from __future__ import annotations

import contextlib
import json
import logging
import re
import secrets
import threading

from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    cast,
    Dict,
    Iterable,
    List,
    Literal,
    NamedTuple,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from dataclasses import asdict, dataclass, field, is_dataclass


from langchain_core._api import beta
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from langchain_community.graph_vectorstores.base import (
    GraphVectorStore,
    Link,
    Node,
    nodes_to_documents,
)
from ._mmr_helper import MmrHelper

from langchain_community.utilities.cassandra import SetupMode
from langchain_community.vectorstores.cassandra import Cassandra as CassandraVectorStore

from cassio.table.mixins.metadata import MetadataMixin
from cassio.config import check_resolve_keyspace, check_resolve_session

if TYPE_CHECKING:
    from cassandra.cluster import ResponseFuture, Session
    from cassandra.query import PreparedStatement, SimpleStatement
    from types import TracebackType

from abc import ABC, abstractmethod

from __future__ import annotations


logger = logging.getLogger(__name__)

ROW_ID = "row_id"
BODY_BLOB = "body_blob"
METADATA_BLOB = "metadata_blob"
METADATA_S = "metadata_s"
LINKS_METADATA_KEY = "graph_links"
VECTOR = "vector"

_CQL_IDENTIFIER_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9_]*")

class _Callback(Protocol):
    def __call__(self, rows: Sequence[Any], /) -> None: ...

class ConcurrentQueries(contextlib.AbstractContextManager["ConcurrentQueries"]):
    """Context manager for concurrent queries."""

    def __init__(self, session: Session) -> None:
        self._session = session
        self._completion = threading.Condition()
        self._pending = 0
        self._error: BaseException | None = None

    def _handle_result(
        self,
        result: Sequence[NamedTuple],
        future: ResponseFuture,
        callback: Callable[[Sequence[NamedTuple]], Any] | None,
    ) -> None:
        if callback is not None:
            callback(result)

        if future.has_more_pages:
            future.start_fetching_next_page()
        else:
            with self._completion:
                self._pending -= 1
                if self._pending == 0:
                    self._completion.notify()

    def _handle_error(self, error: BaseException, future: ResponseFuture) -> None:
        logger.error(
            "Error executing query: %s",
            future.query,
            exc_info=error,
        )
        with self._completion:
            self._error = error
            self._completion.notify()

    def execute(
        self,
        query: PreparedStatement | SimpleStatement,
        parameters: tuple[Any, ...] | None = None,
        callback: _Callback | None = None,
        timeout: float | None = None,
    ) -> None:
        """Execute a query concurrently.

        Because this is done concurrently, it expects a callback if you need
        to inspect the results.

        Args:
            query: The query to execute.
            parameters: Parameter tuple for the query. Defaults to `None`.
            callback: Callback to apply to the results. Defaults to `None`.
            timeout: Timeout to use (if not the session default).
        """
        # TODO: We could have some form of throttling, where we track the number
        # of pending calls and queue things if it exceed some threshold.

        with self._completion:
            self._pending += 1
            if self._error is not None:
                return

        execute_kwargs = {}
        if timeout is not None:
            execute_kwargs["timeout"] = timeout
        future: ResponseFuture = self._session.execute_async(
            query,
            parameters,
            **execute_kwargs,
        )
        future.add_callbacks(
            self._handle_result,
            self._handle_error,
            callback_kwargs={
                "future": future,
                "callback": callback,
            },
            errback_kwargs={
                "future": future,
            },
        )

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_inst: BaseException | None,
        _exc_traceback: TracebackType | None,
    ) -> Literal[False]:
        with self._completion:
            while self._error is None and self._pending > 0:
                self._completion.wait()

        if self._error is not None:
            raise self._error

        # Don't swallow the exception.
        # We don't need to do anything with the exception (`_exc_*` parameters)
        # since returning false here will automatically re-raise it.
        return False

def _serialize_metadata(md: dict[str, Any]) -> str:
    if isinstance(md.get(LINKS_METADATA_KEY), set):
        md = md.copy()
        md[LINKS_METADATA_KEY] = list(md[LINKS_METADATA_KEY])
    return json.dumps(md)


def _serialize_links(links: set[Link]) -> str:
    class SetAndLinkEncoder(json.JSONEncoder):
        def default(self, obj: Any) -> Any:
            if not isinstance(obj, type) and is_dataclass(obj):
                return asdict(obj)

            if isinstance(obj, Iterable):
                return list(obj)

            # Let the base class default method raise the TypeError
            return super().default(obj)

    return json.dumps(list(links), cls=SetAndLinkEncoder)


def _deserialize_metadata(json_blob: str | None) -> dict[str, Any]:
    # We don't need to convert the links list back to a set -- it will be
    # converted when accessed, if needed.
    return cast(dict[str, Any], json.loads(json_blob or ""))


def _deserialize_links(json_blob: str | None) -> set[Link]:
    return {
        Link(kind=link["kind"], direction=link["direction"], tag=link["tag"])
        for link in cast(list[dict[str, Any]], json.loads(json_blob or ""))
    }

def _metadata_s_link_key(link: Link) -> str:
    return "link_" + json.dumps({"kind": link.kind, "tag": link.tag})

def _metadata_s_link_value() -> str:
    return "link"

def _row_to_node(row: Any) -> Node:
    if hasattr(row, METADATA_BLOB):
        metadata_blob = getattr(row, METADATA_BLOB)
        metadata = _deserialize_metadata(metadata_blob)
        links: set[Link] = _deserialize_links(metadata.get(LINKS_METADATA_KEY))
        metadata[LINKS_METADATA_KEY] = links
    else:
        metadata = {}
        links = set()
    return Node(
        id=getattr(row, ROW_ID, ""),
        # TODO: figure out how to pass this when needed
        # embedding=getattr(row, "text_embedding", []),
        text=getattr(row, BODY_BLOB, ""),
        metadata=metadata,
        links=links,
    )

def _incoming_links(node:Node) -> set[Link]:
    return set([l for l in node.links if l.direction in ["in", "bidir"]])

def _outgoing_links(node:Node) -> set[Link]:
    return set([l for l in node.links if l.direction in ["out", "bidir"]])

@beta()
class CassandraGraphVectorStore(GraphVectorStore):
    def __init__(
        self,
        embedding: Embeddings,
        session: Optional[Session] = None,
        keyspace: Optional[str] = None,
        table_name: str = "",
        ttl_seconds: Optional[int] = None,
        *,
        body_index_options: Optional[List[Tuple[str, Any]]] = None,
        setup_mode: SetupMode = SetupMode.SYNC,
        metadata_deny_list: Iterable[str] = [],
    ) -> None:
        """Apache Cassandra(R) for graph-vector-store workloads.

        To use it, you need a recent installation of the `cassio` library
        and a Cassandra cluster / Astra DB instance supporting vector capabilities.

        Example:
            .. code-block:: python

                    from langchain_community.graph_vectorstores import CassandraGraphVectorStore
                    from langchain_openai import OpenAIEmbeddings

                    embeddings = OpenAIEmbeddings()
                    session = ...             # create your Cassandra session object
                    keyspace = 'my_keyspace'  # the keyspace should exist already
                    table_name = 'my_graph_vector_store'
                    vectorstore = CassandraGraphVectorStore(embeddings, session, keyspace, table_name)

        Args:
            embedding: Embedding function to use.
            session: Cassandra driver session. If not provided, it is resolved from
                cassio.
            keyspace: Cassandra keyspace. If not provided, it is resolved from cassio.
            table_name: Cassandra table (required).
            ttl_seconds: Optional time-to-live for the added texts.
            body_index_options: Optional options used to create the body index.
                Eg. body_index_options = [cassio.table.cql.STANDARD_ANALYZER]
            setup_mode: mode used to create the Cassandra table (SYNC,
                ASYNC or OFF).
            metadata_deny_list: Optional list of metadata keys to not index.
                i.e. to fine-tune which of the metadata fields are indexed.
                Note: if you plan to have massive unique text metadata entries,
                consider not indexing them for performance
                (and to overcome max-length limitations).
                Note: the `metadata_indexing` parameter from
                langchain_community.utilities.cassandra.Cassandra is not
                exposed since CassandraGraphVectorStore only supports the
                deny-list option.
        """

        session = check_resolve_session(session)
        keyspace = check_resolve_keyspace(keyspace)

        if not _CQL_IDENTIFIER_PATTERN.fullmatch(keyspace):
            msg = f"Invalid keyspace: {keyspace}"
            raise ValueError(msg)

        if not _CQL_IDENTIFIER_PATTERN.fullmatch(table_name):
            msg = f"Invalid table name: {table_name}"
            raise ValueError(msg)

        self._embedding = embedding
        self._table_name = table_name
        self._session = session
        self._keyspace = keyspace
        self._metadata_deny_list = metadata_deny_list
        self._prepared_query_cache: dict[str, PreparedStatement] = {}

        deny_list = list(metadata_deny_list)
        deny_list.append(LINKS_METADATA_KEY)

        self.store = CassandraVectorStore(
            embedding=embedding,
            session=session,
            keyspace=keyspace,
            table_name=table_name,
            ttl_seconds=ttl_seconds,
            body_index_options=body_index_options,
            setup_mode=setup_mode,
            metadata_indexing=("deny-list", deny_list)
        )

        self._insert_node = session.prepare(
            f"""
            INSERT INTO {keyspace}.{table_name} (
                {ROW_ID}, {BODY_BLOB}, {VECTOR}, {METADATA_BLOB}, {METADATA_S}
            ) VALUES (?, ?, ?, ?, ?)
            """  # noqa: S608
        )

        self._query_by_id = session.prepare(
            f"""
            SELECT {ROW_ID}, {BODY_BLOB}, {METADATA_BLOB}
            FROM {keyspace}.{table_name}
            WHERE content_id = ?
            """  # noqa: S608
        )

        self._query_id_and_metadata_by_id = session.prepare(
            f"""
            SELECT {ROW_ID}, {METADATA_BLOB}
            FROM {keyspace}.{table_name}
            WHERE content_id = ?
            """  # noqa: S608
        )

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return self._embedding

    def _concurrent_queries(self) -> ConcurrentQueries:
        return ConcurrentQueries(self._session)

    # TODO: Async (aadd_nodes)
    def add_nodes(
        self,
        nodes: Iterable[Node],
    ) -> Iterable[str]:
        """Add nodes to the graph store."""
        node_ids: list[str] = []
        texts: list[str] = []
        metadata_list: list[dict[str, Any]] = []
        incoming_links_list: list[set[Link]] = []
        for node in nodes:
            if not node.id:
                node_ids.append(secrets.token_hex(8))
            else:
                node_ids.append(node.id)
            texts.append(node.text)
            combined_metadata = node.metadata.copy()
            combined_metadata[LINKS_METADATA_KEY] = _serialize_links(node.links)
            metadata_list.append(combined_metadata)
            incoming_links_list.append(_incoming_links(node=node))

        text_embeddings = self._embedding.embed_documents(texts)

        with self._concurrent_queries() as cq:
            tuples = zip(node_ids, texts, text_embeddings, metadata_list, incoming_links_list)
            for node_id, text, text_embedding, metadata, incoming_links in tuples:

                metadata_s = {
                    k: MetadataMixin._coerce_string(v)
                    for k, v in metadata.items()
                    if k not in self._metadata_deny_list
                }

                for incoming_link in incoming_links:
                    metadata_s[_metadata_s_link_key(link=incoming_link)] =_metadata_s_link_value()

                metadata_blob = _serialize_metadata(metadata)

                cq.execute(
                    self._insert_node,
                    parameters=(
                        node_id,
                        text,
                        text_embedding,
                        metadata_blob,
                        metadata_s,
                    ),
                    timeout=30.0,
                )

        return node_ids

    @classmethod
    def from_texts(
        cls: Type["CassandraGraphVectorStore"],
        texts: Iterable[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[Iterable[str]] = None,
        **kwargs: Any,
    ) -> "CassandraGraphVectorStore":
        """Return CassandraGraphVectorStore initialized from texts and embeddings."""
        store = cls(embedding, **kwargs)
        store.add_texts(texts, metadatas, ids=ids)
        return store

    @classmethod
    def from_documents(
        cls: Type["CassandraGraphVectorStore"],
        documents: Iterable[Document],
        embedding: Embeddings,
        ids: Optional[Iterable[str]] = None,
        **kwargs: Any,
    ) -> "CassandraGraphVectorStore":
        """Return CassandraGraphVectorStore initialized from documents and
        embeddings."""
        store = cls(embedding, **kwargs)
        store.add_documents(documents, ids=ids)
        return store

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        metadata_filter: dict[str, Any] = {},
        **kwargs: Any,
    ) -> List[Document]:
        # TODO: Deserialize Links
        return self.similarity_search(
            query=query,
            k=k,
            metadata_filter=metadata_filter,
        )

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        metadata_filter: dict[str, Any] = {},
        **kwargs: Any,
    ) -> List[Document]:
        # TODO: Deserialize Links
        return self.store.similarity_search_by_vector(
            embedding,
            k=k,
            metadata_filter=metadata_filter,
        )

    def metadata_search(
        self,
        metadata: dict[str, Any] = {},  # noqa: B006
        n: int = 5,
    ) -> Iterable[Node]:
        """Retrieve nodes based on their metadata."""
        query, params = self._get_search_cql_and_params(
            columns=f"{ROW_ID}, {BODY_BLOB}, {METADATA_BLOB}",
            metadata=metadata,
            limit=n,
        )

        for row in self._session.execute(query, params):
            yield _row_to_node(row)

    def _nodes_with_ids(
        self,
        ids: Iterable[str],
    ) -> list[Node]:
        results: dict[str, Node | None] = {}
        with self._concurrent_queries() as cq:

            def node_callback(rows: Iterable[Any]) -> None:
                # Should always be exactly one row here. We don't need to check
                #   1. The query is for a `ID == ?` query on the primary key.
                #   2. If it doesn't exist, the `get_result` method below will
                #      raise an exception indicating the ID doesn't exist.
                for row in rows:
                    row_id = getattr(row, ROW_ID)
                    results[row_id] = _row_to_node(row)

            for node_id in ids:
                if node_id not in results:
                    # Mark this node ID as being fetched.
                    results[node_id] = None
                    cq.execute(
                        self._query_by_id, parameters=(node_id,), callback=node_callback
                    )

        def get_result(node_id: str) -> Node:
            if (result := results[node_id]) is None:
                msg = f"No node with ID '{node_id}'"
                raise ValueError(msg)
            return result

        return [get_result(node_id) for node_id in ids]

    def mmr_traversal_search(
        self,
        query: str,
        *,
        initial_roots: Sequence[str] = (),
        k: int = 4,
        depth: int = 2,
        fetch_k: int = 100,
        adjacent_k: int = 10,
        lambda_mult: float = 0.5,
        score_threshold: float = float("-inf"),
        metadata_filter: dict[str, Any] = {},  # noqa: B006
    ) -> Iterable[Node]:
        """Retrieve documents from this graph store using MMR-traversal.

        This strategy first retrieves the top `fetch_k` results by similarity to
        the question. It then selects the top `k` results based on
        maximum-marginal relevance using the given `lambda_mult`.

        At each step, it considers the (remaining) documents from `fetch_k` as
        well as any documents connected by edges to a selected document
        retrieved based on similarity (a "root").

        Args:
            query: The query string to search for.
            initial_roots: Optional list of document IDs to use for initializing search.
                The top `adjacent_k` nodes adjacent to each initial root will be
                included in the set of initial candidates. To fetch only in the
                neighborhood of these nodes, set `ftech_k = 0`.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of initial Documents to fetch via similarity.
                Will be added to the nodes adjacent to `initial_roots`.
                Defaults to 100.
            adjacent_k: Number of adjacent Documents to fetch.
                Defaults to 10.
            depth: Maximum depth of a node (number of edges) from a node
                retrieved via similarity. Defaults to 2.
            lambda_mult: Number between 0 and 1 that determines the degree
                of diversity among the results with 0 corresponding to maximum
                diversity and 1 to minimum diversity. Defaults to 0.5.
            score_threshold: Only documents with a score greater than or equal
                this threshold will be chosen. Defaults to -infinity.
            metadata_filter: Optional metadata to filter the results.
        """
        query_embedding = self._embedding.embed_query(query)
        helper = MmrHelper(
            k=k,
            query_embedding=query_embedding,
            lambda_mult=lambda_mult,
            score_threshold=score_threshold,
        )

        # For each unselected node, stores the outgoing links.
        outgoing_links_map: dict[str, set[Link]] = {}
        visited_links: set[Link] = set()


        def fetch_neighborhood(neighborhood: Sequence[str]) -> None:
            nonlocal outgoing_links_map
            nonlocal visited_links

            # Put the neighborhood into the outgoing links, to avoid adding it
            # to the candidate set in the future.
            outgoing_links_map.update({content_id: set() for content_id in neighborhood})

            # Initialize the visited_links with the set of outgoing links from the
            # neighborhood. This prevents re-visiting them.
            visited_links = self._get_outgoing_links(neighborhood)

            # Call `self._get_adjacent` to fetch the candidates.
            adjacent_items = self._get_adjacent(
                links=visited_links,
                query_embedding=query_embedding,
                k_per_link=adjacent_k,
                metadata_filter=metadata_filter,
            )

            new_candidates: dict[str, list[float]] = {}
            for adjacent_node, embedding in adjacent_items:
                if adjacent_node.id not in outgoing_links_map:
                    outgoing_links_map[adjacent_node.id] = _outgoing_links(node=adjacent_node)
                    new_candidates[adjacent_node.id] = embedding
            helper.add_candidates(new_candidates)

        def fetch_initial_candidates() -> None:
            nonlocal outgoing_links_map
            nonlocal visited_links

            initial_candidates_query, params = self._get_search_cql_and_params(
                columns = "content_id, text_embedding, metadata_blob",
                limit=fetch_k,
                metadata=metadata_filter,
                embedding=query_embedding,
            )

            rows = self._session.execute(
                query=initial_candidates_query, parameters=params
            )
            candidates: dict[str, list[float]] = {}
            for row in rows:
                if row.content_id not in outgoing_links_map:
                    node = _row_to_node(row=row)
                    outgoing_links_map[node.id] = node.outgoing_links()
                    candidates[node.id] = node.embedding
            helper.add_candidates(candidates)

        if initial_roots:
            fetch_neighborhood(initial_roots)
        if fetch_k > 0:
            fetch_initial_candidates()

        # Tracks the depth of each candidate.
        depths = {candidate_id: 0 for candidate_id in helper.candidate_ids()}

        # Select the best item, K times.
        for _ in range(k):
            selected_id = helper.pop_best()

            if selected_id is None:
                break

            next_depth = depths[selected_id] + 1
            if next_depth < depth:
                # If the next nodes would not exceed the depth limit, find the
                # adjacent nodes.
                #
                # TODO: For a big performance win, we should track which links we've
                # already incorporated. We don't need to issue adjacent queries for
                # those.

                # Find the links linked to from the selected ID.
                selected_outgoing_links = outgoing_links_map.pop(selected_id)

                # Don't re-visit already visited links.
                selected_outgoing_links.difference_update(visited_links)

                # Find the nodes with incoming links from those links.
                adjacent_nodes = self._get_adjacent(
                    links=selected_outgoing_links,
                    query_embedding=query_embedding,
                    k_per_link=adjacent_k,
                    metadata_filter=metadata_filter,
                )

                # Record the selected_outgoing_links as visited.
                visited_links.update(selected_outgoing_links)

                new_candidates = {}
                for adjacent_node in adjacent_nodes:
                    if adjacent_node.id not in outgoing_links_map:
                        outgoing_links_map[adjacent_node.id] = adjacent_node.outgoing_links()
                        new_candidates[adjacent_node.id] = adjacent_node.embedding
                        if next_depth < depths.get(adjacent_node.id, depth + 1):
                            # If this is a new shortest depth, or there was no
                            # previous depth, update the depths. This ensures that
                            # when we discover a node we will have the shortest
                            # depth available.
                            #
                            # NOTE: No effort is made to traverse from nodes that
                            # were previously selected if they become reachable via
                            # a shorter path via nodes selected later. This is
                            # currently "intended", but may be worth experimenting
                            # with.
                            depths[adjacent_node.id] = next_depth
                helper.add_candidates(new_candidates)

        return self._nodes_with_ids(helper.selected_ids)

    def traversal_search(
        self,
        query: str,
        *,
        k: int = 4,
        depth: int = 1,
        metadata_filter: dict[str, Any] = {},  # noqa: B006
    ) -> Iterable[Node]:
        """Retrieve documents from this knowledge store.

        First, `k` nodes are retrieved using a vector search for the `query` string.
        Then, additional nodes are discovered up to the given `depth` from those
        starting nodes.

        Args:
            query: The query string.
            k: The number of Documents to return from the initial vector search.
                Defaults to 4.
            depth: The maximum depth of edges to traverse. Defaults to 1.
            metadata_filter: Optional metadata to filter the results.

        Returns:
            Collection of retrieved documents.
        """
        # Depth 0:
        #   Query for `k` nodes similar to the question.
        #   Retrieve `content_id` and `outgoing_links()`.
        #
        # Depth 1:
        #   Query for nodes that have an incoming link in the `outgoing_links()` set.
        #   Combine node IDs.
        #   Query for `outgoing_links()` of those "new" node IDs.
        #
        # ...




        with self._concurrent_queries() as cq:
            # Map from visited ID to depth
            visited_ids: dict[str, int] = {}

            # Map from visited link to depth. Allows skipping queries
            # for links that we've already traversed.
            visited_links: dict[Link, int] = {}

            def visit_nodes(d: int, rows: Sequence[Any]) -> None:
                nonlocal visited_ids
                nonlocal visited_links

                # Visit nodes at the given depth.

                # Iterate over nodes, tracking the *new* outgoing links for this
                # depth. These are links that are either new, or newly discovered at a
                # lower depth.
                outgoing_links: Set[Link] = set()
                for row in rows:
                    content_id = row.content_id

                    # Add visited ID. If it is closer it is a new node at this depth:
                    if d <= visited_ids.get(content_id, depth):
                        visited_ids[content_id] = d

                        # If we can continue traversing from this node,
                        if d < depth:
                            node = _row_to_node(row=row)
                            # Record any new (or newly discovered at a lower depth)
                            # links to the set to traverse.
                            for link in node.outgoing_links():
                                if d <= visited_links.get(link, depth):
                                    # Record that we'll query this link at the
                                    # given depth, so we don't fetch it again
                                    # (unless we find it an earlier depth)
                                    visited_links[link] = d
                                    outgoing_links.add(link)

                if outgoing_links:
                    # If there are new links to visit at the next depth, query for the
                    # node IDs.
                    for outgoing_link in outgoing_links:
                        visit_nodes_query, params = self._get_search_cql_and_params(
                            columns="content_id AS target_content_id",
                            metadata=metadata_filter,
                            outgoing_link=outgoing_link,
                        )
                        cq.execute(
                            query=visit_nodes_query,
                            parameters=params,
                            callback=lambda rows, d=d: visit_targets(d, rows),
                        )

            def visit_targets(d: int, rows: Sequence[Any]) -> None:
                nonlocal visited_ids

                new_node_ids_at_next_depth = set()
                for row in rows:
                    content_id = row.target_content_id
                    if d < visited_ids.get(content_id, depth):
                        new_node_ids_at_next_depth.add(content_id)

                if new_node_ids_at_next_depth:
                    for node_id in new_node_ids_at_next_depth:
                        cq.execute(
                            self._query_id_and_metadata_by_id,
                            parameters=(node_id,),
                            callback=lambda rows, d=d: visit_nodes(d + 1, rows),
                        )

            initial_query, params = self._get_search_cql_and_params(
                columns="content_id, metadata_blob",
                limit=k,
                metadata=metadata_filter,
                embedding=self._embedding.embed_query(query),
            )

            cq.execute(
                initial_query,
                parameters=params,
                callback=lambda initial_rows: visit_nodes(0, initial_rows),
            )

        return self._nodes_with_ids(visited_ids.keys())



    def get_node(self, content_id: str) -> Node:
        """Get a node by its id."""
        return self._nodes_with_ids(ids=[content_id])[0]

    def _get_outgoing_links(
        self,
        source_ids: Iterable[str],
    ) -> set[Link]:
        """Return the set of outgoing links for the given source ID(s).

        Args:
            source_ids: The IDs of the source nodes to retrieve outgoing links for.
        """
        links = set()

        def add_sources(rows: Iterable[Any]) -> None:
            for row in rows:
                node = _row_to_node(row=row)
                links.update(node.outgoing_links())

        with self._concurrent_queries() as cq:
            for source_id in source_ids:
                cq.execute(
                    self._query_id_and_metadata_by_id,
                    (source_id,),
                    callback=add_sources,
                )

        return links

    def _get_adjacent(
        self,
        links: set[Link],
        query_embedding: list[float],
        k_per_link: int | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> Iterable[Tuple[Node, list[float]]]:
        """Return the target nodes with incoming links from any of the given links.

        Args:
            links: The links to look for.
            query_embedding: The query embedding. Used to rank target nodes.
            k_per_link: The number of target nodes to fetch for each link.
            metadata_filter: Optional metadata to filter the results.

        Returns:
            List of adjacent edges.
        """
        targets: dict[str, Tuple[Node, list[float]]] = {}

        def add_targets(rows: Iterable[Any]) -> None:
            nonlocal targets

            # TODO: Figure out how to use the "kind" on the edge.
            # This is tricky, since we currently issue one query for anything
            # adjacent via any kind, and we don't have enough information to
            # determine which kind(s) a given target was reached from.
            for row in rows:
                row_id = getattr(row, ROW_ID)
                if row_id not in targets:
                    embedding = getattr(row, VECTOR)
                    targets[row_id] = (_row_to_node(row=row), embedding)

        with self._concurrent_queries() as cq:
            for link in links:
                adjacent_query, params = self._get_search_cql_and_params(
                    columns = "content_id, text_embedding, metadata_blob",
                    limit=k_per_link or 10,
                    metadata=metadata_filter,
                    embedding=query_embedding,
                    outgoing_link=link,
                )

                cq.execute(
                    query=adjacent_query,
                    parameters=params,
                    callback=add_targets,
                )

        # TODO: Consider a combined limit based on the similarity and/or
        # predicated MMR score?
        return targets.values()

    @staticmethod
    def _normalize_metadata_indexing_policy(
        metadata_indexing: tuple[str, Iterable[str]] | str,
    ) -> MetadataIndexingPolicy:
        mode: MetadataIndexingMode
        fields: set[str]
        # metadata indexing policy normalization:
        if isinstance(metadata_indexing, str):
            if metadata_indexing.lower() == "all":
                mode, fields = (MetadataIndexingMode.DEFAULT_TO_SEARCHABLE, set())
            elif metadata_indexing.lower() == "none":
                mode, fields = (MetadataIndexingMode.DEFAULT_TO_UNSEARCHABLE, set())
            else:
                msg = f"Unsupported metadata_indexing value '{metadata_indexing}'"
                raise ValueError(msg)
        else:
            # it's a 2-tuple (mode, fields) still to normalize
            _mode, _field_spec = metadata_indexing
            fields = {_field_spec} if isinstance(_field_spec, str) else set(_field_spec)
            if _mode.lower() in {
                "default_to_unsearchable",
                "allowlist",
                "allow",
                "allow_list",
            }:
                mode = MetadataIndexingMode.DEFAULT_TO_UNSEARCHABLE
            elif _mode.lower() in {
                "default_to_searchable",
                "denylist",
                "deny",
                "deny_list",
            }:
                mode = MetadataIndexingMode.DEFAULT_TO_SEARCHABLE
            else:
                msg = f"Unsupported metadata indexing mode specification '{_mode}'"
                raise ValueError(msg)
        return mode, fields

    @staticmethod
    def _coerce_string(value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, bool):
            # bool MUST come before int in this chain of ifs!
            return json.dumps(value)
        if isinstance(value, int):
            # we don't want to store '1' and '1.0' differently
            # for the sake of metadata-filtered retrieval:
            return json.dumps(float(value))
        if isinstance(value, float) or value is None:
            return json.dumps(value)
        # when all else fails ...
        return str(value)

    def _extract_where_clause_cql(
        self,
        metadata_keys: Sequence[str] = (),
    ) -> str:
        wc_blocks: list[str] = []

        for key in sorted(metadata_keys):
            if _is_metadata_field_indexed(key, self._metadata_indexing_policy):
                wc_blocks.append(f"metadata_s['{key}'] = %s")
            else:
                msg = "Non-indexed metadata fields cannot be used in queries."
                raise ValueError(msg)

        if len(wc_blocks) == 0:
            return ""

        return " WHERE " + " AND ".join(wc_blocks)

    def _extract_where_clause_params(
        self,
        metadata: dict[str, Any],
    ) -> list[Any]:
        params: list[Any] = []

        for key, value in sorted(metadata.items()):
            if _is_metadata_field_indexed(key, self._metadata_indexing_policy):
                params.append(self._coerce_string(value=value))
            else:
                msg = "Non-indexed metadata fields cannot be used in queries."
                raise ValueError(msg)

        return params

    def _get_search_cql_and_params(
        self,
        columns: str,
        limit: int | None = None,
        metadata: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
        outgoing_link: Link | None = None,
    ) -> tuple[PreparedStatement|SimpleStatement, tuple[Any, ...]]:
        if outgoing_link is not None:
            if metadata is None:
                metadata = {}
            else:
                # don't add link search to original metadata dict
                metadata = metadata.copy()
                metadata[_metadata_s_link_key(link=outgoing_link)] = _metadata_s_link_value()

        metadata_keys = list(metadata.keys()) if metadata else []

        where_clause = self._extract_where_clause_cql(metadata_keys=metadata_keys)
        limit_clause = " LIMIT ?" if limit is not None else ""
        order_clause = " ORDER BY text_embedding ANN OF ?" if embedding is not None else ""

        select_cql = SELECT_CQL_TEMPLATE.format(
            columns=columns,
            table_name=self.table_name(),
            where_clause=where_clause,
            order_clause=order_clause,
            limit_clause=limit_clause,
        )

        where_params = self._extract_where_clause_params(metadata=metadata or {})
        limit_params = [limit] if limit is not None else []
        order_params = [embedding] if embedding is not None else []

        params = tuple(list(where_params) + order_params + limit_params)

        if len(metadata_keys) > 0:
            return SimpleStatement(query_string=select_cql, fetch_size=100), params
        elif select_cql in self._prepared_query_cache:
            return self._prepared_query_cache[select_cql], params
        else:
            prepared_query = self._session.prepare(select_cql)
            prepared_query.consistency_level = ConsistencyLevel.ONE
            self._prepared_query_cache[select_cql] = prepared_query
            return prepared_query, params
