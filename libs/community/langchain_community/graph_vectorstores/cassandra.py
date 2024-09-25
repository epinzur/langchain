from __future__ import annotations

import json
import logging
import re
import secrets
import asyncio


from typing import (
    TYPE_CHECKING,
    Any,
    cast,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
)

from dataclasses import asdict, is_dataclass

from langchain_core._api import beta
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from langchain_community.graph_vectorstores.base import (
    GraphVectorStore,
    Link,
    Node,
    nodes_to_documents,
    METADATA_LINKS_KEY,
)
from ._mmr_helper import MmrHelper

from langchain_community.utilities.cassandra import SetupMode
from langchain_community.vectorstores.cassandra import Cassandra as CassandraVectorStore

from cassio.table.mixins.metadata import MetadataMixin

if TYPE_CHECKING:
    from cassandra.cluster import Session

from __future__ import annotations


logger = logging.getLogger(__name__)

_CQL_IDENTIFIER_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9_]*")


def _serialize_metadata(md: dict[str, Any]) -> str:
    if isinstance(md.get(METADATA_LINKS_KEY), set):
        md = md.copy()
        md[METADATA_LINKS_KEY] = list(md[METADATA_LINKS_KEY])
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

def _doc_to_node(doc: Document) -> Node:
    metadata = _deserialize_metadata(doc.metadata)
    links: set[Link] = _deserialize_links(metadata.get(METADATA_LINKS_KEY))
    metadata[METADATA_LINKS_KEY] = links

    return Node(
        id=doc.id,
        text=doc.page_content,
        metadata=metadata,
        links=links,
    )

def _incoming_links(node:Node) -> set[Link]:
    return set([l for l in node.links if l.direction in ["in", "bidir"]])

def _outgoing_links(node:Node) -> set[Link]:
    return set([l for l in node.links if l.direction in ["out", "bidir"]])

class AdjacentNode:
    id: str
    links: list[Link]
    embedding: list[float]

    def __init__(self, node: Node, embedding: List[float]) -> None:
        self.id = node.id
        self.links = node.links
        self.embedding = embedding


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

        self._metadata_deny_list = metadata_deny_list

        deny_list = list(metadata_deny_list)
        deny_list.append(METADATA_LINKS_KEY)

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
                row_id, body_blob, vector, metadata_blob, metadata_s
            ) VALUES (?, ?, ?, ?, ?)
            """  # noqa: S608
        )

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return self.store.embedding

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
            combined_metadata[METADATA_LINKS_KEY] = _serialize_links(node.links)
            metadata_list.append(combined_metadata)
            incoming_links_list.append(_incoming_links(node=node))

        text_embeddings = self.store.embedding.embed_documents(texts)

        futures = []
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

            futures.append(
                self.store.session.execute_async(
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
            )

        for future in futures:
            future.result()

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
        return self.store.similarity_search(
            query=query,
            k=k,
           filter=metadata_filter,
        )

    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        metadata_filter: dict[str, Any] = {},
        **kwargs: Any,
    ) -> List[Document]:
        # TODO: Deserialize Links
        return await self.store.asimilarity_search(
            query=query,
            k=k,
           filter=metadata_filter,
        )

    async def asimilarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        metadata_filter: dict[str, Any] = {},
        **kwargs: Any,
    ) -> List[Document]:
        # TODO: Deserialize Links
        return await self.store.asimilarity_search_by_vector(
            embedding,
            k=k,
            filter=metadata_filter,
        )

    def metadata_search(
        self,
        metadata: dict[str, Any] = {},  # noqa: B006
        n: int = 5,
    ) -> Iterable[Document]:
        """Retrieve nodes based on their metadata."""
        return self.store.metadata_search(
            metadata=metadata,
            n=n,
        )

    async def ametadata_search(
        self,
        metadata: dict[str, Any] = {},  # noqa: B006
        n: int = 5,
    ) -> Iterable[Document]:
        """Retrieve nodes based on their metadata."""
        return await self.store.ametadata_search(
            metadata=metadata,
            n=n,
        )

    async def _documents_with_ids(
        self,
        ids: Iterable[str],
    ) -> list[Document|None]:
        tasks = [
            self.store.aget_by_document_id(document_id=id)
            for id in ids
        ]
        return await asyncio.gather(*tasks)

    async def ammr_traversal_search(
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
        query_embedding = self.store.embedding.embed_query(query)
        helper = MmrHelper(
            k=k,
            query_embedding=query_embedding,
            lambda_mult=lambda_mult,
            score_threshold=score_threshold,
        )

        # For each unselected node, stores the outgoing links.
        outgoing_links_map: dict[str, set[Link]] = {}
        visited_links: set[Link] = set()


        async def fetch_neighborhood(neighborhood: Sequence[str]) -> None:
            nonlocal outgoing_links_map
            nonlocal visited_links

            # Put the neighborhood into the outgoing links, to avoid adding it
            # to the candidate set in the future.
            outgoing_links_map.update({content_id: set() for content_id in neighborhood})

            # Initialize the visited_links with the set of outgoing links from the
            # neighborhood. This prevents re-visiting them.
            visited_links = self._get_outgoing_links(neighborhood)

            # Call `self._get_adjacent` to fetch the candidates.
            adjacent_nodes = await self._get_adjacent(
                links=visited_links,
                query_embedding=query_embedding,
                k_per_link=adjacent_k,
                metadata_filter=metadata_filter,
            )

            new_candidates: dict[str, list[float]] = {}
            for adjacent_node in adjacent_nodes:
                if adjacent_node.id not in outgoing_links_map:
                    outgoing_links_map[adjacent_node.id] = _outgoing_links(node=adjacent_node)
                    new_candidates[adjacent_node.id] = adjacent_node.embedding
            helper.add_candidates(new_candidates)

        async def fetch_initial_candidates() -> None:
            nonlocal outgoing_links_map
            nonlocal visited_links

            results = await self.store.asimilarity_search_with_embedding_id_by_vector(
                embedding=query_embedding,
                k=fetch_k,
                filter=metadata_filter,
            )

            candidates: dict[str, list[float]] = {}
            for (doc, embedding, id) in results:
                if id not in outgoing_links_map:
                    node = _doc_to_node(doc)
                    outgoing_links_map[id] = _outgoing_links(node=node)
                    candidates[id] = embedding
            helper.add_candidates(candidates)

        if initial_roots:
            await fetch_neighborhood(initial_roots)
        if fetch_k > 0:
            await fetch_initial_candidates()

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

                # Find the links linked to from the selected ID.
                selected_outgoing_links = outgoing_links_map.pop(selected_id)

                # Don't re-visit already visited links.
                selected_outgoing_links.difference_update(visited_links)

                # Find the nodes with incoming links from those links.
                adjacent_nodes = await self._get_adjacent(
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
                        outgoing_links_map[adjacent_node.id] = _outgoing_links(node=adjacent_node)
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

        return self._documents_with_ids(helper.selected_ids)

    async def atraversal_search(
        self,
        query: str,
        *,
        k: int = 4,
        depth: int = 1,
        metadata_filter: dict[str, Any] = {},
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

        # Map from visited ID to depth
        visited_ids: Dict[str, int] = {}

        # Map from visited link to depth
        visited_links: Dict[Link, int] = {}

        # Map from id to Document
        retrieved_docs: Dict[str, Document] = {}

        async def visit_nodes(d: int, docs: Iterable[Document]) -> None:
            """Recursively visit nodes and their outgoing links."""
            nonlocal visited_ids, visited_links, retrieved_docs

            # Iterate over nodes, tracking the *new* outgoing links for this
            # depth. These are links that are either new, or newly discovered at a
            # lower depth.
            outgoing_links: Set[Link] = set()
            for doc in docs:
                if doc.id not in retrieved_docs:
                    retrieved_docs[doc.id] = doc

                # If this node is at a closer depth, update visited_ids
                if d <= visited_ids.get(doc.id, depth):
                    visited_ids[doc.id] = d

                    # If we can continue traversing from this node,
                    if d < depth:
                        node = _doc_to_node(doc=doc)
                        # Record any new (or newly discovered at a lower depth)
                        # links to the set to traverse.
                        for link in _outgoing_links(node=node):
                            if d <= visited_links.get(link, depth):
                                # Record that we'll query this link at the
                                # given depth, so we don't fetch it again
                                # (unless we find it an earlier depth)
                                visited_links[link] = d
                                outgoing_links.add(link)

            if outgoing_links:
                tasks = []
                for outgoing_link in outgoing_links:
                    metadata = self._get_metadata_filter(
                        metadata=metadata_filter,
                        outgoing_link=outgoing_link,
                    )
                    task = asyncio.create_task(
                        self.store.ametadata_search(metadata=metadata, n=None)
                    )
                    tasks.append(task)
                results = await asyncio.gather(*tasks)

                # Visit targets concurrently
                tasks = [visit_targets(d=d + 1, docs=docs) for docs in results]
                await asyncio.gather(*tasks)

        async def visit_targets(d: int, docs: Iterable[Document]) -> None:
            """Visit target nodes retrieved from outgoing links."""
            nonlocal visited_ids, retrieved_docs

            new_ids_at_next_depth = set()
            for doc in docs:
                if doc.id not in retrieved_docs:
                    retrieved_docs[doc.id] = doc

                if d <= visited_ids.get(doc.id, depth):
                    new_ids_at_next_depth.add(doc.id)

            if new_ids_at_next_depth:
                fetch_tasks = []
                visit_node_tasks = []
                for id in new_ids_at_next_depth:
                    if id in retrieved_docs:
                        visit_node_tasks.append(visit_nodes(d=d, docs=[retrieved_docs[id]]))
                    else:
                        fetch_task = asyncio.create_task(
                            self.store.aget_by_document_id(document_id=id)
                        )
                        fetch_tasks.append(fetch_task)

                results = await asyncio.gather(*fetch_tasks)
                for result in results:
                    visit_node_tasks.append(visit_nodes(d=d, docs=[result]))

                await asyncio.gather(*visit_node_tasks)

        # Start the traversal
        initial_docs = await self.store.asimilarity_search(
            query=query,
            k=k,
            filter=metadata_filter,
        )
        await visit_nodes(d=0, docs=initial_docs)

        return [retrieved_docs[id] for id in visited_ids.keys()]

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
        return asyncio.run(self.atraversal_search(
            query=query,
            k=k,
            depth=depth,
            metadata_filter=metadata_filter,
        ))


    def get_node(self, content_id: str) -> Node:
        """Get a node by its id."""
        return self._documents_with_ids(ids=[content_id])[0]

    async def _get_outgoing_links(self, source_ids: Iterable[str]) -> Set[Link]:
        """Return the set of outgoing links for the given source IDs asynchronously.

        Args:
            source_ids: The IDs of the source nodes to retrieve outgoing links for.

        Returns:
            A set of `Link` objects representing the outgoing links from the source nodes.
        """
        links = set()

        # Create coroutine objects without scheduling them yet
        coroutines = [
            self.store.aget_by_document_id(document_id=source_id)
            for source_id in source_ids
        ]

        # Schedule and await all coroutines
        docs = await asyncio.gather(*coroutines)

        for doc in docs:
            if doc is not None:
                node = _doc_to_node(doc=doc)
                links.update(_outgoing_links(node=node))
            else:
                # Handle the case where the document is not found
                pass

        return links


    async def _get_adjacent(
        self,
        links: set[Link],
        query_embedding: list[float],
        k_per_link: int | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> Iterable[AdjacentNode]:
        """Return the target nodes with incoming links from any of the given links.

        Args:
            links: The links to look for.
            query_embedding: The query embedding. Used to rank target nodes.
            k_per_link: The number of target nodes to fetch for each link.
            metadata_filter: Optional metadata to filter the results.

        Returns:
            List of adjacent edges.
        """
        targets: dict[str, AdjacentNode] = {}

        tasks = []
        for link in links:
            filter = self._get_metadata_filter(
                metadata=metadata_filter,
                outgoing_link=link,
            )

            tasks.append(
                self.store.asimilarity_search_with_embedding_id_by_vector(
                    embedding=query_embedding,
                    k=k_per_link or 10,
                    filter=filter
                )
            )

        results = await asyncio.gather(*tasks)

        for result in results:
            for (doc, embedding, id) in result:
                if id not in targets:
                    node = _doc_to_node(doc=doc)
                    targets[id] = AdjacentNode(node=node, embedding=embedding)

        # TODO: Consider a combined limit based on the similarity and/or
        # predicated MMR score?
        return targets.values()



    def _get_metadata_filter(
        metadata: dict[str, Any] | None = None,
        outgoing_link: Link | None = None,
    ) -> dict[str, Any]:
        if outgoing_link is None:
            return metadata

        metadata_filter = {} if metadata is None else metadata.copy()
        metadata_filter[_metadata_s_link_key(link=outgoing_link)] = _metadata_s_link_value()
        return metadata_filter
