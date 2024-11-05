"""Apache Cassandra DB graph vector store integration."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    cast,
)

from langchain_core._api import beta, deprecated
from langchain_core.documents import Document

from langchain_community.graph_vectorstores.base import (
    NODE_CLASS_DEPRECATED_SINCE,
    Node,
)
from langchain_community.graph_vectorstores.cassandra_base import (
    CassandraGraphVectorStoreBase,
)
from langchain_community.graph_vectorstores.links import (
    METADATA_LINKS_KEY,
    Link,
    get_links,
    incoming_links,
)
from langchain_community.utilities.cassandra import SetupMode
from langchain_community.vectorstores.cassandra import Cassandra as CassandraVectorStore

CGVST = TypeVar("CGVST", bound="CassandraGraphVectorStore")

if TYPE_CHECKING:
    from cassandra.cluster import Session
    from langchain_core.embeddings import Embeddings


logger = logging.getLogger(__name__)


def _serialize_links(links: list[Link]) -> str:
    class SetAndLinkEncoder(json.JSONEncoder):
        def default(self, obj: Any) -> Any:  # noqa: ANN401
            if not isinstance(obj, type) and is_dataclass(obj):
                return asdict(obj)

            if isinstance(obj, Iterable):
                return list(obj)

            # Let the base class default method raise the TypeError
            return super().default(obj)

    return json.dumps(links, cls=SetAndLinkEncoder)


def _deserialize_links(json_blob: str | None) -> set[Link]:
    return {
        Link(kind=link["kind"], direction=link["direction"], tag=link["tag"])
        for link in cast(list[dict[str, Any]], json.loads(json_blob or "[]"))
    }


def _metadata_link_key(link: Link) -> str:
    return f"link:{link.kind}:{link.tag}"


def _metadata_link_value() -> str:
    return "link"


@beta()
class CassandraGraphVectorStore(CassandraGraphVectorStoreBase):
    def __init__(
        self,
        embedding: Embeddings,
        session: Session | None = None,
        keyspace: str | None = None,
        table_name: str = "",
        ttl_seconds: int | None = None,
        *,
        body_index_options: list[tuple[str, Any]] | None = None,
        setup_mode: SetupMode = SetupMode.SYNC,
        metadata_deny_list: Optional[list[str]] = None,
    ) -> None:
        """Apache Cassandra(R) for graph-vector-store workloads.

        To use it, you need a recent installation of the `cassio` library
        and a Cassandra cluster / Astra DB instance supporting vector capabilities.

        Example:
            .. code-block:: python

                    from langchain_community.graph_vectorstores import
                        CassandraGraphVectorStore
                    from langchain_openai import OpenAIEmbeddings

                    embeddings = OpenAIEmbeddings()
                    session = ...             # create your Cassandra session object
                    keyspace = 'my_keyspace'  # the keyspace should exist already
                    table_name = 'my_graph_vector_store'
                    vectorstore = CassandraGraphVectorStore(
                        embeddings,
                        session,
                        keyspace,
                        table_name,
                    )

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
                deny_list option.
        """
        if metadata_deny_list is None:
            metadata_deny_list = []
        metadata_deny_list.append(METADATA_LINKS_KEY)

        super().__init__(
            vector_store=CassandraVectorStore(
                embedding=embedding,
                session=session,
                keyspace=keyspace,
                table_name=table_name,
                ttl_seconds=ttl_seconds,
                body_index_options=body_index_options,
                setup_mode=setup_mode,
                metadata_indexing=("deny_list", metadata_deny_list),
            )
        )

    def get_metadata_filter(
        self,
        metadata: dict[str, Any] | None = None,
        outgoing_link: Link | None = None,
    ) -> dict[str, Any]:
        if outgoing_link is None:
            return metadata or {}

        metadata_filter = {} if metadata is None else metadata.copy()
        metadata_filter[_metadata_link_key(link=outgoing_link)] = _metadata_link_value()
        return metadata_filter

    def restore_links(self, doc: Document) -> Document:
        """Restores links in a document by deserializing them from metadata.

        Args:
            doc: Document to restore

        Returns:
            The document ready for use in the graph vector store
        """
        links = _deserialize_links(doc.metadata.get(METADATA_LINKS_KEY))
        doc.metadata[METADATA_LINKS_KEY] = links
        # TODO: Could this be skipped if we put these metadata entries
        # only in the searchable `metadata_s` column?
        for incoming_link in incoming_links(links=links):
            incoming_link_key = _metadata_link_key(link=incoming_link)
            if incoming_link_key in doc.metadata:
                del doc.metadata[incoming_link_key]
        return doc

    def get_metadata_for_insertion(self, doc: Document) -> dict[str, Any]:
        links = get_links(doc=doc)
        metadata = doc.metadata.copy()
        metadata[METADATA_LINKS_KEY] = _serialize_links(links=links)
        # TODO: Could we could put these metadata entries
        # only in the searchable `metadata_s` column?
        for incoming_link in incoming_links(links=links):
            metadata[_metadata_link_key(link=incoming_link)] = _metadata_link_value()
        return metadata

    @deprecated(
        since=NODE_CLASS_DEPRECATED_SINCE,
        pending=True,
        alternative="get_document_by_id",
    )
    def get_node(self, node_id: str) -> Node | None:
        """Retrieve a single node from the store, given its ID.

        Args:
            node_id: The node ID

        Returns:
            The the node if it exists. Otherwise None.
        """
        doc = self.get_by_document_id(document_id=node_id)
        if doc is None:
            return None
        return Node(
            id=doc.id,
            text=doc.page_content,
            metadata=doc.metadata,
            links=get_links(doc=doc),
        )

    @classmethod
    def from_texts(
        cls: Type[CGVST],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        *,
        session: Optional[Session] = None,
        keyspace: Optional[str] = None,
        table_name: str = "",
        ids: Optional[List[str]] = None,
        ttl_seconds: Optional[int] = None,
        body_index_options: Optional[List[Tuple[str, Any]]] = None,
        metadata_deny_list: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> CGVST:
        """Create a CassandraGraphVectorStore from raw texts.

        Args:
            texts: Texts to add to the graph vectorstore.
            embedding: Embedding function to use.
            metadatas: Optional list of metadatas associated with the texts.
            session: Cassandra driver session.
                If not provided, it is resolved from cassio.
            keyspace: Cassandra key space.
                If not provided, it is resolved from cassio.
            table_name: Cassandra table (required).
            ids: Optional list of IDs associated with the texts.
            ttl_seconds: Optional time-to-live for the added texts.
            body_index_options: Optional options used to create the body index.
                Eg. body_index_options = [cassio.table.cql.STANDARD_ANALYZER]
            metadata_deny_list: Optional list of metadata keys to not index.
                i.e. to fine-tune which of the metadata fields are indexed.
                Note: if you plan to have massive unique text metadata entries,
                consider not indexing them for performance
                (and to overcome max-length limitations).
                Note: the `metadata_indexing` parameter from
                langchain_community.utilities.cassandra.Cassandra is not
                exposed since CassandraGraphVectorStore only supports the
                deny_list option.

        Returns:
            a CassandraGraphVectorStore.
        """
        store = cls(
            embedding=embedding,
            session=session,
            keyspace=keyspace,
            table_name=table_name,
            ttl_seconds=ttl_seconds,
            body_index_options=body_index_options,
            metadata_deny_list=metadata_deny_list,
            **kwargs,
        )
        store.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
        )
        return store

    @classmethod
    async def afrom_texts(
        cls: Type[CGVST],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        *,
        session: Optional[Session] = None,
        keyspace: Optional[str] = None,
        table_name: str = "",
        ids: Optional[List[str]] = None,
        ttl_seconds: Optional[int] = None,
        body_index_options: Optional[List[Tuple[str, Any]]] = None,
        metadata_deny_list: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> CGVST:
        """Create a CassandraGraphVectorStore from raw texts.

        Args:
            texts: Texts to add to the graph vectorstore.
            embedding: Embedding function to use.
            metadatas: Optional list of metadatas associated with the texts.
            session: Cassandra driver session.
                If not provided, it is resolved from cassio.
            keyspace: Cassandra key space.
                If not provided, it is resolved from cassio.
            table_name: Cassandra table (required).
            ids: Optional list of IDs associated with the texts.
            ttl_seconds: Optional time-to-live for the added texts.
            body_index_options: Optional options used to create the body index.
                Eg. body_index_options = [cassio.table.cql.STANDARD_ANALYZER]
            metadata_deny_list: Optional list of metadata keys to not index.
                i.e. to fine-tune which of the metadata fields are indexed.
                Note: if you plan to have massive unique text metadata entries,
                consider not indexing them for performance
                (and to overcome max-length limitations).
                Note: the `metadata_indexing` parameter from
                langchain_community.utilities.cassandra.Cassandra is not
                exposed since CassandraGraphVectorStore only supports the
                deny_list option.

        Returns:
            a CassandraGraphVectorStore.
        """
        store = cls(
            embedding=embedding,
            session=session,
            keyspace=keyspace,
            table_name=table_name,
            ttl_seconds=ttl_seconds,
            setup_mode=SetupMode.ASYNC,
            body_index_options=body_index_options,
            metadata_deny_list=metadata_deny_list,
            **kwargs,
        )
        await store.aadd_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
        )
        return store

    @classmethod
    def from_documents(
        cls: Type[CGVST],
        documents: List[Document],
        embedding: Embeddings,
        *,
        session: Optional[Session] = None,
        keyspace: Optional[str] = None,
        table_name: str = "",
        ids: Optional[List[str]] = None,
        ttl_seconds: Optional[int] = None,
        body_index_options: Optional[List[Tuple[str, Any]]] = None,
        metadata_deny_list: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> CGVST:
        """Create a CassandraGraphVectorStore from a document list.

        Args:
            documents: Documents to add to the graph vectorstore.
            embedding: Embedding function to use.
            session: Cassandra driver session.
                If not provided, it is resolved from cassio.
            keyspace: Cassandra key space.
                If not provided, it is resolved from cassio.
            table_name: Cassandra table (required).
            ids: Optional list of IDs associated with the documents.
            ttl_seconds: Optional time-to-live for the added documents.
            body_index_options: Optional options used to create the body index.
                Eg. body_index_options = [cassio.table.cql.STANDARD_ANALYZER]
            metadata_deny_list: Optional list of metadata keys to not index.
                i.e. to fine-tune which of the metadata fields are indexed.
                Note: if you plan to have massive unique text metadata entries,
                consider not indexing them for performance
                (and to overcome max-length limitations).
                Note: the `metadata_indexing` parameter from
                langchain_community.utilities.cassandra.Cassandra is not
                exposed since CassandraGraphVectorStore only supports the
                deny_list option.

        Returns:
            a CassandraGraphVectorStore.
        """
        store = cls(
            embedding=embedding,
            session=session,
            keyspace=keyspace,
            table_name=table_name,
            ttl_seconds=ttl_seconds,
            body_index_options=body_index_options,
            metadata_deny_list=metadata_deny_list,
            **kwargs,
        )
        store.add_documents(documents=documents, ids=ids)
        return store

    @classmethod
    async def afrom_documents(
        cls: Type[CGVST],
        documents: List[Document],
        embedding: Embeddings,
        *,
        session: Optional[Session] = None,
        keyspace: Optional[str] = None,
        table_name: str = "",
        ids: Optional[List[str]] = None,
        ttl_seconds: Optional[int] = None,
        body_index_options: Optional[List[Tuple[str, Any]]] = None,
        metadata_deny_list: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> CGVST:
        """Create a CassandraGraphVectorStore from a document list.

        Args:
            documents: Documents to add to the graph vectorstore.
            embedding: Embedding function to use.
            session: Cassandra driver session.
                If not provided, it is resolved from cassio.
            keyspace: Cassandra key space.
                If not provided, it is resolved from cassio.
            table_name: Cassandra table (required).
            ids: Optional list of IDs associated with the documents.
            ttl_seconds: Optional time-to-live for the added documents.
            body_index_options: Optional options used to create the body index.
                Eg. body_index_options = [cassio.table.cql.STANDARD_ANALYZER]
            metadata_deny_list: Optional list of metadata keys to not index.
                i.e. to fine-tune which of the metadata fields are indexed.
                Note: if you plan to have massive unique text metadata entries,
                consider not indexing them for performance
                (and to overcome max-length limitations).
                Note: the `metadata_indexing` parameter from
                langchain_community.utilities.cassandra.Cassandra is not
                exposed since CassandraGraphVectorStore only supports the
                deny_list option.


        Returns:
            a CassandraGraphVectorStore.
        """
        store = cls(
            embedding=embedding,
            session=session,
            keyspace=keyspace,
            table_name=table_name,
            ttl_seconds=ttl_seconds,
            setup_mode=SetupMode.ASYNC,
            body_index_options=body_index_options,
            metadata_deny_list=metadata_deny_list,
            **kwargs,
        )
        await store.aadd_documents(documents=documents, ids=ids)
        return store
