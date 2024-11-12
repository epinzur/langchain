"""Open Search graph vector store integration."""

from __future__ import annotations

import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    TypeVar,
)

from langchain_core.documents import Document

from langchain_community.graph_vectorstores.link_based_gvs import (
    LinkBasedGraphVectorStore,
)
from langchain_community.graph_vectorstores.links import (
    METADATA_LINKS_KEY,
    Link,
    deserialize_links_from_json,
    get_links,
    incoming_links,
    serialize_links_to_json,
)
from langchain_community.graph_vectorstores.wrappers.open_search import (
    OpenSearchVectorStoreForGraph,
)

OpenSearchGVS = TypeVar("OpenSearchGVS", bound="OpenSearchGraphVectorStore")

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings


logger = logging.getLogger(__name__)

METADATA_INCOMING_LINKS_KEY = "__incoming_links"


def _metadata_link_key(link: Link) -> str:
    return f"link:{link.kind}:{link.tag}"


def _metadata_link_value() -> str:
    return "link"


class OpenSearchGraphVectorStore(LinkBasedGraphVectorStore):

    def __init__(
        self,
        opensearch_url: str,
        index_name: str,
        embedding_function: Embeddings,
        **kwargs: Any,
    ) -> None:
        """Initialize with a Open Search client.

        Args:
            opensearch_url: The URL of the Open Search server
            index_name: The Open Search index to store the documents in.
            embedding_function: Embedding function to use.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            vector_store=OpenSearchVectorStoreForGraph(
                opensearch_url=opensearch_url,
                index_name=index_name,
                embedding_function=embedding_function,
                **kwargs,
            )
        )

    def _get_metadata_filter(
        self,
        metadata: dict[str, Any] | None = None,
        outgoing_link: Link | None = None,
    ) -> dict[str, Any]:
        if outgoing_link is None:
            return metadata or {}

        metadata_filter = {} if metadata is None else metadata.copy()
        metadata_filter[_metadata_link_key(link=outgoing_link)] = _metadata_link_value()
        return metadata_filter

    def _restore_links(self, doc: Document) -> Document:
        """Restores links in a document by deserializing them from metadata.

        Args:
            doc: Document to restore

        Returns:
            The document ready for use in the graph vector store
        """
        links = deserialize_links_from_json(doc.metadata.get(METADATA_LINKS_KEY))
        doc.metadata[METADATA_LINKS_KEY] = links
        for incoming_link in incoming_links(links=links):
            incoming_link_key = _metadata_link_key(link=incoming_link)
            doc.metadata.pop(incoming_link_key, None)
        return doc

    def _get_metadata_for_insertion(self, doc: Document) -> dict[str, Any]:
        links = get_links(doc=doc)
        metadata = doc.metadata.copy()
        metadata[METADATA_LINKS_KEY] = serialize_links_to_json(links=links)
        for incoming_link in incoming_links(links=links):
            metadata[_metadata_link_key(link=incoming_link)] = _metadata_link_value()
        return metadata

    @classmethod
    def from_texts(
        cls: Type[OpenSearchGVS],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        *,
        ids: Optional[List[str]] = None,
        opensearch_url: str,
        index_name: str,
        **kwargs: Any,
    ) -> OpenSearchGVS:
        """Create a OpenSearchGraphVectorStore from raw texts.

        Args:
            texts: Texts to add to the graph vectorstore.
            embedding: Embedding function to use.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of IDs associated with the texts.
            opensearch_url: The URL of the Open Search server
            index_name: The Open Search index to store the documents in.
            **kwargs: Additional keyword arguments.

        Returns:
            a OpenSearchGraphVectorStore.
        """
        store = cls(
            opensearch_url=opensearch_url,
            index_name=index_name,
            embedding_function=embedding,
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
        cls: Type[OpenSearchGVS],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        *,
        ids: Optional[List[str]] = None,
        opensearch_url: str,
        index_name: str,
        **kwargs: Any,
    ) -> OpenSearchGVS:
        """Create a OpenSearchGraphVectorStore from raw texts.

        Args:
            texts: Texts to add to the graph vectorstore.
            embedding: Embedding function to use.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of IDs associated with the texts.
            opensearch_url: The URL of the Open Search server
            index_name: The Open Search index to store the documents in.
            **kwargs: Additional keyword arguments.

        Returns:
            a OpenSearchGraphVectorStore.
        """
        store = cls(
            opensearch_url=opensearch_url,
            index_name=index_name,
            embedding_function=embedding,
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
        cls: Type[OpenSearchGVS],
        documents: List[Document],
        embedding: Embeddings,
        *,
        ids: Optional[List[str]] = None,
        opensearch_url: str,
        index_name: str,
        **kwargs: Any,
    ) -> OpenSearchGVS:
        """Create a OpenSearchGraphVectorStore from a document list.

        Args:
            documents: Documents to add to the graph vectorstore.
            embedding: Embedding function to use.
            ids: Optional list of IDs associated with the texts.
            opensearch_url: The URL of the Open Search server
            index_name: The Open Search index to store the documents in.
            **kwargs: Additional keyword arguments.

        Returns:
            a OpenSearchGraphVectorStore.
        """
        store = cls(
            opensearch_url=opensearch_url,
            index_name=index_name,
            embedding_function=embedding,
            **kwargs,
        )
        if ids is None:
            store.add_documents(documents)
        else:
            store.add_documents(documents, ids=ids)
        return store

    @classmethod
    async def afrom_documents(
        cls: Type[OpenSearchGVS],
        documents: List[Document],
        embedding: Embeddings,
        *,
        ids: Optional[List[str]] = None,
        opensearch_url: str,
        index_name: str,
        **kwargs: Any,
    ) -> OpenSearchGVS:
        """Create a OpenSearchGraphVectorStore from a document list.

        Args:
            documents: Documents to add to the graph vectorstore.
            embedding: Embedding function to use.
            ids: Optional list of IDs associated with the texts.
            opensearch_url: The URL of the Open Search server
            index_name: The Open Search index to store the documents in.
            **kwargs: Additional keyword arguments.

        Returns:
            a OpenSearchGraphVectorStore.
        """
        store = cls(
            opensearch_url=opensearch_url,
            index_name=index_name,
            embedding_function=embedding,
            **kwargs,
        )
        if ids is None:
            await store.aadd_documents(documents)
        else:
            await store.aadd_documents(documents, ids=ids)
        return store
