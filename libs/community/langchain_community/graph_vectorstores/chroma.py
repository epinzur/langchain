"""Chroma DB graph vector store integration."""

from __future__ import annotations

import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Type,
    TypeVar,
    cast,
)

from langchain_core.documents import Document

from langchain_community.graph_vectorstores.base import (
    _texts_to_documents,
)
from langchain_community.graph_vectorstores.link_based_gvs import (
    LinkBasedGraphVectorStore,
)
from langchain_community.graph_vectorstores.links import (
    deserialize_links_from_json,
    METADATA_LINKS_KEY,
    Link,
    get_links,
    incoming_links,
    serialize_links_to_json,
)
from langchain_community.graph_vectorstores.wrappers import ChromaVectorStoreForGraph

ChromaGVS = TypeVar("ChromaGVS", bound="ChromaGraphVectorStore")

if TYPE_CHECKING:
    import chromadb
    from langchain_core.embeddings import Embeddings


logger = logging.getLogger(__name__)

METADATA_INCOMING_LINKS_KEY = "__incoming_links"

def _metadata_link_key(link: Link) -> str:
    return f"link:{link.kind}:{link.tag}"


def _metadata_link_value() -> str:
    return "link"


class ChromaGraphVectorStore(LinkBasedGraphVectorStore):
    _LANGCHAIN_DEFAULT_COLLECTION_NAME = "langchain"

    def __init__(
        self,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        embedding_function: Optional[Embeddings] = None,
        persist_directory: Optional[str] = None,
        client_settings: Optional[chromadb.config.Settings] = None,
        collection_metadata: Optional[Dict] = None,
        client: Optional[chromadb.ClientAPI] = None,
        relevance_score_fn: Optional[Callable[[float], float]] = None,
        create_collection_if_not_exists: Optional[bool] = True,
    ) -> None:
        """Initialize with a Chroma client.

        Args:
            collection_name: Name of the collection to create.
            embedding_function: Embedding class object. Used to embed texts.
            persist_directory: Directory to persist the collection.
            client_settings: Chroma client settings
            collection_metadata: Collection configurations.
            client: Chroma client. Documentation:
                    https://docs.trychroma.com/reference/js-client#class:-chromaclient
            relevance_score_fn: Function to calculate relevance score from distance.
                    Used only in `similarity_search_with_relevance_scores`
            create_collection_if_not_exists: Whether to create collection
                    if it doesn't exist. Defaults to True.
        """
        super().__init__(
            vector_store=ChromaVectorStoreForGraph(
                collection_name=collection_name,
                embedding_function=embedding_function,
                persist_directory=persist_directory,
                client_settings=client_settings,
                collection_metadata=collection_metadata,
                client=client,
                relevance_score_fn=relevance_score_fn,
                create_collection_if_not_exists=create_collection_if_not_exists,
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
        # TODO: Could this be skipped if we put these metadata entries
        # only in the searchable `metadata_s` column?
        for incoming_link in incoming_links(links=links):
            incoming_link_key = _metadata_link_key(link=incoming_link)
            doc.metadata.pop(incoming_link_key, None)
        return doc

    def _get_metadata_for_insertion(self, doc: Document) -> dict[str, Any]:
        links = get_links(doc=doc)
        metadata = doc.metadata.copy()
        metadata[METADATA_LINKS_KEY] = serialize_links_to_json(links=links)
        # TODO: Could we could put these metadata entries
        # only in the searchable `metadata_s` column?
        for incoming_link in incoming_links(links=links):
            metadata[_metadata_link_key(link=incoming_link)] = _metadata_link_value()
        return metadata

    @classmethod
    def from_texts(
        cls: Type[ChromaGVS],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        *,
        ids: Optional[List[str]] = None,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        persist_directory: Optional[str] = None,
        client_settings: Optional[chromadb.config.Settings] = None,
        collection_metadata: Optional[Dict] = None,
        client: Optional[chromadb.ClientAPI] = None,
        relevance_score_fn: Optional[Callable[[float], float]] = None,
        create_collection_if_not_exists: Optional[bool] = True,
        **kwargs: Any,
    ) -> ChromaGVS:
        """Create a ChromaGraphVectorStore from raw texts.

        Args:
            texts: Texts to add to the graph vectorstore.
            embedding: Embedding function to use.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of IDs associated with the texts.
            collection_name: Name of the collection to create.
            persist_directory: Directory to persist the collection.
            client_settings: Chroma client settings
            collection_metadata: Collection configurations.
            client: Chroma client. Documentation:
                    https://docs.trychroma.com/reference/js-client#class:-chromaclient
            relevance_score_fn: Function to calculate relevance score from distance.
                    Used only in `similarity_search_with_relevance_scores`
            create_collection_if_not_exists: Whether to create collection
                    if it doesn't exist. Defaults to True.
            **kwargs: Additional keyword arguments.

        Returns:
            a ChromaGraphVectorStore.
        """
        store = cls(
            collection_name=collection_name,
            embedding_function=embedding,
            persist_directory=persist_directory,
            client_settings=client_settings,
            collection_metadata=collection_metadata,
            client=client,
            relevance_score_fn=relevance_score_fn,
            create_collection_if_not_exists=create_collection_if_not_exists,
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
        cls: Type[ChromaGVS],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        *,
        ids: Optional[List[str]] = None,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        persist_directory: Optional[str] = None,
        client_settings: Optional[chromadb.config.Settings] = None,
        collection_metadata: Optional[Dict] = None,
        client: Optional[chromadb.ClientAPI] = None,
        relevance_score_fn: Optional[Callable[[float], float]] = None,
        create_collection_if_not_exists: Optional[bool] = True,
        **kwargs: Any,
    ) -> ChromaGVS:
        """Create a ChromaGraphVectorStore from raw texts.

        Args:
            texts: Texts to add to the graph vectorstore.
            embedding: Embedding function to use.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of IDs associated with the texts.
            collection_name: Name of the collection to create.
            persist_directory: Directory to persist the collection.
            client_settings: Chroma client settings
            collection_metadata: Collection configurations.
            client: Chroma client. Documentation:
                    https://docs.trychroma.com/reference/js-client#class:-chromaclient
            relevance_score_fn: Function to calculate relevance score from distance.
                    Used only in `similarity_search_with_relevance_scores`
            create_collection_if_not_exists: Whether to create collection
                    if it doesn't exist. Defaults to True.
            **kwargs: Additional keyword arguments.

        Returns:
            a ChromaGraphVectorStore.
        """
        store = cls(
            collection_name=collection_name,
            embedding_function=embedding,
            persist_directory=persist_directory,
            client_settings=client_settings,
            collection_metadata=collection_metadata,
            client=client,
            relevance_score_fn=relevance_score_fn,
            create_collection_if_not_exists=create_collection_if_not_exists,
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
        cls: Type[ChromaGVS],
        documents: List[Document],
        embedding: Embeddings,
        *,
        ids: Optional[List[str]] = None,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        persist_directory: Optional[str] = None,
        client_settings: Optional[chromadb.config.Settings] = None,
        collection_metadata: Optional[Dict] = None,
        client: Optional[chromadb.ClientAPI] = None,
        relevance_score_fn: Optional[Callable[[float], float]] = None,
        create_collection_if_not_exists: Optional[bool] = True,
        **kwargs: Any,
    ) -> ChromaGVS:
        """Create a ChromaGraphVectorStore from a document list.

        Args:
            documents: Documents to add to the graph vectorstore.
            embedding: Embedding function to use.
            ids: Optional list of IDs associated with the documents.
            collection_name: Name of the collection to create.
            persist_directory: Directory to persist the collection.
            client_settings: Chroma client settings
            collection_metadata: Collection configurations.
            client: Chroma client. Documentation:
                    https://docs.trychroma.com/reference/js-client#class:-chromaclient
            relevance_score_fn: Function to calculate relevance score from distance.
                    Used only in `similarity_search_with_relevance_scores`
            create_collection_if_not_exists: Whether to create collection
                    if it doesn't exist. Defaults to True.
            **kwargs: Additional keyword arguments.

        Returns:
            a ChromaGraphVectorStore.
        """
        store = cls(
            collection_name=collection_name,
            embedding_function=embedding,
            persist_directory=persist_directory,
            client_settings=client_settings,
            collection_metadata=collection_metadata,
            client=client,
            relevance_score_fn=relevance_score_fn,
            create_collection_if_not_exists=create_collection_if_not_exists,
            **kwargs,
        )
        if ids is None:
            store.add_documents(documents)
        else:
            store.add_documents(documents, ids=ids)
        return store

    @classmethod
    async def afrom_documents(
        cls: Type[ChromaGVS],
        documents: List[Document],
        embedding: Embeddings,
        *,
        ids: Optional[List[str]] = None,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        persist_directory: Optional[str] = None,
        client_settings: Optional[chromadb.config.Settings] = None,
        collection_metadata: Optional[Dict] = None,
        client: Optional[chromadb.ClientAPI] = None,
        relevance_score_fn: Optional[Callable[[float], float]] = None,
        create_collection_if_not_exists: Optional[bool] = True,
        **kwargs: Any,
    ) -> ChromaGVS:
        """Create a ChromaGraphVectorStore from a document list.

        Args:
            documents: Documents to add to the graph vectorstore.
            embedding: Embedding function to use.
            ids: Optional list of IDs associated with the documents.
            collection_name: Name of the collection to create.
            persist_directory: Directory to persist the collection.
            client_settings: Chroma client settings
            collection_metadata: Collection configurations.
            client: Chroma client. Documentation:
                    https://docs.trychroma.com/reference/js-client#class:-chromaclient
            relevance_score_fn: Function to calculate relevance score from distance.
                    Used only in `similarity_search_with_relevance_scores`
            create_collection_if_not_exists: Whether to create collection
                    if it doesn't exist. Defaults to True.
            **kwargs: Additional keyword arguments.

        Returns:
            a ChromaGraphVectorStore.
        """
        store = cls(
            collection_name=collection_name,
            embedding_function=embedding,
            persist_directory=persist_directory,
            client_settings=client_settings,
            collection_metadata=collection_metadata,
            client=client,
            relevance_score_fn=relevance_score_fn,
            create_collection_if_not_exists=create_collection_if_not_exists,
            **kwargs,
        )
        if ids is None:
            await store.aadd_documents(documents)
        else:
            await store.aadd_documents(documents, ids=ids)
        return store
