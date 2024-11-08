"""Chroma DB graph vector store integration."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
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
    Sequence,
    cast,
)

from langchain_core._api import beta, deprecated
from langchain_core.documents import Document

from langchain_community.graph_vectorstores.base import (
    NODE_CLASS_DEPRECATED_SINCE,
    Node,
    _texts_to_documents,
)
from langchain_community.graph_vectorstores.wrappers import ChromaVectorStoreForGraph
from langchain_community.graph_vectorstores.base import GraphVectorStore
from langchain_community.graph_vectorstores.cassandra_base import CassandraGraphVectorStoreBase
from langchain_community.graph_vectorstores.links import (
    METADATA_LINKS_KEY,
    Link,
    get_links,
    incoming_links,
)

ChromaGVS = TypeVar("ChromaGVS", bound="ChromaGraphVectorStore")

if TYPE_CHECKING:
    import chromadb
    from langchain_core.embeddings import Embeddings



logger = logging.getLogger(__name__)

METADATA_INCOMING_LINKS_KEY = "__incoming_links"

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

class ChromaGraphVectorStore(CassandraGraphVectorStoreBase):

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
        links = _deserialize_links(doc.metadata.get(METADATA_LINKS_KEY))
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
        metadata[METADATA_LINKS_KEY] = _serialize_links(links=links)
        # TODO: Could we could put these metadata entries
        # only in the searchable `metadata_s` column?
        for incoming_link in incoming_links(links=links):
            metadata[_metadata_link_key(link=incoming_link)] = _metadata_link_value()
        return metadata

    def add_documents(
        self,
        documents: Iterable[Document],
        **kwargs: Any,
    ) -> list[str]:
        """Run more documents through the embeddings and add to the graph vector store.

        The Links present in the document metadata field `links` will be extracted to
        create the node links.

        Eg if nodes `a` and `b` are connected over a hyperlink `https://some-url`, the
        function call would look like:

        .. code-block:: python

            store.add_documents(
                [
                    Document(
                        id="a",
                        page_content="some text a",
                        metadata={
                            "links": [
                                Link.incoming(kind="hyperlink", tag="http://some-url")
                            ]
                        }
                    ),
                    Document(
                        id="b",
                        page_content="some text b",
                        metadata={
                            "links": [
                                Link.outgoing(kind="hyperlink", tag="http://some-url")
                            ]
                        }
                    ),
                ]

            )

        Args:
            documents: Documents to add to the graph vector store.
                The document's metadata key `links` shall be an iterable of
                :py:class:`~langchain_community.graph_vectorstores.links.Link`.

        Returns:
            List of IDs of the added texts.
        """
        documents = list(self.prep_docs_for_insertion(docs=documents))

        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]
        ids = [d.id if d.id else "" for d in documents]

        return self.vector_store.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
        )

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[Iterable[dict]] = None,
        *,
        ids: Optional[Iterable[str]] = None,
        **kwargs: Any,
    ) -> list[str]:
        """Run more texts through the embeddings and add to the graph vector store.

        The Links present in the metadata field `links` will be extracted to create
        the node links.

        Eg if nodes `a` and `b` are connected over a hyperlink `https://some-url`, the
        function call would look like:

        .. code-block:: python

            store.add_texts(
                ids=["a", "b"],
                texts=["some text a", "some text b"],
                metadatas=[
                    {
                        "links": [
                            Link.incoming(kind="hyperlink", tag="https://some-url")
                        ]
                    },
                    {
                        "links": [
                            Link.outgoing(kind="hyperlink", tag="https://some-url")
                        ]
                    },
                ],
            )

        Args:
            texts: Iterable of strings to add to the graph vector store.
            metadatas: Optional list of metadatas associated with the texts.
                The metadata key `links` shall be an iterable of
                :py:class:`~langchain_community.graph_vectorstores.links.Link`.
            ids: Optional list of IDs associated with the texts.
            **kwargs: vector store specific parameters.

        Returns:
            List of ids from adding the texts into the vector store.
        """

        docs = _texts_to_documents(texts, metadatas, ids)
        return list(self.add_documents(documents=docs, **kwargs))

    @classmethod
    def from_texts(
        cls: Type[ChromaGVS],
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        *,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        embedding_function: Optional[Embeddings] = None,
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
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of IDs associated with the texts.
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
            **kwargs: Additional keyword arguments.

        Returns:
            a ChromaGraphVectorStore.
        """
        store = cls(
            collection_name=collection_name,
            embedding_function=embedding_function,
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
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        *,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        embedding_function: Optional[Embeddings] = None,
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
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of IDs associated with the texts.
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
            **kwargs: Additional keyword arguments.

        Returns:
            a ChromaGraphVectorStore.
        """
        store = cls(
            collection_name=collection_name,
            embedding_function=embedding_function,
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
        ids: Optional[List[str]] = None,
        *,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        embedding_function: Optional[Embeddings] = None,
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
            ids: Optional list of IDs associated with the documents.
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
            **kwargs: Additional keyword arguments.

        Returns:
            a ChromaGraphVectorStore.
        """
        store = cls(
            collection_name=collection_name,
            embedding_function=embedding_function,
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
        ids: Optional[List[str]] = None,
        *,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        embedding_function: Optional[Embeddings] = None,
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
            ids: Optional list of IDs associated with the documents.
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
            **kwargs: Additional keyword arguments.

        Returns:
            a ChromaGraphVectorStore.
        """
        store = cls(
            collection_name=collection_name,
            embedding_function=embedding_function,
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
