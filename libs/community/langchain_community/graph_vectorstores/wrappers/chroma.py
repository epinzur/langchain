from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
)

import chromadb
from chromadb.api.types import IncludeEnum
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import run_in_executor

from langchain_community.graph_vectorstores.interfaces import (
    VectorStoreForGraphInterface,
)

if TYPE_CHECKING:
    import chromadb
    from chromadb.api.types import IncludeEnum
    from langchain_chroma import Chroma


class ChromaVectorStoreForGraph(Chroma, VectorStoreForGraphInterface):
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
        try:
            from langchain_chroma import Chroma  # noqa: F401

        except (ImportError, ModuleNotFoundError):
            raise ImportError(
                "Could not import langchain_chroma python package. "
                "Please install it with `pip install langchain_chroma`."
            )

        super().__init__(
            collection_name=collection_name,
            embedding_function=embedding_function,
            persist_directory=persist_directory,
            client_settings=client_settings,
            collection_metadata=collection_metadata,
            client=client,
            relevance_score_fn=relevance_score_fn,
            create_collection_if_not_exists=create_collection_if_not_exists,
        )

    def _get_safe_embedding(self) -> Embeddings:
        if not self.embeddings:
            msg = "Missing embedding"
            raise ValueError(msg)
        return self.embeddings

    def similarity_search_with_embedding(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> Tuple[List[float], List[Tuple[Document, List[float]]]]:
        """Return docs most similar to the query with embedding.

        Also returns the embedded query vector.

        Args:
            query: Query to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on the metadata to apply.
            **kwargs: Additional keyword arguments.

        Returns:
            (The embedded query vector, The list of (Document, embedding),
            the most similar to the query vector.).
        """
        embedding = self._get_safe_embedding().embed_query(text=query)
        results = self._collection.query(
            query_embeddings=embedding,  # type: ignore
            n_results=k,
            where=filter,  # type: ignore
            include=[
                IncludeEnum.documents,
                IncludeEnum.metadatas,
                IncludeEnum.embeddings,
            ],
            **kwargs,
        )

        docs_with_embeddings: List[Tuple[Document, List[float]]] = [
            (
                Document(
                    page_content=result[0], metadata=result[1] or {}, id=result[2]
                ),
                result[3],
            )
            for result in zip(
                results["documents"][0],  # type: ignore
                results["metadatas"][0],  # type: ignore
                results["ids"][0],  # type: ignore
                results["embeddings"][0],  # type: ignore
            )
        ]
        return embedding, docs_with_embeddings

    async def asimilarity_search_with_embedding(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> Tuple[List[float], List[Tuple[Document, List[float]]]]:
        """Return docs most similar to the query with embedding.

        Also returns the embedded query vector.

        Args:
            query: Query to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on the metadata to apply.
            **kwargs: Additional keyword arguments.

        Returns:
            (The embedded query vector, The list of (Document, embedding),
            the most similar to the query vector.).
        """
        return await run_in_executor(
            None, self.similarity_search_with_embedding, query, k, filter, **kwargs
        )

    def similarity_search_with_embedding_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, List[float]]]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on the metadata to apply.
            **kwargs: Additional keyword arguments.

        Returns:
            List of (Document, embedding), the most similar to the query vector.
        """
        results = self._collection.query(
            query_embeddings=embedding,  # type: ignore
            n_results=k,
            where=filter,  # type: ignore
            include=[
                IncludeEnum.documents,
                IncludeEnum.metadatas,
                IncludeEnum.embeddings,
            ],
            **kwargs,
        )
        return [
            (
                Document(
                    page_content=result[0], metadata=result[1] or {}, id=result[2]
                ),
                result[3],
            )
            for result in zip(
                results["documents"][0],  # type: ignore
                results["metadatas"][0],  # type: ignore
                results["ids"][0],  # type: ignore
                results["embeddings"][0],  # type: ignore
            )
        ]

    async def asimilarity_search_with_embedding_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, List[float]]]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on the metadata to apply.
            **kwargs: Additional keyword arguments.

        Returns:
            List of (Document, embedding), the most similar to the query vector.
        """
        return await run_in_executor(
            None,
            self.similarity_search_with_embedding_by_vector,
            embedding,
            k,
            filter,
            **kwargs,
        )

    def metadata_search(
        self,
        filter: dict[str, Any] | None = None,  # noqa: A002
        n: int = 5,
    ) -> Iterable[Document]:
        """Get document nodes via a metadata search.

        Args:
            filter: the metadata to query for.
            n: the maximum number of documents to return.
        """
        results = self.get(where=filter, limit=n)
        return [
            Document(page_content=result[0], metadata=result[1] or {}, id=result[2])
            for result in zip(
                results["documents"],
                results["metadatas"],
                results["ids"],
            )
        ]

    async def ametadata_search(
        self,
        filter: dict[str, Any] | None = None,  # noqa: A002
        n: int = 5,
    ) -> Iterable[Document]:
        """Get document nodes via a metadata search.

        Args:
            filter: the metadata to query for.
            n: the maximum number of documents to return.
        """
        return await run_in_executor(None, self.metadata_search, filter, n)

    def get_by_document_id(self, document_id: str) -> Document | None:
        """Retrieve a single document node from the graph vector store, given its id.

        Args:
            document_id: The document id

        Returns:
            The the document if it exists. Otherwise None.
        """
        results = self.get(ids=document_id)
        if len(results["documents"]) == 1:
            return Document(
                page_content=results["documents"][0],
                metadata=results["metadatas"][0],
                id=results["ids"][0],
            )
        return None

    async def aget_by_document_id(self, document_id: str) -> Document | None:
        """Retrieve a single document node from the store, given its id.

        Args:
            document_id: The document id

        Returns:
            The the document if it exists. Otherwise None.
        """
        return await run_in_executor(None, self.get_by_document_id, document_id)
