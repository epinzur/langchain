import asyncio
from abc import abstractmethod
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    cast,
)

from langchain_chroma import Chroma
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import run_in_executor
from langchain_core.runnables.config import run_in_executor
from langchain_core.vectorstores import VectorStore
from pydantic import PrivateAttr
from typing_extensions import override

from langchain_community.vectorstores import OpenSearchVectorSearch

METADATA_EMBEDDING_KEY = "__embedding"


class MMRTraversalAdapter(VectorStore):
    _vector_store: VectorStore

    @property
    def _safe_embedding(self) -> Embeddings:
        if not self._vector_store.embeddings:
            msg = "Missing embedding"
            raise ValueError(msg)
        return self._vector_store.embeddings

    def similarity_search_with_embedding(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> Tuple[List[float], List[Document]]:
        """Returns docs (with embeddings) most similar to the query.

        Also returns the embedded query vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on the metadata to apply.
            **kwargs: Additional keyword arguments.

        Returns:
            A tuple of:
                * The embedded query vector
                * List of Documents most similar to the query vector.
                  Documents should have their embedding added to
                  their metadata under the EMBEDDING_KEY key.
        """
        query_embedding = self._safe_embedding.embed_query(text=query)
        docs = self.similarity_search_with_embedding_by_vector(
            embedding=query_embedding,
            k=k,
            filter=filter,
            **kwargs,
        )
        return query_embedding, docs

    async def asimilarity_search_with_embedding(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> Tuple[List[float], List[Document]]:
        """Returns docs (with embeddings) most similar to the query.

        Also returns the embedded query vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on the metadata to apply.
            **kwargs: Additional keyword arguments.

        Returns:
            A tuple of:
                * The embedded query vector
                * List of Documents most similar to the query vector.
                  Documents should have their embedding added to
                  their metadata under the EMBEDDING_KEY key.
        """
        return await run_in_executor(
            None, self.similarity_search_with_embedding, query, k, filter, **kwargs
        )

    @abstractmethod
    def similarity_search_with_embedding_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Returns docs (with embeddings) most similar to the query vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on the metadata to apply.
            **kwargs: Additional keyword arguments.

        Returns:
            List of Documents most similar to the query vector.
                Documents should have their embedding added to
                their metadata under the EMBEDDING_KEY key.
        """

    async def asimilarity_search_with_embedding_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Returns docs (with embeddings) most similar to the query vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on the metadata to apply.
            **kwargs: Additional keyword arguments.

        Returns:
            List of Documents most similar to the query vector.
                Documents should have their embedding added to
                their metadata under the EMBEDDING_KEY key.
        """
        return await run_in_executor(
            None,
            self.similarity_search_with_embedding_by_vector,
            embedding,
            k,
            filter,
            **kwargs,
        )


class OpenSearchMMRTraversalAdapter(MMRTraversalAdapter):
    def __init__(self, vector_store: OpenSearchVectorSearch):
        if vector_store.engine not in ["lucene", "faiss"]:
            msg = (
                f"Invalid engine for MMR Traversal: '{vector_store.engine}'"
                " please instantiate the Open Search Vector Store with"
                " either the 'lucene' or 'faiss' engine"
            )
            raise ValueError(msg)
        self._engine = vector_store.engine
        self._vector_store = vector_store

    def _build_filter(
        self, filter: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]] | None:
        if filter is None:
            return None
        return [
            {
                "terms" if isinstance(value, list) else "term": {
                    f"metadata.{key}.keyword": value
                }
            }
            for key, value in filter.items()
        ]

    def similarity_search_with_embedding_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Returns docs (with embeddings) most similar to the query vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on the metadata to apply.
            **kwargs: Additional keyword arguments.

        Returns:
            List of Documents most similar to the query vector.
                Documents should have their embedding added to
                their metadata under the EMBEDDING_KEY key.
        """
        if filter is not None:
            # use an efficient_filter to collect results that
            # are near the embedding vector until up to 'k'
            # documents that match the filter are found.
            kwargs["efficient_filter"] = {
                "bool": {"must": self._build_filter(filter=filter)}
            }

        docs = self._vector_store.similarity_search_by_vector(
            embedding=embedding,
            k=k,
            metadata_field="*",
            **kwargs,
        )

        # if metadata=="*" on the search, then the document
        # embedding vector and text are included in the
        # document metadata in the returned document.
        #
        # The actual document metadata is moved down into a
        # sub "metadata" key.
        for doc in docs:
            embedding = doc.metadata["vector_field"]
            doc.metadata = doc.metadata["metadata"] or {}
            doc.metadata[METADATA_EMBEDDING_KEY] = embedding

        return docs

    def add_documents(
        self,
        documents: Iterable[Document],
        **kwargs: Any,
    ) -> list[str]:
        """Add document nodes to the graph vector store.

        Args:
            documents: the document nodes to add.
            **kwargs: Additional keyword arguments.
        """
        kwargs["engine"] = self._engine
        return self._vector_store.add_documents(documents, **kwargs)

    @classmethod
    def from_texts(self, *args: Any) -> OpenSearchVectorSearch:
        return OpenSearchVectorSearch.from_texts(*args)

    def similarity_search(self, *args: Any) -> List[Document]:
        return self._vector_store.similarity_search(*args)

    # Delegate all other methods to the underlying vector store
    def __getattr__(self, name: str) -> Any:
        return getattr(self._vector_store, name)
