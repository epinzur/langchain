from abc import abstractmethod
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    cast,
)

from chromadb.api.types import IncludeEnum
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import run_in_executor
from langchain_core.vectorstores import VectorStore

METADATA_EMBEDDING_KEY = "__embedding"


class MMRTraversalAdapter:
    _base_vector_store: VectorStore

    @property
    def _safe_embedding(self) -> Embeddings:
        if not self._base_vector_store.embeddings:
            msg = "Missing embedding"
            raise ValueError(msg)
        return self._base_vector_store.embeddings

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
                  their metadata under the METADATA_EMBEDDING_KEY key.
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
                  their metadata under the METADATA_EMBEDDING_KEY key.
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
                their metadata under the METADATA_EMBEDDING_KEY key.
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
                their metadata under the METADATA_EMBEDDING_KEY key.
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
    def __init__(self, vector_store: VectorStore):
        from langchain_community.vectorstores import OpenSearchVectorSearch

        self._base_vector_store = vector_store
        self._vector_store = cast(OpenSearchVectorSearch, vector_store)
        if self._vector_store.engine not in ["lucene", "faiss"]:
            msg = (
                f"Invalid engine for MMR Traversal: '{self._vector_store.engine}'"
                " please instantiate the Open Search Vector Store with"
                " either the 'lucene' or 'faiss' engine"
            )
            raise ValueError(msg)

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
        """Returns docs (with embeddings) most similar to the query vector."""
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


class ChromaMMRTraversalAdapter(MMRTraversalAdapter):
    def __init__(self, vector_store: VectorStore):
        try:
            from langchain_chroma import Chroma
        except (ImportError, ModuleNotFoundError):
            msg = "please `pip install langchain-chroma`"
            raise ImportError(msg)

        self._vector_store = cast(Chroma, vector_store)
        self._base_vector_store = vector_store

    def similarity_search_with_embedding_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Returns docs (with embeddings) most similar to the query vector."""
        results = self._vector_store._collection.query(
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

        docs: list[Document] = []
        for result in zip(
            results["documents"][0],  # type: ignore
            results["metadatas"][0],  # type: ignore
            results["ids"][0],  # type: ignore
            results["embeddings"][0],  # type: ignore
        ):
            metadata = result[1] or {}
            metadata[METADATA_EMBEDDING_KEY] = result[3]
            docs.append(
                Document(
                    page_content=result[0],
                    metadata=metadata,
                    id=result[2],
                )
            )
        return docs


class AstraMMRTraversalAdapter(MMRTraversalAdapter):
    def __init__(self, vector_store: VectorStore):
        try:
            from langchain_astradb import AstraDBVectorStore
        except (ImportError, ModuleNotFoundError):
            msg = "please `pip install langchain-astradb`"
            raise ImportError(msg)

        self._vector_store = cast(AstraDBVectorStore, vector_store)

    def _build_docs(
        self, docs_with_embeddings: list[tuple[Document, list[float]]]
    ) -> List[Document]:
        docs: List[Document] = []
        for doc, embedding in docs_with_embeddings:
            doc.metadata[METADATA_EMBEDDING_KEY] = embedding
            docs.append(doc)
        return docs

    def similarity_search_with_embedding(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> Tuple[List[float], List[Document]]:
        """Returns docs (with embeddings) most similar to the query."""
        query_embedding, docs_with_embeddings = (
            self._vector_store.similarity_search_with_embedding(
                query=query,
                k=k,
                filter=filter,
                **kwargs,
            )
        )
        return query_embedding, self._build_docs(
            docs_with_embeddings=docs_with_embeddings
        )

    async def asimilarity_search_with_embedding(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> Tuple[List[float], List[Document]]:
        """Returns docs (with embeddings) most similar to the query."""
        (
            query_embedding,
            docs_with_embeddings,
        ) = await self._vector_store.asimilarity_search_with_embedding(
            query=query,
            k=k,
            filter=filter,
            **kwargs,
        )
        return query_embedding, self._build_docs(
            docs_with_embeddings=docs_with_embeddings
        )

    def similarity_search_with_embedding_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Returns docs (with embeddings) most similar to the query vector."""
        docs_with_embeddings = (
            self._vector_store.similarity_search_with_embedding_by_vector(
                embedding=embedding,
                k=k,
                filter=filter,
                **kwargs,
            )
        )
        return self._build_docs(docs_with_embeddings=docs_with_embeddings)

    async def asimilarity_search_with_embedding_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Returns docs (with embeddings) most similar to the query vector."""
        docs_with_embeddings = (
            await self._vector_store.asimilarity_search_with_embedding_by_vector(
                embedding=embedding,
                k=k,
                filter=filter,
                **kwargs,
            )
        )
        return self._build_docs(docs_with_embeddings=docs_with_embeddings)


class CassandraMMRTraversalAdapter(MMRTraversalAdapter):
    def __init__(self, vector_store: VectorStore):
        from langchain_community.vectorstores import Cassandra

        self._vector_store = cast(Cassandra, vector_store)

    def similarity_search_with_embedding(  # type: ignore
        self, **kwargs: Any
    ) -> Tuple[List[float], List[Document]]:
        return self._vector_store.similarity_search_with_embedding(**kwargs)

    async def asimilarity_search_with_embedding(  # type: ignore
        self, **kwargs: Any
    ) -> Tuple[List[float], List[Document]]:
        return await self._vector_store.asimilarity_search_with_embedding(**kwargs)

    def similarity_search_with_embedding_by_vector(  # type: ignore
        self, **kwargs: Any
    ) -> List[Document]:
        return self._vector_store.similarity_search_with_embedding_by_vector(**kwargs)

    async def asimilarity_search_with_embedding_by_vector(  # type: ignore
        self, **kwargs: Any
    ) -> List[Document]:
        return await self._vector_store.asimilarity_search_with_embedding_by_vector(
            **kwargs
        )