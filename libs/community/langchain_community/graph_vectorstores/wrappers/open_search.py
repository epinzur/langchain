from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
)


from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import run_in_executor

from langchain_community.graph_vectorstores.interfaces import (
    VectorStoreForGraphInterface,
)


from langchain_community.vectorstores.opensearch_vector_search import OpenSearchVectorSearch


class OpenSearchVectorStoreForGraph(OpenSearchVectorSearch, VectorStoreForGraphInterface):
    _LANGCHAIN_DEFAULT_COLLECTION_NAME = "langchain"

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
        docs_with_embeddings = self.similarity_search_with_embedding_by_vector(
            embedding=embedding,
            k=k,
            filter=filter,
            **kwargs,
        )
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
        ## TODO: Fix the filter after we know how to do metadata searching.
        docs = self.similarity_search_by_vector(
            embedding=embedding,
            k = k,
            boolean_filter = filter, # this probably needs adjusting
            metadata_field = "*",
            **kwargs,
        )

        print(docs)

        return [
            (
                Document(
                    page_content=doc.page_content, metadata=doc.metadata["metadata"] or {}, id=doc.id
                ),
                doc.metadata["vector_field"],
            )
            for doc in docs
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
        results = self

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
