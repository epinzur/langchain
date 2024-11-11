"""Interface to support Cassandra-based graph vector store integrations."""

from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Tuple,
)

from langchain_core.documents import Document


class CassandraGraphInterface(Protocol):
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

    async def asimilarity_search_with_embedding_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
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

    def get_by_document_id(self, document_id: str) -> Document | None:
        """Retrieve a single document node from the graph vector store, given its id.

        Args:
            document_id: The document id

        Returns:
            The the document if it exists. Otherwise None.
        """

    async def aget_by_document_id(self, document_id: str) -> Document | None:
        """Retrieve a single document node from the store, given its id.

        Args:
            document_id: The document id

        Returns:
            The the document if it exists. Otherwise None.
        """
