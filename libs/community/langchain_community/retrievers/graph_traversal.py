import asyncio
import warnings
from abc import abstractmethod
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import run_in_executor
from langchain_core.vectorstores import VectorStore
from pydantic import Field, PrivateAttr

BASIC_TYPES = (str, bool, int, float, complex, bytes)


class TraversalAdapter:
    _base_vector_store: VectorStore

    @property
    def _safe_embedding(self) -> Embeddings:
        if not self._base_vector_store.embeddings:
            msg = "Missing embedding"
            raise ValueError(msg)
        return self._base_vector_store.embeddings

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> Tuple[List[float], List[Document]]:
        """Return docs most similar to query.

        Also returns the embedded query vector.

        Args:
            query: Input text.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on the metadata to apply.
            **kwargs: Arguments to pass to the search method.

        Returns:
            List of Documents most similar to the query.
        """
        query_embedding = self._safe_embedding.embed_query(text=query)
        docs = self.similarity_search_by_vector(
            embedding=query_embedding,
            k=k,
            filter=filter,
            **kwargs,
        )
        return query_embedding, docs

    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> Tuple[List[float], List[Document]]:
        """Return docs most similar to query.

        Also returns the embedded query vector.

        Args:
            query: Input text.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on the metadata to apply.
            **kwargs: Arguments to pass to the search method.

        Returns:
            List of Documents most similar to the query.
        """
        return await run_in_executor(
            None, self.similarity_search, query, k, filter, **kwargs
        )

    @abstractmethod
    def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on the metadata to apply.
            **kwargs: Arguments to pass to the search method.

        Returns:
            List of Documents most similar to the query vector.
        """

    async def asimilarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Async return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on the metadata to apply.
            **kwargs: Arguments to pass to the search method.

        Returns:
            List of Documents most similar to the query vector.
        """
        return await run_in_executor(
            None, self.similarity_search_by_vector, embedding, k, filter, **kwargs
        )


class Edge:
    """Represents an edge to all nodes with the given key/value incoming."""
    key: str
    value: Any
    is_denormalized: bool

    def __init__(self, key: str, value: Any, is_denormalized: bool = False) -> None:
        self.key = key
        self.value = value
        self.is_denormalized = is_denormalized

    def __str__(self) -> str:
        return f"Edge({self.key}->{self.value}, is_denormalized={self.is_denormalized})"

    def __repr__(self) -> str:
        return f"Edge(key={self.key}, value={self.value}, is_denormalized={self.is_denormalized})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Edge):
            return NotImplemented
        return (
            self.key == other.key
            and self.value == other.value
            and self.is_denormalized == other.is_denormalized
        )

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Edge):
            return NotImplemented
        return (self.__str__()) < (other.__str__())

    def __hash__(self) -> int:
        return hash((self.key, self.value, self.is_denormalized))


class DocumentCache:
    documents: dict[str, Document] = {}

    def add_document(self, doc: Document, depth_found: int) -> None:
        if doc.id is None:
            msg = "All documents should have ids"
            raise ValueError(msg)
        if doc.id not in self.documents:
            doc.metadata["_depth_found"] = depth_found
            self.documents[doc.id] = doc

    def add_documents(self, docs: Iterable[Document], depth_found: int) -> None:
        for doc in docs:
            self.add_document(doc=doc, depth_found=depth_found)

    def get_by_document_ids(
        self,
        ids: Iterable[str],
    ) -> list[Document]:
        docs: list[Document] = []
        for id in ids:
            if id in self.documents:
                docs.append(self.documents[id])
            else:
                msg = f"unexpected, cache should contain id: {id}"
                raise RuntimeError(msg)
        return docs


# this class uses pydantic, so store and edges
# must be provided at init time.
class GraphTraversalRetriever(BaseRetriever):
    store: TraversalAdapter
    edges: List[Union[str, Tuple[str, str]]]
    _edges: List[Tuple[str,str]] = PrivateAttr(default=[])
    start_k: int = Field(default=4)
    max_depth: int = Field(default=4)
    use_denormalized_metadata: bool = Field(default=False)
    denormalized_path_delimiter: str = Field(default=".")
    denormalized_static_value: Any = Field(default=True)


    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        for edge in self.edges:
            if isinstance(edge, str):
                self._edges.append((edge, edge))
            elif (
                isinstance(edge, tuple)
                and len(edge) == 2
                and all(isinstance(item, str) for item in edge)
            ):
                self._edges.append(edge)
            else:
                raise ValueError(
                    "Invalid type for edge. must be 'str' or 'tuple[str,str]'"
                )

    def _get_relevant_documents(
        self,
        query: str,
        *,
        start_k: int | None = None,
        max_depth: int | None = None,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Get documents relevant to a query.

        Args:
            query: String to find relevant documents for.
            run_manager: The callback handler to use.
            start_k: The number of Documents to return from the initial vector search.
                Defaults to 4.
            max_depth: The maximum depth of edges to traverse. Defaults to 1.
            filter: Optional metadata to filter the results.
            **kwargs: Additional keyword arguments.
        Returns:
            List of relevant documents.
        """
        # Depth 0:
        #   Query for `start_k` document nodes similar to the question.
        #   Retrieve `id` and `outgoing_edges`.
        #
        # Depth 1:
        #   Query for document nodes that have an incoming edge in `outgoing_edges`.
        #   Combine node IDs.
        #   Query for `outgoing_edges` of those "new" node IDs.
        #
        # ...
        start_k = self.start_k if start_k is None else start_k
        max_depth = self.max_depth if max_depth is None else max_depth

        # Map from visited ID to depth
        visited_ids: dict[str, int] = {}

        # Map from visited edge to depth
        visited_edges: dict[Edge, int] = {}

        doc_cache = DocumentCache()

        # this gets set after the call to get the initial nodes
        query_embedding: list[float] = []

        def visit_nodes(d: int, nodes: Iterable[Document]) -> None:
            """Recursively visit document nodes and their outgoing edges."""
            _outgoing_edges = self._gather_outgoing_edges(
                nodes=nodes,
                visited_ids=visited_ids,
                visited_edges=visited_edges,
                d=d,
                depth=max_depth,
            )

            if _outgoing_edges:
                for outgoing_edge in _outgoing_edges:
                    metadata_filter = self._get_metadata_filter(
                        metadata=filter,
                        edge=outgoing_edge,
                    )

                    docs = list(
                        self.store.similarity_search_by_vector(
                            embedding=query_embedding,
                            k=1000,
                            filter=metadata_filter,
                            **kwargs,
                        )
                    )
                    doc_cache.add_documents(docs, d + 1)

                    new_ids_at_next_depth: set[str] = set()
                    for doc in docs:
                        if doc.id is not None:
                            if d < visited_ids.get(doc.id, max_depth):
                                new_ids_at_next_depth.add(doc.id)

                    if new_ids_at_next_depth:
                        nodes = doc_cache.get_by_document_ids(ids=new_ids_at_next_depth)
                        visit_nodes(d=d + 1, nodes=nodes)

        # Start the traversal
        query_embedding, initial_nodes = self.store.similarity_search(
            query=query,
            k=start_k,
            filter=filter,
            **kwargs,
        )
        doc_cache.add_documents(docs=initial_nodes, depth_found=0)
        visit_nodes(d=0, nodes=initial_nodes)

        return doc_cache.get_by_document_ids(ids=visited_ids.keys())

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        start_k: int | None = None,
        depth: int | None = None,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Asynchronously get documents relevant to a query.

        Args:
            query: String to find relevant documents for.
            run_manager: The callback handler to use.
            start_k: The number of Documents to return from the initial vector search.
                Defaults to 4.
            depth: The maximum depth of edges to traverse. Defaults to 1.
            filter: Optional metadata to filter the results.
            **kwargs: Additional keyword arguments.
        Returns:
            List of relevant documents
        """
        # Depth 0:
        #   Query for `start_k` document nodes similar to the question.
        #   Retrieve `content_id` and `outgoing_edges()`.
        #
        # Depth 1:
        #   Query for document nodes that have an incoming edge in`outgoing_edges()`.
        #   Combine node IDs.
        #   Query for `outgoing_edges()` of those "new" node IDs.
        #
        # ...
        start_k = self.start_k if start_k is None else start_k
        depth = self.max_depth if depth is None else depth

        # Map from visited ID to depth
        visited_ids: dict[str, int] = {}

        # Map from visited edge to depth
        visited_edges: dict[Edge, int] = {}

        doc_cache = DocumentCache()

        # this gets set after the call to get the initial nodes
        query_embedding: list[float] = []

        async def visit_nodes(d: int, nodes: Iterable[Document]) -> None:
            """Recursively visit document nodes and their outgoing edges."""
            _outgoing_edges = self._gather_outgoing_edges(
                nodes=nodes,
                visited_ids=visited_ids,
                visited_edges=visited_edges,
                d=d,
                depth=depth,
            )

            if _outgoing_edges:
                metadata_search_tasks = [
                    asyncio.create_task(
                        self.store.asimilarity_search_by_vector(
                            embedding=query_embedding,
                            k=1000,
                            filter=self._get_metadata_filter(
                                metadata=filter, edge=outgoing_edge
                            ),
                            **kwargs,
                        )
                    )
                    for outgoing_edge in _outgoing_edges
                ]

                for search_task in asyncio.as_completed(metadata_search_tasks):
                    docs = await search_task
                    docs = list(docs)
                    doc_cache.add_documents(docs, d + 1)

                    new_ids_at_next_depth: set[str] = set()
                    for doc in docs:
                        if doc.id is not None:
                            if d < visited_ids.get(doc.id, depth):
                                new_ids_at_next_depth.add(doc.id)

                    if new_ids_at_next_depth:
                        nodes = doc_cache.get_by_document_ids(ids=new_ids_at_next_depth)
                        await visit_nodes(d=d + 1, nodes=nodes)

        # Start the traversal
        (
            query_embedding,
            initial_nodes,
        ) = await self.store.asimilarity_search(
            query=query,
            k=start_k,
            filter=filter,
            **kwargs,
        )
        doc_cache.add_documents(docs=initial_nodes, depth_found=0)
        await visit_nodes(d=0, nodes=initial_nodes)

        return doc_cache.get_by_document_ids(ids=visited_ids.keys())

    def _get_outgoing_edges(self, doc: Document) -> set[Edge]:
        edges = set()
        for source_key, target_key in self._edges:
            if source_key in doc.metadata:
                value = doc.metadata[source_key]
                if isinstance(value, BASIC_TYPES):
                    edges.add(Edge(key=target_key, value=value))
                elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
                    if self.use_denormalized_metadata:
                        warnings.warn("Iterable metadata values are supported as edges in denormalized metadata")
                    else:
                        for item in value:
                            if isinstance(item, BASIC_TYPES):
                                edges.add(Edge(key=target_key, value=item))
            elif self.use_denormalized_metadata:
                prefix = f"{source_key}{self.denormalized_path_delimiter}"
                matching_keys: list[str] = [key for key in doc.metadata if key.startswith(prefix)]
                for matching_key in matching_keys:
                    if doc.metadata[matching_key] == self.denormalized_static_value:
                        edges.add(Edge(
                            key=target_key, value=matching_key.removeprefix(prefix), is_denormalized=True,
                        ))
        return edges

    def _gather_outgoing_edges(
        self,
        nodes: Iterable[Document],
        visited_ids: dict[str, int],
        visited_edges: dict[Edge, int],
        d: int,
        depth: int,
    ) -> set[Edge]:
        # Iterate over document nodes, tracking the *new* outgoing edges for this
        # depth. These are edges that are either new, or newly discovered at a
        # lower depth.
        _outgoing_edges: set[Edge] = set()
        for node in nodes:
            if node.id is not None:
                # If this document node is at a closer depth, update visited_ids
                if d <= visited_ids.get(node.id, depth):
                    visited_ids[node.id] = d
                    # If we can continue traversing from this document node,
                    if d < depth:
                        # Record any new (or newly discovered at a lower depth)
                        # edges to the set to traverse.
                        for edge in self._get_outgoing_edges(doc=node):
                            if d <= visited_edges.get(edge, depth):
                                # Record that we'll query this edge at the
                                # given depth, so we don't fetch it again
                                # (unless we find it an earlier depth)
                                visited_edges[edge] = d
                                _outgoing_edges.add(edge)
        return _outgoing_edges

    def _get_metadata_filter(
        self,
        metadata: dict[str, Any] | None = None,
        edge: Edge | None = None,
    ) -> dict[str, Any]:
        """Builds a metadata filter to search for documents

        Args:
            metadata: Any metadata that should be used for hybrid search
            edge: An optional outgoing edge to add to the search
        """
        if edge is None:
            return metadata or {}

        metadata_filter = {} if metadata is None else metadata.copy()
        if edge.is_denormalized:
            metadata_filter[f"{edge.key}{self.denormalized_path_delimiter}{edge.value}"] = self.denormalized_static_value
        else:
            metadata_filter[edge.key] = edge.value

        return metadata_filter


class AstraTraversalAdapter(TraversalAdapter):
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
        for doc, _ in docs_with_embeddings:
            docs.append(doc)
        return docs

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> Tuple[List[float], List[Document]]:
        """Returns docs most similar to the query."""
        query_embedding, docs_with_embeddings = (
            self._vector_store.similarity_search_with_embedding(
                query=query,
                k=k,
                filter=filter,
                **kwargs,
            )
        )
        docs = [doc for doc, _ in docs_with_embeddings]
        return query_embedding, docs

    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> Tuple[List[float], List[Document]]:
        """Returns docs most similar to the query."""
        (
            query_embedding,
            docs_with_embeddings,
        ) = await self._vector_store.asimilarity_search_with_embedding(
            query=query,
            k=k,
            filter=filter,
            **kwargs,
        )
        docs = [doc for doc, _ in docs_with_embeddings]
        return query_embedding, docs

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Returns docs most similar to the query vector."""
        docs_with_embeddings = (
            self._vector_store.similarity_search_with_embedding_by_vector(
                embedding=embedding,
                k=k,
                filter=filter,
                **kwargs,
            )
        )
        return [doc for doc, _ in docs_with_embeddings]

    async def asimilarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Returns docs most similar to the query vector."""
        docs_with_embeddings = (
            await self._vector_store.asimilarity_search_with_embedding_by_vector(
                embedding=embedding,
                k=k,
                filter=filter,
                **kwargs,
            )
        )
        return [doc for doc, _ in docs_with_embeddings]


class CassandraTraversalAdapter(TraversalAdapter):
    def __init__(self, vector_store: VectorStore):
        from langchain_community.vectorstores import Cassandra

        self._vector_store = cast(Cassandra, vector_store)
        self._base_vector_store = vector_store

    def similarity_search_by_vector(  # type: ignore
        self, **kwargs: Any
    ) -> List[Document]:
        return self._vector_store.similarity_search_by_vector(**kwargs)

    async def asimilarity_search_by_vector(  # type: ignore
        self, **kwargs: Any
    ) -> List[Document]:
        return await self._vector_store.asimilarity_search_by_vector(**kwargs)


class ChromaTraversalAdapter(TraversalAdapter):
    def __init__(self, vector_store: VectorStore):
        try:
            from langchain_chroma import Chroma
        except (ImportError, ModuleNotFoundError):
            msg = "please `pip install langchain-chroma`"
            raise ImportError(msg)
        self._vector_store = cast(Chroma, vector_store)
        self._base_vector_store = vector_store

    def similarity_search_by_vector(  # type: ignore
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        if k > self._vector_store._collection.count():
            k = self._vector_store._collection.count()
        return self._vector_store.similarity_search_by_vector(
            embedding=embedding, k=k, filter=filter, **kwargs
        )


class OpenSearchTraversalAdapter(TraversalAdapter):
    def __init__(self, vector_store: VectorStore):
        from langchain_community.vectorstores import OpenSearchVectorSearch

        self._base_vector_store = vector_store
        self._vector_store = cast(OpenSearchVectorSearch, vector_store)
        if self._vector_store.engine not in ["lucene", "faiss"]:
            msg = (
                f"Invalid engine for Traversal: '{self._vector_store.engine}'"
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

    def similarity_search_by_vector(
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

        return self._vector_store.similarity_search_by_vector(
            embedding=embedding,
            k=k,
            **kwargs,
        )
