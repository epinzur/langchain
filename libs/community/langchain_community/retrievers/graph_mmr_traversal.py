import asyncio
from abc import abstractmethod
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union, cast

from langchain_chroma import Chroma
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables.config import run_in_executor
from langchain_core.vectorstores import VectorStore
from pydantic import PrivateAttr

from langchain_community.vectorstores import OpenSearchVectorSearch


class Edge:
    direction: str = Literal["bi-dir", "in", "out"]
    key: str
    value: Any

    def __init__(self, direction=str, key=str, value=Any):
        self.direction = direction
        self.key = key
        self.value = value

    def __str__(self):
        return f"{self.direction}:{self.key}:{self.value}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Edge):
            return NotImplemented
        return (
            self.direction == other.direction
            and self.key == other.key
            and self.value == other.value
        )

    def __hash__(self) -> int:
        return hash((self.direction, self.key, self.value))

class MMRTraversalAdapter(VectorStore):
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
        query_embedding = self.embeddings.embed_query(text=query)
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
        return await run_in_executor(None,
            self.similarity_search_with_embedding,
            query, k, filter, **kwargs)

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
        return await run_in_executor(None,
            self.similarity_search_with_embedding_by_vector,
            embedding, k, filter, **kwargs)


class GraphVectorStore(VectorStore):
    @abstractmethod
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
        return await run_in_executor(None, self.metadata_search, filter, n)


class ChromaWrapper(GraphVectorStore):
    def __init__(self, vector_store: Chroma):
        self._vector_store = vector_store

    def metadata_search(
        self,
        filter: dict[str, Any] | None = None,  # noqa: A002
        n: int = 5,
    ) -> Iterable[Document]:
        results = self._vector_store.get(where=filter, limit=n)
        return [
            Document(page_content=result[0], metadata=result[1] or {}, id=result[2])
            for result in zip(
                results["documents"],
                results["metadatas"],
                results["ids"],
            )
        ]

    @classmethod
    def from_texts(self, *args: Any) -> Chroma:
        return Chroma.from_texts(*args)

    def similarity_search(self, *args: Any) -> List[Document]:
        return self._vector_store.similarity_search(*args)




class OpenSearchWrapper(GraphVectorStore):
    def __init__(self, vector_store: OpenSearchVectorSearch):
        self._vector_store = vector_store

    def _hit_to_document(self, hit: Any) -> Document:
        return Document(
            page_content=hit["_source"]["text"],
            metadata=hit["_source"]["metadata"],
            id=hit["_id"],
        )

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

    def metadata_search(
        self,
        filter: dict[str, Any] | None = None,  # noqa: A002
        n: int = 5,
    ) -> Iterable[Document]:
        body = {
            "_source": ["text", "metadata"],
            "query": {"bool": {"must": self._build_filter(filter=filter)}},
            "size": n,
        }

        results = self._vector_store.client.search(
            index=self._vector_store.index_name,
            body=body,
        )

        return [self._hit_to_document(hit) for hit in results["hits"]["hits"]]

    @classmethod
    def from_texts(self, *args: Any) -> OpenSearchVectorSearch:
        return OpenSearchVectorSearch.from_texts(*args)

    def similarity_search(self, *args: Any) -> List[Document]:
        return self._vector_store.similarity_search(*args)

    def __getattr__(self, name: str) -> Any:
        # Delegate attribute access to the underlying vector store
        return getattr(self._vector_store, name)


class DocumentCache:
    documents: dict[str, Document] = {}

    def add_document(self, doc: Document) -> None:
        if doc.id is not None:
            self.documents[doc.id] = doc

    def add_documents(self, docs: Iterable[Document]) -> None:
        for doc in docs:
            self.add_document(doc=doc)

    def get_by_document_ids(
        self,
        doc_ids: Iterable[str],
    ) -> list[Document]:
        docs: list[Document] = []
        for doc_id in doc_ids:
            if doc_id in self.documents:
                docs.append(self.documents[doc_id])
            else:
                msg = f"unexpected, cache should contain id: {doc_id}"
                raise RuntimeError(msg)
        return docs


class GraphMMRTraversalRetriever(BaseRetriever):
    vector_store: VectorStore
    edges: List[Union[str, Tuple[str, str]]]
    _edge_lookup: Dict[str, str] = PrivateAttr(default={})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for edge in self.edges:
            if isinstance(edge, str):
                self._edge_lookup[edge] = edge
            elif (
                isinstance(edge, tuple)
                and len(edge) == 2
                and all(isinstance(item, str) for item in edge)
            ):
                self._edge_lookup[edge[0]] = edge[1]
            else:
                raise ValueError(
                    "Invalid type for edge. must be 'str' or 'tuple[str,str]'"
                )

    @property
    def _graph_vector_store(self) -> GraphVectorStore:
        if isinstance(self.vector_store, Chroma):
            return ChromaWrapper(vector_store=self.vector_store)
        elif isinstance(self.vector_store, OpenSearchVectorSearch):
            return OpenSearchWrapper(vector_store=self.vector_store)
        return cast(GraphVectorStore, self.vector_store)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        k: int = 4,
        depth: int = 1,
        filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        """Get documents relevant to a query.

        Args:
            query: String to find relevant documents for.
            run_manager: The callback handler to use.
            k: The number of Documents to return from the initial vector search.
                Defaults to 4.
            depth: The maximum depth of edges to traverse. Defaults to 1.
            filter: Optional metadata to filter the results.
        Returns:
            List of relevant documents.
        """
        # Depth 0:
        #   Query for `k` document nodes similar to the question.
        #   Retrieve `id` and `outgoing_edges`.
        #
        # Depth 1:
        #   Query for document nodes that have an incoming edge in `outgoing_edges`.
        #   Combine node IDs.
        #   Query for `outgoing_edges` of those "new" node IDs.
        #
        # ...

        # Map from visited ID to depth
        visited_ids: dict[str, int] = {}

        # Map from visited edge to depth
        visited_edges: dict[Edge, int] = {}

        doc_cache = DocumentCache()

        def visit_nodes(d: int, nodes: Iterable[Document]) -> None:
            """Recursively visit document nodes and their outgoing edges."""
            _outgoing_edges = self._gather_outgoing_edges(
                nodes=nodes,
                visited_ids=visited_ids,
                visited_edges=visited_edges,
                d=d,
                depth=depth,
            )

            if _outgoing_edges:
                for outgoing_edge in _outgoing_edges:
                    metadata_filter = self._get_metadata_filter(
                        metadata=filter,
                        outgoing_edge=outgoing_edge,
                    )

                    docs = list(
                        self._graph_vector_store.metadata_search(
                            filter=metadata_filter, n=1000
                        )
                    )
                    doc_cache.add_documents(docs)

                    new_ids_at_next_depth: set[str] = set()
                    for doc in docs:
                        if doc.id is not None:
                            if d < visited_ids.get(doc.id, depth):
                                new_ids_at_next_depth.add(doc.id)

                    if new_ids_at_next_depth:
                        nodes = doc_cache.get_by_document_ids(
                            doc_ids=new_ids_at_next_depth
                        )
                        visit_nodes(d=d + 1, nodes=nodes)

        # Start the traversal
        initial_nodes = self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=filter,
        )
        doc_cache.add_documents(docs=initial_nodes)
        visit_nodes(d=0, nodes=initial_nodes)

        return doc_cache.get_by_document_ids(doc_ids=visited_ids)

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        k: int = 4,
        depth: int = 1,
        filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        """Asynchronously get documents relevant to a query.

        Args:
            query: String to find relevant documents for.
            run_manager: The callback handler to use.
            k: The number of Documents to return from the initial vector search.
                Defaults to 4.
            depth: The maximum depth of edges to traverse. Defaults to 1.
            filter: Optional metadata to filter the results.
        Returns:
            List of relevant documents
        """
        # Depth 0:
        #   Query for `k` document nodes similar to the question.
        #   Retrieve `content_id` and `outgoing_edges()`.
        #
        # Depth 1:
        #   Query for document nodes that have an incoming edge in`outgoing_edges()`.
        #   Combine node IDs.
        #   Query for `outgoing_edges()` of those "new" node IDs.
        #
        # ...

        # Map from visited ID to depth
        visited_ids: dict[str, int] = {}

        # Map from visited edge to depth
        visited_edges: dict[Edge, int] = {}

        doc_cache = DocumentCache()

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
                        self._graph_vector_store.ametadata_search(
                            filter=self._get_metadata_filter(
                                metadata=filter, outgoing_edge=outgoing_edge
                            ),
                            n=1000,
                        )
                    )
                    for outgoing_edge in _outgoing_edges
                ]

                for search_task in asyncio.as_completed(metadata_search_tasks):
                    docs = await search_task
                    docs = list(docs)
                    doc_cache.add_documents(docs)

                    new_ids_at_next_depth: set[str] = set()
                    for doc in docs:
                        if doc.id is not None:
                            if d < visited_ids.get(doc.id, depth):
                                new_ids_at_next_depth.add(doc.id)

                    if new_ids_at_next_depth:
                        nodes = doc_cache.get_by_document_ids(
                            doc_ids=new_ids_at_next_depth
                        )
                        await visit_nodes(d=d + 1, nodes=nodes)

        # Start the traversal
        initial_nodes = await self.vector_store.asimilarity_search(
            query=query,
            k=k,
            filter=filter,
        )
        doc_cache.add_documents(docs=initial_nodes)
        await visit_nodes(d=0, nodes=initial_nodes)

        return doc_cache.get_by_document_ids(doc_ids=visited_ids)

    def _get_edges(self, direction: str, key: str, value: Any) -> set[Edge]:
        if isinstance(value, str):
            return {Edge(direction=direction, key=key, value=value)}
        elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            return {Edge(direction=direction, key=key, value=item) for item in value}
        else:
            raise TypeError(
                "Expected a string or an iterable of strings, but got an unsupported type."
            )

    def _get_outgoing_edges(self, doc: Document) -> set[Edge]:
        outgoing_edges = set()
        for edge in self.edges:
            if isinstance(edge, str):
                if edge in doc.metadata:
                    outgoing_edges.update(
                        self._get_edges(
                            direction="bi-dir", key=edge, value=doc.metadata[edge]
                        )
                    )
            elif (
                isinstance(edge, tuple)
                and len(edge) == 2
                and all(isinstance(item, str) for item in edge)
            ):
                if edge[0] in doc.metadata:
                    outgoing_edges.update(
                        self._get_edges(
                            direction="out", key=edge[0], value=doc.metadata[edge[0]]
                        )
                    )
            else:
                raise ValueError(
                    "Invalid type for edge. must be 'str' or 'tuple[str,str]'"
                )
        return outgoing_edges

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
                        edges = self._get_outgoing_edges(doc=node)
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
        outgoing_edge: Edge | None = None,
    ) -> dict[str, Any]:
        """Builds a metadata filter to search for document

        Args:
            metadata: Any metadata that should be used for hybrid search
            outgoing_edge: An optional outgoing edge to add to the search

        Returns:
            The document metadata ready for insertion into the database
        """
        if outgoing_edge is None:
            return metadata or {}

        metadata_filter = {} if metadata is None else metadata.copy()
        if outgoing_edge.direction == "bi-dir":
            metadata_filter[outgoing_edge.key] = outgoing_edge.value
        elif outgoing_edge.direction == "out":
            in_key = self._edge_lookup[outgoing_edge.key]
            metadata_filter[in_key] = outgoing_edge.value

        return metadata_filter
