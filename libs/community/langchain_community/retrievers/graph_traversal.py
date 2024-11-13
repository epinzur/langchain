import asyncio
from typing import TYPE_CHECKING, Any, AsyncIterable, Iterable, Iterator, Protocol, Tuple, Union

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables.config import run_in_executor
from langchain_core.vectorstores import VectorStore

from abc import abstractmethod

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
    ) -> Iterator[Document]:
        for doc_id in doc_ids:
            if doc_id in self.documents:
                yield self.documents[doc_id]
            else:
                msg = f"unexpected, cache should contain id: {doc_id}"
                raise RuntimeError(msg)

class GraphTraversalRetriever(BaseRetriever):
    vector_store: VectorStore
    edges: Iterable[Union[str, Tuple[str, str]]]
    k: int

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        """Get documents relevant to a query.

        Args:
            query: String to find relevant documents for.
            run_manager: The callback handler to use.
        Returns:
            List of relevant documents.
        """

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        """Asynchronously get documents relevant to a query.

        Args:
            query: String to find relevant documents for
            run_manager: The callback handler to use
        Returns:
            List of relevant documents
        """

    def traversal_search(  # noqa: C901
        self,
        query: str,
        *,
        k: int = 4,
        depth: int = 1,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Iterable[Document]:
        """Retrieve documents from this knowledge store.

        First, `k` nodes are retrieved using a vector search for the `query` string.
        Then, additional nodes are discovered up to the given `depth` from those
        starting nodes.

        Args:
            query: The query string.
            k: The number of Documents to return from the initial vector search.
                Defaults to 4.
            depth: The maximum depth of edges to traverse. Defaults to 1.
            filter: Optional metadata to filter the results.
            **kwargs: Additional keyword arguments.

        Returns:
            Collection of retrieved documents.
        """
        # Depth 0:
        #   Query for `k` document nodes similar to the question.
        #   Retrieve `content_id` and `outgoing_links()`.
        #
        # Depth 1:
        #   Query for document nodes that have an incoming link in `outgoing_links()`.
        #   Combine node IDs.
        #   Query for `outgoing_links()` of those "new" node IDs.
        #
        # ...

        # Map from visited ID to depth
        visited_ids: dict[str, int] = {}

        # Map from visited link to depth
        visited_links: dict[str, int] = {}

        doc_cache = DocumentCache()

        def visit_nodes(d: int, nodes: Iterable[Document]) -> None:
            """Recursively visit document nodes and their outgoing links."""
            _outgoing_links = self._gather_outgoing_links(
                nodes=nodes,
                visited_ids=visited_ids,
                visited_links=visited_links,
                d=d,
                depth=depth,
            )

            if _outgoing_links:
                for outgoing_link in _outgoing_links:
                    metadata_filter = self._get_metadata_filter(
                        metadata=filter,
                        outgoing_link=outgoing_link,
                    )

                    docs = list(self.metadata_search(filter=metadata_filter, n=1000))
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
        initial_nodes = self.similarity_search(
            query=query,
            k=k,
            filter=filter,
        )
        doc_cache.add_documents(docs=initial_nodes)
        visit_nodes(d=0, nodes=initial_nodes)

        return doc_cache.get_by_document_ids(doc_ids=visited_ids)


    async def atraversal_search(  # noqa: C901
        self,
        query: str,
        *,
        k: int = 4,
        depth: int = 1,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[Document]:
        """Retrieve documents from this knowledge store.

        First, `k` nodes are retrieved using a vector search for the `query` string.
        Then, additional nodes are discovered up to the given `depth` from those
        starting nodes.

        Args:
            query: The query string.
            k: The number of Documents to return from the initial vector search.
                Defaults to 4.
            depth: The maximum depth of edges to traverse. Defaults to 1.
            filter: Optional metadata to filter the results.
            **kwargs: Additional keyword arguments.

        Returns:
            Collection of retrieved documents.
        """
        # Depth 0:
        #   Query for `k` document nodes similar to the question.
        #   Retrieve `content_id` and `outgoing_links()`.
        #
        # Depth 1:
        #   Query for document nodes that have an incoming link in`outgoing_links()`.
        #   Combine node IDs.
        #   Query for `outgoing_links()` of those "new" node IDs.
        #
        # ...

        # Map from visited ID to depth
        visited_ids: dict[str, int] = {}

        # Map from visited link to depth
        visited_links: dict[str, int] = {}

        doc_cache = DocumentCache()

        async def visit_nodes(d: int, nodes: Iterable[Document]) -> None:
            """Recursively visit document nodes and their outgoing links."""
            _outgoing_links = self._gather_outgoing_links(
                nodes=nodes,
                visited_ids=visited_ids,
                visited_links=visited_links,
                d=d,
                depth=depth,
            )

            if _outgoing_links:
                metadata_search_tasks = [
                    asyncio.create_task(
                        self.ametadata_search(
                            filter=self._get_metadata_filter(
                                metadata=filter, outgoing_link=outgoing_link
                            ),
                            n=1000,
                        )
                    )
                    for outgoing_link in _outgoing_links
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
        initial_nodes = await self.asimilarity_search(
            query=query,
            k=k,
            filter=filter,
        )
        doc_cache.add_documents(docs=initial_nodes)
        await visit_nodes(d=0, nodes=initial_nodes)

        for doc in doc_cache.get_by_document_ids(doc_ids=visited_ids):
            yield doc