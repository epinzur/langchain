"""Generic Cassandra graph vector store integration."""

from __future__ import annotations

import asyncio
import secrets
from abc import abstractmethod
from typing import Any, AsyncIterable, AsyncIterator, Iterable, Iterator, Sequence, cast

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from typing_extensions import override

from langchain_community.graph_vectorstores.base import GraphVectorStore
from langchain_community.graph_vectorstores.helpers.cassandra_interface import (
    CassandraVectorStoreForGraphInterface,
)
from langchain_community.graph_vectorstores.helpers.mmr_helper import MmrHelper
from langchain_community.graph_vectorstores.links import Link, get_links, outgoing_links

METADATA_EMBEDDING_KEY = "__embedding"


def get_embedding(doc: Document) -> list[float]:
    """Get the embedding from a document."""
    return doc.metadata.get(METADATA_EMBEDDING_KEY, [])


def set_embedding(doc: Document, embedding: list[float]) -> None:
    """Set the embedding on a document."""
    doc.metadata[METADATA_EMBEDDING_KEY] = embedding


def clear_embedding(doc: Document) -> None:
    doc.metadata.pop(METADATA_EMBEDDING_KEY, None)


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


class CassandraGraphVectorStoreBase(GraphVectorStore):
    vector_store: VectorStore
    cassandra_vector_store: CassandraVectorStoreForGraphInterface

    def __init__(self, vector_store: VectorStore):
        super().__init__()
        self.vector_store = vector_store
        self.cassandra_vector_store = cast(
            CassandraVectorStoreForGraphInterface, vector_store
        )

    @abstractmethod
    def get_metadata_for_insertion(self, doc: Document) -> dict[str, Any]:
        """Prepares the links in a document by serializing them to metadata.

        Args:
            doc: Document to prepare

        Returns:
            The document metadata ready for insertion into the database
        """

    @abstractmethod
    def restore_links(self, doc: Document) -> Document:
        """Restores links in a document by deserializing them from metadata.

        Args:
            doc: Document to restore

        Returns:
            The document ready for use in the graph vector store
        """

    @abstractmethod
    def get_metadata_filter(
        self,
        metadata: dict[str, Any] | None = None,
        outgoing_link: Link | None = None,
    ) -> dict[str, Any]:
        """Builds a metadata filter to search for document

        Args:
            metadata: Any metadata that should be used for hybrid search
            outgoing_link: An optional outgoing link to add to the search

        Returns:
            The document metadata ready for insertion into the database
        """

    def restore_links_in_docs(self, docs: Iterable[Document]) -> Iterable[Document]:
        """Restores links in the documents by deserializing them from metadata.

        Args:
            docs: Documents to restore

        Returns:
            The documents ready for use in the graph vector store
        """
        for doc in docs:
            yield self.restore_links(doc=doc)

    def prep_docs_for_insertion(self, docs: Iterable[Document]) -> Iterable[Document]:
        """Prepares the links in documents by serializing them to metadata.

        Args:
            docs: Documents to prepare

        Returns:
            The documents ready for insertion into the database
        """
        for doc in docs:
            if doc.id is None:
                doc.id = secrets.token_hex(8)
            doc.metadata = self.get_metadata_for_insertion(doc=doc)
            yield doc

    @override
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
        docs = self.prep_docs_for_insertion(docs=documents)
        return self.vector_store.add_documents(list(docs), **kwargs)

    @override
    async def aadd_documents(
        self,
        documents: Iterable[Document],
        **kwargs: Any,
    ) -> list[str]:
        """Add document nodes to the graph vector store.

        Args:
            documents: the document nodes to add.
            **kwargs: Additional keyword arguments.
        """
        docs = self.prep_docs_for_insertion(docs=documents)
        return [
            id for id in await self.vector_store.aadd_documents(list(docs), **kwargs)
        ]

    @override
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Retrieve document nodes from this graph vector store.

        Args:
            query: The query string.
            k: The number of Documents to return. Defaults to 4.
            filter: Optional metadata to filter the results.
            **kwargs: Additional keyword arguments.

        Returns:
            Collection of retrieved documents.
        """
        docs = self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=filter,
            **kwargs,
        )
        return list(self.restore_links_in_docs(docs=docs))

    @override
    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Retrieve documents nodes from this graph vector store.

        Args:
            query: The query string.
            k: The number of Documents to return. Defaults to 4.
            filter: Optional metadata to filter the results.
            **kwargs: Additional keyword arguments.

        Returns:
            Collection of retrieved documents.
        """
        docs = await self.vector_store.asimilarity_search(
            query=query,
            k=k,
            filter=filter,
            **kwargs,
        )
        return list(self.restore_links_in_docs(docs=docs))

    @override
    def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return document nodes most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on the metadata to apply.
            **kwargs: Additional arguments are ignored.

        Returns:
            The list of Documents most similar to the embedding vector.
        """
        docs = self.vector_store.similarity_search_by_vector(
            embedding,
            k=k,
            filter=filter,
            **kwargs,
        )
        return list(self.restore_links_in_docs(docs=docs))

    @override
    async def asimilarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return document nodes most similar to the embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on the metadata to apply.
            **kwargs: Additional arguments are ignored.

        Returns:
            The list of Documents most similar to the embedding vector.
        """
        docs = await self.vector_store.asimilarity_search_by_vector(
            embedding,
            k=k,
            filter=filter,
            **kwargs,
        )
        return list(self.restore_links_in_docs(docs=docs))

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
        docs = self.cassandra_vector_store.metadata_search(
            filter=filter or {},
            n=n,
        )
        return self.restore_links_in_docs(docs=docs)

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
        docs = await self.cassandra_vector_store.ametadata_search(
            filter=filter or {},
            n=n,
        )
        return self.restore_links_in_docs(docs=docs)

    def get_by_document_id(self, document_id: str) -> Document | None:
        """Retrieve a single document node from the graph vector store, given its id.

        Args:
            document_id: The document id

        Returns:
            The the document if it exists. Otherwise None.
        """
        doc = self.cassandra_vector_store.get_by_document_id(document_id=document_id)
        return self.restore_links(doc=doc) if doc is not None else None

    async def aget_by_document_id(self, document_id: str) -> Document | None:
        """Retrieve a single document node from the store, given its id.

        Args:
            document_id: The document id

        Returns:
            The the document if it exists. Otherwise None.
        """
        doc = await self.cassandra_vector_store.aget_by_document_id(
            document_id=document_id
        )
        return self.restore_links(doc=doc) if doc is not None else None

    def _return_docs_with_embeddings(
        self, rows: list[tuple[Document, list[float]]]
    ) -> list[Document]:
        docs: list[Document] = []
        for doc, embedding in rows:
            set_embedding(doc=doc, embedding=embedding)
            docs.append(self.restore_links(doc=doc))
        return docs

    def similarity_search_with_embedding(
        self,
        query: str,
        k: int,
        filter: dict[str, Any] | None = None,  # noqa: A002
    ) -> tuple[list[float], list[Document]]:
        (
            query_embedding,
            rows,
        ) = self.cassandra_vector_store.similarity_search_with_embedding(
            query=query,
            k=k,
            filter=filter,
        )
        return query_embedding, self._return_docs_with_embeddings(rows=rows)

    async def asimilarity_search_with_embedding(
        self,
        query: str,
        k: int,
        filter: dict[str, Any] | None = None,  # noqa: A002
    ) -> tuple[list[float], list[Document]]:
        (
            query_embedding,
            rows,
        ) = await self.cassandra_vector_store.asimilarity_search_with_embedding(
            query=query,
            k=k,
            filter=filter,
        )
        return query_embedding, self._return_docs_with_embeddings(rows=rows)

    def similarity_search_with_embedding_by_vector(
        self,
        embedding: list[float],
        k: int,
        filter: dict[str, Any] | None = None,  # noqa: A002
    ) -> list[Document]:
        rows = self.cassandra_vector_store.similarity_search_with_embedding_by_vector(
            embedding=embedding,
            k=k,
            filter=filter,
        )
        return self._return_docs_with_embeddings(rows=rows)

    async def asimilarity_search_with_embedding_by_vector(
        self,
        embedding: list[float],
        k: int,
        filter: dict[str, Any] | None = None,  # noqa: A002
    ) -> list[Document]:
        cvs = self.cassandra_vector_store
        rows = await cvs.asimilarity_search_with_embedding_by_vector(
            embedding=embedding,
            k=k,
            filter=filter,
        )
        return self._return_docs_with_embeddings(rows=rows)

    def get_candidate_embeddings(
        self, nodes: Iterable[Document], outgoing_links_map: dict[str, set[Link]]
    ) -> dict[str, list[float]]:
        """Returns a map of doc.id to doc embedding.

        Only returns document nodes not yet in the outgoing_links_map.
        Updates the outgoing_links_map with the new document node links.
        """

        candidates: dict[str, list[float]] = {}
        for node in nodes:
            if node.id is not None and node.id not in outgoing_links_map:
                outgoing_links_map[node.id] = set(
                    outgoing_links(links=get_links(doc=node))
                )
                candidates[node.id] = get_embedding(doc=node)
        return candidates

    @override
    def mmr_traversal_search(  # noqa: C901
        self,
        query: str,
        *,
        initial_roots: Sequence[str] = (),
        k: int = 4,
        depth: int = 2,
        fetch_k: int = 100,
        adjacent_k: int = 10,
        lambda_mult: float = 0.5,
        score_threshold: float = float("-inf"),
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Iterable[Document]:
        """Retrieve document nodes from this graph vector store using MMR-traversal.

        This strategy first retrieves the top `fetch_k` results by similarity to
        the question. It then selects the top `k` results based on
        maximum-marginal relevance using the given `lambda_mult`.

        At each step, it considers the (remaining) documents from `fetch_k` as
        well as any documents connected by edges to a selected document
        retrieved based on similarity (a "root").

        Args:
            query: The query string to search for.
            initial_roots: Optional list of document IDs to use for initializing search.
                The top `adjacent_k` nodes adjacent to each initial root will be
                included in the set of initial candidates. To fetch only in the
                neighborhood of these nodes, set `fetch_k = 0`.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of initial Documents to fetch via similarity.
                Will be added to the nodes adjacent to `initial_roots`.
                Defaults to 100.
            adjacent_k: Number of adjacent Documents to fetch.
                Defaults to 10.
            depth: Maximum depth of a node (number of edges) from a node
                retrieved via similarity. Defaults to 2.
            lambda_mult: Number between 0 and 1 that determines the degree
                of diversity among the results with 0 corresponding to maximum
                diversity and 1 to minimum diversity. Defaults to 0.5.
            score_threshold: Only documents with a score greater than or equal
                this threshold will be chosen. Defaults to -infinity.
            filter: Optional metadata to filter the results.
            **kwargs: Additional keyword arguments.
        """
        doc_cache = DocumentCache()

        # For each unselected node, stores the outgoing links.
        outgoing_links_map: dict[str, set[Link]] = {}
        visited_links: set[Link] = set()

        def fetch_initial_candidates() -> tuple[list[float], dict[str, list[float]]]:
            """Gets the embedded query and the set of initial candidates.

            If fetch_k is zero, there will be no initial candidates.
            """
            query_embedding, initial_nodes = self._get_initial(
                query=query,
                doc_cache=doc_cache,
                fetch_k=fetch_k,
                filter=filter,
            )
            return query_embedding, self.get_candidate_embeddings(
                nodes=initial_nodes, outgoing_links_map=outgoing_links_map
            )

        def fetch_neighborhood_candidates(
            neighborhood: Sequence[str],
        ) -> dict[str, list[float]]:
            nonlocal outgoing_links_map, visited_links

            # Put the neighborhood into the outgoing links, to avoid adding it
            # to the candidate set in the future.
            outgoing_links_map.update(
                {content_id: set() for content_id in neighborhood}
            )

            # Initialize the visited_links with the set of outgoing links from the
            # neighborhood. This prevents re-visiting them.
            visited_links = set()

            for doc in doc_cache.get_by_document_ids(doc_ids=neighborhood):
                visited_links.update(outgoing_links(links=get_links(doc=doc)))

            # Fetch the candidates.
            adjacent_nodes = self._get_adjacent(
                links=visited_links,
                query_embedding=query_embedding,
                k_per_link=adjacent_k,
                filter=filter,
                doc_cache=doc_cache,
            )

            return self.get_candidate_embeddings(
                nodes=adjacent_nodes, outgoing_links_map=outgoing_links_map
            )

        query_embedding, initial_candidates = fetch_initial_candidates()
        helper = MmrHelper(
            k=k,
            query_embedding=query_embedding,
            lambda_mult=lambda_mult,
            score_threshold=score_threshold,
        )
        helper.add_candidates(candidates=initial_candidates)

        if initial_roots:
            neighborhood_candidates = fetch_neighborhood_candidates(initial_roots)
            helper.add_candidates(candidates=neighborhood_candidates)

        # Tracks the depth of each candidate.
        depths = {candidate_id: 0 for candidate_id in helper.candidate_ids()}

        # Select the best item, K times.
        for _ in range(k):
            selected_id = helper.pop_best()

            if selected_id is None:
                break

            next_depth = depths[selected_id] + 1
            if next_depth < depth:
                # If the next document nodes would not exceed the depth limit, find the
                # adjacent document nodes.

                # Find the links linked to from the selected id.
                selected_outgoing_links = outgoing_links_map.pop(selected_id)

                # Don't re-visit already visited links.
                selected_outgoing_links.difference_update(visited_links)

                # Find the document nodes with incoming links from those links.
                adjacent_nodes = self._get_adjacent(
                    links=selected_outgoing_links,
                    query_embedding=query_embedding,
                    k_per_link=adjacent_k,
                    filter=filter,
                    doc_cache=doc_cache,
                )

                # Record the selected_outgoing_links as visited.
                visited_links.update(selected_outgoing_links)

                for adjacent_node in adjacent_nodes:
                    if (
                        adjacent_node.id is not None
                        and adjacent_node.id not in outgoing_links_map
                    ):
                        if next_depth < depths.get(adjacent_node.id, depth + 1):
                            # If this is a new shortest depth, or there was no
                            # previous depth, update the depths. This ensures that
                            # when we discover a node we will have the shortest
                            # depth available.
                            #
                            # NOTE: No effort is made to traverse from nodes that
                            # were previously selected if they become reachable via
                            # a shorter path via nodes selected later. This is
                            # currently "intended", but may be worth experimenting
                            # with.
                            depths[adjacent_node.id] = next_depth

                new_candidates = self.get_candidate_embeddings(
                    nodes=adjacent_nodes, outgoing_links_map=outgoing_links_map
                )
                helper.add_candidates(new_candidates)

        docs = doc_cache.get_by_document_ids(doc_ids=helper.selected_ids)

        for doc, similarity_score, mmr_score in zip(
            docs,
            helper.selected_similarity_scores,
            helper.selected_mmr_scores,
        ):
            doc.metadata["similarity_score"] = similarity_score
            doc.metadata["mmr_score"] = mmr_score
            clear_embedding(doc)
            yield doc

    @override
    async def ammr_traversal_search(  # noqa: C901
        self,
        query: str,
        *,
        initial_roots: Sequence[str] = (),
        k: int = 4,
        depth: int = 2,
        fetch_k: int = 100,
        adjacent_k: int = 10,
        lambda_mult: float = 0.5,
        score_threshold: float = float("-inf"),
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[Document]:
        """Retrieve documents from this graph store using MMR-traversal.

        This strategy first retrieves the top `fetch_k` results by similarity to
        the question. It then selects the top `k` results based on
        maximum-marginal relevance using the given `lambda_mult`.

        At each step, it considers the (remaining) documents from `fetch_k` as
        well as any documents connected by edges to a selected document
        retrieved based on similarity (a "root").

        Args:
            query: The query string to search for.
            initial_roots: Optional list of document IDs to use for initializing search.
                The top `adjacent_k` nodes adjacent to each initial root will be
                included in the set of initial candidates. To fetch only in the
                neighborhood of these nodes, set `fetch_k = 0`.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of initial Documents to fetch via similarity.
                Will be added to the nodes adjacent to `initial_roots`.
                Defaults to 100.
            adjacent_k: Number of adjacent Documents to fetch.
                Defaults to 10.
            depth: Maximum depth of a node (number of edges) from a node
                retrieved via similarity. Defaults to 2.
            lambda_mult: Number between 0 and 1 that determines the degree
                of diversity among the results with 0 corresponding to maximum
                diversity and 1 to minimum diversity. Defaults to 0.5.
            score_threshold: Only documents with a score greater than or equal
                this threshold will be chosen. Defaults to -infinity.
            filter: Optional metadata to filter the results.
            **kwargs: Additional keyword arguments.
        """
        # For each unselected node, stores the outgoing links.
        outgoing_links_map: dict[str, set[Link]] = {}
        visited_links: set[Link] = set()

        doc_cache = DocumentCache()

        async def fetch_initial_candidates() -> (
            tuple[list[float], dict[str, list[float]]]
        ):
            """Gets the embedded query and the set of initial candidates.

            If fetch_k is zero, there will be no initial candidates.
            """

            query_embedding, initial_nodes = await self._aget_initial(
                query=query,
                doc_cache=doc_cache,
                fetch_k=fetch_k,
                filter=filter,
            )

            return query_embedding, self.get_candidate_embeddings(
                nodes=initial_nodes, outgoing_links_map=outgoing_links_map
            )

        async def fetch_neighborhood_candidates(
            neighborhood: Sequence[str],
        ) -> dict[str, list[float]]:
            nonlocal outgoing_links_map, visited_links

            # Put the neighborhood into the outgoing links, to avoid adding it
            # to the candidate set in the future.
            outgoing_links_map.update(
                {content_id: set() for content_id in neighborhood}
            )

            async def fetch_documents(
                doc_ids: Sequence[str],
            ) -> AsyncIterator[Document]:
                for doc_id in doc_ids:
                    doc = await self.aget_by_document_id(document_id=doc_id)
                    if doc is not None:
                        yield doc

            # Initialize the visited_links with the set of outgoing links from the
            # neighborhood. This prevents re-visiting them.
            visited_links = set()

            async for doc in fetch_documents(doc_ids=neighborhood):
                doc_cache.add_document(doc=doc)
                visited_links.update(outgoing_links(links=get_links(doc=doc)))

            # Fetch the candidates.
            adjacent_nodes = await self._aget_adjacent(
                links=visited_links,
                query_embedding=query_embedding,
                k_per_link=adjacent_k,
                filter=filter,
                doc_cache=doc_cache,
            )

            return self.get_candidate_embeddings(
                nodes=adjacent_nodes, outgoing_links_map=outgoing_links_map
            )

        query_embedding, initial_candidates = await fetch_initial_candidates()
        helper = MmrHelper(
            k=k,
            query_embedding=query_embedding,
            lambda_mult=lambda_mult,
            score_threshold=score_threshold,
        )
        helper.add_candidates(candidates=initial_candidates)

        if initial_roots:
            neighborhood_candidates = await fetch_neighborhood_candidates(initial_roots)
            helper.add_candidates(candidates=neighborhood_candidates)

        # Tracks the depth of each candidate.
        depths = {candidate_id: 0 for candidate_id in helper.candidate_ids()}

        # Select the best item, K times.
        for _ in range(k):
            selected_id = helper.pop_best()

            if selected_id is None:
                break

            next_depth = depths[selected_id] + 1
            if next_depth < depth:
                # If the next document nodes would not exceed the depth limit, find the
                # adjacent document nodes.

                # Find the links linked to from the selected id.
                selected_outgoing_links = outgoing_links_map.pop(selected_id)

                # Don't re-visit already visited links.
                selected_outgoing_links.difference_update(visited_links)

                # Find the document nodes with incoming links from those links.
                adjacent_nodes = await self._aget_adjacent(
                    links=selected_outgoing_links,
                    query_embedding=query_embedding,
                    k_per_link=adjacent_k,
                    filter=filter,
                    doc_cache=doc_cache,
                )

                # Record the selected_outgoing_links as visited.
                visited_links.update(selected_outgoing_links)

                for adjacent_node in adjacent_nodes:
                    if (
                        adjacent_node.id is not None
                        and adjacent_node.id not in outgoing_links_map
                    ):
                        if next_depth < depths.get(adjacent_node.id, depth + 1):
                            # If this is a new shortest depth, or there was no
                            # previous depth, update the depths. This ensures that
                            # when we discover a node we will have the shortest
                            # depth available.
                            #
                            # NOTE: No effort is made to traverse from nodes that
                            # were previously selected if they become reachable via
                            # a shorter path via nodes selected later. This is
                            # currently "intended", but may be worth experimenting
                            # with.
                            depths[adjacent_node.id] = next_depth

                new_candidates = self.get_candidate_embeddings(
                    nodes=adjacent_nodes, outgoing_links_map=outgoing_links_map
                )
                helper.add_candidates(new_candidates)

        docs = doc_cache.get_by_document_ids(doc_ids=helper.selected_ids)

        for doc, similarity_score, mmr_score in zip(
            docs,
            helper.selected_similarity_scores,
            helper.selected_mmr_scores,
        ):
            doc.metadata["similarity_score"] = similarity_score
            doc.metadata["mmr_score"] = mmr_score
            clear_embedding(doc)
            yield doc

    def _get_initial(
        self,
        query: str,
        doc_cache: DocumentCache,
        fetch_k: int,
        filter: dict[str, Any] | None = None,  # noqa: A002
    ) -> tuple[list[float], list[Document]]:
        query_embedding, initial_docs = self.similarity_search_with_embedding(
            query=query, k=fetch_k, filter=filter
        )
        doc_cache.add_documents(docs=initial_docs)
        return query_embedding, initial_docs

    async def _aget_initial(
        self,
        query: str,
        doc_cache: DocumentCache,
        fetch_k: int,
        filter: dict[str, Any] | None = None,  # noqa: A002
    ) -> tuple[list[float], list[Document]]:
        query_embedding, initial_docs = await self.asimilarity_search_with_embedding(
            query=query, k=fetch_k, filter=filter
        )
        doc_cache.add_documents(docs=initial_docs)
        return query_embedding, initial_docs

    def _get_adjacent(
        self,
        links: set[Link],
        query_embedding: list[float],
        doc_cache: DocumentCache,
        k_per_link: int | None = None,
        filter: dict[str, Any] | None = None,  # noqa: A002
    ) -> Iterable[Document]:
        """Return the target docs with incoming links from any of the given links.

        Args:
            links: The links to look for.
            query_embedding: The query embedding. Used to rank target docs.
            doc_cache: A cache of retrieved docs. This will be added to.
            k_per_link: The number of target docs to fetch for each link.
            filter: Optional metadata to filter the results.

        Returns:
            Iterable of adjacent edges.
        """
        targets: dict[str, Document] = {}

        for link in links:
            docs = self.similarity_search_with_embedding_by_vector(
                embedding=query_embedding,
                k=k_per_link or 10,
                filter=self.get_metadata_filter(metadata=filter, outgoing_link=link),
            )
            doc_cache.add_documents(docs)
            targets.update({doc.id: doc for doc in docs if doc.id is not None})

        # TODO: Consider a combined limit based on the similarity and/or
        # predicated MMR score?
        return targets.values()

    async def _aget_adjacent(
        self,
        links: set[Link],
        query_embedding: list[float],
        doc_cache: DocumentCache,
        k_per_link: int | None = None,
        filter: dict[str, Any] | None = None,  # noqa: A002
    ) -> Iterable[Document]:
        """Returns document nodes with incoming links from any of the given links.

        Args:
            links: The links to look for.
            query_embedding: The query embedding. Used to rank target nodes.
            doc_cache: A cache of retrieved docs. This will be added to.
            k_per_link: The number of target nodes to fetch for each link.
            filter: Optional metadata to filter the results.

        Returns:
            Iterable of adjacent edges.
        """
        targets: dict[str, Document] = {}

        tasks = [
            self.asimilarity_search_with_embedding_by_vector(
                embedding=query_embedding,
                k=k_per_link or 10,
                filter=self.get_metadata_filter(metadata=filter, outgoing_link=link),
            )
            for link in links
        ]

        # Process each task as it completes
        for completed_task in asyncio.as_completed(tasks):
            docs = await completed_task
            doc_cache.add_documents(docs)
            targets.update({doc.id: doc for doc in docs if doc.id is not None})

        # TODO: Consider a combined limit based on the similarity and/or
        # predicated MMR score?
        return targets.values()

    @override
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
        visited_links: dict[Link, int] = {}

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
                    metadata_filter = self.get_metadata_filter(
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

    @override
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
        visited_links: dict[Link, int] = {}

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
                            filter=self.get_metadata_filter(
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

    def _gather_outgoing_links(
        self,
        nodes: Iterable[Document],
        visited_ids: dict[str, int],
        visited_links: dict[Link, int],
        d: int,
        depth: int,
    ) -> set[Link]:
        # Iterate over document nodes, tracking the *new* outgoing links for this
        # depth. These are links that are either new, or newly discovered at a
        # lower depth.
        _outgoing_links: set[Link] = set()
        for node in nodes:
            if node.id is not None:
                # If this document node is at a closer depth, update visited_ids
                if d <= visited_ids.get(node.id, depth):
                    visited_ids[node.id] = d

                    # If we can continue traversing from this document node,
                    if d < depth:
                        # Record any new (or newly discovered at a lower depth)
                        # links to the set to traverse.
                        for link in outgoing_links(links=get_links(doc=node)):
                            if d <= visited_links.get(link, depth):
                                # Record that we'll query this link at the
                                # given depth, so we don't fetch it again
                                # (unless we find it an earlier depth)
                                visited_links[link] = d
                                _outgoing_links.add(link)
        return _outgoing_links
