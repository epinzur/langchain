import asyncio
import dataclasses
import warnings
from abc import abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import run_in_executor
from langchain_core.vectorstores import VectorStore
from numpy.typing import NDArray
from pydantic import Field, PrivateAttr

from langchain_community.utils.math import cosine_similarity

if TYPE_CHECKING:
    from numpy.typing import NDArray

BASIC_TYPES = (str, bool, int, float, complex, bytes)
METADATA_EMBEDDING_KEY = "__embedding"
NEG_INF = float("-inf")


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
            query: Input text.
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
            query: Input text.
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

    @abstractmethod
    def get(
        self,
        ids: Sequence[str],
        /,
        **kwargs: Any,
    ) -> list[Document]:
        """Get documents by id.

        Fewer documents may be returned than requested if some IDs are not found or
        if there are duplicated IDs.

        Users should not assume that the order of the returned documents matches
        the order of the input IDs. Instead, users should rely on the ID field of the
        returned documents.

        This method should **NOT** raise exceptions if no documents are found for
        some IDs.

        Args:
            ids: List of IDs to get.
            kwargs: Additional keyword arguments. These are up to the implementation.

        Returns:
            List[Document]: List of documents that were found.
        """

    async def aget(
        self,
        ids: Sequence[str],
        /,
        **kwargs: Any,
    ) -> list[Document]:
        """Get documents by id.

        Fewer documents may be returned than requested if some IDs are not found or
        if there are duplicated IDs.

        Users should not assume that the order of the returned documents matches
        the order of the input IDs. Instead, users should rely on the ID field of the
        returned documents.

        This method should **NOT** raise exceptions if no documents are found for
        some IDs.

        Args:
            ids: List of IDs to get.
            kwargs: Additional keyword arguments. These are up to the implementation.

        Returns:
            List[Document]: List of documents that were found.
        """
        return await run_in_executor(
            None,
            self.get,
            ids,
            **kwargs,
        )


@dataclasses.dataclass
class EmbeddedDocument:
    doc: Document

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EmbeddedDocument):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

    @property
    def id(self) -> str:
        if self.doc.id is None:
            msg = "All documents should have ids"
            raise ValueError(msg)
        return self.doc.id

    @property
    def metadata(self) -> dict[str, Any]:
        """Get the metadata from the document."""
        return self.doc.metadata

    @property
    def embedding(self) -> list[float]:
        """Get the embedding from the document."""
        return self.doc.metadata.get(METADATA_EMBEDDING_KEY, [])

    def document(self) -> Document:
        """Get the underlying document with the embedding removed."""
        self.doc.metadata.pop(METADATA_EMBEDDING_KEY, None)
        return self.doc


@dataclasses.dataclass
class _Candidate:
    doc: Document
    similarity: float
    weighted_similarity: float
    weighted_redundancy: float
    score: float = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self.score = self.weighted_similarity - self.weighted_redundancy

    @property
    def id(self) -> str:
        if self.doc.id is None:
            msg = "All documents should have ids"
            raise ValueError(msg)
        return self.doc.id

    def update_redundancy(self, new_weighted_redundancy: float) -> None:
        if new_weighted_redundancy > self.weighted_redundancy:
            self.weighted_redundancy = new_weighted_redundancy
            self.score = self.weighted_similarity - self.weighted_redundancy


def _emb_to_ndarray(embedding: list[float]) -> NDArray[np.float32]:
    emb_array = np.array(embedding, dtype=np.float32)
    if emb_array.ndim == 1:
        emb_array = np.expand_dims(emb_array, axis=0)
    return emb_array


class MmrHelper:
    """Helper for executing an MMR traversal query.

    Args:
        query_embedding: The embedding of the query to use for scoring.
        lambda_mult: Number between 0 and 1 that determines the degree
            of diversity among the results with 0 corresponding to maximum
            diversity and 1 to minimum diversity. Defaults to 0.5.
        score_threshold: Only documents with a score greater than or equal
            this threshold will be chosen. Defaults to -infinity.
    """

    dimensions: int
    """Dimensions of the embedding."""

    query_embedding: NDArray[np.float32]
    """Embedding of the query as a (1,dim) ndarray."""

    lambda_mult: float
    """Number between 0 and 1.

    Determines the degree of diversity among the results with 0 corresponding to
    maximum diversity and 1 to minimum diversity."""

    lambda_mult_complement: float
    """1 - lambda_mult."""

    score_threshold: float
    """Only documents with a score greater than or equal to this will be chosen."""

    selected_ids: list[str]
    """List of selected IDs (in selection order)."""

    selected_embeddings: NDArray[np.float32]
    """(N, dim) ndarray with a row for each selected node."""

    candidate_id_to_index: dict[str, int]
    """Dictionary of candidate IDs to indices in candidates and candidate_embeddings."""
    candidates: list[_Candidate]
    """List containing information about candidates.

    Same order as rows in `candidate_embeddings`.
    """
    candidate_embeddings: NDArray[np.float32]
    """(N, dim) ndarray with a row for each candidate."""

    best_score: float
    best_id: str | None

    def __init__(
        self,
        k: int,
        query_embedding: list[float],
        lambda_mult: float = 0.5,
        score_threshold: float = NEG_INF,
    ) -> None:
        """Create a new Traversal MMR helper."""
        self.query_embedding = _emb_to_ndarray(query_embedding)
        self.dimensions = self.query_embedding.shape[1]

        self.lambda_mult = lambda_mult
        self.lambda_mult_complement = 1 - lambda_mult
        self.score_threshold = score_threshold

        self.selected_ids = []

        # List of selected embeddings (in selection order).
        self.selected_embeddings = np.ndarray((k, self.dimensions), dtype=np.float32)

        self.candidate_id_to_index = {}

        # List of the candidates.
        self.candidates = []
        # numpy n-dimensional array of the candidate embeddings.
        self.candidate_embeddings = np.ndarray((0, self.dimensions), dtype=np.float32)

        self.best_score = NEG_INF
        self.best_id = None

    def candidate_ids(self) -> Iterable[str]:
        """Return the IDs of the candidates."""
        return self.candidate_id_to_index.keys()

    def _already_selected_embeddings(self) -> NDArray[np.float32]:
        """Return the selected embeddings sliced to the already assigned values."""
        selected = len(self.selected_ids)
        return np.vsplit(self.selected_embeddings, [selected])[0]

    def _pop_candidate(
        self, candidate_id: str
    ) -> tuple[Document, float, NDArray[np.float32]]:
        """Pop the candidate with the given ID.

        Returns:
            The document, similarity score, and embedding of the candidate.
        """
        # Get the embedding for the id.
        index = self.candidate_id_to_index.pop(candidate_id)
        candidate = self.candidates[index]
        if candidate.id != candidate_id:
            msg = (
                "ID in self.candidate_id_to_index doesn't match the ID of the "
                "corresponding index in self.candidates"
            )
            raise ValueError(msg)
        embedding: NDArray[np.float32] = self.candidate_embeddings[index].copy()

        # Swap that index with the last index in the candidates and
        # candidate_embeddings.
        last_index = self.candidate_embeddings.shape[0] - 1

        similarity = 0.0
        if index == last_index:
            # Already the last item. We don't need to swap.
            similarity = self.candidates.pop().similarity
        else:
            self.candidate_embeddings[index] = self.candidate_embeddings[last_index]

            similarity = self.candidates[index].similarity

            old_last = self.candidates.pop()
            self.candidates[index] = old_last
            self.candidate_id_to_index[old_last.id] = index

        self.candidate_embeddings = np.vsplit(self.candidate_embeddings, [last_index])[
            0
        ]

        return candidate.doc, similarity, embedding

    def pop_best(self) -> Document | None:
        """Select and pop the best item being considered.

        Updates the consideration set based on it.

        Returns:
            A tuple containing the ID of the best item.
        """
        if self.best_id is None or self.best_score < self.score_threshold:
            return None

        # Get the selection and remove from candidates.
        selected_id = self.best_id
        selected_doc, selected_similarity, selected_embedding = self._pop_candidate(
            selected_id
        )

        # Add the ID and embedding to the selected information.
        selection_index = len(self.selected_ids)
        self.selected_ids.append(selected_id)
        self.selected_embeddings[selection_index] = selected_embedding

        # Set the scores in the doc metadata
        selected_doc.metadata["_similarity_score"] = selected_similarity
        selected_doc.metadata["_mmr_score"] = self.best_score

        # Reset the best score / best ID.
        self.best_score = NEG_INF
        self.best_id = None

        # Update the candidates redundancy, tracking the best node.
        if self.candidate_embeddings.shape[0] > 0:
            similarity = cosine_similarity(
                self.candidate_embeddings, np.expand_dims(selected_embedding, axis=0)
            )
            for index, candidate in enumerate(self.candidates):
                candidate.update_redundancy(similarity[index][0])
                if candidate.score > self.best_score:
                    self.best_score = candidate.score
                    self.best_id = candidate.id

        return selected_doc

    def add_candidates(self, candidates: dict[str, EmbeddedDocument], depth_found: int) -> None:
        """Add candidates to the consideration set."""
        # Determine the keys to actually include.
        # These are the candidates that aren't already selected
        # or under consideration.
        include_ids_set = set(candidates.keys())
        include_ids_set.difference_update(self.selected_ids)
        include_ids_set.difference_update(self.candidate_id_to_index.keys())
        include_ids = list(include_ids_set)

        # Now, build up a matrix of the remaining candidate embeddings.
        # And add them to the
        new_embeddings: NDArray[np.float32] = np.ndarray(
            (
                len(include_ids),
                self.dimensions,
            )
        )
        offset = self.candidate_embeddings.shape[0]
        for index, candidate_id in enumerate(include_ids):
            if candidate_id in include_ids:
                self.candidate_id_to_index[candidate_id] = offset + index
                new_embeddings[index] = candidates[candidate_id].embedding

        # Compute the similarity to the query.
        similarity = cosine_similarity(new_embeddings, self.query_embedding)

        # Compute the distance metrics of all of pairs in the selected set with
        # the new candidates.
        redundancy = cosine_similarity(
            new_embeddings, self._already_selected_embeddings()
        )
        for index, candidate_id in enumerate(include_ids):
            max_redundancy = 0.0
            if redundancy.shape[0] > 0:
                max_redundancy = redundancy[index].max()
            candidate = _Candidate(
                doc=candidates[candidate_id].document(),
                similarity=similarity[index][0],
                weighted_similarity=self.lambda_mult * similarity[index][0],
                weighted_redundancy=self.lambda_mult_complement * max_redundancy,
            )
            candidate.doc.metadata["_depth_found"] = depth_found
            self.candidates.append(candidate)

            if candidate.score >= self.best_score:
                self.best_score = candidate.score
                self.best_id = candidate.id

        # Add the new embeddings to the candidate set.
        self.candidate_embeddings = np.vstack(
            (
                self.candidate_embeddings,
                new_embeddings,
            )
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


# this class uses pydantic, so store and edges
# must be provided at init time.
class GraphMMRTraversalRetriever(BaseRetriever):
    store: MMRTraversalAdapter
    edges: List[Union[str, Tuple[str, str]]]
    _edges: List[Tuple[str,str]] = PrivateAttr(default=[])
    k: int = Field(default=4)
    depth: int = Field(default=2)
    fetch_k: int = Field(default=100)
    adjacent_k: int = Field(default=10)
    lambda_mult: float = Field(default=0.5)
    score_threshold: float = Field(default=float("-inf"))
    use_denormalized_metadata: bool = Field(default=False)
    denormalized_path_delimiter: str = Field(default=".")
    denormalized_static_value: Any = Field(default=True)

    def __init__(self, **kwargs: Any) -> None:
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
        initial_roots: Sequence[str] = (),
        k: int | None = None,
        depth: int | None = None,
        fetch_k: int | None = None,
        adjacent_k: int | None = None,
        lambda_mult: float | None = None,
        score_threshold: float | None = None,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
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
        k = self.k if k is None else k
        depth = self.depth if depth is None else depth
        fetch_k = self.fetch_k if fetch_k is None else fetch_k
        adjacent_k = self.adjacent_k if adjacent_k is None else adjacent_k
        lambda_mult = self.lambda_mult if lambda_mult is None else lambda_mult
        score_threshold = (
            self.score_threshold if score_threshold is None else score_threshold
        )

        # For each unselected node, stores the outgoing edges.
        outgoing_edges_map: dict[str, set[Edge]] = {}
        visited_edges: set[Edge] = set()

        def fetch_initial_candidates() -> (
            tuple[list[float], dict[str, EmbeddedDocument]]
        ):
            """Gets the embedded query and the set of initial candidates.

            If fetch_k is zero, there will be no initial candidates.
            """
            query_embedding, initial_nodes = self._get_initial(
                query=query,
                fetch_k=fetch_k,
                filter=filter,
                **kwargs,
            )
            return query_embedding, self._get_candidate_embeddings(
                nodes=initial_nodes, outgoing_edges_map=outgoing_edges_map
            )

        def fetch_neighborhood_candidates(
            neighborhood: Sequence[str],
        ) -> dict[str, EmbeddedDocument]:
            nonlocal outgoing_edges_map, visited_edges

            # Put the neighborhood into the outgoing edges, to avoid adding it
            # to the candidate set in the future.
            outgoing_edges_map.update(
                {content_id: set() for content_id in neighborhood}
            )

            # Initialize the visited_edges with the set of outgoing edges from the
            # neighborhood. This prevents re-visiting them.
            visited_edges = set()
            for doc in self.store.get(neighborhood):
                visited_edges.update(
                    self._get_outgoing_edges(doc=EmbeddedDocument(doc=doc))
                )

            # Fetch the candidates.
            adjacent_nodes = self._get_adjacent(
                edges=visited_edges,
                query_embedding=query_embedding,
                k_per_edge=adjacent_k,
                filter=filter,
                **kwargs,
            )

            return self._get_candidate_embeddings(
                nodes=adjacent_nodes, outgoing_edges_map=outgoing_edges_map
            )

        query_embedding, initial_candidates = fetch_initial_candidates()
        helper = MmrHelper(
            k=k,
            query_embedding=query_embedding,
            lambda_mult=lambda_mult,
            score_threshold=score_threshold,
        )
        helper.add_candidates(candidates=initial_candidates, depth_found=0)

        if initial_roots:
            neighborhood_candidates = fetch_neighborhood_candidates(initial_roots)
            helper.add_candidates(candidates=neighborhood_candidates, depth_found=0)

        # Tracks the depth of each candidate.
        depths = {candidate_id: 0 for candidate_id in helper.candidate_ids()}

        # Select the best item, K times.
        selected_docs: list[Document] = []
        for _ in range(k):
            selected_doc = helper.pop_best()
            if selected_doc is None or selected_doc.id is None:
                break

            selected_docs.append(selected_doc)

            next_depth = depths[selected_doc.id] + 1
            if next_depth < depth:
                # If the next document nodes would not exceed the depth limit, find the
                # adjacent document nodes.

                # Find the edges edged to from the selected id.
                selected_outgoing_edges = outgoing_edges_map.pop(selected_doc.id)

                # Don't re-visit already visited edges.
                selected_outgoing_edges.difference_update(visited_edges)

                # Find the document nodes with incoming edges from those edges.
                adjacent_nodes = self._get_adjacent(
                    edges=selected_outgoing_edges,
                    query_embedding=query_embedding,
                    k_per_edge=adjacent_k,
                    filter=filter,
                    **kwargs,
                )

                # Record the selected_outgoing_edges as visited.
                visited_edges.update(selected_outgoing_edges)

                for adjacent_node in adjacent_nodes:
                    if (
                        adjacent_node.id is not None
                        and adjacent_node.id not in outgoing_edges_map
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

                new_candidates = self._get_candidate_embeddings(
                    nodes=adjacent_nodes, outgoing_edges_map=outgoing_edges_map
                )
                helper.add_candidates(new_candidates, depth_found=next_depth)

        return selected_docs

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        initial_roots: Sequence[str] = (),
        k: int | None = None,
        depth: int | None = None,
        fetch_k: int | None = None,
        adjacent_k: int | None = None,
        lambda_mult: float | None = None,
        score_threshold: float | None = None,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
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
        k = self.k if k is None else k
        depth = self.depth if depth is None else depth
        fetch_k = self.fetch_k if fetch_k is None else fetch_k
        adjacent_k = self.adjacent_k if adjacent_k is None else adjacent_k
        lambda_mult = self.lambda_mult if lambda_mult is None else lambda_mult
        score_threshold = (
            self.score_threshold if score_threshold is None else score_threshold
        )

        # For each unselected node, stores the outgoing edges.
        outgoing_edges_map: dict[str, set[Edge]] = {}
        visited_edges: set[Edge] = set()

        async def fetch_initial_candidates() -> (
            tuple[list[float], dict[str, EmbeddedDocument]]
        ):
            """Gets the embedded query and the set of initial candidates.

            If fetch_k is zero, there will be no initial candidates.
            """

            query_embedding, initial_nodes = await self._aget_initial(
                query=query,
                fetch_k=fetch_k,
                filter=filter,
                **kwargs,
            )

            return query_embedding, self._get_candidate_embeddings(
                nodes=initial_nodes, outgoing_edges_map=outgoing_edges_map
            )

        async def fetch_neighborhood_candidates(
            neighborhood: Sequence[str],
        ) -> dict[str, EmbeddedDocument]:
            nonlocal outgoing_edges_map, visited_edges

            # Put the neighborhood into the outgoing edges, to avoid adding it
            # to the candidate set in the future.
            outgoing_edges_map.update(
                {content_id: set() for content_id in neighborhood}
            )

            # Initialize the visited_edges with the set of outgoing edges from the
            # neighborhood. This prevents re-visiting them.
            visited_edges = set()
            for doc in await self.store.aget(neighborhood):
                visited_edges.update(
                    self._get_outgoing_edges(doc=EmbeddedDocument(doc=doc))
                )

            # Fetch the candidates.
            adjacent_nodes = await self._aget_adjacent(
                edges=visited_edges,
                query_embedding=query_embedding,
                k_per_edge=adjacent_k,
                filter=filter,
                **kwargs,
            )

            return self._get_candidate_embeddings(
                nodes=adjacent_nodes, outgoing_edges_map=outgoing_edges_map
            )

        query_embedding, initial_candidates = await fetch_initial_candidates()
        helper = MmrHelper(
            k=k,
            query_embedding=query_embedding,
            lambda_mult=lambda_mult,
            score_threshold=score_threshold,
        )
        helper.add_candidates(candidates=initial_candidates, depth_found=0)

        if initial_roots:
            neighborhood_candidates = await fetch_neighborhood_candidates(initial_roots)
            helper.add_candidates(candidates=neighborhood_candidates, depth_found=0)

        # Tracks the depth of each candidate.
        depths = {candidate_id: 0 for candidate_id in helper.candidate_ids()}

        # Select the best item, K times.
        selected_docs: list[Document] = []
        for _ in range(k):
            selected_doc = helper.pop_best()

            if selected_doc is None or selected_doc.id is None:
                break

            selected_docs.append(selected_doc)

            next_depth = depths[selected_doc.id] + 1
            if next_depth < depth:
                # If the next document nodes would not exceed the depth limit, find the
                # adjacent document nodes.

                # Find the edges edgeed to from the selected id.
                selected_outgoing_edges = outgoing_edges_map.pop(selected_doc.id)

                # Don't re-visit already visited edges.
                selected_outgoing_edges.difference_update(visited_edges)

                # Find the document nodes with incoming edges from those edges.
                adjacent_nodes = await self._aget_adjacent(
                    edges=selected_outgoing_edges,
                    query_embedding=query_embedding,
                    k_per_edge=adjacent_k,
                    filter=filter,
                    **kwargs,
                )

                # Record the selected_outgoing_edges as visited.
                visited_edges.update(selected_outgoing_edges)

                for adjacent_node in adjacent_nodes:
                    if (
                        adjacent_node.id is not None
                        and adjacent_node.id not in outgoing_edges_map
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

                new_candidates = self._get_candidate_embeddings(
                    nodes=adjacent_nodes, outgoing_edges_map=outgoing_edges_map
                )
                helper.add_candidates(new_candidates, depth_found=next_depth)

        return selected_docs

    def _get_candidate_embeddings(
        self,
        nodes: Iterable[EmbeddedDocument],
        outgoing_edges_map: dict[str, set[Edge]],
    ) -> dict[str, EmbeddedDocument]:
        """Returns a map of doc.id to doc embedding.

        Only returns document nodes not yet in the outgoing_edges_map.
        Updates the outgoing_edges_map with the new document node edges.
        """
        candidates: dict[str, EmbeddedDocument] = {}
        for node in nodes:
            if node.id not in outgoing_edges_map:
                outgoing_edges_map[node.id] = self._get_outgoing_edges(doc=node)
                candidates[node.id] = node
        return candidates

    def _get_initial(
        self,
        query: str,
        fetch_k: int,
        filter: dict[str, Any] | None = None,  # noqa: A002
        **kwargs: Any,
    ) -> tuple[list[float], list[EmbeddedDocument]]:
        query_embedding, docs = self.store.similarity_search_with_embedding(
            query=query,
            k=fetch_k,
            filter=filter,
            **kwargs,
        )
        return query_embedding, [EmbeddedDocument(doc=doc) for doc in docs]

    async def _aget_initial(
        self,
        query: str,
        fetch_k: int,
        filter: dict[str, Any] | None = None,  # noqa: A002
        **kwargs: Any,
    ) -> tuple[list[float], list[EmbeddedDocument]]:
        (
            query_embedding,
            docs,
        ) = await self.store.asimilarity_search_with_embedding(
            query=query,
            k=fetch_k,
            filter=filter,
            **kwargs,
        )
        return query_embedding, [EmbeddedDocument(doc=doc) for doc in docs]

    def _get_adjacent(
        self,
        edges: set[Edge],
        query_embedding: list[float],
        k_per_edge: int | None = None,
        filter: dict[str, Any] | None = None,  # noqa: A002
        **kwargs: Any,
    ) -> set[EmbeddedDocument]:
        """Return the target docs with incoming edges from any of the given edges.

        Args:
            edges: The edges to look for.
            query_embedding: The query embedding. Used to rank target docs.
            doc_cache: A cache of retrieved docs. This will be added to.
            k_per_edge: The number of target docs to fetch for each edge.
            filter: Optional metadata to filter the results.

        Returns:
            Iterable of adjacent edges.
        """
        results: set[EmbeddedDocument] = set()
        for edge in edges:
            docs = self.store.similarity_search_with_embedding_by_vector(
                embedding=query_embedding,
                k=k_per_edge or 10,
                filter=self._get_metadata_filter(
                    metadata=filter, edge=edge
                ),
                **kwargs,
            )
            results.update({EmbeddedDocument(doc=doc) for doc in docs})
        return results

    async def _aget_adjacent(
        self,
        edges: set[Edge],
        query_embedding: list[float],
        k_per_edge: int | None = None,
        filter: dict[str, Any] | None = None,  # noqa: A002
        **kwargs: Any,
    ) -> set[EmbeddedDocument]:
        """Returns document nodes with incoming edges from any of the given edges.

        Args:
            edges: The edges to look for.
            query_embedding: The query embedding. Used to rank target nodes.
            doc_cache: A cache of retrieved docs. This will be added to.
            k_per_edge: The number of target nodes to fetch for each edge.
            filter: Optional metadata to filter the results.

        Returns:
            Iterable of adjacent edges.
        """

        tasks = [
            self.store.asimilarity_search_with_embedding_by_vector(
                embedding=query_embedding,
                k=k_per_edge or 10,
                filter=self._get_metadata_filter(
                    metadata=filter, edge=edge
                ),
                **kwargs,
            )
            for edge in edges
        ]

        results: set[EmbeddedDocument] = set()
        for completed_task in asyncio.as_completed(tasks):
            docs = await completed_task
            results.update({EmbeddedDocument(doc=doc) for doc in docs})
        return results

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

    def get(self, ids: Sequence[str], /, **kwargs: Any) -> list[Document]:
        """Get documents by id."""
        docs: list[Document] = []
        for id in ids:
            doc = self._vector_store.get_by_document_id(id, **kwargs)
            if doc is not None:
                docs.append(doc)
        return docs

    async def aget(self, ids: Sequence[str], /, **kwargs: Any) -> list[Document]:
        """Get documents by id."""
        docs: list[Document] = []
        for id in ids:
            doc = await self._vector_store.aget_by_document_id(id, **kwargs)
            if doc is not None:
                docs.append(doc)
        return docs


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

    def get(self, ids: Sequence[str], /, **kwargs: Any) -> list[Document]:
        """Get documents by id."""
        docs: list[Document] = []
        for id in ids:
            doc = self._vector_store.get_by_document_id(id, **kwargs)
            if doc is not None:
                docs.append(doc)
        return docs

    async def aget(self, ids: Sequence[str], /, **kwargs: Any) -> list[Document]:
        """Get documents by id."""
        docs: list[Document] = []
        for id in ids:
            doc = await self._vector_store.aget_by_document_id(id, **kwargs)
            if doc is not None:
                docs.append(doc)
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
        try:
            from chromadb.api.types import IncludeEnum
        except (ImportError, ModuleNotFoundError):
            msg = "please `pip install chromadb`"
            raise ImportError(msg)

        if k > self._vector_store._collection.count():
            k = self._vector_store._collection.count()

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

    def get(self, ids: Sequence[str], /, **kwargs: Any) -> list[Document]:
        """Get documents by id."""
        results = self._vector_store.get(ids=list(ids), **kwargs)
        return [
            Document(
                page_content=text,
                metadata=metadata,
                id=id,
            )
            for (text, metadata, id) in zip(
                results["documents"], results["metadatas"], results["ids"]
            )
        ]


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

    def get(self, ids: Sequence[str], /, **kwargs: Any) -> list[Document]:
        """Get documents by id."""
        try:
            from opensearchpy.exceptions import NotFoundError
        except (ImportError, ModuleNotFoundError):
            msg = "please `pip install opensearch-py`."
            raise ImportError(msg)

        docs: list[Document] = []
        for id in ids:
            try:
                hit = self._vector_store.client.get(
                    index=self._vector_store.index_name,
                    id=id,
                    _source_includes=["text", "metadata"],
                    **kwargs,
                )
                docs.append(
                    Document(
                        page_content=hit["_source"]["text"],
                        metadata=hit["_source"]["metadata"],
                        id=hit["_id"],
                    )
                )
            except NotFoundError:
                pass
        return docs
