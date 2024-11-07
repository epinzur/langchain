"""Opensearch graph vector store integration."""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
from dataclasses import asdict, is_dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    cast,
)

from langchain_core._api import beta, deprecated
from langchain_core.documents import Document
from typing_extensions import override

from langchain_community.graph_vectorstores.base import (
    NODE_CLASS_DEPRECATED_SINCE,
    GraphVectorStore,
    Node,
)
from langchain_community.graph_vectorstores.interfaces.cassandra import (
    CassandraGraphInterface,
)
from langchain_community.graph_vectorstores.links import (
    METADATA_LINKS_KEY,
    Link,
    get_links,
    incoming_links,
    outgoing_links,
)
from langchain_community.graph_vectorstores.document_cache import DocumentCache
from langchain_community.vectorstores import OpenSearchVectorSearch

OSGVST = TypeVar("OSGVST", bound="OpenSearchGraphVectorStore")

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


# region Link serialization and deserialization
def _serialize_links(links: list[Link]) -> str:
    class SetAndLinkEncoder(json.JSONEncoder):
        def default(self, obj: Any) -> Any:
            if not isinstance(obj, type) and is_dataclass(obj):
                return asdict(obj)

            if isinstance(obj, Iterable):
                return list(obj)

            # Let the base class default method raise the TypeError
            return super().default(obj)

    return json.dumps(links, cls=SetAndLinkEncoder)


def _deserialize_links(json_blob: str | None) -> set[Link]:
    return {
        Link(kind=link["kind"], direction=link["direction"], tag=link["tag"])
        for link in cast(list[dict[str, Any]], json.loads(json_blob or "[]"))
    }


def _doc_to_node(doc: Document) -> Node:
    #### Question for Eric - I already have the document's links deserialized.
    metadata = doc.metadata.copy()
    # links = metadata.get(METADATA_LINKS_KEY)
    # metadata[METADATA_LINKS_KEY] = links

    ### Question for Eric -- is the expecatation that node has the serialized links in the metadata?
    return Node(
        id=doc.id,
        text=doc.page_content,
        metadata=metadata,
        links=metadata.get(METADATA_LINKS_KEY),
    )


@beta()
class OpenSearchGraphVectorStore(GraphVectorStore):
    """OpenSearchGraphVectorStore is a class that extends GraphVectorStore to provide
    integration with OpenSearch for storing and searching vectorized documents.

    Attributes:
        embedding (Embeddings): The embedding function to use for vectorizing documents.
        index_name (str): The name of the OpenSearch index.
        opensearch_url (str): The URL of the OpenSearch instance.
        auth (tuple): HTTP authentication credentials for OpenSearch.
        use_ssl (bool): Whether to use SSL for the OpenSearch connection.
        verify_certs (bool): Whether to verify SSL certificates.
        ssl_show_warn (bool): Whether to show SSL warnings.
        reset_index (bool): Whether to reset the OpenSearch index on initialization.
        vector_store (OpenSearchVectorSearch): The OpenSearch vector store instance.

    Methods:
        embeddings: Returns the embedding function.
        add_documents: Adds a sequence of documents to the vector store.
        aadd_documents: Asynchronously adds a sequence of documents to the vector store.
        add_content_graph: Adds the content of a ContentGraph to the vector store.
        aadd_content_graph: Asynchronously adds the content of a ContentGraph to the vector store.
        similarity_search: Performs a similarity search on the OpenSearch vector store.
        similarity_search_with_score: Performs a similarity search and returns documents with their scores.
        get_documents: Retrieves and processes a specified number of documents from OpenSearch.
        from_texts: Creates nodes from texts and adds them to the vector store.
        mmr_traversal_search: Performs a Maximal Marginal Relevance (MMR) traversal search.
        traversal_search: Performs a basic traversal search.
        _restore_links: Restores the links in a document by deserializing them from metadata.
        search_by_id: Retrieves a single document from the store by its document ID.
        asearch_by_id: Asynchronously retrieves a single document from the store by its document ID.
        search_by_metadata: Synchronously searches the index by metadata fields.
        asearch_by_metadata: Asynchronously searches the index by metadata fields.
        add_nodes: Adds nodes to the graph store.
        aadd_nodes: Asynchronously adds nodes to the graph store.
        delete: Deletes documents by their IDs.
        get_node: Retrieves a node by its ID.
        get_links: Retrieves links for a given node ID.
        add_link: Adds a link to the graph store.
        delete_link: Deletes a link from the graph store.
        get_all_nodes: Retrieves all nodes from the graph store.
        get_all_links: Retrieves all links from the graph store.
        get_metadata: Retrieves metadata for a given node ID.
        set_metadata: Sets metadata for a given node ID.
    """

    def __init__(
        self,
        opensearch_url: str,
        index_name: str,
        embedding_function: Embeddings,
        reset_index: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the OpenSearchGraphVectorStore.

        Args:
            embedding (Embeddings): The embedding function to use.
            index_name (str, optional): The name of the OpenSearch index. Defaults to "myindex".
            opensearch_url (str, optional): The URL of the OpenSearch instance. Defaults to "http://localhost:9200".
        Returns:
            None

        """
        self.index_name = index_name

        self.vector_store = OpenSearchVectorSearch(
            embedding_function=embedding_function,
            opensearch_url=opensearch_url,
            index_name=index_name,
            **kwargs,
        )

        if reset_index and self.vector_store.index_exists(index_name=index_name):
            self.vector_store.client.indices.delete(index=index_name)

    @property
    @override
    def embeddings(self) -> Embeddings | None:
        return self.vector_store.embeddings

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
            if doc.metadata is not None and METADATA_LINKS_KEY in doc.metadata:
                metadata = doc.metadata.copy()
                metadata[METADATA_LINKS_KEY] = _serialize_links(
                    metadata[METADATA_LINKS_KEY]
                )
                doc.metadata = metadata
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

    # region Basic Search Methods

    @override
    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Perform a similarity search on the OpenSearch vector store.

        Args:
            query (str): The query string to search for similar documents.
            k (int, optional): The number of top similar documents to return. Defaults to 4.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            List[Document]: A list of documents that are most similar to the query.

        """
        search_type = kwargs.get("search_type", "knn")
        vector_field = kwargs.get("vector_field", "vector_field")
        text_field = kwargs.get("text_field", "text")
        metadata_field = kwargs.get("metadata_field", "metadata")
        query_embedding = self.embeddings.embed_query(query)
        query = {
            "size": k,
            "query": {search_type: {vector_field: {"vector": query_embedding, "k": k}}},
        }

        # Execute the synchronous search
        response = self.vector_store.client.search(
            index=self.index_name, body=query, size=k
        )
        hits = response["hits"]["hits"]

        # Generate and return Document objects from the search results
        return [
            self._restore_links(
                Document(
                    id=hit["_id"],
                    page_content=hit["_source"][text_field],
                    metadata=hit["_source"][metadata_field],
                )
            )
            for hit in hits
        ]

    @override
    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Retrieve documents from this graph store.

        Args:
            query: The query string.
            k: The number of Documents to return. Defaults to 4.
            filter: Optional metadata to filter the results.
            **kwargs: Additional keyword arguments.

        Returns:
            Collection of retrieved documents.

        """
        search_type = kwargs.get("search_type", "knn")
        vector_field = kwargs.get("vector_field", "vector_field")
        index_name = kwargs.get("index_name", self.index_name)
        text_field = kwargs.get("text_field", "text")
        metadata_field = kwargs.get("metadata_field", "metadata")

        query_embedding = self.embeddings.embed_query(query)

        # Define the search query
        query = {
            "size": k,
            "query": {search_type: {vector_field: {"vector": query_embedding, "k": k}}},
        }

        # Perform the asynchronous search
        response = await self.vector_store.async_client.search(
            index=index_name, body=query
        )

        hits = response["hits"]["hits"]

        for hit in hits:
            yield self._restore_links(
                Document(
                    id=hit["_id"],
                    page_content=hit["_source"][text_field],
                    metadata=hit["_source"][metadata_field],
                )
            )

    def from_texts(self, texts: List[str]) -> None:
        """Create nodes from texts and add them to the vector store."""
        # Convert all texts to Document objects at once
        documents = [Document(page_content=text) for text in texts]

        # Add all documents in a single call to add_documents
        self.add_documents(documents)

    def _restore_links(self, doc: Document) -> Document:
        """Restores the links in the document by deserializing them from metadata.

        Args:
            doc: A single Document

        Returns:
            The same Document with restored links.
        """
        links = _deserialize_links(doc.metadata.get(METADATA_LINKS_KEY))
        doc.metadata[METADATA_LINKS_KEY] = links
        return doc

    def get_by_document_id(self, document_id: str, **kwargs: Any) -> Document | None:
        """Retrieve a single document from the store, given its document ID.

            document_id (str): The document ID.

            Document | None: The document if it exists, otherwise None.

        Args:
            document_id: The document ID

        Returns:
            The the document if it exists. Otherwise None.

        """
        try:
            text_field = kwargs.get("text_field", "text")
            metadata_field = kwargs.get("metadata_field", "metadata")
            response = self.vector_store.client.get(
                index=self.index_name, id=document_id
            )
            hit = response["_source"]

            return self._restore_links(
                Document(
                    id=document_id,
                    page_content=hit[text_field],
                    metadata=(
                        hit
                        if metadata_field == "*" or metadata_field not in hit
                        else hit[metadata_field]
                    ),
                )
            )

        except Exception as e:
            logger.error("Error retrieving document: %s", e)

    async def aget_by_document_id(self, document_id: str, **kwargs: Any) -> Document | None:
        """Retrieve a single document from the store, given its document ID.

        Args:
            document_id: The document ID

        Returns:
            The the document if it exists. Otherwise None.

        """
        try:
            text_field = kwargs.get("text_field", "text")
            metadata_field = kwargs.get("metadata_field", "metadata")
            response = await self.vector_store.async_client.get(
                index=self.index_name, id=document_id
            )
            hit = response["_source"]

            return self._restore_links(
                Document(
                    id=document_id,
                    page_content=hit[text_field],
                    metadata=(
                        hit
                        if metadata_field == "*" or metadata_field not in hit
                        else hit[metadata_field]
                    ),
                )
            )

        except Exception as e:
            logger.error("Error retrieving document: %s", e)

    def search_by_metadata(
        self, metadata: Dict[str, Any] | None = None, k: int = 10, **kwargs: Any
    ) -> Iterable[Document]:
        """Search for documents in the OpenSearch vector store based on metadata.

        Args:
            metadata (Dict[str, Any] | None): A dictionary of metadata key-value pairs to search for.
            k (int): The number of top results to return. Defaults to 10.
            **kwargs (Any): Additional keyword arguments.
                - text_field (str): The field name in the document that contains the text content. Defaults to "text".
                - metadata_field (str): The field name in the document that contains the metadata. Defaults to "metadata".

        Returns:
            Iterable[Document]: An iterable of Document objects that match the search criteria.

        """
        text_field = kwargs.get("text_field", "text")
        metadata_field = kwargs.get("metadata_field", "metadata")

        # Build a list of match queries for each metadata field
        query = {
            "bool": {
                "must": [
                    {"match": {f"metadata.{key}": value}}
                    for key, value in metadata.items()
                ]
            }
        }

        # Execute the synchronous search
        response = self.vector_store.client.search(
            index=self.index_name, body={"query": query}, size=k
        )
        hits = response["hits"]["hits"]

        # Generate and return Document objects from the search results
        return [
            self._restore_links(
                Document(
                    id=hit["_id"],
                    page_content=hit["_source"][text_field],
                    metadata=hit["_source"][metadata_field],
                )
            )
            for hit in hits
        ]

    async def asearch_by_metadata(
        self, metadata: dict[str, Any] | None = None, k: int = 10, **kwargs: Any
    ) -> AsyncIterable[Document]:
        text_field = kwargs.get("text_field", "text")
        metadata_field = kwargs.get("metadata_field", "metadata")

        # Build a list of match queries for each metadata field
        if metadata:
            query = {
                "bool": {
                    "must": [
                        {"match": {f"metadata.{key}": value}}
                        for key, value in metadata.items()
                    ]
                }
            }
        else:
            query = {"match_all": {}}

        # Execute the search, Notice that we are using the async client
        response = await self.vector_store.async_client.search(
            index=self.index_name, body={"query": query}, size=k
        )
        hits = response["hits"]["hits"]

        for hit in hits:
            yield self._restore_links(
                Document(
                    id=hit["_id"],
                    page_content=hit["_source"][text_field],
                    metadata=hit["_source"][metadata_field],
                )
            )

    def similarity_search_by_vector_and_metadata(
        self,
        query: str,
        metadata: Dict[str, Any] | None = None,
        k: int = 10,
        **kwargs: Any,
    ) -> Iterable[Document]:
        search_type = kwargs.get("search_type", "knn")
        vector_field = kwargs.get("vector_field", "vector_field")
        text_field = kwargs.get("text_field", "text")
        metadata_field = kwargs.get("metadata_field", "metadata")
        query_vector = self.embeddings.embed_query(query)

        query = {
            "size": k,
            "query": {
                "bool": {
                    "filter": [
                        # Metadata match conditions
                        *[
                            {"match": {f"metadata.{key}": value}}
                            for key, value in metadata.items()
                        ],
                    ],
                    "must": [
                        # Vector similarity using script_score
                        {
                            search_type: {
                                vector_field: {"vector": query_vector, "k": k}
                            },
                        }
                    ],
                }
            },
        }

        # query = {
        #     "size": k,
        #     "query": {
        #         "bool": {
        #             "filter": [
        #                 {
        #                     "term": {
        #                         f"{metadata_field}": metadata_value  # Metadata filtering
        #                     }
        #                 }
        #             ],
        #             "must": [
        #                 {
        #                     "script_score": {
        #                         "query": {"match_all": {}},
        #                         "script": {
        #                             "source": "cosineSimilarity(params.query_vector, doc[params.vector_field]) + 1.0",
        #                             "params": {
        #                                 "query_vector": query_vector,
        #                                 "vector_field": vector_field
        #                             }
        #                         }
        #                     }
        #                 }
        #             ]
        #         }
        #     }
        # }

        # Execute the synchronous search
        response = self.vector_store.client.search(
            index=self.index_name, body=query, size=k
        )
        hits = response["hits"]["hits"]

        # Generate and return Document objects from the search results
        return [
            self._restore_links(
                Document(
                    id=hit["_id"],
                    page_content=hit["_source"][text_field],
                    metadata=hit["_source"][metadata_field],
                )
            )
            for hit in hits
        ]

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

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        pass

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
        pass

    @override
    def add_nodes(
        self,
        nodes: Iterable[Node],
        **kwargs: Any,
    ) -> Iterable[str]:
        """Add nodes to the graph store.

        Args:
            nodes: the nodes to add.
            **kwargs: Additional keyword arguments.
        """
        pass

    def delete(self, ids: List[str], **kwargs: Any) -> None:
        pass

    def get_node(self, node_id: str) -> Optional[Node]:
        pass

    def get_links(self, node_id: str) -> List[Link]:
        pass

    def add_link(self, link: Link) -> None:
        pass

    def delete_link(self, link: Link) -> None:
        pass

    def get_all_nodes(self) -> List[Node]:
        pass

    def get_all_links(self) -> List[Link]:
        pass

    def get_metadata(self, node_id: str) -> dict:
        pass

    def set_metadata(self, node_id: str, metadata: dict) -> None:
        pass

    @override
    def add_nodes(
        self,
        nodes: Iterable[Node],
        **kwargs: Any,
    ) -> Iterable[str]:
        """Add nodes to the graph store.

        Args:
            nodes: the nodes to add.
            **kwargs: Additional keyword arguments.
        """
        pass

    @override
    async def aadd_nodes(
        self,
        nodes: Iterable[Node],
        **kwargs: Any,
    ) -> AsyncIterable[str]:
        """Add nodes to the graph store.

        Args:
            nodes: the nodes to add.
            **kwargs: Additional keyword arguments.
        """
        pass

    # endregion

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