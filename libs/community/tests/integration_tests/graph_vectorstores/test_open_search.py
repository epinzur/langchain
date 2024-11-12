"""Test of Open Search graph vector store class `OpenSearchGraphVectorStore`"""

from typing import Any, Generator, Iterable, List, Optional, cast

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from langchain_community.graph_vectorstores import OpenSearchGraphVectorStore
from langchain_community.graph_vectorstores.links import (
    METADATA_LINKS_KEY,
    Link,
    add_links,
    get_links,
)
from langchain_community.graph_vectorstores.utils.document_embedding import (
    METADATA_EMBEDDING_KEY,
)
from langchain_community.vectorstores.opensearch_vector_search import (
    OpenSearchVectorSearch,
)
from tests.integration_tests.cache.fake_embeddings import (
    AngularTwoDimensionalEmbeddings,
    FakeEmbeddings,
)

TEST_INDEX = "graph_test_index"

OPEN_SEARCH_URL = "http://localhost:9200"


def _result_ids(docs: Iterable[Document]) -> List[Optional[str]]:
    return [doc.id for doc in docs]


def get_graph_vector_store(
    embedding_function: Embeddings,
) -> OpenSearchGraphVectorStore:
    return OpenSearchGraphVectorStore(
        opensearch_url=OPEN_SEARCH_URL,
        index_name=TEST_INDEX,
        embedding_function=embedding_function,
    )


def cleanup(store: OpenSearchGraphVectorStore) -> None:
    os_vectorstore = cast(OpenSearchVectorSearch, store.vector_store)
    os_vectorstore.delete_index(index_name=TEST_INDEX)


@pytest.fixture(scope="function")
def graph_vector_store_angular() -> Generator[OpenSearchGraphVectorStore, None, None]:
    store = get_graph_vector_store(embedding_function=AngularTwoDimensionalEmbeddings())
    yield store
    cleanup(store=store)


@pytest.fixture(scope="function")
def graph_vector_store_earth(
    earth_embeddings: Embeddings,
) -> Generator[OpenSearchGraphVectorStore, None, None]:
    store = get_graph_vector_store(embedding_function=earth_embeddings)
    yield store
    cleanup(store=store)


@pytest.fixture(scope="function")
def graph_vector_store_fake() -> Generator[OpenSearchGraphVectorStore, None, None]:
    store = get_graph_vector_store(embedding_function=FakeEmbeddings())
    yield store
    cleanup(store=store)


@pytest.fixture(scope="function")
def graph_vector_store_d2(
    embedding_d2: Embeddings,
) -> Generator[OpenSearchGraphVectorStore, None, None]:
    store = get_graph_vector_store(embedding_function=embedding_d2)
    yield store
    cleanup(store=store)


@pytest.fixture(scope="function")
def populated_graph_vector_store_d2(
    graph_vector_store_d2: OpenSearchGraphVectorStore,
    graph_vector_store_docs: list[Document],
) -> Generator[OpenSearchGraphVectorStore, None, None]:
    graph_vector_store_d2.add_documents(graph_vector_store_docs)
    yield graph_vector_store_d2


def test_mmr_traversal(graph_vector_store_angular: OpenSearchGraphVectorStore) -> None:
    """ Test end to end construction and MMR search.
    The embedding function used here ensures `texts` become
    the following vectors on a circle (numbered v0 through v3):

           ______ v2
          /      \
         /        |  v1
    v3  |     .    | query
         |        /  v0
          |______/                 (N.B. very crude drawing)

    With fetch_k==2 and k==2, when query is at (1, ),
    one expects that v2 and v0 are returned (in some order)
    because v1 is "too close" to v0 (and v0 is closer than v1)).

    Both v2 and v3 are reachable via edges from v0, so once it is
    selected, those are both considered.
    """
    v0 = Document(id="v0", page_content="-0.124")
    v1 = Document(id="v1", page_content="+0.127")
    v2 = Document(id="v2", page_content="+0.25")
    v3 = Document(id="v3", page_content="+1.0")
    add_links(v0, [Link.outgoing(kind="explicit", tag="link")])
    add_links(v2, [Link.incoming(kind="explicit", tag="link")])
    add_links(v3, [Link.incoming(kind="explicit", tag="link")])

    g_store = graph_vector_store_angular
    g_store.add_documents([v0, v1, v2, v3])

    results = g_store.mmr_traversal_search("0.0", k=2, fetch_k=2)
    assert _result_ids(results) == ["v0", "v2"]

    # With max depth 0, no edges are traversed, so this doesn't reach v2 or v3.
    # So it ends up picking "v1" even though it's similar to "v0".
    results = g_store.mmr_traversal_search("0.0", k=2, fetch_k=2, depth=0)
    assert _result_ids(results) == ["v0", "v1"]

    # With max depth 0 but higher `fetch_k`, we encounter v2
    results = g_store.mmr_traversal_search("0.0", k=2, fetch_k=3, depth=0)
    assert _result_ids(results) == ["v0", "v2"]

    # v0 score is .46, v2 score is 0.16 so it won't be chosen.
    results = g_store.mmr_traversal_search("0.0", k=2, score_threshold=0.2)
    assert _result_ids(results) == ["v0"]

    # with k=4 we should get all of the documents.
    results = g_store.mmr_traversal_search("0.0", k=4)
    assert _result_ids(results) == ["v0", "v2", "v1", "v3"]


def test_write_retrieve_keywords(
    graph_vector_store_earth: OpenSearchGraphVectorStore,
) -> None:
    greetings = Document(
        id="greetings",
        page_content="Typical Greetings",
    )
    add_links(
        greetings,
        [
            Link.incoming(kind="parent", tag="parent"),
        ],
    )

    node1 = Document(
        id="doc1",
        page_content="Hello World",
    )
    add_links(
        node1,
        [
            Link.outgoing(kind="parent", tag="parent"),
            Link.bidir(kind="kw", tag="greeting"),
            Link.bidir(kind="kw", tag="world"),
        ],
    )

    node2 = Document(
        id="doc2",
        page_content="Hello Earth",
    )
    add_links(
        node2,
        [
            Link.outgoing(kind="parent", tag="parent"),
            Link.bidir(kind="kw", tag="greeting"),
            Link.bidir(kind="kw", tag="earth"),
        ],
    )

    g_store = graph_vector_store_earth
    g_store.add_documents(documents=[greetings, node1, node2])

    # Doc2 is more similar, but World and Earth are similar enough that doc1 also
    # shows up.
    results: Iterable[Document] = g_store.similarity_search("Earth", k=2)
    assert _result_ids(results) == ["doc2", "doc1"]

    results = g_store.similarity_search("Earth", k=1)
    assert _result_ids(results) == ["doc2"]

    results = g_store.traversal_search("Earth", k=2, depth=0)
    assert _result_ids(results) == ["doc2", "doc1"]

    results = g_store.traversal_search("Earth", k=2, depth=1)
    assert _result_ids(results) == ["doc2", "doc1", "greetings"]

    # K=1 only pulls in doc2 (Hello Earth)
    results = g_store.traversal_search("Earth", k=1, depth=0)
    assert _result_ids(results) == ["doc2"]

    # K=1 only pulls in doc2 (Hello Earth). Depth=1 traverses to parent and via
    # keyword edge.
    results = g_store.traversal_search("Earth", k=1, depth=1)
    assert set(_result_ids(results)) == {"doc2", "doc1", "greetings"}


def test_metadata(graph_vector_store_fake: OpenSearchGraphVectorStore) -> None:
    links = [
        Link.incoming(kind="hyperlink", tag="http://a"),
        Link.bidir(kind="other", tag="foo"),
    ]
    doc_a = Document(
        id="a",
        page_content="A",
        metadata={"other": "some other field"},
    )
    add_links(doc_a, links)

    g_store = graph_vector_store_fake
    g_store.add_documents([doc_a])
    results = g_store.similarity_search("A")
    assert len(results) == 1
    assert results[0].id == "a"
    metadata = results[0].metadata
    assert metadata["other"] == "some other field"
    # verify against a `set` to avoid flaky ordering issues
    assert set(get_links(doc=results[0])) == {
        Link.incoming(kind="hyperlink", tag="http://a"),
        Link.bidir(kind="other", tag="foo"),
    }


def assert_document_format(doc: Document) -> None:
    assert doc.id is not None
    assert doc.page_content is not None
    assert doc.metadata is not None
    assert METADATA_LINKS_KEY in doc.metadata
    assert METADATA_EMBEDDING_KEY not in doc.metadata


class TestOpenSearchGraphVectorStore:
    def test_gvs_similarity_search_sync(
        self,
        populated_graph_vector_store_d2: OpenSearchGraphVectorStore,
    ) -> None:
        """Simple (non-graph) similarity search on a graph vector g_store."""
        g_store = populated_graph_vector_store_d2
        ss_response = g_store.similarity_search(query="[2, 10]", k=2)
        ss_labels = [doc.metadata["label"] for doc in ss_response]
        assert ss_labels == ["AR", "A0"]
        assert_document_format(ss_response[0])

        ss_by_v_response = g_store.similarity_search_by_vector(embedding=[2, 10], k=2)
        ss_by_v_labels = [doc.metadata["label"] for doc in ss_by_v_response]
        assert ss_by_v_labels == ["AR", "A0"]
        assert_document_format(ss_by_v_response[0])

    async def test_gvs_similarity_search_async(
        self,
        populated_graph_vector_store_d2: OpenSearchGraphVectorStore,
    ) -> None:
        """Simple (non-graph) similarity search on a graph vector store."""
        g_store = populated_graph_vector_store_d2
        ss_response = await g_store.asimilarity_search(query="[2, 10]", k=2)
        ss_labels = [doc.metadata["label"] for doc in ss_response]
        assert ss_labels == ["AR", "A0"]
        assert_document_format(ss_response[0])

        ss_by_v_response = await g_store.asimilarity_search_by_vector(
            embedding=[2, 10], k=2
        )
        ss_by_v_labels = [doc.metadata["label"] for doc in ss_by_v_response]
        assert ss_by_v_labels == ["AR", "A0"]
        assert_document_format(ss_by_v_response[0])

    def test_gvs_traversal_search_sync(
        self,
        populated_graph_vector_store_d2: OpenSearchGraphVectorStore,
    ) -> None:
        """Graph traversal search on a graph vector store."""
        g_store = populated_graph_vector_store_d2
        docs = list(g_store.traversal_search(query="[2, 10]", k=2, depth=2))
        # this is a set, as some of the internals of trav.search are set-driven
        # so ordering is not deterministic:
        ts_labels = {doc.metadata["label"] for doc in docs}
        assert ts_labels == {"AR", "A0", "BR", "B0", "TR", "T0"}
        assert_document_format(docs[0])

        # verify the same works as a retriever
        retriever = g_store.as_retriever(
            search_type="traversal", search_kwargs={"k": 2, "depth": 2}
        )

        ts_labels = {
            doc.metadata["label"]
            for doc in retriever.get_relevant_documents(query="[2, 10]")
        }
        assert ts_labels == {"AR", "A0", "BR", "B0", "TR", "T0"}

    async def test_gvs_traversal_search_async(
        self,
        populated_graph_vector_store_d2: OpenSearchGraphVectorStore,
    ) -> None:
        """Graph traversal search on a graph vector store."""
        g_store = populated_graph_vector_store_d2
        ts_labels = set()
        async for doc in g_store.atraversal_search(query="[2, 10]", k=2, depth=2):
            ts_labels.add(doc.metadata["label"])
            assert_document_format(doc)
        # this is a set, as some of the internals of trav.search are set-driven
        # so ordering is not deterministic:
        assert ts_labels == {"AR", "A0", "BR", "B0", "TR", "T0"}

        # verify the same works as a retriever
        retriever = g_store.as_retriever(
            search_type="traversal", search_kwargs={"k": 2, "depth": 2}
        )

        ts_labels = {
            doc.metadata["label"]
            for doc in await retriever.aget_relevant_documents(query="[2, 10]")
        }
        assert ts_labels == {"AR", "A0", "BR", "B0", "TR", "T0"}

    def test_gvs_mmr_traversal_search_sync(
        self,
        populated_graph_vector_store_d2: OpenSearchGraphVectorStore,
    ) -> None:
        """MMR Graph traversal search on a graph vector store."""
        g_store = populated_graph_vector_store_d2
        docs = list(
            g_store.mmr_traversal_search(
                query="[2, 10]",
                k=2,
                depth=2,
                fetch_k=1,
                adjacent_k=2,
                lambda_mult=0.1,
            )
        )
        # TODO: can this rightfully be a list (or must it be a set)?
        mt_labels = {doc.metadata["label"] for doc in docs}
        assert mt_labels == {"AR", "BR"}
        assert docs[0].metadata
        assert_document_format(docs[0])

    async def test_gvs_mmr_traversal_search_async(
        self,
        populated_graph_vector_store_d2: OpenSearchGraphVectorStore,
    ) -> None:
        """MMR Graph traversal search on a graph vector store."""
        g_store = populated_graph_vector_store_d2
        mt_labels = set()
        async for doc in g_store.ammr_traversal_search(
            query="[2, 10]",
            k=2,
            depth=2,
            fetch_k=1,
            adjacent_k=2,
            lambda_mult=0.1,
        ):
            mt_labels.add(doc.metadata["label"])
        # TODO: can this rightfully be a list (or must it be a set)?
        assert mt_labels == {"AR", "BR"}
        assert_document_format(doc)

    def test_gvs_metadata_search_sync(
        self,
        populated_graph_vector_store_d2: OpenSearchGraphVectorStore,
    ) -> None:
        """Metadata search on a graph vector store."""
        g_store = populated_graph_vector_store_d2
        mt_response = g_store.metadata_search(
            filter={"label": "T0"},
            n=2,
        )
        doc: Document = next(iter(mt_response))
        assert doc.page_content == "[-10, 0]"
        links = doc.metadata["links"]
        assert len(links) == 1
        link: Link = links.pop()
        assert isinstance(link, Link)
        assert link.direction == "in"
        assert link.kind == "at_example"
        assert link.tag == "tag_0"
        assert_document_format(doc)

    async def test_gvs_metadata_search_async(
        self,
        populated_graph_vector_store_d2: OpenSearchGraphVectorStore,
    ) -> None:
        """Metadata search on a graph vector store."""
        g_store = populated_graph_vector_store_d2
        mt_response = await g_store.ametadata_search(
            filter={"label": "T0"},
            n=2,
        )
        doc: Document = next(iter(mt_response))
        assert doc.page_content == "[-10, 0]"
        links: set[Link] = doc.metadata["links"]
        assert len(links) == 1
        link: Link = links.pop()
        assert isinstance(link, Link)
        assert link.direction == "in"
        assert link.kind == "at_example"
        assert link.tag == "tag_0"
        assert_document_format(doc)

    def test_gvs_get_by_document_id_sync(
        self,
        populated_graph_vector_store_d2: OpenSearchGraphVectorStore,
    ) -> None:
        """Get by document_id on a graph vector store."""
        g_store = populated_graph_vector_store_d2
        doc = g_store.get_by_document_id(document_id="FL")
        assert doc is not None
        assert doc.page_content == "[1, -9]"
        links = doc.metadata["links"]
        assert len(links) == 1
        link: Link = links.pop()
        assert isinstance(link, Link)
        assert link.direction == "out"
        assert link.kind == "af_example"
        assert link.tag == "tag_l"
        assert_document_format(doc)

        invalid_doc = g_store.get_by_document_id(document_id="invalid")
        assert invalid_doc is None

    async def test_gvs_get_by_document_id_async(
        self,
        populated_graph_vector_store_d2: OpenSearchGraphVectorStore,
    ) -> None:
        """Get by document_id on a graph vector store."""
        g_store = populated_graph_vector_store_d2
        doc = await g_store.aget_by_document_id(document_id="FL")
        assert doc is not None
        assert doc.page_content == "[1, -9]"
        links = doc.metadata["links"]
        assert len(links) == 1
        link: Link = links.pop()
        assert isinstance(link, Link)
        assert link.direction == "out"
        assert link.kind == "af_example"
        assert link.tag == "tag_l"
        assert_document_format(doc)

        invalid_doc = await g_store.aget_by_document_id(document_id="invalid")
        assert invalid_doc is None

    def test_gvs_from_texts(
        self,
        graph_vector_store_d2: OpenSearchGraphVectorStore,
    ) -> None:
        g_store = graph_vector_store_d2
        g_store.add_texts(
            texts=["[1, 2]"],
            metadatas=[{"md": 1}],
            ids=["x_id"],
        )

        hits = g_store.similarity_search("[2, 1]", k=2)
        assert len(hits) == 1
        assert hits[0].page_content == "[1, 2]"
        assert hits[0].id == "x_id"
        # there may be more re:graph structure.
        assert hits[0].metadata["md"] == 1
        assert_document_format(hits[0])

    def test_gvs_from_documents_containing_ids(
        self,
        graph_vector_store_d2: OpenSearchGraphVectorStore,
    ) -> None:
        the_document = Document(
            page_content="[1, 2]",
            metadata={"md": 1},
            id="x_id",
        )
        g_store = graph_vector_store_d2
        g_store.add_documents([the_document])
        hits = g_store.similarity_search("[2, 1]", k=2)
        assert len(hits) == 1
        assert hits[0].page_content == "[1, 2]"
        assert hits[0].id == "x_id"
        # there may be more re:graph structure.
        assert hits[0].metadata["md"] == 1
        assert_document_format(hits[0])
