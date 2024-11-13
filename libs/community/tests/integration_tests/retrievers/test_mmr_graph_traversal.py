"""Test of Apache Cassandra graph index class `Cassandra`"""

import os
import json
from contextlib import contextmanager
from typing import Any, Generator, Iterable, List

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from langchain_community.retrievers import GraphTraversalRetriever
from langchain_community.vectorstores import Cassandra
from tests.integration_tests.cache.fake_embeddings import (
    AngularTwoDimensionalEmbeddings,
)

TEST_KEYSPACE = "graph_test_keyspace"


def _doc_ids(docs: Iterable[Document]) -> List[str]:
    return [doc.id for doc in docs if doc.id is not None]


class CassandraSession:
    table_name: str
    session: Any

    def __init__(self, table_name: str, session: Any):
        self.table_name = table_name
        self.session = session


class ParserEmbeddings(Embeddings):
    """Parse input texts: if they are json for a List[float], fine.
    Otherwise, return all zeros and call it a day.
    """

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(txt) for txt in texts]

    def embed_query(self, text: str) -> list[float]:
        try:
            vals = json.loads(text)
        except json.JSONDecodeError:
            return [0.0] * self.dimension
        else:
            assert len(vals) == self.dimension
            return vals


@pytest.fixture
def embedding_d2() -> Embeddings:
    return ParserEmbeddings(dimension=2)


@pytest.fixture
def graph_vector_store_docs() -> list[Document]:
    """
    This is a set of Documents to pre-populate a graph vector store,
    with entries placed in a certain way.

    Space of the entries (under Euclidean similarity):

                      A0    (*)
        ....        AL   AR       <....
        :              |              :
        :              |  ^           :
        v              |  .           v
                       |   :
       TR              |   :          BL
    T0   --------------x--------------   B0
       TL              |   :          BR
                       |   :
                       |  .
                       | .
                       |
                    FL   FR
                      F0

    the query point is meant to be at (*).
    the A are bidirectionally with B
    the A are outgoing to T
    the A are incoming from F
    The links are like: L with L, 0 with 0 and R with R.
    """

    docs_a = [
        Document(id="AL", page_content="[-1, 9]", metadata={"label": "AL"}),
        Document(id="A0", page_content="[0, 10]", metadata={"label": "A0"}),
        Document(id="AR", page_content="[1, 9]", metadata={"label": "AR"}),
    ]
    docs_b = [
        Document(id="BL", page_content="[9, 1]", metadata={"label": "BL"}),
        Document(id="B0", page_content="[10, 0]", metadata={"label": "B0"}),
        Document(id="BR", page_content="[9, -1]", metadata={"label": "BR"}),
    ]
    docs_f = [
        Document(id="FL", page_content="[1, -9]", metadata={"label": "FL"}),
        Document(id="F0", page_content="[0, -10]", metadata={"label": "F0"}),
        Document(id="FR", page_content="[-1, -9]", metadata={"label": "FR"}),
    ]
    docs_t = [
        Document(id="TL", page_content="[-9, -1]", metadata={"label": "TL"}),
        Document(id="T0", page_content="[-10, 0]", metadata={"label": "T0"}),
        Document(id="TR", page_content="[-9, 1]", metadata={"label": "TR"}),
    ]
    for doc_a, suffix in zip(docs_a, ["l", "0", "r"]):
        doc_a.metadata["tag"] = f"ab_{suffix}"
        doc_a.metadata["out"] = f"at_{suffix}"
        doc_a.metadata["in"] = f"af_{suffix}"
        # add_links(doc_a, Link.bidir(kind="ab_example", tag=f"tag_{suffix}"))
        # add_links(doc_a, Link.outgoing(kind="at_example", tag=f"tag_{suffix}"))
        # add_links(doc_a, Link.incoming(kind="af_example", tag=f"tag_{suffix}"))
    for doc_b, suffix in zip(docs_b, ["l", "0", "r"]):
        doc_b.metadata["tag"] = f"ab_{suffix}"
        # add_links(doc_b, Link.bidir(kind="ab_example", tag=f"tag_{suffix}"))
    for doc_t, suffix in zip(docs_t, ["l", "0", "r"]):
        doc_t.metadata["in"] = f"at_{suffix}"
        # add_links(doc_t, Link.incoming(kind="at_example", tag=f"tag_{suffix}"))
    for doc_f, suffix in zip(docs_f, ["l", "0", "r"]):
        doc_f.metadata["out"] = f"af_{suffix}"
        # add_links(doc_f, Link.outgoing(kind="af_example", tag=f"tag_{suffix}"))
    return docs_a + docs_b + docs_f + docs_t

@contextmanager
def get_cassandra_session(
    table_name: str, drop: bool = True
) -> Generator[CassandraSession, None, None]:
    """Initialize the Cassandra cluster and session"""
    from cassandra.cluster import Cluster

    if "CASSANDRA_CONTACT_POINTS" in os.environ:
        contact_points = [
            cp.strip()
            for cp in os.environ["CASSANDRA_CONTACT_POINTS"].split(",")
            if cp.strip()
        ]
    else:
        contact_points = None

    cluster = Cluster(contact_points)
    session = cluster.connect()

    try:
        session.execute(
            (
                f"CREATE KEYSPACE IF NOT EXISTS {TEST_KEYSPACE}"
                " WITH replication = "
                "{'class': 'SimpleStrategy', 'replication_factor': 1}"
            )
        )
        if drop:
            session.execute(f"DROP TABLE IF EXISTS {TEST_KEYSPACE}.{table_name}")

        # Yield the session for usage
        yield CassandraSession(table_name=table_name, session=session)
    finally:
        # Ensure proper shutdown/cleanup of resources
        session.shutdown()
        cluster.shutdown()


@pytest.fixture(scope="function")
def vector_store_angular(
    table_name: str = "graph_test_table",
) -> Generator[VectorStore, None, None]:
    with get_cassandra_session(table_name=table_name) as session:
        yield Cassandra(
            embedding=AngularTwoDimensionalEmbeddings(),
            session=session.session,
            keyspace=TEST_KEYSPACE,
            table_name=session.table_name,
        )

@pytest.fixture(scope="function")
def vector_store_d2(
    embedding_d2: Embeddings,
    table_name: str = "graph_test_table",
) -> Generator[VectorStore, None, None]:
    with get_cassandra_session(table_name=table_name) as session:
        yield Cassandra(
            embedding=embedding_d2,
            session=session.session,
            keyspace=TEST_KEYSPACE,
            table_name=session.table_name,
        )

@pytest.fixture(scope="function")
def populated_graph_vector_store_d2(
    vector_store_d2: VectorStore,
    graph_vector_store_docs: list[Document],
) -> Generator[VectorStore, None, None]:
    vector_store_d2.add_documents(graph_vector_store_docs)
    yield vector_store_d2

def test_mmr_traversal(vector_store_angular: VectorStore) -> None:
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

    v0.metadata["outgoing"] = "link"
    v2.metadata["incoming"] = "link"
    v3.metadata["incoming"] = "link"

    vector_store_angular.add_documents([v0, v1, v2, v3])

    retriever = GraphTraversalRetriever(
        vector_store=vector_store_angular,
        edges=[("outgoing", "incoming")],
        fetch_k=2,
        k=2,
        depth=2,
    )

    docs = retriever.invoke("0.0", k=2, fetch_k=2)
    assert _doc_ids(docs) == ["v0", "v2"]

    # With max depth 0, no edges are traversed, so this doesn't reach v2 or v3.
    # So it ends up picking "v1" even though it's similar to "v0".
    docs = retriever.invoke("0.0", k=2, fetch_k=2, depth=0)
    assert _doc_ids(docs) == ["v0", "v1"]

    # With max depth 0 but higher `fetch_k`, we encounter v2
    docs = retriever.invoke("0.0", k=2, fetch_k=3, depth=0)
    assert _doc_ids(docs) == ["v0", "v2"]

    # v0 score is .46, v2 score is 0.16 so it won't be chosen.
    docs = retriever.invoke("0.0", k=2, score_threshold=0.2)
    assert _doc_ids(docs) == ["v0"]

    # with k=4 we should get all of the documents.
    docs = retriever.invoke("0.0", k=4)
    assert _doc_ids(docs) == ["v0", "v2", "v1", "v3"]


def assert_document_format(doc: Document) -> None:
    assert doc.id is not None
    assert doc.page_content is not None
    assert doc.metadata is not None
    assert "__embedding" not in doc.metadata

class TestMmrGraphTraversal:
    def test_invoke_sync(
        self,
        populated_graph_vector_store_d2: VectorStore,
    ) -> None:
        """MMR Graph traversal search on a graph vector store."""
        retriever = GraphTraversalRetriever(
            vector_store=populated_graph_vector_store_d2,
            edges=[("out", "in"), "tag"],
        )

        docs = retriever.invoke(input="[2, 10]")
        # TODO: can this rightfully be a list (or must it be a set)?
        mt_labels = {doc.metadata["label"] for doc in docs}
        assert mt_labels == {"AR", "BR"}
        assert docs[0].metadata
        assert_document_format(docs[0])

    async def test_invoke_async(
        self,
        populated_graph_vector_store_d2: VectorStore,
    ) -> None:
        """MMR Graph traversal search on a graph vector store."""
        retriever = GraphTraversalRetriever(
            vector_store=populated_graph_vector_store_d2,
            edges=[("out", "in"), "tag"],
        )
        mt_labels = set()
        docs = await retriever.ainvoke(input="[2, 10]")
        mt_labels = {doc.metadata["label"] for doc in docs}
        assert mt_labels == {"AR", "BR"}
        assert docs[0].metadata
        assert_document_format(docs[0])
