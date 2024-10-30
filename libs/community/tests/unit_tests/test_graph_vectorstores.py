import pytest
from langchain_core.documents import Document

from langchain_community.graph_vectorstores.base import (
    Node,
    _nodes_to_documents,
    _texts_to_documents,
)
from langchain_community.graph_vectorstores.links import Link


def test_texts_to_documents() -> None:
    assert list(
        _texts_to_documents(["a", "b"], [{"a": "b"}, {"c": "d"}], ["a", "b"])
    ) == [
        Document(id="a", metadata={"a": "b"}, page_content="a"),
        Document(id="b", metadata={"c": "d"}, page_content="b"),
    ]
    assert list(_texts_to_documents(["a", "b"], None, ["a", "b"])) == [
        Document(id="a", metadata={}, page_content="a"),
        Document(id="b", metadata={}, page_content="b"),
    ]
    assert list(_texts_to_documents(["a", "b"], [{"a": "b"}, {"c": "d"}], None)) == [
        Document(metadata={"a": "b"}, page_content="a"),
        Document(metadata={"c": "d"}, page_content="b"),
    ]
    assert list(
        _texts_to_documents(
            ["a"],
            [{"links": [Link.incoming(kind="hyperlink", tag="http://b")]}],
            None,
        )
    ) == [
        Document(
            metadata={"links": [Link.incoming(kind="hyperlink", tag="http://b")]},
            page_content="a",
        )
    ]
    with pytest.raises(ValueError):
        list(_texts_to_documents(["a", "b"], None, ["a"]))
    with pytest.raises(ValueError):
        list(_texts_to_documents(["a", "b"], [{"a": "b"}], None))
    with pytest.raises(ValueError):
        list(_texts_to_documents(["a"], [{"a": "b"}, {"c": "d"}], None))
    with pytest.raises(ValueError):
        list(_texts_to_documents(["a"], None, ["a", "b"]))


def test_nodes_to_documents() -> None:
    nodes = [
        Node(
            id="a",
            text="some text a",
            links=[Link.incoming(kind="hyperlink", tag="http://b")],
        ),
        Node(id="b", text="some text b", metadata={"c": "d"}),
    ]
    assert list(_nodes_to_documents(nodes)) == [
        Document(
            id="a",
            metadata={"links": [Link.incoming(kind="hyperlink", tag="http://b")]},
            page_content="some text a",
        ),
        Document(id="b", metadata={"c": "d"}, page_content="some text b"),
    ]
