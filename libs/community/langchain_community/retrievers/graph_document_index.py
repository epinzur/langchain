import asyncio
from typing import Any, Dict, Iterable, Iterator, List, Literal, Tuple, Union, cast, override

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)

from langchain_core.documents import Document
from langchain_core.indexing import DocumentIndex
from langchain_core.runnables.config import run_in_executor
from langchain_core.vectorstores import VectorStore

from pydantic import BaseModel, PrivateAttr

from abc import abstractmethod

class Edge():
    direction: str = Literal['bi-dir', 'in', 'out']
    key: str
    value: Any

    def __init__(self, direction=str, key=str, value=Any):
        self.direction = direction
        self.key = key
        self.value = value

    def __str__(self):
        return f"{self.direction}:{self.key}:{self.value}"

class GraphVectorStore(VectorStore):
    use_metadata_expansion: bool = False

    def __init__(
        self,
        *pargs: Any,
        use_metadata_expansion: bool = False,
        **kwargs: Any,
    ) -> None:
        self.use_metadata_expansion = use_metadata_expansion
        super().__init__(*pargs, **kwargs)

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

    @override
    def add_documents(self, documents, **kwargs):
        if self.use_metadata_expansion:
            # pre-insertion, expand all list metadata fields
            # remove on return to user
            #
            # can't traverse across documents inserted
            # outside the DocumentIndex
            #
            # need to search two ways for every edge:
            # (key == value) AND (key_value == "constant")
            pass
        return super().add_documents(documents, **kwargs)

    @override
    async def aadd_documents(self, documents, **kwargs):
        return await super().aadd_documents(documents, **kwargs)