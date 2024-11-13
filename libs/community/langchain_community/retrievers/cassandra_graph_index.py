

"""Apache Cassandra DB graph document index implementation."""

from __future__ import annotations

import logging
import secrets
from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from pydantic import Field, PrivateAttr

from langchain_core._api import beta
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.indexing import DocumentIndex, UpsertResponse
from langchain_core.indexing.base import DeleteResponse
from langchain_community.utilities.cassandra import SetupMode
from langchain_community.vectorstores.cassandra import Cassandra as CassandraVectorStore


if TYPE_CHECKING:
    from cassandra.cluster import Session


    from langchain_core.callbacks.manager import (
        CallbackManagerForRetrieverRun,
    )


logger = logging.getLogger(__name__)


class CassandraGraphIndex(DocumentIndex):
    search_type: str
    search_kwargs: dict[str, Any]
    edges: Iterable[Union[str, Tuple[str, str]]]
    k: int
    _vector_store: CassandraVectorStore = PrivateAttr()


    def __init__(
        self,
        search_type: str,
        search_kwargs: dict[str, Any],
        edges: Iterable[Union[str, Tuple[str, str]]],
        k: int,
        embedding: Embeddings,
        session: Optional[Session] = None,
        keyspace: Optional[str] = None,
        table_name: str = "",
        ttl_seconds: Optional[int] = None,
        *,
        body_index_options: Optional[List[Tuple[str, Any]]] = None,
        setup_mode: SetupMode = SetupMode.SYNC,
        metadata_indexing: Union[Tuple[str, Iterable[str]], str] = "all",
        **kwargs: Any,
    ) -> None:
        # TODO: The example needs updating
        """Apache Cassandra(R) for graph-index workloads.

        To use it, you need a recent installation of the `cassio` library
        and a Cassandra cluster / Astra DB instance supporting vector capabilities.

        Example:
            .. code-block:: python

                    from langchain_community.retrievers import
                        CassandraGraphIndex
                    from langchain_openai import OpenAIEmbeddings

                    embeddings = OpenAIEmbeddings()
                    session = ...             # create your Cassandra session object
                    keyspace = 'my_keyspace'  # the keyspace should exist already
                    table_name = 'my_graph_index'
                    index = CassandraGraphIndex(
                        embeddings,
                        session,
                        keyspace,
                        table_name,
                    )

        Args:
            embedding: Embedding function to use.
            session: Cassandra driver session. If not provided, it is resolved from
                cassio.
            keyspace: Cassandra keyspace. If not provided, it is resolved from cassio.
            table_name: Cassandra table (required).
            ttl_seconds: Optional time-to-live for the added texts.
            body_index_options: Optional options used to create the body index.
                Eg. body_index_options = [cassio.table.cql.STANDARD_ANALYZER]
            setup_mode: mode used to create the Cassandra table (SYNC,
                ASYNC or OFF).
            metadata_indexing: Optional specification of a metadata indexing policy,
                i.e. to fine-tune which of the metadata fields are indexed.
                It can be a string ("all" or "none"), or a 2-tuple. The following
                means that all fields except 'f1', 'f2' ... are NOT indexed:
                    metadata_indexing=("allowlist", ["f1", "f2", ...])
                The following means all fields EXCEPT 'g1', 'g2', ... are indexed:
                    metadata_indexing("denylist", ["g1", "g2", ...])
                The default is to index every metadata field.
                Note: if you plan to have massive unique text metadata entries,
                consider not indexing them for performance
                (and to overcome max-length limitations).
        """
        super().__init__(search_type=search_type, search_kwargs=search_kwargs, edges=edges, k=k, **kwargs)

        self._vector_store = CassandraVectorStore(
            embedding=embedding,
            session=session,
            keyspace=keyspace,
            table_name=table_name,
            ttl_seconds=ttl_seconds,
            body_index_options=body_index_options,
            setup_mode=setup_mode,
            metadata_indexing=metadata_indexing,
            **kwargs,
        )




    def __post_init__(self):
        try:
            from cassandra.cluster import Session
        except (ImportError, ModuleNotFoundError):
            raise ImportError(
                "Could not import cassio python package. "
                "Please install it with `pip install cassio`."
            )

        # Assuming embedding and other attributes are initialized as class-level fields or injected post-init



    def upsert(self, items: Sequence[Document], /, **kwargs: Any) -> UpsertResponse:
        """Upsert documents into the index.

        The upsert functionality should utilize the ID field of the content object
        if it is provided. If the ID is not provided, the upsert method is free
        to generate an ID for the content.

        When an ID is specified and the content already exists in the vectorstore,
        the upsert method should update the content with the new data. If the content
        does not exist, the upsert method should add the item to the vectorstore.

        Args:
            items: Sequence of documents to add to the vectorstore.
            **kwargs: Additional keyword arguments.

        Returns:
            UpsertResponse: A response object that contains the list of IDs that were
            successfully added or updated in the vectorstore and the list of IDs that
            failed to be added or updated.
        """
        for item in items:
            if item.id is None:
                item.id = secrets.token_hex(8)
        ids = self._vector_store.add_documents(documents=items)
        return UpsertResponse(succeeded=ids)


    def delete(self, ids: Optional[list[str]] = None, **kwargs: Any) -> DeleteResponse:
        """Delete by IDs or other criteria.

        Calling delete without any input parameters should raise a ValueError!

        Args:
            ids: List of ids to delete.
            kwargs: Additional keyword arguments. This is up to the implementation.
                For example, can include an option to delete the entire index,
                or else issue a non-blocking delete etc.

        Returns:
            DeleteResponse: A response object that contains the list of IDs that were
            successfully deleted and the list of IDs that failed to be deleted.
        """
        result = self._vector_store.delete(ids=ids)
        return DeleteResponse(
            succeeded=ids if result else [],
            num_deleted=len(ids) if result else 0,
            failed=ids if not result else [],
            num_failed=len(ids) if not result else 0,
        )


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
        docs = [
            self._vector_store.get_by_document_id(document_id=id)
            for id in ids
        ]
        return [doc for doc in docs if doc is not None]

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



CassandraGraphIndex.model_rebuild()
