

"""Apache Cassandra DB graph document index implementation."""

from __future__ import annotations

import logging
from typing import (
    TYPE_CHECKING,
    Any,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
)

from langchain_core._api import beta
from langchain_core.documents import Document
from langchain_core.indexing import DocumentIndex, UpsertResponse
from langchain_core.indexing.base import DeleteResponse
from langchain_community.utilities.cassandra import SetupMode
from langchain_community.vectorstores.cassandra import Cassandra as CassandraVectorStore


if TYPE_CHECKING:
    from cassandra.cluster import Session
    from langchain_core.embeddings import Embeddings

    from langchain_core.callbacks.manager import (
        CallbackManagerForRetrieverRun,
    )


logger = logging.getLogger(__name__)


class CassandraGraphIndex(DocumentIndex):

    def __init__(
        self,
        embedding: Embeddings,
        session: Session | None = None,
        keyspace: str | None = None,
        table_name: str = "",
        ttl_seconds: int | None = None,
        *,
        setup_mode: SetupMode = SetupMode.SYNC,
        metadata_deny_list: Optional[list[str]] = None,
    ) -> None:
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
                    table_name = 'my_graph_vector_store'
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
            setup_mode: mode used to create the Cassandra table (SYNC,
                ASYNC or OFF).
            metadata_deny_list: Optional list of metadata keys to not index.
                i.e. to fine-tune which of the metadata fields are indexed.
                Note: if you plan to have massive unique text metadata entries,
                consider not indexing them for performance
                (and to overcome max-length limitations).
                Note: the `metadata_indexing` parameter from
                langchain_community.utilities.cassandra.Cassandra is not
                exposed since CassandraGraphVectorStore only supports the
                deny_list option.
        """

        self.vector_store=CassandraVectorStore(
            embedding=embedding,
            session=session,
            keyspace=keyspace,
            table_name=table_name,
            ttl_seconds=ttl_seconds,
            setup_mode=setup_mode,
            metadata_indexing=("deny_list", metadata_deny_list),
        )



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