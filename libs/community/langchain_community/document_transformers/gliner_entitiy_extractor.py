from typing import Any, Sequence

from langchain_core._api import beta
from langchain_core.documents import Document, BaseDocumentTransformer



@beta()
class GLiNEREntityExtractor(BaseDocumentTransformer):
    """Link documents with common named entities using `GLiNER`_.

    `GLiNER`_ is a Named Entity Recognition (NER) model capable of identifying any
    entity type using a bidirectional transformer encoder (BERT-like).

    The ``GLiNERLinkExtractor`` uses GLiNER to create links between documents that
    have named entities in common.

    Example::

        extractor = GLiNERLinkExtractor(
            labels=["Person", "Award", "Date", "Competitions", "Teams"]
        )
        results = extractor.extract_one("some long text...")

    .. _GLiNER: https://github.com/urchade/GLiNER

    .. seealso::

            - :mod:`How to use a graph vector store <langchain_community.graph_vectorstores>`
            - :class:`How to create links between documents <langchain_community.graph_vectorstores.links.Link>`

    How to link Documents on common named entities
    ==============================================

    Preliminaries
    -------------

    Install the ``gliner`` package:

    .. code-block:: bash

        pip install -q langchain_community gliner

    Usage
    -----

    We load the ``state_of_the_union.txt`` file, chunk it, then for each chunk we
    extract named entity links and add them to the chunk.

    Using extract_one()
    ^^^^^^^^^^^^^^^^^^^

    We can use :meth:`extract_one` on a document to get the links and add the links
    to the document metadata with
    :meth:`~langchain_community.graph_vectorstores.links.add_links`::

        from langchain_community.document_loaders import TextLoader
        from langchain_community.graph_vectorstores import CassandraGraphVectorStore
        from langchain_community.graph_vectorstores.extractors import GLiNERLinkExtractor
        from langchain_community.graph_vectorstores.links import add_links
        from langchain_text_splitters import CharacterTextSplitter

        loader = TextLoader("state_of_the_union.txt")
        raw_documents = loader.load()

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        documents = text_splitter.split_documents(raw_documents)

        ner_extractor = GLiNERLinkExtractor(["Person", "Topic"])
        for document in documents:
            links = ner_extractor.extract_one(document)
            add_links(document, links)

        print(documents[0].metadata)

    .. code-block:: output

        {'source': 'state_of_the_union.txt', 'links': [Link(kind='entity:Person', direction='bidir', tag='President Zelenskyy'), Link(kind='entity:Person', direction='bidir', tag='Vladimir Putin')]}

    Using LinkExtractorTransformer
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    Using the :class:`~langchain_community.graph_vectorstores.extractors.link_extractor_transformer.LinkExtractorTransformer`,
    we can simplify the link extraction::

        from langchain_community.document_loaders import TextLoader
        from langchain_community.graph_vectorstores.extractors import (
            GLiNERLinkExtractor,
            LinkExtractorTransformer,
        )
        from langchain_text_splitters import CharacterTextSplitter

        loader = TextLoader("state_of_the_union.txt")
        raw_documents = loader.load()

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        documents = text_splitter.split_documents(raw_documents)

        ner_extractor = GLiNERLinkExtractor(["Person", "Topic"])
        transformer = LinkExtractorTransformer([ner_extractor])
        documents = transformer.transform_documents(documents)

        print(documents[0].metadata)

    .. code-block:: output

        {'source': 'state_of_the_union.txt', 'links': [Link(kind='entity:Person', direction='bidir', tag='President Zelenskyy'), Link(kind='entity:Person', direction='bidir', tag='Vladimir Putin')]}

    The documents with named entity links can then be added to a :class:`~langchain_community.graph_vectorstores.base.GraphVectorStore`::

        from langchain_community.graph_vectorstores import CassandraGraphVectorStore

        store = CassandraGraphVectorStore.from_documents(documents=documents, embedding=...)

    Args:
        labels: List of kinds of entities to extract.
        kind: Kind of links to produce with this extractor.
        model: GLiNER model to use.
        extract_kwargs: Keyword arguments to pass to GLiNER.
    """  # noqa: E501

    def __init__(
        self,
        labels: List[str],
        *,
        batch_size: int = 8,
        metadata_label_prefix: str = "",
        model: str = "urchade/gliner_mediumv2.1",
    ):
        try:
            from gliner import GLiNER

            self._model = GLiNER.from_pretrained(model)

        except ImportError:
            raise ImportError(
                "gliner is required for GLiNEREntityExtractor. "
                "Please install it with `pip install gliner`."
            ) from None

        self._batch_size = batch_size
        self._labels = labels
        self._metadata_label_prefix = metadata_label_prefix

    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Extracts named entities from documents using GLiNER."""
        for i in range(0, len(documents), self._batch_size):
            batch = documents[i:i+self._batch_size]
            texts = [item.page_content for item in batch]
            extracted = self._model.batch_predict_entities(texts = texts, labels=self._labels, **kwargs)
            for i, entities in enumerate(extracted):
                for entity in entities:
                    label = self._metadata_label_prefix + entity["label"]
                    batch[i].metadata.setdefault(label, []).append(entity["text"])
        return documents
