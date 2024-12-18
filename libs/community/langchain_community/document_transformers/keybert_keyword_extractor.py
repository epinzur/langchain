from typing import Any, Sequence

from langchain_core.documents import BaseDocumentTransformer, Document

class KeybertKeywordExtractor(BaseDocumentTransformer):
    def __init__(
        self,
        *,
        batch_size: int = 8,
        metadata_key: str = "keywords",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        """Extract keywords using `KeyBERT <https://maartengr.github.io/KeyBERT/>`_.

        KeyBERT is a minimal and easy-to-use keyword extraction technique that
        leverages BERT embeddings to create keywords and keyphrases that are most
        similar to a document.

        The KeybertKeywordExtractor uses KeyBERT add a list of keywords to a
        document's metadata.

        Example::

            extractor = KeybertKeywordExtractor()
            results = extractor.extract_one("lorem ipsum...")

        .. seealso::

            - :mod:`How to use a graph vector store <langchain_community.graph_vectorstores>`
            - :class:`How to create links between documents <langchain_community.graph_vectorstores.links.Link>`

        How to link Documents on common keywords using Keybert
        ======================================================

        Preliminaries
        -------------

        Install the keybert package:

        .. code-block:: bash

            pip install -q langchain_community keybert

        Usage
        -----

        We load the ``state_of_the_union.txt`` file, chunk it, then for each chunk we
        extract keyword links and add them to the chunk.

        Using extract_one()
        ^^^^^^^^^^^^^^^^^^^

        We can use :meth:`extract_one` on a document to get the links and add the links
        to the document metadata with
        :meth:`~langchain_community.graph_vectorstores.links.add_links`::

            from langchain_community.document_loaders import TextLoader
            from langchain_community.graph_vectorstores import CassandraGraphVectorStore
            from langchain_community.graph_vectorstores.extractors import KeybertLinkExtractor
            from langchain_community.graph_vectorstores.links import add_links
            from langchain_text_splitters import CharacterTextSplitter

            loader = TextLoader("state_of_the_union.txt")

            raw_documents = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

            documents = text_splitter.split_documents(raw_documents)
            keyword_extractor = KeybertLinkExtractor()

            for document in documents:
                links = keyword_extractor.extract_one(document)
                add_links(document, links)

            print(documents[0].metadata)

        .. code-block:: output

            {'source': 'state_of_the_union.txt', 'links': [Link(kind='kw', direction='bidir', tag='ukraine'), Link(kind='kw', direction='bidir', tag='ukrainian'), Link(kind='kw', direction='bidir', tag='putin'), Link(kind='kw', direction='bidir', tag='vladimir'), Link(kind='kw', direction='bidir', tag='russia')]}

        Using LinkExtractorTransformer
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        Using the :class:`~langchain_community.graph_vectorstores.extractors.link_extractor_transformer.LinkExtractorTransformer`,
        we can simplify the link extraction::

            from langchain_community.document_loaders import TextLoader
            from langchain_community.graph_vectorstores.extractors import (
                KeybertLinkExtractor,
                LinkExtractorTransformer,
            )
            from langchain_text_splitters import CharacterTextSplitter

            loader = TextLoader("state_of_the_union.txt")
            raw_documents = loader.load()

            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            documents = text_splitter.split_documents(raw_documents)

            transformer = LinkExtractorTransformer([KeybertLinkExtractor()])
            documents = transformer.transform_documents(documents)

            print(documents[0].metadata)

        .. code-block:: output

            {'source': 'state_of_the_union.txt', 'links': [Link(kind='kw', direction='bidir', tag='ukraine'), Link(kind='kw', direction='bidir', tag='ukrainian'), Link(kind='kw', direction='bidir', tag='putin'), Link(kind='kw', direction='bidir', tag='vladimir'), Link(kind='kw', direction='bidir', tag='russia')]}

        The documents with keyword links can then be added to a :class:`~langchain_community.graph_vectorstores.base.GraphVectorStore`::

            from langchain_community.graph_vectorstores import CassandraGraphVectorStore

            store = CassandraGraphVectorStore.from_documents(documents=documents, embedding=...)

        Args:
            kind: Kind of links to produce with this extractor.
            embedding_model: Name of the embedding model to use with KeyBERT.
            extract_keywords_kwargs: Keyword arguments to pass to KeyBERT's
                ``extract_keywords`` method.
        """  # noqa: E501
        try:
            import keybert

            self._kw_model = keybert.KeyBERT(model=embedding_model)
        except ImportError:
            raise ImportError(
                "keybert is required for KeybertLinkExtractor. "
                "Please install it with `pip install keybert`."
            ) from None

        self._batch_size = batch_size
        self._metadata_key = metadata_key


    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Extracts properties from documents using KeyBERT."""
        for i in range(0, len(documents), self._batch_size):
            batch = documents[i:i+self._batch_size]
            texts = [item.page_content for item in batch]
            extracted = self._kw_model.extract_keywords(docs = texts, **kwargs)
            if len(texts) == 1:
                # Even though we pass a list, if it contains one item, keybert will
                # flatten it. This means it's easier to just call the special case
                # for one item.
                batch[0].metadata[self._metadata_key] = [kw[0] for kw in extracted]
            else:
                for i, keywords in enumerate(extracted):
                    batch[i].metadata[self._metadata_key] = [kw[0] for kw in keywords]
        return documents
