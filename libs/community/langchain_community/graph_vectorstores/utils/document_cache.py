from typing import Iterable, Iterator

from langchain_core.documents import Document


class DocumentCache:
    documents: dict[str, Document] = {}

    def add_document(self, doc: Document) -> None:
        if doc.id is not None:
            self.documents[doc.id] = doc

    def add_documents(self, docs: Iterable[Document]) -> None:
        for doc in docs:
            self.add_document(doc=doc)

    def get_by_document_ids(
        self,
        doc_ids: Iterable[str],
    ) -> Iterator[Document]:
        for doc_id in doc_ids:
            if doc_id in self.documents:
                yield self.documents[doc_id]
            else:
                msg = f"unexpected, cache should contain id: {doc_id}"
                raise RuntimeError(msg)
