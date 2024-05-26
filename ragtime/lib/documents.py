from typing import Iterable
from langchain_core.documents import Document


def format_docs(docs: Iterable[Document]):
    return "\n\n".join(doc.page_content for doc in docs)
