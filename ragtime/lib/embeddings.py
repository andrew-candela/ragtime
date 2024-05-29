"""
Embedding Models and Document stores

"""

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
import numpy as np
import os
from typing import Any, Iterable
import logging


logger = logging.getLogger(__name__)


CHROMA_COLLECTION_NAME = "RAGTIME"


class OllamaModels:
    quantized_embedding = "mxbai-embed-large"
    quantized_llama2 = "llama2"


def normalize_vector(vector: list[float]) -> list[float]:
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("Zero division error when normalizing")
    return list(np.asarray(vector) / norm)


class NormalizedOllamaEmbeddings(OllamaEmbeddings):
    """
    Wraps the `embed_query` and `embed_documents`
    methods of the base OllamaEmbeddings class with logic that
    normalizes the vectors returned.
    """

    def embed_query(self, text: str) -> list[float]:
        return normalize_vector(super().embed_query(text))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [normalize_vector(vector) for vector in super().embed_documents(texts)]


embedding_provider = (
    NormalizedOllamaEmbeddings(model=OllamaModels.quantized_embedding)
    if os.getenv("EMBEDDING_MODEL", "ollama") != "openai"
    else OpenAIEmbeddings()
)


def get_docs(files: list[str]) -> Iterable[Document]:
    """
    Load the corpus.
    Files can be local files or URI of something on the internet.
    I'm not doing any pre-processing. We just hand the raw text to the
    model for vectorization.
    """
    for file in files:
        yield from PyPDFLoader(file).lazy_load()


class EmbeddingDatabase:
    def __init__(self):
        self._db = Chroma(
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=embedding_provider,
        )

    async def add_doc(self, doc_uris: list[str]):
        """
        Adds a document to the DB
        """
        docs = get_docs(doc_uris)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n"]
        )
        splits = text_splitter.split_documents(docs)
        await self._db.aadd_documents(splits)
        logger.debug("Finished indexing docs: %s", doc_uris)

    async def search(self, search_query: str) -> list[tuple[Document, float]]:
        docs = await self._db.asimilarity_search_with_relevance_scores(
            query=search_query, k=1
        )
        return docs

    def debug(self, page: int | None = None) -> dict[str, Any]:
        return self._db.get(
            include=["embeddings", "metadatas", "documents"], limit=10, offset=page
        )
