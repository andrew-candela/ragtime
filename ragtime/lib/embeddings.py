"""
Embedding Models and Document stores

"""

from abc import ABC, abstractmethod
from itertools import chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
import numpy as np
import os
from typing import Any, Iterable
import logging
from enum import Enum
from threading_practice.aiocrawl import AIOCrawler, AIOProcessor, ScrapedContent


logger = logging.getLogger(__name__)


CHROMA_COLLECTION_NAME = "RAGTIME"


class ContentType(Enum):
    FILE = "file"
    TEXT = "text"


def get_pg_connection() -> str:
    return (
        "postgresql+psycopg://"
        f"{os.environ['POSTGRES_USER']}:"
        f"{os.environ['POSTGRES_PASSWORD']}@"
        f"{os.environ['POSTGRES_HOST']}:"
        f"5432/{os.environ['POSTGRES_DB']}"
    )


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


def docs_from_text(content: str) -> Iterable[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separators=["\n"]
    )
    for doc in splitter.split_text(content):
        yield Document(page_content=doc)


def get_docs(
    contents: list[str], content_type: ContentType = ContentType.FILE
) -> Iterable[Document]:
    """
    Load the corpus.
    Files can be local files or URI of something on the internet.
    Alternatively you can pass raw text.
    I'm not doing any pre-processing. We simply hand the raw text to the
    model for vectorization.
    """
    for content in contents:
        if content_type == ContentType.FILE:
            yield from PyPDFLoader(content).lazy_load()
        else:
            yield from docs_from_text(content)


class EmbeddingDatabase(ABC):

    @abstractmethod
    async def add_doc(self, doc_uris: list[str]): ...

    @abstractmethod
    async def search(self, search_query: str) -> list[tuple[Document, float]]: ...

    @abstractmethod
    def debug(self, page: int | None = None) -> dict[str, Any]: ...


class PGVectorDB(EmbeddingDatabase):
    def __init__(self):
        self._db = PGVector(
            embeddings=embedding_provider,
            collection_name=CHROMA_COLLECTION_NAME,
            connection=get_pg_connection(),
        )

    async def website_processor(self, content: ScrapedContent):
        """
        content.get_text() -> generator str

        """
        docs = (doc for doc in (docs_from_text(part) for part in content.get_text()))
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n"]
        )
        splits = text_splitter.split_documents(chain(*docs))
        await self._db.aadd_documents(splits)

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

    async def crawl_url(self, url):
        """
        Crawls a website and indexes all information held there.
        """
        processor = AIOProcessor(process_function=self.website_processor, workers=1)
        crawler = AIOCrawler(workers=5, max_depth=4, max_pages=30)
        await crawler.crawl(url, processor=processor)

    async def search(self, search_query: str) -> list[tuple[Document, float]]:
        docs = await self._db.asimilarity_search_with_relevance_scores(
            query=search_query, k=1
        )
        return docs

    def debug(self, page: int | None = None) -> dict[str, Any]:
        raise NotImplementedError()


class ChromaVectorDB(EmbeddingDatabase):
    def __init__(self):
        self._db = Chroma(
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=embedding_provider,
            collection_metadata={"hnsw:space": "cosine"},
        )

    async def _website_processor(self, content: ScrapedContent):
        """
        content.get_text() -> generator str

        """
        docs = (doc for doc in (docs_from_text(part) for part in content.get_text()))
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n"]
        )
        splits = text_splitter.split_documents(chain(*docs))
        await self._db.aadd_documents(splits)

    async def crawl_url(self, url):
        """
        Crawls a website and indexes all information held there.
        """
        processor = AIOProcessor(process_function=self._website_processor, workers=1)
        crawler = AIOCrawler(workers=5, max_depth=4, max_pages=30)
        await crawler.crawl(url, processor=processor)

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
