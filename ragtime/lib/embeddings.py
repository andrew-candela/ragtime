"""
Embedding Models and Document stores

"""

from langchain_chroma import Chroma
from langchain.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings

ollama_embedder = OllamaEmbeddings(model="mxbai-embed-large")


def get_docs(files: list[str]) -> list[Document]:
    """
    Load the corpus
    """
    docs = []
    for file in files:
        docs.extend(TextLoader(file).load())
    return docs


def create_chroma_vector_db_and_load_docs(doc_names: list[str]) -> Chroma:
    docs = get_docs(doc_names)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    return Chroma.from_documents(documents=splits, embedding=ollama_embedder)
