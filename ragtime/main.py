"""
Run the pipeline.

Collect input from the command line, then pass it to the RAG pipeline.
Print the LLM output to STDOut
"""

from ragtime.lib.embeddings import create_chroma_vector_db_and_load_docs
from ragtime.lib.documents import format_docs
from ragtime.lib.prompt import rag_prompt
from ragtime.lib.model import ollama_llm
from ragtime.lib.output_parser import string_output_parser
from langchain_core.runnables import RunnablePassthrough
import sys


retriever = create_chroma_vector_db_and_load_docs(
    [
        "corpus/giraffes.txt",
        "corpus/hippos.txt",
        "corpus/monster_trucks.txt",
    ]
).as_retriever()


def main():
    user_input = sys.argv[1]
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | ollama_llm
        | string_output_parser
    )
    print(rag_chain.invoke(user_input))


main()
