from langchain import hub
from langchain_core.prompts import PromptTemplate

rag_prompt_text = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {question}

Context: {context}

Answer:
"""  # noqa: E501


hub_prompt = hub.pull("rlm/rag-prompt")
rag_prompt = PromptTemplate.from_template(rag_prompt_text)
