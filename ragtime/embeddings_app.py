"""
A toy FastAPI app wrapping some features of a vector database (Chroma)

Some example PDFs you might import are:

`https://arxiv.org/pdf/2405.17399` - Transformers Can Do Arithmetic with the
Right Embeddings

`https://arxiv.org/pdf/2405.17428` - NV-Embed: Improved Techniques for Training LLMs
as Generalist Embedding Models


If you want to run this using OpenAI embedding models,
you must set the following environment variables:

```
export EMBEDDING_MODEL=openai
export OPENAI_API_KEY=your-openai-api-key
```

if EMBEDDING_MODEL is unset or anything other than `openai`, the app will
attemt to run an Ollama hosted model.
"""

from fastapi import FastAPI
from ragtime.lib.embeddings import EmbeddingDatabase, Document
from typing import Any


app = FastAPI(
    title="Serve Chroma Database",
    version="0.1",
    description=__doc__,
)

vector_db = EmbeddingDatabase()


@app.post("/embed/")
async def embed_doc(document_uri: str) -> str:
    """
    Add a PDF to the database.
    Pass the location of a local file or valid URL.
    """
    await vector_db.add_doc([document_uri])
    return f"Thanks for indexing {document_uri}"


@app.get("/query/")
async def search(search_query: str) -> list[tuple[Document, float]]:
    """
    Runs a search against the DB by vectorizing the provided query text.
    Returns the closest matching document (actually chunk of a document)
    in the database.
    """
    results = await vector_db.search(search_query)
    return results


@app.get("/inspect/")
async def inspect(page: int | None = None) -> dict[str, Any]:
    """
    Returns contents of document DB.
    Limited to 10 results at a time.
    Page through results by passing the `page` parameter.

    """
    return vector_db.debug(page=page)


def main():
    import uvicorn

    uvicorn.run(app=app, host="localhost", port=8000)


if __name__ == "__main__":
    main()
