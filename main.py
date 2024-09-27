from fastapi import FastAPI
from app.api.v1.routes import documents, embeddings, search

app = FastAPI(
    title="An Open Embeddings API",
    description="An API for managing document retrieval for RAG applications",
    version="0.0.1",
    docs_url="/docs",
)

app.include_router(documents.router, prefix="/api/v1/documents", tags=["Documents"])
app.include_router(embeddings.router, prefix="/api/v1/embeddings", tags=["Embeddings"])
app.include_router(search.router, prefix="/api/v1/search", tags=["Search"])
