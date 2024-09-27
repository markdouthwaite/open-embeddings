import os
from functools import partial
from app.models.embedding import EmbeddingModel
from app.models.document import DocumentChunkingStrategy
from app.services.chunking import fixed_chunking_with_overlap
from app.services.embeddings import init_embedding_model

DATABASE_PASSWORD = os.getenv("DATABASE_PASSWORD")
DATABASE_HOST = os.getenv("DATABASE_HOST")
DATABASE_USER = os.getenv("DATABASE_USER")
DATABASE_NAME = os.getenv("DATABASE_NAME")
DATABASE_PORT = os.getenv("DATABASE_PORT")
DATABASE_URI = f"postgresql://{DATABASE_USER}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"
DATABASE_DEBUG = True
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_chunking_strategy() -> DocumentChunkingStrategy:
    return partial(fixed_chunking_with_overlap, width=1000, overlap=150)


def get_embedding_model() -> EmbeddingModel:
    return init_embedding_model(EMBEDDING_MODEL)
