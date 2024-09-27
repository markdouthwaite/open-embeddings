from fastapi import APIRouter, Depends
from sqlmodel import Session
from app.models.embedding import EmbeddingModel
from app.models.search import SearchResults
from app.services import search
from app.db import get_db
from app.config import get_embedding_model


router = APIRouter()


@router.get("/", response_model=SearchResults)
def get_search_results(
    query: str,
    max_k: int = 3,
    db: Session = Depends(get_db),
    embedding_model: EmbeddingModel = Depends(get_embedding_model),
):
    results = search.get_search_results(db, embedding_model, query, max_k)
    return results
