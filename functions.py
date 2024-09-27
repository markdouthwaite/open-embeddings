from app.services import documents
from app.models.document import Document
from app.db import get_db
from app.config import get_chunking_strategy, get_embedding_model


def handle_create_document(request):
    db = get_db()
    embedding_model = get_embedding_model()
    chunking_strategy = get_chunking_strategy()
    doc = Document(**request.json)
    document = documents.create_document(
        db, doc, model=embedding_model, chunking_strategy=chunking_strategy
    )
    return dict(document)
