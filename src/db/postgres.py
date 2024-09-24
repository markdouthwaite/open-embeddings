from sqlmodel import create_engine, Session
from src.models.embedding import Embedding
from src.models.document import Document
from src.db.base import EmbeddingsRepository, DocumentRepository


class PostgresEmbeddingRepository(EmbeddingsRepository):
    def __init__(self, path: str):
        self.path = path
        self.engine = create_engine(path)

    def add(self, embedding: Embedding):
        with Session(self.engine) as session:
            session.add(embedding)
            session.commit()


class PostgresDocumentRepository(DocumentRepository):
    def __init__(self, path: str):
        self.path = path
        self.engine = create_engine(path)

    def add(self, document: Document):
        with Session(self.engine) as session:
            session.add(document)
            session.commit()
        print("done")