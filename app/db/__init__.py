from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel, Session
from app.config import DATABASE_URI, DATABASE_DEBUG

engine = create_engine(DATABASE_URI, echo=DATABASE_DEBUG)

_Session = sessionmaker(autocommit=False, autoflush=False, bind=engine, class_=Session)


def get_db() -> Generator[Session, None, None]:
    db = _Session()
    try:
        yield db
    finally:
        db.close()


def init_db():
    SQLModel.metadata.create_all(bind=engine)
