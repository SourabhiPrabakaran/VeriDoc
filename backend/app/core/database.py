from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.core.config import settings

# Create engine with sqlite URL
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} # Required for single-threaded SQLite in FastAPI
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    """
    Dependency injection helper to yield db sessions and safely close them.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
