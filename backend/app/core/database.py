import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from app.core.config import settings

logger = logging.getLogger(__name__)

Base = declarative_base()

# Configure database engine
db_url = settings.DATABASE_URL
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

try:
    engine = create_engine(
        db_url,
        pool_pre_ping=True,
        pool_recycle=300,
    )
    # Test connection
    with engine.connect() as conn:
        logger.info("Successfully connected to PostgreSQL database.")
except Exception as e:
    logger.warning(f"Failed to connect to primary DB ({e}). Falling back to local SQLite engine.")
    engine = create_engine(
        "sqlite:///./datareadiness.db",
        connect_args={"check_same_thread": False},
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    import app.models.dataset  # noqa: F401
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialized successfully.")
