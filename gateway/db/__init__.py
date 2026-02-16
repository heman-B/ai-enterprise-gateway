# gateway/db/__init__.py
# Database abstraction layer (SQLite for dev, PostgreSQL for production)

from .database import get_db_connection, is_postgres

__all__ = ["get_db_connection", "is_postgres"]
