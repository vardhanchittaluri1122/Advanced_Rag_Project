
import logging
from functools import lru_cache

from sqlalchemy import create_engine, text

from app.core.config import DATABASE_URL, TABLE_NAME

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_sql_engine():
    return create_engine(
        DATABASE_URL,
        pool_pre_ping=True
    )


def keyword_search(
    query: str,
    top_k: int = 5
):
    """
    Search document chunks using PostgreSQL full-text search.
    """

    sql = text(f"""
        SELECT content, langchain_metadata
        FROM {TABLE_NAME}
        WHERE to_tsvector('english', content)
              @@ websearch_to_tsquery('english', :query)
        ORDER BY ts_rank_cd(
            to_tsvector('english', content),
            websearch_to_tsquery('english', :query)
        ) DESC
        LIMIT :limit
    """)

    try:
        with _get_sql_engine().connect() as connection:

            rows = connection.execute(
                sql,
                {
                    "query": query,
                    "limit": top_k
                }
            ).mappings().all()

            return [
                {
                    "content": row["content"],
                    "metadata": row["langchain_metadata"] or {}
                }
                for row in rows
            ]

    except Exception:
        logger.exception(
            "PostgreSQL keyword search failed"
        )
        raise
