"""
SQLite database layer for storing research paper nodes as JSON.
"""

import json
import sqlite3
import logging
from datetime import datetime, timezone
from typing import Optional
from contextlib import contextmanager

from app.config import settings
from app.models import PaperNode

logger = logging.getLogger(__name__)


def _get_connection() -> sqlite3.Connection:
    """Get a SQLite connection."""
    conn = sqlite3.connect(settings.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def get_db():
    """Context manager for database connections."""
    conn = _get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Initialize the database and create tables if they don't exist."""
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS research_papers (
                unique_id TEXT PRIMARY KEY,
                url TEXT NOT NULL DEFAULT '',
                data TEXT NOT NULL DEFAULT '{}',
                parent_id TEXT,
                depth INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY (parent_id) REFERENCES research_papers(unique_id)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_parent_id 
            ON research_papers(parent_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_depth 
            ON research_papers(depth)
        """)
    logger.info("Database initialized at %s", settings.DATABASE_PATH)


def store_paper_node(
    unique_id: str,
    url: str,
    data: dict,
    parent_id: Optional[str] = None,
    depth: int = 0,
) -> bool:
    """
    Store a paper node in the database.
    Returns True if inserted, False if already exists.
    """
    now = datetime.now(timezone.utc).isoformat()
    data_json = json.dumps(data, ensure_ascii=False)

    with get_db() as conn:
        try:
            conn.execute(
                """
                INSERT INTO research_papers (unique_id, url, data, parent_id, depth, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (unique_id, url, data_json, parent_id, depth, now),
            )
            logger.info("Stored paper node: %s (depth=%d)", unique_id, depth)
            return True
        except sqlite3.IntegrityError:
            logger.warning("Paper node already exists: %s", unique_id)
            return False


def paper_exists(unique_id: str) -> bool:
    """Check if a paper node already exists in the database."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT 1 FROM research_papers WHERE unique_id = ?",
            (unique_id,),
        ).fetchone()
        return row is not None


def get_paper_by_id(unique_id: str) -> Optional[PaperNode]:
    """Retrieve a paper node by its unique ID."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM research_papers WHERE unique_id = ?",
            (unique_id,),
        ).fetchone()

        if not row:
            return None

        return PaperNode(
            unique_id=row["unique_id"],
            url=row["url"],
            data=json.loads(row["data"]),
            parent_id=row["parent_id"],
            depth=row["depth"],
            created_at=row["created_at"],
        )


def get_all_papers(limit: int = 100, offset: int = 0) -> list[PaperNode]:
    """Retrieve all paper nodes with pagination."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM research_papers ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()

        return [
            PaperNode(
                unique_id=row["unique_id"],
                url=row["url"],
                data=json.loads(row["data"]),
                parent_id=row["parent_id"],
                depth=row["depth"],
                created_at=row["created_at"],
            )
            for row in rows
        ]


def get_paper_count() -> int:
    """Get total count of paper nodes."""
    with get_db() as conn:
        row = conn.execute("SELECT COUNT(*) as cnt FROM research_papers").fetchone()
        return row["cnt"]


def get_children(parent_id: str) -> list[PaperNode]:
    """Get all direct children of a paper node."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM research_papers WHERE parent_id = ? ORDER BY created_at",
            (parent_id,),
        ).fetchall()

        return [
            PaperNode(
                unique_id=row["unique_id"],
                url=row["url"],
                data=json.loads(row["data"]),
                parent_id=row["parent_id"],
                depth=row["depth"],
                created_at=row["created_at"],
            )
            for row in rows
        ]


def get_citation_tree(root_id: str) -> Optional[dict]:
    """
    Build a citation tree starting from a root paper node.
    Returns a nested dict representing the tree structure.
    """
    root = get_paper_by_id(root_id)
    if not root:
        return None

    def _build_tree(node: PaperNode) -> dict:
        children = get_children(node.unique_id)
        title = node.data.get("title", "") if isinstance(node.data, dict) else ""
        return {
            "unique_id": node.unique_id,
            "url": node.url,
            "title": title,
            "depth": node.depth,
            "children": [_build_tree(child) for child in children],
        }

    return _build_tree(root)
