"""Feedback storage module.

Stores user feedback on RAG responses using SQLite for later analysis.
"""

import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

FEEDBACK_DB_PATH = Path(__file__).parent / "feedback.db"


@dataclass
class FeedbackEntry:
    question: str
    rewritten_query: Optional[str]
    hyde_used: bool
    chunks_retrieved: str  # JSON string of chunk metadata (no full text)
    final_score: float
    score_level: str  # HIGH / MEDIUM / LOW
    useful: bool  # True = 👍, False = 👎
    timestamp: str

    def to_dict(self) -> dict:
        return asdict(self)


def init_db(db_path: str = None) -> None:
    """Create the feedback table if it doesn't exist."""
    db_path = db_path or str(FEEDBACK_DB_PATH)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            rewritten_query TEXT,
            hyde_used INTEGER NOT NULL,
            chunks_retrieved TEXT NOT NULL,
            final_score REAL NOT NULL,
            score_level TEXT NOT NULL,
            useful INTEGER NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def save_feedback(entry: FeedbackEntry, db_path: str = None) -> None:
    """Save a feedback entry to the database."""
    db_path = db_path or str(FEEDBACK_DB_PATH)
    init_db(db_path)

    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT INTO feedback (question, rewritten_query, hyde_used, chunks_retrieved,
                               final_score, score_level, useful, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            entry.question,
            entry.rewritten_query,
            int(entry.hyde_used),
            entry.chunks_retrieved,
            entry.final_score,
            entry.score_level,
            int(entry.useful),
            entry.timestamp,
        ),
    )
    conn.commit()
    conn.close()


def load_feedback(db_path: str = None) -> List[FeedbackEntry]:
    """Load all feedback entries from the database."""
    db_path = db_path or str(FEEDBACK_DB_PATH)
    init_db(db_path)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM feedback").fetchall()
    conn.close()

    entries = []
    for row in rows:
        entries.append(FeedbackEntry(
            question=row["question"],
            rewritten_query=row["rewritten_query"],
            hyde_used=bool(row["hyde_used"]),
            chunks_retrieved=row["chunks_retrieved"],
            final_score=row["final_score"],
            score_level=row["score_level"],
            useful=bool(row["useful"]),
            timestamp=row["timestamp"],
        ))
    return entries
