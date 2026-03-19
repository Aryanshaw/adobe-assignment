import uuid
from typing import Optional
from datetime import datetime
from pydantic import BaseModel
from app.db.sqlite import sqlite_db
from app.logger import logger

class IngestedFile(BaseModel):
    id: str
    file_name: str
    file_hash: str
    summary: str
    created_at: Optional[datetime] = None

async def create_ingested_files_table():
    query = """
    CREATE TABLE IF NOT EXISTS ingested_files (
        id TEXT PRIMARY KEY,
        file_name TEXT,
        file_hash TEXT UNIQUE,
        summary TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """
    await sqlite_db.execute(query)
    logger.info("ingested_files table verified/created.")

async def create_ingested_file(file_name: str, file_hash: str, summary: str) -> str:
    file_id = str(uuid.uuid4())
    query = """
    INSERT INTO ingested_files (id, file_name, file_hash, summary)
    VALUES (?, ?, ?, ?)
    """
    await sqlite_db.execute(query, (file_id, file_name, file_hash, summary))
    return file_id

async def get_ingested_file_by_hash(file_hash: str) -> Optional[IngestedFile]:
    query = "SELECT id, file_name, file_hash, summary, created_at FROM ingested_files WHERE file_hash = ?"
    rows = await sqlite_db.fetchall(query, (file_hash,))
    if rows:
        row = rows[0]
        # SQLite DATETIME is retrieved as string by default, Pydantic parses it automatically
        return IngestedFile(
            id=row["id"],
            file_name=row["file_name"],
            file_hash=row["file_hash"],
            summary=row["summary"],
            created_at=row["created_at"]
        )
    return None
