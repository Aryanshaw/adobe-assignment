import uuid
from typing import Optional
from datetime import datetime
from pydantic import BaseModel
from app.db.sqlite import sqlite_db
from app.logger import logger

class IngestedFile(BaseModel):
    id: str  # Kept as UUID string
    file_name: str
    file_hash: str
    file_type: str
    table_name: str
    row_count: int
    col_count: int
    summary: str
    created_at: Optional[datetime] = None

async def create_ingested_files_table():
    # We update the schema to accommodate both structured and unstructured
    query = """
    CREATE TABLE IF NOT EXISTS ingested_files (
        id TEXT PRIMARY KEY,
        file_name TEXT NOT NULL,
        file_hash TEXT NOT NULL UNIQUE,
        file_type TEXT NOT NULL,
        table_name TEXT,
        row_count INTEGER,
        col_count INTEGER,
        summary TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """
    await sqlite_db.execute(query)
    # create index
    await sqlite_db.execute("CREATE INDEX IF NOT EXISTS idx_struct_hash ON ingested_files(file_hash)")
    logger.info("ingested_files table verified/created.")

async def create_ingested_file(
    file_name: str, 
    file_hash: str, 
    summary: str, 
    file_type: str = "unstructured",
    table_name: str = "",
    row_count: int = 0,
    col_count: int = 0
) -> str:
    file_id = str(uuid.uuid4())
    query = """
    INSERT INTO ingested_files (id, file_name, file_hash, file_type, table_name, row_count, col_count, summary)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    await sqlite_db.execute(query, (file_id, file_name, file_hash, file_type, table_name, row_count, col_count, summary))
    return file_id

async def get_ingested_file_by_hash(file_hash: str) -> Optional[IngestedFile]:
    query = "SELECT * FROM ingested_files WHERE file_hash = ?"
    rows = await sqlite_db.fetchall(query, (file_hash,))
    if rows:
        row = rows[0]
        return IngestedFile(
            id=row["id"],
            file_name=row["file_name"],
            file_hash=row["file_hash"],
            file_type=row["file_type"],
            table_name=row["table_name"] if row["table_name"] else "",
            row_count=row["row_count"] if row["row_count"] else 0,
            col_count=row["col_count"] if row["col_count"] else 0,
            summary=row["summary"],
            created_at=row["created_at"]
        )
    return None

async def create_schema_registry_table():
    query = """
    CREATE TABLE IF NOT EXISTS schema_registry (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        table_name       TEXT    NOT NULL UNIQUE,
        source_file      TEXT    NOT NULL,
        file_hash        TEXT    NOT NULL,
        file_summary     TEXT,            
        raw_schema       TEXT    NOT NULL, 
        col_descriptions TEXT    NOT NULL, 
        col_names        TEXT    NOT NULL, 
        row_count        INTEGER NOT NULL,
        sample_rows      TEXT    NOT NULL, 
        created_at       TEXT    NOT NULL DEFAULT (datetime('now'))
    )
    """
    await sqlite_db.execute(query)
    await sqlite_db.execute("CREATE INDEX IF NOT EXISTS idx_registry_table ON schema_registry(table_name)")
    await sqlite_db.execute("CREATE INDEX IF NOT EXISTS idx_registry_hash ON schema_registry(file_hash)")
    logger.info("schema_registry table verified/created.")

async def write_schema_registry(
    table_name: str,
    source_file: str,
    file_hash: str,
    file_summary: str,
    raw_schema: str,
    col_descriptions: str,
    col_names: str,
    row_count: int,
    sample_rows: str
):
    query = """
    INSERT OR REPLACE INTO schema_registry
    (table_name, source_file, file_hash, file_summary, raw_schema, col_descriptions, col_names, row_count, sample_rows)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    await sqlite_db.execute(query, (
        table_name, source_file, file_hash, file_summary,
        raw_schema, col_descriptions, col_names, row_count, sample_rows
    ))
