import os
import sqlite3
import aiosqlite
from typing import Any, List, Tuple
from app.logger import logger

class SQLiteConnection:
    def __init__(self, db_path: str = "local.db"):
        self.db_path = db_path
        self.conn = None

    async def connect(self) -> None:
        self.conn = await aiosqlite.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        logger.info(f"Connected to Local SQLite (aiosqlite) at {self.db_path}")

    async def execute(self, query: str, parameters: tuple = ()) -> None:
        if not self.conn:
            logger.warning("SQLite not connected. Cannot execute query.")
            return
            
        await self.conn.execute(query, parameters)
        await self.conn.commit()

    async def fetchall(self, query: str, parameters: tuple = ()) -> List[sqlite3.Row]:
        if not self.conn:
            logger.warning("SQLite not connected. Cannot fetch data.")
            return []
            
        async with self.conn.execute(query, parameters) as cursor:
            return await cursor.fetchall()

    async def disconnect(self) -> None:
        if self.conn:
            await self.conn.close()
            self.conn = None
            logger.info("Disconnected from Local SQLite")

sqlite_db = SQLiteConnection()
