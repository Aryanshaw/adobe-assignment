from contextlib import asynccontextmanager

from fastapi import FastAPI

import app as project_root
from dotenv import load_dotenv

load_dotenv()

from app.logger import logger
from app.db.qdrant import qdrant_db
from app.db.redis import redis_db
from app.db.bm25 import bm25_db
from app.db.sqlite import sqlite_db
from app.db.migrate import run_migrations
from app.api.router import router as ingest_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    project_root.init()
    await qdrant_db.connect()
    await redis_db.connect()
    await bm25_db.connect()
    await sqlite_db.connect()
    logger.info("connections initialized")
    
    await run_migrations()
    
    init_routes(app)
    yield
    await qdrant_db.disconnect()
    await redis_db.disconnect()
    await bm25_db.disconnect()
    await sqlite_db.disconnect()
    logger.info("connections disconnected")


app = FastAPI(lifespan=lifespan)


def init_routes(app_instance: FastAPI) -> None:
    app_instance.include_router(ingest_router)


if __name__ == "__main__":
    project_root.init()
