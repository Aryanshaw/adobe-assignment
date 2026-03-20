import markupsafe  # noqa: F401 — prevents Python 3.13 multiprocessing deadlock on macOS
from contextlib import asynccontextmanager

import app as project_root
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI

from langchain_core.globals import set_verbose, set_debug
from app.logger import logger
from app.db.qdrant import qdrant_db
from app.db.redis import redis_db
from app.db.bm25 import bm25_db
from app.db.sqlite import sqlite_db
from app.db.migrate import run_migrations
from app.api.router import router as ingest_router
from app.api.chat import router as chat_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    project_root.init()
    
    # Enable verbose logging for agents
    set_verbose(False)
    set_debug(False)
    logger.info("Global LangChain Verbose/Debug modes enabled")

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
    app_instance.include_router(chat_router)


if __name__ == "__main__":
    project_root.init()
