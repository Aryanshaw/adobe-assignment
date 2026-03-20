import os
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, VectorParams
from app.logger import logger

class QdrantConnection:
    def __init__(self):
        self.client = None

    async def connect(self) -> None:
        api_key = os.getenv("QDRANT_API_KEY")
        url = os.getenv("QDRANT_CLUSTER_ENDPOINT")
        
        if not url or not api_key:
            logger.warning("Qdrant credentials not found in env.")
            
        self.client = AsyncQdrantClient(url=url, api_key=api_key, timeout=30.0)
        logger.info("Connected to Async Qdrant Cloud")

    async def _ensure_collection(self, collection_name: str) -> None:
        if not self.client:
            return
        if not await self.client.collection_exists(collection_name):
            logger.info(f"Collection '{collection_name}' not found. Creating it now...")
            await self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
            )

    async def _ensure_payload_indexes(self, collection_name: str) -> None:
        if not self.client:
            return
        try:
            await self.client.create_payload_index(
                collection_name=collection_name,
                field_name="doc_type",
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception as exc:
            msg = str(exc).lower()
            if "already exists" not in msg and "duplicate" not in msg:
                raise

    async def upsert(self, collection_name: str, points: list) -> None:
        """Create or Update points in a collection"""
        if self.client:
            await self._ensure_collection(collection_name)
            await self._ensure_payload_indexes(collection_name)
            try:
                await self.client.upsert(collection_name=collection_name, points=points)
            except Exception as e:
                if "vector name" in str(e).lower() or "400" in str(e):
                    logger.warning(f"Vector config mismatch in '{collection_name}'. Recreating collection...")
                    await self.client.delete_collection(collection_name)
                    await self._ensure_collection(collection_name)
                    await self._ensure_payload_indexes(collection_name)
                    await self.client.upsert(collection_name=collection_name, points=points)
                else:
                    raise
            
    async def search(self, collection_name: str, query_vector: list, query_filter=None, limit: int = 20):
        """Read/Search points in a collection"""
        if self.client:
            try:
                response = await self.client.query_points(
                    collection_name=collection_name,
                    query=query_vector,
                    query_filter=query_filter,
                    limit=limit
                )
            except Exception as exc:
                if query_filter and "index required but not found" in str(exc).lower():
                    logger.warning(
                        f"Missing payload index detected for '{collection_name}'. Creating indexes and retrying search."
                    )
                    await self._ensure_payload_indexes(collection_name)
                    response = await self.client.query_points(
                        collection_name=collection_name,
                        query=query_vector,
                        query_filter=query_filter,
                        limit=limit
                    )
                else:
                    raise
            return response.points
        return []
        
    async def delete(self, collection_name: str, points_selector) -> None:
        """Delete points from a collection"""
        if self.client:
            await self.client.delete(collection_name=collection_name, points_selector=points_selector)

    async def disconnect(self) -> None:
        if self.client:
            await self.client.close()
            logger.info("Disconnected from Async Qdrant Cloud")
            self.client = None

qdrant_db = QdrantConnection()
