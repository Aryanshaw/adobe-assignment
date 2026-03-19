import os
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams
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

    async def upsert(self, collection_name: str, points: list) -> None:
        """Create or Update points in a collection"""
        if self.client:
            if not await self.client.collection_exists(collection_name):
                logger.info(f"Collection '{collection_name}' not found. Creating it now...")
                await self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
                )
            try:
                await self.client.upsert(collection_name=collection_name, points=points)
            except Exception as e:
                if "vector name" in str(e).lower() or "400" in str(e):
                    logger.warning(f"Vector config mismatch in '{collection_name}'. Recreating collection...")
                    await self.client.delete_collection(collection_name)
                    await self.client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
                    )
                    await self.client.upsert(collection_name=collection_name, points=points)
                else:
                    raise
            
    async def search(self, collection_name: str, query_vector: list, query_filter=None, limit: int = 20):
        """Read/Search points in a collection"""
        if self.client:
            return await self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit
            )
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
