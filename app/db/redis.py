import os
import redis.asyncio as redis
import json
from typing import Optional, Any
from app.logger import logger

class RedisConnection:
    def __init__(self):
        self.client = None
        self.is_connected = False

    async def connect(self) -> None:
        url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.client = redis.Redis.from_url(url, decode_responses=True)
        # Test the connection to ensure it's alive
        try:
            await self.client.ping()
            self.is_connected = True
            logger.info("Connected to Async Local Redis")
        except redis.ConnectionError as e:
            self.is_connected = False
            logger.warning(f"Could not connect to Local Redis at {url}. Error: {e}")

    async def set(self, key: str, value: Any, expire_seconds: Optional[int] = None) -> bool:
        """Create or Update a key-value pair"""
        if not self.is_connected:
            return False
        if isinstance(value, (dict, list)):
            value = json.dumps(value)
        return await self.client.set(key, value, ex=expire_seconds)

    async def get(self, key: str) -> Optional[str]:
        """Read a value by key"""
        if self.is_connected:
            return await self.client.get(key)
        return None

    async def delete(self, *names: str) -> bool:
        """Delete one or more keys"""
        if self.is_connected:
            return (await self.client.delete(*names)) > 0
        return False

    async def hset(self, name: str, key: str, value: Any) -> bool:
        """Create or Update a field in a hash"""
        if not self.is_connected:
            return False
        if isinstance(value, (dict, list)):
            value = json.dumps(value)
        return (await self.client.hset(name, key, value)) == 1

    async def hget(self, name: str, key: str) -> Optional[str]:
        """Read a field from a hash"""
        if self.is_connected:
            return await self.client.hget(name, key)
        return None
        
    async def keys(self, pattern: str = "*") -> list:
        """Read all keys matching a pattern"""
        if self.is_connected:
            return await self.client.keys(pattern)
        return []

    async def disconnect(self) -> None:
        if self.client:
            try:
                await self.client.aclose()
            except AttributeError:
                await self.client.close()
            self.is_connected = False
            logger.info("Disconnected from Async Local Redis")
            self.client = None

redis_db = RedisConnection()
