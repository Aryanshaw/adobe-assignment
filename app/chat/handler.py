import os
from typing import Optional

from langcache import LangCache

from app.agent.master_agent import get_master_agent_executor
from app.logger import logger


# ---------------------------------------------------------------------------
# LangCache — initialized once at module load, skipped if key is not set
# ---------------------------------------------------------------------------
_langcache: Optional[LangCache] = None
_LANGCACHE_API_KEY = os.getenv("LANGCACHE_API_KEY")
_LANGCACHE_SERVER_URL = os.getenv("LANGCACHE_SERVER_URL")
_LANGCACHE_CACHE_ID = os.getenv("LANGCACHE_CACHE_ID")
_ENABLE_CACHING = os.getenv("ENABLE_RESPONSE_CACHING")

if _ENABLE_CACHING=="True" and _LANGCACHE_API_KEY and _LANGCACHE_SERVER_URL and _LANGCACHE_CACHE_ID:
    _langcache = LangCache(
        server_url=_LANGCACHE_SERVER_URL,
        api_key=_LANGCACHE_API_KEY,
        cache_id=_LANGCACHE_CACHE_ID,
    )
    logger.info(f"LangCache semantic cache enabled ({_LANGCACHE_SERVER_URL})")
else:
    missing = [k for k, v in {
        "ENABLE_RESPONSE_CACHING": _ENABLE_CACHING,
        "LANGCACHE_API_KEY": _LANGCACHE_API_KEY,
        "LANGCACHE_SERVER_URL": _LANGCACHE_SERVER_URL,
        "LANGCACHE_CACHE_ID": _LANGCACHE_CACHE_ID,
    }.items() if not v]
    logger.warning(f"LangCache disabled — missing env vars: {', '.join(missing)}")


# ---------------------------------------------------------------------------
# Chat handler
# ---------------------------------------------------------------------------
async def handle_chat(question: str, session_id: str) -> str:
    try:
        # Step 1 — Semantic cache lookup
        if _langcache:
            try:
                cache_result = await _langcache.search_async(
                    prompt=question,
                    similarity_threshold=0.90,
                )
                if cache_result and cache_result.data:
                    hit = cache_result.data[0]
                    logger.info(
                        f"LangCache HIT (similarity={hit.similarity:.2f})"
                        f" for session {session_id}"
                    )
                    return hit.response
            except Exception as cache_err:
                logger.warning(f"LangCache search failed, falling through to LLM: {cache_err}")

        # Step 2 — LLM invocation (cache miss or cache disabled)
        executor = await get_master_agent_executor()
        response = await executor.ainvoke({"messages": [("user", question)]})
        answer = response["messages"][-1].content

        # Step 3 — Persist answer into LangCache for future hits
        if _langcache and answer:
            try:
                await _langcache.set_async(prompt=question, response=answer)
            except Exception as cache_err:
                logger.warning(f"LangCache set failed: {cache_err}")

        return answer

    except Exception as e:
        logger.error(f"Chat error for session {session_id}: {e}")
        return f"An error occurred while processing your request: {e}"
