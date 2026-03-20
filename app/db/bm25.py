import pickle
import os
import asyncio
from app.logger import logger

class BM25Connection:
    def __init__(self, index_path="bm25_index.pkl", corpus_path="bm25_corpus.pkl"):
        self.index_path = index_path
        self.corpus_path = corpus_path
        self.corpus_path = corpus_path
        self.index = None
        self.corpus = [] # List of dictionaries containing "text", "source_file", etc.

    def _normalize_corpus_item(self, item, idx: int) -> dict:
        if isinstance(item, dict):
            normalized = dict(item)
            if "source" not in normalized:
                normalized["source"] = normalized.get("source_file", "unknown")
            normalized.setdefault("text", "")
            normalized.setdefault("page", None)
            normalized.setdefault("id", f"bm25_{idx}")
            return normalized
        if isinstance(item, str):
            return {
                "id": f"legacy_bm25_{idx}",
                "text": item,
                "source": "legacy_bm25",
                "source_file": "legacy_bm25",
                "page": None,
            }
        return {
            "id": f"bm25_{idx}",
            "text": str(item),
            "source": "unknown",
            "source_file": "unknown",
            "page": None,
        }

    async def connect(self) -> None:
        if os.path.exists(self.index_path) and os.path.exists(self.corpus_path):
            def load_data():
                with open(self.index_path, "rb") as f:
                    idx = pickle.load(f)
                with open(self.corpus_path, "rb") as f:
                    corp = pickle.load(f)
                return idx, corp
                
            self.index, self.corpus = await asyncio.to_thread(load_data)
            self.corpus = [
                self._normalize_corpus_item(item, idx)
                for idx, item in enumerate(self.corpus)
            ]
            logger.info(f"Loaded BM25 index and corpus ({len(self.corpus)} docs)")
        else:
            logger.info("BM25 index/corpus not found. Will be created during first ingestion.")

    async def add_documents(self, new_chunks: list[dict]) -> None:
        """Append new documents (dictionaries) to the corpus and rebuild the index."""
        from rank_bm25 import BM25Okapi
        
        def update():
            # Basic tokenization: lowercase and split by non-word characters
            import re
            def tokenize(text):
                return re.findall(r"\w+", str(text).lower())

            self.corpus.extend(new_chunks)
            tokenized_corpus = [tokenize(doc.get("text", "")) for doc in self.corpus]
            
            self.index = BM25Okapi(tokenized_corpus)
            
            with open(self.index_path, "wb") as f:
                pickle.dump(self.index, f)
            with open(self.corpus_path, "wb") as f:
                pickle.dump(self.corpus, f)
                
        await asyncio.to_thread(update)
        logger.info(f"Updated BM25 index with {len(new_chunks)} new documents. Total: {len(self.corpus)}")

    async def build_index(self, tokenized_corpus: list[list[str]]) -> None:
        """Legacy method for completeness."""
        from rank_bm25 import BM25Okapi
        def do_build():
            self.index = BM25Okapi(tokenized_corpus)
            with open(self.index_path, "wb") as f:
                pickle.dump(self.index, f)
        await asyncio.to_thread(do_build)

    async def get_scores(self, query_tokens: list[str]) -> list[float]:
        """Read: Get the relevance scores for all documents in the index."""
        def do_get_scores():
            if self.index:
                return self.index.get_scores(query_tokens)
            return []
        
        if self.index:
            return await asyncio.to_thread(do_get_scores)
        logger.warning("BM25 index is not loaded. Cannot get scores.")
        return []

    async def get_top_n(self, query_tokens: list[str], corpus: list, n: int = 5) -> list:
        """Read: Retrieve the top N most relevant documents."""
        if self.index:
            return self.index.get_top_n(query_tokens, corpus, n=n)
        logger.warning("BM25 index is not loaded. Cannot get top N.")
        return []

    async def disconnect(self) -> None:
        if self.index:
            self.index = None
            logger.info("Cleared BM25 index from memory")

bm25_db = BM25Connection()
