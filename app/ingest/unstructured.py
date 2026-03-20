import re
import os
import uuid
import asyncio
from pathlib import Path
from typing import Any

from app.logger import logger
from app.db.qdrant import qdrant_db
from app.db.bm25 import bm25_db
from openai import AsyncOpenAI
import tiktoken
from docling.document_converter import DocumentConverter
from qdrant_client.models import PointStruct
from app.ingest.helpers import compute_file_hash, generate_document_summary
from app.models.ingestion import get_ingested_file_by_hash, create_ingested_file
import traceback

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize tiktoken encoding
encoding = tiktoken.get_encoding("cl100k_base")

DOC_TYPE_PATTERNS: dict[str, list[str]] = {
    "financial": [r"annual\.report", r"quarterly", r"q[1-4]", r"revenue", r"earnings", r"balance\.sheet", r"p&l", r"ebitda"],
    "strategy":  [r"strategy", r"roadmap", r"okr", r"vision", r"plan"],
    "operations":[r"ops", r"operational", r"sla", r"incident"],
    "risk":      [r"risk", r"compliance", r"audit", r"control"],
    "hr":        [r"hr", r"headcount", r"hiring", r"attrition"],
}

DATE_PATTERN = re.compile(r"(q[1-4][_\-]?\d{4}|fy\d{4}|\d{4}[_\-]q[1-4])", re.IGNORECASE)


def _tag_chunk(filename: str, folder: str, heading: str) -> dict[str, Any]:
    """Generates rule-based metadata for a chunk."""
    try:
        combined = f"{filename} {folder} {heading}".lower()
        
        doc_type = "general"
        for type_name, patterns in DOC_TYPE_PATTERNS.items():
            if any(re.search(p, combined) for p in patterns):
                doc_type = type_name
                break
                
        date_match = DATE_PATTERN.search(combined)
        dept = Path(folder).name if folder and folder != "." else "unknown"
        
        return {
            "doc_type": doc_type,
            "date": date_match.group(0) if date_match else None,
            "department": dept,
            "source_file": filename,
        }
    except Exception as e:
        logger.error(f"Error in _tag_chunk for {filename}: {e}")
        return {
            "doc_type": "general",
            "date": None,
            "department": "unknown",
            "source_file": filename,
        }

def _semantic_chunk(text: str, max_tokens: int = 512, overlap_ratio: float = 0.1) -> list[str]:
    """Basic semantic chunker: splits by sentences, bounded by token count."""
    try:
        def token_count(s: str) -> int:
            return len(encoding.encode(s))

        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for sent in sentences:
            sent_tokens = token_count(sent)
            if current_tokens + sent_tokens >= max_tokens and current:
                chunks.append(" ".join(current))
                overlap_budget = int(max_tokens * overlap_ratio)
                tail, tail_tokens = [], 0
                for s in reversed(current):
                    if tail_tokens + token_count(s) <= overlap_budget:
                        tail.insert(0, s)
                        tail_tokens += token_count(s)
                    else:
                        break
                current, current_tokens = tail, tail_tokens
                
            current.append(sent)
            current_tokens += sent_tokens

        if current:
            chunks.append(" ".join(current))

        return [c for c in chunks if len(c.strip()) > 50]
    except Exception as e:
        logger.error(f"Error in _semantic_chunk: {e}")
        return []


async def _parse_and_chunk(filepath: Path) -> list[dict[str, Any]]:
    """Runs Docling and extracts block-aware chunks (tables vs prose)."""
    
    def process():
        try:
            converter = DocumentConverter()
            doc = converter.convert(str(filepath)).document
            chunks = []
            
            # We iterate over items (Docling v2.x architecture)
            # Fallback to export_to_markdown if the structure is not easily accessible
            current_section_header = "General"
            try:
                for item, _ in doc.iterate_items():
                    item_label = getattr(item, "label", type(item).__name__).lower()
                    
                    # Track headings to provide context to chunks
                    if "header" in item_label and hasattr(item, "text"):
                        current_section_header = item.text.strip()
                        
                    text = ""
                    if "table" in item_label and hasattr(item, "export_to_markdown"):
                        try:
                            text = item.export_to_markdown(doc=doc)
                        except Exception:
                            text = getattr(item, "text", "")
                    else:
                        text = getattr(item, "text", "")
                        
                    if not text.strip():
                        continue
                        
                    meta = _tag_chunk(filepath.name, str(filepath.parent), current_section_header)
                    meta["page"] = getattr(item, "page_no", None)
                    
                    if "table" in item_label:
                        chunks.append({"id": str(uuid.uuid4()), "text": text, "chunk_type": "table", **meta})
                    else:
                        for prose_chunk in _semantic_chunk(text):
                            chunks.append({"id": str(uuid.uuid4()), "text": prose_chunk, "chunk_type": "prose", **meta})
            except Exception as inner_e:
                logger.warning(f"Structured iteration failed for {filepath.name}, falling back to markdown regex. Error: {inner_e}")
                # Fallback regex parsing if iteration fails
                markdown = doc.export_to_markdown()
                for chunk in _semantic_chunk(markdown, max_tokens=1000):
                    chunks.append({"id": str(uuid.uuid4()), "text": chunk, "chunk_type": "mixed", **_tag_chunk(filepath.name, str(filepath.parent), "")})
                    
            return chunks
        except Exception as e:
            logger.error(f"Error in _parse_and_chunk processing for {filepath.name}: {e}")
            return []

    return await asyncio.to_thread(process)


async def ingest_unstructured_file(filepath: Path) -> int:
    """End-to-end ingestion for a single PDF/DOCX file."""
    try:
        # Check if file is already ingested in the database
        file_hash = compute_file_hash(filepath)
        existing = await get_ingested_file_by_hash(file_hash)
        if existing:
            logger.info(f"File '{filepath.name}' is already ingested (Hash: {file_hash}). Skipping duplicate processing.")
            return 0
            
        logger.info(f"Ingesting unstructured file: {filepath.name} using Docling")
        
        # Parse and chunk the file
        chunks = await _parse_and_chunk(filepath)
        if not chunks:
            logger.warning(f"No text extracted from {filepath.name}")
            return 0
            
        logger.info(f"Extracted {len(chunks)} chunks.")
        
        # Batch Embed using OpenAI
        texts = [c["text"] for c in chunks]
        embeddings = []
        
        logger.info("Generating embeddings for chunks")

        # Process Embeddings asynchronously
        async def get_embeddings():
            batch_size = 100
            result = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                response = await openai_client.embeddings.create(model="text-embedding-3-small", input=batch)
                result.extend([r.embedding for r in response.data])
            return result
            
        embeddings = await get_embeddings()
        
        logger.info("Embeddings generated successfully. Storing in Qdrant")

        # Store in Qdrant using the existing DB wrapper in smaller batches to avoid HTTP timeouts
        points = [
            PointStruct(id=c["id"], vector=embeddings[i], payload={k: v for k, v in c.items() if k != "id"})
            for i, c in enumerate(chunks)
        ]
        
        collection_name = os.getenv("UNSTRUCTURED_QDRANT_COLLECTION_NAME", "leadership_docs")
        
        # Store in Qdrant using the existing DB wrapper in smaller batches to avoid HTTP timeouts
        batch_size = 50
        for i in range(0, len(points), batch_size):
            await qdrant_db.upsert(collection_name=collection_name, points=points[i : i + batch_size])
        
        # Update BM25 index with new text chunks for hybrid search
        await bm25_db.add_documents(chunks)
        
        logger.info("Qdrant and BM25 ingestion completed. Generating summary")
        
        # Generate summary using Groq Llama 3 (Enhanced Sampling)
        total_chunks = len(chunks)
        summary_chunks = []
        if total_chunks <= 15:
            summary_chunks = chunks
        else:
            # first 5 chunks + middle 5 chunks + last 5 chunks (This ensures key financial data or conclusions at the end of long reports are captured in the summary.)
            summary_chunks = chunks[:5] + chunks[total_chunks//2 - 2 : total_chunks//2 + 3] + chunks[-5:]
            
        full_text_for_summary = " ".join([c["text"] for c in summary_chunks])
        summary = await generate_document_summary(full_text_for_summary)
        logger.info(f"Generated summary for {filepath.name}: {summary}")

        # Register file in SQLite
        row_id = await create_ingested_file(filepath.name, file_hash, summary)
        
        logger.info(f"Successfully tracked and ingested {filepath.name} (ID: {row_id}) into '{collection_name}'.")
        return len(chunks)
    except Exception as e:
        import traceback
        logger.error(f"Error in ingest_unstructured_file for {filepath.name}:\n{traceback.format_exc()}")
        return 0
