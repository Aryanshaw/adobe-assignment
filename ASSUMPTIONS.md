# Assumptions & Design Architecture

This document outlines the key assumptions made while building the AI Leadership Insight & Decision Agent, along with the reasoning behind major architectural choices.

---

## 1. Document Corpus

- The system is designed to ingest company documents in PDF, DOCX, TXT, and MD formats, as well as structured data in CSV and XLSX formats.
- Documents are assumed to be internal company materials — annual reports, quarterly reports, strategy notes, and operational updates — as described in the assignment brief.
- No real company data was available during development. Testing was done using publicly available annual reports (e.g. ASX-listed company filings) and synthetic CSV datasets.
- Images, charts, and figures embedded in PDFs are skipped during ingestion. The system is optimised for text and tabular content, which is where leadership insights live.

---

## 2. Retrieval Architecture

- The assignment asked for an agent that answers questions grounded in documents. A RAG (Retrieval-Augmented Generation) pipeline was chosen as the foundation because it grounds every answer in actual document content rather than model memory.
- **Hybrid search** (dense vector + BM25 keyword) was implemented instead of dense-only retrieval. The reasoning: leadership questions often contain exact financial terms, codes, and dates (e.g. "EBITDA Q3 FY2024") that semantic search alone may miss. BM25 catches exact matches; dense search catches meaning. RRF fusion combines both.
- **Cohere rerank-v3.5** is used as a cross-encoder reranker instead of a locally hosted model. This avoids any GPU or CPU load on the host machine while providing superior reranking quality over embedding similarity alone.
- Two separate Qdrant collections are maintained: `unstructured_knowledge` for unstructured content and `structured_knowledge` for column-level descriptions of structured files. Keeping them separate prevents column description chunks from polluting prose retrieval results.

---

## 3. Structured Data Handling

- CSV and XLSX files are treated differently from documents. They are loaded into an in-process SQLite database rather than chunked and embedded, because chunking tabular rows loses the column header context and makes numerical reasoning unreliable.
- **SQLite** was chosen over a full data-warehouse (Clickhouse, Snowflake) because this is a PoC and requires zero infrastructure, ships with Python, and is sufficient for the document-scale data expected in this use case. The design allows swapping to Clickhouse by changing a new connection.
- Column names in real financial exports are often opaque (e.g. `rev_act_q3`, `hc_fte`, `var_pct`). A **Groq LLM call at ingestion time** generates human-readable descriptions for every column. These descriptions are stored in Qdrant and injected into SQL generation prompts at query time, so the model always understands what each column means before writing SQL.
- SQL generation uses **schema injection** — the exact CREATE TABLE schema and column descriptions are passed to the LLM — to prevent hallucinated column names, which is the primary failure mode in text-to-SQL systems.

---

## 4. Agent Architecture

- A **two-layer agent design** was chosen: a master agent (Openai 5.2) that orchestrates retrieval, and a structured sub-agent (OpenAi 5.2/Claude Sonnet 4.5) that handles SQL generation. The unstructured retrieval path is a direct tool function with no sub-agent, since it is a fixed pipeline with no branching logic.
- The master agent uses **ReAct** via LangChain's `create_react_agent`. Tool selection replaces a separate intent classifier — the agent's LLM reasoning step decides which tool to call based on the question and the available data summaries injected into the system prompt.
- **No conversation memory** is maintained between requests. Leadership Q&A questions are mostly independent, and stateless handling simplifies the system significantly for a round 1 submission.
- **LangCache** has been introduced for semantic query caching , if a similar question is asked again it will return the cached response.
- The master agent's system prompt includes a live summary of all ingested files (from the `ingested_files` SQLite table). This gives the agent awareness of what data exists before calling any tool, preventing speculative tool calls for data that was never ingested.

---

## 5. LLM & Model Choices

| Component | Model | Reason |
|---|---|---|
| Master agent | Openai 5.2 | Best reasoning for orchestration and final answer synthesis |
| Structured sub-agent | Openai 5.2 | Accurate SQL generation with faster response time |
| Ingestion enrichment | Groq llama-3.3-70b | Free tier, used only at ingestion time not query time |
| Embeddings | OpenAI text-embedding-3-small | Best cost-to-quality ratio for retrieval; ~$0.02/1M tokens |
| Reranking | Cohere rerank-v3.5 | No local model load; ~$0.001/call |

---

## 6. Cost & Latency Assumptions

- The system is designed for a leadership team making roughly 50–100 queries per day. At this scale the total running cost is estimated at under $10/month.
- Query latency target is under 3 seconds for the unstructured path and under 2 seconds for the structured path. The structured path is faster because it skips embedding and reranking.
- Groq's free tier (14,400 requests/day on llama-3.3-70b) is sufficient for ingestion enrichment at any reasonable document corpus size.

---

## 7. Document Deduplication

- Every ingested file is SHA-256 hashed before processing. If the same file is re-uploaded (identical content, different filename), it is detected as a duplicate and skipped. This prevents redundant Groq calls, duplicate Qdrant entries, and duplicate SQLite tables.
- File hash and ingestion metadata are stored in a unified `ingested_files` SQLite table that tracks both structured and unstructured files, distinguished by a `file_type` column.

---

## 8. What Was Not Built (Explicit Scope Decisions)

- **No frontend UI** — the system exposes a FastAPI REST API. A UI was not in scope for round 1.
- **No authentication or multi-tenancy** — all users share the same document corpus. Adding per-user namespacing in Qdrant is a straightforward future extension.
- **No streaming UI** — the API supports SSE streaming but a streaming frontend was not implemented.
- **Single XLSX sheet** — only the first sheet of an Excel file is ingested. Multi-sheet support is a planned extension.

---

## 9. Open Questions & Known Limitations

- Very large PDFs (100+ pages with heavy image content) take longer to ingest due to Docling's parsing overhead. OCR is disabled to mitigate this — text-layer PDFs parse quickly, scanned image-only PDFs will have limited extraction.
- BM25 index is rebuilt on every new document ingestion. For very large corpora this adds a few seconds to ingestion time but does not affect query performance.
- The SQL retry mechanism makes one correction attempt on failure. Edge cases with highly ambiguous column mappings may still produce incorrect SQL on the second attempt.