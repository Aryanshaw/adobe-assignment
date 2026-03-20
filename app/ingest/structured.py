import re
import os
import json
import uuid
import sqlite3
import asyncio
import pandas as pd
from pathlib import Path
from groq import AsyncGroq
from openai import AsyncOpenAI

from app.logger import logger
from app.db.sqlite import sqlite_db
from app.db.qdrant import qdrant_db
from app.ingest.helpers import compute_file_hash
from app.models.ingestion import get_ingested_file_by_hash, create_ingested_file, write_schema_registry
from qdrant_client.models import PointStruct

groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

STRUCTURED_EXTENSIONS = {'.csv', '.xlsx', '.xls'}

async def load_structured_file(filepath: Path) -> pd.DataFrame:
    if filepath.suffix.lower() not in STRUCTURED_EXTENSIONS:
        raise ValueError(f'Unsupported type: {filepath.suffix}')
    if filepath.stat().st_size == 0:
        raise ValueError(f'Empty file: {filepath.name}')

    def do_load():
        if filepath.suffix.lower() == '.csv':
            try:
                return pd.read_csv(filepath, encoding='utf-8')
            except UnicodeDecodeError:
                return pd.read_csv(filepath, encoding='latin-1')
        else:
            return pd.read_excel(filepath, sheet_name=0)
            
    try:
        df = await asyncio.to_thread(do_load)
    except Exception as e:
        raise RuntimeError(f'Failed to parse {filepath.name}: {e}')

    if df.empty or len(df.columns) == 0:
        raise ValueError(f'No data found in {filepath.name}')

    # Handle completely empty columns
    df.dropna(axis=1, how='all', inplace=True)

    logger.info(f'Loaded {filepath.name}: {len(df)} rows, {len(df.columns)} cols')
    return df

def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = []
    seen = {}
    for i, col in enumerate(df.columns):
        if pd.isna(col) or str(col).strip() == '' or str(col).startswith('Unnamed:'):
            clean = f'col_{i}'
        else:
            clean = str(col).strip().lower()
            clean = re.sub(r'[^\w]', '_', clean)
            clean = re.sub(r'_+', '_', clean).strip('_')

        if clean in seen:
            seen[clean] += 1
            clean = f'{clean}_{seen[clean]}'
        else:
            seen[clean] = 0

        new_cols.append(clean)
    df.columns = new_cols
    return df

async def store_dataframe_to_sqlite(df: pd.DataFrame, table_name: str) -> str:
    """Writes DataFrame to SQLite using a sync thread. Returns schema string."""
    def do_store():
        # pandas requires a synchronous SQLite connection for df.to_sql
        conn = sqlite3.connect(sqlite_db.db_path)
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        schema_sql = pd.io.sql.get_schema(df, table_name)
        
        text_cols = df.select_dtypes(include='object').columns.tolist()
        if text_cols:
            conn.execute(f'CREATE INDEX IF NOT EXISTS idx_{table_name}_{text_cols[0]} ON {table_name}({text_cols[0]})')
        conn.commit()
        conn.close()
        return schema_sql
        
    return await asyncio.to_thread(do_store)

async def generate_structured_file_summary(filepath: Path, df: pd.DataFrame) -> str:
    sample_rows = df.head(5).to_markdown(index=False)
    prompt = f"""You are a business data analyst. Examine this dataset and describe it.

Filename: {filepath.name}
Columns ({len(df.columns)} total): {list(df.columns)}
Row count: {len(df)}
First 5 rows:
{sample_rows}

Return exactly 2 sentences:
Sentence 1: What business data this file contains (what is being measured).
Sentence 2: The likely business context (department, time period, purpose).
No preamble, no explanation. Output only the 2 sentences."""

    try:
        resp = await groq_client.chat.completions.create(
            model='llama-3.3-70b-versatile',
            messages=[
                {'role': 'system', 'content': 'You are a business data analyst. Be concise.'},
                {'role': 'user', 'content': prompt}
            ],
            temperature=0.1,
            max_tokens=150
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f'File summary failed for {filepath.name}: {e}')
        return f'Dataset from {filepath.name} containing {len(df.columns)} columns: {list(df.columns[:5])}... with {len(df)} rows.'

def get_column_samples(df: pd.DataFrame, sample_rows: int = 3) -> str:
    lines = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        samples = df[col].dropna().head(sample_rows).tolist()
        null_count = df[col].isna().sum()
        lines.append(f'  {col} (dtype: {dtype}, nulls: {null_count}): {samples}')
    return '\n'.join(lines)

async def generate_column_descriptions(df: pd.DataFrame, file_summary: str, filename: str) -> dict[str, str]:
    col_samples = get_column_samples(df)
    prompt = f"""You are a data analyst describing a business dataset for a SQL query system.
File context: {file_summary}

For each column listed below, write ONE sentence that describes:
  - What this column measures or represents
  - The unit or format (e.g. USD thousands, percentage, count, date)
  - Any important business meaning (e.g. target vs actual, YTD vs quarterly)

Columns with sample values:
{col_samples}

Return ONLY a valid JSON object. No markdown fences. No explanation.
Format:
{{
  "{df.columns[0]}": "Two sentence description.",
  "other_col": "Two sentence description."
}}"""

    try:
        resp = await groq_client.chat.completions.create(
            model='llama-3.3-70b-versatile',
            messages=[
                {'role': 'system', 'content': 'Return only valid JSON. No markdown, no preamble.'},
                {'role': 'user', 'content': prompt}
            ],
            temperature=0.1,
            max_tokens=1500
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r'^```json\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        
        col_descriptions = json.loads(raw)
        
        for col in df.columns:
            if col not in col_descriptions:
                col_descriptions[col] = f'Column {col} — description unavailable.'
        
        return col_descriptions
    except Exception as e:
        logger.error(f'Column description fallback due to error: {e}')
        return {col: f'{col} ({df[col].dtype})' for col in df.columns}

def build_structured_chunks(filename: str, table_name: str, file_summary: str, col_descriptions: dict, columns: list) -> list[dict]:
    chunks = []
    chunks.append({
        'id': str(uuid.uuid4()),
        'text': f'File {filename}: {file_summary} Table name in database: {table_name}. Columns: {list(columns)}.',
        'chunk_type': 'file_summary',
        'points_to_sqlite': table_name,
        'source_file': filename,
        'doc_type': 'structured'
    })
    for col in columns:
        desc = col_descriptions.get(col, '')
        chunks.append({
            'id': str(uuid.uuid4()),
            'text': f"Column '{col}' in table '{table_name}' ({filename}): {desc}",
            'chunk_type': 'column_description',
            'points_to_sqlite': table_name,
            'column_name': col,
            'source_file': filename,
            'doc_type': 'structured'
        })
    return chunks

async def ingest_structured_file(filepath: Path) -> dict:
    result = {'file': filepath.name, 'status': 'failed', 'rows': 0, 'table': ''}
    try:
        df = await load_structured_file(filepath)
        
        file_hash = compute_file_hash(filepath)
        existing = await get_ingested_file_by_hash(file_hash)
        if existing:
            logger.info(f"File '{filepath.name}' already ingested (hash match). Skipping.")
            result['status'] = 'skipped_duplicate'
            return result
            
        df = normalise_columns(df)
        table_name = filepath.stem.lower()
        table_name = re.sub(r'[^\w]', '_', table_name)
        table_name = re.sub(r'_+', '_', table_name).strip('_')
        
        logger.info(f"Enriching structured file {filepath.name} via Groq LLM...")
        file_summary = await generate_structured_file_summary(filepath, df)
        logger.info(f"File summary for {filepath.name}: {file_summary}")
        col_descriptions = await generate_column_descriptions(df, file_summary, filepath.name)
        
        logger.info(f"Saving data to SQLite table '{table_name}'...")
        raw_schema = await store_dataframe_to_sqlite(df, table_name)
        
        await write_schema_registry(
            table_name=table_name,
            source_file=filepath.name,
            file_hash=file_hash,
            file_summary=file_summary,
            raw_schema=raw_schema,
            col_descriptions=json.dumps(col_descriptions),
            col_names=json.dumps(list(df.columns)),
            row_count=len(df),
            sample_rows=df.head(3).to_json(orient='records')
        )
        
        await create_ingested_file(
            file_name=filepath.name,
            file_hash=file_hash,
            file_type="structured",
            table_name=table_name,
            row_count=len(df),
            col_count=len(df.columns),
            summary=file_summary
        )
        
        logger.info(f"Generating knowledge embeddings for structured file...")
        chunks_data = build_structured_chunks(filepath.name, table_name, file_summary, col_descriptions, df.columns.tolist())
        
        texts = [c['text'] for c in chunks_data]
        resp = await openai_client.embeddings.create(model='text-embedding-3-small', input=texts)
        embeddings = [r.embedding for r in resp.data]
        
        points = [
            PointStruct(id=c['id'], vector=embeddings[i], payload={k: v for k, v in c.items() if k != 'id'})
            for i, c in enumerate(chunks_data)
        ]
        
        await qdrant_db.upsert(collection_name=os.getenv("STRUCTURED_QDRANT_COLLECTION_NAME", "structured_knowledge"), points=points)
        logger.info(f'Stored {len(points)} metadata chunks in structured_knowledge collection')
        
        result.update({'status': 'success', 'rows': len(df), 'table': table_name})
        return result

    except Exception as e:
        import traceback
        logger.error(f"Error ingesting structured file {filepath.name}:\n{traceback.format_exc()}")
        return result
