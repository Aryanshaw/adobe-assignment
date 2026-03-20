import pandas as pd
import sqlite3
import os
import json
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from qdrant_client.models import Filter, FieldCondition, MatchValue

from app.logger import logger
from app.db.sqlite import sqlite_db
from app.db.qdrant import qdrant_db
from app.agent.prompts import STRUCTURED_AGENT_PROMPT
from openai import AsyncOpenAI

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# structured_llm = ChatAnthropic(
#     model="claude-sonnet-4-6",
#     temperature=0.2,
#     api_key=os.getenv("ANTHROPIC_API_KEY"),
#     base_url="https://api.anthropic.com",
#     verbose=True
# )

structured_llm = ChatOpenAI(
    model="gpt-5.2",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
    verbose=True
)

@tool
async def search_structured_knowledge(query: str) -> str:
    """
    Searches structured data knowledge base to find relevant tables
    and columns for a given data requirement.

    Returns table schema and relevant column descriptions
    needed to write an accurate SQL query.

    Always call this FIRST before writing any SQL.

    Args:
        query: the data requirement — what information is needed
    """
    resp = await openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=[query]
    )
    embedding = resp.data[0].embedding

    collection_name = os.getenv("STRUCTURED_QDRANT_COLLECTION_NAME", "structured_knowledge")

    custom_filter = Filter(
        must=[FieldCondition(
            key="doc_type",
            match=MatchValue(value="structured")
        )]
    )

    logger.info(f"Searching structured Qdrant for: '{query[:50]}...'")
    results = await qdrant_db.search(
        collection_name=collection_name,
        query_vector=embedding,
        query_filter=custom_filter,
        limit=15
    )
    logger.info(f"Structured search found {len(results)} matches")

    if not results:
        return "<structured_context>No structured data found relevant to this query.</structured_context>"

    # group by table, skip file-level summary chunks (no column_name)
    tables_found = {}
    for r in results:
        table = r.payload.get("points_to_sqlite")
        col   = r.payload.get("column_name", "")
        text  = r.payload.get("text", "")
        if not table or not col:
            continue
        if table not in tables_found:
            tables_found[table] = []
        tables_found[table].append({
            "column":      col,
            "description": text
        })

    if not tables_found:
        return "<structured_context>No structured data found relevant to this query.</structured_context>"

    conn = sqlite3.connect(sqlite_db.db_path)
    conn.row_factory = sqlite3.Row

    table_blocks = []
    for table_name, columns in tables_found.items():

        # fetch schema
        row = conn.execute(
            "SELECT raw_schema FROM schema_registry WHERE table_name = ?",
            (table_name,)
        ).fetchone()
        schema_str = row["raw_schema"] if row else "Schema not found."

        # strip verbose prefix from description — keep only the meaning
        # descriptions are stored as:
        # "Column 'x' in table 'y' (file.csv): <actual meaning here>"
        # we only want the part after the colon
        col_tags = []
        for col_info in columns:
            raw_desc = col_info["description"]
            meaning  = raw_desc.split(":", 1)[-1].strip() if ":" in raw_desc else raw_desc
            col_tags.append(
                f'      <column name="{col_info["column"]}">{meaning}</column>'
            )

        cols_block  = "\n".join(col_tags)
        table_block = (
            f'  <table name="{table_name}">\n'
            f'    <schema>\n{schema_str}\n    </schema>\n'
            f'    <relevant_columns>\n{cols_block}\n    </relevant_columns>\n'
            f'  </table>'
        )
        table_blocks.append(table_block)

    conn.close()

    output = "<structured_context>\n" + "\n\n".join(table_blocks) + "\n</structured_context>"
    logger.info(f"Returning structured context for {len(table_blocks)} table(s)")
    return output

@tool
def execute_sql(sql: str) -> str:
    """
    Executes a SQLite query and returns results as a markdown table.
    If the query fails, returns the error message for retry.

    Args:
        sql: A valid SQLite SELECT query
    """
    logger.info(f"Executing SQL: {sql}")
    try:
        # We need a sync connection for pandas
        conn = sqlite3.connect(sqlite_db.db_path)
        df = pd.read_sql_query(sql, conn)
        conn.close()

        if df.empty:
            return "Query executed successfully but returned 0 rows."

        # Cap output to avoid token overload
        if len(df) > 50:
            result = df.head(50).to_markdown(index=False)
            result += f"\n[Note: showing 50 of {len(df)} rows]"
        else:
            result = df.to_markdown(index=False)

        return f"SQL RESULT:\n{result}\n\nSQL USED: {sql}"

    except Exception as e:
        return f"SQL_ERROR: {str(e)}\nSQL_ATTEMPTED: {sql}"

@tool
async def validate_and_retry_sql(original_sql: str, error_message: str) -> str:
    """
    Called when execute_sql returns an error.
    Attempts to fix the SQL using the error message as guidance.
    Only call this once — do not loop.

    Args:
        original_sql: the SQL that failed
        error_message: the error returned by execute_sql
    """
    fix_prompt = f"""
    This SQL failed: {original_sql}
    Error: {error_message}
    Fix the SQL and return ONLY the corrected SQL query. Do not use markdown wrappers.
    """
    corrected = await structured_llm.ainvoke(fix_prompt)
    return f"CORRECTED_SQL: {corrected.content.strip()}"

# Sub-agent setup
tools = [search_structured_knowledge, execute_sql, validate_and_retry_sql]

structured_agent_executor = create_react_agent(
    model=structured_llm,
    tools=tools,
    prompt=STRUCTURED_AGENT_PROMPT
)
