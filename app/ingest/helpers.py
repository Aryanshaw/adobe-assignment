import os
import hashlib
from pathlib import Path
from groq import AsyncGroq
import pandas as pd
from app.logger import logger

groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

def compute_file_hash(filepath: Path) -> str:
    """SHA-256 of raw file bytes. Independent of filename."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()

async def generate_document_summary(text: str) -> str:
    """Generates a concise summary using Groq's Llama model, returning only the summary."""
    if not text.strip():
        return "No text available for summary."
    try:
        system_prompt = (
            "You are a highly analytical document summarizer. "
            "Your ONLY task is to return a 3-4 sentence summary of the provided text. "
            "Never use conversational filler, prefixes, or introductory phrases like 'Here is a summary'. "
            "Just output the raw summary text and nothing else."
        )
        
        summary_resp = await groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text[:15000]}
            ],
            temperature=0.1
        )
        return summary_resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Failed to generate summary: {e}")
        return "Summarization API call failed."

def markdown_to_df(markdown_str: str) -> pd.DataFrame:
    """
    Parses a Markdown table string into a pandas DataFrame.
    Assumes standard markdown format with | separators and a dashes row.
    Returns an empty DataFrame if parsing fails or table is invalid.
    """
    try:
        lines = [line.strip() for line in markdown_str.strip().split('\n') if line.strip()]
        if len(lines) < 3:
            return pd.DataFrame()
            
        # Extract headers (first row)
        headers = [col.strip() for col in lines[0].split('|')[1:-1]]
        
        # Verify the second line is a separator row (e.g. |---|---|)
        if not all(cell.strip(' -:') == '' for cell in lines[1].split('|')[1:-1]):
            return pd.DataFrame()
            
        # Extract data rows
        data = []
        for line in lines[2:]:
            cells = [cell.strip() for cell in line.split('|')[1:-1]]
            # Pad or truncate cells to match header length
            if len(cells) < len(headers):
                cells.extend([''] * (len(headers) - len(cells)))
            elif len(cells) > len(headers):
                cells = cells[:len(headers)]
            data.append(cells)
            
        df = pd.DataFrame(data, columns=headers)
        
        # Filter out "empty" columns/rows that might just contain hyphens or spaces
        df = df.replace(r'^[-\s]*$', '', regex=True)
        return df
    except Exception as e:
        logger.warning(f"Failed to parse markdown table to df: {e}")
        return pd.DataFrame()
