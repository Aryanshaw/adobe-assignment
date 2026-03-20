import os
import hashlib
from pathlib import Path
from groq import AsyncGroq
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
