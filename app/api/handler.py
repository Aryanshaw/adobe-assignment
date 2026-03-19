import shutil
from pathlib import Path
from fastapi import UploadFile
from typing import Dict, Any

from app.logger import logger
from app.ingest.unstructured import ingest_unstructured_file

UNSTRUCTURED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".pptx"}

async def handle_file_upload(file: UploadFile) -> Dict[str, Any]:
    """
    Detects the file type and triggers the appropriate ingestion pipeline.
    Saves the UploadFile to a temporary location on disk first.
    """
    if not file.filename:
        return {"error": "No filename provided."}

    ext = Path(file.filename).suffix.lower()
    
    # Save the uploaded file temporarily
    temp_dir = Path("/tmp/agent_uploads")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    temp_filepath = temp_dir / file.filename
    
    try:
        with temp_filepath.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        logger.info(f"Saved uploaded file to {temp_filepath}")

        # Route to appropriate pipeline
        if ext in UNSTRUCTURED_EXTENSIONS:
            chunks_processed = await ingest_unstructured_file(temp_filepath)
            return {
                "status": "success", 
                "message": f"Ingested unstructured file: {file.filename}", 
                "chunks": chunks_processed
            }
        else:
            return {"status": "skipped", "message": f"Unsupported or structured extension: {ext}"}
            
    except Exception as e:
        logger.error(f"Error handling file upload {file.filename}: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        # Cleanup temp file
        if temp_filepath.exists():
            temp_filepath.unlink()
