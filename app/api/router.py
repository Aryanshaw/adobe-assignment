from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
import asyncio
from app.api.handler import handle_file_upload
from app.logger import logger

router = APIRouter(prefix="/ingest", tags=["ingestion"])

@router.post("")
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Accepts a list of multipart file uploads and routes them to the ingestion pipeline.
    """
    logger.info(f"Received file upload request for {len(files)} files.")
    
    sem = asyncio.Semaphore(3)
    
    async def process_with_semaphore(f: UploadFile):
        async with sem:
            return await handle_file_upload(f)
            
    # Process all files concurrently, bounded to 3 active streams at a time
    results = await asyncio.gather(*[process_with_semaphore(f) for f in files])
        
    return {"status": "success", "results": list(results)}
