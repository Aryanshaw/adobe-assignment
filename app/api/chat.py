from fastapi import APIRouter, Body
from app.chat.handler import handle_chat
from typing import Optional
import uuid

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("")
async def chat_endpoint(
    question: str = Body(..., embed=True),
    session_id: Optional[str] = Body(None, embed=True)
):
    if not session_id:
        session_id = str(uuid.uuid4())
        
    answer = await handle_chat(question, session_id)
    return {"session_id": session_id, "answer": answer}
