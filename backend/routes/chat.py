from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

from backend.database import get_client
from backend.executor import execute_query
from backend.query_service import convert_nl_to_mongodb


router = APIRouter()


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str
    query: Optional[Dict[str, Any]] = None
    results: Optional[List[Dict[str, Any]]] = None
    count: int = 0
    confidence: Optional[float] = None
    warnings: Optional[List[str]] = None
    preview: Optional[bool] = None


def _reply(text: str, **kwargs) -> ChatResponse:
    return ChatResponse(reply=text, **kwargs)


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    try:
        # Parse natural language → MongoDB query
        parsed = convert_nl_to_mongodb(message)

        if parsed is None:
            return _reply("I couldn't understand that query. Could you rephrase it?", count=0)

        confidence = parsed.get("confidence", 0)
        warnings = parsed.get("warnings", [])

        # Low confidence warning
        if confidence < 0.3:
            return _reply(
                "I'm not confident I understood your query correctly. "
                "Could you try rephrasing it with more specific field or collection names?",
                query=parsed,
                confidence=confidence,
                warnings=warnings,
                count=0,
            )

        # Execute against MongoDB
        client = get_client()
        if client is None:
            return _reply(
                "Database connection is not available. "
                "I can show you the generated query, but can't fetch results.",
                query=parsed,
                confidence=confidence,
                warnings=warnings,
                count=0,
            )

        exec_result = execute_query(client, parsed)

        # Write operation preview
        if exec_result.get("preview"):
            return _reply(
                exec_result.get("message", "Write operation blocked."),
                query=parsed,
                results=[exec_result.get("generated_query", {})],
                count=0,
                confidence=confidence,
                warnings=warnings,
                preview=True,
            )

        results = exec_result.get("results", [])
        count = exec_result.get("count", 0)

        if count == 0:
            reply = "No results found for your query."
        elif count == 1:
            reply = "Found 1 result."
        else:
            reply = f"Found {count} results."

        return _reply(
            reply,
            query=parsed,
            results=results,
            count=count,
            confidence=confidence,
            warnings=warnings,
        )

    except Exception as e:
        return _reply(f"Something went wrong: {str(e)}", count=0)
