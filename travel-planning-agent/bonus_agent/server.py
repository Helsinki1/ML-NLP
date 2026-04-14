"""Thin FastAPI server exposing the travel agent as an OpenAI-compatible chat/completions endpoint.

This lets arksim (or any OpenAI-compatible client) talk to our LangGraph agent
via the standard chat_completions protocol.

Usage:
    python -m bonus_agent.server          # starts on port 8000
    OPENAI_API_KEY=... python -m bonus_agent.server
"""

import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from bonus_agent.graph import build_graph

# --- Models for the OpenAI-compatible endpoint ---

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "travel-agent"
    messages: list[Message]
    temperature: float = 0.0
    max_tokens: int = 2048


# --- App state ---

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_name = os.environ.get("TRAVEL_AGENT_MODEL", "gpt-4o-mini")
    _state["app_graph"] = build_graph(model_name=model_name, temperature=0.0)
    yield
    _state.clear()


app = FastAPI(title="Travel Planning Agent", lifespan=lifespan)


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    graph = _state["app_graph"]

    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    try:
        result = graph.invoke(
            {"messages": messages},
            config={"recursion_limit": 60},
        )
        all_messages = result["messages"]
        assistant_text = all_messages[-1].content if all_messages else ""
    except Exception as e:
        assistant_text = (
            "I'm sorry, I encountered an issue processing your request. "
            "Let me transfer you to a human agent for assistance. "
            f"(Internal error: {e})"
        )

    response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    return JSONResponse(content={
        "id": response_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "travel-agent",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": assistant_text,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    })


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "bonus_agent.server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
