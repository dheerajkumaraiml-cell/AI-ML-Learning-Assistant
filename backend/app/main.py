
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.services.model_service import ModelService


@asynccontextmanager
async def lifespan(app: FastAPI):
    service = ModelService()
    service.load()
    app.state.model_service = service
    yield
    app.state.model_service = None


app = FastAPI(
    title="AI/ML Learning Assistant API",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)


@app.get("/health")
def health(request: Request):
    return {
        "status": "ok",
        "model_loaded": getattr(
            request.app.state, "model_service", None
        ) is not None,
    }


@app.post("/api/chat")
def chat(request: Request, body: ChatRequest):
    if not body.message.strip():
        raise HTTPException(
            status_code=422,
            detail="Message cannot be empty or whitespace only.",
        )

    service = request.app.state.model_service

    try:
        reply = service.generate(body.message.strip())
        return {"reply": reply}
    except Exception:
        # Do not expose internal model or filesystem details to clients.
        raise HTTPException(
            status_code=500,
            detail="Answer generation failed.",
        )
