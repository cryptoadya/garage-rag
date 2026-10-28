"""Single-process HTTP adapter for the existing Direct/Grounded workflows."""

import os
from contextlib import ExitStack, asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .config import IndexConfig
from .embeddings import E5Embeddings
from .generation import make_llm
from .index import Index
from .profiles import profile_config
from .workflow import run_direct, run_workflow


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    question: str = Field(min_length=1, max_length=4000)
    mode: Literal["direct", "grounded"] = "grounded"
    language: Literal["auto", "en", "de", "ru"] = "auto"

    @field_validator("question")
    @classmethod
    def nonblank_question(cls, value):
        if not value.strip():
            raise ValueError("Question must not be blank")
        return value  # Preserve the exact question sent by the CLI.


class AskResponse(BaseModel):
    supported: bool
    text: str
    reason: str
    language: Literal["en", "de", "ru"]
    workflow: dict


def create_app(*, index_dir: Path | None = None, config: IndexConfig | None = None):
    """Uvicorn factory; importing this module opens no storage or provider client."""
    busy = Lock()
    ready = False
    index = embeddings = llm = None

    @asynccontextmanager
    async def lifespan(app):
        nonlocal ready, index, embeddings, llm
        resources = ExitStack()
        try:
            # Match CLI privacy/cache settings; only read the explicit local .env.
            load_dotenv(Path.cwd() / ".env", override=False)
            os.environ["LANGSMITH_TRACING"] = "false"
            os.environ["LANGCHAIN_TRACING_V2"] = "false"
            os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
            os.environ.setdefault("HF_HOME", str(Path.cwd() / ".cache" / "huggingface"))
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            profile = os.environ.get("GARAGE_RAG_PROFILE", "baseline")
            settings = config if config is not None else profile_config(profile)
            default_dir = "data/qdrant" if profile == "baseline" else f"data/qdrant-{profile}"
            directory = Path(index_dir or os.environ.get("GARAGE_RAG_INDEX", default_dir))
            device = os.environ.get("GARAGE_RAG_DEVICE", "cpu")
            if device not in {"cpu", "mps"} or not (directory / "manifest.json").is_file():
                raise ValueError("Invalid device or missing index")
            # Configuration/credentials are checked without making an LLM request.
            llm = make_llm()
            resources.callback(llm.client.__exit__, None, None, None)
            index = resources.enter_context(Index(directory, settings))
            embeddings = E5Embeddings(settings, device)
        except Exception:
            try:
                resources.close()
            finally:
                raise RuntimeError(
                    "RAG startup failed; check provider configuration, existing index, "
                    "embedding profile and exclusive index access."
                ) from None
        ready = True
        try:
            yield
        finally:
            ready = False
            try:
                resources.close()
            except Exception:
                raise RuntimeError("RAG resource cleanup failed.") from None
            finally:
                index = embeddings = llm = None

    app = FastAPI(title="GarageRAG", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, exc):
        # Default validation responses echo inputs, including unexpected secret fields.
        return JSONResponse(status_code=422, content={"detail": "Invalid request."})

    @app.get("/health")
    def health():
        """Local server readiness; does not call an LLM or probe provider availability."""
        if not ready:
            raise HTTPException(503, "Server is not ready.")
        return {"status": "ok"}

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest):
        if not ready:
            raise HTTPException(503, "Server is not ready.")
        # A synchronous endpoint runs off the event loop. Reject overlap instead of
        # queueing unbounded provider work or sharing Qdrant/model operations.
        if not busy.acquire(blocking=False):
            raise HTTPException(503, "Server is busy.", headers={"Retry-After": "1"})
        try:
            try:
                runner = run_direct if request.mode == "direct" else run_workflow
                run = runner(
                    request.question,
                    lambda k: index.search(request.question, embeddings, k),
                    llm,
                    answer_language=request.language,
                )
                result = run.public_dict()
            except Exception:
                raise HTTPException(500, "Local operation failed.") from None
            if run.error:
                if run.answer.reason == "retrieval_error":
                    raise HTTPException(503, "Local retrieval failed.")
                raise HTTPException(502, "Answer generation failed.")
            return result
        finally:
            busy.release()

    return app
