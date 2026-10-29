"""Single-process HTTP adapter for the existing Direct/Grounded workflows."""

import hashlib
import os
import shutil
import stat
import tempfile
from contextlib import ExitStack, asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pypdf.errors import PdfReadError
from starlette.concurrency import run_in_threadpool

from .answering import query_translation_enabled, run_answer
from .config import IndexConfig
from .embeddings import E5Embeddings
from .generation import make_llm
from .index import Index
from .ingestion import PDFValidationError
from .profiles import profile_config
from .workflow import WorkflowResult

MAX_PDF_BYTES = 50 * 1024 * 1024


class DocumentResponse(BaseModel):
    document: str
    pages: int
    chunks: int


class DocumentsResponse(BaseModel):
    documents: list[DocumentResponse]


class UploadResponse(BaseModel):
    status: Literal["indexed", "unchanged"]
    document: DocumentResponse


def document_info(name, record):
    # Do not serialize the manifest's private metadata or local storage paths.
    return DocumentResponse(document=name, pages=record["pages"], chunks=record["chunks"])


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


class CitationResponse(BaseModel):
    source_id: str
    document: str
    pdf_page: int = Field(ge=1, description="Physical PDF page, starting at 1; not a page label.")
    quote: str
    pdf_sha256: str | None = None


class ClaimResponse(BaseModel):
    text: str
    citations: list[CitationResponse]


class AskResponse(BaseModel):
    supported: bool
    text: str
    reason: str
    language: Literal["en", "de", "ru"]
    workflow: dict
    claims: list[ClaimResponse]


def structured_claims(run: WorkflowResult) -> list[ClaimResponse]:
    """Project only the final supported candidate; never publish rejected drafts."""
    candidate = run.candidate
    if (
        not run.answer.supported
        or candidate is None
        or not candidate.answer.supported
        or candidate.answer.text != run.answer.text
        or candidate.draft is None
        or run.context is None
    ):
        return []
    return [
        ClaimResponse(
            text=claim.text,
            citations=[
                CitationResponse(
                    source_id=evidence.source_id,
                    document=run.context.sources[evidence.source_id].metadata["document"],
                    pdf_page=run.context.sources[evidence.source_id].metadata["pdf_page"],
                    quote=evidence.quote,
                    pdf_sha256=run.context.sources[evidence.source_id].metadata.get("content_sha256"),
                )
                for evidence in claim.evidence
            ],
        )
        for claim in candidate.draft.claims
    ]


def create_app(
    *, index_dir: Path | None = None, config: IndexConfig | None = None,
    pdf_dir: Path | None = None,
):
    """Uvicorn factory; importing this module opens no storage or provider client."""
    busy = Lock()
    ready = False
    index = embeddings = llm = None
    pdf_root = None
    translate_queries = True

    @asynccontextmanager
    async def lifespan(app):
        nonlocal ready, index, embeddings, llm, pdf_root, translate_queries
        resources = ExitStack()
        try:
            # Match CLI privacy/cache settings; only read the explicit local .env.
            load_dotenv(Path.cwd() / ".env", override=False)
            translate_queries = query_translation_enabled()
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
            if device not in {"cpu", "mps"}:
                raise ValueError("Invalid device")
            index_id = hashlib.sha256(str(directory.resolve()).encode()).hexdigest()[:16]
            pdf_root = Path(pdf_dir) if pdf_dir is not None else (
                Path.home() / ".garage-rag" / "pdfs" / index_id
            )
            # Configuration/credentials are checked without making an LLM request.
            llm = make_llm()
            resources.callback(llm.client.__exit__, None, None, None)
            index = resources.enter_context(Index(directory, settings, allow_cross_thread=True))
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

    def require_ready():
        if not ready:
            raise HTTPException(503, "Server is not ready.")

    def busy_error():
        return HTTPException(503, "Server is busy.", headers={"Retry-After": "1"})

    @app.get("/documents", response_model=DocumentsResponse)
    async def documents():
        require_ready()
        # Index.ingest replaces this snapshot only after atomic revision activation.
        # Reading it never touches Qdrant or the embedding adapter.
        snapshot = index.manifest["documents"]
        return DocumentsResponse(documents=[
            document_info(name, record) for name, record in sorted(snapshot.items())
        ])

    @app.get("/documents/pdf")
    def original_pdf(document: str, sha256: str):
        require_ready()
        unavailable = HTTPException(404, "Original PDF unavailable for this citation.")
        if (
            not document.lower().endswith(".pdf")
            or any(c in "/\\" or ord(c) < 32 or ord(c) == 127 for c in document)
        ):
            raise unavailable
        record = index.manifest["documents"].get(document)
        if record is None or record["content_sha256"] != sha256:
            raise unavailable
        # Search only the API's private upload store, never manifest/source paths.
        # Existing uploads need no migration or second path registry. Read a bounded
        # snapshot and hash those exact bytes, avoiding FileResponse path races.
        for folder in pdf_root.glob("pdf-*"):
            if folder.is_symlink() or not folder.is_dir():
                continue
            path = folder / document
            try:
                fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                with os.fdopen(fd, "rb") as source:
                    if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                        continue
                    content = source.read(MAX_PDF_BYTES + 1)
            except OSError:
                continue
            if len(content) > MAX_PDF_BYTES or hashlib.sha256(content).hexdigest() != sha256:
                continue
            # Manifest activation is atomic. Recheck after reading in case an
            # upload replaced the active revision while the snapshot was read.
            active = index.manifest["documents"].get(document, {})
            if active.get("content_sha256") != sha256:
                raise unavailable
            return Response(content, media_type="application/pdf", headers={
                "Content-Disposition": "inline",
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
                "X-PDF-Pages": str(record["pages"]),
            })
        raise unavailable

    def index_upload(folder, path):
        # The worker owns the lock and file cleanup even if its HTTP client leaves.
        acquired = keep = False
        try:
            acquired = busy.acquire(blocking=False)
            if not acquired:
                raise busy_error()
            result = index.ingest(path, embeddings)
            response = UploadResponse(
                status=result["status"],
                document=document_info(path.name, index.manifest["documents"][path.name]),
            )
            keep = True
            return response
        except HTTPException:
            raise
        except (PdfReadError, PDFValidationError):
            raise HTTPException(
                422, "Could not index PDF. Use an unencrypted PDF with readable text; "
                "damaged files and scans without text are not supported.",
            ) from None
        except Exception:
            raise HTTPException(
                500, "PDF indexing failed. Check local storage and embeddings."
            ) from None
        finally:
            if acquired:
                busy.release()
            if not keep:
                shutil.rmtree(folder, ignore_errors=True)

    @app.post("/documents", response_model=UploadResponse)
    async def upload_document(request: Request, filename: str):
        require_ready()
        # Raw PDF body avoids multipart spooling before the size limit is enforced.
        if (
            not filename.lower().endswith(".pdf")
            or len(filename.encode("utf-8")) > 240
            or any(c in "/\\" or ord(c) < 32 or ord(c) == 127 for c in filename)
        ):
            raise HTTPException(400, "Choose a PDF filename without paths or control characters.")
        if request.headers.get("content-type", "").split(";")[0].lower() != "application/pdf":
            raise HTTPException(415, "Send a PDF with Content-Type: application/pdf.")
        length = request.headers.get("content-length")
        if length is not None:
            try:
                size = int(length)
                if size < 0:
                    raise ValueError
            except ValueError:
                raise HTTPException(400, "Invalid upload size.") from None
            if size > MAX_PDF_BYTES:
                raise HTTPException(413, "PDF exceeds the 50 MiB upload limit.")
        if busy.locked():
            raise busy_error()
        folder = None
        try:
            pdf_root.mkdir(mode=0o700, parents=True, exist_ok=True)
            pdf_root.chmod(0o700)
            folder = Path(tempfile.mkdtemp(prefix="pdf-", dir=pdf_root))
            path = folder / filename
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            size = 0
            with os.fdopen(descriptor, "wb") as output:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > MAX_PDF_BYTES:
                        raise HTTPException(413, "PDF exceeds the 50 MiB upload limit.")
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            with path.open("rb") as source:
                if source.read(5) != b"%PDF-":
                    raise HTTPException(422, "The file is empty or is not a PDF.")
        except BaseException as exc:
            if folder is not None:
                shutil.rmtree(folder, ignore_errors=True)
            if isinstance(exc, OSError):
                raise HTTPException(500, "Could not store PDF in private local storage.") from None
            raise
        return await run_in_threadpool(index_upload, folder, path)

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
                run = run_answer(
                    request.question,
                    lambda query, k: index.search(query, embeddings, k),
                    llm,
                    mode=request.mode,
                    answer_language=request.language,
                    translate_query=translate_queries,
                )
                result = run.public_dict()
                result["claims"] = structured_claims(run)
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
