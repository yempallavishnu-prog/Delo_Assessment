import logging
import re
import sqlite3
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from secrets import token_hex
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langgraph.checkpoint.sqlite import SqliteSaver
from ollama import ResponseError
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.config import Settings
from app.ingestion import document_path, load_corpus
from app.retrieval import HybridRetriever
from app.workflow import build_workflow

log = logging.getLogger(__name__)
ROOT = Path(__file__).parent


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str = Field(min_length=1, max_length=2000)
    new_session: bool = False

    @field_validator("message")
    @classmethod
    def meaningful_message(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("Enter a question.")
        return value


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    # Serialize requests to the local model.
    model_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app):
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
        )
        for name in ("httpx", "httpx2", "httpcore", "httpcore2"):
            logging.getLogger(name).setLevel(logging.WARNING)
        app.state.graph = None
        app.state.index_error = None
        app.state.documents = []
        app.state.retriever = None
        settings.session_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(settings.session_path, check_same_thread=False)
        try:
            app.state.documents = load_corpus(settings)
            try:
                retriever = HybridRetriever(settings)
                app.state.retriever = retriever
                app.state.graph = build_workflow(retriever, settings, SqliteSaver(connection))
            except (ValueError, OSError, RuntimeError) as exc:
                log.error("index_unavailable error_type=%s", type(exc).__name__)
                app.state.index_error = (
                    "Search is unavailable. Verify the PDFs and rebuild the index."
                )
            yield
        finally:
            connection.close()

    app = FastAPI(title="Policy RAG", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "testserver"]
    )
    app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
    templates = Jinja2Templates(directory=ROOT / "templates")

    @app.middleware("http")
    async def request_boundary(request, call_next):
        if request.method == "POST":
            origin = request.headers.get("origin")
            if origin and urlparse(origin).netloc != request.headers.get("host"):
                return JSONResponse(
                    {"detail": "Cross-origin requests are not accepted."}, status_code=403
                )
            body = bytearray()
            async for block in request.stream():
                body.extend(block)
                if len(body) > 8192:
                    return JSONResponse({"detail": "Request is too large."}, status_code=413)
            request._body = bytes(body)
        try:
            response = await call_next(request)
        except Exception:
            log.exception("unexpected_request_failure path=%s", request.url.path)
            response = JSONResponse(
                {"detail": "The request failed. Check the local server log."}, status_code=500
            )
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def index(request: Request):
        response = templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "documents": app.state.documents,
                "countries": sorted({d.country for d in app.state.documents}),
                "years": sorted({d.policy_year for d in app.state.documents}),
                "ready": app.state.graph is not None,
            },
        )
        if not re.fullmatch(r"[a-f0-9]{48}", request.cookies.get("policy_session", "")):
            response.set_cookie("policy_session", token_hex(24), httponly=True, samesite="strict")
        return response

    @app.get("/health")
    def health():
        retriever = app.state.retriever
        return {
            "status": "ok",
            "index_available": retriever is not None,
            "documents": len(app.state.documents),
            "chunks": retriever.manifest["chunks"] if retriever else 0,
            "detail": app.state.index_error,
        }

    @app.get("/documents/{document_id}")
    def source_document(document_id: str):
        document = next((d for d in app.state.documents if d.document_id == document_id), None)
        if document is None:
            return JSONResponse({"detail": "Document not found."}, status_code=404)
        path = document_path(settings, document)
        if not path.is_file():
            return JSONResponse(
                {"detail": "The source PDF is not available locally."}, status_code=404
            )
        return FileResponse(
            path,
            media_type="application/pdf",
            content_disposition_type="inline",
            filename=document.filename,
        )

    @app.post("/api/chat")
    def chat(payload: ChatRequest, request: Request):
        if app.state.graph is None:
            return JSONResponse({"detail": app.state.index_error}, status_code=503)
        if not model_lock.acquire(blocking=False):
            return JSONResponse(
                {"detail": "The local model is answering another question. Try again shortly."},
                status_code=503,
            )
        session = request.cookies.get("policy_session", "")
        if payload.new_session or not re.fullmatch(r"[a-f0-9]{48}", session):
            session = token_hex(24)
        started = time.perf_counter()
        try:
            result = app.state.graph.invoke(
                {"message": payload.message}, {"configurable": {"thread_id": session}}
            )
            response = JSONResponse(
                {
                    key: result.get(key)
                    for key in (
                        "status",
                        "answer",
                        "countries",
                        "requested_years",
                        "scopes",
                        "citations",
                        "retrieved_evidence",
                        "diagnostics",
                    )
                }
            )
            response.set_cookie("policy_session", session, httponly=True, samesite="strict")
            log.info(
                "chat session=%s scopes=%s status=%s elapsed_ms=%.1f",
                session[:8],
                result["scopes"],
                result["status"],
                (time.perf_counter() - started) * 1000,
            )
            return response
        except (ConnectionError, httpx.ConnectError):
            return JSONResponse(
                {"detail": "Ollama is unavailable. Start it with ollama serve."}, status_code=503
            )
        except ResponseError as exc:
            detail = (
                f"The Ollama model is missing. Run ollama pull {settings.llm_model}."
                if exc.status_code == 404
                else "Ollama could not complete this answer. Check its local log."
            )
            return JSONResponse({"detail": detail}, status_code=503)
        except httpx.TimeoutException:
            return JSONResponse(
                {"detail": "The local model timed out. Try a shorter question."}, status_code=504
            )
        except ValueError:
            log.exception("invalid_workflow_result session=%s", session[:8])
            return JSONResponse(
                {"detail": "The evidence could not be validated. Try a narrower question."},
                status_code=422,
            )
        finally:
            model_lock.release()

    return app


app = create_app()
