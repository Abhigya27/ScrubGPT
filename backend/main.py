"""FastAPI app + all route definitions.

Run it as ONE process on ONE instance (`uvicorn backend.main:app --host 0.0.0.0 --port $PORT`): job status, the
ingestion queue, chat history and the rate limiters live in this process's memory.
"""
import asyncio
import json
import logging
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import asdict

import requests
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from backend import config, guardrails, jobs, library
from backend.index import qdrant_store
from backend.index.embed import get_sparse
from backend.ingest import transcript as transcript_provider
from backend.ingest.url_parser import parse_url
from backend.models import ChatTurn
from backend.rag import answer, history
from backend.rag.enhance import enhance_query
from backend.rag.retrieve import group_into_periods, public_view, retrieve_chunks
from backend.util import QuotaExhausted, UserFacingError, set_notifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)


def _warm_up() -> None:
    """Verify the Qdrant collection and load the embedding models, in the background.

    Startup must not wait for this: it needs a model download on a cold host and a network round trip to Qdrant, and
    a service that does not bind its port quickly is killed by the platform. If it fails the API still starts and
    /health reports the problem; the first real request retries (qdrant_store.ensure_ready is lazy).
    """
    try:
        qdrant_store.ensure_ready()
        get_sparse()
        log.info("warm-up finished: Qdrant collection verified, embedding models loaded")
    except Exception as exc:  # noqa: BLE001
        log.error(
            "Warm-up failed (%s: %s). Couldn't reach Qdrant Cloud? A free cluster idle for a week is suspended: "
            "resume it in the Qdrant Cloud console (after four idle weeks it is deleted). "
            "Also check QDRANT_URL and QDRANT_API_KEY. The API is up; /health shows the state.",
            type(exc).__name__, exc,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    for note in config.config_warnings():
        log.warning("config: %s", note)  # a missing OPTIONAL setting is a note, never a crash
    threading.Thread(target=_warm_up, name="warm-up", daemon=True).start()
    yield


app = FastAPI(title="YT-Link RAG", lifespan=lifespan)


class IngestRequest(BaseModel):
    url: str


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    question: str


# --- ingest ------------------------------------------------------------------------
@app.post("/ingest")
async def ingest(req: IngestRequest, request: Request):
    """Returns at once. Looking up the playlist, enforcing the limits and indexing all happen in a background job
    (one at a time per backend instance), whose live progress is at GET /ingest/status/{job_id}.

      422  not a YouTube video/playlist link
      409  a DIFFERENT source is being indexed right now
      429  too many submissions
      200  a job id: either a new job, or (already_running=true) the job that is already indexing this source
    """
    try:
        kind, ident = parse_url(req.url)
    except UserFacingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if jobs.active_job() is None:  # attaching to a running job, or being turned away as busy, costs no rate-limit token
        guardrails.check_ingest_rate(request)
    try:
        job, attached = jobs.submit_ingest(kind, ident)
    except jobs.IngestBusy as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"job_id": job.job_id, "source_id": ident, "already_running": attached}


@app.get("/library")
async def get_library():
    """Small local library view; Qdrant remains the persistent chunk source of truth."""
    return {"sources": library.list_sources()}


@app.get("/ingest/status/{job_id}")
async def ingest_status(job_id: str):
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Unknown job id. The backend restarted, so the previous job status was lost. "
                   "Submit the same URL again; already indexed videos will be reused.",
        )
    return asdict(job)


# --- chat ----------------------------------------------------------------------------
async def _chat_events(req: ChatRequest, question: str, mode: str):
    """enhance (follow-ups + related terms) -> retrieve -> merge into sections -> (answer). Yields a progress event as each step starts
    and finishes, then either a "result" or an "error" event. `mode` is "search" or "ask"."""
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def stage(name: str, message: str) -> None:
        queue.put_nowait({"event": "stage", "stage": name, "message": message})

    def notice_from_thread(message: str) -> None:  # LLM rate-limit waits happen in worker threads
        loop.call_soon_threadsafe(queue.put_nowait, {"event": "stage", "stage": "waiting", "message": message})

    def fail(status: int, detail: str, retry_after: int | None = None) -> None:
        queue.put_nowait({"event": "error", "status": status, "detail": detail, "retry_after": retry_after})

    async def work() -> None:
        set_notifier(notice_from_thread)
        try:
            prior = history.get_history(req.session_id)
            if prior:
                stage("condense", "Reading the conversation so far to understand your question...")
            elif config.QUERY_ENHANCE_ENABLED:
                stage("enhance", "Expanding your question into related search terms...")
            # One LangChain call: resolves follow-ups from the history AND proposes rewrites/keywords. It never
            # fails the request: with the model busy or unavailable the question is simply searched as typed.
            enhanced = await asyncio.to_thread(enhance_query, prior, question)
            standalone = enhanced.standalone
            if prior and standalone != question:
                stage("condense", f"Understood as: \"{standalone}\"")
            if enhanced.expansions:
                stage("enhance", "Also searching for: " + "; ".join(f"\"{e}\"" for e in enhanced.expansions))
            elif enhanced.note:
                stage("enhance", enhanced.note)

            stage("retrieve", "Searching the index: keyword and meaning-based matching...")
            chunks = await asyncio.to_thread(retrieve_chunks, standalone, req.source_id, enhanced.expansions)
            if chunks:
                stage("retrieve", f"{len(chunks)} passage{'s' if len(chunks) != 1 else ''} passed the relevance threshold.")
            else:
                stage("retrieve", "No passage passed the relevance threshold.")

            periods = group_into_periods(chunks)  # in playing order
            if periods:
                labels = ", ".join(p["label"] for p in periods[:5]) + (" ..." if len(periods) > 5 else "")
                stage("group", f"Merged into {len(periods)} section{'s' if len(periods) != 1 else ''}, in playing order: {labels}")

            if mode == "search":
                results = [public_view(p) for p in periods]
                history.add_turn(
                    req.session_id,
                    ChatTurn(question=question, standalone_question=standalone, answer=None, citations=results),
                )
                queue.put_nowait({"event": "result", "data": {"results": results, "confident": bool(results)}})
                return

            if periods:
                stage("generate", f"Writing an answer from {len(periods)} section{'s' if len(periods) != 1 else ''}...")
            else:
                stage("generate", "Nothing to answer from, so the language model isn't called.")
            out = await asyncio.to_thread(answer.generate_answer, standalone, periods, prior)
            out["citations"] = [public_view(c) for c in out["citations"]]
            if out["grounded"]:
                stage("generate", f"Answer ready, citing {len(out['citations'])} section{'s' if len(out['citations']) != 1 else ''}.")
            history.add_turn(
                req.session_id,
                ChatTurn(question=question, standalone_question=standalone, answer=out["answer"], citations=out["citations"]),
            )
            queue.put_nowait({"event": "result", "data": out})  # {"answer", "citations", "grounded"}
        except QuotaExhausted as exc:
            fail(503, str(exc), exc.retry_after)
        except UserFacingError as exc:
            fail(503, str(exc))
        except Exception:
            log.exception("chat pipeline failed")
            fail(502, "The language model request failed. Please try again." if mode == "ask" else "Search failed. Please try again.")
        finally:
            queue.put_nowait(None)  # end of stream

    task = asyncio.create_task(work())
    try:
        while True:
            event = await queue.get()
            if event is None:
                break
            yield event
    finally:
        if not task.done():
            task.cancel()  # the client went away


async def _collect(req: ChatRequest, mode: str) -> dict:
    """Non-streaming endpoints: run the same pipeline and return only the final result."""
    question = guardrails.check_question(req.question)
    result = None
    async for event in _chat_events(req, question, mode):
        if event["event"] == "error":
            raise HTTPException(status_code=event["status"], detail=event["detail"])
        if event["event"] == "result":
            result = event["data"]
    return result


async def _ndjson(events):
    async for event in events:
        yield json.dumps(event, ensure_ascii=False) + "\n"


def _stream(req: ChatRequest, mode: str) -> StreamingResponse:
    question = guardrails.check_question(req.question)  # validate BEFORE streaming so a bad request gets a real 422
    return StreamingResponse(
        _ndjson(_chat_events(req, question, mode)),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/search", dependencies=[Depends(guardrails.rate_limit_search)])
async def search(req: ChatRequest):
    """Retrieval-only: no LLM beyond question condensation. Returns time periods in playing order."""
    return await _collect(req, "search")  # {"results", "confident"}


@app.post("/search/stream", dependencies=[Depends(guardrails.rate_limit_search)])
async def search_stream(req: ChatRequest):
    """Same as /search, streamed as newline-delimited JSON: stage events, then a result (or error) event."""
    return _stream(req, "search")


@app.post("/ask", dependencies=[Depends(guardrails.rate_limit_ask)])
async def ask(req: ChatRequest):
    """Generated answer on top of retrieval, behind the three grounding guards in rag/answer.py."""
    return await _collect(req, "ask")  # {"answer", "citations", "grounded"}


@app.post("/ask/stream", dependencies=[Depends(guardrails.rate_limit_ask)])
async def ask_stream(req: ChatRequest):
    """Same as /ask, streamed as newline-delimited JSON."""
    return _stream(req, "ask")


# --- health ----------------------------------------------------------------------------------------
_qdrant_probe = {"at": -1e9, "ok": False}
_probe_lock = threading.Lock()


def _qdrant_reachable() -> bool:
    """One quick GET /collections (3 s timeout). Cached, so a load balancer polling /health does not become traffic
    to Qdrant: 30 s when healthy, 5 s when not (so recovery shows up quickly). Needs no embedding model."""
    if not (config.QDRANT_URL and config.QDRANT_API_KEY):
        return False
    now = time.monotonic()
    with _probe_lock:
        age, ok = now - _qdrant_probe["at"], _qdrant_probe["ok"]
        if age < (30 if ok else 5):
            return ok
    try:
        response = requests.get(
            f"{config.QDRANT_URL.rstrip('/')}/collections", headers={"api-key": config.QDRANT_API_KEY}, timeout=3
        )
        ok = response.status_code == 200
    except requests.RequestException:
        ok = False
    with _probe_lock:
        _qdrant_probe.update(at=time.monotonic(), ok=ok)
    return ok


@app.get("/healthz")
async def healthz():
    """Liveness only: the process is up and serving. Use this as the platform's health-check path, so a Qdrant blip
    can never make a host restart the service (which would also wipe the in-memory job state)."""
    return {"status": "ok"}


@app.get("/health")
async def health():
    """Readiness detail. Lightweight by design:

      - NO YouTube or transcript-provider request (asking YouTube whether it likes us is exactly the traffic that
        gets a host rate-limited; the route status below is read from in-process state);
      - no embedding-model load, no LLM call;
      - one cached 3-second Qdrant ping. Qdrant down -> 503 "unhealthy" quickly.

    A missing optional proxy never makes this unhealthy. No LLM key -> "degraded" (English-only indexing and
    retrieval can still work until an answer/translation call is needed).
    """
    qdrant_ok = await asyncio.to_thread(_qdrant_reachable)
    llm_ok = config.LLM_PROVIDER is not None
    status = "unhealthy" if not qdrant_ok else ("ok" if llm_ok else "degraded")
    body = {
        "status": status,
        "qdrant": qdrant_ok,
        "llm": llm_ok,
        "transcript_provider": ",".join(config.effective_provider_order()),
        "generated_transcripts_allowed": config.ALLOW_GENERATED_TRANSCRIPT,
        "local_mode": config.LOCAL_MODE,
        **transcript_provider.youtube_route_status(),
        "model": config.LLM_MODEL,
        "embed_model": config.EMBED_MODEL,
        "config_warnings": config.config_warnings(),
    }
    return JSONResponse(status_code=503 if status == "unhealthy" else 200, content=body)
