"""The entire UI: one Streamlit page.

Sidebar : paste a link, index it (with live progress), pick the answer mode.
Centre  : a YouTube player (seeks to the moment you click) beside a scrollable chat box.
Both the indexing and every answer show what is happening behind the scenes, step by step.
"""
import json
import os
import time
import uuid

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# The FastAPI backend. Local development keeps both backend and Streamlit on the same machine.
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT = 120  # seconds of silence tolerated on a request/stream
POLL_SECONDS = 1.5
CHAT_HEIGHT = 540
MODE_SEARCH = "Timestamps only"
MODE_ASK = "Generated answer"
ACTIVE = ("queued", "processing")
SETTLED = ("done", "error", "paused")

STAGE_ICONS = {
    # indexing
    "queued": "⏳", "analyzing": "🔎", "measuring": "📏", "checking": "🗂️", "transcript": "📄", "chunking": "✂️",
    "translating": "🌐", "saving": "💾", "done": "✅", "paused": "⏸️", "rejected": "🚫", "failed": "❌",
    # answering
    "condense": "🧠", "enhance": "✨", "retrieve": "🔎", "group": "🧩", "generate": "✍️", "waiting": "⏳",
}
STAGE_TITLES = {
    "queued": "Waiting to start", "analyzing": "Analyzing the link", "measuring": "Checking the length",
    "checking": "Checking the index", "transcript": "Retrieving the transcript",
    "chunking": "Chunking the transcript",
    "translating": "Translating to English", "saving": "Embedding and saving", "done": "Done",
}

st.set_page_config(page_title="YT-Link RAG", page_icon="🎬", layout="wide")


# --- backend access ------------------------------------------------------------
def _detail(resp) -> str:
    try:
        detail = resp.json().get("detail")
    except ValueError:
        detail = None
    if isinstance(detail, list):  # FastAPI validation errors
        detail = "; ".join(str(d.get("msg", d)) for d in detail)
    return detail or f"Request failed ({resp.status_code})."


UNREACHABLE_HINT = "Start the FastAPI backend locally with: uvicorn backend.main:app --host 127.0.0.1 --port 8000"


def _request(method: str, path: str, payload: dict | None = None) -> tuple[dict | None, str | None, int | None]:
    """Returns (json, None, status) on success or (None, user-readable error, http status or None)."""
    try:
        resp = requests.request(method, f"{API_BASE_URL}{path}", json=payload, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as exc:
        return None, f"Can't reach the API at {API_BASE_URL} ({type(exc).__name__}). {UNREACHABLE_HINT}", None
    if resp.status_code >= 400:
        return None, _detail(resp), resp.status_code
    return resp.json(), None, resp.status_code


def api(method: str, path: str, payload: dict | None = None) -> tuple[dict | None, str | None]:
    """Returns (json, None) on success or (None, user-readable error)."""
    data, err, _ = _request(method, path, payload)
    return data, err


def stream_chat(path: str, payload: dict):
    """POST to a streaming endpoint. Yields ("stage", event) as each step happens, then ("result", data)
    or ("error", message)."""
    try:
        with requests.post(f"{API_BASE_URL}{path}", json=payload, stream=True, timeout=(10, REQUEST_TIMEOUT)) as resp:
            if resp.status_code >= 400:
                yield "error", _detail(resp)
                return
            resp.encoding = "utf-8"
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                event = json.loads(line)
                if event["event"] == "stage":
                    yield "stage", event
                elif event["event"] == "result":
                    yield "result", event["data"]
                elif event["event"] == "error":
                    yield "error", event["detail"]
    except requests.RequestException as exc:
        yield "error", f"Lost the connection to the API ({type(exc).__name__}). {UNREACHABLE_HINT}"


# --- state ---------------------------------------------------------------------
def init_state() -> None:
    ss = st.session_state
    ss.setdefault("session_id", str(uuid.uuid4()))
    ss.setdefault("job_id", None)
    ss.setdefault("source_id", None)
    ss.setdefault("job", None)
    ss.setdefault("last_url", None)
    ss.setdefault("ingest_error", None)
    ss.setdefault("messages", [])
    ss.setdefault("player", None)  # the period currently loaded in the player
    ss.setdefault("pending", None)  # a question waiting for its answer
    ss.setdefault("mode", MODE_SEARCH)


def reset_chat() -> None:
    """A fresh backend session too, so old turns can't leak into a new video's follow-ups."""
    ss = st.session_state
    ss.messages = []
    ss.player = None
    ss.pending = None
    ss.session_id = str(uuid.uuid4())


def refresh_job() -> dict | None:
    ss = st.session_state
    job = ss.job
    if ss.job_id and not (job and job.get("status") in SETTLED):
        data, err, status = _request("GET", f"/ingest/status/{ss.job_id}")
        if err:
            if status == 404:
                # The backend restarted (free hosts do this) and its in-memory job status is gone. The INDEX is not:
                # it lives in Qdrant, so nothing already finished was lost.
                err = (
                    "The backend restarted, so the previous job status was lost. "
                    "Submit the same URL again; already indexed videos will be reused."
                )
            ss.ingest_error = err
            ss.job_id = ss.source_id = ss.job = None
            return None
        ss.job = job = data
    return job


def start_ingest(url: str, keep_chat: bool = False) -> None:
    ss = st.session_state
    data, err = api("POST", "/ingest", {"url": url})
    if err:
        ss.ingest_error = err
        return
    ss.ingest_error = None
    ss.last_url = url
    ss.job_id, ss.source_id, ss.job = data["job_id"], data["source_id"], None
    if not keep_chat:
        reset_chat()
    st.rerun()


def indexed_count(job: dict | None) -> int:
    if not job:
        return 0
    return max(0, job.get("processed_videos", 0) - len(job.get("skipped") or []))


# --- sidebar: indexing status ---------------------------------------------------------
def render_activity_log(job: dict, expanded: bool) -> None:
    events = job.get("events") or []
    if not events:
        return
    t0 = events[0]["t"]
    with st.expander("Activity log", expanded=expanded):
        lines = []
        for e in reversed(events[-14:]):  # newest first
            rel = int(e["t"] - t0)
            lines.append(f"+{rel // 60}:{rel % 60:02d}  {e['msg']}")
        st.text("\n".join(lines))


SOURCE_LABELS = {
    "youtube_manual": ("YouTube", "Manually created"),
    "youtube_manual_translated": ("YouTube", "Manually created"),
    "youtube_generated": ("YouTube", "Auto-generated by YouTube"),
    "youtube_generated_translated": ("YouTube", "Auto-generated by YouTube"),
    "existing": ("YouTube", "Already indexed earlier"),
}


def render_transcript_summary(job: dict) -> None:
    """Says exactly where the indexed text came from, rather than implying the app
    transcribed anything itself (spec §6, §21, §45)."""
    indexed = job.get("indexed") or []
    if not indexed:
        return
    with st.expander(f"Transcript source ({len(indexed)} video(s))", expanded=len(indexed) == 1):
        for v in indexed:
            source, kind = SOURCE_LABELS.get(v.get("transcript_source", ""), ("YouTube", "Transcript"))
            st.markdown(f"**{v.get('title', 'Untitled')}**")
            if v.get("reused"):
                st.caption("Already in the index from an earlier run, so it was reused as is.")
                continue
            lines = [f"Transcript source: {source}", f"Type: {kind}", f"Language: {v.get('language_name') or v.get('language') or 'unknown'}"]
            lines.append("Translated to: English" if v.get("translated") else "Translation: not required")
            lines.append(f"Chunks: {v.get('chunks', 0)}")
            st.caption(" · ".join(lines))


def _duration(seconds: float | int) -> str:
    seconds = max(0, int(seconds or 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"


def render_timings(job: dict) -> None:
    timings = job.get("timings") or {}
    if not timings or timings.get("total_seconds", 0) <= 0:
        return
    with st.expander("Processing timings", expanded=False):
        st.caption(
            " · ".join([
                f"Metadata {_duration(timings.get('metadata_seconds'))}",
                f"Transcripts {_duration(timings.get('transcript_seconds'))}",
                f"Translation {_duration(timings.get('translation_seconds'))}",
                f"Embeddings {_duration(timings.get('embedding_seconds'))}",
                f"Qdrant {_duration(timings.get('indexing_seconds'))}",
                f"Total {_duration(timings.get('total_seconds'))}",
            ])
        )


def render_job_status(job: dict | None) -> None:
    ss = st.session_state
    if ss.ingest_error:
        st.error(ss.ingest_error)
    if not job:
        if not ss.ingest_error:
            st.caption("Nothing indexed yet.")
        return

    status, stage, total = job.get("status", "queued"), job.get("stage", "queued"), job.get("total_videos", 0)

    if status in ACTIVE:
        if total:
            frac = (job.get("processed_videos", 0) + job.get("stage_progress", 0.0)) / total
            label = f"Video {job.get('current_video')} of {total}" if job.get("current_video") else f"{total} videos found"
        else:
            frac, label = 0.0, "Getting started..."
        st.progress(min(max(frac, 0.0), 1.0), text=label)
        st.markdown(f"**{STAGE_ICONS.get(stage, '⏳')} {STAGE_TITLES.get(stage, stage.title())}**")
        st.caption(job.get("detail", ""))
        if job.get("current_title"):
            st.caption(f"Now: {job['current_title']}")
        render_activity_log(job, expanded=False)
        return

    if status == "done":
        skipped = job.get("skipped") or []
        st.success(f"✓ Indexed {total - len(skipped)} of {total} video(s).")
        render_transcript_summary(job)
        render_timings(job)
        if skipped:
            with st.expander(f"Skipped videos ({len(skipped)})"):
                for item in skipped:
                    st.markdown(f"- **{item.get('title', 'Untitled')}**: {item.get('reason', 'unknown reason')}")
            st.caption(
                "A skipped video had no usable transcript, or is private or restricted. Nothing is wrong with "
                "the app, and the videos that did index are fully searchable."
            )
    elif status == "paused":
        # A pause is NOT a failure and NOT a skip: it means an outside service asked us
        # to back off. Everything indexed so far is searchable, and Resume continues.
        # The backend's message already says which service (YouTube blocked or rate-limited, the external
        # transcript provider's quota, the language model's quota) and is safe to show as-is.
        st.warning(f"⏸️ Paused: {job.get('error') or 'an external service asked us to wait.'}")
        st.caption("Nothing was lost: finished videos stay indexed and are reused when you resume.")
        wait = job.get("resume_after_seconds")
        if wait:
            st.caption(f"Suggested wait: about {max(1, int(wait) // 60)} minute(s).")
        st.caption(f"{indexed_count(job)} of {total} video(s) are indexed so far, and you can already ask about them.")
        render_transcript_summary(job)
        render_timings(job)
        if ss.last_url:
            if st.button("Resume indexing", key="resume", type="primary", use_container_width=True):
                start_ingest(ss.last_url, keep_chat=True)  # finished videos are recognised and skipped instantly
        else:
            st.caption("Paste the link again and click Index to resume.")
    elif stage == "rejected":
        st.error(f"🚫 {job.get('error')}")
        st.caption("Nothing was indexed.")
    else:
        st.error(f"❌ {job.get('error') or 'Indexing failed.'}")
        if indexed_count(job) > 0:
            st.caption(f"{indexed_count(job)} video(s) indexed before this happened are still searchable.")
        render_timings(job)
    render_activity_log(job, expanded=False)


def render_library() -> None:
    data, err = api("GET", "/library")
    if err or not data:
        return
    sources = data.get("sources") or []
    if not sources:
        return
    with st.expander("My local knowledge base", expanded=False):
        for item in sources[:12]:
            label = item.get("source_id", "unknown")
            kind = item.get("kind", "source")
            st.markdown(f"**{kind.title()}: {label}**")
            st.caption(
                f"{item.get('indexed_count', 0)} / {item.get('videos', 0)} indexed · "
                f"{_duration(item.get('duration_seconds', 0))}"
            )


def render_sidebar(job: dict | None) -> None:
    with st.sidebar:
        st.title("🎬 ScrubGPT")
        st.caption("Ask about a YouTube video or playlist and jump to the exact moment.")
        st.divider()

        with st.form("ingest_form"):
            url = st.text_input("YouTube video or playlist link", placeholder="https://www.youtube.com/watch?v=...")
            submitted = st.form_submit_button("Index", type="primary", use_container_width=True)
        if submitted:
            start_ingest(url)
        render_job_status(job)
        render_library()

        st.divider()
        st.radio(
            "Answer mode",
            [MODE_SEARCH, MODE_ASK],
            key="mode",
            help="Timestamps only: the matching sections. Generated answer: a short written answer that cites them.",
        )
        if st.button("Clear chat", use_container_width=True):
            reset_chat()
            st.rerun()


# --- centre: player ------------------------------------------------------------------
def render_player(multi: bool) -> None:
    p = st.session_state.player
    if not p:
        st.info("Ask a question, then click a result. The video plays here from that moment.")
        return
    where = f"Video {p['position'] + 1} · " if multi else ""
    st.markdown(f"**{where}{p['video_title']}**")
    st.caption(f"Playing from {p['timestamp']}" + (f" · section {p['label']}" if p["is_range"] else ""))
    st.video(f"https://www.youtube.com/watch?v={p['video_id']}", start_time=int(p["start_sec"]))


# --- centre: chat ----------------------------------------------------------------------
def render_matches(matches: list[dict], msg_idx: int, multi: bool) -> None:
    """One button per period (a range like '41:02 - 48:53', or a lone timestamp), in playing order."""
    for j, m in enumerate(matches):
        n = f"[{m['n']}] " if "n" in m else ""
        if st.button(f"▶ {n}{m['label']}", key=f"play_{msg_idx}_{j}", use_container_width=True):
            st.session_state.player = m
            st.rerun()
        where = f"Video {m['position'] + 1} · " if multi else ""
        st.caption(f"{where}{m['video_title']}: {m.get('snippet', '')}")


def render_message(msg: dict, idx: int, multi: bool) -> None:
    st.markdown(msg["content"])
    if msg.get("note"):
        st.caption(msg["note"])
    render_matches(msg.get("matches", []), idx, multi)
    if msg.get("steps"):
        with st.expander("How this was found"):
            for line in msg["steps"]:
                st.write(line)


def build_reply(mode: str, data: dict | None, err: str | None) -> dict:
    if err:
        return {"role": "assistant", "content": f"⚠️ {err}", "matches": []}
    if mode == MODE_SEARCH:
        matches = data["results"]
        if not data["confident"]:
            content = "Nothing in the indexed videos matched that closely enough."
        else:
            n = len(matches)
            content = f"Found {n} relevant section{'s' if n != 1 else ''}, in order of appearance:"
        return {"role": "assistant", "content": content, "matches": matches}
    return {
        "role": "assistant",
        "content": data["answer"],
        "matches": data["citations"],
        "note": None if data["grounded"] else "No supporting excerpts were cited for this answer.",
    }


def handle_pending() -> None:
    """Runs inside the chat box: streams each step of the answer into a live status box, then stores the reply."""
    ss = st.session_state
    pending = ss.pending
    payload = {"session_id": ss.session_id, "source_id": ss.source_id, "question": pending["question"]}
    path = "/search/stream" if pending["mode"] == MODE_SEARCH else "/ask/stream"

    steps, result, error = [], None, None
    with st.chat_message("assistant"):
        status = st.status("Working on it...", expanded=True)
        for kind, value in stream_chat(path, payload):
            if kind == "stage":
                steps.append(f"{STAGE_ICONS.get(value['stage'], '•')} {value['message']}")
                status.write(steps[-1])
                status.update(label=value["message"])
            elif kind == "result":
                result = value
            else:
                error = value
        if error:
            status.update(label="Something went wrong", state="error", expanded=False)
        else:
            status.update(label=f"Done ({len(steps)} steps)", state="complete", expanded=False)

    reply = build_reply(pending["mode"], result, error)
    reply["steps"] = steps
    ss.messages.append(reply)
    if reply["matches"]:
        ss.player = reply["matches"][0]  # load the first section (earliest in sequence) into the player
    ss.pending = None
    st.rerun()  # redraw so the player, which sits above, picks up the new selection


# --- main ------------------------------------------------------------------------------
def main() -> None:
    init_state()
    ss = st.session_state
    job = refresh_job()
    render_sidebar(job)

    multi = bool(job and job.get("total_videos", 0) > 1)
    indexing = bool(job and job.get("status") in ACTIVE)
    # "error" is included: a job that failed part-way still indexed real videos, and
    # refusing to let the user search them would throw that work away.
    ready = bool(job and job.get("status") in ("done", "paused", "error") and indexed_count(job) > 0)

    st.title("ScrubGPT 🎬")
    left, right = st.columns([3, 2], gap="large")

    with left:
        render_player(multi)

    with right:
        st.subheader("Chat")
        chat_box = st.container(height=CHAT_HEIGHT)  # fixed height: scrolls inside instead of growing the page
        with chat_box:
            if not ss.messages and not ss.pending:
                if ready:
                    st.caption("Ask a question about the indexed video(s)." + (" Indexing is paused, but what's done is searchable." if job.get("status") == "paused" else ""))
                elif indexing:
                    st.caption("Indexing... you can ask questions as soon as it finishes. Progress is in the sidebar.")
                else:
                    st.caption("Paste a YouTube link in the sidebar and click Index to start.")
            for i, msg in enumerate(ss.messages):
                with st.chat_message(msg["role"]):
                    render_message(msg, i, multi)
            if ss.pending:
                handle_pending()

    # Called directly in the main flow (not inside columns/containers) so Streamlit pins it to the bottom.
    prompt = st.chat_input("Ask about the videos...", disabled=not ready or bool(ss.pending))
    if prompt:
        ss.messages.append({"role": "user", "content": prompt})
        ss.pending = {"question": prompt, "mode": ss.mode}
        st.rerun()

    if indexing:  # poll at the very end so the page above is already drawn while we wait
        time.sleep(POLL_SECONDS)
        st.rerun()


if __name__ == "__main__":
    main()
