"""Retrieval step (spec §8): hybrid search, then merge the matching chunks into time periods.

Chunks are ~90 s windows, so one topic usually matches several neighbouring chunks.
Returning each as its own timestamp is noisy; instead matches are grouped into
periods and returned IN SEQUENCE (playlist order, then time within each video),
never in score order.
"""
from typing import Sequence

from backend import config
from backend.index import qdrant_store
from backend.util import body_text, format_timestamp


def retrieve_chunks(standalone_question: str, source_id: str, expansions: Sequence[str] = ()) -> list[dict]:
    """The chunks that cleared the hybrid threshold (best first, not yet merged).

    `expansions` are extra queries from rag/enhance.py (LLM rewrites and related keywords). They are searched in
    addition to the question, and the threshold and precision guards apply to the combined candidates exactly as
    they do to a plain search.
    """
    # source_id is always required. Spec §3 calls the source filter "optional" but §7
    # says hybrid_search must "always" filter on it (multi-tenancy); §7 wins.
    return qdrant_store.hybrid_search(
        standalone_question,
        source_id,
        threshold=config.RETRIEVAL_THRESHOLD,
        max_results=config.MAX_RESULTS,
        expansions=expansions,
    )


def retrieve(standalone_question: str, source_id: str, expansions: Sequence[str] = ()) -> list[dict]:
    """Periods for the question, in playing order. Empty if nothing cleared the threshold."""
    return group_into_periods(retrieve_chunks(standalone_question, source_id, expansions))


def group_into_periods(chunks: list[dict], gap_seconds: int | None = None) -> list[dict]:
    """Merge chunk hits into periods.

    Within one video, hits are ordered by time and a new period starts whenever
    a hit begins `gap_seconds` (default 3 minutes) or more after the previous
    hit began. Everything closer than that is one continuous period. This holds
    however many times a topic recurs: each recurrence is its own period.
    """
    gap = config.PERIOD_GAP_SECONDS if gap_seconds is None else gap_seconds

    by_video: dict[str, list[dict]] = {}
    for chunk in chunks:
        by_video.setdefault(chunk["video_id"], []).append(chunk)

    periods: list[dict] = []
    for video_chunks in by_video.values():
        video_chunks.sort(key=lambda c: c["start_sec"])
        group = [video_chunks[0]]
        for chunk in video_chunks[1:]:
            if chunk["start_sec"] - group[-1]["start_sec"] >= gap:
                periods.append(_make_period(group))
                group = [chunk]
            else:
                group.append(chunk)
        periods.append(_make_period(group))

    # Sequence: playlist position first, then time. (video_id keeps videos contiguous on ties.)
    periods.sort(key=lambda p: (p["position"], p["video_id"], p["start_sec"]))
    return periods


def _make_period(group: list[dict]) -> dict:
    start = min(c["start_sec"] for c in group)
    end = max(c["end_sec"] for c in group)
    # A topic that fits inside one chunk window (under CHUNK_SECONDS) is a plain timestamp, not a range.
    is_range = len(group) > 1 and (end - start) >= config.CHUNK_SECONDS
    best = max(group, key=lambda c: c["score"])
    return {
        "video_id": group[0]["video_id"],
        "video_title": group[0]["video_title"],
        "position": group[0].get("position", 0),
        "start_sec": start,
        "end_sec": end,
        "is_range": is_range,
        "timestamp": format_timestamp(start),
        "label": f"{format_timestamp(start)} - {format_timestamp(end)}" if is_range else format_timestamp(start),
        "chunks": len(group),
        "score": best["score"],
        "distance": best["distance"],
        "snippet": _snippet(body_text(best["text"])),
        "text": "\n...\n".join(body_text(c["text"]) for c in group),  # time-ordered; used for /ask, not returned by the API
    }


def _snippet(text: str, limit: int = 200) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "…"


def public_view(period: dict) -> dict:
    """What the API returns: a period without its (large) excerpt text."""
    return {k: v for k, v in period.items() if k != "text"}
