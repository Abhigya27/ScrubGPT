"""Time-domain ~90s chunking over TranscriptLine segments (spec §5 step 4, §16).

This runs on the ORIGINAL-language transcript, before any translation, so every
chunk's start_sec/end_sec is fixed by real transcript timestamps. Translation
(translate.py) only ever rewrites a finished chunk's text afterwards. Do not
reorder this: translating first and chunking second would lose the timestamps
this whole application is built on (spec §16, §33).
"""
import math
import re
from collections import Counter

from backend import config
from backend.models import Chunk, TranscriptLine, Video

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?।。？！])\s+")
_NON_WORD = re.compile(r"[^\w\s]", re.UNICODE)


def chunk_transcript(
    lines: list[TranscriptLine],
    video: Video,
    source_id: str,
    source_language: str,
    transcript_source: str = "youtube_manual",
    position: int = 0,
) -> list[Chunk]:
    """Split one video's transcript into overlapping ~CHUNK_SECONDS windows."""
    segs = [ln for ln in lines if ln.text and ln.text.strip()]
    chunks: list[Chunk] = []
    seen_ids: set[str] = set()
    i, n = 0, len(segs)
    while i < n:
        window_start = segs[i].start
        # Grow the window until the next segment would push it past CHUNK_SECONDS.
        # Always take at least one segment; never split a segment.
        j = i + 1
        while j < n and segs[j].end - window_start <= config.CHUNK_SECONDS:
            j += 1

        chunk = _build_chunk(segs[i:j], video, source_id, source_language, transcript_source, position)
        if chunk is not None and chunk.chunk_id not in seen_ids:
            seen_ids.add(chunk.chunk_id)
            chunks.append(chunk)

        if j >= n:
            break
        i = _next_window_start(segs, i, j)
    return chunks


def _next_window_start(segs: list[TranscriptLine], i: int, j: int) -> int:
    """Index where the next window starts: CHUNK_OVERLAP_SECONDS of re-read overlap,
    snapped to the first segment boundary inside that overlap (not a hard time cut)."""
    overlap_begins = max(seg.end for seg in segs[i:j]) - config.CHUNK_OVERLAP_SECONDS
    for k in range(i + 1, j):
        if segs[k].start >= overlap_begins:
            return k
    return j  # no segment boundary inside the overlap: continue with no overlap


def _build_chunk(
    window: list[TranscriptLine],
    video: Video,
    source_id: str,
    source_language: str,
    transcript_source: str,
    position: int,
) -> Chunk | None:
    texts = [" ".join(seg.text.split()) for seg in window]
    body = " ".join(texts)

    if len(body.split()) < config.MIN_CHUNK_WORDS:
        return None
    if _dominated_by_one_sentence(texts, body):
        return None

    start_sec = int(math.floor(window[0].start))
    end_sec = max(int(math.ceil(max(seg.end for seg in window))), start_sec)
    return Chunk(
        chunk_id=f"{video.video_id}:{start_sec}",  # stable across re-ingest
        video_id=video.video_id,
        video_title=video.title,
        playlist_ids=[source_id],
        start_sec=start_sec,
        end_sec=end_sec,
        # No title prefix. It used to be prepended to every chunk, which made every
        # chunk of a video look a little like the video's overall topic and let
        # off-topic passages clear the relevance threshold. The title is kept in the
        # payload (video_title) and shown with each result instead.
        text=body,
        source_language=source_language,
        transcript_source=transcript_source,
        positions={source_id: position},  # playlist order, so results can be returned in sequence
    )


def _normalize(unit: str) -> str:
    return " ".join(_NON_WORD.sub("", unit.lower()).split())


def _dominated_by_one_sentence(segment_texts: list[str], body: str) -> bool:
    """Repetition/junk filter: True if one sentence (counting repeats) is most of the chunk.

    Some transcripts carry no punctuation at all, so when the text doesn't split
    into at least two sentences we treat each transcript segment as the unit.
    """
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(body) if s.strip()]
    units = sentences if len(sentences) >= 2 else segment_texts
    if len(units) < 2:  # a single unit can't dominate anything; MIN_CHUNK_WORDS covers tiny chunks
        return False
    total_words = sum(len(u.split()) for u in units)
    if total_words == 0:
        return True
    counts = Counter(_normalize(u) for u in units)
    heaviest = max(len(u.split()) * count for u, count in counts.items())
    return heaviest / total_words > config.DOMINANT_SENTENCE_RATIO
