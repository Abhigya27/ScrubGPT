"""Orchestrates one video end-to-end . Called by jobs.py.

    already indexed?  -> reuse the existing points, NO transcript request
            |
    transcript cache / YouTube direct
            |
    chunk the ORIGINAL-language transcript   <- timestamps are fixed here
            |
    translate each finished chunk, only if the source is not English
            |
    embed + upsert to Qdrant
            |
    the video counts as indexed ONLY once Qdrant confirms the write (spec §30)

Order matters: chunk first, translate second. Never translate the whole
transcript and then chunk it — the timestamps would no longer line up with what
was actually said, and timestamps are the entire point of this application.

Every step reports what it is doing through `progress`, so the UI can show it live.
"""
import logging

from backend import config
from backend.index import qdrant_store
from backend.ingest import chunker, translate
from backend.ingest import transcript as transcript_provider
from backend.models import IndexedVideo, Video

log = logging.getLogger(__name__)


class SkipVideo(Exception):
    """An EXPECTED, reportable outcome for one video: no manual transcript, private,
    unusable transcript. The playlist carries on (spec §14, §35).

    Deliberately NOT raised for a rate limit or a bug. A 429 means "ask again
    later", so it pauses the job instead (spec §8); a bug means the job failed and
    should say so rather than masquerading as a skipped video.
    """

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _noop(*args, **kwargs) -> None:
    pass


def _is_english(language: str) -> bool:
    return (language or "").lower().replace("_", "-").split("-")[0] == "en"


def process_video(video: Video, source_id: str, position: int = 0, progress=None, timing=None) -> IndexedVideo:
    """Index one video for `source_id`. `position` is its 0-based place in the playlist.

    `progress(stage, detail, fraction=None, log=True)` is optional.

    Returns an IndexedVideo on success. Raises:
      SkipVideo                          expected, reportable -> the job records it and moves on
      transcript.TemporaryTranscriptFailure  every provider failed temporarily -> the job PAUSES, video not skipped
      QuotaExhausted                     the LLM said "later" -> the job pauses
      TranscriptConfigError / TranscriptParseError / anything else   a real failure -> the job reports an error
    """
    report = progress or _noop
    record_timing = timing or _noop

    # Step 1 (title/duration/language) already happened at analysis time: one flat
    # request for a whole playlist instead of one per video (spec §27).

    # Step 2: dedup. If any point already carries this video_id, don't re-ingest;
    # just add this source to the existing points' playlist_ids. Note this happens
    # BEFORE any transcript request, so re-submitting a link (or resuming a paused
    # playlist) never asks YouTube for a finished video again.
    report("checking", "Checking whether this video is already indexed...", 0.0)
    if qdrant_store.video_exists(video.video_id):
        report("checking", "Already indexed, so it is reused as is (just added to this playlist).", 1.0)
        qdrant_store.add_playlist_tag(video.video_id, source_id, position)
        log.info("video %s already indexed; tagged with source %s", video.video_id, source_id)
        return IndexedVideo(
            video_id=video.video_id, title=video.title, chunks=0, language="", language_name="",
            transcript_source="existing", translated=False, reused=True,
        )

    # Step 3: get the transcript. The transcript layer checks the persistent local cache before contacting YouTube.
    report("transcript", "Checking the local transcript cache / YouTube...", 0.0)
    try:
        result = transcript_provider.get_transcript(
            video.video_id,
            original_language=video.language,
            on_retry=lambda message, seconds: report("transcript", message, 0.1, log=True),
            on_fallback=lambda message: report("transcript", message, 0.1, log=True),
            video_url=f"https://www.youtube.com/watch?v={video.video_id}",
        )
    except (transcript_provider.TranscriptUnavailable, transcript_provider.YouTubeAccessDenied) as exc:
        # Expected: this video simply cannot be indexed. Not an application error.
        log.info("skipping video %s (%s): %s", video.video_id, video.title, exc.reason)
        report("transcript", f"Skipped: {exc.reason}", 1.0)
        raise SkipVideo(exc.reason) from exc

    report("transcript", _found_message(result), 1.0)

    # Step 4: chunk the RAW original-language transcript, so timestamps are final.
    report("chunking", f"Splitting the transcript into ~{config.CHUNK_SECONDS}s chunks (timestamps fixed here)...", 0.0)
    chunks = chunker.chunk_transcript(
        result.lines, video, source_id, result.language, result.source_type, position
    )
    if not chunks:
        log.info("skipping video %s (%s): no usable chunks", video.video_id, video.title)
        report("chunking", "Skipped: the transcript is too short or unusable.", 1.0)
        raise SkipVideo("the YouTube transcript was too short or unusable to index")
    report("chunking", f"Made {len(chunks)} chunks.", 1.0)

    # Step 5: translate each chunk's already-finalized text, only if it isn't English.
    # An English transcript costs zero LLM calls, which is most of the ingest budget (spec §19).
    translated = not _is_english(result.language)
    if translated:
        report("translating", f"Translating {len(chunks)} chunks from {result.language_name} to English...", 0.0)
        chunks = translate.translate_chunks(
            chunks,
            on_progress=lambda done, total: report(
                "translating", f"Translating chunk {done} of {total} ({result.language_name} to English)...",
                done / total, log=False,
            ),
        )
        source_type = f"{result.source_type}_translated"  # provenance survives translation, whichever provider it came from
        chunks = [_retag(c, source_type) for c in chunks]
        report("translating", "Translation finished.", 1.0)
    else:
        source_type = result.source_type
        report("translating", "The transcript is already English, so no translation is needed.", 1.0)

    # Step 6: embed + upsert. The video is only "indexed" once this returns.
    report("saving", f"Embedding {len(chunks)} chunks (keyword + semantic vectors)...", 0.0)
    write_timings = qdrant_store.upsert_chunks(chunks) or {}
    record_timing("embedding_seconds", float(write_timings.get("embedding_seconds", 0.0)))
    record_timing("indexing_seconds", float(write_timings.get("indexing_seconds", 0.0)))
    report("saving", "Saved to Qdrant.", 1.0)
    log.info(
        "indexed video %s: %d chunks (language=%s, source=%s)",
        video.video_id, len(chunks), result.language, source_type,
    )
    return IndexedVideo(
        video_id=video.video_id,
        title=video.title,
        chunks=len(chunks),
        language=result.language,
        language_name=result.language_name,
        transcript_source=source_type,
        translated=translated,
        provider=result.provider,
    )


def _found_message(result) -> str:
    """What the UI's activity log says, with YouTube provenance visible."""
    n = len(result.lines)
    kind = "auto-generated" if result.is_generated else "manual"
    return f"Found a {kind} {result.language_name} YouTube transcript ({n} lines)."


def _retag(chunk, source_type: str):
    from dataclasses import replace

    return replace(chunk, transcript_source=source_type)
