"""Data models (spec §4, §25, §26). Plain dataclasses, no behavior."""
from dataclasses import dataclass, field


@dataclass
class Video:
    video_id: str
    title: str
    duration: int  # seconds
    language: str | None = None  # the video's original spoken language, when YouTube reports it


@dataclass
class TranscriptLine:
    start: float
    end: float
    text: str  # in whatever language the transcript was written in


# The app dealt in "caption tracks" before; everything user-facing now says
# "transcript" (spec §26). The old name is kept as an alias so nothing breaks.
CaptionLine = TranscriptLine


@dataclass
class TranscriptResult:
    """One video's transcript plus its provenance (spec §25).

    `source_type` is what the UI shows the user, and what gets stored on every
    chunk, so the app can say *where the text came from* rather than vaguely
    claiming it transcribed the video (spec §6).

        youtube_manual                 a human wrote it, used as-is
        youtube_generated              YouTube's speech recognition (allowed by ALLOW_GENERATED_TRANSCRIPT)
        youtube_manual_translated       the selected YouTube manual transcript after LLM translation
        youtube_generated_translated    the selected YouTube generated transcript after LLM translation

    A translated source always keeps the original provenance.
    (e.g. youtube_manual_translated). The rest of the pipeline never cares which provider produced it.
    """

    lines: list[TranscriptLine]
    language: str  # BCP-47-ish code, e.g. "en", "hi"
    language_name: str  # human-readable, e.g. "English", "Hindi"
    is_generated: bool  # True when the selected YouTube track is auto-generated; False for manual captions
    provider: str  # "youtube"
    source_type: str


@dataclass
class Chunk:
    chunk_id: str  # f"{video_id}:{start_sec}" — stable across re-ingest
    video_id: str
    video_title: str
    playlist_ids: list[str]  # which submitted playlist(s)/session(s) this chunk belongs to
    start_sec: int
    end_sec: int
    text: str  # ENGLISH, post-translation, post-chunking
    source_language: str  # the original transcript's language code, e.g. "hi", "en"
    transcript_source: str = "youtube_manual"  # provenance; see TranscriptResult.source_type
    positions: dict[str, int] = field(default_factory=dict)  # source_id -> this video's 0-based position in that playlist


@dataclass
class IndexedVideo:
    """What the UI reports about a video that made it into the index (spec §22, §45)."""

    video_id: str
    title: str
    chunks: int
    language: str
    language_name: str
    transcript_source: str
    translated: bool
    reused: bool = False  # it was already in Qdrant; no transcript request was made
    provider: str = ""  # which provider supplied the transcript ("youtube"); "" when reused


@dataclass
class JobStatus:
    job_id: str
    source_id: str  # the playlist_id (or single-video id) this job is indexing
    status: str  # "queued" | "processing" | "paused" | "done" | "error"
    total_videos: int
    processed_videos: int
    skipped: list[dict]  # [{"video_id":..., "title":..., "reason":...}]
    error: str | None
    # --- live progress, so the UI can say what is happening right now ---
    stage: str = "queued"  # analyzing | measuring | checking | transcript | chunking | translating | saving | done | paused | rejected | failed
    detail: str = ""  # one human-readable line for the current step
    current_video: int = 0  # 1-based index of the video being worked on (0 = none yet)
    current_title: str = ""
    stage_progress: float = 0.0  # 0..1 through the current video
    total_seconds: int = 0  # total length of everything submitted, once known
    resume_after_seconds: int | None = None  # set when paused: roughly when trying again makes sense
    pause_reason: str | None = None  # "youtube" | "llm": which service asked us to back off
    indexed: list[dict] = field(default_factory=list)  # IndexedVideo dicts, for the transcript summary
    events: list[dict] = field(default_factory=list)  # recent activity, [{"t": epoch_seconds, "msg": "..."}]
    timings: dict[str, float] = field(default_factory=lambda: {
        "metadata_seconds": 0.0,
        "transcript_seconds": 0.0,
        "translation_seconds": 0.0,
        "embedding_seconds": 0.0,
        "indexing_seconds": 0.0,
        "total_seconds": 0.0,
    })


@dataclass
class ChatTurn:
    question: str
    standalone_question: str  # after history-aware condensation
    answer: str | None
    citations: list[dict]
