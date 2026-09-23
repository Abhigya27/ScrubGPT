"""yt-dlp metadata resolution: titles, durations, playlist contents (spec §3).

This module used to be `captions.py` and owned transcript selection as well. It
no longer does: `transcript.py` handles every transcript decision through
`youtube-transcript-api`, and yt-dlp is kept strictly for what it is good at —
resolving what a link points at (spec §3, §27).

That split is also the main reason the old HTTP 429 problem is gone. Before, a
single video cost TWO full yt-dlp watch-page extractions (one to resolve the
title, one to find the caption track), and the second one is what YouTube
throttled. Now it costs one.
"""
import logging

import yt_dlp
from yt_dlp.utils import DownloadError

from backend import config
from backend.models import Video
from backend.util import UserFacingError

log = logging.getLogger(__name__)

_PLACEHOLDER_TITLES = {"[private video]", "[deleted video]"}


def _opts(**extra) -> dict:
    return {"quiet": True, "no_warnings": True, "skip_download": True, "noprogress": True, **extra}


def friendly(exc: Exception) -> str:
    msg = str(exc).lower()
    if "private" in msg:
        return "this video is private"
    if any(w in msg for w in ("unavailable", "removed", "deleted", "does not exist")):
        return "this video is unavailable"
    if any(w in msg for w in ("sign in", "confirm your age", "members-only", "members only")):
        return "this video requires sign-in, or YouTube is blocking automated access"
    if any(w in msg for w in ("429", "too many requests")):
        return "YouTube is rate-limiting requests from this server right now"
    return "YouTube didn't let us read this video"


def list_playlist(playlist_id: str) -> list[Video]:
    """Every video in the playlist, in playlist order, in one flat round trip (extract_flat).

    The flat listing already returns every entry with its title and (almost
    always) its duration, so counting videos and adding up their length costs
    one request, not N.

    The listing is capped at MAX_VIDEOS_PER_JOB + 1 entries: enough to tell that a playlist is over the
    limit (guardrails.submission_length_error rejects it) without walking a 5,000-video playlist page by page.
    """
    url = f"https://www.youtube.com/playlist?list={playlist_id}"
    extra = {"playlistend": config.MAX_VIDEOS_PER_JOB + 1} if config.MAX_VIDEOS_PER_JOB > 0 else {}
    try:
        with yt_dlp.YoutubeDL(_opts(extract_flat="in_playlist", **extra)) as ydl:
            info = ydl.extract_info(url, download=False)
    except DownloadError as exc:
        raise UserFacingError(f"Couldn't open that playlist ({friendly(exc)}). It may be private or unavailable.") from exc

    entries = [e for e in (info.get("entries") or []) if e and e.get("id")]
    return [
        Video(video_id=e["id"], title=e.get("title") or e["id"], duration=int(e.get("duration") or 0))
        for e in entries
    ]


def resolve_video(video_id: str) -> Video:
    """Title, duration and (when YouTube reports it) original language for one video.

    The language is a hint for transcript selection (spec §47 priority 1). When it
    is missing, transcript selection falls back to "manual English, then any other
    manual transcript" rather than guessing.
    """
    try:
        with yt_dlp.YoutubeDL(_opts(noplaylist=True)) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
    except DownloadError as exc:
        raise UserFacingError(f"Couldn't open that video: {friendly(exc)}.") from exc
    return Video(
        video_id=video_id,
        title=info.get("title") or video_id,
        duration=int(info.get("duration") or 0),
        language=info.get("language") or None,
    )


def _with_duration(video: Video) -> Video:
    try:
        return resolve_video(video.video_id)
    except UserFacingError:
        return video  # unavailable/private: keeps duration 0, and is skipped-and-reported at ingest


def fill_missing_durations(videos: list[Video]) -> list[Video]:
    """The flat listing occasionally omits a duration. Look up just those, one after another,
    so the total-length check is accurate.

    Sequential on purpose: these are yt-dlp requests to YouTube from a shared cloud IP, and firing several
    at once is exactly the burst that gets a host throttled. It is rare (the flat listing almost always has
    durations) and bounded by MAX_VIDEOS_PER_JOB, so the extra seconds are not worth the risk.
    """
    missing = [v for v in videos if v.duration <= 0 and v.title.lower() not in _PLACEHOLDER_TITLES]
    if not missing:
        return videos
    resolved = {v.video_id: _with_duration(v) for v in missing}
    return [
        Video(
            video_id=v.video_id,
            title=v.title,
            duration=resolved[v.video_id].duration if v.video_id in resolved else v.duration,
            language=resolved[v.video_id].language if v.video_id in resolved else v.language,
        )
        for v in videos
    ]
