"""Parse a pasted URL -> ("video" | "playlist", id)  (spec §6).

Only the link's shape is judged here (and instantly, so POST /ingest can answer 422 at once). Private or
unavailable videos/playlists and the total-length limits (90 seconds minimum, 30 hours maximum) need a
YouTube lookup, so they are checked by the background job (jobs._analyze, metadata.list_playlist,
guardrails.submission_length_error) and show up in the job's status.
"""
import re
from urllib.parse import parse_qs, urlparse

from backend.util import UserFacingError

_YT_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be", "www.youtu.be"}
_VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
_PLAYLIST_ID = re.compile(r"^[A-Za-z0-9_-]{10,64}$")
_PATH_VIDEO = re.compile(r"^/(?:shorts|embed|live|v)/([A-Za-z0-9_-]{11})")


def parse_url(url: str) -> tuple[str, str]:
    raw = (url or "").strip()
    if not raw:
        raise UserFacingError("Please paste a YouTube video or playlist link.")
    if "://" not in raw:
        raw = "https://" + raw

    parsed = urlparse(raw)
    host = (parsed.hostname or "").lower()
    if host not in _YT_HOSTS:
        raise UserFacingError("That doesn't look like a YouTube link. Paste a youtube.com or youtu.be video or playlist URL.")

    query = parse_qs(parsed.query)
    playlist_id = (query.get("list") or [None])[0]

    video_id = None
    if host.endswith("youtu.be"):
        video_id = parsed.path.strip("/").split("/")[0] or None
    elif parsed.path == "/watch":
        video_id = (query.get("v") or [None])[0]
    else:
        match = _PATH_VIDEO.match(parsed.path)
        if match:
            video_id = match.group(1)

    # A video URL that also carries list= (e.g. watch?v=X&list=Y) is treated as
    # that VIDEO. The user pasted a specific video, and auto-generated "Mix"
    # lists (list=RD...) are effectively unbounded, so treating them as
    # playlists would just hit the size cap.
    if video_id:
        if not _VIDEO_ID.match(video_id):
            raise UserFacingError("That video link looks malformed (the video id isn't valid).")
        return "video", video_id

    if playlist_id and parsed.path in ("/playlist", "/watch"):
        if not _PLAYLIST_ID.match(playlist_id):
            raise UserFacingError("That playlist link looks malformed (the playlist id isn't valid).")
        return "playlist", playlist_id

    raise UserFacingError("Couldn't find a video or playlist in that link.")
