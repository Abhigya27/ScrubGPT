"""URL parsing: pure, so every case runs offline."""
import pytest

from backend.ingest.url_parser import parse_url
from backend.util import UserFacingError

VID = "dQw4w9WgXcQ"
PLAYLIST = "PLabcdefghijk123"


@pytest.mark.parametrize("url", [
    f"https://www.youtube.com/watch?v={VID}",
    f"https://youtube.com/watch?v={VID}&t=42s",
    f"https://m.youtube.com/watch?v={VID}",
    f"https://music.youtube.com/watch?v={VID}",
    f"youtube.com/watch?v={VID}",  # no scheme
    f"https://youtu.be/{VID}",
    f"https://youtu.be/{VID}?t=30",
    f"https://www.youtube.com/shorts/{VID}",
    f"https://www.youtube.com/embed/{VID}",
    f"https://www.youtube.com/live/{VID}",
    f"  https://www.youtube.com/watch?v={VID}  ",  # surrounding whitespace
])
def test_video_urls(url):
    assert parse_url(url) == ("video", VID)


def test_video_url_with_list_is_treated_as_the_video():
    """A pasted watch?v=X&list=Y is that video: auto-generated Mix lists are effectively unbounded."""
    assert parse_url(f"https://www.youtube.com/watch?v={VID}&list=RDabcdefghijk") == ("video", VID)


@pytest.mark.parametrize("url", [
    f"https://www.youtube.com/playlist?list={PLAYLIST}",
    f"https://youtube.com/watch?list={PLAYLIST}",  # /watch with a list and no v
    f"https://music.youtube.com/playlist?list={PLAYLIST}",
])
def test_playlist_urls(url):
    assert parse_url(url) == ("playlist", PLAYLIST)


@pytest.mark.parametrize("url", ["", "   ", None])
def test_empty_input(url):
    with pytest.raises(UserFacingError, match="paste"):
        parse_url(url)


@pytest.mark.parametrize("url", [
    "https://vimeo.com/12345678",
    f"https://notyoutube.com/watch?v={VID}",
    f"https://youtube.com.evil.example/watch?v={VID}",  # lookalike host
    "not a url at all",
])
def test_non_youtube_hosts_are_rejected(url):
    with pytest.raises(UserFacingError, match="YouTube link"):
        parse_url(url)


def test_malformed_video_id():
    with pytest.raises(UserFacingError, match="malformed"):
        parse_url("https://www.youtube.com/watch?v=tooshort")


def test_malformed_playlist_id():
    with pytest.raises(UserFacingError, match="malformed"):
        parse_url("https://www.youtube.com/playlist?list=abc")


@pytest.mark.parametrize("url", ["https://www.youtube.com/", "https://www.youtube.com/channel/UC123", "https://www.youtube.com/playlist"])
def test_youtube_links_without_a_video_or_playlist(url):
    with pytest.raises(UserFacingError, match="Couldn't find"):
        parse_url(url)
