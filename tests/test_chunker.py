"""Timestamp-preserving chunking: windows, overlap, boundaries and the junk filters."""
import pytest

from backend import config
from backend.ingest import chunker
from backend.models import TranscriptLine
from tests.helpers import demo_video, make_lines


@pytest.fixture(autouse=True)
def chunk_settings(monkeypatch):
    monkeypatch.setattr(config, "CHUNK_SECONDS", 90)
    monkeypatch.setattr(config, "CHUNK_OVERLAP_SECONDS", 20)
    monkeypatch.setattr(config, "MIN_CHUNK_WORDS", 15)
    monkeypatch.setattr(config, "DOMINANT_SENTENCE_RATIO", 0.5)


def chunk(lines, source="playlist", language="hi", position=0):
    return chunker.chunk_transcript(lines, demo_video(), source, language, "youtube_manual", position)


def test_windows_and_overlap_for_ten_second_lines():
    """30 lines x 10 s: 90 s windows that each re-read the last 20 s of the previous one."""
    chunks = chunk(make_lines(30))
    assert [(c.start_sec, c.end_sec) for c in chunks] == [(0, 90), (70, 160), (140, 230), (210, 300)]


def test_consecutive_chunks_overlap_by_at_most_the_configured_amount():
    chunks = chunk(make_lines(30))
    for earlier, later in zip(chunks, chunks[1:]):
        overlap = earlier.end_sec - later.start_sec
        assert 0 <= overlap <= config.CHUNK_OVERLAP_SECONDS


def test_every_second_of_the_transcript_is_covered():
    lines = make_lines(30)
    chunks = chunk(lines)
    assert chunks[0].start_sec == lines[0].start
    assert chunks[-1].end_sec == lines[-1].end
    for earlier, later in zip(chunks, chunks[1:]):
        assert later.start_sec <= earlier.end_sec  # no gap between windows


def test_chunk_boundaries_fall_on_segment_boundaries():
    lines = make_lines(30)
    starts, ends = {int(ln.start) for ln in lines}, {int(ln.end) for ln in lines}
    for c in chunk(lines):
        assert c.start_sec in starts and c.end_sec in ends


def test_a_chunk_never_exceeds_the_window_when_segments_are_short():
    assert all(c.end_sec - c.start_sec <= config.CHUNK_SECONDS for c in chunk(make_lines(60)))


def test_chunk_text_is_the_joined_segments():
    lines = make_lines(30)
    first = chunk(lines)[0]
    assert first.text == " ".join(ln.text for ln in lines[:9])


def test_chunk_ids_are_stable_and_unique():
    first, second = chunk(make_lines(60)), chunk(make_lines(60))
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert len({c.chunk_id for c in first}) == len(first)
    assert first[0].chunk_id == "vid:0" and first[1].chunk_id == f"vid:{first[1].start_sec}"


def test_metadata_is_carried_onto_every_chunk():
    for c in chunk(make_lines(30), source="PLxyz", language="hi", position=3):
        assert (c.video_id, c.video_title, c.playlist_ids) == ("vid", "Demo", ["PLxyz"])
        assert c.positions == {"PLxyz": 3}
        assert (c.source_language, c.transcript_source) == ("hi", "youtube_manual")


def test_a_segment_longer_than_the_window_is_kept_whole():
    long_line = TranscriptLine(0, 200, " ".join(f"word{i}" for i in range(40)))
    chunks = chunk([long_line])
    assert [(c.start_sec, c.end_sec) for c in chunks] == [(0, 200)]


def test_blank_segments_are_ignored():
    lines = make_lines(30)
    lines.insert(5, TranscriptLine(50, 50, "   "))
    assert [c.chunk_id for c in chunk(lines)] == [c.chunk_id for c in chunk(make_lines(30))]


def test_too_short_transcripts_produce_no_chunks():
    assert chunk([TranscriptLine(0, 5, "just five little words here")]) == []
    assert chunk([]) == []


def test_repetition_is_filtered_out():
    """A transcript that is one phrase repeated is junk, not content."""
    spam = [TranscriptLine(i * 10, i * 10 + 10, "thanks for watching") for i in range(20)]
    assert chunk(spam) == []


def test_mixed_content_with_some_repetition_is_kept():
    lines = make_lines(30)
    lines[3] = TranscriptLine(30, 40, "thanks for watching")
    assert chunk(lines)


def test_chunking_preserves_stable_timestamp_identity():
    lines = [TranscriptLine(i * 10, i * 10 + 10, f"sentence {i} with enough words to make a useful chunk") for i in range(12)]
    chunks = chunker.chunk_transcript(lines, demo_video("vid", "Demo", 180), "playlist", "hi", "youtube_manual", 0)
    assert chunks and chunks[0].chunk_id.startswith("vid:")
    assert chunks[0].transcript_source == "youtube_manual"
