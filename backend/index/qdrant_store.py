"""Qdrant Cloud: collection setup, upsert, dedup check, hybrid threshold search (spec §7).

The existing Qdrant persistent store is kept as the index source of truth. The embedded/local
Qdrant client is intentionally not used in this iteration.
"""
import functools
import logging
import math
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from backend import config
from backend.index.embed import get_dense, get_sparse
from backend.models import Chunk
from backend.util import format_timestamp, retry, search_query

log = logging.getLogger(__name__)

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"
# Fixed namespace so point ids are deterministic: uuid5(FIXED_NAMESPACE, chunk_id).
# That makes upserts idempotent and dedup safe (spec §4).
FIXED_NAMESPACE = uuid.UUID("6f0a5c2e-3b1d-4f57-9a3e-1c2d4b5a6e7f")


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(FIXED_NAMESPACE, chunk_id))


@functools.lru_cache(maxsize=1)
def _client() -> QdrantClient:
    if not config.QDRANT_URL or not config.QDRANT_API_KEY:
        raise RuntimeError(
            "QDRANT_URL and QDRANT_API_KEY must be set (Qdrant Cloud). "
            "Local/embedded Qdrant is intentionally unsupported: see spec §7."
        )
    return QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY, timeout=30)


# Bumped to v2 when the video-title prefix was removed from chunk text. Old points were
# embedded WITH the title and would score differently from new ones, so mixing them in one
# collection would make retrieval inconsistent. A new name means a clean re-index; the old
# collection is left alone and can be deleted from the Qdrant console.
_SCHEMA_VERSION = "v2"


def collection_name() -> str:
    return f"ytrag_{_SCHEMA_VERSION}_{get_dense().dim}"  # schema + embedding dimension baked into the name


def ensure_collection() -> None:
    """One collection, with a named dense vector AND a named sparse vector on the same points."""
    client, name = _client(), collection_name()
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config={DENSE_VECTOR: qm.VectorParams(size=get_dense().dim, distance=qm.Distance.COSINE)},
            sparse_vectors_config={SPARSE_VECTOR: qm.SparseVectorParams(modifier=qm.Modifier.IDF)},
        )
        log.info("created Qdrant collection %s", name)
    existing = set((client.get_collection(name).payload_schema or {}).keys())
    for field in ("video_id", "playlist_ids"):  # dedup check / per-session scoping
        if field not in existing:
            client.create_payload_index(collection_name=name, field_name=field, field_schema=qm.PayloadSchemaType.KEYWORD)


_ensured = False
_ensure_lock = threading.Lock()


def ensure_ready() -> None:
    """ensure_collection(), once per process, on first real use.

    The API does NOT block its startup on this: it needs the embedding model (to know the vector size), and a free
    host cold-starts often. Startup warms it up in the background, and every entry point below calls this, so if
    Qdrant was unreachable at boot the first real request simply retries. Raises whatever ensure_collection raises.
    """
    global _ensured
    if _ensured:
        return
    with _ensure_lock:
        if not _ensured:
            ensure_collection()
            _ensured = True


# --- writes ------------------------------------------------------------------
def _payload(c: Chunk) -> dict:
    return {
        "chunk_id": c.chunk_id,
        "video_id": c.video_id,
        "video_title": c.video_title,
        "playlist_ids": c.playlist_ids,
        "start_sec": c.start_sec,
        "end_sec": c.end_sec,
        "text": c.text,
        "source_language": c.source_language,
        "transcript_source": c.transcript_source,  # youtube_manual | youtube_manual_translated | ...
        "positions": c.positions,  # {source_id: 0-based playlist position}
    }


@retry(attempts=3, base_delay=1.5)
def _upsert_batch(points: list[qm.PointStruct]) -> None:
    _client().upsert(collection_name=collection_name(), points=points, wait=True)


def upsert_chunks(chunks: list[Chunk]) -> dict[str, float]:
    """Embed and persist chunks, returning simple stage timings for local optimization."""
    if not chunks:
        return {"embedding_seconds": 0.0, "indexing_seconds": 0.0}
    ensure_ready()
    texts = [c.text for c in chunks]
    started = time.perf_counter()
    dense = get_dense().embed_documents(texts)
    sparse = get_sparse().embed_documents(texts)
    embedding_seconds = time.perf_counter() - started
    points = [
        qm.PointStruct(id=point_id(c.chunk_id), vector={DENSE_VECTOR: d, SPARSE_VECTOR: s}, payload=_payload(c))
        for c, d, s in zip(chunks, dense, sparse)
    ]
    started = time.perf_counter()
    for i in range(0, len(points), 64):
        _upsert_batch(points[i : i + 64])
    indexing_seconds = time.perf_counter() - started
    return {"embedding_seconds": embedding_seconds, "indexing_seconds": indexing_seconds}


# --- dedup (spec §5 step 2) ---------------------------------------------------
def _video_filter(video_id: str) -> qm.Filter:
    return qm.Filter(must=[qm.FieldCondition(key="video_id", match=qm.MatchValue(value=video_id))])


def video_exists(video_id: str) -> bool:
    ensure_ready()
    points, _ = _client().scroll(
        collection_name=collection_name(), scroll_filter=_video_filter(video_id),
        limit=1, with_payload=False, with_vectors=False,
    )
    return bool(points)


def add_playlist_tag(video_id: str, source_id: str, position: int = 0) -> None:
    """Add source_id to playlist_ids (and record the video's position in it) on every point of an
    already-indexed video.

    Qdrant's set_payload REPLACES a field's value, it does not append to a list.
    Writing [source_id] blindly would erase every earlier session's tag and
    silently break multi-tenancy, so read the current lists, merge, then write.
    """
    ensure_ready()
    client, name = _client(), collection_name()
    by_state: dict[tuple, list] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=name, scroll_filter=_video_filter(video_id), limit=256,
            offset=offset, with_payload=["playlist_ids", "positions"], with_vectors=False,
        )
        for p in points:
            payload = p.payload or {}
            tags = tuple(sorted(payload.get("playlist_ids") or []))
            positions = tuple(sorted((payload.get("positions") or {}).items()))
            by_state.setdefault((tags, positions), []).append(p.id)
        if offset is None:
            break

    for (tags, position_items), ids in by_state.items():
        positions = dict(position_items)
        if source_id in tags and positions.get(source_id) == position:
            continue  # already tagged, and the playlist order hasn't changed
        positions[source_id] = position
        client.set_payload(
            collection_name=name,
            payload={"playlist_ids": sorted({*tags, source_id}), "positions": positions},
            points=ids, wait=True,
        )


# --- hybrid search (spec §7) ---------------------------------------------------
_TF_TYPICAL = 1.3  # a matched term's typical BM25 term-frequency weight (fastembed defaults: k=1.2, b=0.75)
_size_cache = {"at": 0.0, "n": 0}


def _collection_size() -> int:
    """Approximate number of chunks in the whole collection (cached for a minute)."""
    now = time.monotonic()
    if now - _size_cache["at"] > 60 or _size_cache["n"] <= 0:
        try:
            _size_cache["n"] = int(_client().count(collection_name=collection_name(), exact=False).count)
        except Exception:
            log.warning("couldn't read the collection size; using the last known value")
        _size_cache["at"] = now
    return max(_size_cache["n"], 2)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _fuse_and_threshold(
    dense_hits, sparse_hits, threshold: float, max_results: int, source_id: str | None = None, n_points: int = 50
) -> list[dict]:
    """Pure function (no I/O): weighted score fusion, then a threshold cutoff.

    ===========================================================================
    REQUIREMENT (spec §7, firm). Do not "simplify" any of these three:
      1. SPARSE IS WEIGHTED ABOVE DENSE (SPARSE_WEIGHT 0.65 / DENSE_WEIGHT 0.35).
         Not an unweighted 50/50 fusion, and not Qdrant's built-in RRF.
      2. SELECTION IS BY THRESHOLD, NOT BY A FIXED TOP-K. Every chunk that
         clears `threshold` is returned, best first. `max_results` is only a
         safety valve against a pathologically broad match.
      3. The threshold is an absolute cutoff, so an irrelevant question
         returns nothing (which is what lets /ask's guard 1 fire).
    ===========================================================================

    Two precision guards run AFTER the threshold, in this order:
      RELATIVE_SCORE_FLOOR   drop chunks scoring below a fraction of the best chunk
      MIN_DENSE_SIMILARITY   drop chunks that are not semantically related at all

    Scale: each leg is mapped into [0, 1] absolutely, never relative to the
    other hits (min-max scaling would always crown a winner and make an
    absolute threshold meaningless).
      dense  = cosine similarity, clamped to [0, 1]
      sparse = raw BM25 score squashed as raw / (raw + half), where
               half = SPARSE_HALF_SATURATION_TERMS * 1.3 * ln(n_points)
    BM25 scores scale with ln(collection size) because of IDF, so `half` scales the
    same way (n_points = points in the whole collection, which is also what Qdrant's
    IDF is computed over). Without that, a fixed constant tuned on a one-video index
    let single-rare-word coincidences through once the index reached ~1,500 chunks.
    fused similarity = (SPARSE_WEIGHT*sparse + DENSE_WEIGHT*dense) / (sum of weights)
    distance         = 1 - fused, and a chunk clears the cutoff if distance <= threshold.

    Consequence worth knowing: a chunk with no lexical overlap has sparse = 0,
    so it can score at most DENSE_WEIGHT (0.35), i.e. distance >= 0.65, and can
    never clear the default 0.55. That follows from weighting sparse above dense
    plus the §11 default. Raise RETRIEVAL_THRESHOLD to admit purely semantic hits.
    """
    half = config.SPARSE_HALF_SATURATION_TERMS * _TF_TYPICAL * math.log(max(n_points, 2))
    candidates: dict[str, dict] = {}
    for hit in dense_hits:
        candidates[str(hit.id)] = {
            "payload": hit.payload or {}, "dense": _clamp01(hit.score), "sparse": 0.0, "raw": 0.0, "in_dense": True,
        }
    for hit in sparse_hits:
        raw = max(0.0, hit.score)
        cand = candidates.setdefault(
            str(hit.id), {"payload": hit.payload or {}, "dense": 0.0, "sparse": 0.0, "raw": 0.0, "in_dense": False},
        )
        cand["raw"] = raw
        cand["sparse"] = raw / (raw + half) if raw else 0.0

    total_weight = config.SPARSE_WEIGHT + config.DENSE_WEIGHT
    ranked = []
    for cand in candidates.values():
        fused = (config.SPARSE_WEIGHT * cand["sparse"] + config.DENSE_WEIGHT * cand["dense"]) / total_weight
        ranked.append((1.0 - fused, fused, cand))
    ranked.sort(key=lambda item: item[0])  # best (smallest distance) first

    kept = [item for item in ranked if item[0] <= threshold]  # the primary selection mechanism
    cleared = len(kept)

    # Precision guard on top of the absolute threshold: next to a strong match, drop chunks that
    # score well below it (they're the "one totally unrelated result" among good ones). The best
    # chunk always survives, so this can never turn a hit into "nothing found".
    if kept:
        floor = config.RELATIVE_SCORE_FLOOR * kept[0][1]
        kept = [item for item in kept if item[1] >= floor]
    after_floor = len(kept)

    # Precision guard 2 (MIN_DENSE_SIMILARITY): a chunk has to be at least vaguely ON TOPIC,
    # not merely share a word. This is what removes the classic "one totally unrelated result":
    # a passage about something else that clears the threshold because one query word happens to
    # appear in it, helped along by a high-ish sparse score. Unrelated text sits near 0.0-0.2
    # cosine, so the gate is cheap insurance rather than a real constraint.
    #
    # It is applied only to candidates the DENSE leg actually returned. A candidate found by the
    # sparse leg alone has dense = 0.0 simply because it fell outside the dense candidate pool,
    # not because it is unrelated, and gating on that would throw away strong keyword matches.
    if config.MIN_DENSE_SIMILARITY > 0:
        kept = [item for item in kept if not item[2]["in_dense"] or item[2]["dense"] >= config.MIN_DENSE_SIMILARITY]

    # Diagnostic for tuning RETRIEVAL_THRESHOLD / SPARSE_HALF_SATURATION_TERMS against real data.
    # 0 candidates means the source_id filter matched nothing (wrong id, or nothing indexed).
    best = "; ".join(
        f"{c['payload'].get('video_id')}@{c['payload'].get('start_sec')}s dense={c['dense']:.2f} bm25={c['raw']:.1f} dist={d:.2f}"
        for d, _, c in ranked[:3]
    )
    log.info(
        "hybrid_search: %d candidates (index %d chunks), %d cleared threshold %.2f, %d after relative floor %.2f, "
        "%d after dense gate %.2f. best: %s",
        len(ranked), n_points, cleared, threshold, after_floor, config.RELATIVE_SCORE_FLOOR,
        len(kept), config.MIN_DENSE_SIMILARITY, best or "-",
    )

    return [
        {
            "chunk_id": cand["payload"].get("chunk_id"),
            "video_id": cand["payload"].get("video_id"),
            "video_title": cand["payload"].get("video_title"),
            "start_sec": cand["payload"].get("start_sec", 0),
            "end_sec": cand["payload"].get("end_sec", 0),
            "timestamp": format_timestamp(cand["payload"].get("start_sec", 0)),
            "text": cand["payload"].get("text", ""),
            "source_language": cand["payload"].get("source_language"),
            "transcript_source": cand["payload"].get("transcript_source", "youtube_manual"),
            "position": int((cand["payload"].get("positions") or {}).get(source_id, 0)),
            "score": round(fused, 4),
            "distance": round(distance, 4),
        }
        for distance, fused, cand in kept[:max_results]  # safety cap only
    ]


class _Hit:
    """A search hit with a (possibly discounted) score. Same three attributes _fuse_and_threshold reads."""

    __slots__ = ("id", "score", "payload")

    def __init__(self, id, score, payload):
        self.id, self.score, self.payload = id, score, payload


def _best_per_point(groups: Sequence[tuple[float, list]]) -> list:
    """Union of several hit lists (one per query), keeping each point's BEST score.

    `groups` is [(weight, hits), ...]; a hit's score is multiplied by its group's weight (1.0 for the user's own
    question, EXPANDED_QUERY_DISCOUNT for an LLM-written extra query). Taking the maximum, rather than adding or
    averaging, keeps dense cosine and raw BM25 on exactly the scale the threshold, the sparse squash and the
    relative floor were tuned for: an extra query can rescue a chunk but never inflate one past its best match.
    """
    best: dict[str, _Hit] = {}
    for weight, hits in groups:
        for hit in hits:
            score = hit.score * weight
            key = str(hit.id)
            current = best.get(key)
            if current is None or score > current.score:
                best[key] = _Hit(hit.id, score, hit.payload)
    return sorted(best.values(), key=lambda h: h.score, reverse=True)


def fetch_candidates(query: str, source_id: str, max_results: int, expansions: Sequence[str] = ()):
    """The retrieval legs, unfiltered by threshold. Returns (dense_hits, sparse_hits, n_points).

    `expansions` are extra queries (rag/enhance.py) searched IN ADDITION to the question. Each one, and the question
    itself, is run through both the dense and the sparse leg, and the per-chunk best score wins.

    Split out so the evaluation tool can fetch once per question and then sweep thresholds offline.
    """
    ensure_ready()
    client, name = _client(), collection_name()

    # Retrieve on the TOPIC, not on the way the question was phrased. "when was self
    # supervised learning taught" searches for "self supervised learning": otherwise the
    # navigational words are embedded too, and passages about teaching, classes and
    # professors compete with passages about the actual subject. Done here rather than in
    # rag/retrieve.py so the evaluation tool, which calls this function directly, measures
    # exactly what the app does. The full question still goes to the answer model.
    topic = search_query(query)
    if topic != query:
        log.info("search query: %r -> %r", query, topic)

    weight = config.EXPANDED_QUERY_DISCOUNT
    queries: list[tuple[float, str]] = [(1.0, topic)]
    seen = {topic.strip().lower()}
    for extra in expansions:
        extra = (extra or "").strip()
        if extra and extra.lower() not in seen:
            seen.add(extra.lower())
            queries.append((weight, extra))
    if len(queries) > 1:
        log.info("query expansion: searching %d queries: %s", len(queries), " | ".join(q for _, q in queries))

    # Multi-tenancy (spec §7): ALWAYS restrict to the requesting session's source_id.
    source_filter = qm.Filter(must=[qm.FieldCondition(key="playlist_ids", match=qm.MatchValue(value=source_id))])

    # Candidate pool per leg. This is depth, not selection: the threshold decides
    # what is returned. Deep enough that a leg's tail rarely matters.
    pool = max(50, max_results * 5)

    # Embedding is local CPU work; the Qdrant round trips are network. Build every (leg, vector) first, then run
    # the round trips together: a few small independent reads, safe to overlap (this client is already used from
    # several threads at once by jobs and chat).
    legs: list[tuple[str, float, object]] = []
    for w, text in queries:
        legs.append(("dense", w, get_dense().embed_query(text)))
        sparse_query = get_sparse().embed_query(text)
        if sparse_query.indices:  # a query of only stopwords has no sparse terms; skip that leg
            legs.append(("sparse", w, sparse_query))

    def run(leg):
        kind, w, vector = leg
        points = client.query_points(
            collection_name=name, query=vector, using=DENSE_VECTOR if kind == "dense" else SPARSE_VECTOR,
            query_filter=source_filter, limit=pool, with_payload=True,
        ).points
        return kind, w, points

    with ThreadPoolExecutor(max_workers=min(4, len(legs))) as pool_executor:
        results = list(pool_executor.map(run, legs))

    dense_hits = _best_per_point([(w, pts) for kind, w, pts in results if kind == "dense"])
    sparse_hits = _best_per_point([(w, pts) for kind, w, pts in results if kind == "sparse"])
    return dense_hits, sparse_hits, _collection_size()


def hybrid_search(query: str, source_id: str, threshold: float, max_results: int, expansions: Sequence[str] = ()) -> list[dict]:
    """Dense + sparse search, sparse-weighted fusion, threshold cutoff. See _fuse_and_threshold.

    `expansions`: extra queries from rag/enhance.py, searched in addition to `query` (best score per chunk wins)."""
    dense_hits, sparse_hits, n_points = fetch_candidates(query, source_id, max_results, expansions)
    return _fuse_and_threshold(dense_hits, sparse_hits, threshold, max_results, source_id, n_points)
