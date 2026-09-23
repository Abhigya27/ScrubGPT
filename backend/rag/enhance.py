"""Query enhancement: history-aware condensation AND multi-query expansion in ONE LangChain call.

Hybrid retrieval (BM25 + dense, see index/qdrant_store.py) is only as good as the words it is given. People
describe a topic differently from the speaker: they ask "how do neural nets learn" and the transcript says
"backpropagation" and "gradient descent". BM25 sees no overlap, and because sparse is weighted above dense a
chunk with no lexical overlap can never clear the default threshold. So, before searching, one small LLM call:

    {"standalone": the question with pronouns/references resolved from the chat history,
     "rewrites":   a couple of alternative phrasings, worded the way a speaker might say it,
     "keywords":   distinctive related terms and synonyms}

The rewrites and one keyword string become EXTRA search queries. Each is searched with both retrieval legs and
the best score per chunk wins (qdrant_store.fetch_candidates), so score scales and the threshold keep their
meaning; extra queries can only surface a chunk the user's own words missed, and they count for slightly less
(EXPANDED_QUERY_DISCOUNT) so a drifting rewrite can never outrank a literal match. The question the user typed is
always searched too.

Why one call: follow-up condensation already needed an LLM call whenever there was history. Folding expansion into
that same call keeps each question to one small enhancement request.

This is an OPTIONAL nicety and never blocks a search:
  - quota exhausted / no key / model error  -> the question is searched exactly as typed;
  - unparsable output                       -> same (plus a plain condensation call if there is history);
  - QUERY_ENHANCE_ENABLED=false             -> the old behaviour (condense only when there is history).
"""
import functools
import logging
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, field

from backend import config
from backend.models import ChatTurn
from backend.rag.condense import condense_question
from backend.rag.history import format_history
from backend.util import UserFacingError, make_chain

log = logging.getLogger(__name__)


@dataclass
class EnhancedQuery:
    standalone: str  # the question, self-contained: what retrieval and the answer step work from
    rewrites: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    used_llm: bool = False  # the enhancement call succeeded
    note: str = ""  # a human-readable reason when enhancement was skipped or degraded

    @property
    def expansions(self) -> list[str]:
        """The EXTRA queries to search besides the question itself: each rewrite, then the keywords as one query."""
        extra = list(self.rewrites)
        if self.keywords:
            extra.append(" ".join(self.keywords))
        return extra


# Note: no literal braces anywhere below. LangChain prompt templates are f-string style, so a stray brace is an error.
_SYSTEM = (
    "You help search the spoken transcript of a YouTube video (or a playlist of them) for the passage that answers "
    "a question.\n"
    "Reply with ONE JSON object and nothing else. It has exactly three keys.\n"
    '"standalone": the latest question rewritten so it can be understood without the chat history. Resolve pronouns '
    'and references such as "it", "that" or "the second one" using the history. If it is already standalone, return '
    "it unchanged. Never answer the question.\n"
    f'"rewrites": a list of at most {config.QUERY_ENHANCE_MAX_REWRITES} alternative phrasings of the question, worded '
    "the way a speaker in the video might have said it. Prefer the technical term when the question is casual, and "
    "plain wording when the question is technical. One of them may be a sentence a speaker could say.\n"
    f'"keywords": a list of at most {config.QUERY_ENHANCE_MAX_KEYWORDS} distinctive words or short phrases, including '
    "synonyms and related technical terms, that would appear in a transcript passage about this. Never generic words "
    'such as "video", "explain" or "topic".\n'
    'Write "rewrites" and "keywords" in English even if the question is in another language: the transcripts are '
    "indexed in English."
)
_HUMAN = "Chat history (oldest first):\n{history}\n\nLatest question: {question}"


@functools.lru_cache(maxsize=1)
def _chain():
    """LCEL: prompt -> quota-aware chat call -> JSON parser. Small attempt/wait caps: this step is optional."""
    from langchain_core.output_parsers import JsonOutputParser

    return make_chain(
        _SYSTEM, _HUMAN, role="enhance", attempts=2, max_wait=config.QUERY_ENHANCE_MAX_WAIT_SECONDS
    ) | JsonOutputParser()


# A repeated question (an interviewer trying the same thing twice) should not spend the quota twice.
_cache: "OrderedDict[tuple[str, str], EnhancedQuery]" = OrderedDict()
_CACHE_MAX = 128
_cache_lock = threading.Lock()


def _norm(text: str) -> str:
    return " ".join(re.findall(r"\w+", (text or "").lower()))


def _clean(value, limit: int) -> str:
    text = " ".join(str(value or "").split()).strip("\"' ")
    return text[:limit].rsplit(" ", 1)[0] if len(text) > limit and " " in text[:limit] else text[:limit]


def _clean_list(value, max_items: int, exclude: set[str], item_limit: int) -> list[str]:
    """Up to `max_items` distinct, non-empty strings (a comma-separated string is accepted too)."""
    if isinstance(value, str):
        value = value.split(",")
    if not isinstance(value, (list, tuple)) or max_items <= 0:
        return []
    out: list[str] = []
    seen = set(exclude)
    for item in value:
        if not isinstance(item, (str, int, float)):
            continue
        text = _clean(item, item_limit)
        key = _norm(text)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= max_items:
            break
    return out


def _from_json(data, question: str, has_history: bool) -> EnhancedQuery:
    """Validate and bound whatever the model returned. Raises ValueError if it is not usable at all."""
    if not isinstance(data, dict):
        raise ValueError("the enhancement reply was not a JSON object")
    standalone = question
    if has_history:  # without history the user's own words are the best standalone question there is
        candidate = _clean(data.get("standalone"), max(300, 4 * len(question)))
        standalone = candidate or question
    already = {_norm(question), _norm(standalone)}
    rewrites = _clean_list(data.get("rewrites"), config.QUERY_ENHANCE_MAX_REWRITES, already, 200)
    keywords = _clean_list(data.get("keywords"), config.QUERY_ENHANCE_MAX_KEYWORDS, set(), 60)
    return EnhancedQuery(standalone=standalone, rewrites=rewrites, keywords=keywords, used_llm=True)


def _safe_condense(history: list[ChatTurn], question: str) -> str:
    try:
        return condense_question(history, question)
    except UserFacingError:
        return question


def enhance_query(history: list[ChatTurn], question: str) -> EnhancedQuery:
    """The standalone question plus extra search queries. Never raises for an LLM problem."""
    if not config.QUERY_ENHANCE_ENABLED:
        return EnhancedQuery(standalone=condense_question(history, question))

    key = (format_history(history, config.HISTORY_WINDOW_TURNS), _norm(question))
    with _cache_lock:
        cached = _cache.get(key)
        if cached is not None:
            _cache.move_to_end(key)
            return cached

    try:
        data = _chain().invoke({"history": key[0], "question": question})
        result = _from_json(data, question, has_history=bool(history))
    except UserFacingError as exc:  # quota used up, no key, request too large
        log.info("query enhancement skipped (%s); searching with the question as typed", exc)
        return EnhancedQuery(
            standalone=question,
            note="The language model is busy or unavailable, so your question was searched exactly as typed.",
        )
    except Exception:  # noqa: BLE001 - a malformed reply or a provider hiccup must never fail a search
        log.warning("query enhancement failed; searching without it", exc_info=True)
        return EnhancedQuery(
            standalone=_safe_condense(history, question) if history else question,
            note="Couldn't expand the question, so it was searched as typed.",
        )

    with _cache_lock:
        _cache[key] = result
        while len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)
    return result


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()
