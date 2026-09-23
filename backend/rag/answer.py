"""Grounding guards, citation logic, and the LangChain answer chain (spec §9)."""
import functools
import re

from backend import config
from backend.models import ChatTurn
from backend.rag.history import format_history
from backend.util import make_chain

# Guard 1 returns this string directly. Guard 2 tells the model to output exactly
# this phrase when the excerpts don't cover the question, and checks for it.
REFUSAL_PHRASE = "I couldn't find an answer to that in the indexed videos."

_SYSTEM = (
    "You answer questions about YouTube videos using ONLY the numbered transcript excerpts provided.\n"
    "Rules:\n"
    "- Base every statement on the excerpts. Do not use outside knowledge.\n"
    "- After each statement, cite the excerpt(s) it came from, like [1] or [1, 2]. Use only the numbers you were given.\n"
    "- Be concise: a short paragraph.\n"
    "- The excerpts are numbered in order of appearance (playlist order, then time within each video). "
    "When your answer draws on several of them, walk through them in that order.\n"
    "- The recent conversation is only there so your answer stays consistent with what was already said. "
    "It is not a source of facts.\n"
    f"- If the excerpts do not actually cover the question, reply with exactly this sentence and nothing else: {REFUSAL_PHRASE}"
)
_HUMAN = "Recent conversation (oldest first):\n{history}\n\nExcerpts:\n{context}\n\nQuestion: {question}"

_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


@functools.lru_cache(maxsize=1)
def _chain():
    return make_chain(_SYSTEM, _HUMAN)


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", "", text.lower()).split())


_REFUSAL_NORM = _norm(REFUSAL_PHRASE)


def _refusal() -> dict:
    return {"answer": REFUSAL_PHRASE, "citations": [], "grounded": False}


def _build_context(results: list[dict]) -> str:
    multi = len({r["video_id"] for r in results}) > 1  # only mention video numbers for playlists
    blocks = []
    for n, r in enumerate(results, 1):
        where = f"Video {r['position'] + 1}, " if multi else ""
        blocks.append(f"[{n}] {where}\"{r['video_title']}\" ({r['label']})\n{r['text']}")
    return "\n\n".join(blocks)


def _renumber_citations(text: str, n_results: int) -> tuple[str, dict[int, int]]:
    """Guard 3: keep only citations the text actually uses, renumbered 1..N by first use.

    Returns (rewritten text, {original_number: new_number}). Numbers that don't
    correspond to a retrieved excerpt are dropped from the text.
    """
    mapping: dict[int, int] = {}

    def rewrite(match: re.Match) -> str:
        kept: list[int] = []
        for original in (int(x) for x in match.group(1).split(",")):
            if not 1 <= original <= n_results:
                continue
            mapping.setdefault(original, len(mapping) + 1)
            if mapping[original] not in kept:
                kept.append(mapping[original])
        return "[" + ", ".join(map(str, kept)) + "]" if kept else ""

    rewritten = _CITATION.sub(rewrite, text)
    rewritten = re.sub(r"[ \t]{2,}", " ", rewritten)
    rewritten = re.sub(r"\s+([.,;:!?])", r"\1", rewritten)
    return rewritten.strip(), mapping


def generate_answer(standalone_question: str, results: list[dict], history: list[ChatTurn]) -> dict:
    """Returns {"answer", "citations", "grounded"}. `history` is the session's turns BEFORE this one."""
    # Guard 1: nothing cleared the retrieval threshold -> fixed refusal, and the LLM is never called.
    if not results:
        return _refusal()

    raw = _chain().invoke(
        {
            "history": format_history(history, config.HISTORY_WINDOW_TURNS),
            "context": _build_context(results),
            "question": standalone_question,
        }
    )

    # Guard 2: the model says the excerpts don't cover the question -> same as guard 1.
    if not raw.strip() or _REFUSAL_NORM in _norm(raw):
        return _refusal()

    # Guard 3: only citations the answer really used, renumbered 1..N.
    answer, mapping = _renumber_citations(raw, len(results))
    citations = [{**results[old - 1], "n": new} for old, new in sorted(mapping.items(), key=lambda kv: kv[1])]
    # An answer that cites nothing valid can't be tied to any excerpt, so it isn't "grounded".
    return {"answer": answer, "citations": citations, "grounded": bool(citations)}
