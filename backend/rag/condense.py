"""LangChain history-aware question condensation (spec §8).

Turn 1: "how does a hashmap work". Turn 2: "what's its time complexity".
Turn 2 is rewritten to "what is the time complexity of a hashmap" before it
ever reaches retrieval.
"""
import functools
import logging

from backend import config
from backend.models import ChatTurn
from backend.rag.history import format_history
from backend.util import UserFacingError, make_chain

log = logging.getLogger(__name__)

_SYSTEM = (
    "Given a chat history and the user's latest question, rewrite the latest question as a standalone, "
    "fully specified question that can be understood without the chat history. Resolve pronouns and "
    "references such as 'it', 'its', 'that' or 'the second one' using the history.\n"
    "Do NOT answer the question. If it is already standalone, return it unchanged. "
    "Output only the rewritten question."
)
_HUMAN = "Chat history (oldest first):\n{history}\n\nLatest question: {question}\n\nStandalone question:"


@functools.lru_cache(maxsize=1)
def _chain():
    return make_chain(_SYSTEM, _HUMAN)


def condense_question(history: list[ChatTurn], question: str) -> str:
    if not history:  # first question of the session: nothing to condense against
        return question
    try:
        out = _chain().invoke({"history": format_history(history, config.HISTORY_WINDOW_TURNS), "question": question})
    except UserFacingError:
        raise
    except Exception:
        log.exception("question condensation failed; searching with the raw question")
        return question
    return out.strip().strip('"').strip() or question
