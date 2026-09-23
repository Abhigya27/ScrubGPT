"""In-memory session/chat-history store (spec §8). Allowed to reset on restart."""
from backend.models import ChatTurn

_SESSIONS: dict[str, list[ChatTurn]] = {}


def get_history(session_id: str) -> list[ChatTurn]:
    return list(_SESSIONS.get(session_id, []))


def add_turn(session_id: str, turn: ChatTurn) -> None:
    _SESSIONS.setdefault(session_id, []).append(turn)


def format_history(turns: list[ChatTurn], window: int) -> str:
    """Last `window` turns as plain text, for the condensation and answer prompts."""
    recent = turns[-window:] if window > 0 else []
    if not recent:
        return "(none)"
    lines = []
    for t in recent:
        lines.append(f"User: {t.question}")
        lines.append(f"Assistant: {t.answer or '(showed matching video timestamps)'}")
    return "\n".join(lines)
