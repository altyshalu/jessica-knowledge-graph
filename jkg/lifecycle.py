"""Public lifecycle types for explicit memory maintenance."""

from dataclasses import dataclass
from enum import Enum


class MemoryAction(str, Enum):
    ADD = "add"
    UPDATE = "update"
    SUPERSEDE = "supersede"
    DELETE = "delete"
    NOOP = "noop"


class FeedbackSignal(str, Enum):
    CONFIRM = "confirm"
    REJECT = "reject"
    USEFUL = "useful"
    NOT_USEFUL = "not_useful"


@dataclass(frozen=True)
class ContextBudget:
    max_chars: int = 12000
    profile_fraction: float = 0.20
    semantic_fraction: float = 0.30
    factual_fraction: float = 0.30
    episodic_fraction: float = 0.15
    procedural_fraction: float = 0.05

    def limit_for(self, layer: str) -> int:
        fraction = getattr(self, f"{layer}_fraction", 0.0)
        return max(0, int(self.max_chars * fraction))
