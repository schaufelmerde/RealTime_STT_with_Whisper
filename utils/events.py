"""The seam between the local pipeline and its consumers.

A single ``TranscriptEvent`` shape flows through the whole pipeline; the
``TranscriptBus`` fans enriched events out to any number of subscribers — the
Streamlit UI today, an agent orchestrator later (see PRD.md, Stretch M1). Adding a
consumer is a new ``subscribe()`` call, not a pipeline change.
"""

import threading
import uuid
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional


@dataclass
class TranscriptEvent:
    """One utterance as it moves through the pipeline.

    The Transcriber produces it raw (``text`` + ``source_lang`` + ``lang_source``); the
    Translator enriches it (``clean_text`` + ``translation`` + ``target_lang``). ``lang_source``
    records how the language was decided — ``"forced"`` (hold key) vs ``"detected"``
    (constrained auto-detect) — so the UI/QA can tell a missed key-press from a detection
    miss. ``speaker`` is
    unused in the MVP but kept on the schema so speaker attribution (PRD.md M2) can be
    added without a migration.
    """

    text: str
    source_lang: str
    ts_start: float
    ts_end: float
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    lang_source: Optional[str] = None  # "forced" (hold key) | "detected" (constrained auto-detect)
    speaker: Optional[str] = None
    clean_text: Optional[str] = None
    translation: Optional[str] = None
    target_lang: Optional[str] = None

    @property
    def display_text(self) -> str:
        """The original line to show — cleaned if available, raw otherwise."""
        return self.clean_text or self.text

    def to_dict(self) -> dict:
        return asdict(self)


class TranscriptBus:
    """Minimal thread-safe publish/subscribe.

    Subscribers register a callback; ``publish()`` fans an event out to all of them.
    Callbacks run on the publisher's thread, so they must be cheap and non-blocking
    (append to a queue/deque, enqueue work — never do I/O inline).
    """

    def __init__(self) -> None:
        self._subscribers: list[Callable[[TranscriptEvent], None]] = []
        self._lock = threading.Lock()

    def subscribe(self, callback: Callable[[TranscriptEvent], None]) -> Callable[[], None]:
        """Register ``callback``; returns an ``unsubscribe()`` to remove it."""
        with self._lock:
            self._subscribers.append(callback)

        def unsubscribe() -> None:
            with self._lock:
                if callback in self._subscribers:
                    self._subscribers.remove(callback)

        return unsubscribe

    def publish(self, event: TranscriptEvent) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        for cb in subscribers:
            try:
                cb(event)
            except Exception:
                # A misbehaving subscriber must not take down the pipeline.
                pass
