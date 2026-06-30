"""Proper-noun / custom-vocabulary store — biases ASR + LLM toward names they'd otherwise miss.

This is **not** model training. ``faster-whisper`` stays a frozen pretrained model; the terms
here are fed two ways, both instant and reversible:

* **LLM cleanup / translation / report (primary, always on)** — the terms are handed to Gemini as
  a "keep these exact spellings" glossary (``utils/translator.py``, ``utils/reporter.py``). Gemini
  sees the ASR text and only fixes a spelling that's already there, so it can't invent a term from
  nothing — making this the safe, default path whenever a Gemini stage runs.
* **ASR decode (opt-in, OFF by default)** — joined into a ``hotwords`` hint string passed to
  ``model.transcribe(...)`` (see ``utils/transcriber.py``). Whisper tokenizes the hint into the
  decoder's *previous-text* context, re-weighting its spelling prior. In practice this conditions
  the decode too strongly: it **over-forces** the terms, emitting them on short/quiet clips even
  when they weren't spoken — the same conditioning-hallucination as ``initial_prompt``. So it's
  gated behind a default-off UI toggle (``hotwords_gate`` in main.py), for the occasional name
  Whisper keeps mangling. The hint is also size-capped (faster-whisper truncates it to half the
  context), so keep the list focused. No gradients, no weights touched either way.

Persisted as a flat JSON list of strings in ``vocabulary.json`` (gitignored — may hold personal
or company names). Thread-safe: the Streamlit UI writes it while the Transcriber's worker thread
reads ``hotwords()`` on every decode, so edits take effect on the next utterance with no restart.
"""

import json
import os
import threading
from typing import List, Optional

# Project-root vocabulary file, resolved relative to this module so it doesn't depend on the
# process CWD (Streamlit may be launched from anywhere).
_DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vocabulary.json"
)


class Glossary:
    """A small, thread-safe list of proper nouns / custom terms.

    One instance is shared between the Streamlit UI (the writer) and the pipeline worker
    threads (readers). All access is guarded by a lock; readers get an immutable snapshot.
    """

    def __init__(self, path: Optional[str] = None):
        self.path = path or _DEFAULT_PATH
        self._lock = threading.Lock()
        self._terms: List[str] = self._load()

    def _load(self) -> List[str]:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return [str(t).strip() for t in data if str(t).strip()]
        except (OSError, ValueError):
            # Missing or corrupt file → empty glossary (the feature simply does nothing).
            pass
        return []

    def terms(self) -> List[str]:
        """A snapshot copy of the current terms (safe to read from any thread)."""
        with self._lock:
            return list(self._terms)

    def set_terms(self, terms: List[str]) -> None:
        """Replace the term list (blanks dropped, de-duped case-insensitively, order kept) and
        persist. The UI calls this when the user edits the vocabulary text area."""
        cleaned: List[str] = []
        seen = set()
        for t in terms:
            t = (t or "").strip()
            key = t.lower()
            if t and key not in seen:
                seen.add(key)
                cleaned.append(t)
        with self._lock:
            self._terms = cleaned
            self._save_locked()

    def hotwords(self) -> str:
        """The terms as one hint string for faster-whisper's ``hotwords`` param.

        Comma-separated so the decoder reads them as distinct items rather than one run-on
        phrase. Empty string when there are no terms (callers treat that as "no biasing").
        """
        with self._lock:
            return ", ".join(self._terms)

    def as_prompt_block(self, label: str = "Known names/terms") -> str:
        """A one-line glossary block for an LLM prompt, or ``""`` if empty. Tells the model to
        preserve the canonical spelling of names it might otherwise "correct"."""
        with self._lock:
            if not self._terms:
                return ""
            joined = ", ".join(self._terms)
        return f"{label} — when these occur, keep their exact spelling: {joined}"

    def _save_locked(self) -> None:
        # Best-effort persistence; a write failure must not crash the UI thread. Write to a temp
        # file then atomically replace so a crash mid-write can't truncate the saved list.
        try:
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._terms, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)
        except OSError:
            pass
