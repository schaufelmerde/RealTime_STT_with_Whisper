"""Per-meeting session persistence — save/load/list transcripts so a meeting survives a restart
or crash, and the sidebar can manage multiple meetings like chat threads.

Each session is one JSON file in a project-local ``sessions/`` dir (gitignored — a transcript can
hold names/PII). Mirrors utils/settings.py: atomic temp-then-replace writes, best-effort, and
never raises into the Streamlit UI thread.

A session file holds::

    {id, title, created, updated, events: [TranscriptEvent.to_dict()...], edited_text|null, report|null}

``events`` are TranscriptEvent dicts (see utils/events.py — ``display_text`` is a property, so it
isn't stored; ``TranscriptEvent(**d)`` reconstructs cleanly). ``edited_text`` is the user's
freeform-edited transcript once they've edited it (from then on the source of truth for the
report). ``report`` is the generated bilingual write-up flattened to ``{title, report_en,
report_ko}`` so loading doesn't depend on pydantic.
"""

import datetime
import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from utils.events import TranscriptEvent

# Project-root sessions dir, resolved relative to this module (not the CWD — Streamlit may be
# launched from anywhere), matching utils/settings.py.
_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sessions")


def _ensure_dir() -> None:
    try:
        os.makedirs(_DIR, exist_ok=True)
    except OSError:
        pass


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def _path(session_id: str) -> str:
    # Our ids are alnum hex, but sanitize regardless so a stray id can't escape the dir.
    safe = "".join(c for c in str(session_id) if c.isalnum())
    return os.path.join(_DIR, f"{safe}.json")


def default_title(ts: Optional[float] = None) -> str:
    """A timestamp-based title for a fresh session, e.g. ``Meeting Jun 30, 14:05``."""
    t = datetime.datetime.fromtimestamp(ts if ts is not None else time.time())
    return t.strftime("Meeting %b %d, %H:%M")


def new_session(title: Optional[str] = None) -> Dict[str, Any]:
    """A fresh, empty session dict (not yet written to disk)."""
    now = time.time()
    return {
        "id": new_id(),
        "title": title or default_title(now),
        "created": now,
        "updated": now,
        "events": [],
        "edited_text": None,
        "report": None,
    }


# --- (de)serialization helpers -------------------------------------------------------
def event_from_dict(d: dict) -> TranscriptEvent:
    """Reconstruct a TranscriptEvent from a stored dict, tolerating missing/unknown keys so an
    older or newer file still loads (the four required fields fall back to safe defaults)."""
    fields = TranscriptEvent.__dataclass_fields__
    kwargs = {k: v for k, v in d.items() if k in fields}
    kwargs.setdefault("text", d.get("text", "") or "")
    kwargs.setdefault("source_lang", d.get("source_lang", "") or "")
    kwargs.setdefault("ts_start", float(d.get("ts_start") or 0.0))
    kwargs.setdefault("ts_end", float(d.get("ts_end") or 0.0))
    return TranscriptEvent(**kwargs)


class StoredReport:
    """Lightweight stand-in for reporter.Report so the UI's attribute access (``report.title`` /
    ``.report_en`` / ``.report_ko``) works on a report loaded from disk without needing pydantic."""

    def __init__(self, title: str, report_en: str, report_ko: str):
        self.title = title
        self.report_en = report_en
        self.report_ko = report_ko


def report_to_dict(report) -> Optional[dict]:
    """Flatten a report (pydantic Report, StoredReport, or dict) to a plain dict for storage."""
    if report is None:
        return None
    if isinstance(report, dict):
        return {"title": report.get("title", ""), "report_en": report.get("report_en", ""),
                "report_ko": report.get("report_ko", "")}
    try:
        return {"title": report.title, "report_en": report.report_en,
                "report_ko": report.report_ko}
    except AttributeError:
        return None


def report_from_dict(d) -> Optional[StoredReport]:
    if not d:
        return None
    # Self-heal reports saved before the report-formatting fix: older files can hold Markdown with
    # literal "\n" sequences instead of real newlines (a structured-output quirk). Normalize on load
    # so they render correctly and get rewritten clean on the next save. Lazy import keeps sessions.py
    # free of a reporter dependency at import time; fall back to the raw strings if it's unavailable.
    try:
        from utils.reporter import _normalize_report_text as _norm
    except Exception:
        _norm = lambda s: s
    return StoredReport(_norm(d.get("title", "")), _norm(d.get("report_en", "")),
                        _norm(d.get("report_ko", "")))


# --- disk I/O ------------------------------------------------------------------------
def exists(session_id: str) -> bool:
    return os.path.exists(_path(session_id))


def save(session: Dict[str, Any]) -> None:
    """Persist a session dict (must contain ``id``). Best-effort and atomic: write a temp file
    then replace, so a crash mid-write can't truncate the saved meeting. Stamps ``updated``."""
    sid = session.get("id")
    if not sid:
        return
    _ensure_dir()
    data = dict(session)
    data["updated"] = time.time()
    path = _path(sid)
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except OSError:
        pass


def load(session_id: str) -> Optional[Dict[str, Any]]:
    """The stored session dict, or ``None`` if it's missing/corrupt."""
    try:
        with open(_path(session_id), "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except (OSError, ValueError):
        pass
    return None


def delete(session_id: str) -> None:
    try:
        os.remove(_path(session_id))
    except OSError:
        pass


def list_meta() -> List[Dict[str, Any]]:
    """Lightweight metadata for every saved session (id/title/created/updated/n_events), newest
    first. Reads each file (one JSON each) but returns only metadata — fine at the dozens-of-
    meetings scale this is built for. The UI caches the result and mutates it in place, so this
    scans disk only on first load / explicit refresh."""
    _ensure_dir()
    out: List[Dict[str, Any]] = []
    try:
        names = os.listdir(_DIR)
    except OSError:
        return out
    for name in names:
        if not name.endswith(".json"):
            continue
        data = load(name[:-5])
        if not data or not data.get("id"):
            continue
        out.append({
            "id": data["id"],
            "title": data.get("title") or "Untitled",
            "created": data.get("created", 0.0),
            "updated": data.get("updated", 0.0),
            "n_events": len(data.get("events") or []),
        })
    out.sort(key=lambda m: m.get("updated", 0.0), reverse=True)
    return out
