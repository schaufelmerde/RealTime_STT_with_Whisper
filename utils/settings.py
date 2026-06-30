"""Cross-session UI settings — remembers the sidebar's controls between launches.

Streamlit's ``session_state`` resets on every process restart, so without this the sidebar
(Whisper model, languages, VAD/beam tuning, translation/assistant toggles, chosen audio
sources) would snap back to defaults each launch. We persist a small flat dict to
``user_settings.json`` and seed the widgets from it at startup, re-saving whenever a control
changes.

Best-effort and defensive — a missing or corrupt file just yields ``{}`` (every setting falls
back to its built-in default), and a write failure never crashes the UI. Same atomic
temp-then-replace write as utils/glossary.py.

Project-local and gitignored: it can hold machine-specific choices (a selected mic/app, a
device index) that shouldn't follow the repo to another machine.
"""

import json
import os
from typing import Any, Dict

# Project-root settings file, resolved relative to this module so it doesn't depend on the
# process CWD (Streamlit may be launched from anywhere). Matches utils/glossary.py.
_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "user_settings.json"
)


def load_settings() -> Dict[str, Any]:
    """The persisted settings dict, or ``{}`` if the file is missing/corrupt."""
    try:
        with open(_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except (OSError, ValueError):
        # Missing or corrupt file → no persisted settings (everything uses its default).
        pass
    return {}


def save_settings(settings: Dict[str, Any]) -> None:
    """Persist ``settings``. Best-effort: a write failure must not crash the UI thread. Write to
    a temp file then atomically replace so a crash mid-write can't truncate the saved state."""
    try:
        tmp = _PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
        os.replace(tmp, _PATH)
    except OSError:
        pass
