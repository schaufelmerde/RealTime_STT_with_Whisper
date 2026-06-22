"""OS-level push-to-force language control.

Owns the active language pair and the "hold key" state. ``forced_lang()`` is the value
the AudioCapture latches onto each segment at VAD onset (see PRD.md "Language selection"):

* hold key down → the **secondary** language (deterministic force, for when you code-switch)
* hold key up   → ``None`` (Transcriber does constrained auto-detect over the pair — the
  hands-free default that covers the other party, who never touches a key)

The key is captured with an OS-level hotkey (``pynput``) because Streamlit can't see
keyup. Listening is best-effort: if ``pynput`` is missing or its listener can't start
(e.g. headless / no permission), ``forced_lang()`` simply always returns ``None`` and the
pipeline still runs in auto-detect-only mode.
"""

import threading
from typing import Optional, Tuple

# Friendly labels for the hold keys we let the UI choose from. Values are pynput
# ``keyboard.Key`` attribute names, resolved lazily in start() so importing this module
# never requires pynput.
HOLD_KEYS = {
    "ctrl_r": "Right Ctrl",
    "shift_r": "Right Shift",
    "alt_r": "Right Alt",
    "ctrl_l": "Left Ctrl",
}
DEFAULT_HOLD_KEY = "ctrl_r"


class LanguageController:
    def __init__(
        self,
        primary: str = "en",
        secondary: str = "ko",
        hold_key: str = DEFAULT_HOLD_KEY,
    ):
        self.primary = primary.lower()
        self.secondary = secondary.lower()
        self.hold_key = hold_key
        self._held = threading.Event()
        self._listener = None
        self._available = False

    @property
    def pair(self) -> Tuple[str, str]:
        return (self.primary, self.secondary)

    @property
    def available(self) -> bool:
        """True once an OS hotkey listener is actually running."""
        return self._available

    def is_held(self) -> bool:
        return self._held.is_set()

    def forced_lang(self) -> Optional[str]:
        """Latched per-segment at onset: secondary while held, else None (auto-detect)."""
        return self.secondary if self._held.is_set() else None

    def start(self):
        try:
            from pynput import keyboard
        except Exception:
            return  # no pynput → auto-detect-only fallback

        target = getattr(keyboard.Key, self.hold_key, None)
        if target is None:
            return

        def on_press(key):
            if key == target:
                self._held.set()

        def on_release(key):
            if key == target:
                self._held.clear()

        try:
            self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
            self._listener.daemon = True
            self._listener.start()
            self._available = True
        except Exception:
            self._listener = None
            self._available = False

    def stop(self):
        if self._listener:
            try:
                self._listener.stop()
            except Exception:
                pass
            self._listener = None
        self._held.clear()
        self._available = False
