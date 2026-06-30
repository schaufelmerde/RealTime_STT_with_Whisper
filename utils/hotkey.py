"""OS-level push-to-ask control.

Owns the "push-to-ask" hold key. ``command_active()`` is the value the AudioCapture latches
onto each segment at VAD onset (see PRD.md "Language selection"):

* hold key down → ``True``: the utterance is a **command** for the AI assistant (it is
  transcribed like everything else, then routed to the agent instead of the transcript).
* hold key up   → ``False``: ordinary conversation that flows to the transcript/translator.

The key is captured with an OS-level hotkey (``pynput``) because Streamlit can't see keyup.
Listening is best-effort: if ``pynput`` is missing or its listener can't start (e.g.
headless / no permission), ``command_active()`` always returns ``False`` and the app simply
runs transcription-only with no push-to-ask.
"""

import threading
from typing import Optional

# Friendly labels for the hold keys we let the UI choose from. Values are pynput
# ``keyboard.Key`` attribute names, resolved lazily in start() so importing this module
# never requires pynput.
HOLD_KEYS = {
    "ctrl_r": "Right Ctrl",
    "shift_r": "Right Shift",
    "alt_r": "Right Alt",
    "ctrl_l": "Left Ctrl",
}
DEFAULT_HOLD_KEY = "ctrl_l"


class CommandHotkey:
    """Tracks the push-to-ask hold key via a global OS hotkey listener."""

    def __init__(self, hold_key: str = DEFAULT_HOLD_KEY):
        self.hold_key = hold_key
        self._held = threading.Event()
        self._listener = None
        self._available = False

    @property
    def available(self) -> bool:
        """True once an OS hotkey listener is actually running."""
        return self._available

    def is_held(self) -> bool:
        return self._held.is_set()

    def command_active(self) -> bool:
        """Latched per-segment at onset: True while the push-to-ask key is held."""
        return self._held.is_set()

    def start(self):
        try:
            from pynput import keyboard
        except Exception:
            return  # no pynput → push-to-ask unavailable; transcription still runs

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
