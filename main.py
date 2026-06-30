import datetime
import html
import json
import math
import os
import queue
import time
import zlib

import streamlit as st

from utils.gemini import get_api_key
from utils.hotkey import HOLD_KEYS, DEFAULT_HOLD_KEY
from utils.settings import load_settings, save_settings
import utils.sessions as sessions

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load GEMINI_API_KEY (and friends) from a .env if present. Optional dependency —
# the app still runs transcription-only without it.
try:
    from dotenv import load_dotenv
    load_dotenv()                              # .env
    load_dotenv(".env.local", override=True)   # .env.local (gitignored secrets) wins
except Exception:
    pass

PAGE_TITLE = "Realtime STT — KO↔EN Assistant"
st.set_page_config(page_title=PAGE_TITLE, page_icon="◎", layout="wide")


# --- Studio Dark styling -------------------------------------------------------------
# Streamlit's own widgets are themed via .streamlit/config.toml; this stylesheet covers the
# custom header, status pill, config strip, cards, transcript bubbles, and assistant panel.
# Inter/JetBrains Mono load with display:swap (no FOIT). Icons use Streamlit's *bundled*
# "Material Symbols Rounded" font: native icon=":material/…:" glyphs AND the expander caret
# render inside <span data-testid="stIconMaterial">; our custom markup uses ._ms / .ms.
# IMPORTANT: the broad Inter rule below also matches Streamlit's icon spans (they carry an
# st-emotion-cache-* class), so we re-assert the icon font on stIconMaterial with !important —
# without it the ligatures show as raw text ("play_arrow", "expand_more") over the labels.
# We also deliberately do NOT @import a network "Material Symbols Rounded" (same family name):
# it shadows the bundled @font-face and flashes raw text on load. Motion is gated behind
# prefers-reduced-motion.
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root{
  /* surfaces & lines */
  --bg:#0b0e14; --surface:#121722; --surface-2:#161d2a; --surface-3:#1b2433;
  --border:#222b3a; --border-2:#2d3a4e;
  /* text tiers (all AA on --bg / --surface) */
  --text:#e8eef6; --dim:#9fadc0; --faint:#828fa3;
  /* accents */
  --brand:#2dd4bf; --brand-soft:rgba(45,212,191,.12); --brand-line:rgba(45,212,191,.40);
  --live:#34d399; --listen:#fbbf24; --idle:#586273; --violet:#a78bfa; --danger:#f87171;
  --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 24px -12px rgba(0,0,0,.6);
}

/* base typography + app shell */
html, body, .stApp, [class*="st-"], button, input, textarea, select{
  font-family:'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;}
.stApp{background:
  radial-gradient(1100px 520px at 18% -8%, rgba(45,212,191,.06), transparent 60%),
  radial-gradient(900px 480px at 110% 0%, rgba(167,139,250,.05), transparent 55%), var(--bg);}
.block-container{padding-top:1.0rem; padding-bottom:3.5rem; max-width:1060px;}
header[data-testid="stHeader"]{background:transparent;}
.ms{font-family:'Material Symbols Rounded'; font-weight:normal; font-style:normal;
  line-height:1; vertical-align:middle; -webkit-font-feature-settings:'liga';
  font-feature-settings:'liga'; -webkit-font-smoothing:antialiased; user-select:none;}
.mono{font-family:'JetBrains Mono', ui-monospace, monospace; font-variant-numeric:tabular-nums;}
/* Antidote to the broad Inter rule above: keep Material Symbols on Streamlit's native icons.
   stIconMaterial covers button glyphs + the expander caret; the icon passed to st.expander
   renders in its OWN span (stExpanderIcon*), so both testids must be re-asserted. Without
   !important the Inter override wins and the ligature name renders as literal text. */
[data-testid="stIconMaterial"],
[data-testid^="stExpanderIcon"]{
  font-family:'Material Symbols Rounded' !important; font-weight:normal !important;
  font-style:normal !important; -webkit-font-feature-settings:'liga' !important;
  font-feature-settings:'liga' !important; -webkit-font-smoothing:antialiased;}

/* ---- header bar ---- */
.app-header{display:flex; align-items:center; justify-content:space-between; gap:16px;
  padding:15px 20px; margin-bottom:14px; border-radius:18px; box-shadow:var(--shadow);
  background:linear-gradient(135deg,#151c28 0%,#0e1320 100%); border:1px solid var(--border);}
.brand{display:flex; align-items:center; gap:13px; min-width:0;}
.brand-mark{display:grid; place-items:center; width:40px; height:40px; border-radius:12px;
  color:#04120f; background:linear-gradient(135deg,#5eead4,#2dd4bf);
  box-shadow:0 6px 18px -6px rgba(45,212,191,.55);}
.brand-mark .ms{font-size:24px; font-variation-settings:'FILL' 1;}
.brand-txt{display:flex; flex-direction:column; gap:1px; min-width:0;}
.brand-title{font-size:1.06rem; font-weight:700; letter-spacing:.2px; color:var(--text); line-height:1.2;}
.brand-sub{color:var(--faint); font-weight:500; font-size:.76rem; letter-spacing:.2px;}
.status-pill{display:inline-flex; align-items:center; gap:8px; font-size:.7rem; font-weight:700;
  padding:7px 15px; border-radius:999px; border:1px solid var(--border-2); color:var(--dim);
  background:var(--surface-2); text-transform:uppercase; letter-spacing:1px; white-space:nowrap;}
.status-pill .dot{width:8px; height:8px; border-radius:50%; background:var(--idle); flex:none;}
.status-pill.live{color:var(--live); border-color:rgba(52,211,153,.45); background:rgba(52,211,153,.10);}
.status-pill.live .dot{background:var(--live); box-shadow:0 0 0 4px rgba(52,211,153,.16); animation:breathe 2.4s ease-in-out infinite;}
.status-pill.listen{color:var(--listen); border-color:rgba(251,191,36,.5); background:rgba(251,191,36,.12);}
.status-pill.listen .dot{background:var(--listen); box-shadow:0 0 0 4px rgba(251,191,36,.18); animation:breathe 1.1s ease-in-out infinite;}
@keyframes breathe{0%,100%{transform:scale(1); opacity:1}50%{transform:scale(1.3); opacity:.65}}

/* ---- config strip (at-a-glance settings) ---- */
.cfg-strip{display:flex; flex-wrap:wrap; gap:7px; margin:0 2px 18px;}
.cfg{display:inline-flex; align-items:center; gap:6px; font-size:.74rem; font-weight:500;
  color:var(--dim); background:var(--surface); border:1px solid var(--border);
  padding:5px 11px; border-radius:9px;}
.cfg .ms{font-size:15px; color:var(--faint);}
.cfg b{color:var(--text); font-weight:600;}
.cfg.on{color:var(--brand); border-color:var(--brand-line); background:var(--brand-soft);}
.cfg.on .ms{color:var(--brand);}
.cfg.off{color:var(--faint);}

/* ---- phone status bar (shown in the main panel only when a Phone source is selected) ---- */
.phone-bar{display:flex; align-items:center; gap:11px; margin:0 2px 16px; padding:9px 14px;
  border-radius:11px; background:var(--surface); border:1px solid var(--border);}
.phone-bar > .ms{color:var(--faint); font-size:18px; flex:none;}
.phone-bar.on > .ms{color:var(--brand);}
.phone-bar .pb-label{font-weight:700; font-size:.8rem; color:var(--text); letter-spacing:.3px; flex:none;}
.phone-bar .pb-state{display:inline-flex; align-items:center; gap:6px; font-size:.72rem;
  font-weight:600; color:var(--faint); flex:none;}
.phone-bar .pb-state .dot{width:7px; height:7px; border-radius:50%; background:var(--idle);}
.phone-bar.on .pb-state{color:var(--live);}
.phone-bar.on .pb-state .dot{background:var(--live); box-shadow:0 0 0 3px rgba(52,211,153,.16);
  animation:breathe 1.8s ease-in-out infinite;}
.phone-bar .lvl{flex:1; max-width:260px; margin-left:auto;}

/* ---- section labels ---- */
.section{display:flex; align-items:center; gap:8px; margin:22px 0 10px;
  font-size:.74rem; font-weight:700; text-transform:uppercase; letter-spacing:1.3px; color:var(--dim);}
.section .ms{font-size:18px; color:var(--brand);}

/* ---- cards (bordered containers) ---- */
div[data-testid="stVerticalBlockBorderWrapper"]{
  background:var(--surface); border:1px solid var(--border) !important; border-radius:14px;
  box-shadow:var(--shadow);}

/* ---- buttons ---- */
.stButton>button{border-radius:10px; font-weight:600; border:1px solid var(--border-2);
  transition:transform .12s ease, background .15s ease, border-color .15s ease, box-shadow .15s ease;}
.stButton>button:hover:not(:disabled){border-color:var(--brand-line); transform:translateY(-1px);}
.stButton>button:active:not(:disabled){transform:translateY(0);}
.stButton>button:focus-visible{outline:2px solid var(--brand); outline-offset:2px;}
.stButton>button[kind="primary"]{box-shadow:0 6px 18px -8px rgba(45,212,191,.6);}

/* sidebar */
[data-testid="stSidebar"]{border-right:1px solid var(--border);}
[data-testid="stSidebar"] .section{margin-top:6px;}
/* Selectboxes are pickers, not text fields. BaseWeb renders a searchable <input> inside
   the control, which shows an I-beam cursor + blinking caret and reads as an editable
   field. Hide the caret and force a pointer cursor so it looks and feels like a dropdown
   (type-to-filter still works for power users — it's just no longer disguised as a textbox). */
[data-baseweb="select"] input{caret-color:transparent !important; cursor:pointer !important;}
[data-baseweb="select"] > div{cursor:pointer;}

/* chosen audio-source rows: compact label, intentionally smaller than the section labels
   above them (the default markdown body size read as oversized for a list item). */
.src-row{display:flex; align-items:center; gap:7px; font-size:.78rem; line-height:1.35; color:var(--dim);}
.src-row .ms{color:var(--brand); flex:none;}
/* truncate a long device/app name to one line with an ellipsis (full text on hover via title=)
   instead of wrapping — keeps each source row a single line. min-width:0 lets the flex child
   shrink below its content width so text-overflow can actually kick in. */
.src-row span{flex:1 1 auto; min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;}
.src-row b{color:var(--text); font-weight:600;}
.src-row.na{color:var(--faint);}
.src-row.na i{font-style:italic;}
/* muted: source stays in the list but is silenced — dim it and grey the category icon so the
   list reads at a glance which sources are live. */
.src-row.muted{opacity:.5;}
.src-row.muted .ms{color:var(--faint);}
.src-row.muted b{color:var(--dim);}
/* per-source input-level meter (a thin track under each source row): its fill tracks the
   live RMS level — whether recording (the capture feeds it) or idle (a lightweight monitor
   does) — so you can see at a glance that audio is coming through and roughly how loud. Fill
   color steps green→amber→red toward clipping. */
.lvl{height:4px; border-radius:3px; background:var(--surface-3); overflow:hidden; width:100%;}
.lvl > i{display:block; height:100%; border-radius:3px;
  transition:width .18s linear, background .2s linear;}
/* mute / gear / remove on each source row: bare icon with padding as the clickable surface — no
   button box. Scoped by the per-widget st-key-<key> class so the "Add source" button (no key)
   keeps its normal styling; selectbox keys like st-key-edit_tgt_* contain no .stButton, so the
   descendant rule can't leak onto them. NOTE: use a descendant combinator (.stButton button),
   NOT .stButton>button — when a button has a help tooltip Streamlit inserts a
   div[data-testid=stTooltipHoverTarget] between .stButton and <button>, so the direct-child
   form matches nothing on these (all three have help). */
[data-testid="stSidebar"] [class*="st-key-mute_"] .stButton button,
[data-testid="stSidebar"] [class*="st-key-edit_"] .stButton button,
[data-testid="stSidebar"] [class*="st-key-rm_"] .stButton button{
  border:none !important; background:transparent !important; box-shadow:none !important;
  border-radius:8px; padding:5px !important; min-height:0; color:var(--faint);
  display:inline-flex; align-items:center; justify-content:center; line-height:1;
  transition:transform .12s ease, color .12s ease;}
/* the icon glyph itself: kill the Material font's intrinsic line box so it sits dead-center
   in the padding box (otherwise descender space pushes it visually low). */
[data-testid="stSidebar"] [class*="st-key-mute_"] .stButton button p,
[data-testid="stSidebar"] [class*="st-key-edit_"] .stButton button p,
[data-testid="stSidebar"] [class*="st-key-rm_"] .stButton button p{
  display:flex; align-items:center; line-height:1; margin:0;}
/* hover: don't paint a box behind the icon (the tooltip wrapper made it read as a stray
   container). Just grow the glyph a touch and brighten it to white. */
[data-testid="stSidebar"] [class*="st-key-mute_"] .stButton button:hover:not(:disabled),
[data-testid="stSidebar"] [class*="st-key-edit_"] .stButton button:hover:not(:disabled),
[data-testid="stSidebar"] [class*="st-key-rm_"] .stButton button:hover:not(:disabled){
  color:#fff; background:transparent !important; border:none !important; transform:scale(1.18);}

/* tabs (report EN/KO) */
[data-baseweb="tab-list"]{gap:4px;}
[data-baseweb="tab"]{border-radius:8px 8px 0 0;}

/* ---- assistant cards ---- */
.ask-q{display:flex; align-items:flex-start; gap:9px; font-weight:600; color:var(--text);
  font-size:.96rem; line-height:1.45;}
.ask-q .ms{font-size:19px; color:var(--violet); margin-top:1px; flex:none;}
.ask-divider{height:1px; background:var(--border); margin:11px 0 9px;}
.ask-pending{display:flex; align-items:center; gap:9px; color:var(--faint); font-size:.82rem;
  font-weight:500; margin-bottom:9px;}
.ask-pending .ms{font-size:17px; color:var(--violet); animation:spin 1s linear infinite;}
.skel{height:11px; border-radius:6px; margin:7px 0;
  background:linear-gradient(90deg,var(--surface-2) 25%,var(--surface-3) 37%,var(--surface-2) 63%);
  background-size:400% 100%; animation:shimmer 1.4s ease infinite;}
@keyframes spin{to{transform:rotate(360deg)}}
@keyframes shimmer{0%{background-position:100% 0}100%{background-position:-100% 0}}

/* ---- transcript bubbles ---- */
.stt-msg{position:relative; background:var(--surface); border:1px solid var(--border);
  border-left:3px solid var(--brand); border-radius:12px; padding:11px 15px 12px; margin:9px 0;
  transition:border-color .15s ease;}
.stt-msg:hover{border-color:var(--border-2);}
.stt-meta{display:flex; align-items:center; gap:8px; margin-bottom:6px; flex-wrap:wrap;}
.stt-badge{display:inline-flex; align-items:center; gap:5px; font-size:.66rem; font-weight:700;
  padding:2px 9px; border-radius:999px; border:1px solid; text-transform:uppercase; letter-spacing:.5px;}
.stt-badge .dot{width:6px; height:6px; border-radius:50%;}
.stt-chip{font-size:.64rem; font-weight:600; color:var(--dim); background:var(--surface-2);
  border:1px solid var(--border); padding:2px 7px; border-radius:6px; letter-spacing:.5px;}
.stt-chip.alt{color:var(--brand); border-color:var(--brand-line); background:var(--brand-soft);}
.stt-lock{font-size:14px; color:var(--listen);}
.stt-time{margin-left:auto; font-size:.68rem; color:var(--faint);}
.stt-text{color:var(--text); font-size:.96rem; line-height:1.6;}
.stt-trans{color:var(--dim); font-size:.87rem; line-height:1.55; margin-top:8px; padding-top:8px;
  border-top:1px dashed var(--border); display:flex; gap:8px; align-items:baseline; flex-wrap:wrap;}
/* live (interim) preview line — tentative, in-progress decode shown before the segment ends */
.stt-msg.partial{border-style:dashed; background:transparent;}
.stt-msg.partial .stt-text{color:var(--dim); font-style:italic;}
.stt-live{display:inline-flex; align-items:center; gap:5px; font-size:.62rem; font-weight:700;
  letter-spacing:.6px; text-transform:uppercase; color:var(--live);}
.stt-live .dot{width:6px; height:6px; border-radius:50%; background:var(--live);
  box-shadow:0 0 0 3px rgba(52,211,153,.16); animation:breathe 1.4s ease-in-out infinite;}

/* contained transcript: a fixed-height region that clips overflow and shows its own
   scrollbar (so the page doesn't grow without bound). A script pins it to the bottom on
   every content change so the latest line / live preview is always in view. */
.stt-scroll{max-height:62vh; overflow-y:auto; overflow-x:hidden; padding-right:6px;}

/* ---- designed empty states ---- */
.empty{display:flex; flex-direction:column; align-items:center; text-align:center; gap:4px;
  padding:30px 20px; background:var(--surface); border:1px dashed var(--border-2);
  border-radius:14px; color:var(--faint);}
.empty .ico{display:grid; place-items:center; width:46px; height:46px; border-radius:13px;
  background:var(--surface-2); border:1px solid var(--border); margin-bottom:6px;}
.empty .ico .ms{font-size:24px; color:var(--brand);}
.empty .et{color:var(--text); font-weight:600; font-size:.92rem;}
.empty .es{font-size:.83rem; max-width:42ch; line-height:1.5;}

/* custom scrollbar */
::-webkit-scrollbar{width:11px; height:11px;}
::-webkit-scrollbar-thumb{background:var(--border-2); border-radius:8px;
  border:3px solid var(--bg);}
::-webkit-scrollbar-thumb:hover{background:#3a4960;}
::-webkit-scrollbar-track{background:transparent;}

@media (prefers-reduced-motion: reduce){
  *,*::before,*::after{animation-duration:.001ms !important; animation-iteration-count:1 !important;
    transition-duration:.001ms !important;}
}
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


# --- Small markup helpers ------------------------------------------------------------
def _ms(name: str, size: int = 18, color: str | None = None, fill: int = 0) -> str:
    """Render a Material Symbols (Rounded) vector icon for use inside custom HTML."""
    style = f"font-size:{size}px;font-variation-settings:'FILL' {fill}"
    if color:
        style += f";color:{color}"
    return f"<span class='ms' style='{style}'>{name}</span>"


def _section(icon: str, title: str) -> None:
    st.markdown(f"<div class='section'>{_ms(icon, 18)}<span>{title}</span></div>",
                unsafe_allow_html=True)


def _empty(icon: str, title: str, sub: str) -> None:
    st.markdown(
        f"<div class='empty'><div class='ico'>{_ms(icon, 24)}</div>"
        f"<div class='et'>{title}</div><div class='es'>{sub}</div></div>",
        unsafe_allow_html=True,
    )


def _level_frac(rms: float) -> float:
    """Map a linear RMS amplitude to a 0..1 meter fraction on a dB scale (the way meters and
    ears both work): ~-60 dBFS floor → empty, 0 dBFS → full, so even quiet speech moves the
    bar while leaving headroom before it pins to the top."""
    if rms <= 1e-6:
        return 0.0
    db = 20 * math.log10(rms)
    return max(0.0, min(1.0, (db + 60.0) / 60.0))


def _level_meter(frac: float, dim: bool = False) -> str:
    """A thin horizontal input-level bar (0..1). Fill color steps green→amber→red so loudness
    reads at a glance; `dim` greys it for a muted row."""
    frac = max(0.0, min(1.0, frac))
    color = "var(--danger)" if frac >= 0.9 else "var(--listen)" if frac >= 0.7 else "var(--live)"
    op = ";opacity:.45" if dim else ""
    return (f"<div class='lvl' style='margin-top:5px{op}'>"
            f"<i style='width:{frac * 100:.0f}%;background:{color}'></i></div>")


def _open_level_monitor(src):
    """Open and start a lightweight input-level monitor for one source. Returns the started
    monitor, or None if the device won't open right now (caller retries on a later run, or shows
    an empty bar). Opening a system-audio loopback reader can take ~1s, so a caller doing it in
    response to a click should wrap it in a spinner."""
    from utils.audio import LevelMonitor, make_reader
    try:
        mon = LevelMonitor(make_reader(src))
        mon.start()
        return mon
    except Exception:
        return None  # device won't open this instant → retry on a later run


# Live-UI poll cadence for the server-side `time.sleep; st.rerun()` loop at the bottom. A no-op
# poll re-executes the whole script; a meeting is mostly silence, so we stay at the responsive
# _POLL_FAST_S tick only while something is live and back off to _POLL_IDLE_S otherwise.
# _ACTIVE_TAIL_S keeps full cadence for a moment after the last received line so continuous speech
# doesn't drop to the slow tick between utterances. The idle branch (not recording, but a level
# monitor open) also ticks at _POLL_FAST_S to animate the sidebar meters.
_POLL_FAST_S, _POLL_IDLE_S, _ACTIVE_TAIL_S = 0.3, 1.0, 2.0


def _reconcile_level_monitors(by_key, keys, muted_keys):
    """Keep a lightweight input-level monitor open for each resolved, unmuted source while NOT
    recording, so the per-row meters are live before you ever hit Start (and again after Stop).
    These just open the reader and compute RMS — no VAD, no ASR. While recording, the real
    AudioCaptures own the devices, so monitors are torn down here (and rebuilt on the next idle
    run). Best-effort: a reader that won't open right now simply yields no monitor (its row
    shows an empty bar) and is retried next run, rather than raising into the sidebar."""
    monitors = st.session_state.level_monitors
    if st.session_state.running:
        # Captures have the devices — let go of every monitor so we never double-open one.
        for mon in monitors.values():
            mon.stop()
        monitors.clear()
        return

    wanted = {k for k in keys if k in by_key and k not in muted_keys}
    # Drop monitors that are no longer wanted (source removed, muted, retargeted, or gone).
    for k in list(monitors):
        if k not in wanted:
            monitors.pop(k).stop()
    # Open monitors for newly-wanted sources (preserve list order; harmless if it fails).
    for k in keys:
        if k in wanted and k not in monitors:
            mon = _open_level_monitor(by_key[k])
            if mon is not None:
                monitors[k] = mon


def _make_command_gate(src_kind, controller, command_gate, phone_ask_gate):
    """Build the get_command callable for a capture — what marks its speech as an assistant
    command (routed to the agent) instead of transcript. A mic uses the OS hold-to-ask key; the
    phone uses its two-tap listening window (phone_ask_gate, toggled by the phone's Ask button).
    Both are additionally gated on the assistant being enabled (command_gate), so a question can't
    be captured with nowhere to go. Any other source kind is never a command channel → None.
    Returns a fresh closure (not a loop-bound lambda) so per-source wiring can't capture the wrong
    variables. The gate dicts are mutable and shared, so live toggles reach the running capture."""
    if src_kind == "mic" and controller is not None:
        return lambda: controller.command_active() and command_gate["on"]
    if src_kind == "phone":
        return lambda: phone_ask_gate["on"] and command_gate["on"]
    return None


def _reconcile_live_captures(by_key, keys, muted_keys, selected_sources,
                             silence_ms, vad_threshold, partial_ms):
    """Bring up a capture for any source added through the picker mid-recording. The pipeline
    fans in through one shared segment queue, so a new source is purely additive: build its
    capture against the live seg_q + command gate and start it — the transcriber/translator
    already drain that queue and dispatch by tag, so nothing downstream restarts. This is the
    same in-place model as mute/unmute, one step further (a new thread instead of a flag flip).
    No-op once every selected source already has a capture, so it costs nothing on the 0.3s poll."""
    if not st.session_state.running:
        return
    seg_q = st.session_state.seg_q
    if seg_q is None:
        return
    from utils.audio import AudioCapture, make_reader

    capture_by_key = st.session_state.capture_by_key
    command_gate = st.session_state.command_gate or {"on": False}
    phone_ask_gate = st.session_state.get("phone_ask_gate") or {"on": False}
    controller = st.session_state.command_hotkey
    added = False
    for key in keys:
        src = by_key.get(key)
        if src is None or key in capture_by_key:
            continue
        # A fresh slot per add → its own (stateful) Silero RNN; never reuse a slot a live capture
        # still holds. The counter only climbs within a session, so removes can't cause a collision.
        slot = st.session_state.vad_slot_seq
        st.session_state.vad_slot_seq = slot + 1
        cap = AudioCapture(
            seg_q, load_vad_model(slot), make_reader(src), source=src.tag,
            silence_ms=silence_ms, threshold=vad_threshold,
            partial_interval_s=partial_ms / 1000.0,
            get_command=_make_command_gate(src.kind, controller, command_gate, phone_ask_gate),
            muted=key in muted_keys,
        )
        cap.start()  # device failures self-report on the thread; the row just shows an empty meter
        capture_by_key[key] = cap
        st.session_state.audio_captures.append(cap)
        added = True
    if added:
        # A live-added system/loopback channel must join the echo-dedup "clean" set (and flip
        # suppression on if it's the first one) — that set is otherwise frozen at Start.
        translator = st.session_state.translator
        if translator is not None:
            clean_tags = {s.tag for s in selected_sources if s.kind in ("loopback", "process")}
            translator.set_clean_sources(
                clean_tags, echo_suppress=st.session_state.echo_suppress and bool(clean_tags))
        st.toast("Source added — now transcribing.", icon="🎙️")


@st.cache_resource
def load_vad_model(slot: int = 0):
    # `slot` is part of the cache key on purpose: each capture source needs its OWN
    # Silero instance (the VAD iterator is a stateful RNN — sharing one across sources
    # corrupts its hidden state). Distinct slots → distinct cached models.
    from silero_vad import load_silero_vad
    return load_silero_vad()


@st.cache_resource
def list_audio_sources():
    from utils.audio import list_sources
    return list_sources()


# max_entries=1: keep only the model currently in use. Without a cap, st.cache_resource
# retains every distinct model_size ever selected — and a GPU large-v3 is ~3GB of VRAM, so
# switching models a couple of times could exhaust the card. Capping evicts the prior model
# (CTranslate2 frees it on GC) before the next is loaded.
@st.cache_resource(max_entries=1)
def load_whisper_model(model_size: str):
    from faster_whisper import WhisperModel

    # GPU (float16) is several times faster than CPU int8 on this workload. Note we do
    # NOT gate on torch.cuda — faster-whisper uses CTranslate2, which ships its own CUDA
    # runtime independent of torch (the bundled torch here is a CPU-only build). Try the
    # GPU directly and fall back if CTranslate2 can't find its cuDNN/cuBLAS DLLs, so the
    # app never crashes just because the GPU path is unavailable.
    try:
        return WhisperModel(model_size, device="cuda", compute_type="float16")
    except Exception as e:
        print(f"[whisper] CUDA unavailable ({e}); falling back to CPU int8.")
        return WhisperModel(model_size, device="cpu", compute_type="int8")


# --- Session persistence (utils/sessions.py) -----------------------------------------
# Each meeting is a session persisted to disk, so a crash/restart never loses a transcript and the
# sidebar can manage meetings like chat threads. The *active* session is just the live UI state
# (transcript / report / edited_text + an id/title); these helpers serialize it, autosave it, and
# swap a different session in. Defined here so the bootstrap + autosave below can call them.
def _transcript_to_text(events) -> str:
    """Flatten transcript events to plain text for the freeform editor — one line per utterance,
    prefixed with its source channel so that context survives a text edit (``[Mic] hello``)."""
    lines = []
    for e in events:
        tag = f"[{e.source}] " if getattr(e, "source", None) else ""
        lines.append(f"{tag}{e.display_text}")
    return "\n".join(lines)


def _continuation_text() -> str:
    """The captured lines recorded *after* the user last saved an edit. ``edited_base_count`` is how
    many events the edit covered; anything past it is a resumed recording the edit hasn't folded in
    yet. Slicing is bounds-safe, so a stale count just yields no continuation."""
    base = st.session_state.get("edited_base_count", 0)
    return _transcript_to_text(st.session_state.transcript[base:])


def _effective_transcript_text():
    """The transcript text the report and the post-stop edited view should use. ``None`` means
    "no edit — use the captured events" (the report formats events itself). Once edited, return the
    edited text PLUS any continuation (see _continuation_text): a recording resumed after an edit is
    appended onto the edit instead of being shadowed by it — otherwise the edit override hides every
    new line until the user reverts. Re-editing folds the continuation back in (see the editor seed)."""
    edited = st.session_state.edited_text
    if edited is None:
        return None
    cont = _continuation_text()
    return edited + ("\n" + cont if cont else "")


def _serialize_active_session() -> dict:
    return {
        "id": st.session_state.session_id,
        "title": st.session_state.session_title,
        "created": st.session_state.session_created or time.time(),
        "events": [e.to_dict() for e in st.session_state.transcript],
        "edited_text": st.session_state.edited_text,
        # How many events the edit covered — events past this index are a continuation appended
        # onto the edit (see _effective_transcript_text). Persisted so a resumed/edited meeting
        # survives a save/load.
        "edited_base_count": st.session_state.get("edited_base_count", 0),
        "report": sessions.report_to_dict(st.session_state.report),
    }


def _touch_index_entry(data: dict) -> None:
    """Update (or insert) this session's entry in the cached sidebar index, newest-first, so the
    sidebar reflects a save without re-scanning disk."""
    idx = st.session_state.sessions_index
    if idx is None:
        st.session_state.sessions_index = sessions.list_meta()
        return
    entry = {
        "id": data["id"],
        "title": data.get("title") or "Untitled",
        "created": data.get("created", 0.0),
        "updated": time.time(),
        "n_events": len(data.get("events") or []),
    }
    idx = [m for m in idx if m.get("id") != data["id"]]
    idx.insert(0, entry)
    st.session_state.sessions_index = idx


def _save_active_session() -> None:
    """Persist the active meeting now and refresh its sidebar-index entry. Skips writing a session
    that has no content and was never saved, so a launch that does nothing leaves no empty file."""
    sid = st.session_state.get("session_id")
    if not sid:
        return
    has_content = bool(st.session_state.transcript or st.session_state.report
                       or st.session_state.edited_text)
    if not has_content and not sessions.exists(sid):
        return
    data = _serialize_active_session()
    sessions.save(data)  # stamps 'updated'
    st.session_state._autosave_len = len(st.session_state.transcript)
    st.session_state._autosave_ts = time.time()
    _touch_index_entry(data)


def _reset_active_view_state() -> None:
    """Clear the per-session live/render state so nothing bleeds across a session swap."""
    st.session_state.partials = {}
    st.session_state.committed_ids = set()
    st.session_state.editing = False
    st.session_state.pop("_tx_html", None)
    st.session_state.pop("_edit_buffer", None)
    st.session_state.pop("_edit_reset", None)


def _set_fresh_session() -> None:
    """Make a brand-new empty session the active one (no disk write until it has content)."""
    fresh = sessions.new_session()
    st.session_state.session_id = fresh["id"]
    st.session_state.session_title = fresh["title"]
    st.session_state.session_created = fresh["created"]
    st.session_state.transcript = []
    st.session_state.edited_text = None
    st.session_state.edited_base_count = 0
    st.session_state.report = None
    _reset_active_view_state()
    st.session_state._autosave_len = 0
    st.session_state._autosave_ts = time.time()


def _apply_session(data: dict) -> None:
    """Load a stored session dict into the live UI state (make it the active meeting)."""
    st.session_state.session_id = data.get("id")
    st.session_state.session_title = data.get("title") or ""
    st.session_state.session_created = data.get("created")
    st.session_state.transcript = [sessions.event_from_dict(d) for d in (data.get("events") or [])]
    st.session_state.edited_text = data.get("edited_text")
    # Migration: sessions saved before the continuation feature have no edited_base_count. Back then
    # an edit always covered the entire transcript, so default a missing count to len(events) — NOT
    # 0, which would re-append every captured line on top of the edit (a duplicate transcript).
    _base = data.get("edited_base_count")
    st.session_state.edited_base_count = (
        int(_base) if _base is not None else len(st.session_state.transcript))
    st.session_state.report = sessions.report_from_dict(data.get("report"))
    _reset_active_view_state()
    st.session_state._autosave_len = len(st.session_state.transcript)
    st.session_state._autosave_ts = time.time()


# --- Session state init ---
_defaults = {
    "running": False,
    "transcript": [],       # list[TranscriptEvent]
    "selected_source_keys": ["mic:default"],  # source.key list chosen via the Add-source picker
    "muted_source_keys": [],  # subset of the above the user has muted — kept in the list but not captured
    "echo_suppress": True,  # drop mic lines that echo system audio (set per mic via its ⚙ Edit dialog)
    "audio_captures": [],   # list[AudioCapture] — one per selected source
    "capture_by_key": {},   # source.key -> its live AudioCapture, for mute/unmute while recording
    "level_monitors": {},   # source.key -> LevelMonitor — drives the per-row meters while NOT recording
    "seg_q": None,          # shared segment queue (set at Start) — lets add-while-recording wire new captures in
    "vad_slot_seq": 0,      # monotonic VAD cache-slot counter, so every live-added capture gets its own Silero RNN
    "command_gate": None,   # {"on": bool} shared with mic captures — live push-to-ask enable while recording
    "transcriber": None,
    "translator": None,
    "command_hotkey": None,
    "agent": None,
    "context_store": None,    # ConversationStore — compacting transcript memory for the agent
    "bus": None,
    "event_sink": None,       # queue.Queue fed by a bus subscriber
    "assistant_sink": None,   # queue.Queue of AssistantMessage from the agent
    "assistant_log": [],      # list[AssistantMessage] rendered in the Assistant panel
    "report": None,           # generated bilingual Report (or None)
    "partial_sink": None,     # queue.Queue of interim (in-progress) TranscriptEvents
    "partials": {},           # segment_id -> latest interim TranscriptEvent (live preview lines)
    "committed_ids": set(),   # segment_ids already finalized — ignore any late partials for them
    "phone_ask_gate": None,       # {"on": bool} shared with the phone capture — the phone's Ask
                                  # button two-tap window flips it; on ⇒ phone speech is a command
    "use_context": False,         # include live conversation context (summary + recent lines) in
                                  # each assistant ask. Off by default: context is only worth its
                                  # tokens/quota when a question references earlier talk — turn it
                                  # on (desktop toggle / phone Context button) when you need it.
    # --- session persistence (utils/sessions.py) ---
    "session_id": None,           # active meeting's id (None ⇒ bootstrap adopts the latest / a fresh one)
    "session_title": "",          # active meeting's title (sidebar label; renamable)
    "session_created": None,      # active meeting's creation time (epoch)
    "edited_text": None,          # freeform-edited transcript; once set, the source of truth for the report
    "edited_base_count": 0,       # # events the edit covered; events past it are an appended continuation
    "editing": False,             # post-stop edit mode is open
    "sessions_index": None,       # cached list_meta() for the sidebar; built lazily, mutated in place
    "_autosave_len": 0,           # transcript length at last autosave (change detection)
    "_autosave_ts": 0.0,          # wall-clock of last autosave (debounce)
    "show_settings": False,       # sidebar settings revealed? off by default to declutter; the
                                  # Settings button toggles it. Audio sources are always shown.
}
# Cross-session UI settings (utils/settings.py): load the saved sidebar state once per session.
# The session_state-backed settings (chosen sources, echo/context toggles) override their
# built-in defaults here; widget-local controls (model, languages, VAD/beam sliders, translation
# toggles) are seeded at their own widgets below. Everything is re-saved on change after the
# sidebar renders (see the persist block at the end of the sidebar).
if "_persisted" not in st.session_state:
    st.session_state["_persisted"] = load_settings()
_persisted = st.session_state["_persisted"]
for _pk in ("echo_suppress", "use_context", "selected_source_keys", "muted_source_keys"):
    if _pk in _persisted:
        _defaults[_pk] = _persisted[_pk]

for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Active-session bootstrap (once per process). On the first run, adopt the most recently saved
# session — continuity, and crash recovery: an autosaved meeting reloads on restart so a quit
# loses nothing. With no saved sessions, start a fresh empty one (written to disk only once it
# has content). session_id stays set thereafter, so this never re-runs within a process.
if st.session_state.session_id is None:
    if st.session_state.sessions_index is None:
        st.session_state.sessions_index = sessions.list_meta()
    _idx = st.session_state.sessions_index
    _boot = sessions.load(_idx[0]["id"]) if _idx else None
    if _boot:
        _apply_session(_boot)
    else:
        _set_fresh_session()


def _seed_index(options, value, default=0):
    """Index of a persisted selectbox value within its options, or ``default`` if the value is
    absent/stale (e.g. an option that no longer exists). Lets selectboxes restore across sessions
    without crashing when the saved choice is gone."""
    try:
        return options.index(value)
    except (ValueError, TypeError):
        return default

# Proper-noun vocabulary, persisted to vocabulary.json. Shared between the UI (writer) and the
# Transcriber/Translator worker threads (readers), so edits apply on the next utterance with no
# restart. See utils/glossary.py — this is decode-time prompt biasing, not model training.
if "glossary" not in st.session_state:
    from utils.glossary import Glossary
    st.session_state.glossary = Glossary()

# Whether the vocabulary also biases the Whisper *decode* (hotwords), not just the LLM cleanup.
# Off by default: faster-whisper's hotwords condition the decoder's previous-text prompt, which
# over-forces the terms on short/quiet clips — the same conditioning-hallucination we removed from
# initial_prompt. Kept as an opt-in via a live gate dict (mutated in place), so the Transcriber's
# get_hotwords reads the current state per utterance, like the app's other live toggles.
if "hotwords_gate" not in st.session_state:
    st.session_state.hotwords_gate = {"on": False}

# The Add source / Edit source / Rename session UIs are inline flag-gated panels (not st.dialogs),
# so the old poll "stand-down + run_every re-arm" dance is gone entirely. There is no dialog
# *fragment* for an st.rerun() to leave un-pruned, so the bottom meter/live poll can fire every run
# unconditionally. The panels are gated on `show_add_source` / `edit_source_key` /
# `rename_session_sid` and render in always-present keyed slots in the sidebar (ghost-safe). This
# removes the entire class of browser-side run_every timer that produced the "fragment ... does not
# exist anymore" console flood — no run_every is armed anywhere now.

# Drain the bus sink before rendering so the transcript is always current. A finalized line
# supersedes its live preview: drop the matching partial and remember the id so a late
# interim decode for the same utterance can't resurrect a ghost preview line.
if st.session_state.running and st.session_state.event_sink:
    while not st.session_state.event_sink.empty():
        ev = st.session_state.event_sink.get_nowait()
        st.session_state.transcript.append(ev)
        st.session_state["_last_rx_ts"] = time.time()  # activity stamp → adaptive poll cadence below
        sid = getattr(ev, "segment_id", None)
        if sid:
            st.session_state.partials.pop(sid, None)
            st.session_state.committed_ids.add(sid)

# Autosave the live transcript so a crash/quit never loses the meeting (the whole reason a hard
# quit hurts today). Debounced: only when the transcript actually grew, and at most every few
# seconds — this script reruns ~3×/s, so we don't touch disk on every tick. Worst case a crash
# loses the last few seconds of lines; Stop forces a final, authoritative save regardless.
_AUTOSAVE_MIN_INTERVAL_S = 4.0
if (st.session_state.running and st.session_state.session_id
        and len(st.session_state.transcript) != st.session_state._autosave_len
        and time.time() - st.session_state._autosave_ts > _AUTOSAVE_MIN_INTERVAL_S):
    _save_active_session()

# Drain interim (in-progress) decodes into the live-preview dict, keyed by segment so each
# open utterance shows a single tentative line that updates in place — unless it's already
# been committed above.
if st.session_state.running and st.session_state.partial_sink:
    while not st.session_state.partial_sink.empty():
        ev = st.session_state.partial_sink.get_nowait()
        sid = getattr(ev, "segment_id", None)
        if not sid:
            continue
        if not getattr(ev, "partial", False):
            # A *final* (not a partial) arrived on the preview channel — two cases, both meaning
            # "this segment's live line is done": a command final (its text goes to the Assistant
            # panel, not the transcript) or a no-text tombstone for a final the quality gate
            # dropped as silence/hallucination. Either way retire the interim line and block a
            # late partial from resurrecting it.
            st.session_state.partials.pop(sid, None)
            st.session_state.committed_ids.add(sid)
        elif sid not in st.session_state.committed_ids:
            st.session_state.partials[sid] = ev
            st.session_state["_last_rx_ts"] = time.time()  # live-preview update → keep poll fast

# Safety net: retire any live-preview line whose segment has clearly closed but whose final
# never arrived to supersede it — e.g. a mic line echo-suppressed at the translator (it's
# dropped there, so no tombstone reaches us), or any final lost downstream. An OPEN segment
# refreshes its partial every preview interval, so a partial gone stale for several seconds is
# a ghost; without this it would hang forever (the bug behind stacked, never-resolving "Live"
# lines). Worst case we drop the preview of an unusually slow final a moment before its
# committed line lands via the bus — no data loss.
_PARTIAL_TTL_S = 6.0
if st.session_state.partials:
    _now = time.time()
    _stale = [sid for sid, ev in st.session_state.partials.items()
              if _now - (getattr(ev, "ts_end", 0) or 0) > _PARTIAL_TTL_S]
    for sid in _stale:
        st.session_state.partials.pop(sid, None)
        st.session_state.committed_ids.add(sid)

# Phone control surface: a handle to the (singleton) phone server, present once a Phone source
# has spun it up. Used to echo assistant answers + confirmed control state to the phone, and to
# drain the phone's inbound control taps. current_server() reads the singleton without starting it.
from utils.phone_server import current_server as _get_phone_server
_phone_server = _get_phone_server() if st.session_state.running else None

# Drain the agent's assistant sink the same way. Messages arrive twice (pending → done)
# under one id; upsert by id so the panel updates in place instead of duplicating.
if st.session_state.running and st.session_state.assistant_sink:
    log = st.session_state.assistant_log
    while not st.session_state.assistant_sink.empty():
        msg = st.session_state.assistant_sink.get_nowait()
        for i, existing in enumerate(log):
            if existing.id == msg.id:
                log[i] = msg
                break
        else:
            log.append(msg)
        # Echo a phone-initiated ask back to the phone screen (pending → done/error). The agent
        # tags each message with the originating channel, so a question spoken into the phone's
        # listening window (source "Phone") routes its answer back to the phone — desktop
        # push-to-ask (a mic source) doesn't.
        if _phone_server is not None and getattr(msg, "source", None) == "Phone":
            _phone_server.push_agent(msg.status, msg.query, msg.answer or "")


# --- Phone control taps (translation / mute / ask) -----------------------------------
# The phone is a PARALLEL control surface. Taps land on the phone server's command queue from
# its asyncio socket thread; we drain and apply them HERE so the live workers are only ever
# mutated from the Streamlit thread — the same place the desktop widgets drive them. Confirmed
# state is pushed back to the phone (wait-for-echo) further below, near the phone status bar.
if _phone_server is not None:
    for _cmd in _phone_server.pop_commands():
        _kind = _cmd.get("cmd")
        if _kind == "translation":
            # Drive the same session key the desktop toggle owns, before that widget renders.
            st.session_state.translate_on = bool(_cmd.get("on"))
        elif _kind == "context":
            # Include/exclude live conversation context in assistant asks. Applied to the running
            # agent in the toggles block below (set_use_context); persists across the poll reruns.
            st.session_state.use_context = bool(_cmd.get("on"))
        elif _kind == "mute":
            _on = bool(_cmd.get("on"))
            _mk = st.session_state.muted_source_keys
            if _on and "phone:default" not in _mk:
                _mk.append("phone:default")
            elif not _on and "phone:default" in _mk:
                _mk.remove("phone:default")
            _cap = st.session_state.capture_by_key.get("phone:default")
            if _cap is not None:
                _cap.set_muted(_on)  # silence transcription; the uplink stays open
        elif _kind == "ask":
            # Two-tap listening window: on=open, off=close. While open, the phone capture's
            # get_command gate is set, so the phone's speech is captured as ONE command (the VAD
            # loop holds it open across pauses, exactly like the desktop hold-to-ask key) and
            # routed to the agent on close. Mutating the shared gate dict reaches the live
            # capture's onset/hold checks; the agent's answer echoes back via the sink drain above.
            _gate = st.session_state.get("phone_ask_gate")
            if _gate is not None:
                _gate["on"] = bool(_cmd.get("on"))


# --- Presentation helpers ------------------------------------------------------------
# Stable-ish per-channel accent so each audio source is visually distinct on the transcript.
_SRC_PALETTE = ["#2dd4bf", "#60a5fa", "#f472b6", "#fbbf24", "#a78bfa", "#34d399", "#fb7185"]


def _src_color(tag: str) -> str:
    # zlib.crc32 (not builtin hash, which is salted per process via PYTHONHASHSEED) so a
    # given channel keeps the same accent across app launches.
    return _SRC_PALETTE[zlib.crc32(tag.encode()) % len(_SRC_PALETTE)] if tag else "#2dd4bf"


def _md_escape(text: str) -> str:
    """Escape characters that would break a Markdown link's label text."""
    return (text or "").replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")


def _clock(ts: float) -> str:
    """Wall-clock HH:MM:SS for a segment's start time (epoch seconds)."""
    try:
        return datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S")
    except Exception:
        return ""


def _render_transcript(events) -> str:
    rows = []
    for e in events:
        src = getattr(e, "source", None) or ""
        color = _src_color(src)
        badge = (f"<span class='stt-badge' style='color:{color};background:{color}1f;"
                 f"border-color:{color}66'><span class='dot' style='background:{color}'></span>"
                 f"{html.escape(src)}</span>") if src else ""
        lang = f"<span class='stt-chip mono'>{e.source_lang.upper()}</span>" if e.source_lang else ""
        lock = (f"{_ms('lock', 14, 'var(--listen)')}"
                if e.lang_source == "forced" else "")
        ts = _clock(getattr(e, "ts_start", 0) or 0)
        time_el = f"<span class='stt-time mono'>{ts}</span>" if ts else ""
        trans = ""
        if e.translation:
            tgt = (e.target_lang or "").upper()
            trans = (f"<div class='stt-trans'><span class='stt-chip alt mono'>{tgt}</span>"
                     f"<span>{html.escape(e.translation)}</span></div>")
        rows.append(
            f"<div class='stt-msg' style='border-left-color:{color}'>"
            f"<div class='stt-meta'>{badge}{lang}{lock}{time_el}</div>"
            f"<div class='stt-text'>{html.escape(e.display_text)}</div>"
            f"{trans}</div>"
        )
    return "".join(rows)


def _render_partials(partials) -> str:
    """Render the in-progress (tentative) lines below the committed transcript. These update
    in place as the speaker talks and are replaced by the committed line at segment end."""
    rows = []
    for e in sorted(partials.values(), key=lambda x: getattr(x, "ts_start", 0) or 0):
        src = getattr(e, "source", None) or ""
        color = _src_color(src)
        badge = (f"<span class='stt-badge' style='color:{color};background:{color}1f;"
                 f"border-color:{color}66'><span class='dot' style='background:{color}'></span>"
                 f"{html.escape(src)}</span>") if src else ""
        lang = f"<span class='stt-chip mono'>{e.source_lang.upper()}</span>" if e.source_lang else ""
        live = "<span class='stt-live'><span class='dot'></span>Live</span>"
        rows.append(
            f"<div class='stt-msg partial' style='border-left-color:{color}'>"
            f"<div class='stt-meta'>{badge}{lang}<span class='stt-time'>{live}</span></div>"
            f"<div class='stt-text'>{html.escape(e.text)}…</div></div>"
        )
    return "".join(rows)


gemini_key_present = bool(get_api_key())


# --- Audio-source picker (Add-source modal) ------------------------------------------
# Category → Material Symbols icon, used by both the chosen-source rows and the modal.
_CAT_ICON = {"Microphone": "mic", "Phone": "smartphone", "System Audio": "volume_up",
             "App": "desktop_windows", "Browser": "public"}


def _phone_pairing_block():
    """Render the phone-mic pairing card (QR + URL + live connection status), starting the
    phone server lazily. Shared by the Add-source and Edit-source dialogs. Best-effort: if the
    optional deps aren't installed (or the server can't bind/cert), explain why instead of
    raising into the dialog."""
    from utils.phone_server import get_server, phone_import_error

    why = phone_import_error()
    if why:
        st.warning(f":material/error: Phone mic unavailable — {why}.")
        return
    try:
        server = get_server()
    except Exception as e:
        st.warning(f":material/error: Phone mic unavailable — {e}.")
        return

    if server.is_connected:
        st.success("Phone connected — audio is streaming.", icon=":material/check_circle:")
    else:
        st.caption(":material/qr_code_scanner: Scan with your phone's camera (Android or "
                   "iPhone), open the link, **accept the certificate warning** once, then tap "
                   "**Start microphone**.")
    st.markdown(
        f"<div style='display:flex;justify-content:center;margin:6px 0 10px'>"
        f"<img src='{server.qr_data_uri()}' width='196' height='196' "
        f"style='border-radius:12px;background:#fff;padding:8px' alt='Pairing QR'/></div>",
        unsafe_allow_html=True,
    )
    st.code(server.url, language=None)
    st.caption(":material/lock: The link is private to your Wi-Fi and carries a one-time access "
               "token. The certificate warning is expected — it's a self-signed cert used only "
               "to satisfy the browser's secure-mic requirement.")


def _render_add_source_panel():
    """Inline 'Add source' panel (replaces the old @st.dialog). Two-step picker: pick a category,
    then the specific target within it. Rendered in an always-present keyed slot gated on
    `show_add_source`. Being inline (not a dialog fragment) means the meter/live poll can st.rerun()
    freely, and a spinner shown here actually reaches the client — so the old pre-warm and
    Rescan-reopen hand-offs (which only existed to dodge the discarded-fragment-render problem) are
    gone: we open the new source's level monitor right here under a visible spinner."""
    from utils.audio import SOURCE_CATEGORIES, source_category, loopback_error

    sources = list_audio_sources()
    by_key = {s.key: s for s in sources}
    chosen = set(st.session_state.selected_source_keys)

    category = st.selectbox(
        "Category", SOURCE_CATEGORIES, key="add_category",
        help="Microphone — an input device. System Audio — everything you hear (all apps "
             "mixed). App — one program's audio, captured natively (no virtual cable). "
             "Browser — a web browser, same native capture.",
    )

    if category in ("App", "Browser"):
        st.caption(":material/info: Captured natively via Windows process loopback (the same "
                   "technique as OBS) — you still hear it normally, no virtual cable. Only "
                   "programs currently playing audio appear; start playback, then Rescan.")

    if category == "Phone":
        _phone_pairing_block()

    candidates = [s for s in sources
                  if source_category(s) == category and s.key not in chosen]

    choice_key = None
    if candidates:
        choice_key = st.selectbox(
            "Target", [s.key for s in candidates], key=f"add_target_{category}",
            format_func=lambda k: by_key[k].target_label or by_key[k].label,
        )
    elif category == "System Audio" and loopback_error():
        st.info(f"No system-audio source available — {loopback_error()}.")
    elif category in ("App", "Browser"):
        st.info(f"No {category.lower()} is playing audio right now (or it's already added). "
                "Start playback, then Rescan.")
    else:
        st.info("Nothing left to add here — every target in this category is already selected.")

    # Primary action on its own full-width row; Rescan/Cancel paired beneath. Three buttons in one
    # row squeezed and wrapped their labels in a narrow sidebar — this keeps each label intact.
    if st.button("Add", type="primary", icon=":material/add:", key="add_confirm",
                 use_container_width=True, disabled=choice_key is None):
        st.session_state.selected_source_keys.append(choice_key)
        # Open the new source's idle input-level monitor (a system-audio loopback stream takes ~1s)
        # under a spinner the user can see — inline, so st.spinner streams normally. Only while idle
        # and unmuted; while recording the per-source capture provides the meter instead.
        with st.spinner("Adding source…"):
            if (not st.session_state.running
                    and choice_key not in st.session_state.muted_source_keys
                    and choice_key in by_key):
                mon = _open_level_monitor(by_key[choice_key])
                if mon is not None:
                    st.session_state.level_monitors[choice_key] = mon
        st.session_state.show_add_source = False
        st.rerun()
    b2, b3 = st.columns(2)
    if b2.button("Rescan", icon=":material/refresh:", key="add_rescan", use_container_width=True,
                 help="Re-detect devices and currently-playing apps."):
        list_audio_sources.clear()
        st.rerun()  # the panel stays open (show_add_source still set) and re-renders the fresh list
    if b3.button("Cancel", key="add_cancel", use_container_width=True):
        st.session_state.show_add_source = False
        st.rerun()


# Shared, multi-line help reused by the per-source mic echo toggle (rows render it via the
# ⚙ Edit-source dialog rather than crowding every mic row with the explanation).
_ECHO_HELP = (
    "For headsets with crosstalk (the mic electrically picks up what you're hearing "
    "through a shared ground), call/video audio gets transcribed twice — once from System "
    "loopback, once echoed through the mic. This drops the mic line when it duplicates a "
    "recent System line. Your own speech is unaffected. Global toggle — only active when "
    "both a mic and a system-audio source are on."
)


def _render_edit_source_panel(key: str):
    """Inline 'Edit source' panel (replaces the old @st.dialog). Re-point an already-added source
    at a different target in its own category, and (for mics) toggle echo suppression — so
    reconfiguring doesn't mean remove + re-add. Gated on `edit_source_key`."""
    from utils.audio import source_category

    sources = list_audio_sources()
    by_key = {s.key: s for s in sources}
    src = by_key.get(key)
    if src is None:
        st.info("This source is no longer available — remove it and add a new one.")
        if st.button("Close", key="edit_close_na", use_container_width=True):
            st.session_state.edit_source_key = None
            st.rerun()
        return

    category = source_category(src)

    # Phone has a single virtual target — there's nothing to re-point. Show the pairing card
    # (QR + live status) so this panel doubles as "reconnect my phone", and stop here.
    if src.kind == "phone":
        _phone_pairing_block()
        if st.button("Done", type="primary", icon=":material/check:", key="edit_phone_done",
                     use_container_width=True):
            st.session_state.edit_source_key = None
            st.rerun()
        return

    chosen = set(st.session_state.selected_source_keys)
    # Same-category targets, minus ones already chosen on other rows; always keep the
    # current target so it shows as the selected option.
    candidates = [s for s in sources if source_category(s) == category
                  and (s.key == key or s.key not in chosen)]
    cand_keys = [s.key for s in candidates]

    # Seed the widget's state to the current target once (rather than passing index= alongside
    # key=, which makes Streamlit warn on every rerun, e.g. when the echo box is toggled).
    sel_key = f"edit_tgt_{key}"
    if sel_key not in st.session_state:
        st.session_state[sel_key] = key

    # Re-target live (no "Done" button): apply on change, so the editor closes the same way it
    # opens — by clicking the gear again. Keep the panel open, now pointed at the new target.
    def _apply_retarget():
        new = st.session_state.get(sel_key)
        if not new or new == key or new not in by_key:
            return
        try:
            pos = st.session_state.selected_source_keys.index(key)
        except ValueError:
            return
        st.session_state.selected_source_keys[pos] = new
        st.session_state.edit_source_key = new

    st.selectbox(
        f"{category} target", cand_keys, key=sel_key,
        format_func=lambda k: by_key[k].target_label or by_key[k].label,
        help="Point this source at a different device or app without removing it.",
        on_change=_apply_retarget,
    )

    if src.kind == "mic":
        # Plain session value (no widget key) so it survives the panel close and is
        # readable at start time, when this panel isn't rendered.
        st.session_state.echo_suppress = st.checkbox(
            "Suppress mic echo of system audio", value=st.session_state.echo_suppress,
            help=_ECHO_HELP,
        )
    st.caption("Changes apply live — click the gear again to close.")


def _render_source_rows(by_key, keys, locked):
    """Render the selected-source rows into PERSISTENT per-row st.empty() slots, explicitly
    clearing surplus slots when the list shrinks.

    The ghost we keep fighting: deleting a source shortens the row list, and a shrinking
    container clears its trailing child *positionally* — which does NOT reliably tear down the
    removed row's DOM, so its mute/gear/close buttons (and the whole row) linger ("pulses and
    returns"). Per-row keying did not help either: a vanished key inside a shrinking parent still
    relies on positional clearing.

    What DOES work is the same idiom the inline edit/add panels use: a persistent `st.empty()`
    slot whose content is *replaced* each run, and `slot.empty()` on the hide path — st.empty()'s
    clear/replace semantics force a real teardown. So we give EACH row index its own st.empty()
    slot, fill the first `len(keys)` with rows and call `.empty()` on the rest. The slot COUNT is a
    per-session high-water mark, so the box's child count only ever GROWS, never shrinks (the other
    half of the ghost rule — a shrinking child count is what ghosts). (dynamic-list ghost guard.)"""
    keys = list(keys)
    n = len(keys)
    hw = max(st.session_state.get("_src_slot_hw", 0), n)
    st.session_state._src_slot_hw = hw

    # Empty-state line in its own persistent slot (always present; shown only with no sources).
    _empty_slot = st.empty()
    if n == 0:
        _empty_slot.caption(":material/info: No sources yet — add one to begin.")
    else:
        _empty_slot.empty()

    for i in range(hw):
        slot = st.empty()
        if i < n:
            with slot.container():
                _render_one_source_row(keys[i], by_key, locked)
        else:
            slot.empty()  # surplus slot from a since-deleted source — force its teardown


def _render_one_source_row(key, by_key, locked):
    """Render one source row (label+meter, mute, gear, close). Placed inside a persistent
    st.empty() slot by _render_source_rows so a deleted source is cleared by slot.empty() rather
    than positional child-shrinking (see that docstring)."""
    from utils.audio import source_category
    src = by_key.get(key)
    muted = key in st.session_state.muted_source_keys
    # label, mute, gear, close
    c1, c2, c3, c4 = st.columns([0.58, 0.14, 0.14, 0.14], vertical_alignment="center")
    if src:
        cat = source_category(src)
        # Live input-level bar under the label. While recording the source's capture provides the
        # level; while idle a lightweight LevelMonitor does (reconciled above). A muted source
        # shows an empty, dimmed track — silenced but still visibly present. No provider yet (a
        # monitor that hasn't opened) → no track.
        provider = (st.session_state.capture_by_key.get(key)
                    or st.session_state.level_monitors.get(key))
        if provider is not None:
            meter = _level_meter(_level_frac(provider.level()), dim=muted)
        elif muted:
            meter = _level_meter(0.0, dim=True)
        else:
            meter = ""
        _lbl = html.escape(src.target_label or src.label)
        c1.markdown(
            f"<div class='src-row{' muted' if muted else ''}'>"
            f"{_ms(_CAT_ICON.get(cat, 'graphic_eq'), 16)}"
            f"<span title='{html.escape(cat)} · {_lbl}'><b>{html.escape(cat)}</b> · "
            f"{_lbl}</span></div>{meter}",
            unsafe_allow_html=True,
        )
    else:
        # An added App/Browser whose process has since closed, or a device unplugged. It's kept
        # in the list (so the choice isn't silently lost) but flagged; Rescan or remove it.
        # Unresolved keys are dropped from selected_sources below.
        kind = key.split(":", 1)[0]
        c1.markdown(
            f"<div class='src-row na'>{_ms('error', 16, 'var(--danger)')}"
            f"<span><i>Unavailable</i> · this {html.escape(kind)} is no longer "
            f"active</span></div>",
            unsafe_allow_html=True,
        )
    # 🔇 mute silences a source without removing its row. It works live: while recording we flip
    # the running capture's flag so it stops/keeps transcribing immediately (the reader stays open
    # for instant unmute); stopped, it just sets the next-Start state. Either way muted sources are
    # dropped from the live count via selected_sources.
    if c2.button(":material/volume_off:" if muted else ":material/volume_up:",
                 key=f"mute_{key}", disabled=src is None,
                 help="Unmute — start transcribing this source again" if muted
                 else "Mute — silence this source now (it stays in the list)"):
        if muted:
            st.session_state.muted_source_keys.remove(key)
        else:
            st.session_state.muted_source_keys.append(key)
        cap = st.session_state.capture_by_key.get(key)
        if cap is not None:
            cap.set_muted(not muted)  # `muted` here is the pre-toggle state
        st.rerun()
    # ⚙ edits the source in place (re-target / mic echo) so a tweak isn't remove + re-add;
    # disabled when the source is gone (nothing to edit) or while recording. The gear is a toggle:
    # clicking it again (or on another row) collapses the inline editor.
    _edit_open = st.session_state.get("edit_source_key") == key
    if c3.button(":material/settings:", key=f"edit_{key}", disabled=locked or src is None,
                 help="Close editor" if _edit_open
                 else "Change this source's target or mic options"):
        st.session_state.edit_source_key = None if _edit_open else key
        st.rerun()
    if c4.button(":material/close:", key=f"rm_{key}", disabled=locked,
                 help="Remove this source"):
        st.session_state.selected_source_keys.remove(key)
        if key in st.session_state.muted_source_keys:
            st.session_state.muted_source_keys.remove(key)
        st.rerun()


# --- Sessions: per-meeting threads (new / switch / rename / delete) -------------------
def _new_session_action():
    _save_active_session()   # don't lose the current meeting
    _set_fresh_session()
    st.rerun()


def _switch_session_action(sid):
    _save_active_session()
    data = sessions.load(sid)
    if data:
        _apply_session(data)
    st.rerun()


def _delete_session_action(sid):
    sessions.delete(sid)
    if st.session_state.sessions_index is not None:
        st.session_state.sessions_index = [
            m for m in st.session_state.sessions_index if m.get("id") != sid]
    if sid == st.session_state.session_id:
        # Deleted the active meeting — adopt the next most recent, or a fresh blank.
        idx = st.session_state.sessions_index or []
        data = sessions.load(idx[0]["id"]) if idx else None
        if data:
            _apply_session(data)
        else:
            _set_fresh_session()
    st.rerun()


def _render_rename_session_panel(sid):
    """Inline 'Rename session' panel (replaces the old @st.dialog). Gated on `rename_session_sid`."""
    if sid == st.session_state.session_id:
        current = st.session_state.session_title
    else:
        _d = sessions.load(sid)
        current = (_d or {}).get("title", "")
    # Seed once (not value= alongside key=, which warns on every rerun). The widget then
    # retains the last-saved title across reopens, which is the current value in practice.
    _in_key = f"rename_input_{sid}"
    if _in_key not in st.session_state:
        st.session_state[_in_key] = current
    new_title = st.text_input("Title", key=_in_key)
    r1, r2 = st.columns([0.6, 0.4])
    if r1.button("Save", type="primary", icon=":material/check:", key="rename_save",
                 use_container_width=True):
        title = (new_title or "").strip() or current or sessions.default_title()
        if sid == st.session_state.session_id:
            st.session_state.session_title = title
            _save_active_session()
        else:
            _d = sessions.load(sid)
            if _d:
                _d["title"] = title
                sessions.save(_d)
                _touch_index_entry(_d)
        st.session_state.rename_session_sid = None
        st.rerun()
    if r2.button("Cancel", key="rename_cancel", use_container_width=True):
        st.session_state.rename_session_sid = None
        st.rerun()


def _render_sessions_sidebar(locked: bool) -> None:
    """A ChatGPT-style list of saved meetings at the top of the sidebar. Switching/new/delete are
    locked while recording (a swap mid-record would write into the wrong session). The whole list
    lives inside one always-rendered expander, and each row inside its own keyed container, so a
    changing session count can't shift — and ghost — the settings expanders below it."""
    if st.session_state.sessions_index is None:
        st.session_state.sessions_index = sessions.list_meta()
    idx = st.session_state.sessions_index
    active_id = st.session_state.session_id
    with st.expander("Sessions", expanded=False, icon=":material/forum:", key="sec_sessions"):
        if st.button("New session", icon=":material/add:", use_container_width=True,
                     disabled=locked, key="new_session",
                     help="Start a fresh meeting. The current one is saved automatically."):
            _new_session_action()
        # Keyed, always-present slot so the lock hint toggling can't change the child count.
        with st.container(key="sess_lock_hint"):
            if locked:
                st.caption(":material/lock: Stop recording to switch or start a session.")
        # The active session may be brand-new and not yet in the index — surface it at the top.
        rows = list(idx)
        if active_id and active_id not in {m["id"] for m in rows}:
            rows.insert(0, {"id": active_id, "title": st.session_state.session_title,
                            "updated": time.time(),
                            "n_events": len(st.session_state.transcript)})
        if not rows:
            st.caption("No saved sessions yet.")
        for m in rows:
            sid = m["id"]
            is_active = sid == active_id
            with st.container(key=f"sessrow_{sid}"):
                c1, c2, c3 = st.columns([0.72, 0.14, 0.14], vertical_alignment="center")
                label = st.session_state.session_title if is_active else (m.get("title") or "Untitled")
                if c1.button(("● " if is_active else "") + label, key=f"sess_{sid}",
                             use_container_width=True, disabled=locked or is_active,
                             help="Currently open" if is_active else "Switch to this meeting"):
                    _switch_session_action(sid)
                if c2.button(":material/edit:", key=f"sessedit_{sid}", disabled=locked,
                             help="Rename"):
                    st.session_state.rename_session_sid = sid
                    st.rerun()
                if c3.button(":material/delete:", key=f"sessdel_{sid}", disabled=locked,
                             help="Delete this meeting (cannot be undone)"):
                    _delete_session_action(sid)

        # Inline rename panel (replaces the old @st.dialog), in an always-present keyed slot after
        # the rows so toggling it can't change the expander's child count (ghost-container guard).
        with st.container(key="rename_session_panel"):
            _rename_slot = st.empty()  # st.empty() so the panel fully clears on close (see edit slot)
            _rs = st.session_state.get("rename_session_sid")
            if _rs is not None:
                with _rename_slot.container():
                    _render_rename_session_panel(_rs)
            else:
                _rename_slot.empty()


# --- Sidebar (grouped, collapsible) --------------------------------------------------
with st.sidebar:
    locked = st.session_state.running
    _render_sessions_sidebar(locked)

    # Settings live behind a single toggle so the sidebar stays uncluttered; Audio sources are the
    # one exception (always shown). The settings widgets must still render on EVERY rerun — the rest
    # of the app reads model/language/VAD/translation values each run, and Streamlit purges a widget's
    # state the instant it unmounts — so "hidden" here means visually hidden via CSS, NOT skipped. The
    # expanders below always render; the style block just display:none's them (by their keys) when
    # collapsed. The button and the style block are BOTH unconditional, so the sidebar's element count
    # never changes between show/hide — otherwise the last expander would ghost (see the lock_hint note).
    if st.button(
        "Hide settings" if st.session_state.show_settings else "Settings",
        icon=":material/expand_less:" if st.session_state.show_settings else ":material/tune:",
        use_container_width=True, key="settings_toggle",
        help="Show or hide transcription, vocabulary, and assistant/translation settings. "
             "Audio sources stay visible.",
    ):
        st.session_state.show_settings = not st.session_state.show_settings
        st.rerun()
    _hide_settings_css = (
        "[class*='st-key-sec_transcription'],[class*='st-key-sec_vocab'],"
        "[class*='st-key-sec_assist'],[class*='st-key-lock_hint']{display:none !important;}"
    ) if not st.session_state.show_settings else ""
    st.markdown(f"<style>{_hide_settings_css}</style>", unsafe_allow_html=True)
    # The "locked" hint exists only while recording, so every Stop↔Start (and mute/unmute, via
    # the in-panel hints below) would change the sidebar's element count and shift the index of
    # every expander beneath it. Streamlit reconciles layout containers positionally, so that
    # shift left a ghost copy of the last expander ("Assistant & translation"). Keeping each such
    # conditional inside an ALWAYS-rendered keyed container holds the element count constant — the
    # caption appears/disappears as the container's only child, so nothing below ever shifts.
    with st.container(key="lock_hint"):
        if locked:
            st.caption(":material/lock: Locked while recording — stop to change.")

    with st.expander("Transcription", expanded=True, icon=":material/graphic_eq:",
                     key="sec_transcription"):
        _model_opts = ["tiny", "base", "small", "medium", "large-v3"]
        model_size = st.selectbox(
            "Whisper model", _model_opts,
            index=_seed_index(_model_opts, _persisted.get("model"), 4),
            disabled=locked,
            help="large-v3 is by far the most accurate, especially for Korean — your GPU has "
                 "the headroom for it. Drop to medium/small only if you need lower latency; "
                 "on CPU-only, prefer base/tiny.",
        )
        _primary_opts = ["English (en)", "Korean (ko)"]
        primary = st.selectbox(
            "Primary language", _primary_opts,
            index=_seed_index(_primary_opts, _persisted.get("primary"), 0), disabled=locked,
            help="Speech is auto-detected, constrained to this pair — hands-free for both parties.",
        )
        _secondary_opts = ["Korean (ko)", "English (en)"]
        secondary = st.selectbox(
            "Secondary language", _secondary_opts,
            index=_seed_index(_secondary_opts, _persisted.get("secondary"), 0), disabled=locked,
            help="The other member of the detection pair.",
        )
        # Advanced VAD/decoder tuning lives in a popover so the everyday controls (model +
        # languages) stay uncluttered. Why a popover and not a nested expander or a reveal-toggle:
        # Streamlit forbids expander-in-expander, and a reveal-toggle would *unmount* these
        # widgets when collapsed — Streamlit then purges their state, breaking persistence. A
        # popover keeps them mounted every run, so values survive reruns and seed/save cleanly.
        with st.popover(":material/tune: Advanced", use_container_width=True):
            silence_ms = st.slider(
                "Silence cutoff (ms)", 400, 2000, int(_persisted.get("silence_ms", 500)),
                step=100, disabled=locked,
                help="Silence duration that triggers end of a speech segment. Lower = less "
                     "latency but may split sentences at natural pauses.",
            )
            vad_threshold = st.slider(
                "VAD sensitivity", 0.1, 0.9, float(_persisted.get("vad_threshold", 0.5)),
                step=0.05, disabled=locked,
                help="Lower = picks up quieter speech",
            )
            partial_ms = st.slider(
                "Live preview update (ms)", 0, 2000, int(_persisted.get("partial_ms", 1000)),
                step=250, disabled=locked,
                help="Show a tentative, in-progress transcription this often while someone is "
                     "still speaking, so text streams in instead of appearing only when they "
                     "pause. The finalized line replaces it at the end of each utterance. "
                     "0 = off (finalized lines only). Lower = snappier but more compute.",
            )
            beam_size = st.slider(
                "Beam size", 1, 10, int(_persisted.get("beam_size", 5)), step=1, disabled=locked,
                help="Whisper decoding beams for finalized lines: higher = more accurate but "
                     "slower, 1 = greedy. Live previews always use 1. Bump this to clean up "
                     "occasional misrecognitions. Takes effect at the next Start.",
            )

    # Proper-noun vocabulary: editable even while recording (the Transcriber re-reads it per
    # utterance), since the whole point is to correct a name mid-conversation.
    with st.expander("Proper nouns / vocabulary", expanded=False, icon=":material/spellcheck:",
                     key="sec_vocab"):
        glossary = st.session_state.glossary
        st.caption(":material/info: Names, companies, or jargon Whisper mis-hears. The Gemini "
                   "cleanup/translation keeps these spellings automatically — **no model "
                   "retraining**, effective on the next utterance. Keep the list focused.")
        # Seed the widget once so we don't pass value= alongside key= on every rerun (Streamlit
        # warns on that). The widget then owns its state across reruns.
        if "vocab_text" not in st.session_state:
            st.session_state.vocab_text = "\n".join(glossary.terms())
        st.text_area(
            "One term per line", key="vocab_text", height=120,
            placeholder="Acme Corp\nSilero\n김민준",
            help="Proper nouns, product names, acronyms — one per line.",
        )
        # Persist only on actual change (this reruns every ~0.3s while recording). Mirror the
        # Glossary's own cleaning so the comparison is stable and we don't rewrite every poll.
        _seen: set = set()
        _candidate = []
        for _t in st.session_state.vocab_text.splitlines():
            _t = _t.strip()
            if _t and _t.lower() not in _seen:
                _seen.add(_t.lower())
                _candidate.append(_t)
        if _candidate != glossary.terms():
            glossary.set_terms(_candidate)
        st.caption(f":material/sell: {len(glossary.terms())} term"
                   f"{'' if len(glossary.terms()) == 1 else 's'} active.")
        hotwords_on = st.toggle(
            "Also bias the speech recognizer (hotwords)",
            value=bool(_persisted.get("hotwords_on", False)),
            help="Off by default. When on, the terms are also fed to Whisper's decoder as "
                 "hotwords. That can rescue a name Whisper keeps mangling — but it conditions the "
                 "decode, so it tends to *over*-force the terms, inserting them on short or quiet "
                 "audio. Leave it off unless a specific name won't come out right; the glossary "
                 "above already fixes most spellings during cleanup without the forcing.",
        )
        st.session_state.hotwords_gate["on"] = bool(hotwords_on)

    with st.expander("Audio sources", expanded=True, icon=":material/headphones:",
                     key="sec_sources"):
        sources = list_audio_sources()
        by_key = {s.key: s for s in sources}
        keys = st.session_state.selected_source_keys

        # Drive the per-row meters while idle: open/close a lightweight level monitor per
        # resolved, unmuted source (no-op while recording, where the captures do this). Runs
        # before the rows render so each row can read its monitor's level this same run.
        _reconcile_level_monitors(by_key, keys, set(st.session_state.muted_source_keys))

        # One always-present keyed box holds the rows, so the expander's own child count never
        # changes when sources are added/removed. Inside it each row is its own keyed container,
        # so deleting a source reconciles by key and can't leave the removed row's mute/gear/close
        # buttons ghosting (see _render_source_rows; ghost-container guard, dynamic-list variant).
        with st.container(key="source_rows_box"):
            _render_source_rows(by_key, keys, locked)

        # Inline edit panel for the gear-selected source (replaces the old @st.dialog), in an
        # always-present keyed slot after the rows so toggling it can't change the child count.
        with st.container(key="edit_source_panel"):
            # Render the panel into an st.empty() slot, and explicitly clear that slot when closed.
            # A bare keyed container that produces ZERO children on close does NOT emit a clear delta,
            # so the previously-rendered widgets ghost on screen — that's why the gear "wouldn't
            # collapse" even though edit_source_key was already None. st.empty()'s replace/clear
            # semantics force the removal. (ghost-container guard, render-side.)
            _edit_slot = st.empty()
            _ek = st.session_state.get("edit_source_key")
            if _ek is not None and _ek in keys:
                with _edit_slot.container():
                    _render_edit_source_panel(_ek)
            else:
                _edit_slot.empty()
                if _ek is not None:
                    st.session_state.edit_source_key = None  # source removed while panel was open

        # Not locked while recording: a source is purely additive (shared queue, per-source
        # capture), so it can be added live without a restart — its capture spins up on the next
        # run via _reconcile_live_captures. Retarget/remove still require Stop.
        if st.button("Add source", icon=":material/add:", use_container_width=True):
            st.session_state.show_add_source = True
            st.rerun()
        # Inline add-source panel (replaces the old @st.dialog), always-present keyed slot.
        with st.container(key="add_source_panel"):
            _add_slot = st.empty()  # st.empty() so the panel fully clears on close (see edit slot)
            if st.session_state.get("show_add_source"):
                with _add_slot.container():
                    _render_add_source_panel()
            else:
                _add_slot.empty()

        # Resolve chosen keys → sources, preserving order and dropping any that no longer exist
        # (e.g. an app closed since it was added). capture_sources is every resolved source —
        # each gets a capture at Start so mute/unmute can take effect live; selected_sources is
        # the unmuted subset, used for the at-a-glance live count and the echo-dedup clean tags.
        muted_keys = set(st.session_state.muted_source_keys)
        capture_sources = [by_key[k] for k in keys if k in by_key]
        selected_sources = [s for s in capture_sources if s.key not in muted_keys]

        # Add-while-recording: if a source was added through the picker mid-run, bring up its
        # capture now (no-op otherwise). Runs here, where the live silence/VAD/preview settings
        # are in scope, so the new capture matches the running session's tuning.
        if st.session_state.running:
            _reconcile_live_captures(by_key, keys, muted_keys, selected_sources,
                                     silence_ms, vad_threshold, partial_ms)

        if not any(s.kind == "loopback" for s in sources):
            from utils.audio import loopback_error
            why = loopback_error()
            st.caption(f":material/info: No system-audio (loopback) source found{f' — {why}' if why else ''}. "
                       "Install `soundcard` (`pip install soundcard`) to capture program/system audio.")

    with st.expander("Assistant & translation", expanded=True, icon=":material/smart_toy:",
                     key="sec_assist"):
        # Push-to-ask can only hear you through a live mic, so the assistant toggle follows mic
        # state — it un-greys the moment you unmute. The assistant/translate toggles take effect
        # live while recording (applied to the running workers just below); only the hotkey
        # binding and translation direction stay fixed for the session.
        has_live_mic = any(s.kind == "mic" for s in selected_sources)
        _hold_opts = list(HOLD_KEYS.keys())
        hold_key = st.selectbox(
            "Push-to-ask key", _hold_opts,
            index=_seed_index(_hold_opts, _persisted.get("hold_key"),
                              _hold_opts.index(DEFAULT_HOLD_KEY)),
            format_func=lambda k: HOLD_KEYS[k], disabled=locked,
            help="Global OS hotkey. Hold it and speak to ask the AI assistant instead of "
                 "transcribing into the conversation; release when done.",
        )
        assist_on = st.toggle(
            "Ask assistant (Gemini + web search)",
            value=bool(_persisted.get("assist_on", gemini_key_present)),
            disabled=locked and not has_live_mic,
            help="Hold the push-to-ask key and speak a question; the assistant searches the "
                 "web (Google Search grounding) and answers in the Assistant panel.",
        )
        # Keyed, always-present slots so these hints toggling (on unmute / key state) never
        # change this expander's child count — see the lock_hint note above.
        with st.container(key="assist_mic_hint"):
            if locked and not has_live_mic:
                st.caption(":material/mic_off: Push-to-ask is paused — unmute a mic to ask the assistant.")
        with st.container(key="assist_key_warn"):
            if assist_on and not gemini_key_present:
                st.warning("GEMINI_API_KEY not set — assistant disabled. Free key at "
                           "aistudio.google.com/apikey.")
        # Keyed so the phone can flip it too (parallel switch): the phone-command drain at the top
        # of the script writes st.session_state.translate_on before this widget renders, and the
        # desktop toggle reads/writes the same key — either surface drives translation. Seeded once
        # (rather than value=) so setting the key from the phone-drain doesn't fight a value= arg.
        if "translate_on" not in st.session_state:
            st.session_state.translate_on = bool(_persisted.get("translate_on", False))
        translate_on = st.toggle(
            "Translate (Gemini)", key="translate_on",
            help="KO↔EN translation via Gemini. Also togglable from the phone.",
        )
        _dir_opts = ["Auto (to the other language)", "Korean → English", "English → Korean"]
        direction = st.selectbox(
            "Translation direction", _dir_opts,
            index=_seed_index(_dir_opts, _persisted.get("direction"), 0),
            disabled=locked or not translate_on,
        )
        # Off by default: instrumentation showed raw Whisper is usually more faithful than the
        # Gemini rewrite. Live (set_cleanup below), so NOT locked while recording — only gated on
        # translation being on, since the rewrite is a by-product of the same Gemini call.
        cleanup_on = st.toggle(
            "Polish transcript (AI cleanup)",
            value=bool(_persisted.get("cleanup_on", False)),
            disabled=not translate_on,
            help="While translating, also let Gemini rewrite the *original* transcript line to "
                 "fix ASR errors, spacing, and punctuation. Off (default) shows Whisper's raw "
                 "output, which is usually more faithful — the rewrite can drift. The translation "
                 "is produced either way.",
        )
        with st.container(key="translate_key_warn"):
            if translate_on and not gemini_key_present:
                st.warning("GEMINI_API_KEY not set — translation disabled. Transcription still works.")

        # Apply the toggles to the already-running workers so a mid-recording flip takes effect
        # without a restart (both loops read .enabled live; the gate drives push-to-ask). The
        # widgets above stay in sync because their state persists across the 0.3s poll reruns.
        if st.session_state.running:
            if st.session_state.translator is not None:
                st.session_state.translator.set_enabled(translate_on)
                st.session_state.translator.set_cleanup(cleanup_on)
            if st.session_state.agent is not None:
                st.session_state.agent.set_enabled(assist_on)
                # Live context preference (the phone's Context button drives use_context above).
                st.session_state.agent.set_use_context(bool(st.session_state.use_context))
            if st.session_state.command_gate is not None:
                st.session_state.command_gate["on"] = assist_on

    # Persist the full sidebar state so it survives a restart (utils/settings.py). Cheap: this
    # script reruns ~3×/s while recording, but we only touch disk when something actually changed
    # (steady state writes nothing). Stale source keys are harmless — the source rows render them
    # as "Unavailable" and drop them from the live set, so restoring them never crashes.
    _current_settings = {
        "model": model_size, "primary": primary, "secondary": secondary,
        "silence_ms": silence_ms, "vad_threshold": vad_threshold, "partial_ms": partial_ms,
        "beam_size": beam_size,
        "hold_key": hold_key, "assist_on": bool(assist_on),
        "translate_on": bool(st.session_state.translate_on), "direction": direction,
        "cleanup_on": bool(cleanup_on),
        "hotwords_on": bool(hotwords_on),
        "echo_suppress": bool(st.session_state.echo_suppress),
        "use_context": bool(st.session_state.use_context),
        "selected_source_keys": list(st.session_state.selected_source_keys),
        "muted_source_keys": list(st.session_state.muted_source_keys),
    }
    if _current_settings != _persisted:
        save_settings(_current_settings)
        st.session_state._persisted = _current_settings
        _persisted = _current_settings


# Make the sidebar selectboxes true read-only pickers. Streamlit's st.selectbox has no
# `searchable=False`, and the CSS above only hides the caret — once BaseWeb focuses the
# embedded <input> you can still type to filter. So we run a small script that sets
# input.readOnly = true: still focusable + arrow/Enter selectable (works as a picker) but
# ignores typing. st.html injects inline into the app document (not an iframe), so the
# script reaches the selectboxes directly — no window.parent / cross-origin hop; it replaces
# the deprecated st.components.v1.html srcdoc trick. unsafe_allow_javascript=True lets the
# script execute. A MutationObserver re-applies readOnly across Streamlit reruns (which
# recreate the inputs); the window guard keeps a single observer from accumulating on reruns.
# The same script also fixes a BaseWeb quirk where clicking an open dropdown to dismiss it
# flickers shut and reopens (see the inline comment for the focus-race detail).
st.html(
    """
    <script>
      if (!window.__selectLockInstalled) {
        window.__selectLockInstalled = true;
        const lock = () => document.querySelectorAll('[data-baseweb="select"] input')
          .forEach(el => { if (!el.readOnly) el.readOnly = true; });
        lock();
        const sidebar = document.querySelector('[data-testid="stSidebar"]');
        new MutationObserver(lock).observe(sidebar || document.body, {childList: true, subtree: true});

        // A second click on an OPEN selectbox should dismiss it. BaseWeb's searchable Select
        // races a focus-driven reopen (openAfterFocus) against its own click-to-close, so the
        // menu flickers shut then springs back. The fix is just the mousedown half: when the
        // control — not an option — is clicked while open, preventDefault the mousedown in the
        // capture phase. That blocks the input from refocusing, so openAfterFocus can't fire,
        // while BaseWeb's own (untouched) click handler still toggles the open menu shut.
        // Do NOT also swallow the trailing click — that suppresses the toggle-close and leaves
        // the menu stuck open on a normal click (only a long press, which let an internal timer
        // lapse, slipped a real click through and closed it).
        const dismissTarget = (t) => {
          if (!t || !t.closest) return null;
          // A click inside the open menu is a selection, never a dismiss — leave it alone.
          if (t.closest('[role="listbox"],[role="option"],[data-baseweb="popover"],[data-baseweb="menu"]')) return null;
          const sel = t.closest('[data-baseweb="select"]');
          if (!sel) return null;
          const open = sel.matches('[aria-expanded="true"]') || sel.querySelector('[aria-expanded="true"]');
          return open ? sel : null;
        };
        document.addEventListener('mousedown', (e) => {
          const sel = dismissTarget(e.target);
          if (!sel) return;                 // closed (or not a select) → let BaseWeb open it
          e.preventDefault();               // block the refocus → openAfterFocus reopen
          e.stopPropagation();
          const input = sel.querySelector('input');
          if (input) input.blur();          // drop focus; the click then toggles it closed
        }, true);
      }
    </script>
    """,
    unsafe_allow_javascript=True,
)

# Tab-switch recovery. The live UI is driven by a server-side `time.sleep; st.rerun()` poll that
# only runs while this tab's Streamlit WebSocket is connected. Backgrounding the PC tab lets Chrome
# freeze it: the socket times out, the server halts the rerun loop, and the page sticks on its last
# frame (levels bar frozen, no response) even after you return. The capture/transcribe threads keep
# running server-side the whole time and the bus sink is unbounded, so nothing is lost — the only
# problem is the UI never resumes on its own. On refocus we kick Streamlit's connection manager to
# reconnect to the SAME session (a freeze preserves its in-memory session id; a page *reload* would
# not — it starts a fresh session and drops the transcript, which is why we never reload here).
# Reconnecting triggers a rerun, which drains everything captured while away. Dispatching `online`
# is the documented nudge that pulls Streamlit out of its reconnect backoff immediately instead of
# waiting it out; we fire it a few times because the JS engine is still spinning up right after a
# freeze and a single event can be missed.
st.html(
    """
    <script>
      if (!window.__sttTabRecover) {
        window.__sttTabRecover = true;
        // Probe: if this line doesn't appear in the PC browser console, st.html isn't executing
        // injected <script> in this Streamlit build and the recovery below is a no-op (see review).
        console.log('[stt] tab-recover installed');
        const wake = () => {
          if (document.visibilityState !== 'visible') return;
          let n = 0;
          const kick = () => {
            try { window.dispatchEvent(new Event('online')); } catch (e) {}
            if (++n < 5) setTimeout(kick, 400);
          };
          kick();
        };
        // ONLY real tab background→foreground transitions: visibilitychange (tab hide/show)
        // and pageshow (bfcache restore). We deliberately do NOT listen on window 'focus':
        // that also fires when you alt-tab back from ANOTHER app while this browser tab stayed
        // visible (visibilityState never left 'visible') — e.g. clicking back after starting
        // playback to set up a System Audio source. Each such focus fired the `online` storm,
        // and a reconnect rerun landing mid-interaction pre-empted button handlers (historically
        // the Add-source modal wouldn't close; the inline panels are sturdier but the guard stays).
        document.addEventListener('visibilitychange', wake);
        window.addEventListener('pageshow', wake);
      }
    </script>
    """,
    unsafe_allow_javascript=True,
)

# Tab-title flicker fix. Streamlit re-runs the whole script on every poll tick — this app does so
# ~3×/s via the time.sleep+st.rerun loop at the bottom. On each run the frontend briefly resets
# the tab title to its default ("main · Streamlit") BEFORE our st.set_page_config message re-asserts
# page_title, so at ~3 resets/second the tab text visibly flickers. set_page_config can't stop it
# (the reset lands first). Pin the title client-side instead: a MutationObserver on <head> re-asserts
# our title the instant Streamlit clears it — within the same frame, so the reset never paints.
# Idempotent install (window guard); we only write when the title actually differs, so our own write
# can't loop. Observing <head> (not just the <title> node) also survives Streamlit replacing the
# element wholesale rather than mutating its text. WANT is injected from the same PAGE_TITLE constant
# set_page_config uses, so the two can never drift.
st.html(
    f"""
    <script>
      if (!window.__sttTitleLock) {{
        window.__sttTitleLock = true;
        const WANT = {json.dumps(PAGE_TITLE)};
        const pin = () => {{ if (document.title !== WANT) document.title = WANT; }};
        pin();
        new MutationObserver(pin).observe(
          document.head, {{childList: true, subtree: true, characterData: true}});
      }}
    </script>
    """,
    unsafe_allow_javascript=True,
)

lang_map = {"Korean (ko)": "ko", "English (en)": "en"}
primary_code = lang_map[primary]
secondary_code = lang_map[secondary]
mode_map = {
    "Auto (to the other language)": "auto",
    "Korean → English": "ko-en",
    "English → Korean": "en-ko",
}
translate_mode = mode_map[direction]


# --- Header bar (brand + live-status pill) -------------------------------------------
running = st.session_state.running
ctrl = st.session_state.command_hotkey
held = bool(running and ctrl and ctrl.is_held())
pill_cls = "listen" if held else ("live" if running else "")
pill_txt = "Listening" if held else ("Live" if running else "Idle")
st.markdown(
    "<div class='app-header'>"
    "<div class='brand'>"
    f"<div class='brand-mark'>{_ms('graphic_eq', 24)}</div>"
    "<div class='brand-txt'>"
    "<span class='brand-title'>Realtime STT</span>"
    "<span class='brand-sub'>Korean ↔ English live transcription &amp; assistant</span>"
    "</div></div>"
    f"<div class='status-pill {pill_cls}'><span class='dot'></span>{pill_txt}</div>"
    "</div>",
    unsafe_allow_html=True,
)

# --- Config strip: the active setup at a glance --------------------------------------
_src_n = len(selected_sources)
_chips = [
    f"<span class='cfg'>{_ms('memory', 15)}<b>{model_size}</b> model</span>",
    f"<span class='cfg'>{_ms('language', 15)}<b>{primary_code.upper()} ↔ {secondary_code.upper()}</b></span>",
    f"<span class='cfg'>{_ms('headphones', 15)}<b>{_src_n}</b> source{'' if _src_n == 1 else 's'}</span>",
    f"<span class='cfg {'on' if translate_on else 'off'}'>"
    f"{_ms('g_translate', 15)}Translate</span>",
    f"<span class='cfg {'on' if assist_on else 'off'}'>{_ms('smart_toy', 15)}Assistant</span>",
]
st.markdown(f"<div class='cfg-strip'>{''.join(_chips)}</div>", unsafe_allow_html=True)

# --- Phone status bar + connect/disconnect toasts ------------------------------------
# Only when a Phone source is in the selected list. current_server() never starts the server —
# it just reads the live singleton (started lazily by the source's reader/monitor), so this
# poll is side-effect-free. is_connected is wall-clock-fresh (audio within ~3s), so a phone
# that locks its screen flips to "Waiting" on its own.
_phone_present = any(k.startswith("phone:") for k in st.session_state.selected_source_keys)
if _phone_present:
    from utils.phone_server import current_server
    _phone_srv = current_server()
    _phone_conn = bool(_phone_srv and _phone_srv.is_connected)

    # Mirror the authoritative control state to the phone (wait-for-echo): a desktop-side flip of
    # translate/mute, or the recording state itself, reaches the phone here without it having asked.
    # set_phone_state no-ops unless something changed, so this is cheap on the ~3/s poll.
    #
    # ONLY the recording session mirrors state. The phone server is a process-wide singleton SHARED
    # by every Streamlit session (browser tab), so an idle session that merely has the phone selected
    # would otherwise push recording=False on its own ~0.3s poll and clobber the recording session's
    # recording=True every tick — strobing the phone's "Press Start…" hint. Gating on `running` makes
    # exactly one session (the live one) the writer; idle/extra tabs stay silent. The Stop handler
    # pushes a one-shot recording=False so the phone disables its controls when recording ends.
    if _phone_srv is not None and running:
        _phone_srv.set_phone_state(
            translation=bool(translate_on),
            mute=("phone:default" in st.session_state.muted_source_keys),
            recording=True,
            # Gates the phone's Ask button: no point opening a listening window if the assistant
            # is off (the spoken question would just land in the transcript with no answer).
            assist=bool(assist_on),
            # Reflects the Context toggle so the phone button renders the current truth on connect.
            context=bool(st.session_state.use_context),
        )

    # Edge-detected toasts: fire once per transition (the ~3/s poll would otherwise repeat them).
    _prev_conn = st.session_state.get("_phone_conn_prev", False)
    if _phone_conn and not _prev_conn:
        st.toast("Phone connected — mic is streaming.", icon="📱")
    elif not _phone_conn and _prev_conn:
        st.toast("Phone disconnected.", icon="🔌")
    st.session_state["_phone_conn_prev"] = _phone_conn

    # Live level: while recording the phone's capture provides it; while idle its LevelMonitor
    # does (same providers the sidebar row reads) — so the meter moves before Start, too.
    _pp = (st.session_state.capture_by_key.get("phone:default")
           or st.session_state.level_monitors.get("phone:default"))
    _frac = _level_frac(_pp.level()) if _pp is not None else 0.0
    _mcolor = ("var(--danger)" if _frac >= 0.9 else
               "var(--listen)" if _frac >= 0.7 else "var(--live)")
    _state_txt = "Streaming" if _phone_conn else "Waiting for phone…"
    st.markdown(
        f"<div class='phone-bar {'on' if _phone_conn else ''}'>{_ms('smartphone', 18)}"
        f"<span class='pb-label'>Phone</span>"
        f"<span class='pb-state'><span class='dot'></span>{_state_txt}</span>"
        f"<div class='lvl'><i style='width:{_frac * 100:.0f}%;background:{_mcolor}'></i></div>"
        f"</div>",
        unsafe_allow_html=True,
    )
else:
    # No phone selected → reset the edge tracker so re-adding a phone later toasts cleanly.
    st.session_state["_phone_conn_prev"] = False

# --- Controls ---
col1, col2 = st.columns(2)
start_btn = col1.button("Start", icon=":material/play_arrow:", disabled=running,
                        use_container_width=True, type="primary")
stop_btn = col2.button("Stop", icon=":material/stop:", disabled=not running,
                       use_container_width=True)

if start_btn and not capture_sources:
    st.warning("Select at least one audio source before starting.")
elif start_btn:
    # Release the idle level-monitors before opening the real captures so a device is never
    # opened twice (reconcile won't rebuild them now that we're about to be running).
    for mon in st.session_state.level_monitors.values():
        mon.stop()
    st.session_state.level_monitors = {}
    with st.spinner(f"Loading {model_size} model…"):
        whisper = load_whisper_model(model_size)
        # One Silero instance per source (stateful RNN — must not be shared).
        vads = [load_vad_model(i) for i in range(len(capture_sources))]

    from utils.audio import AudioCapture, make_reader
    from utils.transcriber import Transcriber
    from utils.translator import Translator
    from utils.agent import AgentService
    from utils.context import ConversationStore
    from utils.events import TranscriptBus
    from utils.hotkey import CommandHotkey

    seg_q: queue.Queue = queue.Queue()
    # Slots 0..N-1 go to the sources started here; live adds continue the count from N (see
    # _reconcile_live_captures) so no two captures ever share a stateful Silero slot.
    st.session_state.vad_slot_seq = len(capture_sources)
    transcript_q: queue.Queue = queue.Queue()
    command_q: queue.Queue = queue.Queue()     # command utterances → the agent
    partial_q: queue.Queue = queue.Queue()     # interim (in-progress) decodes → the UI live preview
    assistant_q: queue.Queue = queue.Queue()   # AssistantMessage → the UI panel
    bus = TranscriptBus()
    sink: queue.Queue = queue.Queue()
    bus.subscribe(sink.put)  # the UI is one bus subscriber; agents would be another

    controller = CommandHotkey(hold_key=hold_key)
    controller.start()  # best-effort OS hotkey; without it, push-to-ask is unavailable

    # One capture per source, all feeding the shared segment queue tagged with their
    # channel. The push-to-ask key marks *your* speech as a command — wire it to mic
    # sources only (and only when the assistant is on); loopback (the other party / system
    # audio) is never a command channel.
    # One capture per resolved source. Muted sources start silenced but still get a live
    # capture so they can be unmuted mid-recording (and live ones muted) without spinning
    # capture threads up and down. capture_by_key lets a row's mute toggle reach its capture.
    # Live push-to-ask gate: a tiny mutable shared by every mic capture's command check, so
    # toggling "Ask assistant" mid-recording takes effect at once (the closure reads it on each
    # speech onset, from the capture thread). Wired on ALL mics — not just when the assistant
    # started on — so unmuting a mic + flipping the toggle later still enables push-to-ask. When
    # off, the key never marks a command, so held-key speech is just transcribed normally.
    command_gate = {"on": assist_on}
    st.session_state.command_gate = command_gate
    # Phone ask gate: the phone's "Ask assistant" two-tap window flips this on/off (above). It's
    # the phone's equivalent of the desktop hold-to-ask key — the phone capture's get_command
    # reads it, so while it's on the phone's speech is captured as one command for the agent.
    phone_ask_gate = {"on": False}
    st.session_state.phone_ask_gate = phone_ask_gate
    captures = []
    capture_by_key = {}
    for src, vad in zip(capture_sources, vads):
        cap = AudioCapture(
            seg_q, vad, make_reader(src), source=src.tag,
            silence_ms=silence_ms, threshold=vad_threshold,
            partial_interval_s=partial_ms / 1000.0,
            get_command=_make_command_gate(src.kind, controller, command_gate, phone_ask_gate),
            muted=src.key in muted_keys,
        )
        captures.append(cap)
        capture_by_key[src.key] = cap
    # Command segments are split off to command_q (→ agent); interim decodes go to partial_q
    # (→ UI live preview, never translated); everything else flows to the translator and on
    # to the transcript.
    transcriber = Transcriber(
        seg_q, transcript_q, whisper,
        lang_pair=(primary_code, secondary_code), command_queue=command_q,
        partial_queue=partial_q,
        # Final-decode beam width from the Advanced popover (fixed for the session, like model).
        beam_size=beam_size,
        # Proper-noun biasing, read live each decode so vocabulary edits apply mid-recording.
        # Gated by the (default-off) hotwords toggle: when off, return "" so the decode gets no
        # hotwords at all — the terms still reach Gemini via the glossary block below.
        get_hotwords=(lambda g=st.session_state.glossary, gate=st.session_state.hotwords_gate:
                      g.hotwords() if gate["on"] else ""),
    )
    # Crosstalk is one-directional (system → mic), so loopback tags are the clean/authoritative
    # side; mic lines duplicating them are dropped. No loopback selected ⇒ nothing to dedup.
    clean_tags = {src.tag for src in selected_sources if src.kind in ("loopback", "process")}
    translator = Translator(
        transcript_q, bus, mode=translate_mode, enabled=translate_on,
        echo_suppress=st.session_state.echo_suppress and bool(clean_tags), clean_sources=clean_tags,
        # Show Gemini's rewrite of the original only when the user opted in (default off → raw ASR).
        cleanup=cleanup_on,
        # Glossary as a text backstop: keep proper-noun spellings through cleanup/translation.
        get_glossary_block=st.session_state.glossary.as_prompt_block,
    )
    # Conversation memory the assistant can query on demand. A bus subscriber (the seam's whole
    # point): ingest is append-only and never calls Gemini, and compaction runs on demand inside
    # an ask (ConversationStore.compact_now), so it makes zero Gemini calls when the assistant is
    # off — or on but never invoked.
    context_store = ConversationStore()
    bus.subscribe(context_store.ingest)
    agent = AgentService(command_q, assistant_q, enabled=assist_on, context_store=context_store,
                         use_context=bool(st.session_state.use_context))

    for capture in captures:
        capture.start()
    transcriber.start()
    translator.start()
    context_store.start()
    agent.start()

    st.session_state.bus = bus
    st.session_state.context_store = context_store
    st.session_state.event_sink = sink
    st.session_state.partial_sink = partial_q
    st.session_state.partials = {}
    st.session_state.committed_ids = set()
    st.session_state.assistant_sink = assistant_q
    st.session_state.audio_captures = captures
    st.session_state.capture_by_key = capture_by_key
    st.session_state.seg_q = seg_q  # exposed so add-while-recording can wire new captures into it
    st.session_state.transcriber = transcriber
    st.session_state.translator = translator
    st.session_state.agent = agent
    st.session_state.command_hotkey = controller
    # Discard any phone control taps queued while stopped (e.g. a tap that beat the phone's
    # "recording off" state) so they can't fire late against the freshly-started session.
    from utils.phone_server import current_server as _cps_start
    _ps_start = _cps_start()
    if _ps_start is not None:
        _ps_start.pop_commands()
    # Leave any post-stop edit mode (the transcript is about to grow again).
    st.session_state.editing = False
    st.session_state.pop("_edit_buffer", None)
    st.session_state.running = True
    st.rerun()

if stop_btn:
    for capture in st.session_state.audio_captures:
        capture.stop()
    if st.session_state.transcriber:
        st.session_state.transcriber.stop()
    if st.session_state.translator:
        st.session_state.translator.stop()
    if st.session_state.agent:
        st.session_state.agent.stop()
    if st.session_state.context_store:
        st.session_state.context_store.stop()
    if st.session_state.command_hotkey:
        st.session_state.command_hotkey.stop()
    # Workers are stopped — drain any finalized lines still in the sink so the saved transcript is
    # complete, then persist this meeting authoritatively (autosave only ran every few seconds).
    if st.session_state.event_sink is not None:
        while not st.session_state.event_sink.empty():
            ev = st.session_state.event_sink.get_nowait()
            st.session_state.transcript.append(ev)
    _save_active_session()
    st.session_state.running = False
    # This session is the phone's state writer only while recording (see the mirror block above), so
    # push the final recording=False here — otherwise the phone would keep its controls enabled until
    # some other session happened to write. assist=False greys the Ask button in the same frame.
    _ps_stop = _get_phone_server()
    if _ps_stop is not None:
        _ps_stop.set_phone_state(recording=False, assist=False)
    st.session_state.command_gate = None
    st.session_state.seg_q = None
    st.session_state.audio_captures = []
    st.session_state.capture_by_key = {}
    st.session_state.transcriber = None
    st.session_state.translator = None
    st.session_state.agent = None
    st.session_state.context_store = None
    st.session_state.command_hotkey = None
    st.session_state.bus = None
    st.session_state.event_sink = None
    st.session_state.partial_sink = None
    st.session_state.partials = {}
    st.session_state.committed_ids = set()
    st.session_state.assistant_sink = None
    st.session_state.phone_ask_gate = None  # listening window can't outlive the captures it drove
    st.rerun()

if running and ctrl and not ctrl.available:
    st.caption(":material/warning: Push-to-ask key unavailable (pynput not active) — "
               "transcription only.")

# --- Assistant panel ---
_section("smart_toy", "Assistant")
if not st.session_state.assistant_log:
    _empty("forum",
           "No questions yet",
           "Hold the push-to-ask key and speak a question — grounded answers appear here.")
else:
    for msg in st.session_state.assistant_log:
        with st.container(border=True):
            st.markdown(
                f"<div class='ask-q'>{_ms('record_voice_over', 19)}"
                f"<span>{html.escape(msg.query)}</span></div>",
                unsafe_allow_html=True,
            )
            st.markdown("<div class='ask-divider'></div>", unsafe_allow_html=True)
            if msg.status == "pending":
                # Shimmer skeleton conveys "working" without a frozen UI (UX: loading-states).
                st.markdown(
                    f"<div class='ask-pending'>{_ms('search', 17)}Searching the web…</div>"
                    "<div class='skel' style='width:92%'></div>"
                    "<div class='skel' style='width:80%'></div>"
                    "<div class='skel' style='width:64%'></div>",
                    unsafe_allow_html=True,
                )
            else:
                # Answer is the model's output; st.markdown without unsafe_allow_html escapes
                # any raw HTML while still rendering markdown (bold, links).
                st.markdown(msg.answer or "")
                if msg.citations:
                    # Escape markdown-significant chars in the title so a ']' or ')' in a
                    # grounding source title can't malform the link.
                    links = " · ".join(
                        f"[{_md_escape(t)}]({u})" for t, u in msg.citations
                    )
                    st.caption(f":material/link: Sources: {links}")

# --- Transcript ---
_section("forum", "Transcript")
# Freeform post-stop editing: once stopped, the transcript can be edited like a text document
# (fix names, delete junk) before a report. The edited text then becomes the source of truth for
# the report and the saved session — per-line source/translation structure is intentionally given
# up once you edit as plain text. Unavailable while recording (the list is still growing).
_has_content = bool(st.session_state.transcript or st.session_state.edited_text)

if st.session_state.editing:
    st.caption(":material/edit: Editing transcript — fix or delete anything, then Save. "
               "This text becomes the basis for the report.")
    # Seed the buffer before the widget renders (Streamlit forbids setting a widget-keyed value
    # after instantiation); a pending reset repopulates it from the captured lines.
    if st.session_state.pop("_edit_reset", False):
        st.session_state._edit_buffer = _transcript_to_text(st.session_state.transcript)
    if "_edit_buffer" not in st.session_state:
        # Seed from the *effective* text (the edit plus any lines recorded after it) so re-editing
        # after a resumed recording shows the continuation in the editor instead of dropping it.
        st.session_state._edit_buffer = (
            _effective_transcript_text() if st.session_state.edited_text is not None
            else _transcript_to_text(st.session_state.transcript))
    st.text_area("Transcript", key="_edit_buffer", height=440, label_visibility="collapsed")
    e1, e2, e3 = st.columns([0.28, 0.28, 0.44])
    if e1.button("Save", type="primary", icon=":material/check:", use_container_width=True):
        # Don't pop _edit_buffer here: it's a live widget key this run, and clearing a widget-keyed
        # value after the widget was instantiated raises. The Edit button re-seeds it next entry.
        st.session_state.edited_text = st.session_state._edit_buffer
        # The edit now covers every event captured so far; anything recorded later is appended as a
        # continuation (see _effective_transcript_text), not shadowed by this snapshot.
        st.session_state.edited_base_count = len(st.session_state.transcript)
        st.session_state.editing = False
        _save_active_session()
        st.rerun()
    if e2.button("Cancel", icon=":material/close:", use_container_width=True):
        st.session_state.editing = False
        st.rerun()
    if e3.button("Reset from captured lines", icon=":material/restart_alt:",
                 use_container_width=True,
                 help="Discard edits and reload the originally transcribed lines into the editor."):
        st.session_state["_edit_reset"] = True
        st.rerun()
else:
    tcol1, tcol2, tcol3 = st.columns(3)
    if tcol1.button("Clear transcript", icon=":material/delete_sweep:", use_container_width=True):
        st.session_state.transcript = []
        st.session_state.partials = {}
        st.session_state.committed_ids = set()
        st.session_state.report = None
        st.session_state.edited_text = None
        st.session_state.edited_base_count = 0
        st.session_state.pop("_tx_html", None)
        _save_active_session()
        st.rerun()
    if tcol2.button("Edit transcript", icon=":material/edit:", use_container_width=True,
                    disabled=st.session_state.running or not _has_content,
                    help="Edit the transcript text before a report (stop recording first)."):
        st.session_state.editing = True
        st.session_state.pop("_edit_buffer", None)
        st.rerun()
    # Report generation is a post-meeting action — available once stopped with content.
    gen_report = tcol3.button(
        "Generate report", icon=":material/summarize:", use_container_width=True,
        disabled=st.session_state.running or not _has_content,
        help="Summarize the discussion into a formatted report, in English and Korean.",
    )
    if gen_report:
        if not gemini_key_present:
            st.warning("GEMINI_API_KEY not set — can't generate a report.")
        else:
            from utils.reporter import generate_report
            with st.spinner("Writing up the report…"):
                try:
                    st.session_state.report = generate_report(
                        st.session_state.transcript,
                        glossary_block=st.session_state.glossary.as_prompt_block(),
                        # Edit + any post-edit continuation; None ⇒ unedited, use captured events.
                        transcript_text=_effective_transcript_text(),
                    )
                    _save_active_session()
                except Exception as e:
                    st.session_state.report = None
                    st.error(f"Report generation failed ({type(e).__name__}: {e}).")

    # Render the most recent report (persists until Clear transcript / regenerate).
    report = st.session_state.report
    if report is not None:
        with st.container(border=True):
            st.markdown(f"#### {report.title}")
            # Language picker instead of st.tabs. st.tabs remounts its (un)hidden panels on every
            # rerun, and this page repaints the whole script ~3×/s while idle (the sidebar level-
            # meter poll at the bottom), so the report's EN/한국어 tabs visibly blinked after a
            # meeting even though nothing was recording. A segmented_control is one stable widget
            # and a plain st.markdown reconciles byte-identical content without remounting, so the
            # report sits still across those no-op repaints — same reasoning as the cached
            # transcript box above. Element count stays constant (one markdown + one download
            # button) regardless of selection, so no ghost-duplication either.
            _lang = st.segmented_control(
                "Report language", ["English", "한국어"], default="English",
                key="report_lang", label_visibility="collapsed",
            )
            if _lang == "한국어":
                st.markdown(report.report_ko)
                st.download_button("Download (KO)", report.report_ko, icon=":material/download:",
                                   file_name="meeting_report_ko.md", mime="text/markdown")
            else:  # "English" or deselected (segmented_control allows None) → default to English
                st.markdown(report.report_en)
                st.download_button("Download (EN)", report.report_en, icon=":material/download:",
                                   file_name="meeting_report_en.md", mime="text/markdown")

if st.session_state.editing:
    pass  # the editor is shown above; no read-only transcript body while editing
elif st.session_state.edited_text is not None and not st.session_state.running:
    # Edit plus any lines captured after it (a resumed recording). Show the combined text — that's
    # exactly what the report uses — and tell the user when a continuation has been appended.
    _eff = _effective_transcript_text()
    if _continuation_text():
        st.caption(":material/edit_note: Showing your edited transcript plus the lines recorded "
                   "since — this is what the report uses. Edit again to revise the new lines.")
    else:
        st.caption(":material/edit_note: Showing your edited transcript — this is what the report uses.")
    st.markdown(
        "<div class='stt-scroll' style='white-space:pre-wrap;color:var(--text);"
        f"font-size:.95rem;line-height:1.65'>{html.escape(_eff)}</div>",
        unsafe_allow_html=True,
    )
    if st.button("Revert to captured transcript", icon=":material/undo:",
                 help="Discard the edited version and show the originally transcribed lines."):
        st.session_state.edited_text = None
        st.session_state.edited_base_count = 0
        st.session_state.pop("_tx_html", None)
        _save_active_session()
        st.rerun()
elif not st.session_state.transcript and not st.session_state.partials:
    _empty("graphic_eq",
           "Waiting for speech",
           "Pick your audio sources in the sidebar and press Start — transcribed "
           "lines will stream in here, each tagged by source and language.")
else:
    # The committed transcript is append-only between Clears, and an event is immutable once it
    # lands here (the translator enriches it before publishing), so cache its rendered HTML and
    # render only newly-appended rows each rerun. Otherwise the whole (unbounded) transcript is
    # re-serialized to HTML ~3×/s — a cost that grows with meeting length and runs even during
    # silence. Cache key is the committed count: a smaller count means Clear reset the list, so
    # rebuild; a larger one means new finals landed, so render just the tail and append.
    _tx = st.session_state.transcript
    _cache = st.session_state.get("_tx_html")
    if _cache is None or _cache[0] > len(_tx):       # first run, or transcript cleared/shrank
        _committed = _render_transcript(_tx)
    elif _cache[0] < len(_tx):                        # new committed lines → render just the tail
        _committed = _cache[1] + _render_transcript(_tx[_cache[0]:])
    else:                                            # unchanged → reuse
        _committed = _cache[1]
    st.session_state["_tx_html"] = (len(_tx), _committed)

    # Partials are few and change every preview tick, so they're always rendered fresh.
    inner = _committed
    if st.session_state.running and st.session_state.partials:
        inner += _render_partials(st.session_state.partials)
    # Self-scrolling box: fixed-height, clips overflow, shows its own scrollbar (CSS
    # .stt-scroll). The id lets the pin script below target it.
    st.markdown(f"<div class='stt-scroll' id='stt-scroll'>{inner}</div>",
                unsafe_allow_html=True)

    # Pin to the very bottom so the newest line / live partial is always visible. The box's
    # inner DOM is recreated on each rerun (scrollTop would reset to 0), so we re-pin after
    # every render. The nonce (content length) is baked into the script body, so Streamlit
    # re-executes the script whenever the transcript changes — a new line or a growing
    # partial — but a no-op status poll (byte-identical HTML) doesn't, leaving a brief
    # scroll-up during silence undisturbed. requestAnimationFrame re-pins after layout.
    # st.html injects inline (not an iframe), so it reaches #stt-scroll directly.
    _stt_nonce = len(inner)
    st.html(
        f"""
        <script>
          (function() {{
            function pin() {{
              const box = document.getElementById('stt-scroll');
              if (box) box.scrollTop = box.scrollHeight;
            }}
            pin();
            requestAnimationFrame(pin);
            /* nonce:{_stt_nonce} */
          }})();
        </script>
        """,
        unsafe_allow_javascript=True,
    )

# --- Polling loop (header pill shows status) ---
# The poll runs unconditionally now — no dialog stand-down. The Add/Edit/Rename UIs are inline
# flag-gated panels, not st.dialog fragments, so there is nothing an st.rerun() could leave
# un-pruned, and no run_every re-arm is needed to restart the poll after a close.

# Adaptive poll cadence (constants defined up top, shared with the idle meter fragments). A no-op
# poll still re-executes the whole script, and a meeting is mostly silence — so stay at the
# responsive 0.3s tick only while something is actually live: an in-progress partial, the
# push-to-ask key held, an assistant answer still pending, or a final / partial received in the last
# couple of seconds (stamped as `_last_rx_ts` in the drains above). Otherwise back off to ~1s. The
# recent-receive tail keeps continuous conversation at full cadence; only true pauses slow down.
# This changes ONLY how often the UI repaints — the capture/transcribe threads run regardless, so
# nothing in the pipeline is dropped or delayed; the visible cost is up to ~1s of extra latency
# before the meters/pill react to the first sound after a lull (lower _POLL_IDLE_S to trade).
_poll_active = (
    bool(st.session_state.partials)
    or held
    or any(getattr(m, "status", "") == "pending" for m in st.session_state.assistant_log)
    or (time.time() - st.session_state.get("_last_rx_ts", 0.0) < _ACTIVE_TAIL_S)
)

if st.session_state.running:
    time.sleep(_POLL_FAST_S if _poll_active else _POLL_IDLE_S)
    st.rerun()
elif st.session_state.level_monitors:
    # Not recording, but at least one source is selected: keep a gentle full-app poll alive purely
    # to animate the sidebar level meters (the idle LevelMonitors reconciled in the Audio sources
    # expander). This repaints the whole script ~3×/s, which is invisible for the markdown-based
    # main pane (the report uses segmented_control + st.markdown, which reconciles byte-identical
    # content without remounting — see the report block). A full rerun (not a fragment tick) is what
    # keeps top-level buttons — New session / edit / delete, mute/remove — responsive: each click is
    # picked up by the next poll within ~0.3s. Stands down while a dialog is open/closing.
    time.sleep(_POLL_FAST_S)
    st.rerun()
