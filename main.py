import html
import os
import queue
import time

import streamlit as st

from utils.hotkey import HOLD_KEYS, DEFAULT_HOLD_KEY

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load ANTHROPIC_API_KEY (and friends) from a .env if present. Optional dependency —
# the app still runs transcription-only without it.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

st.set_page_config(page_title="Korean–English Real-Time Assistant", layout="wide")


@st.cache_resource
def load_vad_model():
    from silero_vad import load_silero_vad
    return load_silero_vad()


@st.cache_resource
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


# --- Session state init ---
_defaults = {
    "running": False,
    "transcript": [],       # list[TranscriptEvent]
    "audio_capture": None,
    "transcriber": None,
    "translator": None,
    "language_controller": None,
    "bus": None,
    "event_sink": None,     # queue.Queue fed by a bus subscriber
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Drain the bus sink before rendering so the transcript is always current.
if st.session_state.running and st.session_state.event_sink:
    while not st.session_state.event_sink.empty():
        st.session_state.transcript.append(st.session_state.event_sink.get_nowait())

# --- Layout ---
st.title("Korean–English Real-Time Assistant Transcriber")

api_key_present = bool(os.environ.get("ANTHROPIC_API_KEY"))

with st.sidebar:
    st.header("Settings")
    locked = st.session_state.running

    model_size = st.selectbox(
        "Whisper model",
        ["tiny", "base", "small", "medium"],
        index=2,
        disabled=locked,
        help="small = good speed/accuracy balance. On GPU you can go larger (medium) "
             "with little latency cost; on CPU-only, prefer base/tiny.",
    )
    primary = st.selectbox(
        "Primary language",
        ["English (en)", "Korean (ko)"],
        index=0,
        disabled=locked,
        help="Auto-detected by default (constrained to the pair) — hands-free for the other party.",
    )
    secondary = st.selectbox(
        "Secondary — hold key forces",
        ["Korean (ko)", "English (en)"],
        index=0,
        disabled=locked,
        help="Hold the key to force this language; release to auto-detect over the pair.",
    )
    hold_key = st.selectbox(
        "Hold key (force secondary)",
        list(HOLD_KEYS.keys()),
        index=list(HOLD_KEYS.keys()).index(DEFAULT_HOLD_KEY),
        format_func=lambda k: HOLD_KEYS[k],
        disabled=locked,
        help="Global OS hotkey. Hold to force the secondary language for the next utterances.",
    )
    silence_ms = st.slider(
        "Silence cutoff (ms)", 400, 2000, 500, step=100,
        disabled=locked,
        help="Silence duration that triggers end of a speech segment. Lower = less "
             "latency but may split sentences at natural pauses.",
    )
    vad_threshold = st.slider(
        "VAD sensitivity", 0.1, 0.9, 0.5, step=0.05,
        disabled=locked,
        help="Lower = picks up quieter speech",
    )

    st.divider()
    translate_on = st.toggle(
        "Translate (Claude)", value=api_key_present, disabled=locked,
        help="Cleanup + KO↔EN translation via the LLM.",
    )
    direction = st.selectbox(
        "Translation direction",
        ["Auto (to the other language)", "Korean → English", "English → Korean"],
        index=0,
        disabled=locked or not translate_on,
    )
    if translate_on and not api_key_present:
        st.warning("ANTHROPIC_API_KEY not set — translation is disabled. "
                   "Transcription still works. Set the key in a .env file.")

    st.divider()
    st.caption("Settings are locked while recording. Stop first to change them.")

lang_map = {"Korean (ko)": "ko", "English (en)": "en"}
primary_code = lang_map[primary]
secondary_code = lang_map[secondary]
if primary_code == secondary_code:
    st.sidebar.warning("Primary and secondary are the same language — the hold key has no effect.")
mode_map = {
    "Auto (to the other language)": "auto",
    "Korean → English": "ko-en",
    "English → Korean": "en-ko",
}
translate_mode = mode_map[direction]

# --- Controls ---
col1, col2 = st.columns(2)
start_btn = col1.button("▶  Start", disabled=st.session_state.running, use_container_width=True, type="primary")
stop_btn = col2.button("■  Stop", disabled=not st.session_state.running, use_container_width=True)

status_box = st.empty()

if start_btn:
    with st.spinner(f"Loading {model_size} model…"):
        vad = load_vad_model()
        whisper = load_whisper_model(model_size)

    from utils.audio import AudioCapture
    from utils.transcriber import Transcriber
    from utils.translator import Translator
    from utils.events import TranscriptBus
    from utils.hotkey import LanguageController

    seg_q: queue.Queue = queue.Queue()
    transcript_q: queue.Queue = queue.Queue()
    bus = TranscriptBus()
    sink: queue.Queue = queue.Queue()
    bus.subscribe(sink.put)  # the UI is one bus subscriber; agents would be another

    controller = LanguageController(primary_code, secondary_code, hold_key=hold_key)
    controller.start()  # best-effort OS hotkey; falls back to auto-detect-only

    capture = AudioCapture(
        seg_q, vad, silence_ms=silence_ms, threshold=vad_threshold,
        get_forced_lang=controller.forced_lang,
    )
    transcriber = Transcriber(seg_q, transcript_q, whisper, lang_pair=controller.pair)
    translator = Translator(transcript_q, bus, mode=translate_mode, enabled=translate_on)

    capture.start()
    transcriber.start()
    translator.start()

    st.session_state.bus = bus
    st.session_state.event_sink = sink
    st.session_state.audio_capture = capture
    st.session_state.transcriber = transcriber
    st.session_state.translator = translator
    st.session_state.language_controller = controller
    st.session_state.running = True
    st.rerun()

if stop_btn:
    if st.session_state.audio_capture:
        st.session_state.audio_capture.stop()
    if st.session_state.transcriber:
        st.session_state.transcriber.stop()
    if st.session_state.translator:
        st.session_state.translator.stop()
    if st.session_state.language_controller:
        st.session_state.language_controller.stop()
    st.session_state.running = False
    st.session_state.audio_capture = None
    st.session_state.transcriber = None
    st.session_state.translator = None
    st.session_state.language_controller = None
    st.session_state.bus = None
    st.session_state.event_sink = None
    st.rerun()

# --- Transcript ---
if st.button("Clear transcript"):
    st.session_state.transcript = []

if not st.session_state.transcript:
    st.info("Transcript will appear here once you start recording.")
else:
    for event in st.session_state.transcript:
        lang_tag = f"`{event.source_lang.upper()}` " if event.source_lang else ""
        forced_tag = "🔒 " if event.lang_source == "forced" else ""
        st.markdown(f"{forced_tag}{lang_tag}{event.display_text}")
        if event.translation:
            # event.translation is raw LLM output derived from arbitrary audio — escape it
            # before embedding in HTML (unsafe_allow_html would otherwise execute injected markup).
            st.markdown(
                f"<span style='color:#888'>↳ `{(event.target_lang or '').upper()}` "
                f"{html.escape(event.translation)}</span>",
                unsafe_allow_html=True,
            )

# --- Status + polling loop ---
if st.session_state.running:
    controller = st.session_state.language_controller
    if controller and controller.is_held():
        status_box.warning(f"🎙  Recording — forcing **{controller.secondary.upper()}** (key held)")
    else:
        status_box.success("🎙  Recording — speak now (auto-detect over the pair)")
    if controller and not controller.available:
        st.caption("⚠️ Hold key unavailable (pynput not active) — running auto-detect-only.")
    time.sleep(0.3)
    st.rerun()
else:
    status_box.info("Press **Start** to begin transcription")
