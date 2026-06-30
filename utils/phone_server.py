"""Phone-as-microphone bridge over Wi-Fi.

A phone (Android or iPhone) scans a QR code, opens a tiny page in its browser, grants mic
access, and streams 32-bit float PCM to this local server over a secure WebSocket. A
``WebMicReader`` (utils/audio.py) drains those samples into the very same VAD/ASR path every
other source uses — so the phone shows up as just another channel on the transcript.

Why a SEPARATE server (not Streamlit's Tornado):
  * ``getUserMedia`` only works in a **secure context** — HTTPS, or ``localhost``. The phone
    reaches us over the LAN by IP, never localhost, so the capture page MUST be served over
    HTTPS. We stand up a small aiohttp app with a self-signed cert on its own port, fully
    decoupled from Streamlit (which keeps serving plain HTTP to the desktop).
  * The phone never touches Streamlit — it loads our page, streams audio over ``wss://``, and
    we hand the samples to the pipeline. Clean seam, no Streamlit-internals surgery.

Security model:
  * The QR/URL carries a random per-launch **token**; the WebSocket handshake rejects any
    connection that doesn't present it, so another device on the same Wi-Fi can't push audio.
  * TLS (self-signed) encrypts the link AND supplies the secure context ``getUserMedia``
    needs. The phone shows a one-time certificate warning the user accepts; the cert is
    persisted (``~/.realtime_stt``) so that acceptance sticks across app restarts.

Lifecycle: a process-wide singleton (``get_server()``), started lazily the first time a Phone
source is used and left running for the app's lifetime — it must outlive Streamlit's ~3/s
reruns, and the phone stays "plugged in" across Start/Stop just like a real mic would.
"""

from __future__ import annotations

import asyncio
import base64
import datetime
import io
import ipaddress
import json
import os
import queue
import secrets
import socket
import ssl
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# Surfaced to the UI (like utils.audio.loopback_error) so a Phone source can explain *why*
# it's unavailable — a missing optional dependency vs. a bind/cert error — instead of a
# silent dead source.
_IMPORT_ERROR: Optional[str] = None
try:
    from aiohttp import web, WSMsgType
    import segno
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
except Exception as e:  # pragma: no cover - exercised only when extras aren't installed
    _IMPORT_ERROR = (
        "phone-mic deps missing (pip install aiohttp segno cryptography)"
        f" — {type(e).__name__}: {e}"
    )


DEFAULT_PORT = 8533
_PORT_SCAN = 12          # try DEFAULT_PORT .. DEFAULT_PORT+_PORT_SCAN if the first is taken
_AUDIO_QUEUE_MAX = 96    # cap buffered blocks: with no reader draining (idle gaps) the oldest
                         # are dropped so we never grow without bound and always stay current
_CERT_DIR = Path.home() / ".realtime_stt"


def phone_import_error() -> Optional[str]:
    """The reason the phone-mic feature can't run (missing deps), or None if it can."""
    return _IMPORT_ERROR


# --------------------------------------------------------------------------------------
# Process-wide singleton
# --------------------------------------------------------------------------------------
_server_singleton: Optional["PhoneAudioServer"] = None
_singleton_lock = threading.Lock()


def current_server() -> Optional["PhoneAudioServer"]:
    """The running server if one has been started, else None — WITHOUT starting it. For status
    polls (connection state / level) that must never spin a server up as a side effect."""
    return _server_singleton


def get_server() -> "PhoneAudioServer":
    """Return the running phone-audio server, starting it on first use.

    Raises RuntimeError with a user-facing message if the optional dependencies aren't
    installed or the server can't bind/serve — callers (the UI) catch and display it.
    """
    if _IMPORT_ERROR:
        raise RuntimeError(_IMPORT_ERROR)
    global _server_singleton
    with _singleton_lock:
        if _server_singleton is None:
            srv = PhoneAudioServer()
            srv.start()
            _server_singleton = srv
        return _server_singleton


def _detect_lan_ip() -> str:
    """Best-effort primary LAN IPv4. The UDP connect picks the OS's default-route source
    address without sending a packet; falls back to loopback if there's no network."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def _ensure_cert(cert_path: Path, key_path: Path, ip: str) -> None:
    """Mint a long-lived self-signed cert once and persist it, so the phone only has to accept
    the certificate warning a single time (ever), not on every app launch. The current LAN IP
    is added as a SubjectAltName when available — a self-signed cert warns regardless, but a
    matching SAN keeps the warning to just 'unknown issuer' rather than also 'wrong host'."""
    if cert_path.exists() and key_path.exists():
        return
    cert_path.parent.mkdir(parents=True, exist_ok=True)

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Realtime STT Phone Mic")])
    sans = [x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
    try:
        addr = ipaddress.ip_address(ip)
        if addr not in (a.value for a in sans if isinstance(a, x509.IPAddress)):
            sans.append(x509.IPAddress(addr))
    except ValueError:
        pass

    # Timezone-aware UTC: datetime.utcnow() is deprecated, and cryptography itself deprecates
    # naive datetimes in not_valid_before/after — an aware UTC value satisfies both.
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=3650))
        .add_extension(x509.SubjectAlternativeName(sans), critical=False)
        .sign(key, hashes.SHA256())
    )
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))


class PhoneAudioServer:
    """A self-contained HTTPS+WSS server that turns a phone browser into a mic source.

    Runs its asyncio loop on a daemon thread. Inbound audio (float32 mono PCM blocks, each
    tagged with the browser's sample rate) lands on a bounded thread-safe queue that the
    synchronous WebMicReader drains. Exposes the pairing URL, a QR data-URI, and live
    connection state for the UI.
    """

    def __init__(self, port: int = DEFAULT_PORT):
        self.port = port
        self.lan_ip = _detect_lan_ip()
        self.token = base64.urlsafe_b64encode(os.urandom(9)).decode().rstrip("=")
        self._audio_q: "queue.Queue[Tuple[np.ndarray, int]]" = queue.Queue(maxsize=_AUDIO_QUEUE_MAX)
        self._clients = 0
        self._last_rx = 0.0          # wall-clock of the most recent audio frame (for staleness)
        # Control channel (phone ⇄ desktop). Inbound taps (translation/mute/ask) queue here for
        # the Streamlit side to drain and apply on its own thread; outbound state/agent frames are
        # pushed to every connected phone socket. _state is the last-broadcast truth, so a phone
        # reconnecting is told the current state at once and set_phone_state can no-op when unchanged.
        self._cmd_q: "queue.Queue[dict]" = queue.Queue()
        self._ctrl_sockets: set = set()
        # assist = the desktop assistant is enabled; gates the phone's "Ask assistant" button so a
        # tap can't open a listening window that would leak the spoken question into the transcript.
        # context = include live conversation context in assistant asks (default on); the phone's
        # Context button flips it.
        self._state: dict = {"translation": False, "mute": False,
                             "recording": False, "assist": False, "context": True}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._start_error: Optional[str] = None

    # ---- public API ----------------------------------------------------------------
    @property
    def url(self) -> str:
        return f"https://{self.lan_ip}:{self.port}/?t={self.token}"

    @property
    def is_connected(self) -> bool:
        """A phone is connected AND has sent audio in the last ~3s (covers a socket that's
        open but idle because the user hasn't tapped Start / the page is backgrounded)."""
        return self._clients > 0 and (time.time() - self._last_rx) < 3.0

    @property
    def client_count(self) -> int:
        return self._clients

    def qr_data_uri(self) -> str:
        """A PNG data-URI of the pairing URL's QR, for inline <img> embedding in the UI."""
        buff = io.BytesIO()
        segno.make(self.url, error="m").save(
            buff, kind="png", scale=6, border=2, dark="#0b0e14", light="#ffffff"
        )
        return "data:image/png;base64," + base64.b64encode(buff.getvalue()).decode()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True, name="phone-audio-server")
        self._thread.start()
        # Block briefly so callers can surface a bind/cert failure synchronously instead of
        # handing back a half-started server.
        if not self._ready.wait(timeout=8.0):
            raise RuntimeError(self._start_error or "phone server failed to start (timeout)")
        if self._start_error:
            raise RuntimeError(self._start_error)

    def flush(self) -> None:
        """Drop any audio buffered before a new reader takes over, so it starts on live
        samples rather than replaying a backlog captured while nothing was consuming."""
        try:
            while True:
                self._audio_q.get_nowait()
        except queue.Empty:
            pass

    def read_block(self, timeout: float = 0.1) -> Optional[Tuple[np.ndarray, int]]:
        """Pull one (samples, sample_rate) block, or None on timeout. Called by WebMicReader."""
        try:
            return self._audio_q.get(timeout=timeout)
        except queue.Empty:
            return None

    # ---- control channel (phone ⇄ desktop) ------------------------------------------
    def pop_commands(self) -> list:
        """Drain inbound phone control taps. The UI calls this each rerun and applies them on
        the Streamlit thread, so the audio/asyncio thread never mutates a worker directly."""
        out: list = []
        try:
            while True:
                out.append(self._cmd_q.get_nowait())
        except queue.Empty:
            pass
        return out

    def set_phone_state(self, **fields) -> None:
        """Update the control state shown on the phone and broadcast it iff it changed. Safe to
        call on every ~3/s rerun. This is what makes the phone a mirror — a desktop-side flip
        reaches it here without the phone having asked (wait-for-echo from either origin)."""
        changed = False
        for k, v in fields.items():
            v = bool(v)
            if self._state.get(k) != v:
                self._state[k] = v
                changed = True
        if changed:
            self._broadcast_threadsafe({"type": "state", **self._state})

    def push_agent(self, status: str, query: str = "", text: str = "") -> None:
        """Echo a phone-initiated assistant exchange to the phone screen (pending → done/error)."""
        self._broadcast_threadsafe(
            {"type": "agent", "status": status, "query": query or "", "text": text or ""}
        )

    def _broadcast_threadsafe(self, obj: dict) -> None:
        """Schedule a broadcast onto the server's event loop from any (e.g. Streamlit) thread."""
        loop = self._loop
        if loop is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(self._broadcast(obj), loop)
        except Exception:
            pass  # loop closing — drop the frame

    async def _broadcast(self, obj: dict) -> None:
        data = json.dumps(obj)
        for ws in list(self._ctrl_sockets):
            try:
                await ws.send_str(data)
            except Exception:
                self._ctrl_sockets.discard(ws)

    # ---- server internals ----------------------------------------------------------
    def _push(self, samples: np.ndarray, rate: int) -> None:
        self._last_rx = time.time()
        try:
            self._audio_q.put_nowait((samples, rate))
        except queue.Full:
            # No consumer is keeping up — drop the oldest block to stay bounded and current.
            try:
                self._audio_q.get_nowait()
                self._audio_q.put_nowait((samples, rate))
            except queue.Empty:
                pass

    def _run(self) -> None:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop

            cert_path = _CERT_DIR / "phone_cert.pem"
            key_path = _CERT_DIR / "phone_key.pem"
            _ensure_cert(cert_path, key_path, self.lan_ip)
            ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_ctx.load_cert_chain(certfile=str(cert_path), keyfile=str(key_path))

            app = web.Application()
            app.add_routes([
                web.get("/", self._handle_page),
                web.get("/worklet.js", self._handle_worklet),
                web.get("/ws", self._handle_ws),
            ])
            runner = web.AppRunner(app)
            loop.run_until_complete(runner.setup())

            bound = False
            for port in range(DEFAULT_PORT, DEFAULT_PORT + _PORT_SCAN):
                try:
                    site = web.TCPSite(runner, host="0.0.0.0", port=port, ssl_context=ssl_ctx)
                    loop.run_until_complete(site.start())
                    self.port = port
                    bound = True
                    break
                except OSError:
                    continue
            if not bound:
                self._start_error = (
                    f"no free port in {DEFAULT_PORT}..{DEFAULT_PORT + _PORT_SCAN}"
                )
                self._ready.set()
                return

            self._ready.set()
            loop.run_forever()
        except Exception as e:  # pragma: no cover - defensive: report, don't crash the app
            self._start_error = f"{type(e).__name__}: {e}"
            self._ready.set()

    async def _handle_page(self, request: "web.Request") -> "web.Response":
        return web.Response(text=_PAGE_HTML, content_type="text/html")

    async def _handle_worklet(self, request: "web.Request") -> "web.Response":
        # Must be a real JS MIME or the browser refuses to load it as an AudioWorklet module.
        return web.Response(text=_WORKLET_JS, content_type="application/javascript")

    async def _handle_ws(self, request: "web.Request") -> "web.WebSocketResponse":
        # Token gate: reject before upgrading so a bad token never reaches the audio path.
        if not secrets.compare_digest(request.query.get("t", ""), self.token):
            return web.Response(status=403, text="forbidden")

        ws = web.WebSocketResponse(max_msg_size=4 * 1024 * 1024)
        await ws.prepare(request)
        self._clients += 1
        self._ctrl_sockets.add(ws)
        # Greet the phone with the current control state so its buttons render the truth at once.
        try:
            await ws.send_str(json.dumps({"type": "state", **self._state}))
        except Exception:
            pass
        rate = 48000  # overwritten by the client's hello; a sane default if it never arrives
        try:
            async for msg in ws:
                if msg.type == WSMsgType.BINARY:
                    samples = np.frombuffer(msg.data, dtype=np.float32)
                    if samples.size:
                        self._push(samples, rate)
                elif msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                    except (ValueError, TypeError):
                        continue
                    mtype = data.get("type")
                    if mtype == "hello":
                        rate = int(data.get("sampleRate", rate)) or rate
                    elif mtype in ("translation", "mute", "ask", "context"):
                        # A control tap — queue it for the Streamlit side to apply to the workers.
                        # "ask" carries on/off: it opens (on) and closes (off) a listening window,
                        # so the whole spoken question is captured as ONE command for the agent.
                        # "context" carries on/off: whether asks include live conversation context.
                        self._cmd_q.put({"cmd": mtype, "on": bool(data.get("on"))})
                elif msg.type == WSMsgType.ERROR:
                    break
        finally:
            self._clients = max(0, self._clients - 1)
            self._ctrl_sockets.discard(ws)
        return ws


# --------------------------------------------------------------------------------------
# Client assets (served over HTTPS to the phone)
# --------------------------------------------------------------------------------------
# NOTE: plain strings (no f-strings) so JS/CSS braces need no escaping. The token is read
# from the page URL in JS, not injected server-side, which keeps these assets static.

_WORKLET_JS = """
// Accumulates mic samples and posts ~2048-frame mono float32 chunks to the main thread,
// which forwards them over the WebSocket. Fewer, larger messages than posting every 128-frame
// render quantum. We don't write to outputs, so connecting this node to the destination (done
// on the page, to keep process() being pulled) produces silence — no mic-to-speaker feedback.
class PCMWorklet extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buf = [];
    this._count = 0;
    this._target = 2048;
  }
  process(inputs) {
    const input = inputs[0];
    if (input && input[0]) {
      const ch = input[0];
      this._buf.push(new Float32Array(ch));
      this._count += ch.length;
      if (this._count >= this._target) {
        const out = new Float32Array(this._count);
        let off = 0;
        for (const b of this._buf) { out.set(b, off); off += b.length; }
        this.port.postMessage(out, [out.buffer]);
        this._buf = [];
        this._count = 0;
      }
    }
    return true;
  }
}
registerProcessor('pcm-worklet', PCMWorklet);
"""

_PAGE_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<title>Realtime STT — Phone Mic</title>
<style>
  :root{ --bg:#0b0e14; --surface:#161d2a; --border:#222b3a; --text:#e8eef6; --dim:#9fadc0;
    --brand:#2dd4bf; --live:#34d399; --danger:#f87171; }
  *{box-sizing:border-box;}
  html,body{margin:0;height:100%;}
  body{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,
    'Segoe UI',Roboto,sans-serif;display:flex;align-items:center;justify-content:center;
    padding:24px;-webkit-tap-highlight-color:transparent;}
  .card{width:100%;max-width:420px;background:var(--surface);border:1px solid var(--border);
    border-radius:20px;padding:26px 22px;text-align:center;box-shadow:0 10px 40px -12px #000;}
  h1{font-size:1.15rem;margin:0 0 4px;}
  p.sub{color:var(--dim);font-size:.85rem;margin:0 0 22px;line-height:1.5;}
  button{width:100%;border:none;border-radius:14px;padding:18px;font-size:1.05rem;font-weight:700;
    color:#04120f;background:linear-gradient(135deg,#5eead4,#2dd4bf);cursor:pointer;
    transition:transform .12s ease,opacity .15s ease;}
  button:active{transform:scale(.98);}
  button:disabled{opacity:.5;}
  button.stop{background:linear-gradient(135deg,#fca5a5,#f87171);color:#1a0606;}
  .status{display:flex;align-items:center;justify-content:center;gap:9px;margin-top:20px;
    font-size:.9rem;font-weight:600;color:var(--dim);}
  .dot{width:10px;height:10px;border-radius:50%;background:#586273;}
  .status.live .dot{background:var(--live);box-shadow:0 0 0 4px rgba(52,211,153,.18);
    animation:b 1.6s ease-in-out infinite;}
  .status.live{color:var(--live);}
  .status.err{color:var(--danger);} .status.err .dot{background:var(--danger);}
  @keyframes b{0%,100%{transform:scale(1);opacity:1}50%{transform:scale(1.4);opacity:.6}}
  .meter{height:6px;border-radius:4px;background:#1b2433;overflow:hidden;margin-top:18px;}
  .meter > i{display:block;height:100%;width:0;background:var(--live);transition:width .1s linear;}
  .hint{margin-top:18px;font-size:.72rem;color:#828fa3;line-height:1.5;}
  /* ---- control panel (translation / mute / ask) — shown once the link is up ---- */
  .controls{margin-top:20px;display:flex;flex-direction:column;gap:12px;}
  .controls[hidden]{display:none;}
  .ctl-hint{font-size:.74rem;color:var(--dim);line-height:1.5;}
  .ctl-hint[hidden]{display:none;}
  .ctl-row{display:flex;gap:12px;}
  .ctl{flex:1;display:flex;flex-direction:column;align-items:center;gap:7px;background:var(--surface);
    border:1px solid var(--border);border-radius:14px;padding:14px 10px;color:var(--text);
    font-size:.86rem;font-weight:700;}
  .ctl .pill{font-size:.68rem;font-weight:700;padding:3px 11px;border-radius:999px;
    background:#0f1520;color:var(--dim);border:1px solid var(--border);}
  .ctl.on{border-color:var(--brand);color:var(--brand);}
  .ctl.on .pill{background:var(--brand);color:#04120f;border-color:var(--brand);}
  .ctl.muted{border-color:var(--danger);color:var(--danger);}
  .ctl.muted .pill{background:var(--danger);color:#1a0606;border-color:var(--danger);}
  .ctl.pending{opacity:.6;}
  .ctl:disabled{opacity:.4;}
  .ask{background:linear-gradient(135deg,#c4b5fd,#a78bfa);color:#140a2e;}
  /* listening window open (tap 1 → tap 2): red, with the same pulse as the live status dot, so
     it's unmistakably "recording your question now — tap again to send". */
  .ask.listening{background:linear-gradient(135deg,#fca5a5,#f87171);color:#1a0606;
    animation:b 1.6s ease-in-out infinite;}
  .answer{background:#0f1520;border:1px solid var(--border);border-radius:14px;padding:13px 14px;
    text-align:left;font-size:.9rem;line-height:1.55;color:var(--text);max-height:42vh;overflow:auto;
    white-space:pre-wrap;}
  .answer[hidden]{display:none;}
  .answer .q{color:var(--dim);font-size:.78rem;font-weight:600;margin-bottom:8px;}
  .answer .working{color:#a78bfa;font-weight:600;}
</style>
</head>
<body>
  <div class="card">
    <h1>Phone microphone</h1>
    <p class="sub">Stream this phone's mic to your computer's transcriber over Wi-Fi.
      Keep this page open and the screen awake while talking.</p>
    <button id="btn">Start microphone</button>
    <div class="meter"><i id="lvl"></i></div>
    <div class="status" id="status"><span class="dot"></span><span id="statusText">Idle</span></div>
    <div class="controls" id="controls" hidden>
      <div class="ctl-hint" id="ctlHint">Press <b>Start</b> on the computer to enable controls.</div>
      <div class="ctl-row">
        <button class="ctl" id="translateBtn" type="button" disabled>
          <span class="ctl-label">Translate</span><span class="pill" id="translatePill">Off</span>
        </button>
        <button class="ctl" id="muteBtn" type="button" disabled>
          <span class="ctl-label">Mic</span><span class="pill" id="mutePill">Live</span>
        </button>
        <button class="ctl" id="contextBtn" type="button" disabled>
          <span class="ctl-label">Context</span><span class="pill" id="contextPill">On</span>
        </button>
      </div>
      <button class="ask" id="askBtn" type="button" disabled>Ask assistant</button>
      <div class="answer" id="answer" hidden></div>
    </div>
    <div class="hint" id="hint">Tip: turn the screen-lock timeout up — phones suspend audio when the screen sleeps.</div>
  </div>
<script>
const token = new URLSearchParams(location.search).get('t') || '';
const btn = document.getElementById('btn');
const statusEl = document.getElementById('status');
const statusText = document.getElementById('statusText');
const lvl = document.getElementById('lvl');
const controls = document.getElementById('controls');
const ctlHint = document.getElementById('ctlHint');
const translateBtn = document.getElementById('translateBtn');
const translatePill = document.getElementById('translatePill');
const muteBtn = document.getElementById('muteBtn');
const mutePill = document.getElementById('mutePill');
const contextBtn = document.getElementById('contextBtn');
const contextPill = document.getElementById('contextPill');
const askBtn = document.getElementById('askBtn');
const answerEl = document.getElementById('answer');
let ctx, node, stream, ws, running = false;
let confirmedTranslation = false, confirmedMute = false, recording = false, assistOn = false;
// Whether assistant asks fold in live conversation context. Defaults on (matches the server) so the
// button reads true until the first state frame; the desktop is authoritative thereafter.
let confirmedContext = true;
// Ask is a two-tap window: askListening = window open (tap 1, speaking), askPending = window
// closed and awaiting the answer (tap 2). askTimer resets the button if no answer ever lands
// (e.g. the window was closed without anything being said).
let askListening = false, askPending = false, askTimer = null;

function setStatus(cls, text){ statusEl.className = 'status ' + cls; statusText.textContent = text; }

async function start(){
  try{
    setStatus('', 'Requesting mic…');
    stream = await navigator.mediaDevices.getUserMedia({
      audio:{ channelCount:1, echoCancellation:true, noiseSuppression:true, autoGainControl:true },
      video:false
    });
    ctx = new (window.AudioContext || window.webkitAudioContext)();
    await ctx.resume();
    await ctx.audioWorklet.addModule('/worklet.js');
    const src = ctx.createMediaStreamSource(stream);
    node = new AudioWorkletNode(ctx, 'pcm-worklet');

    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    ws = new WebSocket(proto + '://' + location.host + '/ws?t=' + encodeURIComponent(token));
    ws.binaryType = 'arraybuffer';
    ws.onopen = () => {
      ws.send(JSON.stringify({ type:'hello', sampleRate: ctx.sampleRate }));
      setStatus('live', 'Streaming');
      controls.hidden = false;          // reveal the control panel once the link is up
    };
    ws.onmessage = (e) => { try { handleControl(JSON.parse(e.data)); } catch(_){} };
    ws.onclose = () => { if (running) stop(true); };
    ws.onerror = () => setStatus('err', 'Connection error');

    node.port.onmessage = (e) => {
      const buf = e.data;
      // Level meter (RMS of this chunk) + backpressure guard: skip sends if the socket is backed up.
      const f = new Float32Array(buf);
      let s = 0; for (let i=0;i<f.length;i++) s += f[i]*f[i];
      const rms = Math.sqrt(s / Math.max(1,f.length));
      lvl.style.width = Math.min(100, rms * 320) + '%';
      if (ws && ws.readyState === 1 && ws.bufferedAmount < 1<<20) ws.send(buf);
    };
    src.connect(node);
    node.connect(ctx.destination);  // pull process() (silent — we don't write outputs)

    running = true;
    btn.textContent = 'Stop microphone';
    btn.classList.add('stop');
  }catch(err){
    setStatus('err', (err && err.name === 'NotAllowedError') ? 'Mic permission denied' : 'Could not start mic');
  }
}

function stop(fromClose){
  running = false;
  // Close any open ask window before the socket goes away, so the desktop doesn't keep marking
  // (now-absent) phone audio as a command.
  if (askListening && ws && ws.readyState === 1 && !fromClose){ try{ sendCtl({ type:'ask', on:false }); }catch(e){} }
  askListening = false; askPending = false; clearTimeout(askTimer);
  try{ if (node) node.disconnect(); }catch(e){}
  try{ if (stream) stream.getTracks().forEach(t => t.stop()); }catch(e){}
  try{ if (ctx) ctx.close(); }catch(e){}
  try{ if (ws && !fromClose) ws.close(); }catch(e){}
  lvl.style.width = '0%';
  controls.hidden = true;
  answerEl.hidden = true;
  btn.textContent = 'Start microphone';
  btn.classList.remove('stop');
  setStatus('', fromClose ? 'Disconnected' : 'Idle');
}

// ---- control panel: wait-for-echo buttons + assistant answer ----
function escapeHtml(s){ return (s||'').replace(/[&<>"']/g, c => (
  {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }

function sendCtl(obj){ if (ws && ws.readyState === 1) ws.send(JSON.stringify(obj)); }

function renderControls(){
  translateBtn.classList.toggle('on', confirmedTranslation);
  translateBtn.classList.remove('pending');
  translatePill.textContent = confirmedTranslation ? 'On' : 'Off';
  muteBtn.classList.toggle('muted', confirmedMute);
  muteBtn.classList.remove('pending');
  mutePill.textContent = confirmedMute ? 'Muted' : 'Live';
  contextBtn.classList.toggle('on', confirmedContext);
  contextBtn.classList.remove('pending');
  contextPill.textContent = confirmedContext ? 'On' : 'Off';
  translateBtn.disabled = !recording;
  muteBtn.disabled = !recording;
  contextBtn.disabled = !recording;
  // Ask is a two-tap window. Idle → "Ask assistant"; tap 1 opens the listening window (red,
  // "tap to send"); tap 2 closes it and waits for the answer ("Thinking…", disabled). It needs
  // both recording AND the desktop assistant enabled — otherwise a tap would just leak the
  // spoken question into the transcript with no answer.
  askBtn.classList.toggle('listening', askListening);
  if (askListening){
    askBtn.disabled = false;
    askBtn.textContent = 'Listening… tap to send';
  } else if (askPending){
    askBtn.disabled = true;
    askBtn.textContent = 'Thinking…';
  } else {
    askBtn.disabled = !recording || !assistOn;
    askBtn.textContent = recording && !assistOn ? 'Assistant is off' : 'Ask assistant';
  }
  ctlHint.hidden = recording;
}

function renderAnswer(m){
  answerEl.hidden = false;
  const q = m.query ? '<div class="q">' + escapeHtml(m.query) + '</div>' : '';
  if (m.status === 'pending'){
    // The agent now has the transcribed question — show it with a spinner. This only arrives
    // after tap 2 (the window is already closed), so the button stays in its "Thinking…" state.
    answerEl.innerHTML = q + '<div class="working">Thinking…</div>';
  } else {
    answerEl.innerHTML = q + '<div>' + escapeHtml(m.text || '(no answer)') + '</div>';
    askPending = false;
    clearTimeout(askTimer);
    renderControls();
  }
}

function handleControl(m){
  if (!m || !m.type) return;
  if (m.type === 'state'){
    confirmedTranslation = !!m.translation;
    confirmedMute = !!m.mute;
    recording = !!m.recording;
    assistOn = !!m.assist;
    confirmedContext = !!m.context;
    // Recording stopped (or the assistant was switched off) mid-window → abandon the ask so the
    // button can't be stuck "Listening…/Thinking…" against a capture that's gone.
    if ((!recording || !assistOn) && (askListening || askPending)){
      askListening = false; askPending = false; clearTimeout(askTimer);
    }
    renderControls();
  } else if (m.type === 'agent'){
    renderAnswer(m);
  }
}

// A tap only *requests* a change; the on/off state flips when the server's state frame
// confirms it (wait-for-echo). The pending class is a transient affordance, cleared on the
// echo — or after a short timeout if none arrives (e.g. the desktop isn't recording).
translateBtn.addEventListener('click', () => {
  if (translateBtn.disabled) return;
  translateBtn.classList.add('pending');
  setTimeout(() => translateBtn.classList.remove('pending'), 3000);
  sendCtl({ type:'translation', on: !confirmedTranslation });
});
muteBtn.addEventListener('click', () => {
  if (muteBtn.disabled) return;
  muteBtn.classList.add('pending');
  setTimeout(() => muteBtn.classList.remove('pending'), 3000);
  sendCtl({ type:'mute', on: !confirmedMute });
});
// Context on/off: whether assistant asks include live conversation context. Same wait-for-echo
// dance — the flip confirms when the desktop's state frame echoes back.
contextBtn.addEventListener('click', () => {
  if (contextBtn.disabled) return;
  contextBtn.classList.add('pending');
  setTimeout(() => contextBtn.classList.remove('pending'), 3000);
  sendCtl({ type:'context', on: !confirmedContext });
});
// Two-tap ask. Tap 1 opens a listening window: your speech is marked as one command and routed
// to the agent (not the transcript). Tap 2 closes it and sends the whole window as a single
// prompt — pauses and all. The answer echoes back over the control channel (renderAnswer). The
// safety timer clears the spinner if the window closed with nothing said (no answer will come).
askBtn.addEventListener('click', () => {
  if (askBtn.disabled) return;
  if (!askListening){
    askListening = true;
    sendCtl({ type:'ask', on:true });
    renderControls();
  } else {
    askListening = false;
    askPending = true;
    sendCtl({ type:'ask', on:false });
    answerEl.hidden = false;
    answerEl.innerHTML = '<div class="working">Thinking…</div>';
    renderControls();
    clearTimeout(askTimer);
    askTimer = setTimeout(() => {
      if (askPending){ askPending = false; answerEl.hidden = true; renderControls(); }
    }, 20000);
  }
});

btn.addEventListener('click', () => running ? stop(false) : start());
</script>
</body>
</html>
"""
