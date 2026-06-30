"""
Per-application audio capture via the Windows **Process Loopback API** — the same
technique OBS uses for "Application Audio Capture". A non-destructive tap on one process
(and its child processes): the app keeps playing to your speakers (no lag, you still hear
it), and we get a private copy of just that app's audio. No virtual cable, no third-party
software — only ~3 Windows API calls, reimplemented here in pure ctypes/COM.

Documented minimum is Windows build 20348; in practice it also works on Win10 22H2
(19045) — verified on this machine. Everything here is Windows-only and import-guarded by
the caller; on failure we simply contribute no process sources.

Two public entry points:
  * list_audio_sessions() -> [{"pid", "exe", "name"}]  apps with an audio session, one per
        app (sessions are rolled up to the root process so "Brave" appears once).
  * ProcessLoopbackCapture(pid)  context manager whose .read() yields (frames, channels)
        float32 — mirrors soundcard's recorder so the reader in utils/audio.py is trivial.
"""

import os
import ctypes
from ctypes import (
    wintypes, POINTER, Structure, Union, WINFUNCTYPE, cast, byref,
    c_void_p, c_ulong, c_uint32, c_uint64, c_longlong, c_int, c_byte, c_wchar,
)

import numpy as np

# ---- constants -----------------------------------------------------------------------
_VAD_PROCESS_LOOPBACK = "VAD\\Process_Loopback"
_INCLUDE_TREE = 0
_ACT_TYPE_PROCESS_LOOPBACK = 1
_VT_BLOB = 0x0041
_SHARED = 0
_STREAMFLAGS_LOOPBACK = 0x00020000
_BUFFERFLAGS_SILENT = 0x2
_WAVE_FORMAT_PCM = 1
_S_OK = 0
_E_NOINTERFACE = 0x80004002
_COINIT_MULTITHREADED = 0x0
_CLSCTX_INPROC_SERVER = 0x1
_TH32CS_SNAPPROCESS = 0x2
_STATE_EXPIRED = 2

CAPTURE_RATE = 48000   # request 48k stereo PCM16 (the format the spike proved); the caller
CAPTURE_CHANNELS = 2   # downmixes + resamples to the 16k ASR rate, as LoopbackReader does.
_BITS = 16


def _hr_ok(hr):
    return (hr & 0xFFFFFFFF) == 0


# ---- GUID + COM call helper ----------------------------------------------------------
class GUID(Structure):
    _fields_ = [("Data1", c_ulong), ("Data2", wintypes.WORD),
                ("Data3", wintypes.WORD), ("Data4", c_byte * 8)]


def _guid(d1, d2, d3, tail):
    g = GUID()
    g.Data1, g.Data2, g.Data3 = d1, d2, d3
    for i, b in enumerate(tail):
        g.Data4[i] = b - 256 if b > 127 else b
    return g


IID_IUnknown = _guid(0x00000000, 0x0000, 0x0000, [0xC0, 0, 0, 0, 0, 0, 0, 0x46])
IID_ICompletionHandler = _guid(0x41D949AB, 0x9862, 0x444A, [0x80, 0xF6, 0xC2, 0x61, 0x33, 0x4D, 0xA5, 0xEB])
IID_IAgileObject = _guid(0x94EA2B94, 0xE9CC, 0x49E0, [0xC0, 0xFF, 0xEE, 0x64, 0xCA, 0x8F, 0x5B, 0x90])
IID_IAudioClient = _guid(0x1CB9AD4C, 0xDBFA, 0x4C32, [0xB1, 0x78, 0xC2, 0xF5, 0x68, 0xA7, 0x03, 0xB2])
IID_IAudioCaptureClient = _guid(0xC8ADBD64, 0xE71E, 0x48A0, [0xA4, 0xDE, 0x18, 0x5C, 0x39, 0x5C, 0xD3, 0x17])
CLSID_MMDeviceEnumerator = _guid(0xBCDE0395, 0xE52F, 0x467C, [0x8E, 0x3D, 0xC4, 0x57, 0x92, 0x91, 0x69, 0x2E])
IID_IMMDeviceEnumerator = _guid(0xA95664D2, 0x9614, 0x4F35, [0xA7, 0x46, 0xDE, 0x8D, 0xB6, 0x36, 0x17, 0xE6])
IID_IAudioSessionManager2 = _guid(0x77AA99A0, 0x1BD6, 0x484F, [0x8B, 0xC7, 0x2C, 0x65, 0x4C, 0x9A, 0x9B, 0x6F])
IID_IAudioSessionEnumerator = _guid(0xE2F5BB11, 0x0570, 0x40CA, [0xAC, 0xDD, 0x3A, 0xA0, 0x12, 0x77, 0xDE, 0xE8])
IID_IAudioSessionControl2 = _guid(0xBFB7FF88, 0x7239, 0x4FC9, [0x8F, 0xA2, 0x07, 0xC9, 0x50, 0xBE, 0x9C, 0x6D])


def _vcall(interface, index, restype, argtypes, *args):
    """Call COM method #index on an interface pointer (c_void_p)."""
    vtbl = cast(interface, POINTER(c_void_p))
    table = cast(vtbl[0], POINTER(c_void_p))
    proto = WINFUNCTYPE(restype, c_void_p, *argtypes)
    return proto(table[index])(interface, *args)


def _release(interface):
    if interface and interface.value:
        _vcall(interface, 2, ctypes.c_long, [])


def _ensure_com():
    try:
        ctypes.windll.ole32.CoInitializeEx(None, _COINIT_MULTITHREADED)
    except OSError:
        pass


# ---- structs for activation ----------------------------------------------------------
class WAVEFORMATEX(Structure):
    _fields_ = [("wFormatTag", wintypes.WORD), ("nChannels", wintypes.WORD),
                ("nSamplesPerSec", wintypes.DWORD), ("nAvgBytesPerSec", wintypes.DWORD),
                ("nBlockAlign", wintypes.WORD), ("wBitsPerSample", wintypes.WORD),
                ("cbSize", wintypes.WORD)]


class _PROC_LOOPBACK_PARAMS(Structure):
    _fields_ = [("TargetProcessId", wintypes.DWORD), ("ProcessLoopbackMode", c_int)]


class _ACT_UNION(Union):
    _fields_ = [("ProcessLoopbackParams", _PROC_LOOPBACK_PARAMS)]


class _ACTIVATION_PARAMS(Structure):
    _fields_ = [("ActivationType", c_int), ("u", _ACT_UNION)]


class _BLOB(Structure):
    _fields_ = [("cbSize", c_ulong), ("pBlobData", c_void_p)]


class PROPVARIANT(Structure):
    _fields_ = [("vt", wintypes.WORD), ("r1", wintypes.WORD), ("r2", wintypes.WORD),
                ("r3", wintypes.WORD), ("blob", _BLOB)]


# ---- process table (for rolling sessions up to their root process) -------------------
class _PROCESSENTRY32W(Structure):
    _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
                ("th32ProcessID", wintypes.DWORD), ("th32DefaultHeapID", POINTER(c_ulong)),
                ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
                ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", ctypes.c_long),
                ("dwFlags", wintypes.DWORD), ("szExeFile", c_wchar * 260)]


def _process_map():
    """{pid: (parent_pid, exe_basename_lower)} for every running process."""
    k32 = ctypes.windll.kernel32
    snap = k32.CreateToolhelp32Snapshot(_TH32CS_SNAPPROCESS, 0)
    if snap == -1 or snap == 0xFFFFFFFFFFFFFFFF:
        return {}
    out = {}
    try:
        entry = _PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(_PROCESSENTRY32W)
        ok = k32.Process32FirstW(snap, byref(entry))
        while ok:
            out[entry.th32ProcessID] = (entry.th32ParentProcessID, entry.szExeFile.lower())
            ok = k32.Process32NextW(snap, byref(entry))
    finally:
        k32.CloseHandle(snap)
    return out


def _root_pid(pid, pmap):
    """Climb parents while the image name stays the same → the app's root process.

    A Chromium app registers its audio session under a child utility process; targeting the
    root (browser) PID with INCLUDE_TARGET_PROCESS_TREE captures the whole app. Single-process
    apps just return themselves.
    """
    info = pmap.get(pid)
    if not info:
        return pid
    exe = info[1]
    seen = {pid}
    cur = pid
    while True:
        ppid, _ = pmap.get(cur, (0, ""))
        parent = pmap.get(ppid)
        if not parent or parent[1] != exe or ppid in seen:
            return cur
        seen.add(ppid)
        cur = ppid


def _friendly(exe):
    """'brave.exe' → 'Brave'."""
    stem = exe[:-4] if exe.lower().endswith(".exe") else exe
    return stem[:1].upper() + stem[1:] if stem else exe


# ---- session enumeration -------------------------------------------------------------
def list_audio_sessions():
    """Apps with an audio session on the default render endpoint, one entry per app.

    Returns [{"pid": int (root process), "exe": str, "name": str}]. Best-effort: any COM
    failure yields an empty list rather than raising. Excludes the system-sounds session,
    PID 0, and our own process.
    """
    _ensure_com()
    ole32 = ctypes.windll.ole32
    enum = c_void_p()
    hr = ole32.CoCreateInstance(byref(CLSID_MMDeviceEnumerator), None, _CLSCTX_INPROC_SERVER,
                                byref(IID_IMMDeviceEnumerator), byref(enum))
    if not _hr_ok(hr) or not enum.value:
        return []

    device = c_void_p()
    mgr = c_void_p()
    sess_enum = c_void_p()
    try:
        # GetDefaultAudioEndpoint(eRender=0, eConsole=0)
        if not _hr_ok(_vcall(enum, 4, ctypes.c_long, [c_int, c_int, POINTER(c_void_p)],
                             0, 0, byref(device))) or not device.value:
            return []
        # IMMDevice::Activate(IID_IAudioSessionManager2)
        if not _hr_ok(_vcall(device, 3, ctypes.c_long,
                             [POINTER(GUID), wintypes.DWORD, c_void_p, POINTER(c_void_p)],
                             byref(IID_IAudioSessionManager2), _CLSCTX_INPROC_SERVER, None,
                             byref(mgr))) or not mgr.value:
            return []
        # GetSessionEnumerator
        if not _hr_ok(_vcall(mgr, 5, ctypes.c_long, [POINTER(c_void_p)], byref(sess_enum))) \
                or not sess_enum.value:
            return []

        count = c_int()
        if not _hr_ok(_vcall(sess_enum, 3, ctypes.c_long, [POINTER(c_int)], byref(count))):
            return []

        pmap = _process_map()
        self_pid = os.getpid()
        by_root = {}
        for i in range(count.value):
            ctrl = c_void_p()
            if not _hr_ok(_vcall(sess_enum, 4, ctypes.c_long, [c_int, POINTER(c_void_p)],
                                 i, byref(ctrl))) or not ctrl.value:
                continue
            ctrl2 = c_void_p()
            try:
                if not _hr_ok(_vcall(ctrl, 0, ctypes.c_long, [POINTER(GUID), POINTER(c_void_p)],
                                     byref(IID_IAudioSessionControl2), byref(ctrl2))) \
                        or not ctrl2.value:
                    continue
                state = c_int()
                _vcall(ctrl2, 3, ctypes.c_long, [POINTER(c_int)], byref(state))
                if state.value == _STATE_EXPIRED:
                    continue
                # IsSystemSoundsSession (index 15) → S_OK means it's the system chime session
                if _hr_ok(_vcall(ctrl2, 15, ctypes.c_long, [])):
                    continue
                pid = wintypes.DWORD()
                if not _hr_ok(_vcall(ctrl2, 14, ctypes.c_long, [POINTER(wintypes.DWORD)], byref(pid))):
                    continue
                raw_pid = pid.value
                if raw_pid in (0, self_pid):
                    continue
                root = _root_pid(raw_pid, pmap)
                exe = pmap.get(root, (0, ""))[1] or pmap.get(raw_pid, (0, "audio"))[1]
                by_root.setdefault(root, exe)
            finally:
                _release(ctrl2)
                _release(ctrl)
        return [{"pid": pid, "exe": exe, "name": _friendly(exe)} for pid, exe in by_root.items()]
    finally:
        _release(sess_enum)
        _release(mgr)
        _release(device)
        _release(enum)


# ---- capture -------------------------------------------------------------------------
class ProcessLoopbackCapture:
    """Context manager: opens process loopback for `pid`, .read() drains available audio.

    Usage mirrors soundcard's recorder so utils/audio.py's ProcessReader stays tiny:

        with ProcessLoopbackCapture(pid) as cap:
            data = cap.read()           # (frames, channels) float32, may be empty
    """

    def __init__(self, pid, samplerate=CAPTURE_RATE, channels=CAPTURE_CHANNELS):
        self.pid = int(pid)
        self.samplerate = samplerate
        self.channels = channels
        self._client = None
        self._capture = None
        self._block_align = channels * _BITS // 8
        self._done_evt = None
        self._cb_refs = []  # keep WINFUNCTYPE callbacks alive

    # -- COM completion handler (agile, event-signalling) ------------------------------
    def _make_handler(self):
        k32 = ctypes.windll.kernel32
        self._done_evt = k32.CreateEventW(None, True, False, None)
        accept = {bytes(memoryview(g)) for g in (IID_IUnknown, IID_ICompletionHandler, IID_IAgileObject)}

        QI_T = WINFUNCTYPE(ctypes.c_long, c_void_p, c_void_p, POINTER(c_void_p))
        REF_T = WINFUNCTYPE(c_ulong, c_void_p)
        ACT_T = WINFUNCTYPE(ctypes.c_long, c_void_p, c_void_p)

        def qi(this, riid, ppv):
            if ctypes.string_at(riid, ctypes.sizeof(GUID)) in accept:
                cast(ppv, POINTER(c_void_p))[0] = this
                return _S_OK
            cast(ppv, POINTER(c_void_p))[0] = None
            return _E_NOINTERFACE

        def addref(this):
            return 1

        def release(this):
            return 1

        def activated(this, op):
            k32.SetEvent(self._done_evt)
            return _S_OK

        cbs = [QI_T(qi), REF_T(addref), REF_T(release), ACT_T(activated)]
        self._cb_refs = cbs

        class VTBL(Structure):
            _fields_ = [("QueryInterface", QI_T), ("AddRef", REF_T),
                        ("Release", REF_T), ("ActivateCompleted", ACT_T)]

        class HANDLER(Structure):
            _fields_ = [("lpVtbl", POINTER(VTBL))]

        vtbl = VTBL(*cbs)
        handler = HANDLER(ctypes.pointer(vtbl))
        self._cb_refs += [vtbl, handler]  # pin against GC
        return handler

    def _open(self):
        _ensure_com()
        k32 = ctypes.windll.kernel32
        mmdev = ctypes.WinDLL("Mmdevapi.dll")

        act = _ACTIVATION_PARAMS()
        act.ActivationType = _ACT_TYPE_PROCESS_LOOPBACK
        act.u.ProcessLoopbackParams.TargetProcessId = self.pid
        act.u.ProcessLoopbackParams.ProcessLoopbackMode = _INCLUDE_TREE
        self._cb_refs.append(act)

        pv = PROPVARIANT()
        pv.vt = _VT_BLOB
        pv.blob.cbSize = ctypes.sizeof(act)
        pv.blob.pBlobData = cast(byref(act), c_void_p)

        handler = self._make_handler()

        fn = mmdev.ActivateAudioInterfaceAsync
        fn.restype = ctypes.c_long
        fn.argtypes = [wintypes.LPCWSTR, POINTER(GUID), POINTER(PROPVARIANT), c_void_p, POINTER(c_void_p)]
        async_op = c_void_p()
        hr = fn(_VAD_PROCESS_LOOPBACK, byref(IID_IAudioClient), byref(pv),
                cast(byref(handler), c_void_p), byref(async_op))
        if not _hr_ok(hr):
            raise OSError(f"ActivateAudioInterfaceAsync failed: 0x{hr & 0xFFFFFFFF:08X}")

        if k32.WaitForSingleObject(self._done_evt, 5000) != 0:
            _release(async_op)
            raise OSError("process-loopback activation timed out")

        activate_hr = ctypes.c_long()
        client = c_void_p()
        _vcall(async_op, 3, ctypes.c_long, [POINTER(ctypes.c_long), POINTER(c_void_p)],
               byref(activate_hr), byref(client))
        _release(async_op)
        if not _hr_ok(activate_hr.value) or not client.value:
            raise OSError(f"GetActivateResult failed: 0x{activate_hr.value & 0xFFFFFFFF:08X}")
        self._client = client

        fmt = WAVEFORMATEX(_WAVE_FORMAT_PCM, self.channels, self.samplerate,
                           self.samplerate * self._block_align, self._block_align, _BITS, 0)
        hr = _vcall(client, 3, ctypes.c_long,  # IAudioClient::Initialize
                    [c_int, wintypes.DWORD, c_longlong, c_longlong, POINTER(WAVEFORMATEX), c_void_p],
                    _SHARED, _STREAMFLAGS_LOOPBACK, 2_000_000, 0, byref(fmt), None)
        if not _hr_ok(hr):
            raise OSError(f"IAudioClient::Initialize failed: 0x{hr & 0xFFFFFFFF:08X}")

        capture = c_void_p()
        hr = _vcall(client, 14, ctypes.c_long, [POINTER(GUID), POINTER(c_void_p)],
                    byref(IID_IAudioCaptureClient), byref(capture))
        if not _hr_ok(hr) or not capture.value:
            raise OSError(f"GetService(IAudioCaptureClient) failed: 0x{hr & 0xFFFFFFFF:08X}")
        self._capture = capture

        hr = _vcall(client, 10, ctypes.c_long, [])  # Start
        if not _hr_ok(hr):
            raise OSError(f"IAudioClient::Start failed: 0x{hr & 0xFFFFFFFF:08X}")

    def read(self):
        """Drain all currently-available packets → (frames, channels) float32. May be empty."""
        cap = self._capture
        chunks = []
        packet = c_uint32()
        _vcall(cap, 5, ctypes.c_long, [POINTER(c_uint32)], byref(packet))  # GetNextPacketSize
        while packet.value > 0:
            pdata = c_void_p()
            nframes = c_uint32()
            flags = wintypes.DWORD()
            devpos = c_uint64()
            qpc = c_uint64()
            hr = _vcall(cap, 3, ctypes.c_long,  # GetBuffer
                        [POINTER(c_void_p), POINTER(c_uint32), POINTER(wintypes.DWORD),
                         POINTER(c_uint64), POINTER(c_uint64)],
                        byref(pdata), byref(nframes), byref(flags), byref(devpos), byref(qpc))
            if not _hr_ok(hr):
                break
            n = nframes.value
            if n:
                if (flags.value & _BUFFERFLAGS_SILENT) or not pdata.value:
                    chunks.append(np.zeros((n, self.channels), dtype=np.float32))
                else:
                    raw = ctypes.string_at(pdata, n * self._block_align)
                    arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                    chunks.append(arr.reshape(-1, self.channels))
            _vcall(cap, 4, ctypes.c_long, [c_uint32], n)  # ReleaseBuffer
            _vcall(cap, 5, ctypes.c_long, [POINTER(c_uint32)], byref(packet))
        if not chunks:
            return np.empty((0, self.channels), dtype=np.float32)
        return np.concatenate(chunks, axis=0)

    def close(self):
        if self._client:
            _vcall(self._client, 11, ctypes.c_long, [])  # Stop
        _release(self._capture)
        _release(self._client)
        self._capture = None
        self._client = None
        if self._done_evt:
            ctypes.windll.kernel32.CloseHandle(self._done_evt)
            self._done_evt = None
        self._cb_refs = []

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, *exc):
        self.close()
        return False
