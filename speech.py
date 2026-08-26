"""
Offline TTS with per-object cooldown, same-info suppression, and global
rate limiting.

Windows SAPI must be created and used on the *same* thread (STA + COM).
The previous design initialized the engine on the UI thread and called
runAndWait from a new thread per phrase, which fails silently — no audio.

This module owns a single worker thread that:
  1. Initializes COM
  2. Creates the voice on that thread
  3. Speaks queued alerts through the default audio output

Anti-spam strategy:
  • Per-distance cooldowns (near=3 s, mid=6 s, far=12 s)
  • Same-info multiplier: if label+direction+distance haven't changed,
    double the cooldown so the user isn't told the same thing repeatedly
  • Global rate limiter: at most one announcement every GLOBAL_SPEAK_GAP
  • Escalation override: if an object moves to a closer bucket, the
    cooldown is bypassed immediately
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from typing import List, Optional, Protocol

from tracker import Track
from config import (
    COOLDOWN_SEC,
    COOLDOWN_BY_DISTANCE,
    GLOBAL_SPEAK_GAP,
    SAME_INFO_MULTIPLIER,
    TTS_RATE,
    TTS_VOLUME,
    TTS_QUEUE_SIZE,
)

_BUCKET_ORDER = {"far": 0, "mid": 1, "near": 2}

_DIR_WORDS = {
    "left": "on the left",
    "right": "on the right",
    "center": "ahead",
}


class _Voice(Protocol):
    def speak(self, text: str) -> None: ...


def _coinitialize() -> None:
    try:
        import pythoncom

        pythoncom.CoInitialize()
        return
    except Exception:
        pass
    try:
        import comtypes

        comtypes.CoInitialize()
    except Exception:
        pass


class _SapiVoice:
    """Direct Windows SAPI voice — goes to the default playback device."""

    def __init__(self):
        import comtypes.client

        self._voice = comtypes.client.CreateObject("SAPI.SpVoice")
        self._voice.Volume = int(max(0.0, min(1.0, TTS_VOLUME)) * 100)
        # SAPI rate is -10..10; map ~175 wpm to a slightly brisk setting
        self._voice.Rate = 1

    def speak(self, text: str) -> None:
        self._voice.Speak(str(text), 0)


class _PyttsxVoice:
    def __init__(self):
        import pyttsx3

        driver = "sapi5" if sys.platform == "win32" else None
        self._engine = pyttsx3.init(driver) if driver else pyttsx3.init()
        self._engine.setProperty("rate", TTS_RATE)
        self._engine.setProperty("volume", float(TTS_VOLUME))

    def speak(self, text: str) -> None:
        self._engine.say(text)
        self._engine.runAndWait()


class _PowerShellVoice:
    """Last-resort Windows TTS via .NET SpeechSynthesizer (default device)."""

    def speak(self, text: str) -> None:
        import subprocess

        safe = str(text).replace("'", "''")
        flags = 0
        if sys.platform == "win32":
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-STA",
                "-Command",
                "Add-Type -AssemblyName System.Speech; "
                "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                "$s.Volume = 100; "
                f"$s.Speak('{safe}')",
            ],
            check=False,
            timeout=45,
            creationflags=flags,
        )


def _make_voice() -> _Voice:
    if sys.platform == "win32":
        try:
            v = _SapiVoice()
            print("[speech] Using Windows SAPI (default audio device)")
            return v
        except Exception as exc:
            print(f"[speech] SAPI init failed: {exc}")
    try:
        v = _PyttsxVoice()
        print("[speech] Using pyttsx3")
        return v
    except Exception as exc:
        print(f"[speech] pyttsx3 init failed: {exc}")
    if sys.platform == "win32":
        print("[speech] Falling back to PowerShell SpeechSynthesizer")
        return _PowerShellVoice()
    raise RuntimeError("No TTS backend available")


def format_alert(det: dict, approaching: bool = False) -> str:
    """Human-spoken alert for a fused detection."""
    label = det.get("label", "obstacle")
    direction = _DIR_WORDS.get(det.get("direction", "center"), "ahead")
    bucket = det.get("distance", "far")
    phrase = f"{label} {direction}, {bucket}"
    if approaching:
        phrase += ", approaching"
    if bucket == "near":
        return f"Warning. {phrase}"
    return phrase


def _info_key(det: dict, approaching: bool = False) -> str:
    """Return a hashable summary of what the user would *hear*.

    Used to detect when the spoken info hasn't changed (same label,
    same direction, same distance bucket, same approaching state).
    """
    appr = "A" if approaching else ""
    return f"{det.get('label')}|{det.get('direction')}|{det.get('distance')}|{appr}"


class Speaker:
    """Queue-based TTS announcer with adaptive cooldowns and rate limiting."""

    def __init__(self, enabled: bool = True, log_file: Optional[str] = None):
        self.enabled = enabled
        self.available = False
        self._q: queue.Queue[Optional[tuple[str, bool]]] = queue.Queue(
            maxsize=TTS_QUEUE_SIZE
        )
        self._running = True
        self._voice: _Voice | None = None
        self._lock = threading.Lock()

        # Per-track cooldown state:
        #   track_id → (expiry_time, last_bucket, last_info_key)
        self._cooldowns: dict[int, tuple[float, str, str]] = {}

        # Global rate limiter: timestamp of the last enqueued alert
        self._last_speak_time: float = 0.0

        self._log_fh = None
        if log_file:
            self._log_fh = open(log_file, "a", encoding="utf-8")
        self._fault_cd: float = 0.0

        self._thread = threading.Thread(
            target=self._worker, name="ObstacleGuard-TTS", daemon=True
        )
        self._thread.start()

    def _worker(self) -> None:
        if sys.platform == "win32":
            _coinitialize()
        try:
            self._voice = _make_voice()
            self.available = True
        except Exception as exc:
            print(f"[speech] TTS init failed: {exc}")
            self.available = False
            self.enabled = False
            return

        while self._running:
            try:
                item = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                break
            text, _urgent = item
            try:
                if self._voice is not None:
                    self._voice.speak(text)
            except Exception as exc:
                print(f"[speech] speak failed: {exc}")
                # Recreate the voice once and retry
                try:
                    if sys.platform == "win32":
                        _coinitialize()
                    self._voice = _make_voice()
                    self._voice.speak(text)
                except Exception as exc2:
                    print(f"[speech] retry failed: {exc2}")

    def _enqueue(self, text: str, urgent: bool = False) -> None:
        if not self.enabled or not text:
            return
        if urgent:
            try:
                while True:
                    self._q.get_nowait()
            except queue.Empty:
                pass
        try:
            self._q.put_nowait((text, urgent))
        except queue.Full:
            if not urgent:
                return
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait((text, urgent))
            except queue.Full:
                pass

    def speak(self, text: str, urgent: bool = False) -> None:
        """Speak *text* now (used for tests and fault banners)."""
        self._enqueue(text, urgent=urgent)
        self._log(text)

    def _log(self, text: str) -> None:
        if self._log_fh:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            self._log_fh.write(f"[{ts}] {text}\n")
            self._log_fh.flush()

    def announce(self, top_tracks: List[Track]) -> list[str]:
        """Speak the single highest-priority detection, respecting cooldowns.

        Anti-spam rules (applied in order):
          1. Per-track adaptive cooldown based on distance bucket.
             → Escalation override: object moved to a *closer* bucket
               bypasses cooldown immediately.
          2. Same-info suppression: cooldown is doubled when the spoken
             content (label + direction + distance + approaching) hasn't
             changed since the last announcement for this track.
          3. Global rate limit: at most one announcement every
             GLOBAL_SPEAK_GAP; near and escalation overrides bypass this.
          4. At most ONE alert per call — the highest-priority one that
             passes all checks.  This prevents two tracks from talking
             over each other in the same frame.
        """
        spoken: list[str] = []
        now = time.monotonic()

        for trk in top_tracks:
            det = trk.det
            bucket = det.get("distance", "far")
            tid = trk.id
            info_key = _info_key(det, trk.approaching)
            phrase = format_alert(det, approaching=trk.approaching)
            escalated = False

            # ── per-track cooldown check ────────────────────────
            if tid in self._cooldowns:
                expiry, last_bucket, last_info = self._cooldowns[tid]
                escalated = _BUCKET_ORDER.get(bucket, 0) > _BUCKET_ORDER.get(
                    last_bucket, 0
                )
                if now < expiry and not escalated:
                    continue  # still on cooldown

            # ── global rate limit ───────────────────────────────
            elapsed = now - self._last_speak_time
            if elapsed < GLOBAL_SPEAK_GAP:
                # Near objects and escalation can override the global gap
                if bucket != "near" and not escalated:
                    continue

            # ── compute next cooldown ───────────────────────────
            base_cd = COOLDOWN_BY_DISTANCE.get(bucket, COOLDOWN_SEC)

            # Same-info suppression: if what the user would hear has not
            # changed at all, double the cooldown before storing it.
            if tid in self._cooldowns:
                _, _, prev_info = self._cooldowns[tid]
                if info_key == prev_info:
                    base_cd *= SAME_INFO_MULTIPLIER

            self._cooldowns[tid] = (now + base_cd, bucket, info_key)
            self._last_speak_time = now

            # Near obstacles interrupt whatever is queued
            self._enqueue(phrase, urgent=(bucket == "near"))
            self._log(phrase)
            spoken.append(phrase)

            # Only one alert per frame to prevent audio overlap
            break

        return spoken

    def fault(self, msg: str = "Camera fault"):
        now = time.monotonic()
        if now >= self._fault_cd:
            self._enqueue(msg, urgent=True)
            self._log(msg)
            self._fault_cd = now + 5.0

    def shutdown(self):
        self._running = False
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._log_fh:
            self._log_fh.close()
            self._log_fh = None
