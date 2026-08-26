#!/usr/bin/env python3
"""
ObstacleGuard desktop GUI.

Load a video (or webcam), run the same detection / depth / priority pipeline
as the CLI, show the annotated stream, list priority flags, and play TTS / beeps.
"""

from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from capture import Capture
from pipeline import FrameResult, ObstaclePipeline, load_models


_BG = "#101418"
_PANEL = "#1a2128"
_TEXT = "#e8eef4"
_MUTED = "#8b9aab"
_ACCENT = "#3d8bfd"
_NEAR = "#ff4d4d"
_MID = "#ffc14d"
_FAR = "#3dd68c"


class ObstacleGuardApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("ObstacleGuard")
        self.root.minsize(1100, 680)
        self.root.configure(bg=_BG)

        self._detector = None
        self._depth_est = None
        self._pipe: ObstaclePipeline | None = None
        self._cap: Capture | None = None

        self._worker: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._frame_q: queue.Queue = queue.Queue(maxsize=2)
        self._photo = None
        self._source_path: str | None = None
        self._running = False
        self._models_ready = False

        self._build_style()
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(80, self._drain_queue)
        self.root.after(200, self._start_model_load)

    # ── style ───────────────────────────────────────────────────
    def _build_style(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(".", background=_BG, foreground=_TEXT, fieldbackground=_PANEL)
        style.configure("TFrame", background=_BG)
        style.configure("Panel.TFrame", background=_PANEL)
        style.configure("TLabel", background=_BG, foreground=_TEXT, font=("Segoe UI", 10))
        style.configure("Muted.TLabel", background=_BG, foreground=_MUTED)
        style.configure("Title.TLabel", background=_BG, foreground=_TEXT, font=("Segoe UI", 16, "bold"))
        style.configure("Panel.TLabel", background=_PANEL, foreground=_TEXT)
        style.configure("TCheckbutton", background=_BG, foreground=_TEXT, font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 10), padding=6)
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))
        style.configure("TProgressbar", troughcolor=_PANEL, background=_ACCENT)
        style.configure(
            "Treeview",
            background=_PANEL,
            foreground=_TEXT,
            fieldbackground=_PANEL,
            rowheight=26,
            font=("Consolas", 9),
        )
        style.configure("Treeview.Heading", background="#24303a", foreground=_TEXT, font=("Segoe UI", 9, "bold"))
        style.map("Treeview", background=[("selected", "#2a4a73")])

    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(outer)
        header.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(header, text="ObstacleGuard", style="Title.TLabel").pack(side=tk.LEFT)
        ttk.Label(
            header,
            text="  Phase-0  ·  detect  ·  depth  ·  priority  ·  audio",
            style="Muted.TLabel",
        ).pack(side=tk.LEFT, padx=(8, 0))
        self.status_var = tk.StringVar(value="Loading models…")
        ttk.Label(header, textvariable=self.status_var, style="Muted.TLabel").pack(side=tk.RIGHT)

        body = ttk.Frame(outer)
        body.pack(fill=tk.BOTH, expand=True)

        # Video
        left = ttk.Frame(body)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        self.video_label = tk.Label(
            left,
            bg="#0a0c0e",
            fg=_MUTED,
            text="Choose a video file, then press Start",
            font=("Segoe UI", 12),
            width=80,
            height=24,
        )
        self.video_label.pack(fill=tk.BOTH, expand=True)
        self.progress = ttk.Progressbar(left, mode="determinate")
        self.progress.pack(fill=tk.X, pady=(8, 0))
        self.progress_lbl = ttk.Label(left, text="No source loaded", style="Muted.TLabel")
        self.progress_lbl.pack(anchor=tk.W, pady=(4, 0))

        # Side panel
        right = ttk.Frame(body, width=380)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        right.pack_propagate(False)

        ttk.Label(right, text="Priority flags", font=("Segoe UI", 11, "bold")).pack(anchor=tk.W)
        cols = ("role", "id", "label", "dir", "dist", "appr", "score")
        self.tree = ttk.Treeview(right, columns=cols, show="headings", height=8)
        headings = {
            "role": "Role",
            "id": "ID",
            "label": "Class",
            "dir": "Dir",
            "dist": "Dist",
            "appr": "Appr",
            "score": "Score",
        }
        widths = {"role": 58, "id": 36, "label": 72, "dir": 56, "dist": 48, "appr": 40, "score": 48}
        for c in cols:
            self.tree.heading(c, text=headings[c])
            self.tree.column(c, width=widths[c], anchor=tk.CENTER)
        self.tree.pack(fill=tk.X, pady=(4, 8))
        self.tree.tag_configure("near", foreground=_NEAR)
        self.tree.tag_configure("mid", foreground=_MID)
        self.tree.tag_configure("far", foreground=_FAR)
        self.tree.tag_configure("alert", background="#2a1c1c")

        stats = ttk.Frame(right)
        stats.pack(fill=tk.X, pady=(0, 8))
        self.fps_var = tk.StringVar(value="FPS: —")
        self.obj_var = tk.StringVar(value="Objects: 0")
        self.near_var = tk.StringVar(value="Nearest: —")
        ttk.Label(stats, textvariable=self.fps_var).pack(side=tk.LEFT)
        ttk.Label(stats, textvariable=self.obj_var).pack(side=tk.LEFT, padx=12)
        ttk.Label(stats, textvariable=self.near_var).pack(side=tk.LEFT)

        ttk.Label(right, text="Alert log", font=("Segoe UI", 11, "bold")).pack(anchor=tk.W, pady=(4, 0))
        log_wrap = ttk.Frame(right)
        log_wrap.pack(fill=tk.BOTH, expand=True, pady=(4, 8))
        self.log = tk.Text(
            log_wrap,
            height=12,
            bg=_PANEL,
            fg=_TEXT,
            insertbackground=_TEXT,
            relief=tk.FLAT,
            font=("Consolas", 9),
            wrap=tk.WORD,
        )
        scroll = ttk.Scrollbar(log_wrap, command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        self.log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log.tag_configure("near", foreground=_NEAR)
        self.log.tag_configure("mid", foreground=_MID)
        self.log.tag_configure("far", foreground=_FAR)
        self.log.tag_configure("sys", foreground=_MUTED)
        self.log.configure(state=tk.DISABLED)

        # Controls
        controls = ttk.Frame(outer)
        controls.pack(fill=tk.X, pady=(12, 0))

        ttk.Button(controls, text="Open video…", command=self._browse).pack(side=tk.LEFT)
        ttk.Button(controls, text="Webcam", command=self._use_webcam).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(controls, text="Start", style="Accent.TButton", command=self._start).pack(
            side=tk.LEFT, padx=(16, 0)
        )
        ttk.Button(controls, text="Pause / resume", command=self._toggle_pause).pack(
            side=tk.LEFT, padx=(6, 0)
        )
        ttk.Button(controls, text="Stop", command=self._stop).pack(side=tk.LEFT, padx=(6, 0))

        self.speech_var = tk.BooleanVar(value=True)
        self.beep_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            controls, text="Speech (TTS)", variable=self.speech_var, command=self._sync_audio
        ).pack(side=tk.LEFT, padx=(20, 0))
        ttk.Checkbutton(
            controls, text="Proximity beeps", variable=self.beep_var, command=self._sync_audio
        ).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(controls, text="Test voice", command=self._test_voice).pack(
            side=tk.LEFT, padx=(12, 0)
        )

        ttk.Button(controls, text="Clear log", command=self._clear_log).pack(side=tk.RIGHT)

        self.file_var = tk.StringVar(value="Source: none")
        ttk.Label(outer, textvariable=self.file_var, style="Muted.TLabel").pack(
            anchor=tk.W, pady=(8, 0)
        )

    # ── model load ──────────────────────────────────────────────
    def _start_model_load(self):
        self._log_line("Loading YOLOv8n and MiDaS-small (first run downloads weights)…", "sys")

        def work():
            try:
                det, depth = load_models(lambda msg: self.root.after(0, self.status_var.set, msg))
                self.root.after(0, lambda: self._on_models_ready(det, depth))
            except Exception as exc:
                self.root.after(0, lambda: self._on_models_failed(exc))

        threading.Thread(target=work, daemon=True).start()

    def _on_models_ready(self, det, depth):
        self._detector = det
        self._depth_est = depth
        self._pipe = ObstaclePipeline(
            det, depth, speech=self.speech_var.get(), beep=self.beep_var.get()
        )
        self._models_ready = True
        self.status_var.set("Ready — open a video and press Start")
        self._log_line("Models loaded. Pipeline ready.", "sys")
        self.root.after(600, self._voice_ready_chime)

    def _on_models_failed(self, exc: Exception):
        self.status_var.set("Model load failed")
        self._log_line(f"Failed to load models: {exc}", "near")
        messagebox.showerror(
            "ObstacleGuard",
            f"Could not load detection / depth models:\n\n{exc}",
        )

    # ── source ──────────────────────────────────────────────────
    def _browse(self):
        path = filedialog.askopenfilename(
            title="Open video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mkv *.mov *.webm *.mpg *.mpeg"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self._source_path = path
        self.file_var.set(f"Source: {path}")
        self.status_var.set("Video selected — press Start")
        self._log_line(f"Selected video: {Path(path).name}", "sys")

    def _use_webcam(self):
        self._source_path = 0
        self.file_var.set("Source: webcam (index 0)")
        self.status_var.set("Webcam selected — press Start")
        self._log_line("Selected webcam 0", "sys")

    # ── run control ─────────────────────────────────────────────
    def _start(self):
        if not self._models_ready or self._pipe is None:
            messagebox.showinfo("ObstacleGuard", "Models are still loading. Please wait.")
            return
        if self._source_path is None:
            messagebox.showinfo("ObstacleGuard", "Open a video file (or choose Webcam) first.")
            return
        if self._worker is not None and self._worker.is_alive():
            messagebox.showinfo("ObstacleGuard", "Already running. Press Stop first.")
            return

        try:
            cap = Capture(self._source_path)
        except Exception as exc:
            messagebox.showerror("ObstacleGuard", str(exc))
            return

        self._cap = cap
        self._pipe.reset()
        self._sync_audio()
        self._stop_event.clear()
        self._pause_event.clear()
        self._running = True
        self.status_var.set("Running")
        self._log_line("Pipeline started.", "sys")

        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    def _toggle_pause(self):
        if not self._running:
            return
        if self._pause_event.is_set():
            self._pause_event.clear()
            if self._pipe:
                self._pipe.set_beep(self.beep_var.get())
            self.status_var.set("Running")
            self._log_line("Resumed.", "sys")
        else:
            self._pause_event.set()
            if self._pipe:
                self._pipe.beeper.update(None)
            self.status_var.set("Paused")
            self._log_line("Paused.", "sys")

    def _stop(self):
        if not self._running and (self._worker is None or not self._worker.is_alive()):
            return
        self._stop_event.set()
        self._pause_event.clear()
        if self._pipe:
            self._pipe.beeper.update(None)
        self.status_var.set("Stopping…")

    def _loop(self):
        cap = self._cap
        pipe = self._pipe
        if cap is None or pipe is None:
            return
        try:
            while not self._stop_event.is_set():
                if self._pause_event.is_set():
                    time.sleep(0.05)
                    continue
                if self._cap is not cap:
                    break
                frame = cap.read()
                if frame is None:
                    if cap.eof:
                        self._put({"kind": "eof"})
                        break
                    if cap.fault:
                        result = pipe.process_fault()
                        self._put({"kind": "frame", "result": result, "progress": cap.progress()})
                        pipe.speaker.fault("Camera feed lost")
                    time.sleep(0.02)
                    continue
                result = pipe.process_frame(frame)
                self._put({"kind": "frame", "result": result, "progress": cap.progress()})
        except Exception as exc:
            self._put({"kind": "error", "error": str(exc)})
        finally:
            if self._cap is cap:
                cap.release()
                self._cap = None
            self._put({"kind": "done"})

    def _put(self, item: dict):
        try:
            self._frame_q.put_nowait(item)
        except queue.Full:
            try:
                self._frame_q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frame_q.put_nowait(item)
            except queue.Full:
                pass

    def _drain_queue(self):
        try:
            while True:
                item = self._frame_q.get_nowait()
                kind = item.get("kind")
                if kind == "frame":
                    self._show_result(item["result"], item.get("progress"))
                elif kind == "eof":
                    self.status_var.set("Video finished")
                    self._log_line("End of video.", "sys")
                    if self._pipe:
                        self._pipe.beeper.update(None)
                elif kind == "error":
                    self.status_var.set("Error")
                    self._log_line(item["error"], "near")
                    messagebox.showerror("ObstacleGuard", item["error"])
                elif kind == "done":
                    was_stop = self.status_var.get() == "Stopping…"
                    self._running = False
                    if was_stop:
                        self.status_var.set("Stopped")
                        self.progress_lbl.configure(text="Stopped")
                        self._log_line("Pipeline stopped.", "sys")
                    elif self.status_var.get() not in ("Video finished", "Error"):
                        self.status_var.set("Idle")
        except queue.Empty:
            pass
        self.root.after(30, self._drain_queue)

    def _show_result(self, result: FrameResult, progress):
        rgb = cv2.cvtColor(result.vis, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb)
        # Fit the label width without upscaling too far
        lw = max(self.video_label.winfo_width(), 640)
        lh = max(self.video_label.winfo_height(), 480)
        im.thumbnail((lw, lh), Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(image=im)
        self.video_label.configure(image=self._photo, text="")

        self.fps_var.set(f"FPS: {result.fps:.1f}")
        active = [t for t in result.tracks if t.lost == 0]
        self.obj_var.set(f"Objects: {len(active)}")
        nearest = result.nearest_distance or "—"
        self.near_var.set(f"Nearest: {nearest}")

        if progress:
            pos, total = progress
            if total > 0:
                self.progress["maximum"] = total
                self.progress["value"] = pos
                self.progress_lbl.configure(text=f"Frame {pos} / {total}")
            else:
                self.progress["value"] = 0
                self.progress_lbl.configure(text=f"Live  ·  frame {pos}")

        self._refresh_tree(result)

        for phrase in result.announced:
            tag = "far"
            if ", near" in phrase:
                tag = "near"
            elif ", mid" in phrase:
                tag = "mid"
            self._log_line(phrase, tag)

    def _refresh_tree(self, result: FrameResult):
        self.tree.delete(*self.tree.get_children())
        top_ids = {t.id for t in result.top}
        rows = []
        for trk in result.tracks:
            if trk.lost > 0:
                continue
            det = trk.det
            score = result.scores.get(trk.id, 0.0)
            rows.append((score, trk, det))
        rows.sort(key=lambda r: r[0], reverse=True)
        for score, trk, det in rows:
            role = "ALERT" if trk.id in top_ids else "watch"
            dist = det["distance"]
            tags = (dist, "alert") if role == "ALERT" else (dist,)
            self.tree.insert(
                "",
                tk.END,
                values=(
                    role,
                    trk.id,
                    det["label"],
                    det["direction"],
                    dist,
                    "yes" if trk.approaching else "",
                    f"{score:.1f}",
                ),
                tags=tags,
            )

    def _voice_ready_chime(self, tries: int = 0):
        if self._pipe is None:
            return
        sp = self._pipe.speaker
        if not sp.available:
            if tries < 8:
                self.root.after(400, lambda: self._voice_ready_chime(tries + 1))
            else:
                self._log_line(
                    "Voice engine did not start. Click Test voice after unmuting speakers.",
                    "near",
                )
            return
        self._log_line("Voice engine ready (Windows SAPI / default speakers).", "sys")
        if self.speech_var.get():
            sp.speak("ObstacleGuard voice alerts are on", urgent=True)
            self._log_line("Spoken: ObstacleGuard voice alerts are on", "sys")

    def _test_voice(self):
        if self._pipe is None:
            messagebox.showinfo("ObstacleGuard", "Wait for models to finish loading first.")
            return
        self.speech_var.set(True)
        self._pipe.set_speech(True)
        if not self._pipe.speaker.available:
            messagebox.showwarning(
                "ObstacleGuard",
                "The voice engine is not available yet.\n\n"
                "Unmute speakers, wait a second, and try Test voice again.",
            )
            return
        phrase = "Warning. person ahead, near"
        self._pipe.speaker.speak(phrase, urgent=True)
        self._log_line(f"Voice test: {phrase}", "near")

    def _sync_audio(self):
        if self._pipe is None:
            return
        self._pipe.set_speech(self.speech_var.get())
        self._pipe.set_beep(self.beep_var.get() and self._running and not self._pause_event.is_set())

    def _log_line(self, text: str, tag: str = "sys"):
        ts = time.strftime("%H:%M:%S")
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, f"[{ts}] {text}\n", tag)
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    def _clear_log(self):
        self.log.configure(state=tk.NORMAL)
        self.log.delete("1.0", tk.END)
        self.log.configure(state=tk.DISABLED)

    def _on_close(self):
        self._stop_event.set()
        self._running = False
        if self._pipe is not None:
            self._pipe.shutdown()
            self._pipe = None
        self.root.destroy()


def run_gui():
    root = tk.Tk()
    ObstacleGuardApp(root)
    root.mainloop()


if __name__ == "__main__":
    run_gui()
