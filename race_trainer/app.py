"""A responsive desktop menu and training monitor. Opening it never starts a job."""

import json
import os
import queue
import subprocess
import sys
import textwrap
import threading
from collections import deque
from datetime import datetime
from pathlib import Path

import pygame

from .config import TEST_SEED_BASE, TRACK_NAMES, EnvConfig, TrainConfig, generated_tracks
from .rendering import (
    ACCENT,
    BG,
    MUTED,
    ORANGE,
    PANEL,
    SIZE,
    TEXT,
    button,
    font,
    text,
    track_layer,
)
from .tracks import make_track

# The menu offers the built-in layouts and a few generated ones. None are trained on.
MENU_TRACKS = TRACK_NAMES + generated_tracks(TEST_SEED_BASE, 9)
STEP_INCREMENT = 250_000


class RaceApp:
    def __init__(self):
        pygame.display.init()
        self.screen = pygame.display.set_mode(SIZE)
        pygame.display.set_caption("Race Lab | ML Race Trainer")
        self.clock = pygame.time.Clock()
        self.track_index = 0
        self.device_index = 0
        self.devices = ("auto", "cuda", "cpu")
        self.steps = TrainConfig().total_steps
        self.mode = "menu"
        self.running = True
        self.message = "Choose a circuit, then take the wheel."
        self.model_path = None
        self.process = None
        self.events = queue.Queue()
        self.progress = deque(maxlen=180)
        self.last_progress = {}
        self.last_evaluation = {}
        self.output = None
        self.stop_requested = self.closing = False
        self.latest_event = None
        self._preview_name = self._preview = None

    @property
    def track_name(self):
        return MENU_TRACKS[self.track_index]

    @property
    def track_label(self):
        return make_track(self.track_name).label

    @property
    def busy(self):
        return self.process is not None and self.process.poll() is None

    def _buttons(self):
        if self.mode == "monitor":
            return {"stop" if self.busy else "back": pygame.Rect(856, 545, 264, 48)}
        buttons = {"track": pygame.Rect(856, 167, 264, 44)}
        if self.mode == "menu":
            buttons.update(
                drive=pygame.Rect(856, 250, 264, 50),
                watch=pygame.Rect(856, 316, 264, 50),
                setup=pygame.Rect(856, 382, 264, 50),
            )
        else:
            buttons.update(
                device=pygame.Rect(856, 260, 264, 44),
                fewer=pygame.Rect(856, 357, 54, 40),
                more=pygame.Rect(1066, 357, 54, 40),
                start=pygame.Rect(856, 466, 264, 50),
                back=pygame.Rect(856, 530, 264, 44),
            )
        return buttons

    def _restore_display(self):
        pygame.display.init()
        self.screen = pygame.display.set_mode(SIZE)
        pygame.display.set_caption("Race Lab | ML Race Trainer")

    def click(self, action):
        if action == "track":
            self.track_index = (self.track_index + 1) % len(MENU_TRACKS)
        elif action == "device":
            self.device_index = (self.device_index + 1) % len(self.devices)
        elif action == "fewer":
            self.steps = max(STEP_INCREMENT, self.steps - STEP_INCREMENT)
        elif action == "more":
            self.steps += STEP_INCREMENT
        elif action == "setup":
            self.mode = "setup"
            self.message = "Review the session settings, then start when ready."
        elif action == "back":
            self.mode = "menu"
            self.message = "Ready for another session."
        elif action == "drive":
            from .viewer import play

            self.running = not play(EnvConfig(track=self.track_name))
            if self.running:
                self._restore_display()
        elif action == "watch":
            self._watch()
        elif action == "start":
            self._launch_training()
        elif action == "stop":
            self.request_stop()

    def _watch(self):
        try:
            import tkinter as tk
            from tkinter.filedialog import askopenfilename

            root = tk.Tk()
            root.withdraw()
            try:
                selected = askopenfilename(
                    title="Choose a version 3 racing model",
                    initialdir=str(Path("runs").resolve()),
                    filetypes=[("PyTorch checkpoints", "*.pt *.pth")],
                )
            finally:
                root.destroy()
            if not selected:
                return
            # Watching from the menu is deliberately light; CUDA replay is available in the CLI.
            from dataclasses import replace

            from .checkpoint import load_policy
            from .viewer import play

            policy, payload = load_policy(selected, "cpu")
            self.model_path = Path(selected)
            config = replace(EnvConfig(**payload["env_config"]), track=self.track_name, laps=1)
            self.running = not play(config, policy)
            if self.running:
                self._restore_display()
        except Exception as exc:
            self.message = str(exc)
            self._restore_display()

    def _launch_training(self):
        if self.busy:
            return
        self.output = Path("runs") / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        command = [
            sys.executable,
            "-m",
            "race_trainer",
            "train",
            "--steps",
            str(self.steps),
            "--device",
            self.devices[self.device_index],
            "--output",
            str(self.output),
            "--json",
        ]
        environment = {
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "PYGAME_HIDE_SUPPORT_PROMPT": "1",
            "PYTHONIOENCODING": "utf-8",
        }
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=environment,
            creationflags=creationflags,
        )
        self.progress.clear()
        self.last_progress, self.last_evaluation = {}, {}
        self.latest_event = None
        self.stop_requested = False
        self.mode = "monitor"
        self.message = "Starting the training process…"

        process = self.process

        def read_output():
            with process.stdout as stream:
                for line in stream:
                    self.events.put(line.rstrip())

        threading.Thread(target=read_output, daemon=True).start()

    def request_stop(self):
        self.stop_requested = True
        self.message = "Stopping and saving the latest checkpoint…"
        if self.output and self.output.exists():
            (self.output / "stop.request").touch()

    def poll_job(self):
        if self.stop_requested and self.busy and self.output and self.output.exists():
            marker = self.output / "stop.request"
            if not marker.exists():
                marker.touch()
        while True:
            try:
                line = self.events.get_nowait()
            except queue.Empty:
                break
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                if line:
                    self.message = line
                continue
            self.latest_event = event.get("event")
            if self.latest_event == "started":
                self.message = event["reason"]
            elif self.latest_event == "progress":
                self.last_progress = event
                self.progress.append(event["mean_progress"])
            elif self.latest_event == "evaluation":
                self.last_evaluation = event["summary"]
            elif self.latest_event in ("finished", "stopped"):
                self.message = f"{self.latest_event.title()}. Checkpoint saved in {self.output}."
                self.model_path = Path(event["checkpoint"])
        if self.process is not None and not self.busy and self.process.returncode != 0:
            self.message = (
                f"Training exited with code {self.process.returncode}. {self.message}"
                if not self.message.startswith("Training exited")
                else self.message
            )
        if self.closing and not self.busy:
            self.running = False

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            if self.busy:
                self.closing = True
                self.request_stop()
            else:
                self.running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            if self.busy:
                self.request_stop()
            elif self.mode != "menu":
                self.click("back")
            else:
                self.running = False
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for action, rect in self._buttons().items():
                if rect.collidepoint(event.pos):
                    try:
                        self.click(action)
                    except (OSError, ValueError, RuntimeError, ImportError) as exc:
                        self.message = str(exc)
                    break

    def draw(self):
        screen = self.screen
        screen.fill(BG)
        text(screen, "RACE LAB", (24, 21), 43, bold=True)
        text(screen, "ML RACE TRAINER   /   VERSION 3", (27, 80), 13, ACCENT, True)
        state = "TRAINING" if self.busy else "READY"
        text(screen, state, (1040, 42), 14, ORANGE if self.busy else ACCENT, True)
        pygame.draw.rect(screen, PANEL, (840, 122, 296, 590), border_radius=12)
        if self.mode == "monitor":
            self._draw_monitor()
        else:
            self._draw_preview()
            text(
                screen,
                "TRAINING SETUP" if self.mode == "setup" else "YOUR SESSION",
                (856, 136),
                13,
                ACCENT,
                True,
            )
        labels = {
            "track": self.track_label + "  ›",
            "drive": "Drive manually",
            "watch": "Watch saved model",
            "setup": "Set up training",
            "back": "Back to menu",
            "device": f"Device: {self.devices[self.device_index].upper()}  ›",
            "fewer": "−",
            "more": "+",
            "start": "Start training",
            "stop": "Saving…" if self.stop_requested else "Stop & save",
        }
        mouse = pygame.mouse.get_pos()
        for action, rect in self._buttons().items():
            button(
                screen,
                rect,
                labels[action],
                primary=action in ("drive", "start"),
                hover=rect.collidepoint(mouse),
            )
        if self.mode == "setup":
            text(screen, "COMPUTE", (856, 231), 12, MUTED, True)
            text(screen, "TRAINING STEPS", (856, 329), 12, MUTED, True)
            number = font(23, True).render(f"{self.steps:,}", True, TEXT)
            screen.blit(number, number.get_rect(center=(988, 377)))
            text(screen, "Trains on generated tracks, so", (856, 411), 14, MUTED)
            text(screen, "every layout shown here is unseen.", (856, 431), 14, MUTED)
        elif self.mode == "menu":
            text(screen, "Explore the circuit by hand,", (856, 477), 15, MUTED)
            text(screen, "or replay a saved driver.", (856, 501), 15, MUTED)
            text(screen, "Training runs save to /runs", (856, 543), 14, MUTED)
            text(screen, "Manual drive uses your CPU.", (856, 566), 14, MUTED)
        message_lines = textwrap.wrap(self.message, width=34)[:4]
        for i, line in enumerate(message_lines):
            text(screen, line, (856, 611 + 20 * i), 13, MUTED)
        text(
            screen,
            "W A S D / ARROWS  Drive   SPACE  Pause   R  Restart   TAB  Sensors   N  New track",
            (25, 726),
            14,
            MUTED,
        )

    def _draw_preview(self):
        if self._preview_name != self.track_name:
            self._preview_name = self.track_name
            self._preview_track = make_track(self.track_name)
            self._preview = track_layer(self._preview_track)
        self.screen.blit(self._preview, (24, 112))
        text(self.screen, self.track_label, (47, 132), 24, bold=True)
        track = self._preview_track
        description = (
            f"{len(track.gates)} gates  /  {'one lap' if track.closed else 'point-to-point sprint'}"
        )
        text(self.screen, description, (49, 169), 14, MUTED)
        text(
            self.screen,
            "Hold a clean line. Reach each gate in order.",
            (49, 674),
            14,
            MUTED,
        )

    def _draw_monitor(self):
        screen = self.screen
        pygame.draw.rect(screen, PANEL, (24, 122, 800, 590), border_radius=12)
        text(screen, "Training progress", (48, 145), 28, bold=True)
        text(screen, "AVERAGE TRACK COMPLETED WHILE EXPLORING", (49, 213), 13, MUTED, True)
        chart = pygame.Rect(56, 264, 730, 306)
        for i in range(5):
            y = chart.y + chart.height * i // 4
            pygame.draw.line(screen, (42, 57, 72), (chart.x, y), (chart.right, y))
        if len(self.progress) > 1:
            points = [
                (
                    chart.x + i * chart.width / (len(self.progress) - 1),
                    chart.bottom - min(1.0, value) * chart.height,
                )
                for i, value in enumerate(self.progress)
            ]
            pygame.draw.lines(screen, ACCENT, False, points, 2)
            text(screen, "100%", (chart.x, chart.y - 24), 14, MUTED)
            text(screen, "0%", (chart.x, chart.bottom + 10), 14, MUTED)
        else:
            text(screen, "Waiting for the first progress report...", (205, 405), 21, MUTED)
        text(
            screen,
            "Stop safely, then continue from your latest checkpoint.",
            (49, 664),
            14,
            MUTED,
        )
        text(screen, "SESSION STATUS", (856, 139), 13, ACCENT, True)
        steps = self.last_progress.get("steps", 0)
        text(screen, f"{steps / self.steps:.0%}", (856, 178), 34, bold=True)
        text(screen, f"{steps:,} of {self.steps:,} steps", (856, 229), 15, MUTED)
        progress = min(1.0, steps / self.steps)
        pygame.draw.rect(screen, (42, 57, 72), (856, 274, 264, 6), border_radius=3)
        pygame.draw.rect(screen, ACCENT, (856, 274, int(progress * 264), 6), border_radius=3)
        loss = self.last_progress.get("loss")
        completion = self.last_evaluation.get("completion_rate")
        speed = self.last_progress.get("steps_per_second")
        rows = [
            ("Exploration", f"{self.last_progress.get('epsilon', 1):.1%}"),
            ("Loss", f"{loss:.4f}" if loss is not None else "warming up"),
            ("Unseen-track finishes", f"{completion:.0%}" if completion is not None else "pending"),
            ("Speed", f"{speed:,.0f} steps/s" if speed else "pending"),
        ]
        for i, (label, value) in enumerate(rows):
            text(screen, label, (856, 310 + i * 49), 15, MUTED)
            rendered = font(16, True).render(value, True, TEXT)
            screen.blit(rendered, (1120 - rendered.get_width(), 310 + i * 49))

    def run(self):
        try:
            while self.running:
                self.poll_job()
                for event in pygame.event.get():
                    self.handle_event(event)
                if not self.running:
                    break
                self.draw()
                pygame.display.flip()
                self.clock.tick(30)
        finally:
            pygame.display.quit()


def run_app():
    RaceApp().run()
