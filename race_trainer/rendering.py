"""Pygame presentation. Headless rendering uses a Surface, never a GPU context."""

import math
from collections import deque

import numpy as np
import pygame

SIZE = (1160, 760)
OFFSET = np.array([24, 112])
BG = (12, 18, 26)
PANEL = (20, 29, 40)
FIELD = (19, 34, 37)
ROAD = (40, 52, 64)
EDGE = (94, 122, 131)
TEXT = (231, 241, 241)
MUTED = (143, 164, 177)
ACCENT = (74, 220, 194)
ORANGE = (255, 177, 88)
RED = (248, 117, 115)
TRACK_LABELS = {"oval": "Oval circuit", "s_track": "S-curve sprint", "square": "Rounded square"}

_fonts = {}


def font(size, bold=False):
    if not pygame.font.get_init():
        pygame.font.init()
    key = (size, bold)
    if key not in _fonts:
        _fonts[key] = pygame.font.SysFont("Segoe UI", size, bold=bold)
    return _fonts[key]


def text(surface, message, position, size=18, color=TEXT, bold=False):
    rendered = font(size, bold).render(str(message), True, color)
    surface.blit(rendered, position)
    return rendered.get_rect(topleft=position)


def button(surface, rect, label, *, primary=False, enabled=True, hover=False):
    color = ACCENT if primary and enabled else (34, 49, 63) if hover else PANEL
    pygame.draw.rect(surface, color, rect, border_radius=9)
    if not primary:
        pygame.draw.rect(surface, (58, 78, 91), rect, 1, border_radius=9)
    fg = BG if primary and enabled else TEXT if enabled else MUTED
    label_surface = font(18, True).render(label, True, fg)
    surface.blit(label_surface, label_surface.get_rect(center=rect.center))


def track_layer(track):
    surface = pygame.Surface((800, 600))
    surface.fill(FIELD)
    for x in range(20, 800, 40):
        for y in range(20, 600, 40):
            pygame.draw.circle(surface, (29, 47, 49), (x, y), 1)
    pygame.draw.polygon(surface, ROAD, track.outer)
    if track.inner is not None:
        pygame.draw.polygon(surface, FIELD, track.inner)
    for polygon in track.polygons:
        pygame.draw.lines(surface, EDGE, True, polygon, 2)
    for distance in np.arange(0, track.length - 12, 30):
        a, _ = track.sample(distance)
        b, _ = track.sample(distance + 12)
        pygame.draw.line(surface, (67, 90, 101), a, b, 1)
    # The final gate is also the start line on circuits.
    gate = track.gates[-1]
    normal = np.array([-gate.tangent[1], gate.tangent[0]])
    for row in range(2):
        for col in range(12):
            a = gate.position + normal * ((col / 12 - 0.5) * track.half_width * 2)
            a += gate.tangent * (row * 5 - 5)
            width = normal * (track.half_width / 6)
            depth = gate.tangent * 5
            color = TEXT if (row + col) % 2 == 0 else BG
            pygame.draw.polygon(surface, color, [a, a + width, a + width + depth, a + depth])
    return surface


class RaceRenderer:
    def __init__(self, mode="human"):
        self.mode = mode
        if mode == "human":
            pygame.display.init()
            self.surface = pygame.display.set_mode(SIZE)
            pygame.display.set_caption("Race Lab | ML Race Trainer")
        else:
            self.surface = pygame.Surface(SIZE)
        self.clock = pygame.time.Clock()
        self.show_sensors = True
        self._track = self._layer = None
        self.trail = deque(maxlen=180)
        self.last_steps = -1

    def draw(self, env, *, mode_label="SIMULATION", paused=False, message=None):
        screen = self.surface
        screen.fill(BG)
        text(screen, "RACE LAB", (24, 19), 35, bold=True)
        text(screen, mode_label, (25, 66), 13, ACCENT, True)
        text(screen, TRACK_LABELS[env.track.name], (840, 33), 20, bold=True)
        text(screen, "SPRINT" if not env.track.closed else "CIRCUIT", (840, 65), 12, MUTED, True)
        if self._track is not env.track:
            self._track, self._layer = env.track, track_layer(env.track)
            self.trail.clear()
        if env.steps < self.last_steps:
            self.trail.clear()
        self.last_steps = env.steps
        screen.blit(self._layer, OFFSET)
        if not self.trail or np.linalg.norm(env.position - self.trail[-1]) > 1:
            self.trail.append(env.position.copy())
        if len(self.trail) > 1:
            pygame.draw.lines(screen, (65, 122, 117), False, np.array(self.trail) + OFFSET, 2)
        gate = env.track.gates[env.current_checkpoint]
        normal = np.array([-gate.tangent[1], gate.tangent[0]]) * (env.track.half_width - 4)
        pygame.draw.line(
            screen, ACCENT, gate.position - normal + OFFSET, gate.position + normal + OFFSET, 2
        )
        if self.show_sensors and not env.crashed:
            rays = pygame.Surface(SIZE, pygame.SRCALPHA)
            for end in env.sensor_ends:
                pygame.draw.line(rays, (*ACCENT, 60), env.position + OFFSET, end + OFFSET)
                pygame.draw.circle(rays, (*ACCENT, 150), end + OFFSET, 2)
            screen.blit(rays, (0, 0))
        corners = env.corners() + OFFSET
        pygame.draw.polygon(screen, (8, 14, 18), corners + 3)
        pygame.draw.polygon(screen, RED if env.crashed else ORANGE, corners)
        c, s = math.cos(env.angle), math.sin(env.angle)
        glass = np.array([[2, -4], [7, -4], [7, 4], [2, 4]])
        pygame.draw.polygon(
            screen, (29, 53, 65), glass @ np.array([[c, s], [-s, c]]) + env.position + OFFSET
        )
        self._hud(env)
        if paused or env._finished:
            label = message or (
                "FINISHED"
                if env.completed
                else "CRASHED"
                if env.crashed
                else "EPISODE ENDED"
                if env._finished
                else "PAUSED"
            )
            overlay = pygame.Surface((800, 600), pygame.SRCALPHA)
            overlay.fill((7, 13, 21, 155))
            screen.blit(overlay, OFFSET)
            heading = font(38, True).render(label, True, TEXT)
            screen.blit(heading, heading.get_rect(center=(424, 376)))
            hint = font(18).render("R to restart  /  Esc to return", True, MUTED)
            screen.blit(hint, hint.get_rect(center=(424, 423)))
        text(
            screen,
            "SPACE  Pause     R  Restart     TAB  Sensors     ESC  Return",
            (25, 726),
            14,
            MUTED,
        )
        text(screen, "30 Hz simulation", (996, 726), 14, MUTED)
        if self.mode == "human":
            pygame.display.flip()
            pygame.event.pump()
            self.clock.tick(30)
            return None
        return np.transpose(pygame.surfarray.array3d(screen), (1, 0, 2)).copy()

    def _hud(self, env):
        screen = self.surface
        pygame.draw.rect(screen, PANEL, (840, 112, 296, 600), border_radius=12)
        x = 860
        text(screen, "LIVE TELEMETRY", (x, 133), 13, ACCENT, True)
        text(screen, f"{env.speed:03.0f}", (x, 161), 64, bold=True)
        text(screen, "speed · units / second", (x, 240), 15, MUTED)
        pygame.draw.rect(screen, (38, 52, 65), (x, 277, 256, 5), border_radius=2)
        pygame.draw.rect(
            screen,
            ORANGE,
            (x, 277, int(256 * env.speed / env.config.max_speed), 5),
            border_radius=2,
        )
        info = env._info()
        rows = [
            ("Checkpoints", f"{info['checkpoints_passed']} / {info['total_checkpoints']}"),
            ("Race time", f"{info['elapsed_seconds']:.2f} s"),
            ("Progress", f"{info['progress']:.0%}"),
            ("Reward", f"{env.episode_reward:+.1f}"),
        ]
        for i, (label, value) in enumerate(rows):
            y = 310 + i * 44
            text(screen, label, (x, y), 16, MUTED)
            rendered = font(18, True).render(value, True, TEXT)
            screen.blit(rendered, (1115 - rendered.get_width(), y - 1))
        pygame.draw.line(screen, (43, 59, 72), (x, 500), (1116, 500))
        text(screen, "DRIVER CONTROLS", (x, 521), 13, ACCENT, True)
        for i, line in enumerate(
            (
                "W / ↑      Accelerate",
                "S / ↓        Brake",
                "A D / ← →    Steer",
                "Release throttle to coast",
            )
        ):
            text(screen, line, (x, 553 + 29 * i), 15, MUTED)

    def close(self):
        if self.mode == "human":
            pygame.display.quit()
