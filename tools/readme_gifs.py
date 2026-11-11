"""Record the README GIFs from a trained model. Not part of training or tests.

Every track shown is held out of training. Run from the repository root with Pillow
importable (it is not a project dependency), for example:

    uv pip install --target <folder> pillow
    $env:PYTHONPATH = '<folder>'
    & $racePython tools\\readme_gifs.py --model runs\\v3-default\\best.pt

Frames come from the same simulator and renderer as `watch`; the simulator runs at 30
steps per second and every second step is kept, so the GIFs play in real time.
"""

import argparse
import os
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np  # noqa: E402
import pygame  # noqa: E402
from PIL import Image  # noqa: E402

from race_trainer.agent import ACTIONS  # noqa: E402
from race_trainer.checkpoint import load_policy  # noqa: E402
from race_trainer.config import EnvConfig  # noqa: E402
from race_trainer.environment import RaceCarEnv  # noqa: E402
from race_trainer.rendering import (  # noqa: E402
    ACCENT,
    BG,
    MUTED,
    ORANGE,
    RED,
    TEXT,
    RaceRenderer,
    text,
    track_layer,
)
from race_trainer.simulation import RaceSim  # noqa: E402
from race_trainer.tracks import make_track  # noqa: E402

FRAME_SKIP = 2
FRAME_MS = 1000 * FRAME_SKIP // 30
HOLD_FRAMES = 20


def to_image(surface, size=None):
    pixels = np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2))
    image = Image.fromarray(pixels)
    return image.resize(size, Image.LANCZOS) if size else image


CAR_COLOURS = [
    ORANGE,
    RED,
    (122, 182, 255),
    (196, 140, 255),
    (255, 226, 102),
    (120, 230, 150),
]


def save_gif(frames, path):
    # One shared palette keeps colours stable between frames and the file small. It is
    # built from frames across the clip plus a swatch, so small cars keep their colour.
    width, height = frames[0].size
    picks = frames[:: max(1, len(frames) // 6)][:6]
    montage = Image.new("RGB", (width, height * len(picks) + 40))
    for i, frame in enumerate(picks):
        montage.paste(frame, (0, i * height))
    swatch = [*CAR_COLOURS, ACCENT, TEXT, MUTED]
    for i, colour in enumerate(swatch):
        montage.paste(
            colour, (i * width // len(swatch), height * len(picks), width, montage.height)
        )
    palette = montage.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    quantized = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    path.parent.mkdir(parents=True, exist_ok=True)
    quantized[0].save(
        path,
        save_all=True,
        append_images=quantized[1:],
        duration=FRAME_MS,
        loop=0,
        optimize=True,
        disposal=1,
    )
    print(f"{path}  {len(frames)} frames  {path.stat().st_size / 1e6:.2f} MB")


def draw_car(surface, corners, angle, position, color):
    pygame.draw.polygon(surface, (8, 14, 18), corners + 3)
    pygame.draw.polygon(surface, color, corners)
    c, s = np.cos(angle), np.sin(angle)
    glass = np.array([[2, -4], [7, -4], [7, 4], [2, 4]])
    pygame.draw.polygon(surface, (29, 53, 65), glass @ np.array([[c, s], [-s, c]]) + position)


def draw_rays(surface, position, ends, alpha=110):
    rays = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    for end in ends:
        pygame.draw.line(rays, (*ACCENT, alpha), position, end)
        pygame.draw.circle(rays, (*ACCENT, 160), end, 2)
    surface.blit(rays, (0, 0))


def replay(policy, env_config, track, seed):
    """Full `watch` screen for one run, including the telemetry panel."""
    env = RaceCarEnv(replace(env_config, track=track))
    renderer = RaceRenderer("rgb_array")
    observation, _ = env.reset(seed=seed)
    frames = []
    while not env._finished:
        for _ in range(FRAME_SKIP):
            if env._finished:
                break
            observation, *_ = env.step(ACTIONS[policy.act(observation)])
        renderer.draw(env, mode_label="MODEL REPLAY")
        frames.append(to_image(renderer.surface, (696, 456)))
    frames += [frames[-1]] * HOLD_FRAMES
    result = "finished" if env.completed else "crashed" if env.crashed else "stopped"
    print(f"{track}: {result} in {env.steps / 30:.1f} s")
    env.close()
    return frames


def grid(policy, env_config, tracks, seed):
    """One car on each of four tracks, driven at the same time."""
    envs = [RaceCarEnv(replace(env_config, track=t)) for t in tracks]
    observations = [env.reset(seed=seed)[0] for env in envs]
    layers = [track_layer(env.track) for env in envs]
    trails = [[env.position.copy()] for env in envs]
    frames = []
    while not all(env._finished for env in envs):
        for _ in range(FRAME_SKIP):
            for i, env in enumerate(envs):
                if not env._finished:
                    observations[i], *_ = env.step(ACTIONS[policy.act(observations[i])])
                    trails[i].append(env.position.copy())
        canvas = Image.new("RGB", (800, 600))
        for i, env in enumerate(envs):
            cell = layers[i].copy()
            pygame.draw.lines(cell, (65, 122, 117), False, np.array(trails[i]), 3)
            if not env._finished:
                draw_rays(cell, env.position, env.sensor_ends)
            color = RED if env.crashed else ORANGE
            draw_car(cell, env.corners(), env.angle, env.position, color)
            pygame.draw.rect(cell, BG, (0, 0, 800, 54))
            text(cell, env.track.label, (18, 8), 30, TEXT, True)
            status = (
                f"FINISHED  {env.steps / 30:.1f} s"
                if env.completed
                else "CRASHED"
                if env.crashed
                else f"{env.speed:3.0f} units/s"
            )
            text(cell, status, (520, 14), 24, ACCENT if env.completed else MUTED, True)
            canvas.paste(to_image(cell, (400, 300)), ((i % 2) * 400, (i // 2) * 300))
        frames.append(canvas)
    frames += [frames[-1]] * HOLD_FRAMES
    for env in envs:
        print(f"{env.track.name}: {'finished' if env.completed else 'not finished'}")
    return frames


def swarm(policy, env_config, track, count, seconds, seed):
    """Many cars on one track at once, like training. Cars do not collide with each other."""
    config = replace(env_config, track=track)
    sim = RaceSim(config, count)
    course = make_track(track)
    rng = np.random.default_rng(seed)
    for i in range(count):
        sim.reset_env(i, course, rng, random_position=True)
    layer = track_layer(course)
    trails = [[p.copy()] for p in sim.position]
    hues = CAR_COLOURS
    frames = []
    for _ in range(int(seconds * 30 / FRAME_SKIP)):
        for _ in range(FRAME_SKIP):
            active = ~sim.finished
            actions = ACTIONS[policy.act_batch(sim.observe())]
            actions[~active] = 0
            sim.step(actions)
            for i in np.flatnonzero(active):
                trails[i].append(sim.position[i].copy())
                del trails[i][:-45]
        surface = layer.copy()
        corners = sim.corners()
        for i in range(count):
            if len(trails[i]) > 1:
                pygame.draw.lines(surface, (65, 122, 117), False, np.array(trails[i]), 3)
        for i in range(count):
            draw_car(surface, corners[i], sim.angle[i], sim.position[i], hues[i % len(hues)])
        pygame.draw.rect(surface, BG, (0, 0, 800, 40))
        text(
            surface, f"{count} cars, one shared policy  |  {course.label}", (14, 6), 22, TEXT, True
        )
        frames.append(to_image(surface, (640, 480)))
    crashed = int(sim.crashed.sum())
    print(f"{track}: {count} cars, {crashed} crashed in {seconds} s")
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, default=Path("runs/v3-default/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("docs/media"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    pygame.init()
    policy, payload = load_policy(args.model, "cpu")
    env_config = replace(EnvConfig(**payload["env_config"]), laps=1)
    save_gif(replay(policy, env_config, "random:2000000000", args.seed), args.output / "replay.gif")
    unseen = ("random:2000000003", "random:2000000001", "random:2000000006", "s_track")
    save_gif(grid(policy, env_config, unseen, args.seed), args.output / "unseen-tracks.gif")
    save_gif(
        swarm(policy, env_config, "random:2000000002", 12, 8, args.seed),
        args.output / "many-cars.gif",
    )


if __name__ == "__main__":
    main()
