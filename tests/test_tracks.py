import math

import numpy as np
import pytest

from race_trainer.config import (
    TEST_SEED_BASE,
    TRACK_NAMES,
    VALIDATION_SEED_BASE,
    EnvConfig,
    check_track,
    heldout_tracks,
    validation_tracks,
)
from race_trainer.environment import RaceCarEnv
from race_trainer.simulation import SENSOR_ANGLES
from race_trainer.tracks import CANVAS, TrackBatch, generate_track, make_track

GENERATED = tuple(f"random:{seed}" for seed in range(24))
SAMPLED = (*TRACK_NAMES, *GENERATED[:9])


def test_generated_tracks_are_reproducible_varied_and_inside_the_canvas():
    tracks = [make_track(name) for name in GENERATED]
    for name, track in zip(GENERATED, tracks, strict=True):
        again = generate_track(int(name.split(":")[1]))
        np.testing.assert_array_equal(track.centerline, again.centerline)
        assert track.name == name and 28 <= track.half_width <= 44
        for polygon in track.polygons:
            assert (polygon > 0).all() and (polygon < np.array(CANVAS)).all()
        assert len(track.gates) >= 3 and track.length > 400
    assert {track.closed for track in tracks} == {True, False}
    assert len({track.centerline.tobytes() for track in tracks}) == len(tracks)
    assert make_track("random:3") is make_track("random:3")


def test_generated_tracks_turn_both_ways_and_respect_the_car():
    left = right = 0
    for name in GENERATED:
        track = make_track(name)
        headings = np.arctan2(track.tangents[:, 1], track.tangents[:, 0])
        turns = (np.diff(headings) + np.pi) % (2 * np.pi) - np.pi
        left += np.count_nonzero(turns > 0.02)
        right += np.count_nonzero(turns < -0.02)
        # Three-segment turning never exceeds the tightest corner the generator allows.
        window = np.abs(turns[:-2] + turns[1:-1] + turns[2:]) / 36
        assert window.max() <= 1 / max(track.half_width + 18, 55) + 1e-9
    assert left > 100 and right > 100


def test_track_names_and_held_out_seed_ranges():
    for name in ("oval", "random:0", "random:2147483647"):
        assert check_track(name) == name
    for name in ("figure8", "random", "random:-1", "random:x", "random:2147483648"):
        with pytest.raises(ValueError, match="track"):
            check_track(name)
    assert check_track("random", allow_pool=True) == "random"
    validation, heldout = validation_tracks(16), heldout_tracks(20)
    assert validation[0] == f"random:{VALIDATION_SEED_BASE}"
    assert heldout[:3] == TRACK_NAMES and heldout[3] == f"random:{TEST_SEED_BASE}"
    assert not set(validation) & set(heldout) and len(heldout) == 23
    labels = [make_track(name).label for name in ("oval", "random:3", validation[1], heldout[5])]
    assert labels[0] == "Oval circuit" and labels[1].startswith("Generated ")
    assert labels[2].startswith("Validation ") and labels[2].endswith(" 1")
    assert labels[3].startswith("Test ") and labels[3].endswith(" 2")


@pytest.mark.parametrize("name", SAMPLED)
def test_sensors_intersect_the_actual_collision_boundaries(name):
    track = make_track(name)
    for distance in np.linspace(track.start_distance + 10, track.finish_distance - 10, 8):
        position, tangent = track.sample(distance)
        angle = math.atan2(tangent[1], tangent[0])
        directions = np.column_stack((np.cos(angle + SENSOR_ANGLES), np.sin(angle + SENSOR_ANGLES)))
        lengths = track.raycast(position, directions, 1000)
        before = position + directions * (lengths - 0.01)[:, None]
        after = position + directions * (lengths + 0.01)[:, None]
        assert track.contains(before).all()
        assert not track.contains(after).any()


def test_a_mixed_batch_answers_exactly_like_each_track_alone():
    names = ("oval", "random:1", "s_track", "random:3", "square")
    batch = TrackBatch(len(names))
    tracks = [make_track(name) for name in names]
    # Assign the largest layout last so earlier rows are re-padded when storage grows.
    for index in np.argsort([len(track.wall_starts) for track in tracks]):
        batch.assign(index, tracks[index])
    rng = np.random.default_rng(4)
    origins = np.array([t.sample(rng.uniform(30, t.length - 30))[0] for t in tracks])
    origins += rng.uniform(-8, 8, origins.shape)
    angles = rng.uniform(-np.pi, np.pi, (len(names), 7))
    directions = np.stack((np.cos(angles), np.sin(angles)), axis=2)
    points = origins[:, None, :] + rng.uniform(-70, 70, (len(names), 12, 2))
    rays = batch.raycast(origins, directions, 500)
    road = batch.on_road(points)
    distance, lateral = batch.project(origins)
    assert road.any() and not road.all()
    for index, track in enumerate(tracks):
        alone = track.raycast(origins[index], directions[index], 500)
        np.testing.assert_array_equal(rays[index], alone)
        np.testing.assert_array_equal(road[index], track.contains(points[index]))
        assert (distance[index], lateral[index]) == track.project(origins[index])
    subset = np.array([3, 1])
    np.testing.assert_array_equal(
        batch.raycast(origins[subset], directions[subset], 500, subset), rays[subset]
    )


@pytest.mark.parametrize("name", (*TRACK_NAMES, *GENERATED))
def test_tracks_are_drivable_with_a_fixed_geometry_controller(name):
    """Physical lap validation using fixed arithmetic, with no model or learning."""
    env = RaceCarEnv(EnvConfig(track=name, random_start=False))
    env.reset()
    for _ in range(env.config.max_steps):
        distance, _ = env.track.project(env.position)
        target, _ = env.track.sample(distance + 45)
        delta = target - env.position
        alpha = math.atan2(delta[1], delta[0]) - env.angle
        steering = math.atan2(2 * env.config.wheelbase * math.sin(alpha), np.linalg.norm(delta))
        throttle = np.clip(
            (90 - env.speed) * 0.1 + env.config.drag / env.config.acceleration, -1, 1
        )
        _, _, terminated, truncated, info = env.step([throttle, steering / env.config.max_steering])
        if terminated or truncated:
            break
    assert info["completed"], info
    assert info["progress"] == 1.0 and not info["crashed"]
