"""One cached geometry for rendering, collision, raycasts, progress, and gates.

Every query is written once, in batched form. A single track answers through the same
functions as a batch of many environments, so the simulator cannot drift from the picture.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .config import (
    TEST_SEED_BASE,
    TRACK_NAMES,
    VALIDATION_SEED_BASE,
    check_track,
    track_seed,
)

CANVAS = (800.0, 600.0)
GENERATOR_VERSION = 1
_FAR = 1e9


SPRINT_RUNOFF = 100.0
_RAYCAST_ELEMENTS = 80_000


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def batch_raycast(wall_starts, wall_vectors, origins, directions, max_distance):
    """Distances from N origins along R rays each to the nearest of W wall segments.

    Shapes: walls (N, W, 2), origins (N, 2), directions (N, R, 2). Returns (N, R).
    """
    # This is the simulator's hot loop. Environments are processed in cache-sized slices
    # and temporaries are reused; each element sees the same arithmetic in any batch.
    count, rays, walls = len(origins), directions.shape[1], wall_starts.shape[1]
    result = np.empty((count, rays))
    size = max(1, _RAYCAST_ELEMENTS // (rays * walls))
    for first in range(0, count, size):
        part = slice(first, first + size)
        ray_x, ray_y = directions[part, :, 0, None], directions[part, :, 1, None]
        wall_x, wall_y = wall_vectors[part, None, :, 0], wall_vectors[part, None, :, 1]
        relative = wall_starts[part] - origins[part, None, :]
        relative_x, relative_y = relative[:, None, :, 0], relative[:, None, :, 1]
        denominator = ray_x * wall_y
        scratch = ray_y * wall_x
        denominator -= scratch
        miss = np.abs(denominator, out=scratch) <= 1e-10
        denominator[miss] = 1.0
        ray_t = (relative_x * wall_y - relative_y * wall_x) / denominator
        wall_t = np.multiply(relative_x, ray_y, out=scratch)
        wall_t -= relative_y * ray_x
        wall_t /= denominator
        miss |= ray_t < 0
        miss |= wall_t < 0
        miss |= wall_t > 1
        ray_t[miss] = np.inf
        result[part] = ray_t.min(axis=2)
    return np.minimum(result, max_distance, out=result)


def batch_on_road(wall_starts, wall_end_y, wall_slopes, points):
    """Even-odd test of P points per environment against all road boundary segments.

    The outer and inner boundaries are both in the wall list: a point on the road crosses
    an odd number of them. Shapes: walls (N, W, ...), points (N, P, 2). Returns (N, P).
    """
    start_x, start_y = wall_starts[:, None, :, 0], wall_starts[:, None, :, 1]
    point_x, point_y = points[:, :, 0, None], points[:, :, 1, None]
    crosses = (start_y > point_y) != (wall_end_y[:, None, :] > point_y)
    hit_x = start_x + (point_y - start_y) * wall_slopes[:, None, :]
    return np.count_nonzero(crosses & (point_x < hit_x), axis=2) % 2 == 1


def batch_project(starts, vectors, lengths, cumulative, tangents, points):
    """Arc distance along the centerline and signed lateral offset for N points."""
    relative = points[:, None, :] - starts
    fractions = np.clip(np.sum(relative * vectors, axis=2) / (lengths * lengths), 0, 1)
    differences = relative - fractions[:, :, None] * vectors
    rows = np.arange(len(points))
    index = np.argmin(np.sum(differences * differences, axis=2), axis=1)
    distance = cumulative[rows, index] + fractions[rows, index] * lengths[rows, index]
    lateral = _cross(tangents[rows, index], differences[rows, index])
    return distance, lateral


def gate_crossed(previous, current, position, tangent, half_width):
    """Forward crossings of a gate line within the road width; accepts batches."""
    before = np.sum((previous - position) * tangent, axis=-1)
    after = np.sum((current - position) * tangent, axis=-1)
    forward = (before < 0) & (after >= 0)
    span = np.where(forward, after - before, 1.0)
    intersection = previous + (-before / span)[..., None] * (current - previous)
    return forward & (np.abs(_cross(tangent, intersection - position)) <= half_width)


@dataclass(frozen=True)
class Gate:
    position: np.ndarray
    tangent: np.ndarray
    distance: float

    def crossed(self, previous, current, half_width):
        return bool(
            gate_crossed(
                np.asarray(previous, dtype=np.float64),
                np.asarray(current, dtype=np.float64),
                self.position,
                self.tangent,
                half_width,
            )
        )


class Track:
    def __init__(self, name, centerline, half_width, closed, checkpoint_count):
        self.name, self.closed, self.half_width = name, bool(closed), float(half_width)
        self.centerline = np.asarray(centerline, dtype=np.float64)
        self.starts = self.centerline if closed else self.centerline[:-1]
        ends = np.roll(self.centerline, -1, axis=0) if closed else self.centerline[1:]
        self.vectors = ends - self.starts
        self.lengths = np.linalg.norm(self.vectors, axis=1)
        self.tangents = self.vectors / self.lengths[:, None]
        self.cumulative = np.r_[0.0, np.cumsum(self.lengths)]
        self.length = float(self.cumulative[-1])
        normals = np.column_stack((-self.tangents[:, 1], self.tangents[:, 0]))
        if closed:
            before, after = np.roll(normals, 1, axis=0), normals
        else:
            before = np.vstack((normals[0], normals))
            after = np.vstack((normals, normals[-1]))
        bisector = before + after
        bisector /= np.linalg.norm(bisector, axis=1, keepdims=True)
        offset = bisector * (half_width / np.sum(bisector * after, axis=1))[:, None]
        left, right = self.centerline + offset, self.centerline - offset
        if closed:
            # The loop may run in either direction; the larger offset loop is the outside.
            left_is_outer = abs(_signed_area(left)) > abs(_signed_area(right))
            self.outer, self.inner = (left, right) if left_is_outer else (right, left)
            self.polygons = (self.outer, self.inner)
        else:
            self.outer, self.inner = np.vstack((left, right[::-1])), None
            self.polygons = (self.outer,)
        self.wall_starts = np.vstack(self.polygons)
        wall_ends = np.vstack([np.roll(p, -1, axis=0) for p in self.polygons])
        self.wall_vectors = wall_ends - self.wall_starts
        self.wall_end_y = wall_ends[:, 1].copy()
        rise = self.wall_vectors[:, 1]
        self.wall_slopes = self.wall_vectors[:, 0] / np.where(np.abs(rise) > 1e-12, rise, 1.0)
        # A sprint ends in a wall. Its finish line sits a braking distance before that wall:
        # a car that reads only its sensors cannot tell a finish from a dead end, so it
        # must be able to cross the line at speed and still have road left.
        self.start_distance = 0.0 if closed else 25.0
        self.finish_distance = self.length if closed else self.length - SPRINT_RUNOFF
        distances = np.linspace(self.start_distance, self.finish_distance, checkpoint_count + 1)[1:]
        self.gate_distances = distances
        self.gates = tuple(Gate(*self.sample(s), float(s)) for s in distances)
        self.gate_positions = np.array([gate.position for gate in self.gates])
        self.gate_tangents = np.array([gate.tangent for gate in self.gates])

    @property
    def label(self):
        seed = track_seed(self.name)
        if seed is None:
            return {
                "oval": "Oval circuit",
                "s_track": "S-curve sprint",
                "square": "Rounded square",
            }[self.name]
        # The seed range says what a generated layout is for; show that, not ten digits.
        kind = "circuit" if self.closed else "sprint"
        if seed >= TEST_SEED_BASE:
            return f"Test {kind} {seed - TEST_SEED_BASE}"
        if seed >= VALIDATION_SEED_BASE:
            return f"Validation {kind} {seed - VALIDATION_SEED_BASE}"
        return f"Generated {kind} {seed}"

    def sample(self, distance):
        distance = distance % self.length if self.closed else np.clip(distance, 0, self.length)
        index = min(
            int(np.searchsorted(self.cumulative, distance, side="right") - 1), len(self.lengths) - 1
        )
        point = self.starts[index] + self.tangents[index] * (distance - self.cumulative[index])
        return point, self.tangents[index]

    def project(self, point):
        distance, lateral = batch_project(
            self.starts[None],
            self.vectors[None],
            self.lengths[None],
            self.cumulative[None, :-1],
            self.tangents[None],
            np.asarray(point, dtype=np.float64)[None],
        )
        return float(distance[0]), float(lateral[0])

    def contains(self, points):
        points = np.atleast_2d(np.asarray(points, dtype=np.float64))
        return batch_on_road(
            self.wall_starts[None], self.wall_end_y[None], self.wall_slopes[None], points[None]
        )[0]

    def raycast(self, origin, directions, max_distance):
        return batch_raycast(
            self.wall_starts[None],
            self.wall_vectors[None],
            np.asarray(origin, dtype=np.float64)[None],
            np.asarray(directions, dtype=np.float64)[None],
            max_distance,
        )[0]


class TrackBatch:
    """Padded geometry for N environments, each of which may use a different track."""

    def __init__(self, count):
        self.count = count
        self.tracks = [None] * count
        self.length = np.ones(count)
        self.half_width = np.ones(count)
        self.closed = np.zeros(count, dtype=bool)
        self.gate_count = np.ones(count, dtype=np.int64)
        self._allocate(walls=1, segments=1, gates=1)

    def _allocate(self, walls, segments, gates):
        n = self.count
        # Padding sits far away with zero extent: no ray, point, or projection reaches it.
        self.wall_starts = np.full((n, walls, 2), _FAR)
        self.wall_vectors = np.zeros((n, walls, 2))
        self.wall_end_y = np.full((n, walls), _FAR)
        self.wall_slopes = np.zeros((n, walls))
        self.starts = np.full((n, segments, 2), _FAR)
        self.vectors = np.zeros((n, segments, 2))
        self.lengths = np.ones((n, segments))
        self.cumulative = np.zeros((n, segments))
        self.tangents = np.zeros((n, segments, 2))
        self.gate_positions = np.zeros((n, gates, 2))
        self.gate_tangents = np.zeros((n, gates, 2))

    def _grow(self, walls, segments, gates):
        shape = (self.wall_starts.shape[1], self.starts.shape[1], self.gate_positions.shape[1])
        if walls <= shape[0] and segments <= shape[1] and gates <= shape[2]:
            return
        self._allocate(max(walls, shape[0]), max(segments, shape[1]), max(gates, shape[2]))
        for index, track in enumerate(self.tracks):
            if track is not None:
                self._write(index, track)

    def _write(self, index, track):
        walls, segments, gates = len(track.wall_starts), len(track.starts), len(track.gates)
        self.wall_starts[index, :walls] = track.wall_starts
        self.wall_starts[index, walls:] = _FAR
        self.wall_vectors[index, :walls] = track.wall_vectors
        self.wall_vectors[index, walls:] = 0
        self.wall_end_y[index, :walls] = track.wall_end_y
        self.wall_end_y[index, walls:] = _FAR
        self.wall_slopes[index, :walls] = track.wall_slopes
        self.wall_slopes[index, walls:] = 0
        self.starts[index, :segments] = track.starts
        self.starts[index, segments:] = _FAR
        self.vectors[index, :segments] = track.vectors
        self.vectors[index, segments:] = 0
        self.lengths[index, :segments] = track.lengths
        self.lengths[index, segments:] = 1
        self.cumulative[index, :segments] = track.cumulative[:-1]
        self.tangents[index, :segments] = track.tangents
        self.gate_positions[index, :gates] = track.gate_positions
        self.gate_tangents[index, :gates] = track.gate_tangents

    def assign(self, index, track):
        if self.tracks[index] is track:
            return
        self.tracks[index] = track
        self._grow(len(track.wall_starts), len(track.starts), len(track.gates))
        self._write(index, track)
        self.length[index], self.half_width[index] = track.length, track.half_width
        self.closed[index], self.gate_count[index] = track.closed, len(track.gates)

    def raycast(self, origins, directions, max_distance, rows=None):
        if rows is None:
            return batch_raycast(
                self.wall_starts, self.wall_vectors, origins, directions, max_distance
            )
        return batch_raycast(
            self.wall_starts[rows], self.wall_vectors[rows], origins, directions, max_distance
        )

    def on_road(self, points):
        return batch_on_road(self.wall_starts, self.wall_end_y, self.wall_slopes, points)

    def project(self, points):
        return batch_project(
            self.starts, self.vectors, self.lengths, self.cumulative, self.tangents, points
        )


def _signed_area(polygon):
    x, y = polygon[:, 0], polygon[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _resample_loop(points, spacing):
    loop = np.vstack((points, points[:1]))
    cumulative = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(loop, axis=0), axis=1))]
    count = max(int(round(cumulative[-1] / spacing)), 12)
    targets = np.linspace(0, cumulative[-1], count, endpoint=False)
    return np.column_stack(
        (np.interp(targets, cumulative, loop[:, 0]), np.interp(targets, cumulative, loop[:, 1]))
    )


def _loop_is_drivable(loop, half_width, spacing):
    """Reject loops that are too sharp, pinch against themselves, or leave the canvas."""
    margin = half_width + 10
    if (loop < margin).any() or (loop > np.array(CANVAS) - margin).any():
        return False
    vectors = np.roll(loop, -1, axis=0) - loop
    headings = np.arctan2(vectors[:, 1], vectors[:, 0])
    turns = (np.diff(np.r_[headings, headings[0]]) + np.pi) % (2 * np.pi) - np.pi
    # Averaging three turns measures the corner radius rather than resampling noise.
    window = np.abs(turns + np.roll(turns, 1) + np.roll(turns, -1)) / (3 * spacing)
    minimum_radius = max(half_width + 18, 55.0)
    if window.max() > 1 / minimum_radius or np.abs(turns).max() > 1.6 * spacing / minimum_radius:
        return False
    count = len(loop)
    steps = np.abs(np.arange(count)[:, None] - np.arange(count)[None, :])
    arc = np.minimum(steps, count - steps) * spacing
    gaps = np.linalg.norm(loop[:, None, :] - loop[None, :, :], axis=2)
    # A chord of a legal corner is at least 0.6 of its arc; anything closer is a pinch.
    return bool((gaps >= np.minimum(2 * half_width + 20, 0.6 * arc) - 1e-9).all())


def _counter_turning(loop):
    """Radians of turning against the loop's overall direction: zero for a plain oval."""
    vectors = np.roll(loop, -1, axis=0) - loop
    headings = np.arctan2(vectors[:, 1], vectors[:, 0])
    turns = (np.diff(np.r_[headings, headings[0]]) + np.pi) % (2 * np.pi) - np.pi
    return (np.abs(turns).sum() - abs(turns.sum())) / 2


def _random_loop(rng, half_width, spacing):
    """A smooth closed curve from a few random polar harmonics, scaled into the canvas."""
    theta = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    radius = np.ones_like(theta)
    amplitude = rng.uniform(0.2, 0.4)
    for mode in range(2, int(rng.integers(3, 6)) + 1):
        phase = rng.uniform(0, 2 * np.pi)
        radius += rng.uniform(0, amplitude) / np.sqrt(mode - 1) * np.cos(mode * theta + phase)
    if radius.min() < 0.25:
        return None
    points = radius[:, None] * np.column_stack((np.cos(theta), np.sin(theta)))
    low, high = points.min(axis=0), points.max(axis=0)
    room = np.array(CANVAS) - 2 * (half_width + 11)
    size = room * rng.uniform(0.8, 1.0, 2)
    corner = half_width + 11 + (room - size) * rng.uniform(0, 1, 2)
    loop = _resample_loop(corner + (points - low) / (high - low) * size, spacing)
    return loop if _loop_is_drivable(loop, half_width, spacing) else None


def generate_track(seed):
    """A reproducible random layout: mostly circuits, sometimes point-to-point sprints."""
    rng = np.random.default_rng([GENERATOR_VERSION, seed])
    spacing = 12.0
    half_width = float(rng.uniform(28, 44))
    # Most layouts keep the twistiest of a few candidates so S-bends are common, while
    # the rest take the first valid shape and so include simple ovals.
    wanted = 1 if rng.random() < 0.3 else 3
    loops = []
    for _ in range(5000):
        loop = _random_loop(rng, half_width, spacing)
        if loop is not None:
            loops.append(loop)
            if len(loops) == wanted:
                break
    if not loops:
        raise RuntimeError(f"Could not generate a drivable track for seed {seed}")
    loop = max(loops, key=_counter_turning)
    if rng.random() < 0.5:
        loop = loop[::-1]
    loop = np.roll(loop, -int(rng.integers(len(loop))), axis=0)
    name = f"random:{seed}"
    if rng.random() < 0.2:
        # A sprint is an open stretch of a valid loop, so it is valid as well.
        keep = int(len(loop) * rng.uniform(0.5, 0.8))
        return Track(name, loop[: keep + 1], half_width, False, max(3, round(keep / 9)))
    return Track(name, loop, half_width, True, max(6, round(len(loop) / 9)))


def _builtin(name):
    if name == "oval":
        theta = np.linspace(0, 2 * np.pi, 96, endpoint=False)
        points = np.column_stack((400 + 270 * np.cos(theta), 300 + 190 * np.sin(theta)))
        return Track(name, points, 44, True, 12)
    if name == "s_track":
        t = np.linspace(0, 1, 65)
        points = np.column_stack((120 + 560 * t, 300 + 135 * np.sin(2 * np.pi * t)))
        return Track(name, points, 43, False, 6)
    points = []
    for center, start in (
        ((585, 195), -90),
        ((585, 405), 0),
        ((215, 405), 90),
        ((215, 195), 180),
    ):
        angles = np.deg2rad(np.linspace(start, start + 90, 17))
        points.extend(
            np.column_stack((center[0] + 65 * np.cos(angles), center[1] + 65 * np.sin(angles)))
        )
    return Track(name, points, 40, True, 12)


@lru_cache(maxsize=4096)
def make_track(name):
    """Build a named layout or ``random:<seed>``. Tracks are immutable and cached."""
    check_track(name)
    return _builtin(name) if name in TRACK_NAMES else generate_track(track_seed(name))
