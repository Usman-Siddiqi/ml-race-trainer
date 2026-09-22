"""One cached geometry for rendering, collision, raycasts, progress, and gates."""

from dataclasses import dataclass

import numpy as np

from .config import TRACK_NAMES


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _inside_polygon(points, polygon):
    points = np.atleast_2d(points)
    a, b = polygon, np.roll(polygon, -1, axis=0)
    px, py = points[:, 0, None], points[:, 1, None]
    crosses_y = (a[:, 1] > py) != (b[:, 1] > py)
    dy = b[:, 1] - a[:, 1]
    safe_dy = np.where(np.abs(dy) > 1e-12, dy, 1.0)
    hit_x = a[:, 0] + (py - a[:, 1]) * (b[:, 0] - a[:, 0]) / safe_dy
    return np.count_nonzero(crosses_y & (px < hit_x), axis=1) % 2 == 1


@dataclass(frozen=True)
class Gate:
    position: np.ndarray
    tangent: np.ndarray
    distance: float

    def crossed(self, previous, current, half_width):
        before = float(np.dot(previous - self.position, self.tangent))
        after = float(np.dot(current - self.position, self.tangent))
        if not (before < 0 <= after):
            return False
        fraction = -before / (after - before)
        intersection = previous + fraction * (current - previous)
        return abs(float(_cross(self.tangent, intersection - self.position))) <= half_width


class Track:
    def __init__(self, name, centerline, half_width, closed, checkpoint_count):
        self.name, self.closed, self.half_width = name, closed, float(half_width)
        self.centerline = np.asarray(centerline, dtype=np.float64)
        self.starts = self.centerline if closed else self.centerline[:-1]
        ends = np.roll(self.centerline, -1, axis=0) if closed else self.centerline[1:]
        self.vectors = ends - self.starts
        self.lengths = np.linalg.norm(self.vectors, axis=1)
        self.lengths_squared = self.lengths**2
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
            # All built-in loops have positive signed area in screen coordinates.
            self.outer, self.inner = right, left
            self.polygons = (self.outer, self.inner)
        else:
            self.outer, self.inner = np.vstack((left, right[::-1])), None
            self.polygons = (self.outer,)
        self.wall_starts = np.vstack(self.polygons)
        wall_ends = np.vstack([np.roll(p, -1, axis=0) for p in self.polygons])
        self.wall_vectors = wall_ends - self.wall_starts
        self.start_distance = 0.0 if closed else 25.0
        self.finish_distance = self.length if closed else self.length - 25.0
        distances = np.linspace(self.start_distance, self.finish_distance, checkpoint_count + 1)[1:]
        self.gates = tuple(Gate(*self.sample(s), float(s)) for s in distances)

    def sample(self, distance):
        distance = distance % self.length if self.closed else np.clip(distance, 0, self.length)
        index = min(
            int(np.searchsorted(self.cumulative, distance, side="right") - 1), len(self.lengths) - 1
        )
        point = self.starts[index] + self.tangents[index] * (distance - self.cumulative[index])
        return point, self.tangents[index]

    def project(self, point):
        relative = point - self.starts
        fractions = np.clip(np.sum(relative * self.vectors, axis=1) / self.lengths_squared, 0, 1)
        nearest = self.starts + fractions[:, None] * self.vectors
        differences = point - nearest
        index = int(np.argmin(np.sum(differences**2, axis=1)))
        distance = self.cumulative[index] + fractions[index] * self.lengths[index]
        lateral = float(_cross(self.tangents[index], differences[index]))
        return float(distance), lateral

    def contains(self, points):
        inside = _inside_polygon(points, self.outer)
        if self.inner is not None:
            inside &= ~_inside_polygon(points, self.inner)
        return inside

    def raycast(self, origin, directions, max_distance):
        # All beams intersect all boundary segments in one NumPy operation.
        rays = np.asarray(directions)[:, None, :]
        walls = self.wall_vectors[None, :, :]
        relative = self.wall_starts[None, :, :] - origin
        denominator = _cross(rays, walls)
        safe = np.where(np.abs(denominator) > 1e-10, denominator, 1.0)
        ray_t = _cross(relative, walls) / safe
        wall_t = _cross(relative, rays) / safe
        valid = (np.abs(denominator) > 1e-10) & (ray_t >= 0) & (wall_t >= 0) & (wall_t <= 1)
        return np.minimum(np.min(np.where(valid, ray_t, np.inf), axis=1), max_distance)


def make_track(name):
    if name == "oval":
        theta = np.linspace(0, 2 * np.pi, 96, endpoint=False)
        points = np.column_stack((400 + 270 * np.cos(theta), 300 + 190 * np.sin(theta)))
        return Track(name, points, 44, True, 12)
    if name == "s_track":
        t = np.linspace(0, 1, 65)
        points = np.column_stack((120 + 560 * t, 300 + 135 * np.sin(2 * np.pi * t)))
        return Track(name, points, 43, False, 6)
    if name == "square":
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
    raise ValueError(f"Unknown track {name!r}; choose from {TRACK_NAMES}")
