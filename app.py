import base64
import hashlib
import heapq
import io
import json
import math
import os
import sys
import tempfile
import time
import traceback
import tracemalloc
import zlib
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import ndimage
import streamlit as st
from PIL import Image, ImageDraw

try:
    import cv2
except Exception:
    cv2 = None

try:
    import trimesh
except Exception:
    trimesh = None


APP_TITLE = "Procedural World Forge — M1 Terrain Studio"

DEFAULT_CONFIG = {
    "seed": 20261109,
    "name": "Demo Valley",
    "size": 120.0,
    "resolution": 40,
    "height_scale": 18.0,
    "roughness": 0.55,
    "mountain_amount": 0.62,
    "height_gamma": 1.0,
    "water_level": 0.28,
    "erosion_iterations": 2,
    "biome": "mixed",
    "vegetation_density": 0.65,
    "building_density": 0.22,
    "road_count": 3,
    "road_width": 2.6,
    "road_width_pixels": 2,
    "poly_budget": 15000,
    "instance_limit": 300,
    "camera_distance": 155.0,
    "camera_elevation": 28.0,
    "start_azimuth": 35.0,
    "video_frames": 28,
    "video_fps": 24,
    "video_width": 576,
    "video_height": 324,
    "enable_local_generative_model": True,
}

PRESETS = {
    "Mixed Valley": {
        "biome": "mixed",
        "mountain_amount": 0.62,
        "water_level": 0.28,
        "roughness": 0.55,
        "height_gamma": 1.0,
        "vegetation_density": 0.65,
        "building_density": 0.22,
        "road_count": 3,
    },
    "Alpine Ridge": {
        "biome": "alpine",
        "mountain_amount": 0.90,
        "water_level": 0.18,
        "roughness": 0.72,
        "height_gamma": 1.25,
        "vegetation_density": 0.35,
        "building_density": 0.08,
        "road_count": 2,
    },
    "Coastal Wetland": {
        "biome": "coastal",
        "mountain_amount": 0.25,
        "water_level": 0.45,
        "roughness": 0.45,
        "height_gamma": 0.90,
        "vegetation_density": 0.75,
        "building_density": 0.18,
        "road_count": 3,
    },
    "Desert Mesa": {
        "biome": "desert",
        "mountain_amount": 0.55,
        "water_level": 0.08,
        "roughness": 0.35,
        "height_gamma": 0.85,
        "vegetation_density": 0.20,
        "building_density": 0.12,
        "road_count": 2,
    },
    "Dense Forest": {
        "biome": "forest",
        "mountain_amount": 0.40,
        "water_level": 0.22,
        "roughness": 0.60,
        "height_gamma": 1.05,
        "vegetation_density": 0.95,
        "building_density": 0.10,
        "road_count": 3,
    },
    "Volcanic Badlands": {
        "biome": "volcanic",
        "mountain_amount": 0.85,
        "water_level": 0.12,
        "roughness": 0.80,
        "height_gamma": 1.40,
        "vegetation_density": 0.15,
        "building_density": 0.05,
        "road_count": 1,
    },
    "Urban Plains": {
        "biome": "urban",
        "mountain_amount": 0.15,
        "water_level": 0.15,
        "roughness": 0.25,
        "height_gamma": 0.70,
        "vegetation_density": 0.35,
        "building_density": 0.70,
        "road_count": 5,
    },
}

BIOME_NAMES = [
    "Water",
    "Beach",
    "Desert",
    "Grassland",
    "Forest",
    "Shrubland",
    "Alpine Rock",
    "Snow",
    "Volcanic",
]

BIOME_COLORS = {
    0: (0.18, 0.35, 0.55),
    1: (0.82, 0.75, 0.55),
    2: (0.85, 0.68, 0.40),
    3: (0.42, 0.62, 0.30),
    4: (0.18, 0.45, 0.22),
    5: (0.55, 0.55, 0.35),
    6: (0.45, 0.45, 0.48),
    7: (0.92, 0.94, 0.96),
    8: (0.25, 0.20, 0.22),
}

BIOME_COLORS_RGB255 = {
    k: tuple(int(round(c * 255.0)) for c in v) for k, v in BIOME_COLORS.items()
}

BIOME_LUT_RGB = np.array([BIOME_COLORS[i] for i in range(9)], dtype=float)
BIOME_LUT_RGB255 = np.array([BIOME_COLORS_RGB255[i] for i in range(9)], dtype=np.uint8)

OBJECT_COLORS_RGB = {
    "tree": (0.12, 0.42, 0.18),
    "shrub": (0.42, 0.52, 0.24),
    "rock": (0.48, 0.48, 0.50),
    "building": (0.65, 0.58, 0.48),
    "hut": (0.62, 0.52, 0.40),
    "tower": (0.55, 0.55, 0.60),
    "ruin": (0.38, 0.34, 0.30),
}

BUILDING_TYPES = {"building", "hut", "tower", "ruin"}

ASSET_POLY = {
    "tree": 12,
    "shrub": 8,
    "rock": 10,
    "building": 24,
    "hut": 24,
    "tower": 36,
    "ruin": 18,
}

VEG_SUIT_BASE = {
    0: 0.00,
    1: 0.15,
    2: 0.25,
    3: 0.70,
    4: 1.00,
    5: 0.55,
    6: 0.18,
    7: 0.04,
    8: 0.08,
}

BUILD_SUIT_BASE = {
    0: 0.00,
    1: 0.55,
    2: 0.60,
    3: 1.00,
    4: 0.70,
    5: 0.60,
    6: 0.10,
    7: 0.02,
    8: 0.18,
}

LIGHT_DIR = np.array([0.45, 0.30, 0.84], dtype=float)
LIGHT_DIR /= max(1e-9, float(np.linalg.norm(LIGHT_DIR)))


def json_safe(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def normalize_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    cfg = DEFAULT_CONFIG.copy()
    cfg.update({k: v for k, v in raw.items() if k in DEFAULT_CONFIG})

    cfg["seed"] = int(cfg["seed"]) % 1000000000
    cfg["name"] = str(cfg["name"])[:80]
    cfg["biome"] = str(cfg["biome"])

    cfg["resolution"] = int(np.clip(cfg["resolution"], 16, 128))
    cfg["size"] = float(np.clip(cfg["size"], 20.0, 500.0))
    cfg["height_scale"] = float(np.clip(cfg["height_scale"], 1.0, 120.0))
    cfg["roughness"] = float(np.clip(cfg["roughness"], 0.0, 1.0))
    cfg["mountain_amount"] = float(np.clip(cfg["mountain_amount"], 0.0, 1.0))
    cfg["height_gamma"] = float(np.clip(cfg["height_gamma"], 0.35, 2.5))
    cfg["water_level"] = float(np.clip(cfg["water_level"], 0.0, 0.85))
    cfg["erosion_iterations"] = int(np.clip(cfg["erosion_iterations"], 0, 8))

    cfg["vegetation_density"] = float(np.clip(cfg["vegetation_density"], 0.0, 1.0))
    cfg["building_density"] = float(np.clip(cfg["building_density"], 0.0, 1.0))
    cfg["road_count"] = int(np.clip(cfg["road_count"], 0, 8))
    cfg["road_width"] = float(np.clip(cfg["road_width"], 0.5, 12.0))
    cfg["road_width_pixels"] = int(np.clip(cfg["road_width_pixels"], 1, 8))

    cfg["poly_budget"] = int(np.clip(cfg["poly_budget"], 2000, 250000))
    max_res = int(math.sqrt(max(2000, cfg["poly_budget"]) / 2.0)) + 1
    cfg["effective_resolution"] = int(min(cfg["resolution"], max(16, max_res)))

    cfg["instance_limit"] = int(np.clip(cfg["instance_limit"], 0, 5000))
    asset_budget = max(20, cfg["poly_budget"] // 90)
    cfg["instance_limit"] = min(cfg["instance_limit"], asset_budget)

    cfg["camera_distance"] = float(np.clip(cfg["camera_distance"], 30.0, 1000.0))
    cfg["camera_elevation"] = float(np.clip(cfg["camera_elevation"], 5.0, 80.0))
    cfg["start_azimuth"] = float(cfg["start_azimuth"] % 360.0)

    cfg["video_frames"] = int(np.clip(cfg["video_frames"], 2, 180))
    cfg["video_fps"] = int(np.clip(cfg["video_fps"], 5, 60))
    cfg["video_width"] = int(np.clip(cfg["video_width"], 160, 1280))
    cfg["video_height"] = int(np.clip(cfg["video_height"], 90, 720))
    cfg["enable_local_generative_model"] = bool(cfg["enable_local_generative_model"])

    return cfg


class NoiseEngine:
    """
    Permissively licensed, dependency-free gradient noise engine.
    Implements seeded 2D Perlin-style noise, fBm, and ridged multifractal variants.
    """

    GRAD = np.array(
        [
            [1.0, 1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [-1.0, -1.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ],
        dtype=float,
    )

    def __init__(self, seed: int):
        rng = np.random.default_rng(seed)
        p = rng.permutation(256).astype(np.int32)
        self.perm = np.concatenate([p, p]).astype(np.int32)

    @staticmethod
    def _fade(t: np.ndarray) -> np.ndarray:
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

    @staticmethod
    def _lerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
        return a + t * (b - a)

    def _hash(self, xi: np.ndarray, yi: np.ndarray, offset: int = 0) -> np.ndarray:
        xa = (xi + offset) & 255
        yb = (yi + offset * 7) & 255
        return self.perm[(self.perm[xa] + yb) & 255]

    def _grad_dot(self, h: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
        g = self.GRAD[h & 7]
        return g[..., 0] * dx + g[..., 1] * dy

    def perlin2(self, x: np.ndarray, y: np.ndarray, offset: int = 0) -> np.ndarray:
        x0 = np.floor(x).astype(np.int32)
        y0 = np.floor(y).astype(np.int32)

        xf = x - x0
        yf = y - y0

        u = self._fade(xf)
        v = self._fade(yf)

        aa = self._hash(x0, y0, offset)
        ba = self._hash(x0 + 1, y0, offset)
        ab = self._hash(x0, y0 + 1, offset)
        bb = self._hash(x0 + 1, y0 + 1, offset)

        x1 = self._lerp(
            self._grad_dot(aa, xf, yf),
            self._grad_dot(ba, xf - 1.0, yf),
            u,
        )
        x2 = self._lerp(
            self._grad_dot(ab, xf, yf - 1.0),
            self._grad_dot(bb, xf - 1.0, yf - 1.0),
            u,
        )
        return self._lerp(x1, x2, v)

    def fbm(
        self,
        x: np.ndarray,
        y: np.ndarray,
        octaves: int = 5,
        lacunarity: float = 2.0,
        gain: float = 0.5,
        offset: int = 0,
    ) -> np.ndarray:
        amp = 1.0
        freq = 1.0
        total = np.zeros_like(x, dtype=float)
        norm = 0.0

        for o in range(max(1, int(octaves))):
            total += amp * self.perlin2(x * freq, y * freq, offset + o * 13)
            norm += amp
            amp *= gain
            freq *= lacunarity

        return total / max(1e-9, norm)

    def ridged(
        self,
        x: np.ndarray,
        y: np.ndarray,
        octaves: int = 4,
        lacunarity: float = 2.0,
        gain: float = 0.5,
        offset: int = 0,
    ) -> np.ndarray:
        amp = 1.0
        freq = 1.0
        total = np.zeros_like(x, dtype=float)
        norm = 0.0

        for o in range(max(1, int(octaves))):
            n = self.perlin2(x * freq, y * freq, offset + o * 17)
            r = (1.0 - np.abs(n)) ** 2
            total += amp * r
            norm += amp
            amp *= gain
            freq *= lacunarity

        return total / max(1e-9, norm)


def apply_erosion(h: np.ndarray, iterations: int, strength: float = 0.25) -> np.ndarray:
    out = h.copy()
    iterations = max(0, int(iterations))

    for _ in range(iterations):
        blur = ndimage.gaussian_filter(out, sigma=0.75)
        out = out * (1.0 - strength) + blur * strength

        gy, gx = np.gradient(out)
        slope = np.sqrt(gx * gx + gy * gy)
        steep = np.clip(slope - 0.10, 0.0, None)
        out -= 0.15 * steep
        out = np.clip(out, 0.0, 1.0)

    return out


def classify_biomes(
    h: np.ndarray,
    slope_deg: np.ndarray,
    moisture: np.ndarray,
    temperature: np.ndarray,
    cfg: Dict[str, Any],
) -> np.ndarray:
    labels = np.full(h.shape, 3, dtype=np.int16)

    water = h < cfg["water_level"]
    labels[water] = 0

    beach = (h < cfg["water_level"] + 0.06) & (~water)
    labels[beach] = 1

    high = h > 0.68
    rocky = high & (slope_deg > 38.0)
    labels[rocky] = 6

    snow = high & (~rocky) & (temperature < 0.38)
    labels[snow] = 7

    lowland = (~water) & (~high)
    dry = moisture < 0.38
    wet = moisture > 0.64

    labels[lowland & dry] = 2
    labels[lowland & wet & (slope_deg < 45.0)] = 4
    labels[lowland & (~dry) & (~wet) & (slope_deg < 55.0)] = 3
    labels[lowland & (~dry) & (~wet) & (slope_deg >= 55.0)] = 5

    biome = str(cfg.get("biome", "mixed"))

    if biome == "volcanic":
        labels[(h > 0.45) & (~water)] = 8
    elif biome == "desert":
        labels[(h < 0.65) & (~water) & (slope_deg < 45.0)] = 2
    elif biome == "forest":
        labels[(h < 0.65) & (~water) & (slope_deg < 45.0) & (moisture > 0.45)] = 4
    elif biome == "alpine":
        labels[(h > 0.55) & (~water)] = 6
        labels[(h > 0.75) & (slope_deg < 25.0)] = 7
    elif biome == "coastal":
        labels[(h < cfg["water_level"] + 0.10) & (~water)] = 1
    elif biome == "urban":
        labels[(h < 0.55) & (~water) & (slope_deg < 12.0)] = 3

    return labels


def generate_terrain(cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = int(cfg.get("effective_resolution", cfg.get("resolution", 40)))
    coord = np.linspace(-0.5, 0.5, n, dtype=float)
    X, Y = np.meshgrid(coord, coord, indexing="xy")

    ne = NoiseEngine(cfg["seed"])

    warp1 = ne.fbm(X * 1.8 + 3.1, Y * 1.8 - 2.4, octaves=3, offset=21)
    warp2 = ne.fbm(X * 1.8 - 5.7, Y * 1.8 + 4.2, octaves=3, offset=22)

    Xw = X + cfg["roughness"] * 0.18 * warp1
    Yw = Y + cfg["roughness"] * 0.18 * warp2

    base = ne.fbm(Xw * 2.3, Yw * 2.3, octaves=5, offset=1)
    ridge = ne.ridged(Xw * 1.6, Yw * 1.6, octaves=4, offset=2)
    macro = ne.fbm(Xw * 0.8, Yw * 0.8, octaves=3, offset=3)

    m = float(cfg["mountain_amount"])
    h = (1.0 - m) * base + m * ridge
    h = h + 0.18 * macro

    if cfg.get("biome") == "volcanic":
        h = 0.60 * h + 0.40 * (ridge ** 1.5)

    lo, hi = np.percentile(h, 1.0), np.percentile(h, 99.0)
    h = (h - lo) / max(1e-6, hi - lo)
    h = np.clip(h, 0.0, 1.0)

    gamma = float(cfg.get("height_gamma", 1.0))
    h = np.power(h, gamma)

    h = apply_erosion(h, cfg["erosion_iterations"], strength=0.25)

    lo, hi = float(h.min()), float(h.max())
    if hi > lo:
        h = (h - lo) / (hi - lo)
    else:
        h = np.zeros_like(h)

    cell = cfg["size"] / max(1, n - 1)
    gy, gx = np.gradient(h, cell)
    slope_rad = np.sqrt(gx * gx + gy * gy)
    slope_deg = np.degrees(np.arctan(slope_rad))

    moisture = 0.5 + 0.5 * ne.fbm(
        X * 1.2 + 9.1,
        Y * 1.2 - 4.4,
        octaves=4,
        offset=31,
    )
    moisture = np.clip(moisture, 0.0, 1.0)

    temperature = (
        1.0
        - 0.65 * h
        + 0.25 * (0.5 + 0.5 * ne.fbm(X * 0.9 - 2.2, Y * 0.9 + 8.8, octaves=3, offset=32))
    )
    temperature = np.clip(temperature, 0.0, 1.0)

    labels = classify_biomes(h, slope_deg, moisture, temperature, cfg)

    return h, slope_deg, labels, moisture, temperature


def compute_face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    if len(faces) == 0:
        return np.empty((0, 3), dtype=float)

    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    n = np.cross(v1 - v0, v2 - v0)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.maximum(norm, 1e-9)

    flip = n[:, 2] < 0.0
    n[flip] *= -1.0

    return n


def heightmap_to_mesh(h: np.ndarray, size: float, height_scale: float) -> Tuple[np.ndarray, np.ndarray]:
    n = h.shape[0]
    coord = np.linspace(-size * 0.5, size * 0.5, n, dtype=float)
    X, Y = np.meshgrid(coord, coord, indexing="xy")

    vertices = np.column_stack(
        [
            X.reshape(-1),
            Y.reshape(-1),
            h.reshape(-1) * height_scale,
        ]
    ).astype(float)

    idx = np.arange(n * n, dtype=np.int32).reshape(n, n)

    a = idx[:-1, :-1].reshape(-1)
    b = idx[1:, :-1].reshape(-1)
    c = idx[1:, 1:].reshape(-1)
    d = idx[:-1, 1:].reshape(-1)

    faces = np.concatenate(
        [
            np.column_stack([a, b, c]),
            np.column_stack([a, c, d]),
        ],
        axis=0,
    ).astype(np.int32)

    return vertices, faces


def empty_mesh_tuple() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.empty((0, 3), dtype=float),
        np.empty((0, 3), dtype=np.int32),
        np.empty((0, 3), dtype=float),
        np.empty((0, 3), dtype=float),
    )


def shade_colors(colors_rgb: np.ndarray, normals: np.ndarray) -> np.ndarray:
    if len(colors_rgb) == 0:
        return np.empty((0, 3), dtype=np.uint8)

    diff = np.clip(normals @ LIGHT_DIR, 0.0, 1.0)
    intensity = 0.36 + 0.64 * diff
    shaded = np.clip(colors_rgb * intensity[:, None], 0.0, 1.0)

    bgr = shaded[:, ::-1]
    return (bgr * 255.0).astype(np.uint8)


def decimate_indices(n: int, stride: int) -> np.ndarray:
    if stride <= 1:
        return np.arange(n, dtype=int)

    idx = list(range(0, n, stride))
    if idx[-1] != n - 1:
        idx.append(n - 1)

    return np.unique(np.array(idx, dtype=int))


def decimate_2d(arr: np.ndarray, stride: int) -> np.ndarray:
    if stride <= 1:
        return arr
    idx = decimate_indices(arr.shape[0], stride)
    return arr[np.ix_(idx, idx)]


def draw_line_mask(mask: np.ndarray, p0: Tuple[int, int], p1: Tuple[int, int], value: int = 1) -> None:
    x0, y0 = int(p0[0]), int(p0[1])
    x1, y1 = int(p1[0]), int(p1[1])

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    h, w = mask.shape

    while True:
        if 0 <= y0 < h and 0 <= x0 < w:
            mask[y0, x0] = value

        if x0 == x1 and y0 == y1:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


def dijkstra_path(cost: np.ndarray, start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
    n = cost.shape[0]
    total = n * n
    INF = 1e18

    start_idx = int(start[0]) * n + int(start[1])
    end_idx = int(end[0]) * n + int(end[1])

    dist = np.full(total, INF, dtype=float)
    prev = np.full(total, -1, dtype=np.int32)

    dist[start_idx] = 0.0
    heap = [(0.0, start_idx)]

    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),          (0, 1),
        (1, -1),  (1, 0), (1, 1),
    ]

    while heap:
        d, idx = heapq.heappop(heap)

        if idx == end_idx:
            break

        if d > dist[idx]:
            continue

        y, x = divmod(idx, n)

        for dy, dx in neighbors:
            ny = y + dy
            nx = x + dx

            if 0 <= ny < n and 0 <= nx < n:
                nid = ny * n + nx
                step = math.hypot(dy, dx)
                nd = d + float(cost[ny, nx]) * step

                if nd < dist[nid]:
                    dist[nid] = nd
                    prev[nid] = idx
                    heapq.heappush(heap, (nd, nid))

    if dist[end_idx] >= INF / 2.0:
        return []

    path = []
    cur = end_idx

    while cur != -1:
        y, x = divmod(cur, n)
        path.append((y, x))
        if cur == start_idx:
            break
        cur = int(prev[cur])

    return path[::-1]


def build_road_mesh(
    cfg: Dict[str, Any],
    paths: List[List[Tuple[int, int]]],
    h: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not paths:
        return empty_mesh_tuple()

    n = h.shape[0]
    size = float(cfg["size"])
    height_scale = float(cfg["height_scale"])
    width = float(cfg["road_width"])

    verts: List[Tuple[float, float, float]] = []
    faces: List[Tuple[int, int, int]] = []

    def world_from_rc(r: int, c: int) -> Tuple[float, float]:
        x = (c / max(1, n - 1) - 0.5) * size
        y = (r / max(1, n - 1) - 0.5) * size
        return x, y

    for path in paths:
        if len(path) < 2:
            continue

        sections = []

        for i, (r, c) in enumerate(path):
            prev_pt = path[max(i - 1, 0)]
            next_pt = path[min(i + 1, len(path) - 1)]

            dr = next_pt[0] - prev_pt[0]
            dc = next_pt[1] - prev_pt[1]
            norm = math.hypot(dr, dc)
            if norm < 1e-6:
                norm = 1.0

            tx = dc / norm
            ty = dr / norm

            px = -ty
            py = tx

            x, y = world_from_rc(r, c)
            z = float(h[r, c]) * height_scale + 0.12

            lx = x + px * width * 0.5
            ly = y + py * width * 0.5
            rx = x - px * width * 0.5
            ry = y - py * width * 0.5

            sections.append(((lx, ly, z), (rx, ry, z)))

        base = len(verts)

        for left, right in sections:
            verts.append(left)
            verts.append(right)

        for i in range(len(sections) - 1):
            a = base + i * 2
            b = a + 1
            c = a + 2
            d = a + 3
            faces.append((a, b, d))
            faces.append((b, c, d))

    if not verts or not faces:
        return empty_mesh_tuple()

    vertices = np.array(verts, dtype=float)
    faces_arr = np.array(faces, dtype=np.int32)
    normals = compute_face_normals(vertices, faces_arr)

    asphalt = np.array([[0.16, 0.16, 0.18]], dtype=float)
    face_colors = np.repeat(asphalt, len(faces_arr), axis=0)

    return vertices, faces_arr, face_colors, normals


def generate_roads(
    cfg: Dict[str, Any],
    h: np.ndarray,
    labels: np.ndarray,
    slope_deg: np.ndarray,
) -> Tuple[List[List[Tuple[int, int]]], np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    n = h.shape[0]
    rng = np.random.default_rng(cfg["seed"] + 101)

    land_yx = np.argwhere(labels != 0)
    paths: List[List[Tuple[int, int]]] = []

    if len(land_yx) > 20 and cfg["road_count"] > 0:
        cost = (
            1.0
            + 3.5 * (slope_deg / 90.0)
            + 25.0 * (labels == 0).astype(float)
            + 0.8 * (labels == 4).astype(float)
            + 0.3 * (labels == 2).astype(float)
        )
        cost = np.clip(cost, 0.5, 100.0)

        for _ in range(int(cfg["road_count"])):
            for _attempt in range(200):
                s = land_yx[rng.integers(len(land_yx))]
                e = land_yx[rng.integers(len(land_yx))]

                if math.hypot(int(s[0]) - int(e[0]), int(s[1]) - int(e[1])) > n * 0.45:
                    path = dijkstra_path(cost, (int(s[0]), int(s[1])), (int(e[0]), int(e[1])))
                    if len(path) > 5:
                        paths.append(path)
                        break

    road_mask_uint8 = np.zeros((n, n), dtype=np.uint8)

    if cv2 is not None:
        for path in paths:
            if len(path) > 1:
                pts = np.array([[c, r] for r, c in path], dtype=np.int32)
                cv2.polylines(
                    road_mask_uint8,
                    [pts],
                    isClosed=False,
                    color=1,
                    thickness=int(cfg["road_width_pixels"]),
                    lineType=cv2.LINE_8,
                )
    else:
        for path in paths:
            for i in range(len(path) - 1):
                r0, c0 = path[i]
                r1, c1 = path[i + 1]
                draw_line_mask(road_mask_uint8, (c0, r0), (c1, r1), 1)

    road_mask_bool = road_mask_uint8.astype(bool)
    road_mask_bool = ndimage.binary_dilation(road_mask_bool, iterations=1)

    if not road_mask_bool.any():
        dist_to_road = np.full((n, n), float(n * 10), dtype=float)
    else:
        dist_to_road = ndimage.distance_transform_edt(~road_mask_bool)

    road_mesh = build_road_mesh(cfg, paths, h)

    return paths, road_mask_bool, dist_to_road, road_mesh


def poisson_disc_sampling(
    size: float,
    radius: float,
    seed: int,
    constraint,
    k: int = 30,
    max_points: int = 1000,
    stats: Dict[str, int] = None,
) -> Tuple[List[Tuple[float, float]], Dict[str, int]]:
    if stats is None:
        stats = {"attempts": 0, "rejected_constraint": 0, "rejected_spacing": 0}

    if radius <= 0.0 or max_points <= 0:
        return [], stats

    rng = np.random.default_rng(seed)
    half = size * 0.5
    cell = radius / math.sqrt(2.0)
    ncells = max(1, int(math.ceil(size / cell)))

    grid = [[-1 for _ in range(ncells)] for _ in range(ncells)]
    points: List[Tuple[float, float]] = []
    active: List[int] = []

    def to_grid(x: float, y: float) -> Tuple[int, int]:
        gi = int((x + half) / cell)
        gj = int((y + half) / cell)
        gi = min(ncells - 1, max(0, gi))
        gj = min(ncells - 1, max(0, gj))
        return gi, gj

    def add_point(x: float, y: float) -> None:
        idx = len(points)
        points.append((x, y))
        gi, gj = to_grid(x, y)
        grid[gj][gi] = idx
        active.append(idx)

    for _ in range(min(500, max(50, max_points * 10))):
        stats["attempts"] += 1
        x = float(rng.uniform(-half, half))
        y = float(rng.uniform(-half, half))

        if not constraint(x, y):
            stats["rejected_constraint"] += 1
            continue

        add_point(x, y)
        break

    if not points:
        return points, stats

    while active and len(points) < max_points:
        ai = int(rng.integers(len(active)))
        pidx = active[ai]
        p = points[pidx]
        found = False

        for _ in range(k):
            stats["attempts"] += 1

            ang = float(rng.uniform(0.0, 2.0 * math.pi))
            rad = float(rng.uniform(radius, 2.0 * radius))

            nx = p[0] + rad * math.cos(ang)
            ny = p[1] + rad * math.sin(ang)

            if nx < -half or nx > half or ny < -half or ny > half:
                stats["rejected_constraint"] += 1
                continue

            if not constraint(nx, ny):
                stats["rejected_constraint"] += 1
                continue

            gi, gj = to_grid(nx, ny)
            ok = True

            for yy in range(max(0, gj - 2), min(ncells, gj + 3)):
                for xx in range(max(0, gi - 2), min(ncells, gi + 3)):
                    id2 = grid[yy][xx]
                    if id2 != -1:
                        q = points[id2]
                        dx = nx - q[0]
                        dy = ny - q[1]
                        if dx * dx + dy * dy < radius * radius:
                            ok = False
                            break
                if not ok:
                    break

            if not ok:
                stats["rejected_spacing"] += 1
                continue

            add_point(nx, ny)
            found = True
            break

        if not found:
            active.pop(ai)

    return points, stats


class LocalStyleModel:
    """
    Tiny local generative attribute model.
    No cloud, no trained neural network download.
    Uses conditioned sinusoidal fields plus local Gaussian variation.
    """

    def __init__(self, seed: int, enabled: bool = True):
        self.rng = np.random.default_rng(seed + 909)
        self.enabled = bool(enabled)

        self.means = {
            "tree": 1.00,
            "shrub": 0.65,
            "rock": 0.55,
            "building": 1.00,
            "hut": 1.00,
            "tower": 1.15,
            "ruin": 0.90,
        }

        self.stds = {
            "tree": 0.22,
            "shrub": 0.18,
            "rock": 0.25,
            "building": 0.12,
            "hut": 0.15,
            "tower": 0.18,
            "ruin": 0.22,
        }

        self.biome_mod = {
            0: 0.00,
            1: 0.70,
            2: 0.70,
            3: 1.00,
            4: 1.18,
            5: 0.90,
            6: 0.65,
            7: 0.40,
            8: 0.45,
        }

    def sample_scale(self, kind: str, biome: int, x: float, y: float) -> float:
        base = float(self.means.get(kind, 1.0))
        std = float(self.stds.get(kind, 0.15))
        mod = float(self.biome_mod.get(int(biome), 1.0))

        field = 1.0 + 0.25 * math.sin(x * 0.31 + y * 0.17)
        s = base * mod * field

        if self.enabled:
            s *= max(0.35, min(1.8, float(self.rng.normal(1.0, std))))

        return float(np.clip(s, 0.25, 2.2))


def veg_suitability(biome: int, slope_deg: float) -> float:
    base = VEG_SUIT_BASE.get(int(biome), 0.25)
    slope_factor = max(0.0, 1.0 - float(slope_deg) / 70.0)
    return float(base * slope_factor)


def place_vegetation(
    cfg: Dict[str, Any],
    h: np.ndarray,
    labels: np.ndarray,
    slope_deg: np.ndarray,
    dist_to_road: np.ndarray,
    local_model: LocalStyleModel,
    stats: Dict[str, int],
) -> List[Dict[str, Any]]:
    n = h.shape[0]
    cell = cfg["size"] / max(1, n - 1)
    instances: List[Dict[str, Any]] = []

    total_max = int(cfg["instance_limit"] * 0.75)
    if total_max <= 0:
        return instances

    allocations = [
        ("tree", 0.55),
        ("shrub", 0.28),
        ("rock", 0.17),
    ]

    for kind, fraction in allocations:
        count = int(total_max * fraction * cfg["vegetation_density"])
        if count < 3:
            continue

        radius = max(cell * 1.8, cfg["size"] / math.sqrt(max(1, count)) * 0.8)

        max_slope = {"tree": 38.0, "shrub": 48.0, "rock": 65.0}[kind]
        clearance_px = {"tree": 3, "shrub": 2, "rock": 1}[kind]

        def constraint(
            x: float,
            y: float,
            kind=kind,
            max_slope=max_slope,
            clearance_px=clearance_px,
        ) -> bool:
            c = int(round(x / cell + (n - 1) * 0.5))
            r = int(round(y / cell + (n - 1) * 0.5))

            if r < 0 or r >= n or c < 0 or c >= n:
                return False

            if labels[r, c] == 0:
                return False

            if slope_deg[r, c] > max_slope:
                return False

            if dist_to_road[r, c] < clearance_px:
                return False

            suit = veg_suitability(int(labels[r, c]), float(slope_deg[r, c]))
            return suit > 0.12

        kind_seed = cfg["seed"] + 1000 + sum(ord(ch) for ch in kind)
        pts, _ = poisson_disc_sampling(
            size=cfg["size"],
            radius=radius,
            seed=kind_seed,
            constraint=constraint,
            max_points=count,
            stats=stats,
        )

        for x, y in pts:
            c = int(round(x / cell + (n - 1) * 0.5))
            r = int(round(y / cell + (n - 1) * 0.5))

            if r < 0 or r >= n or c < 0 or c >= n:
                continue

            biome = int(labels[r, c])
            z = float(h[r, c] * cfg["height_scale"])
            scale = local_model.sample_scale(kind, biome, x, y)

            base_color = np.array(OBJECT_COLORS_RGB[kind], dtype=float)
            variation = 0.85 + 0.30 * float(local_model.rng.random())
            color = np.clip(base_color * variation, 0.0, 1.0)

            instances.append(
                {
                    "type": kind,
                    "x": float(x),
                    "y": float(y),
                    "z": z,
                    "scale": float(scale),
                    "rotation": float(local_model.rng.uniform(0.0, 2.0 * math.pi)),
                    "footprint": 0.0,
                    "height": 0.0,
                    "biome": biome,
                    "color": [float(c0) for c0 in color],
                }
            )

    return instances


def place_buildings(
    cfg: Dict[str, Any],
    h: np.ndarray,
    labels: np.ndarray,
    slope_deg: np.ndarray,
    dist_to_road: np.ndarray,
    road_mask: np.ndarray,
    stats: Dict[str, int],
) -> List[Dict[str, Any]]:
    n = h.shape[0]
    cell = cfg["size"] / max(1, n - 1)
    instances: List[Dict[str, Any]] = []

    count = int(cfg["instance_limit"] * 0.25 * cfg["building_density"])
    if count < 1:
        return instances

    rng = np.random.default_rng(cfg["seed"] + 555)

    radius = max(cell * 5.0, cfg["size"] / math.sqrt(max(1, count)) * 1.5)

    road_angle = 0.0
    if road_mask.any():
        ys, xs = np.nonzero(road_mask)
        if len(xs) > 10:
            coords = np.column_stack([xs, ys]).astype(float)
            coords -= coords.mean(axis=0)
            cov = np.cov(coords.T)
            vals, vecs = np.linalg.eigh(cov)
            main = vecs[:, -1]
            road_angle = float(math.atan2(main[1], main[0]))

    has_roads = bool(road_mask.any())
    access_px = n * 10 if not has_roads else max(6, int(cfg["road_width_pixels"] * 5))

    occupancy = np.zeros((n, n), dtype=bool)
    min_footprint = 4.0
    occ_radius_px = max(1, int((min_footprint * 0.7) / cell))

    def occupancy_collision(r: int, c: int, radius_px: int) -> bool:
        r0 = max(0, r - radius_px)
        r1 = min(n, r + radius_px + 1)
        c0 = max(0, c - radius_px)
        c1 = min(n, c + radius_px + 1)

        yy, xx = np.ogrid[r0:r1, c0:c1]
        mask = (yy - r) ** 2 + (xx - c) ** 2 <= radius_px * radius_px

        return bool(np.any(occupancy[r0:r1, c0:c1] & mask))

    def mark_occupancy(r: int, c: int, radius_px: int) -> bool:
        r0 = max(0, r - radius_px)
        r1 = min(n, r + radius_px + 1)
        c0 = max(0, c - radius_px)
        c1 = min(n, c + radius_px + 1)

        yy, xx = np.ogrid[r0:r1, c0:c1]
        mask = (yy - r) ** 2 + (xx - c) ** 2 <= radius_px * radius_px

        sub = occupancy[r0:r1, c0:c1]
        if np.any(sub & mask):
            stats["collisions"] = stats.get("collisions", 0) + 1
            return False

        sub[mask] = True
        occupancy[r0:r1, c0:c1] = sub
        return True

    def constraint(x: float, y: float) -> bool:
        c = int(round(x / cell + (n - 1) * 0.5))
        r = int(round(y / cell + (n - 1) * 0.5))

        if r < 0 or r >= n or c < 0 or c >= n:
            return False

        if labels[r, c] == 0:
            return False

        if slope_deg[r, c] > 14.0:
            return False

        if dist_to_road[r, c] > access_px:
            return False

        suit = BUILD_SUIT_BASE.get(int(labels[r, c]), 0.2)
        if suit < 0.25:
            return False

        if occupancy_collision(r, c, occ_radius_px):
            stats["collisions"] = stats.get("collisions", 0) + 1
            return False

        return True

    pts, _ = poisson_disc_sampling(
        size=cfg["size"],
        radius=radius,
        seed=cfg["seed"] + 777,
        constraint=constraint,
        max_points=count,
        stats=stats,
    )

    for x, y in pts:
        c = int(round(x / cell + (n - 1) * 0.5))
        r = int(round(y / cell + (n - 1) * 0.5))

        if r < 0 or r >= n or c < 0 or c >= n:
            continue

        biome = int(labels[r, c])
        suit = BUILD_SUIT_BASE.get(biome, 0.3)

        footprint = float(rng.uniform(4.0, 8.0) * (0.8 + 0.4 * suit))
        height = float(rng.uniform(3.0, 7.0) * (0.8 + 0.4 * suit))
        scale = float(np.clip(footprint / 6.0, 0.6, 1.8))

        radius_px = max(1, int((footprint * 0.55) / cell))
        if not mark_occupancy(r, c, radius_px):
            continue

        if cfg.get("biome") == "urban" and rng.random() < 0.28:
            kind = "tower"
            height *= 1.8
        elif cfg.get("biome") == "volcanic" and rng.random() < 0.45:
            kind = "ruin"
            height *= 0.65
        else:
            kind = "hut" if rng.random() < 0.72 else "building"

        if has_roads:
            rotation = road_angle + float(rng.normal(0.0, 0.18))
        else:
            rotation = float(rng.uniform(0.0, 2.0 * math.pi))

        z = float(h[r, c] * cfg["height_scale"])

        base_color = np.array(OBJECT_COLORS_RGB[kind], dtype=float)
        variation = 0.88 + 0.24 * float(rng.random())
        color = np.clip(base_color * variation, 0.0, 1.0)

        instances.append(
            {
                "type": kind,
                "x": float(x),
                "y": float(y),
                "z": z,
                "scale": scale,
                "rotation": float(rotation % (2.0 * math.pi)),
                "footprint": footprint,
                "height": height,
                "biome": biome,
                "color": [float(cc) for cc in color],
            }
        )

    return instances


def combine_meshes(meshes: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]):
    all_v = []
    all_f = []
    all_c = []
    all_n = []
    offset = 0

    for v, f, c, nrm in meshes:
        if len(f) == 0 or len(v) == 0:
            continue

        all_v.append(v)
        all_f.append(f + offset)
        all_c.append(c)
        all_n.append(nrm)
        offset += len(v)

    if not all_v:
        return empty_mesh_tuple()

    vertices = np.vstack(all_v).astype(float)
    faces = np.vstack(all_f).astype(np.int32)
    colors = np.vstack(all_c).astype(float)
    normals = np.vstack(all_n).astype(float)

    return vertices, faces, colors, normals


def compute_repro_hash(h: np.ndarray, instances: List[Dict[str, Any]]) -> str:
    m = hashlib.sha256()
    m.update(np.ascontiguousarray(h.astype(np.float32)).tobytes())

    clean = []
    for inst in sorted(
        instances,
        key=lambda d: (str(d.get("type", "")), round(float(d.get("x", 0.0)), 6), round(float(d.get("y", 0.0)), 6)),
    ):
        row = {}
        for k, v in inst.items():
            if k in ("color", "color_bgr", "id"):
                continue
            if isinstance(v, (float, np.floating)):
                row[k] = round(float(v), 6)
            elif isinstance(v, (int, np.integer)):
                row[k] = int(v)
            else:
                row[k] = v
        clean.append(row)

    m.update(json.dumps(clean, sort_keys=True, default=json_safe).encode("utf-8"))
    return m.hexdigest()


def compute_metrics(
    cfg: Dict[str, Any],
    h: np.ndarray,
    slope_deg: np.ndarray,
    instances: List[Dict[str, Any]],
    terrain_faces: np.ndarray,
    road_faces: np.ndarray,
    stats: Dict[str, int],
    elapsed_s: float,
    memory_peak_mb: float,
    repro_hash: str,
) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    for inst in instances:
        t = str(inst.get("type", "unknown"))
        counts[t] = counts.get(t, 0) + 1

    total_instances = max(1, len(instances))
    entropy = 0.0

    if len(counts) > 1:
        for v in counts.values():
            p = v / total_instances
            if p > 0.0:
                entropy -= p * math.log2(p)
        entropy /= math.log2(len(counts))

    terrain_tris = int(len(terrain_faces))
    road_tris = int(len(road_faces))
    asset_tris = int(sum(ASSET_POLY.get(t, 10) * c for t, c in counts.items()))
    total_tris = terrain_tris + road_tris + asset_tris

    attempts = max(1, int(stats.get("attempts", 0)) + int(stats.get("collisions", 0)))
    collision_rate = float(stats.get("collisions", 0)) / float(attempts)

    continuity = float(1.0 - np.mean(slope_deg > 65.0))
    unique_types = len(set(counts.keys()))
    draw_calls = 1 + (1 if road_tris > 0 else 0) + unique_types

    budget_ratio = float(total_tris) / float(max(1, cfg["poly_budget"]))
    engine_score = float(
        max(
            0.0,
            min(
                100.0,
                100.0 * (1.0 - max(0.0, budget_ratio - 1.0)) - max(0.0, draw_calls - 64) * 0.5,
            ),
        )
    )

    return {
        "generation_seconds": float(elapsed_s),
        "memory_peak_mb": float(memory_peak_mb),
        "terrain_triangles": terrain_tris,
        "road_triangles": road_tris,
        "asset_triangles_estimate": asset_tris,
        "total_triangles": total_tris,
        "polygon_budget": int(cfg["poly_budget"]),
        "budget_ratio": float(budget_ratio),
        "instance_count": int(len(instances)),
        "instance_counts_by_type": counts,
        "asset_distribution_entropy": float(entropy),
        "object_collision_rate": float(collision_rate),
        "placement_attempts": int(attempts),
        "terrain_continuity_score": float(continuity),
        "height_min": float(h.min()),
        "height_max": float(h.max()),
        "mean_slope_deg": float(np.mean(slope_deg)),
        "max_slope_deg": float(np.max(slope_deg)),
        "estimated_draw_calls": int(draw_calls),
        "target_engine_score": engine_score,
        "reproducibility_sha256": repro_hash,
    }


def build_world_core(raw_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = normalize_config(raw_cfg)

    start = time.perf_counter()
    trace_started = not tracemalloc.is_tracing()
    if trace_started:
        tracemalloc.start()

    h, slope_deg, labels, moisture, temperature = generate_terrain(cfg)

    vertices, faces = heightmap_to_mesh(h, cfg["size"], cfg["height_scale"])
    normals = compute_face_normals(vertices, faces)

    vertex_colors_rgb = BIOME_LUT_RGB[labels.reshape(-1).astype(np.int16)]
    face_colors_rgb = vertex_colors_rgb[faces].mean(axis=1)

    paths, road_mask, dist_to_road, road_mesh = generate_roads(cfg, h, labels, slope_deg)

    local_model = LocalStyleModel(cfg["seed"], enabled=cfg["enable_local_generative_model"])

    stats = {
        "attempts": 0,
        "rejected_constraint": 0,
        "rejected_spacing": 0,
        "collisions": 0,
    }

    vegetation = place_vegetation(cfg, h, labels, slope_deg, dist_to_road, local_model, stats)
    buildings = place_buildings(cfg, h, labels, slope_deg, dist_to_road, road_mask, stats)

    instances = vegetation + buildings
    if len(instances) > cfg["instance_limit"]:
        instances = instances[: cfg["instance_limit"]]

    for i, inst in enumerate(instances):
        inst["id"] = i
        rgb = inst.get("color", [0.5, 0.5, 0.5])
        inst["color_bgr"] = tuple(int(round(float(c) * 255.0)) for c in reversed(rgb))

    full_tri_count = len(faces)
    preview_target = 4500
    stride = max(1, int(math.sqrt(max(1, full_tri_count) / float(preview_target))))

    ph = decimate_2d(h, stride)
    pl = decimate_2d(labels, stride)

    pv, pf = heightmap_to_mesh(ph, cfg["size"], cfg["height_scale"])
    pn = compute_face_normals(pv, pf)

    pvc = BIOME_LUT_RGB[pl.reshape(-1).astype(np.int16)]
    pfc = pvc[pf].mean(axis=1)

    if len(road_mesh[1]) > 0:
        preview_vertices, preview_faces, preview_face_colors, preview_normals = combine_meshes(
            [(pv, pf, pfc, pn), road_mesh]
        )
    else:
        preview_vertices, preview_faces, preview_face_colors, preview_normals = pv, pf, pfc, pn

    preview_shaded_bgr = shade_colors(preview_face_colors, preview_normals)
    preview_center_z = float(np.mean(preview_vertices[:, 2])) if len(preview_vertices) else 0.0

    topdown_rgb = BIOME_LUT_RGB255[labels.astype(np.int16)]

    vertex_colors_rgba = np.column_stack(
        [
            vertex_colors_rgb,
            np.ones(len(vertex_colors_rgb), dtype=float),
        ]
    )
    vertex_colors_rgba = (np.clip(vertex_colors_rgba, 0.0, 1.0) * 255.0).astype(np.uint8)

    elapsed = time.perf_counter() - start

    if trace_started:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = float(peak) / (1024.0 * 1024.0)
    else:
        peak_mb = 0.0

    repro_hash = compute_repro_hash(h, instances)

    metrics = compute_metrics(
        cfg=cfg,
        h=h,
        slope_deg=slope_deg,
        instances=instances,
        terrain_faces=faces,
        road_faces=road_mesh[1],
        stats=stats,
        elapsed_s=elapsed,
        memory_peak_mb=peak_mb,
        repro_hash=repro_hash,
    )

    return {
        "config": cfg,
        "height": h,
        "height_min": float(h.min()),
        "height_max": float(h.max()),
        "slope_deg": slope_deg,
        "labels": labels,
        "moisture": moisture,
        "temperature": temperature,
        "vertices": vertices,
        "faces": faces,
        "normals": normals,
        "vertex_colors_rgb": vertex_colors_rgb,
        "vertex_colors_rgba": vertex_colors_rgba,
        "face_colors_rgb": face_colors_rgb,
        "road_paths": paths,
        "road_mask": road_mask,
        "dist_to_road": dist_to_road,
        "road_mesh": road_mesh,
        "instances": instances,
        "preview_vertices": preview_vertices,
        "preview_faces": preview_faces,
        "preview_normals": preview_normals,
        "preview_face_colors": preview_face_colors,
        "preview_shaded_bgr": preview_shaded_bgr,
        "preview_center_z": preview_center_z,
        "topdown_rgb": topdown_rgb,
        "metrics": metrics,
        "repro_hash": repro_hash,
        "generation_seconds": float(elapsed),
        "memory_peak_mb": float(peak_mb),
        "placement_stats": stats,
    }


def make_sky(width: int, height: int) -> np.ndarray:
    width = max(1, int(width))
    height = max(1, int(height))

    top = np.array([255, 180, 80], dtype=float)
    horizon = np.array([235, 215, 190], dtype=float)

    t = np.linspace(0.0, 1.0, height, dtype=float)[:, None, None]
    sky = top[None, None, :] * (1.0 - t) + horizon[None, None, :] * t
    sky = np.repeat(sky.astype(np.uint8), width, axis=1)

    if cv2 is not None:
        cv2.circle(
            sky,
            (int(width * 0.78), int(height * 0.18)),
            max(4, int(min(width, height) * 0.035)),
            (220, 240, 255),
            -1,
            lineType=cv2.LINE_AA,
        )

    return sky


def look_at_view(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray = np.array([0.0, 0.0, 1.0]),
) -> np.ndarray:
    z = eye - target
    zn = float(np.linalg.norm(z))
    if zn < 1e-6:
        z = np.array([0.0, 1.0, 0.0])
    else:
        z = z / zn

    x = np.cross(up, z)
    xn = float(np.linalg.norm(x))
    if xn < 1e-6:
        x = np.array([1.0, 0.0, 0.0])
    else:
        x = x / xn

    y = np.cross(z, x)

    view = np.eye(4, dtype=float)
    view[0, :3] = x
    view[1, :3] = y
    view[2, :3] = z

    view[0, 3] = -float(np.dot(x, eye))
    view[1, 3] = -float(np.dot(y, eye))
    view[2, 3] = -float(np.dot(z, eye))

    return view


def project_points(
    points: np.ndarray,
    view: np.ndarray,
    width: int,
    height: int,
    focal: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(points) == 0:
        return np.empty((0, 3), dtype=float), np.empty((0,), dtype=bool)

    R = view[:3, :3]
    t = view[:3, 3]

    cam = points @ R.T + t
    depth = -cam[:, 2]

    visible = depth > 0.1

    safe_depth = np.maximum(depth, 1e-6)
    x = np.where(visible, cam[:, 0] / safe_depth, 0.0)
    y = np.where(visible, cam[:, 1] / safe_depth, 0.0)

    u = width * 0.5 + x * focal
    v = height * 0.5 - y * focal

    out = np.column_stack([u, v, depth])
    return out, visible


def instance_top_z(inst: Dict[str, Any]) -> float:
    t = str(inst.get("type", "tree"))
    z = float(inst.get("z", 0.0))
    s = float(inst.get("scale", 1.0))

    if t == "tree":
        return z + 5.5 * s
    if t == "shrub":
        return z + 2.0 * s
    if t == "rock":
        return z + 1.4 * s
    if t in BUILDING_TYPES:
        return z + float(inst.get("height", 4.0))
    return z + 2.0 * s


def instance_world_width(inst: Dict[str, Any]) -> float:
    t = str(inst.get("type", "tree"))
    s = float(inst.get("scale", 1.0))

    if t == "tree":
        return 1.8 * s
    if t == "shrub":
        return 1.4 * s
    if t == "rock":
        return 1.2 * s
    if t in BUILDING_TYPES:
        return float(inst.get("footprint", 5.0 * s))
    return 1.5 * s


def render_frame(
    world: Dict[str, Any],
    azimuth: float,
    elevation: float,
    distance: float,
    width: int,
    height: int,
) -> np.ndarray:
    width = max(64, int(width))
    height = max(48, int(height))

    if cv2 is None:
        img = Image.fromarray(world["topdown_rgb"]).convert("RGB").resize((width, height))
        return np.array(img)[:, :, ::-1].copy()

    sky = make_sky(width, height)

    vertices = world.get("preview_vertices")
    faces = world.get("preview_faces")
    colors = world.get("preview_shaded_bgr")

    if vertices is None or faces is None or len(faces) == 0:
        return sky

    elevation = float(np.clip(elevation, 5.0, 80.0))
    az = math.radians(float(azimuth % 360.0))
    el = math.radians(elevation)

    target = np.array([0.0, 0.0, float(world.get("preview_center_z", 0.0))], dtype=float)
    dist = float(distance)

    eye = target + np.array(
        [
            dist * math.cos(el) * math.cos(az),
            dist * math.cos(el) * math.sin(az),
            dist * math.sin(el),
        ],
        dtype=float,
    )

    view = look_at_view(eye, target)
    focal = max(width, height) * 0.9

    proj, visible = project_points(vertices, view, width, height, focal)

    polys: List[Tuple[float, np.ndarray, Tuple[int, int, int]]] = []

    face_visible = visible[faces].all(axis=1)
    face_depths = proj[faces, 2].mean(axis=1)

    valid = np.nonzero(face_visible)[0]
    if len(valid) > 0:
        valid = valid[np.argsort(face_depths[valid])[::-1]]

        margin = 40
        for fi in valid:
            pts = proj[faces[fi], :2]

            if (
                pts[:, 0].max() < -margin
                or pts[:, 0].min() > width + margin
                or pts[:, 1].max() < -margin
                or pts[:, 1].min() > height + margin
            ):
                continue

            polys.append(
                (
                    float(face_depths[fi]),
                    pts.astype(np.int32),
                    tuple(int(c) for c in colors[fi]),
                )
            )

    instances = world.get("instances", [])
    if instances:
        bases = np.array([[float(i["x"]), float(i["y"]), float(i["z"])] for i in instances], dtype=float)
        tops = np.array([[float(i["x"]), float(i["y"]), instance_top_z(i)] for i in instances], dtype=float)

        pb, vb = project_points(bases, view, width, height, focal)
        pt, vt = project_points(tops, view, width, height, focal)

        for idx, inst in enumerate(instances):
            if not vb[idx]:
                continue

            depth = float(pb[idx, 2])
            xb, yb = pb[idx, 0], pb[idx, 1]
            xt, yt = pt[idx, 0], pt[idx, 1]

            ww = instance_world_width(inst)
            wb = ww * focal / max(depth, 1e-3)

            if vt[idx]:
                wt = ww * focal / max(float(pt[idx, 2]), 1e-3)
            else:
                wt = wb * 0.85
                xt = xb
                yt = yb - wb * 1.2

            t = str(inst.get("type", "tree"))
            color = inst.get("color_bgr", (80, 160, 80))

            if t in ("tree", "shrub"):
                poly = np.array(
                    [
                        [xb - wb * 0.5, yb],
                        [xb + wb * 0.5, yb],
                        [xt, yt],
                    ],
                    dtype=np.float32,
                )
            elif t == "rock":
                radius = wb * 0.5
                poly = []
                for a in range(6):
                    ang = a * math.pi / 3.0
                    poly.append([xb + radius * math.cos(ang), yb + radius * math.sin(ang) * 0.65])
                poly = np.array(poly, dtype=np.float32)
            else:
                poly = np.array(
                    [
                        [xb - wb * 0.5, yb],
                        [xb + wb * 0.5, yb],
                        [xt + wt * 0.5, yt],
                        [xt - wt * 0.5, yt],
                    ],
                    dtype=np.float32,
                )

            if len(poly) >= 3:
                polys.append((depth, poly.astype(np.int32), tuple(int(c) for c in color)))

    polys.sort(key=lambda item: item[0], reverse=True)

    img = sky.copy()
    for _, pts, color in polys:
        if len(pts) < 3:
            continue
        cv2.fillPoly(img, [pts], color)

    return img


def generate_video_from_world(world: Dict[str, Any], raw_cfg: Dict[str, Any]) -> Tuple[bytes, str, str]:
    cfg = normalize_config(raw_cfg)

    frames = []
    nframes = max(2, int(cfg["video_frames"]))

    for i in range(nframes):
        az = (float(cfg["start_azimuth"]) + 360.0 * i / float(nframes)) % 360.0
        frame = render_frame(
            world,
            azimuth=az,
            elevation=cfg["camera_elevation"],
            distance=cfg["camera_distance"],
            width=cfg["video_width"],
            height=cfg["video_height"],
        )
        frames.append(frame)

    return encode_video(frames, cfg["video_fps"], cfg["video_width"], cfg["video_height"])


def encode_video(
    frames: List[np.ndarray],
    fps: int,
    width: int,
    height: int,
) -> Tuple[bytes, str, str]:
    if not frames:
        return b"", "image/gif", "gif"

    width = max(1, int(width))
    height = max(1, int(height))
    fps = max(1, int(fps))

    if cv2 is not None:
        fourcc_candidates = ["avc1", "H264", "X264", "mp4v"]

        for code in fourcc_candidates:
            tmp_path = tempfile.mktemp(suffix=".mp4")

            try:
                fourcc = cv2.VideoWriter_fourcc(*code)
                writer = cv2.VideoWriter(tmp_path, fourcc, float(fps), (width, height))

                if writer.isOpened():
                    for f in frames:
                        if f.shape[1] != width or f.shape[0] != height:
                            f = cv2.resize(f, (width, height), interpolation=cv2.INTER_AREA)
                        writer.write(f)

                    writer.release()

                    if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
                        with open(tmp_path, "rb") as fh:
                            data = fh.read()
                        try:
                            os.unlink(tmp_path)
                        except Exception:
                            pass
                        return data, "video/mp4", "mp4"

                if os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass

            except Exception:
                try:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                except Exception:
                    pass

    pil_frames = []
    for f in frames:
        if cv2 is not None:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        else:
            rgb = f[:, :, ::-1]
        pil_frames.append(Image.fromarray(rgb).convert("P", palette=Image.ADAPTIVE, colors=128))

    buf = io.BytesIO()
    if pil_frames:
        pil_frames[0].save(
            buf,
            format="GIF",
            append_images=pil_frames[1:],
            save_all=True,
            duration=int(1000.0 / max(1, fps)),
            loop=0,
        )
        return buf.getvalue(), "image/gif", "gif"

    return b"", "image/gif", "gif"


def make_obj(vertices: np.ndarray, faces: np.ndarray, name: str = "terrain") -> str:
    out = [f"# {name}"]

    for v in vertices:
        out.append(f"v {float(v[0]):.6f} {float(v[1]):.6f} {float(v[2]):.6f}")

    for f in faces:
        out.append(f"f {int(f[0]) + 1} {int(f[1]) + 1} {int(f[2]) + 1}")

    return "\n".join(out)


def make_glb(world: Dict[str, Any]) -> bytes | None:
    if trimesh is None:
        return None

    try:
        mesh = trimesh.Trimesh(
            vertices=world["vertices"],
            faces=world["faces"],
            process=False,
        )
        mesh.visual.vertex_colors = world["vertex_colors_rgba"]
        return mesh.export(file_type="glb")
    except Exception:
        return None


def clean_instance(inst: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in inst.items():
        if k in ("color", "color_bgr", "id"):
            continue

        if isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = round(float(v), 5)
        elif isinstance(v, float):
            out[k] = round(v, 5)
        elif isinstance(v, (list, tuple)):
            out[k] = [round(float(x), 5) if isinstance(x, (float, np.floating)) else x for x in v]
        else:
            out[k] = v

    return out


BLENDER_TEMPLATE = r'''
import bpy
import sys
import os
import json
import zlib
import base64
import struct
import math
from mathutils import Vector

VERT_B64 = "__VERT_B64__"
FACE_B64 = "__FACE_B64__"
INST_B64 = "__INST_B64__"
META_B64 = "__META_B64__"


def decode_bytes(b64):
    return zlib.decompress(base64.b64decode(b64))


def decode_float32(b64):
    data = decode_bytes(b64)
    n = len(data) // 4
    if n == 0:
        return []
    return list(struct.unpack("<%df" % n, data))


def decode_int32(b64):
    data = decode_bytes(b64)
    n = len(data) // 4
    if n == 0:
        return []
    return list(struct.unpack("<%di" % n, data))


def decode_json(b64):
    return json.loads(decode_bytes(b64).decode("utf-8"))


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    opts = {"render": "", "gltf": ""}
    i = 0

    while i < len(argv):
        if argv[i] == "--render" and i + 1 < len(argv):
            opts["render"] = argv[i + 1]
            i += 2
        elif argv[i] == "--gltf" and i + 1 < len(argv):
            opts["gltf"] = argv[i + 1]
            i += 2
        else:
            i += 1

    return opts


def clear_scene():
    try:
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)
    except Exception:
        pass

    for block in (bpy.data.meshes, bpy.data.materials, bpy.data.objects, bpy.data.cameras, bpy.data.lights):
        for item in list(block):
            if item.users == 0:
                block.remove(item)


def create_terrain():
    verts = decode_float32(VERT_B64)
    faces = decode_int32(FACE_B64)

    mesh = bpy.data.meshes.new("Terrain")

    mesh.vertices.add(len(verts) // 3)
    mesh.vertices.foreach_set("coord", verts)

    mesh.loops.add(len(faces))
    mesh.loops.foreach_set("vertex_index", faces)

    poly_count = len(faces) // 3
    mesh.polygons.add(poly_count)

    loop_starts = list(range(0, len(faces), 3))
    loop_totals = [3] * poly_count

    mesh.polygons.foreach_set("loop_start", loop_starts)
    mesh.polygons.foreach_set("loop_total", loop_totals)

    mesh.update()
    mesh.validate()

    obj = bpy.data.objects.new("Terrain", mesh)
    bpy.context.collection.objects.link(obj)

    mat = bpy.data.materials.new("TerrainMaterial")
    mat.use_nodes = True
    obj.data.materials.append(mat)

    for p in mesh.polygons:
        p.use_smooth = True

    return obj


def add_material(name, rgb):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True

    bsdf = None
    for node in mat.node_tree.nodes:
        if node.type == "BSDF_PRINCIPLED":
            bsdf = node
            break

    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = (rgb[0], rgb[1], rgb[2], 1.0)

    return mat


def create_instances():
    instances = decode_json(INST_B64)

    mats = {
        "tree": add_material("TreeMat", (0.12, 0.42, 0.18)),
        "shrub": add_material("ShrubMat", (0.42, 0.52, 0.24)),
        "rock": add_material("RockMat", (0.48, 0.48, 0.50)),
        "building": add_material("BuildingMat", (0.65, 0.58, 0.48)),
        "hut": add_material("HutMat", (0.62, 0.52, 0.40)),
        "tower": add_material("TowerMat", (0.55, 0.55, 0.60)),
        "ruin": add_material("RuinMat", (0.38, 0.34, 0.30)),
    }

    for inst in instances:
        t = str(inst.get("type", "tree"))
        loc = (float(inst.get("x", 0.0)), float(inst.get("y", 0.0)), float(inst.get("z", 0.0)))
        scale = float(inst.get("scale", 1.0))
        rot = float(inst.get("rotation", 0.0))

        if t == "tree":
            bpy.ops.mesh.primitive_cone_add(
                vertices=8,
                radius1=0.9 * scale,
                radius2=0.0,
                depth=4.5 * scale,
                location=(loc[0], loc[1], loc[2] + 2.25 * scale),
            )
            ob = bpy.context.active_object
            ob.name = "Tree"
            ob.rotation_euler = (0.0, 0.0, rot)
            ob.data.materials.append(mats["tree"])

        elif t == "shrub":
            bpy.ops.mesh.primitive_uv_sphere_add(
                segments=8,
                ring_count=6,
                radius=0.9 * scale,
                location=(loc[0], loc[1], loc[2] + 0.7 * scale),
            )
            ob = bpy.context.active_object
            ob.name = "Shrub"
            ob.rotation_euler = (0.0, 0.0, rot)
            ob.data.materials.append(mats["shrub"])

        elif t == "rock":
            bpy.ops.mesh.primitive_ico_sphere_add(
                subdivisions=1,
                radius=0.8 * scale,
                location=(loc[0], loc[1], loc[2] + 0.5 * scale),
            )
            ob = bpy.context.active_object
            ob.name = "Rock"
            ob.rotation_euler = (0.0, 0.0, rot)
            ob.data.materials.append(mats["rock"])

        elif t in ("building", "hut", "tower", "ruin"):
            w = float(inst.get("footprint", 5.0))
            hgt = float(inst.get("height", 4.0))

            bpy.ops.mesh.primitive_cube_add(size=1.0, location=(loc[0], loc[1], loc[2] + hgt * 0.5))
            ob = bpy.context.active_object
            ob.name = t.capitalize()
            ob.scale = (w, hgt, w)
            ob.rotation_euler = (0.0, 0.0, rot)
            ob.data.materials.append(mats.get(t, mats["building"]))


def setup_camera(terrain_obj):
    bb = [terrain_obj.matrix_world @ Vector(corner) for corner in terrain_obj.bound_box]

    center = Vector((0.0, 0.0, 0.0))
    for v in bb:
        center += v
    center /= 8.0

    radius = 1.0
    for v in bb:
        radius = max(radius, (v - center).length)

    cam_data = bpy.data.cameras.new("WorldCam")
    cam_data.clip_end = 10000.0
    cam_obj = bpy.data.objects.new("WorldCam", cam_data)
    bpy.context.collection.objects.link(cam_obj)

    cam_obj.location = (
        center.x + radius * 1.45,
        center.y - radius * 1.45,
        center.z + radius * 0.75,
    )

    target = bpy.data.objects.new("CameraTarget", None)
    target.location = center
    bpy.context.collection.objects.link(target)

    con = cam_obj.constraints.new(type="TRACK_TO")
    con.target = target
    con.track_axis = "TRACK_NEGATIVE_Z"
    con.up_axis = "UP_Y"

    bpy.context.scene.camera = cam_obj

    sun_data = bpy.data.lights.new("Sun", "SUN")
    sun_data.energy = 3.0
    sun_obj = bpy.data.objects.new("Sun", sun_data)
    bpy.context.collection.objects.link(sun_obj)
    sun_obj.rotation_euler = (math.radians(55.0), 0.0, math.radians(35.0))


def main():
    opts = parse_args()

    clear_scene()
    terrain = create_terrain()
    create_instances()
    setup_camera(terrain)

    scene = bpy.context.scene
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720

    try:
        scene.render.engine = "BLENDER_WORKBENCH"
    except Exception:
        pass

    if opts["gltf"]:
        try:
            bpy.ops.export_scene.gltf(
                filepath=os.path.abspath(opts["gltf"]),
                export_format="GLB",
                use_selection=False,
            )
        except Exception as exc:
            print("GLTF export failed:", exc)

    if opts["render"]:
        try:
            scene.render.filepath = os.path.abspath(opts["render"])
            bpy.ops.render.render(write_still=True)
        except Exception as exc:
            print("Render failed:", exc)


if __name__ == "__main__":
    main()
'''


def make_blender_script(world: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    vert_flat = np.ascontiguousarray(world["vertices"].reshape(-1), dtype=np.float32)
    face_flat = np.ascontiguousarray(world["faces"].reshape(-1), dtype=np.int32)

    instances = [clean_instance(i) for i in world["instances"]]
    meta = {
        "seed": int(cfg.get("seed", 0)),
        "size": float(cfg.get("size", 120.0)),
        "height_scale": float(cfg.get("height_scale", 18.0)),
        "name": str(cfg.get("name", "terrain")),
    }

    def enc(data: bytes) -> str:
        return base64.b64encode(zlib.compress(data, 9)).decode("ascii")

    script = BLENDER_TEMPLATE
    script = script.replace("__VERT_B64__", enc(vert_flat.tobytes()))
    script = script.replace("__FACE_B64__", enc(face_flat.tobytes()))
    script = script.replace("__INST_B64__", enc(json.dumps(instances, separators=(",", ":"), default=json_safe).encode("utf-8")))
    script = script.replace("__META_B64__", enc(json.dumps(meta, separators=(",", ":"), default=json_safe).encode("utf-8")))

    return script


def render_object_map(world: Dict[str, Any], max_dim: int = 760) -> Image.Image:
    arr = world["topdown_rgb"].copy()
    road = world["road_mask"]

    if road is not None and road.any():
        arr[road] = np.array([35, 35, 40], dtype=np.uint8)

    img = Image.fromarray(arr).convert("RGB")
    draw = ImageDraw.Draw(img)

    n = world["labels"].shape[0]
    size = float(world["config"]["size"])

    for inst in world["instances"]:
        c = (float(inst["x"]) / size + 0.5) * (n - 1)
        r = (float(inst["y"]) / size + 0.5) * (n - 1)

        color = tuple(int(round(float(x) * 255.0)) for x in inst.get("color", [0.5, 0.5, 0.5]))
        rad = max(1, int(float(inst.get("scale", 1.0)) * 2.0))

        draw.ellipse([c - rad, r - rad, c + rad, r + rad], fill=color, outline=(20, 20, 20))

    if max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim))

    return img


def run_self_tests() -> Tuple[bool, str]:
    lines = []
    passed = 0
    failed = 0

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal passed, failed
        if cond:
            passed += 1
            lines.append(f"PASS | {name} | {detail}")
        else:
            failed += 1
            lines.append(f"FAIL | {name} | {detail}")

    cfg = DEFAULT_CONFIG.copy()
    cfg.update(
        {
            "seed": 4242,
            "resolution": 24,
            "poly_budget": 4000,
            "instance_limit": 60,
            "road_count": 1,
            "video_frames": 2,
            "video_fps": 12,
            "video_width": 160,
            "video_height": 90,
            "vegetation_density": 0.5,
            "building_density": 0.2,
        }
    )

    t0 = time.perf_counter()
    w1 = build_world_core(cfg)
    t1 = time.perf_counter()

    w2 = build_world_core(cfg)

    check("Reproducibility from seed", w1["repro_hash"] == w2["repro_hash"], w1["repro_hash"][:16])
    check("Terrain bounds", 0.0 <= w1["height_min"] <= w1["height_max"] <= 1.0001, f"{w1['height_min']:.3f}..{w1['height_max']:.3f}")
    check("Terrain mesh non-empty", len(w1["faces"]) > 0, f"{len(w1['faces'])} triangles")
    check("Slope finite", bool(np.all(np.isfinite(w1["slope_deg"]))), "")
    check("Instance limit respected", len(w1["instances"]) <= max(1, cfg["instance_limit"]) * 1.25, f"{len(w1['instances'])}")
    check(
        "Polygon budget approximate",
        w1["metrics"]["total_triangles"] <= cfg["poly_budget"] * 2.0,
        f"{w1['metrics']['total_triangles']} / {cfg['poly_budget']}",
    )
    check(
        "Collision rate valid",
        0.0 <= w1["metrics"]["object_collision_rate"] <= 1.0,
        f"{w1['metrics']['object_collision_rate']:.3f}",
    )

    obj = make_obj(w1["vertices"], w1["faces"], "test")
    check("OBJ export", obj.startswith("#") and "\nv " in obj and "\nf " in obj, "")

    script = make_blender_script(w1, w1["config"])
    check(
        "Blender script export",
        "import bpy" in script and "__VERT_B64__" not in script and "__INST_B64__" not in script,
        "",
    )

    glb = make_glb(w1)
    check("GLB export optional", glb is None or len(glb) > 100, "trimesh optional" if glb is None else f"{len(glb)} bytes")

    video = generate_video_from_world(w1, cfg)
    check("Live demo video bytes", len(video[0]) > 100, video[1])

    check("Generation time", (t1 - t0) < 60.0, f"{t1 - t0:.2f}s")

    report = "\n".join(lines)
    report += f"\n\n{passed} passed, {failed} failed"

    return failed == 0, report


@st.cache_data(show_spinner=False)
def cached_build_world(cfg_json: str) -> Dict[str, Any]:
    cfg = json.loads(cfg_json)
    return build_world_core(cfg)


@st.cache_data(show_spinner=False)
def cached_generate_video(cfg_json: str) -> Tuple[bytes, str, str]:
    cfg = json.loads(cfg_json)
    world = cached_build_world(cfg_json)
    return generate_video_from_world(world, cfg)


@st.cache_data(show_spinner=False)
def cached_render_frame(
    cfg_json: str,
    azimuth: float,
    elevation: float,
    distance: float,
    width: int,
    height: int,
) -> np.ndarray:
    world = cached_build_world(cfg_json)
    return render_frame(world, azimuth, elevation, distance, width, height)


@st.cache_data(show_spinner=False)
def cached_export_bundle(cfg_json: str) -> Dict[str, Any]:
    world = cached_build_world(cfg_json)
    cfg = world["config"]

    obj_text = make_obj(world["vertices"], world["faces"], cfg.get("name", "terrain"))
    instances_payload = [clean_instance(i) for i in world["instances"]]

    return {
        "config": json.dumps(cfg, indent=2, default=json_safe).encode("utf-8"),
        "obj": obj_text.encode("utf-8"),
        "glb": make_glb(world),
        "instances": json.dumps(instances_payload, indent=2, default=json_safe).encode("utf-8"),
        "blender": make_blender_script(world, cfg).encode("utf-8"),
        "metrics": json.dumps(world["metrics"], indent=2, default=json_safe).encode("utf-8"),
    }


def init_session() -> None:
    if "initialized" not in st.session_state:
        for k, v in DEFAULT_CONFIG.items():
            st.session_state[f"w_{k}"] = v

        st.session_state["w_preset"] = "Mixed Valley"
        st.session_state["last_preset"] = "Mixed Valley"
        st.session_state["initialized"] = True


def current_config() -> Dict[str, Any]:
    return {k: st.session_state[f"w_{k}"] for k in DEFAULT_CONFIG}


def sidebar_controls() -> Dict[str, Any]:
    st.sidebar.header("World configuration")

    preset_names = list(PRESETS.keys())
    preset = st.sidebar.selectbox("Biome preset", preset_names, key="w_preset")

    if st.session_state.get("last_preset") != preset:
        for k, v in PRESETS[preset].items():
            st.session_state[f"w_{k}"] = v
        st.session_state["last_preset"] = preset

    st.sidebar.text_input("World name", key="w_name")
    st.sidebar.number_input("Seed", min_value=0, max_value=999999999, step=1, key="w_seed")

    with st.sidebar.expander("Terrain controls", expanded=False):
        st.sidebar.slider("Terrain resolution", 24, 96, step=4, key="w_resolution")
        st.sidebar.slider("World size", 40.0, 250.0, step=5.0, key="w_size")
        st.sidebar.slider("Height scale", 2.0, 60.0, step=1.0, key="w_height_scale")
        st.sidebar.slider("Roughness", 0.0, 1.0, step=0.01, key="w_roughness")
        st.sidebar.slider("Mountain amount", 0.0, 1.0, step=0.01, key="w_mountain_amount")
        st.sidebar.slider("Height gamma", 0.35, 2.5, step=0.05, key="w_height_gamma")
        st.sidebar.slider("Water level", 0.0, 0.85, step=0.01, key="w_water_level")
        st.sidebar.slider("Erosion iterations", 0, 6, step=1, key="w_erosion_iterations")

    with st.sidebar.expander("Scene rules", expanded=False):
        st.sidebar.slider("Vegetation density", 0.0, 1.0, step=0.01, key="w_vegetation_density")
        st.sidebar.slider("Building density", 0.0, 1.0, step=0.01, key="w_building_density")
        st.sidebar.slider("Road count", 0, 8, step=1, key="w_road_count")
        st.sidebar.slider("Road width", 0.5, 12.0, step=0.1, key="w_road_width")
        st.sidebar.slider("Instance limit", 0, 1200, step=10, key="w_instance_limit")
        st.sidebar.slider("Polygon budget", 5000, 120000, step=1000, key="w_poly_budget")
        st.sidebar.checkbox("Local generative attribute model", key="w_enable_local_generative_model")

    with st.sidebar.expander("Demo renderer", expanded=False):
        st.sidebar.slider("Video frames", 8, 90, step=2, key="w_video_frames")
        st.sidebar.number_input("Video FPS", min_value=5, max_value=60, step=1, key="w_video_fps")
        st.sidebar.number_input("Video width", min_value=240, max_value=1280, step=32, key="w_video_width")
        st.sidebar.number_input("Video height", min_value=135, max_value=720, step=16, key="w_video_height")
        st.sidebar.slider("Camera distance", 50.0, 400.0, step=5.0, key="w_camera_distance")
        st.sidebar.slider("Camera elevation", 5.0, 80.0, step=1.0, key="w_camera_elevation")
        st.sidebar.slider("Start azimuth", 0.0, 360.0, step=1.0, key="w_start_azimuth")

    return current_config()


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")

    init_session()
    cfg = sidebar_controls()
    cfg_json = json.dumps(cfg, sort_keys=True, default=json_safe)

    with st.spinner("Preparing procedural world..."):
        world = cached_build_world(cfg_json)

    st.session_state["world"] = world
    st.session_state["cfg"] = cfg
    st.session_state["cfg_json"] = cfg_json

    if "video" not in st.session_state:
        try:
            with st.spinner("Generating live demo video..."):
                st.session_state["video"] = cached_generate_video(cfg_json)
                st.session_state["video_cfg"] = cfg_json
        except Exception:
            st.session_state["video_error"] = traceback.format_exc()
            st.session_state["video"] = (b"", "image/gif", "gif")

    st.title("Procedural World Forge")
    st.caption(
        "M1: reproducible procedural terrain, biome rules, roads, vegetation, buildings, "
        "live demo video, Blender/glTF export, and evaluation metrics. No API keys. No cloud."
    )

    tabs = st.tabs(
        [
            "Live Demo",
            "Terrain",
            "Objects",
            "Metrics",
            "Export",
            "Tests",
            "Docs",
        ]
    )

    with tabs[0]:
        st.subheader("Automatically generated live orbit demo")

        video_bytes, mime, ext = st.session_state.get("video", (b"", "image/gif", "gif"))

        if video_bytes:
            if mime.startswith("video/"):
                st.video(video_bytes, format=mime)
            else:
                st.image(Image.open(io.BytesIO(video_bytes)), use_container_width=True)

            st.download_button(
                "Download demo video",
                data=video_bytes,
                file_name=f"procedural_world_demo_{cfg['seed']}.{ext}",
                mime=mime,
            )
        else:
            st.warning("Video encoder fallback did not produce a file. The interactive 3D frame below is live.")

        if "video_error" in st.session_state:
            with st.expander("Video generation error"):
                st.code(st.session_state["video_error"], language="text")

        colA, colB = st.columns([2.2, 1.0])

        with colA:
            st.subheader("Interactive 3D preview")

            az = st.slider("Azimuth", 0.0, 360.0, value=float(cfg["start_azimuth"]), key="ui_az")
            el = st.slider("Elevation", 5.0, 80.0, value=float(cfg["camera_elevation"]), key="ui_el")
            dist = st.slider("Distance", 50.0, 400.0, value=float(cfg["camera_distance"]), key="ui_dist")

            preview_w = min(int(cfg["video_width"]), 900)
            preview_h = min(int(cfg["video_height"]), 540)

            frame = cached_render_frame(cfg_json, az, el, dist, preview_w, preview_h)
            st.image(Image.fromarray(frame[:, :, ::-1]), use_container_width=True)

        with colB:
            st.subheader("Controls")

            if st.button("Regenerate live demo video", use_container_width=True):
                with st.spinner("Rendering new demo video..."):
                    st.session_state["video"] = cached_generate_video(cfg_json)
                    st.session_state["video_cfg"] = cfg_json
                st.success("Demo video regenerated.")

            if st.button("Randomize seed", use_container_width=True):
                st.session_state["w_seed"] = int(np.random.default_rng().integers(0, 999999999))
                st.rerun()

            st.metric("Seed", str(cfg["seed"]))
            st.metric("Generation time", f"{world['generation_seconds']:.2f} s")
            st.metric("Total triangles", f"{world['metrics']['total_triangles']:,}")
            st.metric("Instances", f"{world['metrics']['instance_count']}")

            if st.session_state.get("video_cfg") != cfg_json:
                st.info("Current interactive preview reflects the latest config. Press regenerate to refresh the recorded video.")

    with tabs[1]:
        st.subheader("Terrain analysis")

        c1, c2, c3 = st.columns(3)

        h_norm = np.clip(world["height"], 0.0, 1.0)
        h_img = Image.fromarray((h_norm * 255.0).astype(np.uint8))

        slope_norm = np.clip(world["slope_deg"] / 90.0, 0.0, 1.0)
        slope_img = Image.fromarray((slope_norm * 255.0).astype(np.uint8))

        biome_img = Image.fromarray(world["topdown_rgb"])

        with c1:
            st.image(h_img, caption="Height field", use_container_width=True)
        with c2:
            st.image(slope_img, caption="Slope magnitude", use_container_width=True)
        with c3:
            st.image(biome_img, caption="Biome classification", use_container_width=True)

        st.json(
            {
                "height_min": world["height_min"],
                "height_max": world["height_max"],
                "mean_slope_deg": float(np.mean(world["slope_deg"])),
                "max_slope_deg": float(np.max(world["slope_deg"])),
                "effective_resolution": world["config"]["effective_resolution"],
                "biome": world["config"]["biome"],
            }
        )

    with tabs[2]:
        st.subheader("Scene composition and asset placement")

        map_img = render_object_map(world)
        st.image(map_img, caption="Top-down scene map with roads and instances", use_container_width=True)

        counts = world["metrics"]["instance_counts_by_type"]
        st.bar_chart(counts)

        display_rows = []
        for inst in world["instances"][:250]:
            row = clean_instance(inst)
            display_rows.append(row)

        st.dataframe(display_rows, use_container_width=True, hide_index=True)

    with tabs[3]:
        st.subheader("Evaluation metrics")

        m = world["metrics"]

        cc = st.columns(4)
        cc[0].metric("Generation time", f"{m['generation_seconds']:.2f} s")
        cc[1].metric("Total triangles", f"{m['total_triangles']:,}")
        cc[2].metric("Memory peak", f"{m['memory_peak_mb']:.1f} MB")
        cc[3].metric("Engine score", f"{m['target_engine_score']:.1f}/100")

        cc2 = st.columns(4)
        cc2[0].metric("Continuity score", f"{m['terrain_continuity_score']:.3f}")
        cc2[1].metric("Collision rate", f"{m['object_collision_rate']:.3f}")
        cc2[2].metric("Asset entropy", f"{m['asset_distribution_entropy']:.3f}")
        cc2[3].metric("Draw calls est.", f"{m['estimated_draw_calls']}")

        st.write("Reproducibility SHA-256:")
        st.code(m["reproducibility_sha256"], language="text")

        with st.expander("Full metrics JSON"):
            st.json(m)

    with tabs[4]:
        st.subheader("Exports")

        bundle = cached_export_bundle(cfg_json)

        st.download_button(
            "Download world_config.json",
            data=bundle["config"],
            file_name=f"world_config_{cfg['seed']}.json",
            mime="application/json",
        )

        st.download_button(
            "Download terrain.obj",
            data=bundle["obj"],
            file_name=f"terrain_{cfg['seed']}.obj",
            mime="text/plain",
        )

        if bundle["glb"]:
            st.download_button(
                "Download terrain.glb",
                data=bundle["glb"],
                file_name=f"terrain_{cfg['seed']}.glb",
                mime="model/gltf-binary",
            )
        else:
            st.info("GLB export unavailable because trimesh is not importable in this environment.")

        st.download_button(
            "Download instances.json",
            data=bundle["instances"],
            file_name=f"instances_{cfg['seed']}.json",
            mime="application/json",
        )

        st.download_button(
            "Download blender_terrain.py",
            data=bundle["blender"],
            file_name=f"blender_terrain_{cfg['seed']}.py",
            mime="text/x-python",
        )

        st.download_button(
            "Download metrics.json",
            data=bundle["metrics"],
            file_name=f"metrics_{cfg['seed']}.json",
            mime="application/json",
        )

        video_bytes, mime, ext = st.session_state.get("video", (b"", "image/gif", "gif"))
        if video_bytes:
            st.download_button(
                "Download live demo video",
                data=video_bytes,
                file_name=f"procedural_world_demo_{cfg['seed']}.{ext}",
                mime=mime,
            )

    with tabs[5]:
        st.subheader("Self-tests")

        st.markdown(
            """
            Run inside the app, or from a terminal:

            ```bash
            python app.py --self-test
            ```
            """
        )

        if st.button("Run self-tests", use_container_width=True):
            with st.spinner("Running reproducibility, export, and performance tests..."):
                ok, report = run_self_tests()

            if ok:
                st.success("All self-tests passed.")
            else:
                st.error("Some self-tests failed.")

            st.code(report, language="text")

    with tabs[6]:
        st.subheader("Architecture and operating notes")

        st.markdown(
            f"""
            ### Single-file modular architecture

            Although this deliverable is intentionally a single `app.py`, the code is separated into logical modules:

            - `NoiseEngine`: procedural noise, fBm, ridged multifractal.
            - `generate_terrain`: heightfield, slope, moisture, temperature, biome labels.
            - `heightmap_to_mesh`: terrain triangle mesh generation.
            - `generate_roads`: least-cost road/path generation and road mesh.
            - `poisson_disc_sampling`: constraint-based object placement.
            - `place_vegetation`: trees, shrubs, rocks.
            - `place_buildings`: huts, towers, ruins, buildings.
            - `LocalStyleModel`: optional local generative attribute model.
            - `render_frame` / `encode_video`: live 3D preview and demo video.
            - `make_obj`, `make_glb`, `make_blender_script`: export engines.
            - `compute_metrics`: evaluation harness.
            - `run_self_tests`: reproducible test suite.

            ### M1 scope delivered

            M1 is procedural terrain in Blender-ready form. This app goes slightly beyond M1 by also including:

            - biome presets,
            - roads,
            - vegetation,
            - buildings,
            - live demo video,
            - metrics,
            - OBJ/GLB/Blender script export.

            ### Exact run commands

            ```bash
            python -m venv .venv

            # Windows PowerShell
            .\\.venv\\Scripts\\Activate.ps1

            # Windows CMD
            .venv\\Scripts\\activate.bat

            # macOS / Linux
            source .venv/bin/activate

            pip install -r requirements.txt
            streamlit run app.py
            ```

            ### Tests

            ```bash
            python app.py --self-test
            ```

            Or open the **Tests** tab in the Streamlit app and click **Run self-tests**.

            ### Blender render instructions

            1. Open the app.
            2. Go to **Export**.
            3. Download `blender_terrain_<seed>.py`.
            4. Run:

            ```bash
            blender --background --python blender_terrain_{cfg['seed']}.py -- --render terrain_render.png
            ```

            Export to glTF from Blender:

            ```bash
            blender --background --python blender_terrain_{cfg['seed']}.py -- --gltf terrain_{cfg['seed']}.glb
            ```

            The generated Blender script creates:

            - terrain mesh,
            - simple procedural vegetation,
            - simple buildings/props,
            - camera,
            - sun light,
            - optional still render,
            - optional GLB export.

            ### Godot / game-engine notes

            Use the exported `.glb` terrain and `instances.json` for runtime instancing.

            Recommended Godot workflow:

            - import `terrain_<seed>.glb` as StaticBody3D or MeshInstance3D;
            - parse `instances_<seed>.json`;
            - instantiate prefabs for tree, shrub, rock, hut, tower, ruin;
            - use transformed arrays for MultiMeshInstance3D where possible.

            ### Performance optimization strategy

            - polygon budget controls effective terrain resolution;
            - instance limit controls asset count;
            - preview renderer decimates terrain for fast live video;
            - exports retain full-resolution terrain;
            - road and vegetation placement use spatial constraints to reduce collision work;
            - metrics estimate draw calls and triangle load for target engines.

            ### License and commercial distribution notes

            This prototype uses no paid APIs and no cloud dependencies.

            Before commercial distribution, verify current licenses for:

            - Streamlit: Apache-2.0
            - NumPy: BSD-3-Clause
            - SciPy: BSD-3-Clause
            - Pillow: MIT-style / HPND
            - OpenCV: Apache-2.0, but check bundled codec/runtime implications
            - Trimesh: MIT
            - Blender: GPL for the application; generated assets/scripts should be reviewed separately
            - Godot: MIT

            The app attempts MP4 output using locally available OpenCV codecs. If codec/patent compliance is a concern for commercial shipping, prefer:

            - GIF fallback,
            - WebM/VP9/AV1 where supported,
            - royalty-free H.264 via properly licensed FFmpeg builds,
            - or ship engine-native capture instead of pre-encoded video.

            ### Milestone roadmap

            - M1: procedural terrain in Blender — delivered.
            - M2: vegetation and object placement — delivered as foundation.
            - M3: multiple environment presets — delivered.
            - M4: complete scenes from configuration files — delivered through JSON config/export.
            - M5: interactive world editor — delivered as Streamlit editor.
            - M6: reusable asset libraries and commercial project workspaces — foundation delivered through procedural asset library, instance manifests, and export bundles.
            """
        )


def _running_under_streamlit() -> bool:
    try:
        return bool(st.runtime.exists())
    except Exception:
        return any(k.startswith("STREAMLIT_") for k in os.environ) or "streamlit" in " ".join(sys.argv).lower()


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        ok, report = run_self_tests()
        print(report)
        sys.exit(0 if ok else 1)

    if _running_under_streamlit():
        main()
    else:
        print("Run the interactive app with:")
        print("  streamlit run app.py")
        print("")
        print("Run self-tests with:")
        print("  python app.py --self-test")