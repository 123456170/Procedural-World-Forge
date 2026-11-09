from __future__ import annotations

import base64
import io
import json
import math
import os
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from shapely.geometry import Polygon
from shapely.ops import unary_union

try:
    import cv2
except Exception:
    cv2 = None

try:
    from PIL import Image
except Exception:
    Image = None


st.set_page_config(
    page_title="Architectural Concept Studio - M1",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# Logical module: style_engine
# =============================================================================

STYLES: Dict[str, Dict[str, str]] = {
    "Scandinavian Light": {
        "floor": "#d8c3a5",
        "wall": "#f8f9fa",
        "trim": "#ffffff",
        "accent": "#4f6d7a",
        "light": "#fff4e6",
        "color_temp": "2700K warm-neutral",
    },
    "Japandi Calm": {
        "floor": "#c9b79c",
        "wall": "#f2eee8",
        "trim": "#e8dfd3",
        "accent": "#6b705c",
        "light": "#fff1d6",
        "color_temp": "3000K warm",
    },
    "Industrial Loft": {
        "floor": "#8d8680",
        "wall": "#d9d9d9",
        "trim": "#4a4a4a",
        "accent": "#c1666b",
        "light": "#ffe8c2",
        "color_temp": "3500K neutral-warm",
    },
    "Minimal Warm": {
        "floor": "#e3cdb2",
        "wall": "#fbf8f3",
        "trim": "#efe6da",
        "accent": "#9c6644",
        "light": "#fff0dc",
        "color_temp": "2700K warm",
    },
    "Commercial Neutral": {
        "floor": "#cfd4ce",
        "wall": "#f5f5f5",
        "trim": "#dddddd",
        "accent": "#2f6690",
        "light": "#f8f8ff",
        "color_temp": "4000K neutral",
    },
}


# =============================================================================
# Logical module: spatial_model
# =============================================================================

@dataclass
class RoomParams:
    width: float = 4.2
    depth: float = 3.6
    height: float = 2.7
    wall_thickness: float = 0.12

    door_wall: str = "south"
    door_offset: float = 1.2
    door_width: float = 0.9
    door_height: float = 2.05
    hinge_side: str = "left"

    style_name: str = "Scandinavian Light"
    target_lux: int = 300

    min_side_clearance: float = 0.15
    approach_depth: float = 1.2

    show_ceiling: bool = True
    show_lights: bool = True

    seed: int = 20261109


DEFAULT_SESSION: Dict[str, Any] = {
    "width": 4.2,
    "depth": 3.6,
    "height": 2.7,
    "wall_thickness": 0.12,
    "door_wall": "south",
    "door_offset": 1.2,
    "door_width": 0.9,
    "door_height": 2.05,
    "hinge_side": "left",
    "style_name": "Scandinavian Light",
    "target_lux": 300,
    "min_side_clearance": 0.15,
    "approach_depth": 1.2,
    "show_ceiling": True,
    "show_lights": True,
    "auto_update_video": False,
    "test_results": None,
    "test_timestamp": None,
}


BENCHMARKS: Dict[str, Dict[str, Any]] = {
    "Demo living room": {
        "width": 4.2,
        "depth": 3.6,
        "height": 2.7,
        "door_wall": "south",
        "door_offset": 1.2,
        "door_width": 0.9,
        "door_height": 2.05,
        "hinge_side": "left",
        "style_name": "Scandinavian Light",
        "approach_depth": 1.2,
        "min_side_clearance": 0.15,
    },
    "Narrow bedroom": {
        "width": 2.8,
        "depth": 3.4,
        "height": 2.5,
        "door_wall": "south",
        "door_offset": 0.25,
        "door_width": 0.82,
        "door_height": 2.0,
        "hinge_side": "left",
        "style_name": "Japandi Calm",
        "approach_depth": 1.0,
        "min_side_clearance": 0.1,
    },
    "Office cell": {
        "width": 3.6,
        "depth": 3.0,
        "height": 2.75,
        "door_wall": "north",
        "door_offset": 1.35,
        "door_width": 0.95,
        "door_height": 2.1,
        "hinge_side": "right",
        "style_name": "Commercial Neutral",
        "approach_depth": 1.2,
        "min_side_clearance": 0.2,
    },
    "Tight corridor door": {
        "width": 2.2,
        "depth": 4.5,
        "height": 2.4,
        "door_wall": "west",
        "door_offset": 1.5,
        "door_width": 0.9,
        "door_height": 2.05,
        "hinge_side": "left",
        "style_name": "Minimal Warm",
        "approach_depth": 1.2,
        "min_side_clearance": 0.15,
    },
    "Stress test wide door": {
        "width": 2.0,
        "depth": 2.0,
        "height": 2.3,
        "door_wall": "south",
        "door_offset": 0.0,
        "door_width": 1.8,
        "door_height": 2.1,
        "hinge_side": "left",
        "style_name": "Industrial Loft",
        "approach_depth": 2.5,
        "min_side_clearance": 0.3,
    },
}


# =============================================================================
# Logical module: room_parser / constraint_solver utilities
# =============================================================================

def sanitize_params(p: RoomParams) -> RoomParams:
    p.width = float(np.clip(p.width, 1.5, 15.0))
    p.depth = float(np.clip(p.depth, 1.5, 12.0))
    p.height = float(np.clip(p.height, 2.0, 5.0))
    p.wall_thickness = float(np.clip(p.wall_thickness, 0.03, 0.50))

    if p.door_wall not in {"south", "north", "east", "west"}:
        p.door_wall = "south"
    if p.hinge_side not in {"left", "right"}:
        p.hinge_side = "left"
    if p.style_name not in STYLES:
        p.style_name = "Scandinavian Light"

    L = wall_length(p.door_wall, p)
    p.door_width = float(np.clip(p.door_width, 0.40, max(0.40, L)))
    p.door_offset = float(np.clip(p.door_offset, 0.0, max(0.0, L - p.door_width)))
    p.door_height = float(np.clip(p.door_height, 1.50, p.height))

    p.min_side_clearance = float(np.clip(p.min_side_clearance, 0.0, 1.0))
    p.approach_depth = float(np.clip(p.approach_depth, 0.0, 3.0))
    p.target_lux = int(np.clip(p.target_lux, 50, 1000))

    p.show_ceiling = bool(p.show_ceiling)
    p.show_lights = bool(p.show_lights)
    p.seed = int(p.seed)

    return p


def wall_length(wall: str, p: RoomParams) -> float:
    return p.width if wall in ("south", "north") else p.depth


def floor_point(wall: str, u: float, s: float, p: RoomParams) -> Tuple[float, float]:
    if wall == "south":
        return (u, s)
    if wall == "north":
        return (u, p.depth - s)
    if wall == "west":
        return (s, u)
    if wall == "east":
        return (p.width - s, u)
    raise ValueError("Unknown wall: " + str(wall))


def wall_surface_point(wall: str, u: float, v: float, p: RoomParams) -> Tuple[float, float, float]:
    if wall == "south":
        return (u, 0.0, v)
    if wall == "north":
        return (u, p.depth, v)
    if wall == "west":
        return (0.0, u, v)
    if wall == "east":
        return (p.width, u, v)
    raise ValueError("Unknown wall: " + str(wall))


def room_polygon(p: RoomParams) -> Polygon:
    return Polygon([(0.0, 0.0), (p.width, 0.0), (p.width, p.depth), (0.0, p.depth)])


def door_local_hinge(p: RoomParams) -> Tuple[float, float]:
    L = wall_length(p.door_wall, p)
    hinge_u = p.door_offset if p.hinge_side == "left" else p.door_offset + p.door_width
    hinge_u = float(np.clip(hinge_u, 0.0, L))
    closed_dir = 1.0 if p.hinge_side == "left" else -1.0
    return hinge_u, closed_dir


def door_swing_polygon(p: RoomParams, segments: int = 28) -> Polygon:
    hinge_u, closed_dir = door_local_hinge(p)
    dw = p.door_width

    pts_local: List[Tuple[float, float]] = [(hinge_u, 0.0)]
    for th in np.linspace(0.0, math.pi / 2.0, segments):
        u = hinge_u + closed_dir * dw * math.cos(th)
        s = dw * math.sin(th)
        if math.hypot(u - hinge_u, s) > 1e-9:
            pts_local.append((u, s))

    pts_world = [floor_point(p.door_wall, u, s, p) for u, s in pts_local]
    poly = Polygon(pts_world)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def door_approach_polygon(p: RoomParams) -> Polygon:
    if p.approach_depth <= 1e-9:
        return Polygon()

    u1 = p.door_offset
    u2 = p.door_offset + p.door_width
    s2 = p.approach_depth

    pts_local = [(u1, 0.0), (u2, 0.0), (u2, s2), (u1, s2)]
    pts_world = [floor_point(p.door_wall, u, s, p) for u, s in pts_local]
    poly = Polygon(pts_world)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def door_leaf_segment(p: RoomParams, theta: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    hinge_u, closed_dir = door_local_hinge(p)
    dw = p.door_width

    hinge_xy = floor_point(p.door_wall, hinge_u, 0.0, p)
    leaf_u = hinge_u + closed_dir * dw * math.cos(theta)
    leaf_s = dw * math.sin(theta)
    leaf_xy = floor_point(p.door_wall, leaf_u, leaf_s, p)
    return hinge_xy, leaf_xy


# =============================================================================
# Logical module: architecture_geometry
# =============================================================================

def make_rect_surface(
    wall: str,
    u1: float,
    u2: float,
    v1: float,
    v2: float,
    p: RoomParams,
) -> List[Tuple[float, float, float]]:
    return [
        wall_surface_point(wall, u1, v1, p),
        wall_surface_point(wall, u2, v1, p),
        wall_surface_point(wall, u2, v2, p),
        wall_surface_point(wall, u1, v2, p),
    ]


def build_surfaces(p: RoomParams) -> List[Tuple[str, List[Tuple[float, float, float]]]]:
    W, D, H = p.width, p.depth, p.height
    surfaces: List[Tuple[str, List[Tuple[float, float, float]]]] = []

    surfaces.append(("floor", [(0.0, 0.0, 0.0), (W, 0.0, 0.0), (W, D, 0.0), (0.0, D, 0.0)]))

    if p.show_ceiling:
        surfaces.append(("ceiling", [(0.0, 0.0, H), (W, 0.0, H), (W, D, H), (0.0, D, H)]))

    L = wall_length(p.door_wall, p)
    off = p.door_offset
    dw = p.door_width
    dh = p.door_height
    tol = 1e-6

    if off > tol:
        surfaces.append(("wall_" + p.door_wall + "_left", make_rect_surface(p.door_wall, 0.0, off, 0.0, H, p)))

    if L - (off + dw) > tol:
        surfaces.append(("wall_" + p.door_wall + "_right", make_rect_surface(p.door_wall, off + dw, L, 0.0, H, p)))

    if H - dh > tol:
        surfaces.append(("wall_" + p.door_wall + "_header", make_rect_surface(p.door_wall, off, off + dw, dh, H, p)))

    for wall in ("south", "north", "east", "west"):
        if wall == p.door_wall:
            continue
        Lw = wall_length(wall, p)
        surfaces.append(("wall_" + wall, make_rect_surface(wall, 0.0, Lw, 0.0, H, p)))

    return surfaces


def build_wireframe_edges(p: RoomParams) -> List[Tuple[str, List[Tuple[float, float, float]]]]:
    W, D, H = p.width, p.depth, p.height
    edges: List[Tuple[str, List[Tuple[float, float, float]]]] = []

    floor = [(0.0, 0.0, 0.0), (W, 0.0, 0.0), (W, D, 0.0), (0.0, D, 0.0)]
    top = [(x, y, H) for x, y, _ in floor]

    for i in range(4):
        edges.append(("floor", [floor[i], floor[(i + 1) % 4]]))
        edges.append(("top", [top[i], top[(i + 1) % 4]]))
        edges.append(("wall", [floor[i], top[i]]))

    u1 = p.door_offset
    u2 = p.door_offset + p.door_width
    v1 = 0.0
    v2 = p.door_height

    door_pts = [
        wall_surface_point(p.door_wall, u1, v1, p),
        wall_surface_point(p.door_wall, u2, v1, p),
        wall_surface_point(p.door_wall, u2, v2, p),
        wall_surface_point(p.door_wall, u1, v2, p),
    ]

    for i in range(4):
        edges.append(("door", [door_pts[i], door_pts[(i + 1) % 4]]))

    return edges


# =============================================================================
# Logical module: constraint_solver / evaluation
# =============================================================================

def validate_with_polys(
    p: RoomParams,
    room: Polygon,
    swing: Polygon,
    approach: Polygon,
) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []

    def add(check: str, status: str, detail: str) -> None:
        results.append({"Check": check, "Status": status, "Detail": detail})

    L = wall_length(p.door_wall, p)
    off = p.door_offset
    dw = p.door_width
    dh = p.door_height
    H = p.height

    left_clear = off
    right_clear = L - off - dw

    add(
        "Door fits wall",
        "pass" if -1e-9 <= off and off + dw <= L + 1e-9 else "fail",
        "offset=" + format(off, ".3f") + " m, width=" + format(dw, ".3f") + " m, wall length=" + format(L, ".3f") + " m",
    )

    add(
        "Door height fits room",
        "pass" if dh <= H + 1e-9 else "fail",
        "door height=" + format(dh, ".3f") + " m, room height=" + format(H, ".3f") + " m",
    )

    if left_clear < -1e-9 or right_clear < -1e-9:
        side_status = "fail"
    elif left_clear < p.min_side_clearance - 1e-9 or right_clear < p.min_side_clearance - 1e-9:
        side_status = "warn"
    else:
        side_status = "pass"

    add(
        "Side clearance configurable local rule",
        side_status,
        "left=" + format(left_clear, ".3f") + " m, right=" + format(right_clear, ".3f") +
        " m, target=" + format(p.min_side_clearance, ".3f") + " m",
    )

    if swing.is_empty or swing.area <= 1e-9:
        add("Door swing boundary", "fail", "Swing polygon has zero or invalid area.")
    elif room.buffer(1e-8).covers(swing):
        add("Door swing boundary", "pass", "Swing area=" + format(swing.area, ".3f") + " m2 inside room boundary.")
    else:
        add("Door swing boundary", "fail", "Swing polygon extends outside room boundary.")

    if p.approach_depth <= 1e-9:
        add("Approach boundary", "pass", "No approach zone configured.")
    elif approach.is_empty or approach.area <= 1e-9:
        add("Approach boundary", "fail", "Approach polygon has zero or invalid area.")
    elif room.buffer(1e-8).covers(approach):
        add("Approach boundary", "pass", "Approach area=" + format(approach.area, ".3f") + " m2 inside room boundary.")
    else:
        add("Approach boundary", "fail", "Approach polygon extends outside room boundary.")

    opposite_dist = p.depth if p.door_wall in ("south", "north") else p.width
    clear_beyond = opposite_dist - p.approach_depth

    if clear_beyond >= 0.05:
        opp_status = "pass"
    elif clear_beyond >= 0.0:
        opp_status = "warn"
    else:
        opp_status = "fail"

    add(
        "Approach opposite clearance",
        opp_status,
        "approach depth=" + format(p.approach_depth, ".3f") +
        " m, opposite distance=" + format(opposite_dist, ".3f") +
        " m, remaining=" + format(clear_beyond, ".3f") + " m",
    )

    return results


def compute_metrics_with_polys(
    p: RoomParams,
    room: Polygon,
    swing: Polygon,
    approach: Polygon,
) -> Dict[str, float]:
    geoms = [g for g in (swing, approach) if (not g.is_empty) and g.area > 1e-9]
    obstruction = unary_union(geoms) if geoms else Polygon()

    if obstruction.is_empty:
        obstruction_area = 0.0
    else:
        clipped = obstruction.intersection(room)
        obstruction_area = 0.0 if clipped.is_empty else float(clipped.area)

    gross = float(room.area)
    usable = max(0.0, gross - obstruction_area)
    utilization = (usable / gross * 100.0) if gross > 0 else 0.0

    L = wall_length(p.door_wall, p)
    left_clear = p.door_offset
    right_clear = L - p.door_offset - p.door_width
    opposite_dist = p.depth if p.door_wall in ("south", "north") else p.width
    min_clear = max(0.0, min(left_clear, right_clear, opposite_dist - p.approach_depth))

    door_opening_area = float(p.door_width * p.door_height)
    gross_wall_area = float(2.0 * (p.width + p.depth) * p.height)
    net_wall_area = max(0.0, gross_wall_area - door_opening_area)

    return {
        "gross_floor_area": gross,
        "usable_floor_area": usable,
        "utilization_pct": utilization,
        "obstruction_area": obstruction_area,
        "door_swing_area": 0.0 if swing.is_empty else float(swing.area),
        "door_approach_area": 0.0 if approach.is_empty else float(approach.area),
        "volume": float(p.width * p.depth * p.height),
        "perimeter": float(2.0 * (p.width + p.depth)),
        "gross_wall_area": gross_wall_area,
        "net_wall_area": net_wall_area,
        "door_opening_area": door_opening_area,
        "min_configured_clearance_m": min_clear,
    }


# =============================================================================
# Logical module: lighting
# =============================================================================

def lighting_layout(p: RoomParams, style: Dict[str, str]) -> Dict[str, Any]:
    area = p.width * p.depth

    if p.target_lux <= 250:
        max_area_per_light = 6.0
    elif p.target_lux <= 350:
        max_area_per_light = 4.5
    else:
        max_area_per_light = 3.5

    planned_count = max(1, math.ceil(area / max_area_per_light))
    aspect = p.width / max(p.depth, 1e-6)
    cols = max(1, int(round(math.sqrt(planned_count * aspect))))
    rows = max(1, math.ceil(planned_count / cols))

    inset = min(0.6, min(p.width, p.depth) / 4.0)

    if cols > 1 and p.width - 2.0 * inset > 1e-6:
        xs = np.linspace(inset, p.width - inset, cols)
    else:
        xs = np.array([p.width / 2.0])

    if rows > 1 and p.depth - 2.0 * inset > 1e-6:
        ys = np.linspace(inset, p.depth - inset, rows)
    else:
        ys = np.array([p.depth / 2.0])

    fixtures: List[List[float]] = []
    for x in xs:
        for y in ys:
            fixtures.append([round(float(x), 3), round(float(y), 3), round(p.height - 0.05, 3)])

    spacing = math.sqrt(area / max(1, len(fixtures)))

    return {
        "fixtures": fixtures,
        "count": len(fixtures),
        "planned_count": planned_count,
        "spacing_m": round(spacing, 3),
        "color_temp": style.get("color_temp", "3000K"),
        "note": "Conceptual lighting layout only. Not a photometric or regulatory compliance calculation.",
    }


# =============================================================================
# Logical module: scene_builder
# =============================================================================

def build_scene(p: RoomParams) -> Dict[str, Any]:
    p = sanitize_params(p)
    style = STYLES.get(p.style_name, STYLES["Scandinavian Light"])

    surfaces = build_surfaces(p)
    edges = build_wireframe_edges(p)

    room = room_polygon(p)
    swing = door_swing_polygon(p)
    approach = door_approach_polygon(p)

    validations = validate_with_polys(p, room, swing, approach)
    metrics = compute_metrics_with_polys(p, room, swing, approach)

    util = metrics["utilization_pct"]
    if util >= 65.0:
        util_status = "pass"
    elif util >= 45.0:
        util_status = "warn"
    else:
        util_status = "fail"

    validations.append(
        {
            "Check": "Spatial utilization heuristic",
            "Status": util_status,
            "Detail": "Usable floor area=" + format(metrics["usable_floor_area"], ".2f") +
                      " m2 (" + format(util, ".1f") + "% of gross area).",
        }
    )

    lights = lighting_layout(p, style)

    return {
        "params": p,
        "style": style,
        "surfaces": surfaces,
        "edges": edges,
        "room": room,
        "swing": swing,
        "approach": approach,
        "validations": validations,
        "metrics": metrics,
        "lights": lights,
        "milestone": "M1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# =============================================================================
# Logical module: visualization / 3D viewer
# =============================================================================

def add_3d_polygon_mesh(
    fig: go.Figure,
    poly: List[Tuple[float, float, float]],
    color: str,
    opacity: float,
    name: str,
) -> None:
    if len(poly) < 3:
        return

    xs = [pt[0] for pt in poly]
    ys = [pt[1] for pt in poly]
    zs = [pt[2] for pt in poly]

    i = [0] * (len(poly) - 2)
    j = list(range(1, len(poly) - 1))
    k = list(range(2, len(poly)))

    fig.add_trace(
        go.Mesh3d(
            x=xs,
            y=ys,
            z=zs,
            i=i,
            j=j,
            k=k,
            color=color,
            opacity=opacity,
            name=name,
            flatshading=True,
            hoverinfo="name",
        )
    )


def add_flat_polygon_mesh(
    fig: go.Figure,
    polygon: Polygon,
    z: float,
    color: str,
    opacity: float,
    name: str,
) -> None:
    if polygon.is_empty or polygon.area <= 1e-9:
        return

    coords = list(polygon.exterior.coords)[:-1]
    if len(coords) < 3:
        return

    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    zs = [z] * len(coords)

    i = [0] * (len(coords) - 2)
    j = list(range(1, len(coords) - 1))
    k = list(range(2, len(coords)))

    fig.add_trace(
        go.Mesh3d(
            x=xs,
            y=ys,
            z=zs,
            i=i,
            j=j,
            k=k,
            color=color,
            opacity=opacity,
            name=name,
            hoverinfo="name",
        )
    )


def build_plotly_figure(scene: Dict[str, Any]) -> go.Figure:
    p: RoomParams = scene["params"]
    style = scene["style"]
    fig = go.Figure()

    for name, poly in scene["surfaces"]:
        if name == "floor":
            color = style["floor"]
            opacity = 0.95
        elif name == "ceiling":
            color = style["trim"]
            opacity = 0.20
        else:
            color = style["wall"]
            opacity = 0.45
        add_3d_polygon_mesh(fig, poly, color, opacity, name.replace("_", " ").title())

    add_flat_polygon_mesh(fig, scene["swing"], 0.015, style["accent"], 0.22, "Door swing")
    add_flat_polygon_mesh(fig, scene["approach"], 0.020, "#9aa0a6", 0.14, "Door approach")

    edge_colors = {
        "floor": style["trim"],
        "wall": style["wall"],
        "top": style["trim"],
        "door": style["accent"],
    }

    groups: Dict[str, Dict[str, List[float]]] = {}
    nan = float("nan")

    for label, seg in scene["edges"]:
        g = groups.setdefault(label, {"x": [], "y": [], "z": []})
        for pt in seg:
            g["x"].append(pt[0])
            g["y"].append(pt[1])
            g["z"].append(pt[2])
        g["x"].append(nan)
        g["y"].append(nan)
        g["z"].append(nan)

    for label, g in groups.items():
        fig.add_trace(
            go.Scatter3d(
                x=g["x"],
                y=g["y"],
                z=g["z"],
                mode="lines",
                line=dict(color=edge_colors.get(label, "#555555"), width=4),
                name=label.capitalize(),
                hoverinfo="skip",
            )
        )

    hinge_xy, leaf_xy = door_leaf_segment(p, 0.0)
    fig.add_trace(
        go.Scatter3d(
            x=[hinge_xy[0], leaf_xy[0]],
            y=[hinge_xy[1], leaf_xy[1]],
            z=[0.02, 0.02],
            mode="lines",
            line=dict(color=style["accent"], width=8),
            name="Door leaf",
            hoverinfo="name",
        )
    )

    if p.show_lights and scene["lights"]["fixtures"]:
        fx = [f[0] for f in scene["lights"]["fixtures"]]
        fy = [f[1] for f in scene["lights"]["fixtures"]]
        fz = [f[2] for f in scene["lights"]["fixtures"]]
        fig.add_trace(
            go.Scatter3d(
                x=fx,
                y=fy,
                z=fz,
                mode="markers",
                marker=dict(size=5, color=style["light"], symbol="diamond", line=dict(width=1, color="#555555")),
                name="Lighting concept",
                hoverinfo="name",
            )
        )

    fig.update_layout(
        title="M1 Interactive 3D Room - " + p.style_name,
        height=690,
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        scene=dict(
            xaxis=dict(title="X m", range=[-0.45, p.width + 0.45], gridcolor="#d9d9d9", zeroline=False),
            yaxis=dict(title="Y m", range=[-0.45, p.depth + 0.45], gridcolor="#d9d9d9", zeroline=False),
            zaxis=dict(title="Z m", range=[-0.25, p.height + 0.45], gridcolor="#d9d9d9", zeroline=False),
            aspectmode="data",
            bgcolor="white",
            camera=dict(eye=dict(x=1.7, y=-1.7, z=0.95)),
        ),
    )

    return fig


# =============================================================================
# Logical module: visualization / floor plan SVG
# =============================================================================

def build_floor_plan_svg(scene: Dict[str, Any]) -> Tuple[str, int]:
    p: RoomParams = scene["params"]
    style = scene["style"]

    scale = max(35, min(90, int(760 / max(p.width, p.depth))))
    margin = 75
    svg_w = int(2 * margin + p.width * scale)
    svg_h = int(2 * margin + p.depth * scale + 40)

    def X(x: float) -> float:
        return margin + x * scale

    def Y(y: float) -> float:
        return margin + (p.depth - y) * scale

    parts: List[str] = []
    parts.append(
        '<svg xmlns="http://www.w3.org/2000/svg" width="' + str(svg_w) + '" height="' + str(svg_h) +
        '" viewBox="0 0 ' + str(svg_w) + ' ' + str(svg_h) + '">'
    )
    parts.append('<rect x="0" y="0" width="' + str(svg_w) + '" height="' + str(svg_h) + '" fill="#ffffff" />')

    parts.append(
        '<text x="' + format(svg_w / 2.0, ".1f") + '" y="28" text-anchor="middle" '
        'font-family="Arial, sans-serif" font-size="16" fill="#222222">M1 Floor Plan - ' +
        p.style_name + '</text>'
    )

    parts.append(
        '<rect x="' + format(X(0.0), ".1f") + '" y="' + format(Y(p.depth), ".1f") +
        '" width="' + format(p.width * scale, ".1f") + '" height="' + format(p.depth * scale, ".1f") +
        '" fill="' + style["floor"] + '" stroke="none" />'
    )

    stroke_w = max(2.0, p.wall_thickness * scale)
    wall_stroke = "#64748b"

    def add_wall_line(wall: str, u1: float, u2: float) -> None:
        if u2 - u1 <= 1e-6:
            return
        x1, y1 = floor_point(wall, u1, 0.0, p)
        x2, y2 = floor_point(wall, u2, 0.0, p)
        parts.append(
            '<line x1="' + format(X(x1), ".1f") + '" y1="' + format(Y(y1), ".1f") +
            '" x2="' + format(X(x2), ".1f") + '" y2="' + format(Y(y2), ".1f") +
            '" stroke="' + wall_stroke + '" stroke-width="' + format(stroke_w, ".1f") +
            '" stroke-linecap="butt" />'
        )

    L = wall_length(p.door_wall, p)
    off = p.door_offset
    dw = p.door_width

    add_wall_line(p.door_wall, 0.0, off)
    add_wall_line(p.door_wall, off + dw, L)

    for wall in ("south", "north", "east", "west"):
        if wall == p.door_wall:
            continue
        Lw = wall_length(wall, p)
        add_wall_line(wall, 0.0, Lw)

    hinge_xy, closed_xy = door_leaf_segment(p, 0.0)
    parts.append(
        '<line x1="' + format(X(hinge_xy[0]), ".1f") + '" y1="' + format(Y(hinge_xy[1]), ".1f") +
        '" x2="' + format(X(closed_xy[0]), ".1f") + '" y2="' + format(Y(closed_xy[1]), ".1f") +
        '" stroke="' + style["accent"] + '" stroke-width="3" stroke-linecap="round" />'
    )

    swing = scene["swing"]
    if (not swing.is_empty) and swing.area > 1e-9:
        pts = list(swing.exterior.coords)[:-1]
        point_str = " ".join(format(X(x), ".1f") + "," + format(Y(y), ".1f") for x, y in pts)
        parts.append(
            '<polyline points="' + point_str + '" fill="none" stroke="' + style["accent"] +
            '" stroke-width="1.5" stroke-dasharray="5 4" opacity="0.85" />'
        )

    approach = scene["approach"]
    if (not approach.is_empty) and approach.area > 1e-9:
        pts = list(approach.exterior.coords)[:-1]
        point_str = " ".join(format(X(x), ".1f") + "," + format(Y(y), ".1f") for x, y in pts)
        parts.append(
            '<polyline points="' + point_str + '" fill="none" stroke="#9aa0a6" '
            'stroke-width="1.2" stroke-dasharray="3 4" opacity="0.9" />'
        )

    parts.append(
        '<text x="' + format(X(p.width / 2.0), ".1f") + '" y="' + format(Y(0.0) + 38.0, ".1f") +
        '" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#333333">' +
        format(p.width, ".2f") + ' m</text>'
    )

    parts.append(
        '<text x="' + format(X(0.0) - 38.0, ".1f") + '" y="' + format(Y(p.depth / 2.0), ".1f") +
        '" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#333333" '
        'transform="rotate(-90 ' + format(X(0.0) - 38.0, ".1f") + ' ' + format(Y(p.depth / 2.0), ".1f") +
        ')">' + format(p.depth, ".2f") + ' m</text>'
    )

    parts.append(
        '<text x="' + format(X(hinge_xy[0]) + 8.0, ".1f") + '" y="' + format(Y(hinge_xy[1]) - 8.0, ".1f") +
        '" font-family="Arial, sans-serif" font-size="11" fill="' + style["accent"] + '">door</text>'
    )

    parts.append("</svg>")
    return "\n".join(parts), svg_h


# =============================================================================
# Logical module: api/export helpers local file exports only
# =============================================================================

def build_json_export(scene: Dict[str, Any]) -> bytes:
    p: RoomParams = scene["params"]
    data = {
        "milestone": scene["milestone"],
        "generated_at": scene["generated_at"],
        "disclaimer": (
            "This tool provides conceptual design and configurable local-rule checks. "
            "It does not automatically certify regulatory or building-code compliance."
        ),
        "params": asdict(p),
        "metrics": scene["metrics"],
        "validations": scene["validations"],
        "lighting_concept": scene["lights"],
        "geometry_wkt": {
            "room": scene["room"].wkt,
            "door_swing": scene["swing"].wkt,
            "door_approach": scene["approach"].wkt,
        },
    }
    return json.dumps(data, indent=2, default=str).encode("utf-8")


def build_obj_export(scene: Dict[str, Any]) -> bytes:
    lines_v: List[str] = ["# Architectural Concept Studio M1 OBJ"]
    lines_f: List[str] = []
    idx = 1

    for _, poly in scene["surfaces"]:
        if len(poly) < 3:
            continue
        start = idx
        for pt in poly:
            lines_v.append("v " + format(pt[0], ".6f") + " " + format(pt[1], ".6f") + " " + format(pt[2], ".6f"))
            idx += 1
        for i in range(1, len(poly) - 1):
            lines_f.append("f " + str(start) + " " + str(start + i) + " " + str(start + i + 1))

    return ("\n".join(lines_v + lines_f) + "\n").encode("utf-8")


def build_export_zip(scene: Dict[str, Any], json_bytes: bytes, svg_bytes: bytes, obj_bytes: bytes) -> bytes:
    buf = io.BytesIO()
    readme = (
        "Architectural Concept Studio M1 export bundle\n"
        "==============================================\n\n"
        "Files:\n"
        "- room_spec.json: parametric room, validation, metrics, lighting concept\n"
        "- floor_plan.svg: 2D floor plan with door swing and approach zone\n"
        "- room_m1.obj: simple 3D geometry export for further modeling\n\n"
        "Disclaimer:\n"
        "This package is for conceptual design and local configurable checks only.\n"
        "It does not automatically determine regulatory or building-code compliance.\n"
    ).encode("utf-8")

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("room_spec.json", json_bytes)
        zf.writestr("floor_plan.svg", svg_bytes)
        zf.writestr("room_m1.obj", obj_bytes)
        zf.writestr("README.txt", readme)

    return buf.getvalue()


# =============================================================================
# Logical module: visualization / live demo video renderer
# =============================================================================

def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.lstrip("#")
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return (b, g, r)


def render_frames(
    p: RoomParams,
    style: Dict[str, str],
    fps: int = 15,
    duration: float = 3.0,
    width: int = 640,
    height: int = 360,
) -> List[np.ndarray]:
    W, D, H = p.width, p.depth, p.height
    edges = build_wireframe_edges(p)

    swing_poly = door_swing_polygon(p)
    swing_coords = list(swing_poly.exterior.coords)[:-1] if not swing_poly.is_empty else []

    room = room_polygon(p)
    approach = door_approach_polygon(p)
    validations = validate_with_polys(p, room, swing_poly, approach)
    pass_count = sum(1 for v in validations if v["Status"] == "pass")
    total_count = len(validations)

    floor_col = hex_to_bgr(style["floor"])
    wall_col = hex_to_bgr(style["wall"])
    trim_col = hex_to_bgr(style["trim"])
    accent_col = hex_to_bgr(style["accent"])

    bg = np.array([28, 24, 20], dtype=np.uint8)
    focal = width / (2.0 * math.tan(math.radians(30.0)))
    cx, cy = width / 2.0, height / 2.0
    target = np.array([W / 2.0, D / 2.0, H * 0.45], dtype=float)

    n_frames = max(2, int(fps * duration))
    frames: List[np.ndarray] = []

    for frame_idx in range(n_frames):
        progress = frame_idx / max(1, n_frames - 1)
        az = 2.0 * math.pi * progress
        el = 0.36
        r = max(W, D) * 2.1 + H * 0.9

        cam = target + np.array(
            [
                r * math.cos(az) * math.cos(el),
                r * math.sin(az) * math.cos(el),
                r * math.sin(el),
            ],
            dtype=float,
        )

        forward = target - cam
        fnorm = np.linalg.norm(forward)
        if fnorm < 1e-9:
            forward = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            forward = forward / fnorm

        world_up = np.array([0.0, 0.0, 1.0], dtype=float)
        right = np.cross(forward, world_up)
        rn = np.linalg.norm(right)
        if rn < 1e-6:
            right = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            right = right / rn

        up = np.cross(right, forward)

        def proj(pt: Tuple[float, float, float]) -> Optional[Tuple[float, float, float]]:
            v = np.array(pt, dtype=float) - cam
            zc = float(np.dot(v, forward))
            if zc <= 0.10:
                return None
            xc = float(np.dot(v, right))
            yc = float(np.dot(v, up))
            u = cx + focal * xc / zc
            vv = cy - focal * yc / zc
            return (u, vv, zc)

        img = np.tile(bg, (height, width, 1))

        floor_pts = [(0.0, 0.0, 0.0), (W, 0.0, 0.0), (W, D, 0.0), (0.0, D, 0.0)]
        projected_floor = [proj(q) for q in floor_pts]
        if all(q is not None for q in projected_floor):
            poly = np.array([[int(q[0]), int(q[1])] for q in projected_floor], dtype=np.int32)
            dark_floor = tuple(int(c * 0.55) for c in floor_col)
            cv2.fillPoly(img, [poly], dark_floor)

        for label, seg in edges:
            a = proj(seg[0])
            b = proj(seg[1])
            if a is None or b is None:
                continue
            color = {
                "floor": floor_col,
                "wall": wall_col,
                "top": trim_col,
                "door": accent_col,
            }.get(label, wall_col)
            thickness = 3 if label == "door" else 1
            cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), color, thickness)

        prev: Optional[Tuple[int, int]] = None
        dark_accent = tuple(int(c * 0.75) for c in accent_col)
        for x, y in swing_coords:
            q = proj((x, y, 0.02))
            if q is None:
                prev = None
                continue
            cur = (int(q[0]), int(q[1]))
            if prev is not None:
                cv2.line(img, prev, cur, dark_accent, 1)
            prev = cur

        theta = (math.pi / 2.0) * (0.5 - 0.5 * math.cos(2.0 * math.pi * progress))
        hinge_xy, leaf_xy = door_leaf_segment(p, theta)
        ah = proj((hinge_xy[0], hinge_xy[1], 0.02))
        al = proj((leaf_xy[0], leaf_xy[1], 0.02))
        if ah is not None and al is not None:
            cv2.line(img, (int(ah[0]), int(ah[1])), (int(al[0]), int(al[1])), accent_col, 4)

        cv2.rectangle(img, (10, 10), (width - 10, 92), (15, 15, 18), -1)
        cv2.putText(
            img,
            "M1 Procedural Room + Door Simulation",
            (20, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            "Room " + format(W, ".2f") + " x " + format(D, ".2f") + " x " + format(H, ".2f") +
            " m | Door " + p.door_wall + " " + format(p.door_width, ".2f") + " x " + format(p.door_height, ".2f"),
            (20, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            trim_col,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            "Style " + p.style_name + " | Checks " + str(pass_count) + "/" + str(total_count) +
            " | Orbit " + format(math.degrees(az), ".0f") + " deg",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 225, 180),
            1,
            cv2.LINE_AA,
        )

        frames.append(img)

    return frames


def generate_demo_assets(p: RoomParams, style: Dict[str, str]) -> Tuple[Optional[bytes], Optional[bytes], str]:
    if cv2 is None:
        return None, None, "OpenCV is not available. Install opencv-python-headless from requirements.txt."

    p = sanitize_params(p)
    frames = render_frames(p, style, fps=15, duration=3.0, width=640, height=360)
    if not frames:
        return None, None, "Renderer produced no frames."

    fps = 15
    size = (frames[0].shape[1], frames[0].shape[0])
    tmp_dir = tempfile.mkdtemp(prefix="archstudio_video_")
    path = os.path.join(tmp_dir, "demo.mp4")
    mp4 = None
    errors: List[str] = []

    for codec in ("avc1", "H264", "mp4v"):
        writer = None
        try:
            if os.path.exists(path):
                os.remove(path)

            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(path, fourcc, fps, size)

            if not writer.isOpened():
                errors.append(codec + ": writer could not open")
                continue

            for fr in frames:
                writer.write(fr)

            writer.release()
            writer = None

            if os.path.exists(path) and os.path.getsize(path) > 0:
                with open(path, "rb") as f:
                    candidate = f.read()
                if candidate:
                    mp4 = candidate
                    break
                errors.append(codec + ": output file was empty")

        except Exception as exc:
            errors.append(codec + ": " + str(exc))

        finally:
            if writer is not None:
                try:
                    writer.release()
                except Exception:
                    pass

    try:
        if os.path.exists(path):
            os.remove(path)
        os.rmdir(tmp_dir)
    except Exception:
        pass

    gif = None
    gif_error = ""

    if Image is not None:
        try:
            rgb_frames = [Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)) for fr in frames]
            buf = io.BytesIO()
            rgb_frames[0].save(
                buf,
                format="GIF",
                append_images=rgb_frames[1:],
                duration=int(1000 / fps),
                loop=0,
                optimize=False,
            )
            gif = buf.getvalue()
        except Exception as exc:
            gif_error = str(exc)
    else:
        gif_error = "Pillow is not available."

    messages: List[str] = []

    if mp4 is None:
        if errors:
            messages.append("MP4 generation failed: " + "; ".join(errors))
        else:
            messages.append("MP4 generation failed.")

    if gif is None:
        if gif_error:
            messages.append("GIF fallback unavailable: " + gif_error)
        else:
            messages.append("GIF fallback unavailable.")

    if mp4 is not None and gif is not None:
        msg = ""
    elif mp4 is not None:
        msg = "MP4 generated. " + messages[0]
    elif gif is not None:
        msg = "GIF generated. " + messages[0]
    else:
        msg = " ".join(messages)

    return mp4, gif, msg.strip()


def b64_data(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def build_demo_html(mp4: Optional[bytes], gif: Optional[bytes]) -> str:
    header = '<!DOCTYPE html><html><head><meta charset="utf-8"></head><body style="margin:0; background:#111;">'
    footer = "</body></html>"
    container_start = '<div style="max-width:640px; margin:0 auto;">'
    container_end = "</div>"

    if mp4:
        video_style = "width:100%; height:auto; background:#000; display:block;"
        parts = [
            header,
            container_start,
            '<video controls autoplay muted loop playsinline style="' + video_style + '">',
            '<source src="data:video/mp4;base64,' + b64_data(mp4) + '" type="video/mp4">',
        ]

        if gif:
            parts.append(
                '<img src="data:image/gif;base64,' + b64_data(gif) +
                '" alt="M1 demo animation" style="width:100%; height:auto; display:block;">'
            )
        else:
            parts.append('<div style="color:#ffffff; padding:16px;">MP4 source could not be played by this browser.</div>')

        parts.append("</video>")
        parts.append(container_end)
        parts.append(footer)
        return "".join(parts)

    if gif:
        parts = [
            header,
            container_start,
            '<img src="data:image/gif;base64,' + b64_data(gif) +
            '" alt="M1 demo animation" style="width:100%; height:auto; display:block;">',
            container_end,
            footer,
        ]
        return "".join(parts)

    return (
        header +
        container_start +
        '<div style="color:#ffffff; padding:16px;">No demo video or animation was generated.</div>' +
        container_end +
        footer
    )


# =============================================================================
# Logical module: tests / built-in evaluation
# =============================================================================

def run_self_tests() -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []

    def record(name: str, fn) -> None:
        try:
            fn()
            out.append({"Check": name, "Result": "PASS", "Detail": ""})
        except AssertionError as exc:
            out.append({"Check": name, "Result": "FAIL", "Detail": str(exc)})
        except Exception as exc:
            out.append({"Check": name, "Result": "ERROR", "Detail": type(exc).__name__ + ": " + str(exc)})

    def reproducibility() -> None:
        p = RoomParams()
        s1 = build_scene(p)
        s2 = build_scene(p)
        assert json.dumps(s1["metrics"], sort_keys=True, default=str) == json.dumps(
            s2["metrics"], sort_keys=True, default=str
        )
        assert len(s1["edges"]) == len(s2["edges"])

    def geometric_accuracy() -> None:
        p = RoomParams(width=3.0, depth=4.0, height=2.5)
        m = build_scene(p)["metrics"]
        assert abs(m["gross_floor_area"] - 12.0) < 1e-9
        assert abs(m["volume"] - 30.0) < 1e-9

    def boundary_violation() -> None:
        p = RoomParams()
        s = build_scene(p)
        assert s["room"].buffer(1e-8).covers(s["swing"])

    def overlap_consistency() -> None:
        p = RoomParams()
        s = build_scene(p)
        union = unary_union([s["swing"], s["approach"]])
        assert union.area <= s["swing"].area + s["approach"].area + 1e-9

    def circulation_clearance_detection() -> None:
        p = RoomParams(
            width=2.0,
            depth=2.0,
            door_wall="south",
            door_offset=0.0,
            door_width=1.8,
            door_height=2.1,
            min_side_clearance=0.3,
            approach_depth=2.5,
        )
        vals = {v["Check"]: v["Status"] for v in build_scene(p)["validations"]}
        assert vals.get("Approach opposite clearance") in ("fail", "warn")

    def layout_feasibility_demo() -> None:
        p = RoomParams()
        statuses = [v["Status"] for v in build_scene(p)["validations"]]
        assert "fail" not in statuses

    def live_video_renderer() -> None:
        if cv2 is None:
            raise AssertionError("OpenCV unavailable")
        p = RoomParams()
        style = STYLES[p.style_name]
        frames = render_frames(p, style, fps=5, duration=0.4, width=160, height=90)
        assert len(frames) >= 2
        assert frames[0].shape == (90, 160, 3)

    record("Reproducibility of generated layout", reproducibility)
    record("Geometric accuracy", geometric_accuracy)
    record("Boundary violation check", boundary_violation)
    record("Overlap consistency check", overlap_consistency)
    record("Circulation clearance detection", circulation_clearance_detection)
    record("Layout feasibility for demo room", layout_feasibility_demo)
    record("Live video renderer", live_video_renderer)

    return out


# =============================================================================
# Logical module: frontend / Streamlit application
# =============================================================================

def main() -> None:
    if "arch_initialized" not in st.session_state:
        st.session_state.arch_initialized = True
        for k, v in DEFAULT_SESSION.items():
            st.session_state[k] = v

    st.title("Architectural and Interior Concept Generation Platform")
    st.caption(
        "Milestone 1: parametric 3D room with walls, floor, one animated door, configurable local clearance checks, "
        "spatial utilization analysis, lighting concept, interactive editing, and local exports. "
        "This tool does not automatically claim regulatory compliance."
    )

    st.sidebar.header("Room and Door Inputs")

    with st.sidebar.expander("Benchmark rooms"):
        st.caption("Click a benchmark to load realistic test geometry.")
        for name, vals in BENCHMARKS.items():
            if st.button(name, key="bench_" + name):
                st.session_state.update(vals)
                st.session_state.video_force = True
                st.rerun()

    width = st.sidebar.slider("Room width m", 1.5, 15.0, step=0.1, key="width")
    depth = st.sidebar.slider("Room depth m", 1.5, 12.0, step=0.1, key="depth")
    height = st.sidebar.slider("Room height m", 2.0, 5.0, step=0.05, key="height")
    wall_thickness = st.sidebar.slider("Wall thickness m", 0.03, 0.50, step=0.01, key="wall_thickness")

    st.sidebar.subheader("Door Placement")
    door_wall = st.sidebar.selectbox("Door wall", ["south", "north", "east", "west"], key="door_wall")

    L = float(width if door_wall in ("south", "north") else depth)

    if "door_width" not in st.session_state:
        st.session_state["door_width"] = 0.9

    dw_min = 0.40
    dw_max = max(dw_min + 0.01, L)
    dw_current = float(st.session_state["door_width"])
    dw_current = min(max(dw_current, dw_min), dw_max)
    st.session_state["door_width"] = dw_current

    door_width = st.sidebar.slider(
        "Door width m",
        dw_min,
        dw_max,
        step=0.01,
        key="door_width",
    )

    max_offset = max(0.0, L - float(door_width))

    if "door_offset" not in st.session_state:
        st.session_state["door_offset"] = 1.2

    off_current = float(st.session_state["door_offset"])
    off_current = min(max(off_current, 0.0), max_offset)
    st.session_state["door_offset"] = off_current

    door_offset = st.sidebar.slider(
        "Door offset from wall origin m",
        0.0,
        max(0.01, max_offset),
        step=0.01,
        key="door_offset",
    )

    dh_min = 1.50
    dh_max = max(dh_min + 0.01, float(height))

    if "door_height" not in st.session_state:
        st.session_state["door_height"] = 2.05

    dh_current = float(st.session_state["door_height"])
    dh_current = min(max(dh_current, dh_min), dh_max)
    st.session_state["door_height"] = dh_current

    door_height = st.sidebar.slider(
        "Door height m",
        dh_min,
        dh_max,
        step=0.01,
        key="door_height",
    )

    hinge_side = st.sidebar.selectbox("Hinge side", ["left", "right"], key="hinge_side")

    st.sidebar.subheader("Configurable Local Rules")
    st.sidebar.caption("These are user-configurable design checks, not automatic legal compliance decisions.")
    min_side_clearance = st.sidebar.slider("Minimum side clearance m", 0.0, 1.0, step=0.01, key="min_side_clearance")
    approach_depth = st.sidebar.slider("Door approach depth m", 0.0, 3.0, step=0.01, key="approach_depth")

    st.sidebar.subheader("Interior Style and Lighting Concept")
    style_name = st.sidebar.selectbox("Interior style", list(STYLES.keys()), key="style_name")
    target_lux = st.sidebar.slider("Target illuminance concept lux", 50, 1000, step=10, key="target_lux")
    show_ceiling = st.sidebar.checkbox("Show ceiling", key="show_ceiling")
    show_lights = st.sidebar.checkbox("Show lighting concept", key="show_lights")

    st.sidebar.subheader("Live Demo Video")
    auto_update_video = st.sidebar.checkbox("Auto-regenerate video on change", key="auto_update_video")
    regen_clicked = st.sidebar.button("Regenerate live demo video", key="regen_video")
    if regen_clicked:
        st.session_state.video_force = True

    p = RoomParams(
        width=width,
        depth=depth,
        height=height,
        wall_thickness=wall_thickness,
        door_wall=door_wall,
        door_offset=door_offset,
        door_width=door_width,
        door_height=door_height,
        hinge_side=hinge_side,
        style_name=style_name,
        target_lux=target_lux,
        min_side_clearance=min_side_clearance,
        approach_depth=approach_depth,
        show_ceiling=show_ceiling,
        show_lights=show_lights,
    )

    p = sanitize_params(p)
    scene = build_scene(p)
    style = scene["style"]

    current_key = json.dumps(asdict(p), sort_keys=True, default=str)
    video_info = st.session_state.get("current_video")
    force_video = bool(st.session_state.pop("video_force", False))

    should_generate_video = (
        force_video
        or video_info is None
        or (bool(auto_update_video) and video_info.get("key") != current_key)
    )

    if should_generate_video:
        with st.spinner("Generating live demo assets..."):
            mp4, gif, msg = generate_demo_assets(p, style)
        st.session_state.current_video = {
            "key": current_key,
            "mp4": mp4,
            "gif": gif,
            "msg": msg,
        }
        video_info = st.session_state.current_video

    json_bytes = build_json_export(scene)
    svg_string, svg_height = build_floor_plan_svg(scene)
    svg_bytes = svg_string.encode("utf-8")
    obj_bytes = build_obj_export(scene)
    zip_bytes = build_export_zip(scene, json_bytes, svg_bytes, obj_bytes)

    tab_demo, tab_3d, tab_plan, tab_analysis, tab_exports, tab_help = st.tabs(
        ["Live Demo", "Interactive 3D", "Floor Plan", "Analysis", "Exports", "Run and Tests"]
    )

    with tab_demo:
        st.subheader("Automatically generated live demo")
        st.write(
            "The app loads a realistic demo room on startup and renders a short 3D orbit simulation with animated door swing. "
            "Use the sidebar to edit dimensions, door placement, style, and configurable clearance rules."
        )

        mp4 = None
        gif = None
        msg = ""

        if video_info:
            mp4 = video_info.get("mp4")
            gif = video_info.get("gif")
            msg = video_info.get("msg") or ""

        if mp4 or gif:
            html = build_demo_html(mp4, gif)
            components.html(html, height=430, scrolling=False)

            if msg:
                st.info(msg)

            d1, d2, d3 = st.columns(3)

            if mp4:
                d1.download_button(
                    "Download MP4",
                    mp4,
                    file_name="demo_m1.mp4",
                    mime="video/mp4",
                    key="dl_mp4",
                )

            if gif:
                d2.download_button(
                    "Download GIF",
                    gif,
                    file_name="demo_m1.gif",
                    mime="image/gif",
                    key="dl_gif",
                )
        else:
            st.error(msg if msg else "No demo assets were generated.")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Gross floor area", format(scene["metrics"]["gross_floor_area"], ".2f") + " m2")
        col2.metric("Usable floor area", format(scene["metrics"]["usable_floor_area"], ".2f") + " m2")
        col3.metric("Utilization", format(scene["metrics"]["utilization_pct"], ".1f") + "%")
        col4.metric(
            "Passing checks",
            str(sum(1 for v in scene["validations"] if v["Status"] == "pass")) + "/" + str(len(scene["validations"])),
        )

    with tab_3d:
        st.subheader("Interactive 3D scene")
        st.caption("Rotate, zoom, and pan. This is a procedural M1 room shell with one parametric door opening.")
        fig = build_plotly_figure(scene)
        st.plotly_chart(fig, use_container_width=True)

    with tab_plan:
        st.subheader("2D floor plan with door swing and approach zone")
        components.html(svg_string, height=svg_height + 10, scrolling=False)

    with tab_analysis:
        st.subheader("Spatial analysis and configurable checks")

        m = scene["metrics"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Volume", format(m["volume"], ".2f") + " m3")
        c2.metric("Perimeter", format(m["perimeter"], ".2f") + " m")
        c3.metric("Net wall area", format(m["net_wall_area"], ".2f") + " m2")
        c4.metric("Min configured clearance", format(m["min_configured_clearance_m"], ".2f") + " m")

        progress_value = min(1.0, max(0.0, m["utilization_pct"] / 100.0))
        st.progress(progress_value)
        st.caption(
            "Spatial utilization heuristic: " + format(m["utilization_pct"], ".1f") +
            "% of gross floor area remains usable after door swing and approach zones."
        )

        failures = [v for v in scene["validations"] if v["Status"] == "fail"]
        warnings = [v for v in scene["validations"] if v["Status"] == "warn"]

        if failures:
            st.error(str(len(failures)) + " check(s) failed. Review the validation table below.")
        elif warnings:
            st.warning(str(len(warnings)) + " check(s) produced warnings. Review the validation table below.")
        else:
            st.success("All configurable M1 checks passed for this concept layout.")

        st.subheader("Validation results")
        st.dataframe(scene["validations"], use_container_width=True)

        st.info(
            "Validation output is a configurable design heuristic. It does not replace local building regulations, "
            "accessibility standards, fire egress analysis, or professional code review."
        )

        st.subheader("Lighting layout concept")
        lights = scene["lights"]
        st.write(
            "Planned fixtures: **" + str(lights["count"]) + "**, conceptual spacing: **" +
            format(lights["spacing_m"], ".2f") + " m**, color temperature: **" + lights["color_temp"] + "**"
        )
        st.caption(lights["note"])

        if lights["fixtures"]:
            fixture_rows = [
                {"Fixture": i + 1, "X m": f[0], "Y m": f[1], "Z m": f[2]}
                for i, f in enumerate(lights["fixtures"])
            ]
            st.dataframe(fixture_rows, use_container_width=True)

    with tab_exports:
        st.subheader("Local exports")
        st.write("Download conceptual design artifacts. All exports are generated locally.")

        e1, e2, e3, e4 = st.columns(4)

        e1.download_button(
            "Download JSON spec",
            json_bytes,
            file_name="room_spec_m1.json",
            mime="application/json",
            key="dl_json",
        )

        e2.download_button(
            "Download SVG floor plan",
            svg_bytes,
            file_name="floor_plan_m1.svg",
            mime="image/svg+xml",
            key="dl_svg",
        )

        e3.download_button(
            "Download OBJ 3D geometry",
            obj_bytes,
            file_name="room_m1.obj",
            mime="model/obj",
            key="dl_obj",
        )

        e4.download_button(
            "Download ZIP bundle",
            zip_bytes,
            file_name="architectural_concept_m1_bundle.zip",
            mime="application/zip",
            key="dl_zip",
        )

    with tab_help:
        st.subheader("Windows plus VS Code run instructions")
        st.markdown(
            r"""
1. Install Python from the official Python website.
2. Open the project folder in VS Code.
3. Open a terminal in VS Code.
4. Create a virtual environment:

```bash
python -m venv .venv
"""
)

st.subheader("Built-in evaluation suite")
st.write(
"This M1 app includes automated checks for reproducibility, geometric accuracy, boundary violations, "
"overlap consistency, circulation clearance detection, layout feasibility, and the live video renderer."
)

if st.button("Run built-in evaluation suite", key="run_tests"):
    with st.spinner("Running tests..."):
        st.session_state.test_results = run_self_tests()
st.session_state.test_timestamp = datetime.now(timezone.utc).isoformat()

test_results = st.session_state.get("test_results")

if test_results:
    st.caption("Last run: " + str(st.session_state.get("test_timestamp", "")))
    st.dataframe(test_results, use_container_width=True)

    passed = sum(1 for r in test_results if r["Result"] == "PASS")
    st.metric("Passed tests", str(passed) + "/" + str(len(test_results)))
else:
    st.info("Click the button above to run the built-in evaluation suite.")

st.subheader("Milestone roadmap")
st.markdown(
"""
"""
)

st.subheader("Architecture mapping")
st.markdown(
"""
"""
)
if os.name == "main":
    main()