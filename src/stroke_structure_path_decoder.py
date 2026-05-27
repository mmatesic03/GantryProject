"""Path-first reconstruction for rich stroke-structure ML outputs.

This experimental decoder is intentionally separate from the existing oracle
and ML graph decoders. It treats the centreline mask as the primary geometry
and uses endpoint/corner/junction predictions as soft evidence rather than
hard pen-lift instructions.

The script can either run the stroke-structure model directly or decode a
saved ``structure_prediction_arrays.npz`` file from a previous inference run.
It exports the same Arduino command format used by the gantry pipeline:

    x_mm,y_mm,mode

where mode 0 is pen-up travel and mode 1 is pen-down drawing.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from oracle_stroke_structure_reconstruction import (
    DecodeResult,
    StructureLabels,
    claimed_pixels_from_strokes,
    component_mask,
    compute_structure_metrics,
    connected_components,
    expanded_mask,
    heatmap_count,
    neighbor_count,
    pixel_neighbors,
    save_reconstruction_debug,
    save_structure_overlay,
    tangent_consistency,
)
from stroke_based_pipeline import (
    map_paths_to_gantry_mm,
    parse_firmware_constants,
    paths_to_arduino_commands,
    save_arduino_commands,
    save_gantry_preview,
    save_json,
    simplify_polyline,
    validate_arduino_commands,
)


Pixel = tuple[int, int]


@dataclass
class EvidencePoint:
    kind: str
    pixel: Pixel
    x: float
    y: float
    strength: float
    terminal: bool = False


@dataclass
class PathDecoderDiagnostics:
    input_endpoint_count: int
    input_corner_count: int
    input_junction_count: int
    terminal_hint_count: int
    reclassified_endpoint_count: int
    bridge_count: int
    component_count_after_bridging: int
    pre_merge_stroke_count: int
    post_merge_stroke_count: int
    dropped_short_stroke_count: int


@dataclass
class InternalEdge:
    id: int
    u: int
    v: int
    points: np.ndarray
    used: bool = False


@dataclass
class InternalNode:
    id: int
    pixels: list[Pixel]
    x: float
    y: float
    kind: str


def load_grayscale(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def prob_panel(array: np.ndarray) -> Image.Image:
    clipped = np.clip(array, 0.0, 1.0)
    return Image.fromarray((clipped * 255.0).astype(np.uint8), mode="L").convert("RGB")


def mask_panel(mask: np.ndarray) -> Image.Image:
    return Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L").convert("RGB")


def distance_px(a: Pixel | np.ndarray, b: Pixel | np.ndarray) -> float:
    ay, ax = pixel_to_yx(a)
    by, bx = pixel_to_yx(b)
    return float(math.hypot(ax - bx, ay - by))


def pixel_to_yx(value: Pixel | np.ndarray) -> tuple[float, float]:
    if isinstance(value, tuple):
        return float(value[0]), float(value[1])
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape[0] != 2:
        raise ValueError(f"Expected 2D point, got shape {arr.shape}")
    return float(arr[1]), float(arr[0])


def yx_to_xy(pixel: Pixel) -> np.ndarray:
    y, x = pixel
    return np.array([x, y], dtype=np.float32)


def xy_to_yx(point: np.ndarray) -> Pixel:
    x, y = point
    return int(round(float(y))), int(round(float(x)))


def line_pixels(start: Pixel, end: Pixel, shape: tuple[int, int]) -> list[Pixel]:
    y0, x0 = start
    y1, x1 = end
    distance = max(abs(y1 - y0), abs(x1 - x0))
    steps = max(1, int(distance))
    pixels: list[Pixel] = []
    seen: set[Pixel] = set()
    for t in np.linspace(0.0, 1.0, steps + 1):
        y = int(round(y0 * (1.0 - t) + y1 * t))
        x = int(round(x0 * (1.0 - t) + x1 * t))
        if 0 <= y < shape[0] and 0 <= x < shape[1] and (y, x) not in seen:
            seen.add((y, x))
            pixels.append((y, x))
    return pixels


def draw_line_on_mask(mask: np.ndarray, start: Pixel, end: Pixel) -> None:
    for y, x in line_pixels(start, end, mask.shape):
        mask[y, x] = True


def nearest_mask_pixel(mask: np.ndarray, yx: tuple[float, float], max_radius: float | None = None) -> Pixel | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    target_y, target_x = yx
    distances = (ys.astype(np.float32) - target_y) ** 2 + (xs.astype(np.float32) - target_x) ** 2
    idx = int(np.argmin(distances))
    if max_radius is not None and float(math.sqrt(float(distances[idx]))) > max_radius:
        return None
    return int(ys[idx]), int(xs[idx])


def tangent_axis_at(labels: StructureLabels, pixel: Pixel) -> np.ndarray | None:
    y, x = pixel
    if not (0 <= y < labels.tangent_valid.shape[0] and 0 <= x < labels.tangent_valid.shape[1]):
        return None
    if not labels.tangent_valid[y, x]:
        return None
    tangent = np.array([labels.tangent_cos[y, x], labels.tangent_sin[y, x]], dtype=np.float32)
    norm = float(np.linalg.norm(tangent))
    if norm < 1e-6:
        return None
    return tangent / norm


def tangent_alignment_between(labels: StructureLabels, start: Pixel, end: Pixel) -> float:
    sy, sx = start
    ey, ex = end
    vector = np.array([ex - sx, ey - sy], dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-6:
        return 1.0
    direction = vector / norm
    scores = []
    for pixel in (start, end):
        tangent = tangent_axis_at(labels, pixel)
        if tangent is not None:
            scores.append(abs(float(np.dot(direction, tangent))))
    return float(np.mean(scores)) if scores else 0.5


def support_fraction_between(labels: StructureLabels, start: Pixel, end: Pixel) -> float:
    pixels = line_pixels(start, end, labels.support.shape)
    if not pixels:
        return 0.0
    supported = 0
    for y, x in pixels:
        if labels.support[y, x] or labels.centreline[y, x] or labels.tangent_valid[y, x]:
            supported += 1
    return supported / len(pixels)


def heatmap_components(heatmap: np.ndarray, threshold: float) -> list[list[Pixel]]:
    return connected_components(heatmap >= threshold)


def snap_evidence_points(
    kind: str,
    heatmap: np.ndarray,
    threshold: float,
    mask: np.ndarray,
    max_radius: float,
) -> list[EvidencePoint]:
    points: list[EvidencePoint] = []
    for component in heatmap_components(heatmap, threshold):
        ys = np.array([p[0] for p in component], dtype=np.float32)
        xs = np.array([p[1] for p in component], dtype=np.float32)
        snap = nearest_mask_pixel(mask, (float(np.mean(ys)), float(np.mean(xs))), max_radius=max_radius)
        if snap is None:
            continue
        strength = float(np.max(heatmap[ys.astype(np.int32), xs.astype(np.int32)]))
        points.append(EvidencePoint(kind=kind, pixel=snap, x=float(snap[1]), y=float(snap[0]), strength=strength))
    return merge_nearby_evidence(points, merge_radius=2.0)


def merge_nearby_evidence(points: list[EvidencePoint], merge_radius: float) -> list[EvidencePoint]:
    merged: list[EvidencePoint] = []
    used = [False] * len(points)
    for i, point in enumerate(points):
        if used[i]:
            continue
        group = [point]
        used[i] = True
        for j in range(i + 1, len(points)):
            if used[j] or points[j].kind != point.kind:
                continue
            if distance_px(point.pixel, points[j].pixel) <= merge_radius:
                used[j] = True
                group.append(points[j])
        weights = np.array([max(p.strength, 1e-3) for p in group], dtype=np.float32)
        y = float(np.average([p.y for p in group], weights=weights))
        x = float(np.average([p.x for p in group], weights=weights))
        strongest = max(group, key=lambda p: p.strength)
        merged.append(
            EvidencePoint(
                kind=point.kind,
                pixel=strongest.pixel,
                x=x,
                y=y,
                strength=float(max(p.strength for p in group)),
                terminal=any(p.terminal for p in group),
            )
        )
    return merged


def local_min_degree(mask: np.ndarray, pixel: Pixel, radius: int) -> int:
    degree = neighbor_count(mask)
    y, x = pixel
    y0, y1 = max(0, y - radius), min(mask.shape[0], y + radius + 1)
    x0, x1 = max(0, x - radius), min(mask.shape[1], x + radius + 1)
    local = degree[y0:y1, x0:x1][mask[y0:y1, x0:x1]]
    if local.size == 0:
        return 0
    return int(np.min(local))


def classify_endpoint_hints(
    mask: np.ndarray,
    endpoints: list[EvidencePoint],
    args: argparse.Namespace,
) -> tuple[list[EvidencePoint], list[EvidencePoint]]:
    terminals: list[EvidencePoint] = []
    reclassified: list[EvidencePoint] = []
    for point in endpoints:
        point.terminal = local_min_degree(mask, point.pixel, args.endpoint_terminal_radius_px) <= 1
        if point.terminal:
            terminals.append(point)
        else:
            reclassified.append(EvidencePoint("corner", point.pixel, point.x, point.y, point.strength, terminal=False))
    return terminals, reclassified


def add_degree_terminals(mask: np.ndarray, existing: list[EvidencePoint], args: argparse.Namespace) -> list[EvidencePoint]:
    degree = neighbor_count(mask)
    terminals = list(existing)
    for y, x in map(tuple, np.argwhere(mask & (degree <= 1))):
        pixel = (int(y), int(x))
        if any(distance_px(pixel, point.pixel) <= args.terminal_merge_px for point in terminals):
            continue
        terminals.append(EvidencePoint("degree_terminal", pixel, float(x), float(y), 0.25, terminal=True))
    return terminals


def remove_small_components(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    if min_pixels <= 1:
        return mask.astype(bool).copy()
    output = np.zeros_like(mask, dtype=bool)
    for component in connected_components(mask):
        if len(component) >= min_pixels:
            for y, x in component:
                output[y, x] = True
    return output


def prune_short_spurs(mask: np.ndarray, protected: np.ndarray, max_length: int) -> np.ndarray:
    if max_length <= 0:
        return mask.astype(bool).copy()
    pruned = mask.astype(bool).copy()
    changed = True
    while changed:
        changed = False
        degree = neighbor_count(pruned)
        endpoints = list(map(tuple, np.argwhere(pruned & (degree <= 1) & ~protected)))
        for start in endpoints:
            path = [start]
            previous: Pixel | None = None
            current = start
            while len(path) <= max_length + 1:
                candidates = [p for p in pixel_neighbors(current, pruned) if p != previous]
                if len(candidates) != 1:
                    break
                previous, current = current, candidates[0]
                path.append(current)
                if protected[current]:
                    break
                if degree[current] != 2:
                    break
            if len(path) <= max_length and not any(protected[p] for p in path):
                for y, x in path:
                    pruned[y, x] = False
                changed = True
    return pruned


def bridge_small_gaps(
    mask: np.ndarray,
    labels: StructureLabels,
    args: argparse.Namespace,
) -> tuple[np.ndarray, int]:
    bridged = mask.astype(bool).copy()
    bridge_count = 0
    for _ in range(max(1, args.bridge_iterations)):
        degree = neighbor_count(bridged)
        components = connected_components(bridged)
        component_id = np.full(bridged.shape, -1, dtype=np.int32)
        terminals: list[tuple[Pixel, int]] = []
        for idx, component in enumerate(components):
            for y, x in component:
                component_id[y, x] = idx
            for y, x in component:
                if degree[y, x] <= 1:
                    terminals.append(((y, x), idx))
        candidates: list[tuple[float, int, int, Pixel, Pixel]] = []
        for i, (first, first_component) in enumerate(terminals):
            for j in range(i + 1, len(terminals)):
                second, second_component = terminals[j]
                distance = distance_px(first, second)
                if distance > args.max_bridge_gap_px:
                    continue
                if first_component == second_component and distance > args.close_loop_gap_px:
                    continue
                support_fraction = support_fraction_between(labels, first, second)
                tangent_score = tangent_alignment_between(labels, first, second)
                very_close = distance <= args.endpoint_merge_px
                if not very_close and support_fraction < args.bridge_support_fraction:
                    continue
                if not very_close and tangent_score < args.bridge_tangent_min:
                    continue
                score = distance - 3.0 * support_fraction - tangent_score
                candidates.append((score, i, j, first, second))
        if not candidates:
            break
        candidates.sort(key=lambda item: item[0])
        used_terminal_indexes: set[int] = set()
        added_this_round = 0
        for _, i, j, first, second in candidates:
            if i in used_terminal_indexes or j in used_terminal_indexes:
                continue
            draw_line_on_mask(bridged, first, second)
            used_terminal_indexes.add(i)
            used_terminal_indexes.add(j)
            added_this_round += 1
            bridge_count += 1
        if added_this_round == 0:
            break
    return bridged, bridge_count


def shortest_path(mask: np.ndarray, start: Pixel, goal: Pixel) -> list[Pixel]:
    queue = [start]
    parents: dict[Pixel, Pixel | None] = {start: None}
    head = 0
    while head < len(queue):
        current = queue[head]
        head += 1
        if current == goal:
            break
        for nbr in pixel_neighbors(current, mask):
            if nbr not in parents:
                parents[nbr] = current
                queue.append(nbr)
    if goal not in parents:
        return []
    path: list[Pixel] = []
    current: Pixel | None = goal
    while current is not None:
        path.append(current)
        current = parents[current]
    path.reverse()
    return path


def longest_shortest_path(mask: np.ndarray, candidates: list[Pixel], limit: int = 32) -> list[Pixel]:
    if len(candidates) < 2:
        return []
    if len(candidates) > limit:
        candidates = sorted(candidates)[:limit]
    best: list[Pixel] = []
    for i, start in enumerate(candidates):
        for goal in candidates[i + 1 :]:
            path = shortest_path(mask, start, goal)
            if len(path) > len(best):
                best = path
    return best


def trace_unvisited_path(mask: np.ndarray, start: Pixel, labels: StructureLabels) -> list[Pixel]:
    ordered = [start]
    visited = {start}
    previous: Pixel | None = None
    current = start
    while True:
        candidates = [p for p in pixel_neighbors(current, mask) if p != previous and p not in visited]
        if not candidates:
            break
        candidates.sort(key=lambda p: tangent_step_score(labels, previous, current, p), reverse=True)
        nxt = candidates[0]
        ordered.append(nxt)
        visited.add(nxt)
        previous, current = current, nxt
    if len(ordered) > 2 and distance_px(ordered[-1], ordered[0]) <= 1.5:
        ordered.append(ordered[0])
    return ordered


def tangent_step_score(labels: StructureLabels, previous: Pixel | None, current: Pixel, candidate: Pixel) -> float:
    tangent_score = tangent_alignment_between(labels, current, candidate)
    if previous is None:
        return tangent_score
    py, px = previous
    cy, cx = current
    ny, nx = candidate
    incoming = np.array([cx - px, cy - py], dtype=np.float32)
    outgoing = np.array([nx - cx, ny - cy], dtype=np.float32)
    in_norm = float(np.linalg.norm(incoming))
    out_norm = float(np.linalg.norm(outgoing))
    smooth = 0.0 if in_norm < 1e-6 or out_norm < 1e-6 else float(np.dot(incoming / in_norm, outgoing / out_norm))
    return 0.65 * smooth + 0.35 * tangent_score


def edge_key(a: Pixel, b: Pixel) -> tuple[Pixel, Pixel]:
    return tuple(sorted([a, b]))


def edge_cover_walk(mask: np.ndarray, start: Pixel, labels: StructureLabels) -> list[Pixel]:
    """Trace a connected component with high coverage when graph routing fails.

    This is deliberately a fallback, not the preferred decoder. It may revisit
    pixels, but it prevents a useful centreline component from being reduced to
    many tiny graph fragments.
    """
    adjacency = {pixel: pixel_neighbors(pixel, mask) for pixel in map(tuple, np.argwhere(mask))}
    visited_edges: set[tuple[Pixel, Pixel]] = set()
    path = [start]
    stack: list[tuple[Pixel, Pixel | None]] = [(start, None)]
    while stack:
        current, previous = stack[-1]
        candidates = [pixel for pixel in adjacency.get(current, []) if edge_key(current, pixel) not in visited_edges]
        if candidates:
            candidates.sort(key=lambda pixel: tangent_step_score(labels, previous, current, pixel), reverse=True)
            nxt = candidates[0]
            visited_edges.add(edge_key(current, nxt))
            path.append(nxt)
            stack.append((nxt, current))
            continue
        stack.pop()
        if stack:
            parent, _ = stack[-1]
            if any(edge_key(parent, pixel) not in visited_edges for pixel in adjacency.get(parent, [])):
                path.append(parent)
    return path


def component_route_coverage(strokes: list[np.ndarray], mask: np.ndarray) -> float:
    if not strokes:
        return 0.0
    drawn = claimed_pixels_from_strokes(strokes, mask.shape, radius=1)
    return float(np.count_nonzero(drawn & mask) / max(np.count_nonzero(mask), 1))


def coverage_walk_component(
    mask: np.ndarray,
    terminals: list[Pixel],
    labels: StructureLabels,
) -> np.ndarray:
    degree_endpoints = component_degree_endpoints(mask)
    start = terminals[0] if terminals else degree_endpoints[0] if degree_endpoints else tuple(map(int, np.argwhere(mask)[0]))
    path = edge_cover_walk(mask, start, labels)
    return np.array([(x, y) for y, x in path], dtype=np.float32)


def component_terminals(component_mask_: np.ndarray, terminals: list[EvidencePoint], args: argparse.Namespace) -> list[Pixel]:
    result: list[Pixel] = []
    zone = expanded_mask(component_mask_, args.endpoint_terminal_radius_px)
    for point in terminals:
        y, x = point.pixel
        if 0 <= y < zone.shape[0] and 0 <= x < zone.shape[1] and zone[y, x]:
            snap = nearest_mask_pixel(component_mask_, (point.y, point.x), max_radius=args.endpoint_terminal_radius_px + 2)
            if snap is not None and snap not in result:
                result.append(snap)
    return result


def component_degree_endpoints(mask: np.ndarray) -> list[Pixel]:
    degree = neighbor_count(mask)
    return list(map(tuple, np.argwhere(mask & (degree <= 1))))


def trace_simple_component(
    mask: np.ndarray,
    terminals: list[Pixel],
    labels: StructureLabels,
) -> np.ndarray:
    degree_endpoints = component_degree_endpoints(mask)
    candidates = terminals or degree_endpoints
    path = longest_shortest_path(mask, candidates) if len(candidates) >= 2 else []
    if not path:
        start = candidates[0] if candidates else tuple(map(int, np.argwhere(mask)[0]))
        path = trace_unvisited_path(mask, start, labels)
    return np.array([(x, y) for y, x in path], dtype=np.float32)


def build_node_map(mask: np.ndarray, terminals: list[Pixel], args: argparse.Namespace) -> tuple[list[InternalNode], np.ndarray]:
    degree = neighbor_count(mask)
    node_zone = mask & ((degree != 2) | expanded_mask(component_mask(terminals, mask.shape), args.node_radius_px))
    label_map = np.full(mask.shape, -1, dtype=np.int32)
    nodes: list[InternalNode] = []
    for node_id, component in enumerate(connected_components(node_zone)):
        ys = np.array([p[0] for p in component], dtype=np.float32)
        xs = np.array([p[1] for p in component], dtype=np.float32)
        max_degree = int(np.max(degree[ys.astype(np.int32), xs.astype(np.int32)]))
        min_degree = int(np.min(degree[ys.astype(np.int32), xs.astype(np.int32)]))
        kind = "branch" if max_degree >= 3 else "terminal" if min_degree <= 1 else "pass"
        nodes.append(InternalNode(node_id, component, float(np.mean(xs)), float(np.mean(ys)), kind))
        for y, x in component:
            label_map[y, x] = node_id
    return nodes, label_map


def trace_edge_from_node(
    mask: np.ndarray,
    label_map: np.ndarray,
    start_node: int,
    start_pixel: Pixel,
    next_pixel: Pixel,
    visited_links: set[tuple[Pixel, Pixel]],
    labels: StructureLabels,
) -> tuple[int, np.ndarray] | None:
    points = [start_pixel]
    previous = start_pixel
    current = next_pixel
    while True:
        link = tuple(sorted([previous, current]))
        if link in visited_links:
            return None
        visited_links.add(link)
        points.append(current)
        node_id = int(label_map[current])
        if node_id >= 0 and node_id != start_node:
            return node_id, np.array([(x, y) for y, x in points], dtype=np.float32)
        candidates = [p for p in pixel_neighbors(current, mask) if p != previous]
        if not candidates:
            return node_id if node_id >= 0 else start_node, np.array([(x, y) for y, x in points], dtype=np.float32)
        if len(candidates) > 1:
            candidates.sort(key=lambda p: tangent_step_score(labels, previous, current, p), reverse=True)
        previous, current = current, candidates[0]


def build_internal_edges(nodes: list[InternalNode], label_map: np.ndarray, mask: np.ndarray, labels: StructureLabels) -> list[InternalEdge]:
    edges: list[InternalEdge] = []
    visited_links: set[tuple[Pixel, Pixel]] = set()
    for node in nodes:
        for pixel in node.pixels:
            for nbr in pixel_neighbors(pixel, mask):
                if int(label_map[nbr]) == node.id:
                    continue
                traced = trace_edge_from_node(mask, label_map, node.id, pixel, nbr, visited_links, labels)
                if traced is None:
                    continue
                end_node, points = traced
                if len(points) >= 2:
                    edges.append(InternalEdge(len(edges), node.id, int(end_node), points))
    return edges


def orient_internal_edge(edge: InternalEdge, node_id: int) -> np.ndarray:
    return edge.points if edge.u == node_id else edge.points[::-1].copy()


def other_internal_node(edge: InternalEdge, node_id: int) -> int:
    return edge.v if edge.u == node_id else edge.u


def edge_vector_at(edge: InternalEdge, node_id: int, leaving: bool) -> np.ndarray:
    points = orient_internal_edge(edge, node_id)
    if not leaving:
        points = points[::-1]
    if len(points) < 2:
        return np.zeros(2, dtype=np.float32)
    vector = points[1] - points[0]
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-6 else np.zeros(2, dtype=np.float32)


def continuation_score(current_edge: InternalEdge, node_id: int, candidate: InternalEdge, labels: StructureLabels) -> float:
    incoming = -edge_vector_at(current_edge, node_id, leaving=False)
    outgoing = edge_vector_at(candidate, node_id, leaving=True)
    smooth = float(np.dot(incoming, outgoing))
    point = xy_to_yx(orient_internal_edge(candidate, node_id)[0])
    tangent = tangent_axis_at(labels, point)
    tangent_score = abs(float(np.dot(outgoing, tangent))) if tangent is not None else 0.0
    return smooth + 0.35 * tangent_score


def compose_internal_edges(edges: list[InternalEdge], nodes: list[InternalNode], labels: StructureLabels, args: argparse.Namespace) -> list[np.ndarray]:
    incident: dict[int, list[InternalEdge]] = {node.id: [] for node in nodes}
    for edge in edges:
        incident.setdefault(edge.u, []).append(edge)
        incident.setdefault(edge.v, []).append(edge)

    starts = sorted(
        edges,
        key=lambda edge: (
            nodes[edge.u].kind != "terminal" and nodes[edge.v].kind != "terminal",
            -len(edge.points),
            edge.id,
        ),
    )
    strokes: list[np.ndarray] = []
    for start_edge in starts:
        if start_edge.used:
            continue
        start_node = start_edge.u
        if nodes[start_edge.v].kind == "terminal" and nodes[start_edge.u].kind != "terminal":
            start_node = start_edge.v
        parts = []
        current_edge: InternalEdge | None = start_edge
        current_node = start_node
        while current_edge is not None and not current_edge.used:
            current_edge.used = True
            oriented = orient_internal_edge(current_edge, current_node)
            parts.append(oriented if not parts else oriented[1:])
            current_node = other_internal_node(current_edge, current_node)
            available = [edge for edge in incident.get(current_node, []) if not edge.used and edge.id != current_edge.id]
            if not available:
                break
            scored = [(continuation_score(current_edge, current_node, edge, labels), edge) for edge in available]
            scored.sort(key=lambda item: item[0], reverse=True)
            best_score, best_edge = scored[0]
            if nodes[current_node].kind == "terminal" and best_score < args.terminal_continue_min_score:
                break
            if nodes[current_node].kind == "branch" and best_score < args.branch_continue_min_score:
                break
            current_edge = best_edge
        if parts:
            strokes.append(np.vstack(parts).astype(np.float32))
    return strokes


def trace_branched_component(
    mask: np.ndarray,
    terminals: list[Pixel],
    labels: StructureLabels,
    args: argparse.Namespace,
) -> list[np.ndarray]:
    nodes, label_map = build_node_map(mask, terminals, args)
    edges = build_internal_edges(nodes, label_map, mask, labels)
    if not edges:
        simple = trace_simple_component(mask, terminals, labels)
        return [simple] if len(simple) >= args.min_points else []
    return compose_internal_edges(edges, nodes, labels, args)


def component_has_branches(mask: np.ndarray, args: argparse.Namespace) -> bool:
    degree = neighbor_count(mask)
    branch_pixels = int(np.count_nonzero(mask & (degree >= 3)))
    return branch_pixels > args.branch_pixel_tolerance


def polyline_length(stroke: np.ndarray) -> float:
    if len(stroke) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(stroke, axis=0), axis=1)))


def split_simplify_with_corners(
    stroke: np.ndarray,
    corner_points: list[EvidencePoint],
    args: argparse.Namespace,
) -> np.ndarray:
    if len(stroke) <= 2:
        return stroke.astype(np.float32)
    pin_indices = {0, len(stroke) - 1}
    for corner in corner_points:
        distances = np.linalg.norm(stroke - np.array([corner.x, corner.y], dtype=np.float32), axis=1)
        idx = int(np.argmin(distances))
        if float(distances[idx]) <= args.corner_preserve_radius_px:
            pin_indices.add(idx)
    ordered = sorted(pin_indices)
    pieces: list[np.ndarray] = []
    for start, end in zip(ordered[:-1], ordered[1:]):
        if end <= start:
            continue
        segment = stroke[start : end + 1]
        simplified = simplify_polyline(segment.astype(np.float32), args.simplification_epsilon)
        pieces.append(simplified if not pieces else simplified[1:])
    if pieces:
        return np.vstack(pieces).astype(np.float32)
    return simplify_polyline(stroke.astype(np.float32), args.simplification_epsilon).astype(np.float32)


def merge_oriented_strokes(first: np.ndarray, first_end: int, second: np.ndarray, second_end: int) -> np.ndarray:
    left = first[::-1].copy() if first_end == 0 else first
    right = second if second_end == 0 else second[::-1].copy()
    if np.linalg.norm(left[-1] - right[0]) < 1e-6:
        return np.vstack([left, right[1:]])
    return np.vstack([left, right])


def merge_close_strokes(
    strokes: list[np.ndarray],
    labels: StructureLabels,
    args: argparse.Namespace,
) -> list[np.ndarray]:
    merged = [stroke.copy() for stroke in strokes if len(stroke) >= args.min_points]
    while True:
        best: tuple[float, int, int, int, int] | None = None
        for i, first in enumerate(merged):
            for j in range(i + 1, len(merged)):
                second = merged[j]
                for first_end, first_xy in enumerate([first[0], first[-1]]):
                    first_pixel = xy_to_yx(first_xy)
                    for second_end, second_xy in enumerate([second[0], second[-1]]):
                        second_pixel = xy_to_yx(second_xy)
                        distance = distance_px(first_pixel, second_pixel)
                        if distance > args.post_merge_gap_px:
                            continue
                        support_fraction = support_fraction_between(labels, first_pixel, second_pixel)
                        tangent_score = tangent_alignment_between(labels, first_pixel, second_pixel)
                        coincident = distance <= args.endpoint_merge_px
                        if not coincident and support_fraction < args.post_merge_support_fraction:
                            continue
                        if not coincident and tangent_score < args.post_merge_tangent_min:
                            continue
                        score = distance - 2.0 * support_fraction - 0.75 * tangent_score
                        candidate = (score, i, first_end, j, second_end)
                        if best is None or candidate < best:
                            best = candidate
        if best is None:
            break
        _, i, first_end, j, second_end = best
        merged[i] = merge_oriented_strokes(merged[i], first_end, merged[j], second_end)
        del merged[j]
    return merged


def decode_path_first(labels: StructureLabels, args: argparse.Namespace) -> tuple[DecodeResult, PathDecoderDiagnostics]:
    base_mask = remove_small_components(labels.centreline.astype(bool), args.min_component_pixels)
    endpoints = snap_evidence_points("endpoint", labels.endpoint, args.endpoint_threshold, base_mask, args.evidence_snap_radius_px)
    corners = snap_evidence_points("corner", labels.corner, args.corner_threshold, base_mask, args.evidence_snap_radius_px)
    junctions = snap_evidence_points("junction", labels.junction, args.junction_threshold, base_mask, args.evidence_snap_radius_px)
    protected = np.zeros_like(base_mask, dtype=bool)
    for point in endpoints + corners + junctions:
        y, x = point.pixel
        protected[max(0, y - 1) : min(base_mask.shape[0], y + 2), max(0, x - 1) : min(base_mask.shape[1], x + 2)] = True
    base_mask = prune_short_spurs(base_mask, protected, args.spur_prune_length_px)
    bridge_mask, bridge_count = bridge_small_gaps(base_mask, labels, args)
    endpoints = snap_evidence_points("endpoint", labels.endpoint, args.endpoint_threshold, bridge_mask, args.evidence_snap_radius_px)
    corners = snap_evidence_points("corner", labels.corner, args.corner_threshold, bridge_mask, args.evidence_snap_radius_px)
    terminal_hints, reclassified = classify_endpoint_hints(bridge_mask, endpoints, args)
    corner_points = corners + reclassified
    terminal_hints = add_degree_terminals(bridge_mask, terminal_hints, args)

    raw_strokes: list[np.ndarray] = []
    components = connected_components(bridge_mask)
    for points in components:
        if len(points) < args.min_component_pixels:
            continue
        mask = component_mask(points, bridge_mask.shape)
        terminals = component_terminals(mask, terminal_hints, args)
        if component_has_branches(mask, args):
            component_strokes = trace_branched_component(mask, terminals, labels, args)
        else:
            stroke = trace_simple_component(mask, terminals, labels)
            component_strokes = [stroke] if len(stroke) >= args.min_points else []
        if component_route_coverage(component_strokes, mask) < args.component_route_min_coverage:
            fallback = coverage_walk_component(mask, terminals, labels)
            if len(fallback) >= args.min_points:
                component_strokes = [fallback]
        raw_strokes.extend(component_strokes)

    pre_merge_count = len(raw_strokes)
    merged = merge_close_strokes(raw_strokes, labels, args)
    post_merge_count = len(merged)
    simplified: list[np.ndarray] = []
    dropped = 0
    for stroke in merged:
        if len(stroke) < args.min_points or polyline_length(stroke) < args.min_stroke_length_px:
            dropped += 1
            continue
        simplified_stroke = split_simplify_with_corners(stroke, corner_points, args)
        if len(simplified_stroke) < args.min_points or polyline_length(simplified_stroke) < args.min_stroke_length_px:
            dropped += 1
            continue
        simplified.append(simplified_stroke.astype(np.float32))

    claimed = claimed_pixels_from_strokes(simplified, labels.support.shape, radius=args.coverage_radius_px) & labels.support
    missed = labels.support & ~claimed
    endpoint_components = heatmap_components(labels.endpoint, args.endpoint_threshold)
    traced_endpoints = 0
    for component in endpoint_components:
        cy = float(np.mean([p[0] for p in component]))
        cx = float(np.mean([p[1] for p in component]))
        for stroke in simplified:
            if len(stroke) and (
                np.linalg.norm(stroke[0] - [cx, cy]) <= args.endpoint_usage_radius_px
                or np.linalg.norm(stroke[-1] - [cx, cy]) <= args.endpoint_usage_radius_px
            ):
                traced_endpoints += 1
                break

    result = DecodeResult(
        strokes_px=simplified,
        raw_strokes_px=merged,
        edges=[],
        nodes=[],
        claimed_support_mask=claimed,
        missed_support_mask=missed,
        endpoint_count=len(endpoint_components),
        corner_count=heatmap_count(labels.corner, args.corner_threshold),
        junction_count=heatmap_count(labels.junction, args.junction_threshold),
        traced_endpoint_count=traced_endpoints,
        tangent_consistency_score=tangent_consistency(simplified, labels),
        continuity_metadata_used=False,
    )
    diagnostics = PathDecoderDiagnostics(
        input_endpoint_count=len(endpoints),
        input_corner_count=len(corners),
        input_junction_count=len(junctions),
        terminal_hint_count=len([point for point in terminal_hints if point.kind == "endpoint"]),
        reclassified_endpoint_count=len(reclassified),
        bridge_count=bridge_count,
        component_count_after_bridging=len(components),
        pre_merge_stroke_count=pre_merge_count,
        post_merge_stroke_count=post_merge_count,
        dropped_short_stroke_count=dropped,
    )
    return result, diagnostics


def save_prediction_mask_outputs(
    probabilities: dict[str, np.ndarray],
    labels: StructureLabels,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    masks = {
        "support": labels.support,
        "centreline": labels.centreline,
        "endpoint": labels.endpoint >= args.endpoint_threshold,
        "corner": labels.corner >= args.corner_threshold,
        "junction": labels.junction >= args.junction_threshold,
        "tangent_valid": labels.tangent_valid,
    }
    np.savez_compressed(
        output_dir / "structure_prediction_arrays.npz",
        support_probability=probabilities["support"].astype(np.float32),
        centreline_probability=probabilities["centreline"].astype(np.float32),
        endpoint_probability=probabilities["endpoint"].astype(np.float32),
        corner_probability=probabilities["corner"].astype(np.float32),
        junction_probability=probabilities["junction"].astype(np.float32),
        tangent_cos=probabilities["tangent_cos"].astype(np.float32),
        tangent_sin=probabilities["tangent_sin"].astype(np.float32),
        support_mask=masks["support"].astype(np.uint8),
        centreline_mask=masks["centreline"].astype(np.uint8),
        endpoint_mask=masks["endpoint"].astype(np.uint8),
        corner_mask=masks["corner"].astype(np.uint8),
        junction_mask=masks["junction"].astype(np.uint8),
        tangent_valid_mask=masks["tangent_valid"].astype(np.uint8),
    )
    for name in ["support", "centreline", "endpoint", "corner", "junction"]:
        prob_panel(probabilities[name]).save(output_dir / f"ml_{name}_probability.png")
        mask_panel(masks[name]).save(output_dir / f"ml_{name}_mask.png")
    mask_panel(masks["tangent_valid"]).save(output_dir / "ml_tangent_valid_mask.png")
    panels = [
        Image.fromarray(labels.gray, mode="L").convert("RGB"),
        mask_panel(masks["support"]),
        mask_panel(masks["centreline"]),
        mask_panel(masks["endpoint"]),
        mask_panel(masks["corner"]),
        mask_panel(masks["junction"]),
        mask_panel(masks["tangent_valid"]),
    ]
    height, width = labels.gray.shape
    canvas = Image.new("RGB", (width * len(panels), height), "white")
    for i, panel in enumerate(panels):
        canvas.paste(panel, (i * width, 0))
    canvas.save(output_dir / "structure_mask_debug.png")


def save_probability_debug(gray: np.ndarray, probabilities: dict[str, np.ndarray], labels: StructureLabels, output_path: Path) -> None:
    panels = [
        Image.fromarray(gray, mode="L").convert("RGB"),
        prob_panel(probabilities["support"]),
        prob_panel(probabilities["centreline"]),
        prob_panel(probabilities["endpoint"]),
        prob_panel(probabilities["corner"]),
        prob_panel(probabilities["junction"]),
    ]
    overlay = Image.fromarray(np.dstack([gray, gray, gray]).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(overlay)
    h, w = gray.shape
    for y in range(4, h, 12):
        for x in range(4, w, 12):
            if labels.tangent_valid[y, x]:
                dx = float(labels.tangent_cos[y, x]) * 5.0
                dy = float(labels.tangent_sin[y, x]) * 5.0
                draw.line((x - dx, y - dy, x + dx, y + dy), fill=(30, 80, 230), width=1)
    panels.append(overlay)
    canvas = Image.new("RGB", (w * len(panels), h), "white")
    for i, panel in enumerate(panels):
        canvas.paste(panel, (i * w, 0))
    canvas.save(output_path)


def save_false_positive_drawn(labels: StructureLabels, strokes: list[np.ndarray], output_path: Path, radius: int) -> None:
    drawn = claimed_pixels_from_strokes(strokes, labels.support.shape, radius=radius)
    false_positive = drawn & ~labels.support
    rgb = np.dstack([labels.gray, labels.gray, labels.gray]).astype(np.uint8)
    rgb[drawn] = [40, 140, 230]
    rgb[false_positive] = [230, 50, 60]
    Image.fromarray(rgb, mode="RGB").save(output_path)


def load_labels_from_prediction_arrays(
    image_path: Path,
    arrays_path: Path,
    args: argparse.Namespace,
) -> tuple[StructureLabels, dict[str, np.ndarray], dict]:
    gray = load_grayscale(image_path)
    arrays = np.load(arrays_path)
    support_prob = arrays["support_probability"].astype(np.float32)
    centreline_prob = arrays["centreline_probability"].astype(np.float32)
    endpoint_prob = arrays["endpoint_probability"].astype(np.float32)
    corner_prob = arrays["corner_probability"].astype(np.float32)
    junction_prob = arrays["junction_probability"].astype(np.float32)
    tangent_cos = arrays["tangent_cos"].astype(np.float32)
    tangent_sin = arrays["tangent_sin"].astype(np.float32)
    labels = StructureLabels(
        gray=gray,
        support=support_prob >= args.support_threshold,
        centreline=centreline_prob >= args.centreline_threshold,
        endpoint=endpoint_prob,
        corner=corner_prob,
        junction=junction_prob,
        tangent_cos=tangent_cos,
        tangent_sin=tangent_sin,
        tangent_valid=(support_prob >= args.tangent_valid_threshold) | (centreline_prob >= args.centreline_threshold),
        stroke_id_map=None,
        vector_strokes=None,
        closed_stroke_ids=set(),
        model_path=str(args.model_path) if args.model_path else None,
        label_schema="saved_structure_prediction_arrays",
        source_record=None,
    )
    probabilities = {
        "support": support_prob,
        "centreline": centreline_prob,
        "endpoint": endpoint_prob,
        "corner": corner_prob,
        "junction": junction_prob,
        "tangent_cos": tangent_cos,
        "tangent_sin": tangent_sin,
    }
    return labels, probabilities, {"model_config": {}, "label_schema": labels.label_schema}


def load_labels(args: argparse.Namespace) -> tuple[StructureLabels, dict[str, np.ndarray], dict]:
    if args.prediction_arrays:
        return load_labels_from_prediction_arrays(Path(args.image), Path(args.prediction_arrays), args)
    if not args.model_path:
        raise ValueError("Provide either --model-path or --prediction-arrays.")
    from stroke_structure_reconstruction_pipeline import infer_structure_labels

    return infer_structure_labels(Path(args.image), Path(args.model_path), args)


def run_pipeline(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels, probabilities, checkpoint = load_labels(args)
    result, diagnostics = decode_path_first(labels, args)
    paths_mm, transform_info = map_paths_to_gantry_mm(
        result.strokes_px,
        labels.support.shape,
        work_w_mm=args.work_width_mm,
        work_h_mm=args.work_height_mm,
        margin_mm=args.margin_mm,
    )
    commands = paths_to_arduino_commands(paths_mm)
    validate_arduino_commands(commands, args.work_width_mm, args.work_height_mm)
    repo_root = Path(__file__).resolve().parents[1]
    firmware_constants = parse_firmware_constants(repo_root.parent / "Arduino Code" / "ArduinoCode.ino")
    metrics = compute_structure_metrics(
        commands,
        result,
        labels,
        transform_info,
        firmware_constants,
        args,
        pipeline_name="ml_stroke_structure_path_decoder",
    )
    metrics["checkpoint_model_config"] = checkpoint.get("model_config", {})
    metrics["thresholds"] = {
        "support_threshold": args.support_threshold,
        "centreline_threshold": args.centreline_threshold,
        "endpoint_threshold": args.endpoint_threshold,
        "corner_threshold": args.corner_threshold,
        "junction_threshold": args.junction_threshold,
        "tangent_valid_threshold": args.tangent_valid_threshold,
    }
    metrics["path_decoder"] = diagnostics.__dict__
    metrics["path_decoder_settings"] = {
        "max_bridge_gap_px": args.max_bridge_gap_px,
        "post_merge_gap_px": args.post_merge_gap_px,
        "endpoint_merge_px": args.endpoint_merge_px,
        "component_route_min_coverage": args.component_route_min_coverage,
        "branch_continue_min_score": args.branch_continue_min_score,
        "terminal_continue_min_score": args.terminal_continue_min_score,
        "simplification_epsilon": args.simplification_epsilon,
    }
    save_arduino_commands(commands, output_dir / "arduino_commands.txt")
    save_json(metrics, output_dir / "stroke_metrics.json")
    save_prediction_mask_outputs(probabilities, labels, output_dir, args)
    save_probability_debug(labels.gray, probabilities, labels, output_dir / "structure_probability_debug.png")
    save_structure_overlay(labels, output_dir / "endpoint_corner_junction_tangent_overlay.png")
    save_reconstruction_debug(labels, result, output_dir / "reconstruction_debug.png")
    Image.fromarray(np.where(result.missed_support_mask, 255, 0).astype(np.uint8), mode="L").save(output_dir / "missed_support_pixels.png")
    save_false_positive_drawn(labels, result.strokes_px, output_dir / "false_positive_drawn_pixels.png", radius=args.coverage_radius_px)
    save_gantry_preview(paths_mm, output_dir / "gantry_path_preview.png", args.work_width_mm, args.work_height_mm, args.margin_mm)
    print(f"Output directory: {output_dir}")
    print(f"Strokes: {metrics['stroke_count']}")
    print(f"Commands: {metrics['command_count']}")
    print(f"Support coverage: {metrics['line_support_coverage']:.3f}")
    print(f"Bounds valid: {metrics['bounds_validation_passed']}")
    print(f"Path decoder bridges: {diagnostics.bridge_count}")
    print(f"Path decoder merged strokes: {diagnostics.pre_merge_stroke_count} -> {diagnostics.post_merge_stroke_count}")
    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run path-first reconstruction from rich stroke-structure masks.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--prediction-arrays")
    parser.add_argument("--output-dir", default="output/stroke_structure_path_decoder")
    parser.add_argument("--support-threshold", type=float, default=0.45)
    parser.add_argument("--centreline-threshold", type=float, default=0.45)
    parser.add_argument("--endpoint-threshold", type=float, default=0.35)
    parser.add_argument("--corner-threshold", type=float, default=0.35)
    parser.add_argument("--junction-threshold", type=float, default=0.35)
    parser.add_argument("--tangent-valid-threshold", type=float, default=0.35)
    parser.add_argument("--evidence-snap-radius-px", type=float, default=8.0)
    parser.add_argument("--endpoint-terminal-radius-px", type=int, default=3)
    parser.add_argument("--terminal-merge-px", type=float, default=4.0)
    parser.add_argument("--endpoint-merge-px", type=float, default=5.0)
    parser.add_argument("--node-radius-px", type=int, default=2)
    parser.add_argument("--coverage-radius-px", type=int, default=2)
    parser.add_argument("--endpoint-usage-radius-px", type=float, default=5.0)
    parser.add_argument("--min-points", type=int, default=2)
    parser.add_argument("--min-component-pixels", type=int, default=4)
    parser.add_argument("--min-stroke-length-px", type=float, default=2.0)
    parser.add_argument("--spur-prune-length-px", type=int, default=2)
    parser.add_argument("--branch-pixel-tolerance", type=int, default=3)
    parser.add_argument("--component-route-min-coverage", type=float, default=0.72)
    parser.add_argument("--max-bridge-gap-px", type=float, default=8.0)
    parser.add_argument("--close-loop-gap-px", type=float, default=5.0)
    parser.add_argument("--bridge-iterations", type=int, default=2)
    parser.add_argument("--bridge-support-fraction", type=float, default=0.45)
    parser.add_argument("--bridge-tangent-min", type=float, default=0.35)
    parser.add_argument("--post-merge-gap-px", type=float, default=10.0)
    parser.add_argument("--post-merge-support-fraction", type=float, default=0.35)
    parser.add_argument("--post-merge-tangent-min", type=float, default=0.25)
    parser.add_argument("--branch-continue-min-score", type=float, default=0.15)
    parser.add_argument("--terminal-continue-min-score", type=float, default=0.55)
    parser.add_argument("--simplification-epsilon", type=float, default=2.5)
    parser.add_argument("--corner-preserve-radius-px", type=float, default=5.0)
    parser.add_argument("--work-width-mm", type=float, default=150.0)
    parser.add_argument("--work-height-mm", type=float, default=270.0)
    parser.add_argument("--margin-mm", type=float, default=5.0)
    parser.add_argument("--stitch-gap-px", type=float, default=0.0)
    parser.add_argument("--stitch-support-fraction", type=float, default=0.5)
    parser.add_argument("--disable-stitching", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
