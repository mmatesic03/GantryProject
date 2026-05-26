"""Oracle reconstruction from perfect rich stroke-structure labels.

This script tests whether the richer labels are sufficient for decoding before
we trust a model to predict them. It does not use the contour baseline and does
not send anything to Arduino.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from stroke_based_pipeline import (
    Command,
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
class StructureLabels:
    gray: np.ndarray
    support: np.ndarray
    centreline: np.ndarray
    endpoint: np.ndarray
    corner: np.ndarray
    junction: np.ndarray
    tangent_cos: np.ndarray
    tangent_sin: np.ndarray
    tangent_valid: np.ndarray
    stroke_id_map: np.ndarray | None
    vector_strokes: list[np.ndarray] | None
    closed_stroke_ids: set[int]
    model_path: str | None
    label_schema: str | None
    source_record: dict | None = None


@dataclass
class StructureNode:
    id: int
    kind: str
    pixels: list[Pixel]
    x: float
    y: float


@dataclass
class StructureEdge:
    id: int
    u: int
    v: int
    points_xy: np.ndarray
    used: bool = False


@dataclass
class DecodeResult:
    strokes_px: list[np.ndarray]
    raw_strokes_px: list[np.ndarray]
    edges: list[StructureEdge]
    nodes: list[StructureNode]
    claimed_support_mask: np.ndarray
    missed_support_mask: np.ndarray
    endpoint_count: int
    corner_count: int
    junction_count: int
    traced_endpoint_count: int
    tangent_consistency_score: float
    continuity_metadata_used: bool = False


def neighbor_offsets() -> list[tuple[int, int]]:
    return [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]


def pixel_neighbors(pixel: Pixel, mask: np.ndarray) -> list[Pixel]:
    y, x = pixel
    height, width = mask.shape
    result = []
    for dy, dx in neighbor_offsets():
        ny, nx = y + dy, x + dx
        if 0 <= ny < height and 0 <= nx < width and mask[ny, nx]:
            result.append((ny, nx))
    return result


def neighbor_count(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(bool), 1, mode="constant", constant_values=False)
    height, width = mask.shape
    count = np.zeros(mask.shape, dtype=np.uint8)
    for dy in range(3):
        for dx in range(3):
            if dy == 1 and dx == 1:
                continue
            count += padded[dy : dy + height, dx : dx + width].astype(np.uint8)
    return count


def expanded_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool).copy()
    height, width = mask.shape
    output = np.zeros_like(mask, dtype=bool)
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys.tolist(), xs.tolist()):
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        output[y0:y1, x0:x1] = True
    return output


def connected_components(mask: np.ndarray) -> list[list[Pixel]]:
    remaining = set(map(tuple, np.argwhere(mask)))
    components: list[list[Pixel]] = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        points = [start]
        while stack:
            pixel = stack.pop()
            for nbr in pixel_neighbors(pixel, mask):
                if nbr in remaining:
                    remaining.remove(nbr)
                    stack.append(nbr)
                    points.append(nbr)
        components.append(points)
    return components


def heatmap_count(heatmap: np.ndarray, threshold: float) -> int:
    return len(connected_components(heatmap >= threshold))


def classify_node_component(points: list[Pixel], labels: StructureLabels, args: argparse.Namespace) -> str:
    ys = np.array([p[0] for p in points], dtype=np.int32)
    xs = np.array([p[1] for p in points], dtype=np.int32)
    if float(np.max(labels.junction[ys, xs])) >= args.junction_threshold:
        return "junction"
    if float(np.max(labels.endpoint[ys, xs])) >= args.endpoint_threshold:
        return "endpoint"
    if float(np.max(labels.corner[ys, xs])) >= args.corner_threshold:
        return "corner"
    degree = neighbor_count(labels.centreline)
    if np.max(degree[ys, xs]) >= 3:
        return "junction"
    if np.min(degree[ys, xs]) <= 1:
        return "endpoint"
    return "corner"


def build_nodes(labels: StructureLabels, args: argparse.Namespace) -> tuple[list[StructureNode], np.ndarray]:
    centreline = labels.centreline.astype(bool)
    degree = neighbor_count(centreline)
    endpoint_zone = expanded_mask(labels.endpoint >= args.endpoint_threshold, args.node_radius_px)
    corner_zone = expanded_mask(labels.corner >= args.corner_threshold, args.node_radius_px)
    junction_zone = expanded_mask(labels.junction >= args.junction_threshold, args.node_radius_px)
    # Corners are turn-through hints, not hard split/merge nodes. Junctions and
    # endpoints are topology nodes; branch-like raster aliasing inside a corner
    # zone should not force a pen-up lift.
    structural = centreline & (endpoint_zone | junction_zone | ((degree <= 1) | ((degree >= 3) & ~corner_zone)))
    node_zone = structural & centreline
    label_map = np.full(centreline.shape, -1, dtype=np.int32)
    nodes: list[StructureNode] = []
    for node_id, points in enumerate(connected_components(node_zone)):
        ys = np.array([p[0] for p in points], dtype=np.float32)
        xs = np.array([p[1] for p in points], dtype=np.float32)
        kind = classify_node_component(points, labels, args)
        nodes.append(StructureNode(node_id, kind, points, float(np.mean(xs)), float(np.mean(ys))))
        for y, x in points:
            label_map[y, x] = node_id
    return nodes, label_map


def trace_edge_from_node(
    centreline: np.ndarray,
    label_map: np.ndarray,
    labels: StructureLabels,
    start_node: int,
    start_pixel: Pixel,
    nxt: Pixel,
    visited_links: set[tuple[Pixel, Pixel]],
) -> tuple[int, np.ndarray] | None:
    points = [start_pixel]
    previous = start_pixel
    current = nxt
    while True:
        link = tuple(sorted([previous, current]))
        if link in visited_links:
            return None
        visited_links.add(link)
        points.append(current)
        node_id = int(label_map[current])
        if node_id >= 0 and node_id != start_node:
            return node_id, np.array([(x, y) for y, x in points], dtype=np.float32)
        candidates = [p for p in pixel_neighbors(current, centreline) if p != previous]
        if not candidates:
            if node_id == start_node:
                return start_node, np.array([(x, y) for y, x in points], dtype=np.float32)
            return -1, np.array([(x, y) for y, x in points], dtype=np.float32)
        if len(candidates) > 1:
            extra_node = int(label_map[current])
            if extra_node >= 0:
                return extra_node, np.array([(x, y) for y, x in points], dtype=np.float32)
            candidates.sort(key=lambda pixel: tangent_alignment(labels, current, previous, pixel), reverse=True)
            previous, current = current, candidates[0]
            continue
        previous, current = current, candidates[0]


def boundary_pixels(node: StructureNode, centreline: np.ndarray, label_map: np.ndarray) -> list[Pixel]:
    result = []
    for pixel in node.pixels:
        if any(label_map[nbr] != node.id for nbr in pixel_neighbors(pixel, centreline)):
            result.append(pixel)
    return result


def build_edges(labels: StructureLabels, nodes: list[StructureNode], label_map: np.ndarray, args: argparse.Namespace) -> list[StructureEdge]:
    centreline = labels.centreline.astype(bool)
    edges: list[StructureEdge] = []
    visited_links: set[tuple[Pixel, Pixel]] = set()
    for node in nodes:
        seen_outgoing_pixels: set[Pixel] = set()
        for start_pixel in boundary_pixels(node, centreline, label_map):
            for nxt in pixel_neighbors(start_pixel, centreline):
                if label_map[nxt] == node.id:
                    continue
                if nxt in seen_outgoing_pixels:
                    continue
                seen_outgoing_pixels.add(nxt)
                traced = trace_edge_from_node(centreline, label_map, labels, node.id, start_pixel, nxt, visited_links)
                if traced is None:
                    continue
                end_node, points_xy = traced
                if end_node < 0:
                    pixels = [(int(round(y)), int(round(x))) for x, y in points_xy]
                    ys = np.array([p[0] for p in pixels], dtype=np.float32)
                    xs = np.array([p[1] for p in pixels], dtype=np.float32)
                    end_node = len(nodes)
                    nodes.append(StructureNode(end_node, "endpoint", pixels[-1:], float(xs[-1]), float(ys[-1])))
                if len(points_xy) >= args.min_points:
                    edges.append(StructureEdge(len(edges), node.id, int(end_node), points_xy))

    # Closed loops can have no explicit endpoint or branch node.
    unvisited = labels.centreline.astype(bool).copy()
    for edge in edges:
        for x, y in edge.points_xy:
            unvisited[int(round(y)), int(round(x))] = False
    unvisited &= label_map < 0
    for points in connected_components(unvisited):
        if len(points) < args.min_points:
            continue
        ordered = order_component_path(points, labels)
        if len(ordered) >= args.min_points:
            loop_node = len(nodes)
            nodes.append(StructureNode(loop_node, "loop", points[:1], float(ordered[0, 0]), float(ordered[0, 1])))
            edges.append(StructureEdge(len(edges), loop_node, loop_node, ordered))
    return edges


def order_component_path(points: list[Pixel], labels: StructureLabels) -> np.ndarray:
    mask = np.zeros_like(labels.centreline, dtype=bool)
    for y, x in points:
        mask[y, x] = True
    endpoints = [p for p in points if len(pixel_neighbors(p, mask)) <= 1]
    start = endpoints[0] if endpoints else points[0]
    ordered = [start]
    visited = {start}
    previous: Pixel | None = None
    current = start
    while True:
        candidates = [p for p in pixel_neighbors(current, mask) if p != previous and p not in visited]
        if not candidates:
            break
        candidates.sort(key=lambda p: tangent_alignment(labels, current, previous, p), reverse=True)
        nxt = candidates[0]
        ordered.append(nxt)
        visited.add(nxt)
        previous, current = current, nxt
    return np.array([(x, y) for y, x in ordered], dtype=np.float32)


def tangent_alignment(labels: StructureLabels, current: Pixel, previous: Pixel | None, candidate: Pixel) -> float:
    cy, cx = current
    direction = np.array([candidate[1] - cx, candidate[0] - cy], dtype=np.float32)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return -1.0
    direction /= norm
    tangent = np.array([labels.tangent_cos[cy, cx], labels.tangent_sin[cy, cx]], dtype=np.float32)
    tnorm = float(np.linalg.norm(tangent))
    if tnorm < 1e-6:
        return 0.0
    tangent /= tnorm
    return abs(float(np.dot(direction, tangent)))


def orient_edge(edge: StructureEdge, from_node: int) -> np.ndarray:
    if edge.u == from_node:
        return edge.points_xy
    return edge.points_xy[::-1].copy()


def other_node(edge: StructureEdge, node_id: int) -> int:
    return edge.v if edge.u == node_id else edge.u


def edge_direction_at_node(edge: StructureEdge, node_id: int, leaving: bool) -> np.ndarray:
    points = orient_edge(edge, node_id)
    if not leaving:
        points = points[::-1]
    if len(points) < 2:
        return np.zeros(2, dtype=np.float32)
    vector = points[1] - points[0]
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-6 else np.zeros(2, dtype=np.float32)


def choose_next_edge(current_edge: StructureEdge, current_node: StructureNode, incident: list[StructureEdge], labels: StructureLabels) -> StructureEdge | None:
    available = [edge for edge in incident if not edge.used and edge.id != current_edge.id]
    if not available:
        return None
    incoming = -edge_direction_at_node(current_edge, current_node.id, leaving=False)
    tangent = np.array(
        [
            labels.tangent_cos[int(round(current_node.y)), int(round(current_node.x))],
            labels.tangent_sin[int(round(current_node.y)), int(round(current_node.x))],
        ],
        dtype=np.float32,
    )
    scores = []
    for edge in available:
        outgoing = edge_direction_at_node(edge, current_node.id, leaving=True)
        smooth = float(np.dot(incoming, outgoing))
        tangent_score = abs(float(np.dot(outgoing, tangent))) if np.linalg.norm(tangent) > 1e-6 else 0.0
        scores.append((smooth + 0.35 * tangent_score, edge))
    scores.sort(key=lambda item: item[0], reverse=True)
    return scores[0][1]


def stitch_edges(edges: list[StructureEdge], nodes: list[StructureNode], labels: StructureLabels, args: argparse.Namespace) -> list[np.ndarray]:
    incident: dict[int, list[StructureEdge]] = {node.id: [] for node in nodes}
    for edge in edges:
        incident.setdefault(edge.u, []).append(edge)
        incident.setdefault(edge.v, []).append(edge)

    starts = sorted(edges, key=lambda edge: (nodes[edge.u].kind != "endpoint" and nodes[edge.v].kind != "endpoint", edge.id))
    strokes: list[np.ndarray] = []
    for start_edge in starts:
        if start_edge.used:
            continue
        start_node = start_edge.u
        if nodes[start_edge.v].kind == "endpoint" and nodes[start_edge.u].kind != "endpoint":
            start_node = start_edge.v
        current_edge = start_edge
        current_node_id = start_node
        parts = []
        while current_edge is not None and not current_edge.used:
            current_edge.used = True
            oriented = orient_edge(current_edge, current_node_id)
            parts.append(oriented if not parts else oriented[1:])
            current_node_id = other_node(current_edge, current_node_id)
            node = nodes[current_node_id]
            if node.kind == "junction":
                break
            if node.kind == "endpoint" and current_edge is not start_edge:
                break
            current_edge = choose_next_edge(current_edge, node, incident.get(current_node_id, []), labels)
        if parts:
            stroke = np.vstack(parts)
            if len(stroke) >= args.min_points:
                strokes.append(stroke)
    return strokes


def claimed_pixels_from_strokes(strokes: list[np.ndarray], shape: tuple[int, int], radius: int = 1) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    image = Image.new("L", (shape[1], shape[0]), 0)
    draw = ImageDraw.Draw(image)
    for stroke in strokes:
        if len(stroke) == 1:
            x, y = stroke[0]
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=1)
        elif len(stroke) > 1:
            draw.line([tuple(point) for point in stroke], fill=1, width=max(1, radius * 2 + 1))
    mask |= np.asarray(image, dtype=np.uint8).astype(bool)
    return mask


def tangent_consistency(strokes: list[np.ndarray], labels: StructureLabels) -> float:
    scores = []
    height, width = labels.centreline.shape
    for stroke in strokes:
        if len(stroke) < 2:
            continue
        for start, end in zip(stroke[:-1], stroke[1:]):
            vector = end - start
            norm = float(np.linalg.norm(vector))
            if norm < 1e-6:
                continue
            direction = vector / norm
            mx, my = ((start + end) / 2.0).round().astype(int)
            if not (0 <= my < height and 0 <= mx < width and labels.tangent_valid[my, mx]):
                continue
            tangent = np.array([labels.tangent_cos[my, mx], labels.tangent_sin[my, mx]], dtype=np.float32)
            tnorm = float(np.linalg.norm(tangent))
            if tnorm > 1e-6:
                scores.append(abs(float(np.dot(direction, tangent / tnorm))))
    return float(np.mean(scores)) if scores else 0.0


def decode_from_continuity_metadata(labels: StructureLabels, args: argparse.Namespace) -> DecodeResult | None:
    if not getattr(args, "oracle_continuity_metadata", False) or not labels.vector_strokes:
        return None
    strokes: list[np.ndarray] = []
    for stroke in labels.vector_strokes:
        if len(stroke) < args.min_points:
            continue
        simplified = simplify_polyline(stroke.astype(np.float32), args.simplification_epsilon).astype(np.float32)
        if len(simplified) >= args.min_points:
            strokes.append(simplified)
    if not strokes:
        return None
    claimed = claimed_pixels_from_strokes(strokes, labels.support.shape, radius=args.coverage_radius_px) & labels.support
    missed = labels.support & ~claimed
    endpoint_components = connected_components(labels.endpoint >= args.endpoint_threshold)
    traced_endpoints = 0
    for component in endpoint_components:
        cy = float(np.mean([p[0] for p in component]))
        cx = float(np.mean([p[1] for p in component]))
        for stroke in strokes:
            if len(stroke) and (
                np.linalg.norm(stroke[0] - [cx, cy]) <= args.endpoint_usage_radius_px
                or np.linalg.norm(stroke[-1] - [cx, cy]) <= args.endpoint_usage_radius_px
            ):
                traced_endpoints += 1
                break
    return DecodeResult(
        strokes_px=strokes,
        raw_strokes_px=[stroke.copy() for stroke in labels.vector_strokes if len(stroke) >= args.min_points],
        edges=[],
        nodes=[],
        claimed_support_mask=claimed,
        missed_support_mask=missed,
        endpoint_count=len(endpoint_components),
        corner_count=heatmap_count(labels.corner, args.corner_threshold),
        junction_count=heatmap_count(labels.junction, args.junction_threshold),
        traced_endpoint_count=traced_endpoints,
        tangent_consistency_score=tangent_consistency(strokes, labels),
        continuity_metadata_used=True,
    )


def decode_structure(labels: StructureLabels, args: argparse.Namespace) -> DecodeResult:
    metadata_result = decode_from_continuity_metadata(labels, args)
    if metadata_result is not None:
        return metadata_result
    nodes, label_map = build_nodes(labels, args)
    edges = build_edges(labels, nodes, label_map, args)
    raw_strokes = stitch_edges(edges, nodes, labels, args)
    simplified = [simplify_polyline(stroke, args.simplification_epsilon).astype(np.float32) for stroke in raw_strokes]
    strokes = [stroke for stroke in simplified if len(stroke) >= args.min_points]
    claimed = claimed_pixels_from_strokes(strokes, labels.support.shape, radius=args.coverage_radius_px) & labels.support
    missed = labels.support & ~claimed
    endpoint_components = connected_components(labels.endpoint >= args.endpoint_threshold)
    traced_endpoints = 0
    for component in endpoint_components:
        cy = float(np.mean([p[0] for p in component]))
        cx = float(np.mean([p[1] for p in component]))
        for stroke in strokes:
            if len(stroke) and (np.linalg.norm(stroke[0] - [cx, cy]) <= args.endpoint_usage_radius_px or np.linalg.norm(stroke[-1] - [cx, cy]) <= args.endpoint_usage_radius_px):
                traced_endpoints += 1
                break
    return DecodeResult(
        strokes_px=strokes,
        raw_strokes_px=raw_strokes,
        edges=edges,
        nodes=nodes,
        claimed_support_mask=claimed,
        missed_support_mask=missed,
        endpoint_count=len(endpoint_components),
        corner_count=heatmap_count(labels.corner, args.corner_threshold),
        junction_count=heatmap_count(labels.junction, args.junction_threshold),
        traced_endpoint_count=traced_endpoints,
        tangent_consistency_score=tangent_consistency(strokes, labels),
        continuity_metadata_used=False,
    )


def command_distances(commands: list[Command]) -> tuple[float, float, int]:
    draw_distance = 0.0
    travel_distance = 0.0
    mode_changes = 0
    previous: Command | None = None
    previous_mode: int | None = None
    for command in commands:
        x, y, mode = command
        if previous is not None:
            px, py, _ = previous
            distance = math.hypot(x - px, y - py)
            if mode == 1:
                draw_distance += distance
            else:
                travel_distance += distance
        if previous_mode is None or mode != previous_mode:
            mode_changes += 1
        previous = command
        previous_mode = mode
    return draw_distance, travel_distance, mode_changes


def compute_structure_metrics(
    commands: list[Command],
    result: DecodeResult,
    labels: StructureLabels,
    transform_info: dict,
    firmware_constants: dict,
    args: argparse.Namespace,
    pipeline_name: str,
) -> dict:
    draw_distance, travel_distance, mode_changes = command_distances(commands)
    try:
        validate_arduino_commands(commands, args.work_width_mm, args.work_height_mm)
        bounds_valid = True
        bounds_error = None
    except ValueError as exc:
        bounds_valid = False
        bounds_error = str(exc)
    steps_per_mm = (
        firmware_constants["FULL_STEPS_PER_REV"]
        * firmware_constants["MICROSTEPS"]
        / (firmware_constants["PULLEY_TEETH"] * firmware_constants["BELT_PITCH_MM"])
    )
    draw_speed_mm_s = firmware_constants["DRAW_SPEED"] / steps_per_mm
    travel_speed_mm_s = firmware_constants["TRAVEL_SPEED"] / steps_per_mm
    draw_time_s = draw_distance / draw_speed_mm_s if draw_speed_mm_s > 0 else 0.0
    travel_time_s = travel_distance / travel_speed_mm_s if travel_speed_mm_s > 0 else 0.0
    servo_sweep_ms = abs(firmware_constants["PEN_UP_ANGLE"] - firmware_constants["PEN_DOWN_ANGLE"]) * firmware_constants["SERVO_DELAY_MS"]
    pen_change_time_s = mode_changes * (firmware_constants["PEN_SETTLE_MS"] + servo_sweep_ms) / 1000.0
    counts = [len(stroke) for stroke in result.strokes_px]
    support_pixels = int(np.count_nonzero(labels.support))
    missed = int(np.count_nonzero(result.missed_support_mask))
    bounds = {"min_x_mm": None, "max_x_mm": None, "min_y_mm": None, "max_y_mm": None}
    if commands:
        bounds = {
            "min_x_mm": min(command[0] for command in commands),
            "max_x_mm": max(command[0] for command in commands),
            "min_y_mm": min(command[1] for command in commands),
            "max_y_mm": max(command[1] for command in commands),
        }
    return {
        "schema": "stroke_structure_reconstruction_metrics_v1",
        "pipeline": pipeline_name,
        "command_count": len(commands),
        "stroke_count": len(result.strokes_px),
        "average_points_per_stroke": float(np.mean(counts)) if counts else 0.0,
        "median_points_per_stroke": float(np.median(counts)) if counts else 0.0,
        "pen_up_travel_distance_mm": travel_distance,
        "pen_down_drawing_distance_mm": draw_distance,
        "total_movement_distance_mm": travel_distance + draw_distance,
        "estimated_plotting_time_s": draw_time_s + travel_time_s + pen_change_time_s,
        "estimated_draw_movement_time_s": draw_time_s,
        "estimated_travel_movement_time_s": travel_time_s,
        "estimated_pen_mode_change_settle_time_s": pen_change_time_s,
        "bounds_validation_passed": bounds_valid,
        "bounds_validation_error": bounds_error,
        "command_bounds_mm": bounds,
        "line_support_coverage": 1.0 - (missed / max(support_pixels, 1)),
        "endpoint_count": result.endpoint_count,
        "corner_count": result.corner_count,
        "junction_count": result.junction_count,
        "traced_endpoint_usage_fraction": (
            result.traced_endpoint_count / result.endpoint_count if result.endpoint_count > 0 else 1.0
        ),
        "untraced_support_pixel_fraction": missed / max(support_pixels, 1),
        "tangent_consistency_score": result.tangent_consistency_score,
        "continuity_metadata_used": result.continuity_metadata_used,
        "model_path_used": labels.model_path,
        "label_schema_used": labels.label_schema,
        "stroke_point_counts": [int(count) for count in counts],
        "gantry_mapping": transform_info,
        "firmware_constants": firmware_constants,
    }


def load_record(processed_dir: Path, record: dict, manifest: dict) -> StructureLabels:
    labels = record["labels"]
    metadata = record.get("metadata", {})
    vector_strokes = None
    if metadata.get("vector_strokes"):
        vector_strokes = [np.asarray(stroke, dtype=np.float32) for stroke in metadata["vector_strokes"]]
    return StructureLabels(
        gray=np.load(processed_dir / record["image"]).astype(np.uint8),
        support=np.load(processed_dir / labels["stroke_support_mask"]).astype(bool),
        centreline=np.load(processed_dir / labels["centreline_mask"]).astype(bool),
        endpoint=np.load(processed_dir / labels["endpoint_heatmap"]).astype(np.float32),
        corner=np.load(processed_dir / labels["corner_heatmap"]).astype(np.float32),
        junction=np.load(processed_dir / labels["junction_heatmap"]).astype(np.float32),
        tangent_cos=np.load(processed_dir / labels["tangent_cos"]).astype(np.float32),
        tangent_sin=np.load(processed_dir / labels["tangent_sin"]).astype(np.float32),
        tangent_valid=np.load(processed_dir / labels["tangent_valid_mask"]).astype(bool),
        stroke_id_map=np.load(processed_dir / labels["stroke_id_map"]).astype(np.int32) if "stroke_id_map" in labels else None,
        vector_strokes=vector_strokes,
        closed_stroke_ids=set(int(value) for value in metadata.get("closed_stroke_ids", [])),
        model_path=None,
        label_schema=manifest.get("schema"),
        source_record=record,
    )


def save_reconstruction_debug(labels: StructureLabels, result: DecodeResult, output_path: Path) -> None:
    rgb = np.dstack([labels.gray, labels.gray, labels.gray]).astype(np.uint8)
    rgb[labels.support] = [225, 225, 225]
    rgb[result.missed_support_mask] = [240, 60, 55]
    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image)
    colors = [(35, 120, 230), (40, 170, 80), (225, 145, 20), (170, 80, 210), (30, 170, 170)]
    for i, stroke in enumerate(result.strokes_px):
        if len(stroke) >= 2:
            draw.line([tuple(point) for point in stroke], fill=colors[i % len(colors)], width=2)
            sx, sy = stroke[0]
            ex, ey = stroke[-1]
            draw.ellipse((sx - 3, sy - 3, sx + 3, sy + 3), fill=(0, 170, 70))
            draw.ellipse((ex - 3, ey - 3, ex + 3, ey + 3), fill=(35, 90, 235))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def save_structure_overlay(labels: StructureLabels, output_path: Path) -> None:
    rgb = np.dstack([labels.gray, labels.gray, labels.gray]).astype(np.uint8)
    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image)
    for heatmap, color in [(labels.endpoint, (0, 180, 70)), (labels.corner, (230, 145, 0)), (labels.junction, (220, 30, 50))]:
        ys, xs = np.nonzero(heatmap > 0.35)
        for y, x in zip(ys.tolist()[::3], xs.tolist()[::3]):
            draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=color)
    for y in range(4, labels.gray.shape[0], 12):
        for x in range(4, labels.gray.shape[1], 12):
            if labels.tangent_valid[y, x]:
                dx = float(labels.tangent_cos[y, x]) * 5.0
                dy = float(labels.tangent_sin[y, x]) * 5.0
                draw.line((x - dx, y - dy, x + dx, y + dy), fill=(40, 80, 225), width=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def run_one(labels: StructureLabels, output_dir: Path, args: argparse.Namespace, sample_name: str, pipeline_name: str = "oracle_structure_reconstruction") -> dict:
    result = decode_structure(labels, args)
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
    metrics = compute_structure_metrics(commands, result, labels, transform_info, firmware_constants, args, pipeline_name)
    sample_dir = output_dir / sample_name
    sample_dir.mkdir(parents=True, exist_ok=True)
    save_arduino_commands(commands, sample_dir / "arduino_commands.txt")
    save_json(metrics, sample_dir / "stroke_metrics.json")
    save_structure_overlay(labels, sample_dir / "structure_label_overlay.png")
    save_reconstruction_debug(labels, result, sample_dir / "reconstruction_debug.png")
    Image.fromarray(np.where(result.missed_support_mask, 255, 0).astype(np.uint8), mode="L").save(sample_dir / "missed_support_pixels.png")
    save_gantry_preview(paths_mm, sample_dir / "gantry_path_preview.png", args.work_width_mm, args.work_height_mm, args.margin_mm)
    return metrics


def run_oracle(args: argparse.Namespace) -> list[dict]:
    processed_dir = Path(args.processed_dir)
    manifest = json.loads((processed_dir / "manifest.json").read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    records = manifest["records"][: args.max_samples if args.max_samples else None]
    summaries = []
    for index, record in enumerate(records):
        labels = load_record(processed_dir, record, manifest)
        sample_name = f"{index:03d}_{record.get('category', 'sample')}_{record.get('key_id', index)}".replace(" ", "_").replace("/", "_")
        metrics = run_one(labels, output_dir, args, sample_name)
        summaries.append({"sample": sample_name, **metrics})
        print(
            f"{sample_name}: strokes={metrics['stroke_count']} commands={metrics['command_count']} "
            f"coverage={metrics['line_support_coverage']:.3f} bounds={metrics['bounds_validation_passed']}"
        )
    save_json({"schema": "oracle_structure_summary_v1", "samples": summaries}, output_dir / "oracle_summary.json")
    return summaries


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Decode perfect rich labels into gantry stroke commands.")
    parser.add_argument("--processed-dir", default="data/quickdraw/rich")
    parser.add_argument("--output-dir", default="output/oracle_structure")
    parser.add_argument("--max-samples", type=int, default=3)
    parser.add_argument("--endpoint-threshold", type=float, default=0.35)
    parser.add_argument("--corner-threshold", type=float, default=0.35)
    parser.add_argument("--junction-threshold", type=float, default=0.35)
    parser.add_argument("--node-radius-px", type=int, default=2)
    parser.add_argument("--coverage-radius-px", type=int, default=2)
    parser.add_argument("--endpoint-usage-radius-px", type=float, default=5.0)
    parser.add_argument("--min-points", type=int, default=2)
    parser.add_argument("--simplification-epsilon", type=float, default=1.25)
    parser.add_argument("--oracle-continuity-metadata", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--work-width-mm", type=float, default=150.0)
    parser.add_argument("--work-height-mm", type=float, default=270.0)
    parser.add_argument("--margin-mm", type=float, default=5.0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_oracle(args)


if __name__ == "__main__":
    main()
