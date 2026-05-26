"""Clean-slate ML support reconstruction pipeline for gantry stroke planning.

This experimental path is intentionally separate from the contour baseline,
the heuristic stroke pipeline, and the earlier ML graph pipeline. It starts
from ML probability masks, recovers continuous drawable support geometry, then
extracts ordered stroke paths for Arduino serial commands.

The output command format remains:

    x_mm,y_mm,mode

where mode 0 is travel / pen up and mode 1 is draw / pen down.
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
    connected_components,
    map_paths_to_gantry_mm,
    parse_firmware_constants,
    paths_to_arduino_commands,
    save_arduino_commands,
    save_gantry_preview,
    save_json,
    validate_arduino_commands,
)
from stroke_ml_graph_pipeline import MLProbabilities, load_torch_probabilities, probability_image, save_labeled_grid


Pixel = tuple[int, int]


@dataclass
class SupportComponent:
    id: int
    pixels_yx: np.ndarray
    bbox: tuple[int, int, int, int]
    area: int
    line_pixel_count: int
    node_pixel_count: int
    line_fraction: float
    node_fraction: float
    mean_support_probability: float
    max_support_probability: float
    discarded: bool = False
    discard_reason: str = ""


@dataclass
class NodeBlobInfo:
    id: int
    class_name: str
    confidence: float
    area: int
    bbox: tuple[int, int, int, int]
    centroid_x: float
    centroid_y: float
    weighted_centroid_x: float
    weighted_centroid_y: float
    probability_mean: float
    probability_max: float
    port_count: int
    port_directions: list[tuple[float, float]]
    points_yx: np.ndarray


@dataclass
class ReconstructionResult:
    line_mask: np.ndarray
    node_mask: np.ndarray
    support_prob: np.ndarray
    support_mask: np.ndarray
    centreline_mask: np.ndarray
    components: list[SupportComponent]
    node_blobs: list[NodeBlobInfo]
    raw_strokes_px: list[np.ndarray]
    strokes_px: list[np.ndarray]
    endpoint_count: int
    branch_count: int
    claimed_support_pixel_count: int
    support_coverage_fraction: float
    unclaimed_support_pixel_count: int


def shifted_neighbors(mask: np.ndarray) -> list[np.ndarray]:
    padded = np.pad(mask.astype(bool), 1, mode="constant", constant_values=False)
    height, width = mask.shape
    slices = [
        padded[0:height, 1 : width + 1],
        padded[0:height, 2 : width + 2],
        padded[1 : height + 1, 2 : width + 2],
        padded[2 : height + 2, 2 : width + 2],
        padded[2 : height + 2, 1 : width + 1],
        padded[2 : height + 2, 0:width],
        padded[1 : height + 1, 0:width],
        padded[0:height, 0:width],
    ]
    return slices


def transition_count(neighbors: list[np.ndarray]) -> np.ndarray:
    total = np.zeros_like(neighbors[0], dtype=np.uint8)
    for current, nxt in zip(neighbors, neighbors[1:] + neighbors[:1]):
        total += (~current & nxt).astype(np.uint8)
    return total


def zhang_suen_thinning(mask: np.ndarray, max_iterations: int = 120) -> np.ndarray:
    skeleton = mask.astype(bool).copy()
    if not np.any(skeleton):
        return skeleton
    for _ in range(max_iterations):
        changed = False
        p2, p3, p4, p5, p6, p7, p8, p9 = shifted_neighbors(skeleton)
        neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
        b = sum(n.astype(np.uint8) for n in neighbors)
        a = transition_count(neighbors)
        remove = skeleton & (b >= 2) & (b <= 6) & (a == 1) & ~(p2 & p4 & p6) & ~(p4 & p6 & p8)
        if np.any(remove):
            skeleton[remove] = False
            changed = True

        p2, p3, p4, p5, p6, p7, p8, p9 = shifted_neighbors(skeleton)
        neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
        b = sum(n.astype(np.uint8) for n in neighbors)
        a = transition_count(neighbors)
        remove = skeleton & (b >= 2) & (b <= 6) & (a == 1) & ~(p2 & p4 & p8) & ~(p2 & p6 & p8)
        if np.any(remove):
            skeleton[remove] = False
            changed = True
        if not changed:
            break
    return skeleton


def neighbor_directions(connectivity: int = 8) -> list[tuple[int, int]]:
    if connectivity == 4:
        return [(-1, 0), (0, 1), (1, 0), (0, -1)]
    return [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]


def pixel_neighbors(pixel: Pixel, mask: np.ndarray) -> list[Pixel]:
    y, x = pixel
    height, width = mask.shape
    result: list[Pixel] = []
    for dy, dx in neighbor_directions(8):
        ny = y + dy
        nx = x + dx
        if 0 <= ny < height and 0 <= nx < width and mask[ny, nx]:
            result.append((ny, nx))
    return result


def neighbor_count(mask: np.ndarray) -> np.ndarray:
    return sum(n.astype(np.uint8) for n in shifted_neighbors(mask))


def expanded_mask(mask: np.ndarray, radius_px: int) -> np.ndarray:
    if radius_px <= 0:
        return mask.astype(bool).copy()
    height, width = mask.shape
    ys, xs = np.nonzero(mask)
    output = np.zeros_like(mask, dtype=bool)
    for y, x in zip(ys.tolist(), xs.tolist()):
        y0 = max(0, y - radius_px)
        y1 = min(height, y + radius_px + 1)
        x0 = max(0, x - radius_px)
        x1 = min(width, x + radius_px + 1)
        output[y0:y1, x0:x1] = True
    return output


def build_support_fields(probabilities: MLProbabilities, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    line_mask = probabilities.line_prob >= args.line_threshold
    node_mask = probabilities.node_prob >= args.node_threshold
    nearby_node_mask = node_mask & expanded_mask(line_mask, args.node_support_radius_px)
    support_prob = np.maximum(probabilities.line_prob, args.node_support_weight * probabilities.node_prob)
    if args.background_suppression_weight > 0:
        support_prob = support_prob * np.clip(1.0 - args.background_suppression_weight * probabilities.background_prob, 0.0, 1.0)
    support_mask = (support_prob >= args.support_threshold) & (line_mask | nearby_node_mask)
    return line_mask, node_mask, support_prob, support_mask


def describe_support_components(
    support_mask: np.ndarray,
    line_mask: np.ndarray,
    node_mask: np.ndarray,
    support_prob: np.ndarray,
    min_pixels: int,
) -> list[SupportComponent]:
    components: list[SupportComponent] = []
    for component_id, component in enumerate(connected_components(support_mask, connectivity=8)):
        points = np.array(component, dtype=np.int32)
        if len(points) == 0:
            continue
        ys = points[:, 0]
        xs = points[:, 1]
        x0 = int(np.min(xs))
        x1 = int(np.max(xs))
        y0 = int(np.min(ys))
        y1 = int(np.max(ys))
        line_count = int(np.count_nonzero(line_mask[ys, xs]))
        node_count = int(np.count_nonzero(node_mask[ys, xs]))
        values = support_prob[ys, xs]
        discarded = len(points) < min_pixels
        components.append(
            SupportComponent(
                id=component_id,
                pixels_yx=points,
                bbox=(x0, y0, x1, y1),
                area=int(len(points)),
                line_pixel_count=line_count,
                node_pixel_count=node_count,
                line_fraction=line_count / max(len(points), 1),
                node_fraction=node_count / max(len(points), 1),
                mean_support_probability=float(np.mean(values)),
                max_support_probability=float(np.max(values)),
                discarded=discarded,
                discard_reason="below min component pixels" if discarded else "",
            )
        )
    return components


def component_mask(shape: tuple[int, int], points_yx: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if len(points_yx) > 0:
        mask[points_yx[:, 0], points_yx[:, 1]] = True
    return mask


def centreline_from_components(components: list[SupportComponent], support_shape: tuple[int, int], args: argparse.Namespace) -> np.ndarray:
    centreline = np.zeros(support_shape, dtype=bool)
    for component in components:
        if component.discarded:
            continue
        local_mask = component_mask(support_shape, component.pixels_yx)
        if args.centreline_mode == "thinning":
            centreline |= zhang_suen_thinning(local_mask)
        else:
            centreline |= zhang_suen_thinning(local_mask)
    return centreline


def edge_key(a: Pixel, b: Pixel) -> tuple[Pixel, Pixel]:
    return (a, b) if a <= b else (b, a)


def trace_path_from(skeleton: np.ndarray, start: Pixel, nxt: Pixel, node_mask: np.ndarray, visited_links: set[tuple[Pixel, Pixel]]) -> list[Pixel]:
    path = [start, nxt]
    previous = start
    current = nxt
    visited_links.add(edge_key(start, nxt))
    while True:
        if node_mask[current] and current != start:
            break
        candidates = [pixel for pixel in pixel_neighbors(current, skeleton) if pixel != previous]
        unvisited = [pixel for pixel in candidates if edge_key(current, pixel) not in visited_links]
        if not unvisited:
            break
        if len(unvisited) > 1:
            break
        following = unvisited[0]
        visited_links.add(edge_key(current, following))
        previous, current = current, following
        path.append(current)
    return path


def trace_centreline_paths(skeleton: np.ndarray, min_points: int, min_length_px: float) -> tuple[list[np.ndarray], int, int]:
    degree = neighbor_count(skeleton)
    endpoints = skeleton & (degree == 1)
    branches = skeleton & (degree >= 3)
    nodes = endpoints | branches
    visited_links: set[tuple[Pixel, Pixel]] = set()
    paths: list[np.ndarray] = []

    node_pixels = [tuple(pixel.tolist()) for pixel in np.argwhere(nodes)]
    for start in node_pixels:
        for nxt in pixel_neighbors(start, skeleton):
            if edge_key(start, nxt) in visited_links:
                continue
            path = trace_path_from(skeleton, start, nxt, nodes, visited_links)
            array = np.array([(x, y) for y, x in path], dtype=np.float32)
            if keep_path(array, min_points, min_length_px):
                paths.append(array)

    for y, x in np.argwhere(skeleton):
        pixel = (int(y), int(x))
        for nxt in pixel_neighbors(pixel, skeleton):
            if edge_key(pixel, nxt) in visited_links:
                continue
            path = trace_path_from(skeleton, pixel, nxt, nodes, visited_links)
            array = np.array([(x, y) for y, x in path], dtype=np.float32)
            if keep_path(array, min_points, min_length_px):
                paths.append(array)

    return paths, int(np.count_nonzero(endpoints)), int(np.count_nonzero(branches))


def path_length_px(path: np.ndarray) -> float:
    if len(path) < 2:
        return 0.0
    diffs = np.diff(path.astype(np.float64), axis=0)
    return float(np.sum(np.sqrt(np.sum(diffs * diffs, axis=1))))


def keep_path(path: np.ndarray, min_points: int, min_length_px: float) -> bool:
    return len(path) >= min_points and path_length_px(path) >= min_length_px


def perpendicular_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    segment = end - start
    denom = float(np.dot(segment, segment))
    if denom <= 1e-12:
        return float(np.linalg.norm(point - start))
    t = float(np.clip(np.dot(point - start, segment) / denom, 0.0, 1.0))
    projection = start + t * segment
    return float(np.linalg.norm(point - projection))


def simplify_polyline(points: np.ndarray, epsilon: float) -> np.ndarray:
    if epsilon <= 0 or len(points) <= 2:
        return points.copy()
    start = points[0]
    end = points[-1]
    distances = [perpendicular_distance(points[i], start, end) for i in range(1, len(points) - 1)]
    if not distances:
        return points.copy()
    max_index = int(np.argmax(distances))
    max_distance = distances[max_index]
    if max_distance <= epsilon:
        return np.vstack([start, end]).astype(np.float32)
    split = max_index + 1
    left = simplify_polyline(points[: split + 1], epsilon)
    right = simplify_polyline(points[split:], epsilon)
    return np.vstack([left[:-1], right]).astype(np.float32)


def order_strokes_nearest_neighbor(strokes: list[np.ndarray]) -> list[np.ndarray]:
    remaining = [stroke.copy() for stroke in strokes if len(stroke) >= 2]
    ordered: list[np.ndarray] = []
    current: np.ndarray | None = None
    while remaining:
        if current is None:
            idx = min(range(len(remaining)), key=lambda i: (remaining[i][0, 1], remaining[i][0, 0]))
            ordered.append(remaining.pop(idx))
            current = ordered[-1][-1]
            continue
        best: tuple[float, int, bool] | None = None
        for i, stroke in enumerate(remaining):
            d_start = float(np.linalg.norm(stroke[0] - current))
            d_end = float(np.linalg.norm(stroke[-1] - current))
            candidate = (min(d_start, d_end), i, d_end < d_start)
            if best is None or candidate < best:
                best = candidate
        _, idx, reverse = best
        stroke = remaining.pop(idx)
        if reverse:
            stroke = stroke[::-1].copy()
        ordered.append(stroke)
        current = stroke[-1]
    return ordered


def weighted_centroid(points_yx: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    values = probability[points_yx[:, 0], points_yx[:, 1]].astype(np.float64)
    total = float(np.sum(values))
    if total <= 1e-9:
        y, x = np.mean(points_yx, axis=0)
    else:
        y = float(np.sum(points_yx[:, 0] * values) / total)
        x = float(np.sum(points_yx[:, 1] * values) / total)
    return x, y


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-9 or nb <= 1e-9:
        return 180.0
    dot = float(np.clip(np.dot(a / na, b / nb), -1.0, 1.0))
    return math.degrees(math.acos(dot))


def classify_node_blob(port_directions: list[tuple[float, float]], area: int, aspect_ratio: float) -> tuple[str, float]:
    count = len(port_directions)
    if count == 0:
        return "noisy_or_ambiguous", 0.35
    if count == 1:
        return "endpoint", 0.70
    angles = []
    for i in range(count):
        for j in range(i + 1, count):
            angles.append(angle_between(np.array(port_directions[i]), np.array(port_directions[j])))
    max_angle = max(angles) if angles else 180.0
    min_angle = min(angles) if angles else 180.0
    if count == 2:
        if aspect_ratio > 3.0 and max_angle > 130:
            return "line_like_node_fragment", 0.70
        if max_angle > 135:
            return "smooth_bend", 0.68
        return "sharp_corner", 0.74
    if count == 3:
        return "t_junction", 0.70
    if count >= 4:
        if max_angle > 145 and min_angle < 55:
            return "crossing_or_overlap", 0.65
        return "multi_junction", 0.60
    return "noisy_or_ambiguous", 0.30


def analyze_node_blobs(node_mask: np.ndarray, node_prob: np.ndarray, centreline: np.ndarray, args: argparse.Namespace) -> list[NodeBlobInfo]:
    blobs: list[NodeBlobInfo] = []
    for blob_id, component in enumerate(connected_components(node_mask, connectivity=8)):
        points = np.array(component, dtype=np.int32)
        if len(points) < args.min_node_blob_pixels:
            continue
        ys = points[:, 0]
        xs = points[:, 1]
        x0 = int(np.min(xs))
        x1 = int(np.max(xs))
        y0 = int(np.min(ys))
        y1 = int(np.max(ys))
        width = x1 - x0 + 1
        height = y1 - y0 + 1
        aspect = max(width, height) / max(min(width, height), 1)
        centroid_y = float(np.mean(ys))
        centroid_x = float(np.mean(xs))
        weighted_x, weighted_y = weighted_centroid(points, node_prob)
        local = expanded_mask(component_mask(node_mask.shape, points), args.node_blob_port_radius_px) & centreline
        port_directions: list[tuple[float, float]] = []
        for port_component in connected_components(local, connectivity=8):
            port_points = np.array(port_component, dtype=np.int32)
            if len(port_points) == 0:
                continue
            py = float(np.mean(port_points[:, 0]))
            px = float(np.mean(port_points[:, 1]))
            direction = np.array([px - weighted_x, py - weighted_y], dtype=np.float32)
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-9:
                continue
            direction = direction / norm
            if all(angle_between(direction, np.array(existing)) > args.node_blob_port_angle_degrees for existing in port_directions):
                port_directions.append((float(direction[0]), float(direction[1])))
        class_name, confidence = classify_node_blob(port_directions, len(points), aspect)
        values = node_prob[ys, xs]
        blobs.append(
            NodeBlobInfo(
                id=blob_id,
                class_name=class_name,
                confidence=confidence,
                area=int(len(points)),
                bbox=(x0, y0, x1, y1),
                centroid_x=centroid_x,
                centroid_y=centroid_y,
                weighted_centroid_x=float(weighted_x),
                weighted_centroid_y=float(weighted_y),
                probability_mean=float(np.mean(values)),
                probability_max=float(np.max(values)),
                port_count=len(port_directions),
                port_directions=port_directions,
                points_yx=points,
            )
        )
    return blobs


def claimed_pixels_from_strokes(strokes: list[np.ndarray], shape: tuple[int, int], radius: int = 1) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    height, width = shape
    for stroke in strokes:
        for x_f, y_f in stroke:
            x = int(round(float(x_f)))
            y = int(round(float(y_f)))
            if not (0 <= y < height and 0 <= x < width):
                continue
            y0 = max(0, y - radius)
            y1 = min(height, y + radius + 1)
            x0 = max(0, x - radius)
            x1 = min(width, x + radius + 1)
            mask[y0:y1, x0:x1] = True
    return mask


def reconstruct(probabilities: MLProbabilities, args: argparse.Namespace) -> ReconstructionResult:
    line_mask, node_mask, support_prob, support_mask = build_support_fields(probabilities, args)
    components = describe_support_components(support_mask, line_mask, node_mask, support_prob, args.min_component_pixels)
    centreline = centreline_from_components(components, support_mask.shape, args)
    raw_strokes, endpoint_count, branch_count = trace_centreline_paths(
        centreline,
        min_points=args.min_stroke_points,
        min_length_px=args.min_stroke_length_px,
    )
    simplified = [
        simplify_polyline(stroke, args.simplification_epsilon)
        for stroke in raw_strokes
        if keep_path(stroke, args.min_stroke_points, args.min_stroke_length_px)
    ]
    simplified = [stroke for stroke in simplified if keep_path(stroke, 2, args.min_stroke_length_px)]
    strokes = order_strokes_nearest_neighbor(simplified)
    node_blobs = analyze_node_blobs(node_mask, probabilities.node_prob, centreline, args)
    claimed = claimed_pixels_from_strokes(strokes, support_mask.shape, radius=args.coverage_radius_px) & support_mask
    claimed_count = int(np.count_nonzero(claimed))
    support_count = int(np.count_nonzero(support_mask))
    return ReconstructionResult(
        line_mask=line_mask,
        node_mask=node_mask,
        support_prob=support_prob,
        support_mask=support_mask,
        centreline_mask=centreline,
        components=components,
        node_blobs=node_blobs,
        raw_strokes_px=raw_strokes,
        strokes_px=strokes,
        endpoint_count=endpoint_count,
        branch_count=branch_count,
        claimed_support_pixel_count=claimed_count,
        support_coverage_fraction=claimed_count / max(support_count, 1),
        unclaimed_support_pixel_count=max(support_count - claimed_count, 0),
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


def compute_metrics(
    commands: list[Command],
    result: ReconstructionResult,
    probabilities: MLProbabilities,
    transform_info: dict,
    firmware_constants: dict,
    args: argparse.Namespace,
) -> dict:
    draw_distance, travel_distance, mode_changes = command_distances(commands)
    try:
        validate_arduino_commands(commands, args.work_width_mm, args.work_height_mm)
        bounds_valid = True
        bounds_error = None
    except ValueError as exc:
        bounds_valid = False
        bounds_error = str(exc)

    bounds = {"min_x_mm": None, "max_x_mm": None, "min_y_mm": None, "max_y_mm": None}
    if commands:
        bounds = {
            "min_x_mm": min(command[0] for command in commands),
            "max_x_mm": max(command[0] for command in commands),
            "min_y_mm": min(command[1] for command in commands),
            "max_y_mm": max(command[1] for command in commands),
        }

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

    stroke_point_counts = [len(stroke) for stroke in result.strokes_px]
    raw_point_counts = [len(stroke) for stroke in result.raw_strokes_px]
    node_classes: dict[str, int] = {}
    for blob in result.node_blobs:
        node_classes[blob.class_name] = node_classes.get(blob.class_name, 0) + 1

    warnings: list[str] = []
    if not commands:
        warnings.append("No Arduino commands were generated.")
    if np.count_nonzero(result.support_mask) == 0:
        warnings.append("Support mask is empty; thresholds are likely too strict.")
    if result.support_coverage_fraction < args.coverage_warning_fraction and np.any(result.support_mask):
        warnings.append("Recovered strokes cover a low fraction of the ML support mask.")
    if len(result.strokes_px) > args.fragment_warning_strokes:
        warnings.append("Stroke output is highly fragmented.")
    if not bounds_valid:
        warnings.append("Generated commands failed gantry bounds validation.")
    discarded = sum(1 for component in result.components if component.discarded)
    if discarded > len(result.components) * 0.5 and result.components:
        warnings.append("Most support components were discarded as too small.")

    return {
        "schema": "stroke_ml_reconstruction_metrics_v1",
        "pipeline": "ml_support_reconstruction",
        "uses_skeletonisation": bool(args.centreline_mode == "thinning"),
        "command_count": len(commands),
        "stroke_count": len(result.strokes_px),
        "average_points_per_stroke": float(np.mean(stroke_point_counts)) if stroke_point_counts else 0.0,
        "median_points_per_stroke": float(np.median(stroke_point_counts)) if stroke_point_counts else 0.0,
        "raw_stroke_count": len(result.raw_strokes_px),
        "raw_points_total": int(sum(raw_point_counts)),
        "simplified_points_total": int(sum(stroke_point_counts)),
        "pen_down_drawing_distance_mm": draw_distance,
        "pen_up_travel_distance_mm": travel_distance,
        "total_movement_distance_mm": draw_distance + travel_distance,
        "estimated_draw_movement_time_s": draw_time_s,
        "estimated_travel_movement_time_s": travel_time_s,
        "estimated_pen_mode_change_settle_time_s": pen_change_time_s,
        "estimated_total_plotting_time_s": draw_time_s + travel_time_s + pen_change_time_s,
        "mode_change_count": mode_changes,
        "command_bounds_mm": bounds,
        "bounds_validation_passed": bounds_valid,
        "bounds_validation_error": bounds_error,
        "segmentation_mode_used": "ML support reconstruction",
        "model_path_used": probabilities.model_path,
        "ml_probability_diagnostics": probabilities.diagnostics,
        "reconstruction": {
            "line_mask_pixel_count": int(np.count_nonzero(result.line_mask)),
            "node_mask_pixel_count": int(np.count_nonzero(result.node_mask)),
            "support_mask_pixel_count": int(np.count_nonzero(result.support_mask)),
            "support_connected_component_count": len(result.components),
            "reconstructed_component_count": sum(1 for component in result.components if not component.discarded),
            "discarded_component_count": discarded,
            "centreline_pixel_count": int(np.count_nonzero(result.centreline_mask)),
            "endpoint_count": result.endpoint_count,
            "branch_or_junction_count": result.branch_count,
            "node_blob_count": len(result.node_blobs),
            "node_blob_class_counts": node_classes,
            "claimed_support_pixel_count": result.claimed_support_pixel_count,
            "unclaimed_support_pixel_count": result.unclaimed_support_pixel_count,
            "recovered_support_coverage_fraction": result.support_coverage_fraction,
            "stroke_point_counts": [int(count) for count in stroke_point_counts],
        },
        "thresholds": {
            "line_threshold": args.line_threshold,
            "node_threshold": args.node_threshold,
            "support_threshold": args.support_threshold,
            "node_support_weight": args.node_support_weight,
            "node_support_radius_px": args.node_support_radius_px,
            "background_suppression_weight": args.background_suppression_weight,
            "centreline_mode": args.centreline_mode,
            "simplification_epsilon": args.simplification_epsilon,
            "min_component_pixels": args.min_component_pixels,
            "min_stroke_points": args.min_stroke_points,
            "min_stroke_length_px": args.min_stroke_length_px,
        },
        "gantry_mapping": transform_info,
        "firmware_constants": firmware_constants,
        "time_estimation_assumptions": {
            "speed_units": "DRAW_SPEED and TRAVEL_SPEED are treated as AccelStepper steps/s.",
            "steps_per_mm_formula": "(FULL_STEPS_PER_REV * MICROSTEPS) / (PULLEY_TEETH * BELT_PITCH_MM)",
            "steps_per_mm": steps_per_mm,
            "draw_speed_mm_s": draw_speed_mm_s,
            "travel_speed_mm_s": travel_speed_mm_s,
            "acceleration": "Acceleration and cornering dynamics are not modelled.",
        },
        "warnings": warnings,
    }


def mask_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L").convert("RGB")


def save_probability_debug(probabilities: MLProbabilities, result: ReconstructionResult, output_path: Path) -> None:
    gray = Image.fromarray(probabilities.gray, mode="L").convert("RGB")
    save_labeled_grid(
        [
            (gray, "grayscale"),
            (probability_image(probabilities.line_prob), "line probability"),
            (probability_image(probabilities.node_prob), "node/corner probability"),
            (probability_image(result.support_prob), "support probability"),
        ],
        output_path,
    )


def save_support_debug(result: ReconstructionResult, output_path: Path) -> None:
    save_labeled_grid(
        [
            (mask_image(result.line_mask), "line mask"),
            (mask_image(result.node_mask), "node mask"),
            (mask_image(result.support_mask), "support mask"),
            (mask_image(result.centreline_mask), "centreline"),
        ],
        output_path,
    )


def save_component_debug(probabilities: MLProbabilities, result: ReconstructionResult, output_path: Path) -> None:
    image = Image.fromarray(probabilities.gray, mode="L").convert("RGB")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    colors = [(230, 57, 70, 80), (42, 157, 143, 80), (29, 53, 87, 80), (244, 162, 97, 80), (131, 56, 236, 80)]
    for component in result.components:
        color = (120, 120, 120, 50) if component.discarded else colors[component.id % len(colors)]
        for y, x in component.pixels_yx:
            draw.point((int(x), int(y)), fill=color)
        x0, y0, x1, y1 = component.bbox
        draw.rectangle((x0, y0, x1, y1), outline=(0, 0, 0, 180), width=1)
        draw.text((x0 + 2, y0 + 2), str(component.id), fill=(0, 0, 0, 255))
    Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB").save(output_path)


def save_centreline_debug(probabilities: MLProbabilities, result: ReconstructionResult, output_path: Path) -> None:
    image = Image.fromarray(probabilities.gray, mode="L").convert("RGB")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for y, x in np.argwhere(result.centreline_mask):
        draw.point((int(x), int(y)), fill=(230, 0, 0, 220))
    degree = neighbor_count(result.centreline_mask)
    for y, x in np.argwhere(result.centreline_mask & (degree == 1)):
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(40, 120, 255, 240))
    for y, x in np.argwhere(result.centreline_mask & (degree >= 3)):
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(255, 180, 0, 240))
    Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB").save(output_path)


def save_node_blob_debug(probabilities: MLProbabilities, result: ReconstructionResult, output_path: Path) -> None:
    image = Image.fromarray(probabilities.gray, mode="L").convert("RGB")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    colors = {
        "endpoint": (40, 120, 255, 180),
        "sharp_corner": (230, 60, 60, 180),
        "smooth_bend": (30, 170, 90, 180),
        "t_junction": (245, 160, 20, 180),
        "crossing_or_overlap": (150, 70, 220, 180),
        "multi_junction": (0, 160, 180, 180),
        "line_like_node_fragment": (220, 80, 170, 180),
        "noisy_or_ambiguous": (120, 120, 120, 120),
    }
    for blob in result.node_blobs:
        color = colors.get(blob.class_name, (0, 0, 0, 150))
        for y, x in blob.points_yx:
            draw.point((int(x), int(y)), fill=color)
        cx = blob.weighted_centroid_x
        cy = blob.weighted_centroid_y
        draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=(0, 0, 0, 235))
        for dx, dy in blob.port_directions:
            draw.line((cx, cy, cx + dx * 18, cy + dy * 18), fill=(0, 0, 0, 220), width=2)
        draw.text((cx + 5, cy + 3), f"{blob.id}:{blob.class_name}", fill=(0, 0, 0, 255))
    Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB").save(output_path)


def save_stroke_sequence_debug(probabilities: MLProbabilities, result: ReconstructionResult, output_path: Path) -> None:
    image = Image.fromarray(probabilities.gray, mode="L").convert("RGB")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    colors = [(230, 57, 70, 230), (42, 157, 143, 230), (29, 53, 87, 230), (244, 162, 97, 230), (131, 56, 236, 230)]
    for i, stroke in enumerate(result.strokes_px):
        if len(stroke) < 2:
            continue
        points = [(float(x), float(y)) for x, y in stroke]
        draw.line(points, fill=colors[i % len(colors)], width=3)
        sx, sy = points[0]
        ex, ey = points[-1]
        draw.ellipse((sx - 4, sy - 4, sx + 4, sy + 4), fill=(0, 190, 70, 245), outline=(0, 0, 0, 245))
        draw.ellipse((ex - 3, ey - 3, ex + 3, ey + 3), fill=(40, 90, 255, 235), outline=(0, 0, 0, 235))
        draw.text((sx + 5, sy + 5), str(i), fill=(0, 0, 0, 255))
    Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB").save(output_path)


def run_pipeline(args: argparse.Namespace) -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    image_path = (repo_root / args.image).resolve() if not Path(args.image).is_absolute() else Path(args.image)
    model_path = (repo_root / args.model_path).resolve() if not Path(args.model_path).is_absolute() else Path(args.model_path)
    output_dir = (repo_root / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    probabilities = load_torch_probabilities(image_path, model_path)
    result = reconstruct(probabilities, args)
    paths_mm, transform_info = map_paths_to_gantry_mm(
        result.strokes_px,
        image_shape=probabilities.gray.shape,
        work_w_mm=args.work_width_mm,
        work_h_mm=args.work_height_mm,
        margin_mm=args.margin_mm,
        centre_on_page=True,
    )
    commands = paths_to_arduino_commands(paths_mm)
    validate_arduino_commands(commands, args.work_width_mm, args.work_height_mm)
    firmware_constants = parse_firmware_constants(repo_root.parent / "Arduino Code" / "ArduinoCode.ino")
    metrics = compute_metrics(commands, result, probabilities, transform_info, firmware_constants, args)

    save_arduino_commands(commands, output_dir / "arduino_commands.txt")
    save_json(metrics, output_dir / "stroke_metrics.json")
    save_probability_debug(probabilities, result, output_dir / "ml_probability_debug.png")
    save_support_debug(result, output_dir / "support_mask_debug.png")
    save_component_debug(probabilities, result, output_dir / "component_debug.png")
    save_centreline_debug(probabilities, result, output_dir / "centreline_debug.png")
    save_node_blob_debug(probabilities, result, output_dir / "node_blob_debug.png")
    save_node_blob_debug(probabilities, result, output_dir / "topology_debug.png")
    save_stroke_sequence_debug(probabilities, result, output_dir / "stroke_sequence_debug.png")
    save_gantry_preview(paths_mm, output_dir / "gantry_path_preview.png", args.work_width_mm, args.work_height_mm, args.margin_mm)

    print(f"Image: {image_path}")
    print(f"Output directory: {output_dir}")
    print(f"Model path: {model_path}")
    print("Segmentation mode used: ML support reconstruction")
    print(f"Support components: {len(result.components)}")
    print(f"Centreline pixels: {int(np.count_nonzero(result.centreline_mask))}")
    print(f"Strokes: {metrics['stroke_count']}")
    print(f"Commands: {metrics['command_count']}")
    print(f"Bounds valid: {metrics['bounds_validation_passed']}")
    for warning in metrics["warnings"]:
        print(f"WARNING: {warning}")
    print(f"Estimated total plotting time: {metrics['estimated_total_plotting_time_s']:.2f} s")
    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Clean-slate ML support reconstruction pipeline for gantry strokes.")
    parser.add_argument("--image", default=str(repo_root / "input_images" / "square.png"))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", default=str(repo_root / "output" / "ml_reconstruction"))
    parser.add_argument("--line-threshold", type=float, default=0.35)
    parser.add_argument("--node-threshold", type=float, default=0.35)
    parser.add_argument("--support-threshold", type=float, default=0.30)
    parser.add_argument("--node-support-weight", type=float, default=0.65)
    parser.add_argument("--node-support-radius-px", type=int, default=4)
    parser.add_argument("--background-suppression-weight", type=float, default=0.0)
    parser.add_argument("--min-component-pixels", type=int, default=8)
    parser.add_argument("--centreline-mode", choices=["thinning", "ridge"], default="thinning")
    parser.add_argument("--simplification-epsilon", type=float, default=1.0)
    parser.add_argument("--min-stroke-points", type=int, default=3)
    parser.add_argument("--min-stroke-length-px", type=float, default=4.0)
    parser.add_argument("--coverage-radius-px", type=int, default=2)
    parser.add_argument("--coverage-warning-fraction", type=float, default=0.55)
    parser.add_argument("--fragment-warning-strokes", type=int, default=80)
    parser.add_argument("--min-node-blob-pixels", type=int, default=3)
    parser.add_argument("--node-blob-port-radius-px", type=int, default=16)
    parser.add_argument("--node-blob-port-angle-degrees", type=float, default=25.0)
    parser.add_argument("--work-width-mm", type=float, default=150.0)
    parser.add_argument("--work-height-mm", type=float, default=270.0)
    parser.add_argument("--margin-mm", type=float, default=5.0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
