"""
Experimental ML-only graph pipeline for gantry stroke planning.

This module is intentionally separate from stroke_based_pipeline.py. The
heuristic pipeline keeps its skeletonisation path there; this file follows the
Raghav et al.-style idea more directly:

image -> ML background/line/node-corner probabilities -> corner components as
graph vertices -> line-probability edge scoring -> recursive stroke extraction
-> Arduino serial commands.

No skeletonisation is used in this ML-only graph pipeline.
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
    class_index_for,
    connected_components,
    load_grayscale,
    map_paths_to_gantry_mm,
    normalize_class_map,
    output_to_probabilities,
    parse_firmware_constants,
    paths_to_arduino_commands,
    save_arduino_commands,
    save_gantry_preview,
    save_json,
    validate_arduino_commands,
)


Command = tuple[float, float, int]


@dataclass
class MLProbabilities:
    gray: np.ndarray
    probs: np.ndarray
    background_prob: np.ndarray
    line_prob: np.ndarray
    node_prob: np.ndarray
    model_path: str
    diagnostics: dict


@dataclass
class MLGraphVertex:
    id: int
    x: float
    y: float
    area: int
    mean_probability: float
    max_probability: float
    source_component_id: int
    split_from_large_component: bool = False


@dataclass
class MLGraphEdge:
    id: int
    u: int
    v: int
    distance_px: float
    mean_line_probability: float
    max_line_probability: float
    support_fraction: float
    score: float
    accepted: bool
    reason: str
    polyline_px: np.ndarray


@dataclass
class MLGraphResult:
    vertices: list[MLGraphVertex]
    candidate_edges: list[MLGraphEdge]
    accepted_edges: list[MLGraphEdge]
    strokes_px: list[np.ndarray]
    raw_strokes_px: list[np.ndarray]
    node_mask: np.ndarray
    line_mask: np.ndarray


def load_torch_probabilities(image_path: Path, model_path: Path) -> MLProbabilities:
    import torch

    from stroke_ml_model import build_stroke_unet

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    gray = load_grayscale(image_path)
    tensor = torch.from_numpy(gray.astype(np.float32) / 255.0)[None, None, :, :]
    class_map = None

    try:
        model = torch.jit.load(str(model_path), map_location="cpu")
        model_config = {"num_classes": 3, "base_channels": None}
        checkpoint_keys = ["torchscript"]
    except Exception:
        checkpoint = torch.load(str(model_path), map_location="cpu")
        model_config = checkpoint.get("model_config", {}) if isinstance(checkpoint, dict) else {}
        class_map = checkpoint.get("class_map") if isinstance(checkpoint, dict) else None
        model = build_stroke_unet(
            num_classes=int(model_config.get("num_classes", 3)),
            base_channels=int(model_config.get("base_channels", 16)),
        )
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
            checkpoint_keys = list(checkpoint.keys())
        else:
            state_dict = checkpoint
            checkpoint_keys = ["state_dict"]
        model.load_state_dict(state_dict)

    model.eval()
    with torch.no_grad():
        output = model(tensor)
        output_np = output.detach().cpu().numpy()
        probs = output_to_probabilities(output_np)

    normalized_class_map = normalize_class_map(class_map)
    background_class = class_index_for(normalized_class_map, ("background",), default=0)
    line_class = class_index_for(normalized_class_map, ("line",), default=1)
    node_class = class_index_for(normalized_class_map, ("node", "corner"), default=2)
    if probs.shape[0] <= max(background_class, line_class, node_class):
        raise ValueError(f"Model output has {probs.shape[0]} channels; expected classes {normalized_class_map}")

    diagnostics = {
        "model_path": str(model_path),
        "output_shape": list(output_np.shape),
        "probability_shape": list(probs.shape),
        "class_map": normalized_class_map,
        "background_class_index": background_class,
        "line_class_index": line_class,
        "node_class_index": node_class,
        "checkpoint_keys": checkpoint_keys,
        "model_config": model_config,
        "background_probability_mean": float(np.mean(probs[background_class])),
        "line_probability_mean": float(np.mean(probs[line_class])),
        "line_probability_max": float(np.max(probs[line_class])),
        "node_probability_mean": float(np.mean(probs[node_class])),
        "node_probability_max": float(np.max(probs[node_class])),
    }
    return MLProbabilities(
        gray=gray,
        probs=probs,
        background_prob=probs[background_class],
        line_prob=probs[line_class],
        node_prob=probs[node_class],
        model_path=str(model_path),
        diagnostics=diagnostics,
    )


def weighted_centroid(points_yx: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    values = probability[points_yx[:, 0], points_yx[:, 1]].astype(np.float64)
    total = float(np.sum(values))
    if total <= 1e-9:
        y, x = np.mean(points_yx, axis=0)
    else:
        y = float(np.sum(points_yx[:, 0] * values) / total)
        x = float(np.sum(points_yx[:, 1] * values) / total)
    return x, y


def split_large_component_peaks(
    points_yx: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    min_separation_px: float,
    max_vertices: int,
) -> list[tuple[float, float, int, float, float]]:
    values = probability[points_yx[:, 0], points_yx[:, 1]]
    order = np.argsort(values)[::-1]
    selected: list[tuple[float, float, int, float, float]] = []
    for idx in order:
        y = float(points_yx[idx, 0])
        x = float(points_yx[idx, 1])
        value = float(values[idx])
        if value < threshold:
            break
        if any(math.hypot(x - sx, y - sy) < min_separation_px for sx, sy, _, _, _ in selected):
            continue
        selected.append((x, y, 1, value, value))
        if len(selected) >= max_vertices:
            break
    return selected


def extract_vertices_from_nodes(
    node_prob: np.ndarray,
    threshold: float,
    min_area: int,
    min_separation_px: float,
    large_component_area: int,
    max_large_component_vertices: int,
    max_vertices: int,
) -> tuple[list[MLGraphVertex], np.ndarray]:
    node_mask = node_prob >= threshold
    vertices: list[MLGraphVertex] = []
    component_id = 0
    for component in connected_components(node_mask, connectivity=8):
        points_yx = np.array(component, dtype=np.int32)
        area = int(len(points_yx))
        if area < min_area:
            continue
        values = node_prob[points_yx[:, 0], points_yx[:, 1]]
        if area >= large_component_area:
            candidates = split_large_component_peaks(
                points_yx,
                node_prob,
                threshold=threshold,
                min_separation_px=min_separation_px,
                max_vertices=max_large_component_vertices,
            )
            for x, y, peak_area, mean_probability, max_probability in candidates:
                vertices.append(
                    MLGraphVertex(
                        id=len(vertices),
                        x=float(x),
                        y=float(y),
                        area=peak_area,
                        mean_probability=float(mean_probability),
                        max_probability=float(max_probability),
                        source_component_id=component_id,
                        split_from_large_component=True,
                    )
                )
        else:
            x, y = weighted_centroid(points_yx, node_prob)
            vertices.append(
                MLGraphVertex(
                    id=len(vertices),
                    x=float(x),
                    y=float(y),
                    area=area,
                    mean_probability=float(np.mean(values)),
                    max_probability=float(np.max(values)),
                    source_component_id=component_id,
                )
            )
        component_id += 1

    vertices.sort(key=lambda vertex: vertex.max_probability, reverse=True)
    vertices = vertices[:max_vertices]
    for i, vertex in enumerate(vertices):
        vertex.id = i
    vertices.sort(key=lambda vertex: (vertex.y, vertex.x))
    for i, vertex in enumerate(vertices):
        vertex.id = i
    return vertices, node_mask


def sample_edge_polyline(p0: tuple[float, float], p1: tuple[float, float], step_px: float) -> np.ndarray:
    x0, y0 = p0
    x1, y1 = p1
    distance = math.hypot(x1 - x0, y1 - y0)
    n = max(2, int(math.ceil(distance / max(step_px, 1e-6))) + 1)
    ts = np.linspace(0.0, 1.0, n)
    xs = x0 + (x1 - x0) * ts
    ys = y0 + (y1 - y0) * ts
    return np.column_stack([xs, ys])


def edge_probability_samples(
    line_prob: np.ndarray,
    p0: tuple[float, float],
    p1: tuple[float, float],
    corridor_radius_px: int,
    step_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    centerline = sample_edge_polyline(p0, p1, step_px=step_px)
    x0, y0 = p0
    x1, y1 = p1
    dx = x1 - x0
    dy = y1 - y0
    length = math.hypot(dx, dy)
    if length <= 1e-9:
        return np.array([], dtype=np.float32), centerline
    nx = -dy / length
    ny = dx / length
    height, width = line_prob.shape
    samples = []
    seen: set[tuple[int, int]] = set()
    for x, y in centerline:
        for offset in range(-corridor_radius_px, corridor_radius_px + 1):
            sx = int(round(x + nx * offset))
            sy = int(round(y + ny * offset))
            if 0 <= sx < width and 0 <= sy < height and (sy, sx) not in seen:
                samples.append(float(line_prob[sy, sx]))
                seen.add((sy, sx))
    return np.asarray(samples, dtype=np.float32), centerline


def build_candidate_edges(
    vertices: list[MLGraphVertex],
    line_prob: np.ndarray,
    max_distance_px: float,
    nearest_neighbors: int,
    edge_score_threshold: float,
    line_threshold: float,
    support_fraction_threshold: float,
    corridor_radius_px: int,
    densify_step_px: float,
) -> tuple[list[MLGraphEdge], list[MLGraphEdge]]:
    pairs: set[tuple[int, int]] = set()
    coords = np.array([[vertex.x, vertex.y] for vertex in vertices], dtype=np.float32)
    for i, vertex in enumerate(vertices):
        if len(vertices) <= 1:
            continue
        distances = np.linalg.norm(coords - np.array([vertex.x, vertex.y], dtype=np.float32), axis=1)
        order = np.argsort(distances)
        added = 0
        for j in order:
            j = int(j)
            if j == i:
                continue
            distance = float(distances[j])
            if distance > max_distance_px:
                continue
            pairs.add((min(i, j), max(i, j)))
            added += 1
            if added >= nearest_neighbors:
                break

    candidate_edges: list[MLGraphEdge] = []
    accepted_edges: list[MLGraphEdge] = []
    for u, v in sorted(pairs):
        p0 = (vertices[u].x, vertices[u].y)
        p1 = (vertices[v].x, vertices[v].y)
        samples, polyline = edge_probability_samples(
            line_prob,
            p0,
            p1,
            corridor_radius_px=corridor_radius_px,
            step_px=densify_step_px,
        )
        if len(samples) == 0:
            mean_prob = 0.0
            max_prob = 0.0
            support_fraction = 0.0
        else:
            mean_prob = float(np.mean(samples))
            max_prob = float(np.max(samples))
            support_fraction = float(np.count_nonzero(samples >= line_threshold) / len(samples))
        score = 0.65 * mean_prob + 0.35 * support_fraction
        accepted = bool(score >= edge_score_threshold and support_fraction >= support_fraction_threshold)
        reason = "accepted" if accepted else "score/support below threshold"
        edge = MLGraphEdge(
            id=len(candidate_edges),
            u=u,
            v=v,
            distance_px=float(math.hypot(p1[0] - p0[0], p1[1] - p0[1])),
            mean_line_probability=mean_prob,
            max_line_probability=max_prob,
            support_fraction=support_fraction,
            score=score,
            accepted=accepted,
            reason=reason,
            polyline_px=polyline,
        )
        candidate_edges.append(edge)
        if accepted:
            edge.id = len(accepted_edges)
            accepted_edges.append(edge)
    return candidate_edges, accepted_edges


def orient_edge(edge: MLGraphEdge, from_vertex: int) -> np.ndarray:
    if edge.u == from_vertex:
        return edge.polyline_px
    return edge.polyline_px[::-1].copy()


def extract_recursive_strokes(vertices: list[MLGraphVertex], edges: list[MLGraphEdge]) -> list[np.ndarray]:
    adjacency: dict[int, set[int]] = {vertex.id: set() for vertex in vertices}
    edge_by_id = {edge.id: edge for edge in edges}
    for edge in edges:
        adjacency.setdefault(edge.u, set()).add(edge.id)
        adjacency.setdefault(edge.v, set()).add(edge.id)

    unused = set(edge_by_id.keys())
    strokes: list[np.ndarray] = []
    while unused:
        odd_vertices = [
            vertex_id
            for vertex_id, edge_ids in adjacency.items()
            if len(edge_ids & unused) % 2 == 1 and len(edge_ids & unused) > 0
        ]
        if odd_vertices:
            start = min(odd_vertices, key=lambda vertex_id: vertices[vertex_id].y)
        else:
            start = min(
                (vertex_id for vertex_id, edge_ids in adjacency.items() if edge_ids & unused),
                key=lambda vertex_id: vertices[vertex_id].y,
            )

        current = start
        stroke_parts: list[np.ndarray] = []
        while True:
            available = list(adjacency.get(current, set()) & unused)
            if not available:
                break
            available.sort(key=lambda edge_id: edge_by_id[edge_id].score, reverse=True)
            edge = edge_by_id[available[0]]
            unused.remove(edge.id)
            oriented = orient_edge(edge, current)
            stroke_parts.append(oriented if not stroke_parts else oriented[1:])
            current = edge.v if current == edge.u else edge.u
        if stroke_parts:
            stroke = np.vstack(stroke_parts)
            if len(stroke) >= 2:
                strokes.append(stroke)
    return strokes


def build_ml_graph(probabilities: MLProbabilities, args: argparse.Namespace) -> MLGraphResult:
    vertices, node_mask = extract_vertices_from_nodes(
        probabilities.node_prob,
        threshold=args.node_threshold,
        min_area=args.min_node_area,
        min_separation_px=args.min_vertex_separation_px,
        large_component_area=args.large_node_component_area,
        max_large_component_vertices=args.max_large_component_vertices,
        max_vertices=args.max_vertices,
    )
    line_mask = probabilities.line_prob >= args.line_threshold
    candidate_edges, accepted_edges = build_candidate_edges(
        vertices,
        probabilities.line_prob,
        max_distance_px=args.max_edge_distance_px,
        nearest_neighbors=args.nearest_neighbors,
        edge_score_threshold=args.edge_score_threshold,
        line_threshold=args.line_threshold,
        support_fraction_threshold=args.support_fraction_threshold,
        corridor_radius_px=args.edge_corridor_radius_px,
        densify_step_px=args.densify_step_px,
    )
    strokes = extract_recursive_strokes(vertices, accepted_edges)
    return MLGraphResult(
        vertices=vertices,
        candidate_edges=candidate_edges,
        accepted_edges=accepted_edges,
        strokes_px=strokes,
        raw_strokes_px=[stroke.copy() for stroke in strokes],
        node_mask=node_mask,
        line_mask=line_mask,
    )


def path_closed(path: np.ndarray, tolerance: float = 2.0) -> bool:
    if len(path) < 3:
        return False
    return float(np.linalg.norm(path[0] - path[-1])) <= tolerance


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
    graph: MLGraphResult,
    probabilities: MLProbabilities,
    transform_info: dict,
    firmware_constants: dict,
    args: argparse.Namespace,
) -> dict:
    draw_distance, travel_distance, mode_changes = command_distances(commands)
    try:
        validate_arduino_commands(commands, args.work_width_mm, args.work_height_mm)
        bounds_pass = True
        bounds_error = None
    except ValueError as exc:
        bounds_pass = False
        bounds_error = str(exc)

    if commands:
        xs = [command[0] for command in commands]
        ys = [command[1] for command in commands]
        bounds = {
            "min_x_mm": min(xs),
            "max_x_mm": max(xs),
            "min_y_mm": min(ys),
            "max_y_mm": max(ys),
        }
    else:
        bounds = {"min_x_mm": None, "max_x_mm": None, "min_y_mm": None, "max_y_mm": None}

    steps_per_mm = (
        firmware_constants["FULL_STEPS_PER_REV"]
        * firmware_constants["MICROSTEPS"]
        / (firmware_constants["PULLEY_TEETH"] * firmware_constants["BELT_PITCH_MM"])
    )
    draw_speed_mm_s = firmware_constants["DRAW_SPEED"] / steps_per_mm
    travel_speed_mm_s = firmware_constants["TRAVEL_SPEED"] / steps_per_mm
    draw_time_s = draw_distance / draw_speed_mm_s if draw_speed_mm_s > 0 else None
    travel_time_s = travel_distance / travel_speed_mm_s if travel_speed_mm_s > 0 else None
    servo_sweep_ms = abs(firmware_constants["PEN_UP_ANGLE"] - firmware_constants["PEN_DOWN_ANGLE"]) * firmware_constants["SERVO_DELAY_MS"]
    pen_change_time_s = mode_changes * (firmware_constants["PEN_SETTLE_MS"] + servo_sweep_ms) / 1000.0
    edge_scores = [edge.score for edge in graph.accepted_edges]
    candidate_scores = [edge.score for edge in graph.candidate_edges]
    stroke_point_counts = [len(stroke) for stroke in graph.strokes_px]
    warnings = []
    if not graph.vertices:
        warnings.append("No ML node/corner components became graph vertices.")
    if not graph.accepted_edges:
        warnings.append("No ML graph edges passed the line-probability edge thresholds.")
    if graph.vertices and len(graph.accepted_edges) < max(1, len(graph.vertices) // 3):
        warnings.append("Accepted edge count is low relative to vertex count; edge thresholds may be too strict or line probabilities too weak.")
    if np.count_nonzero(graph.node_mask) > np.count_nonzero(graph.line_mask) * 2:
        warnings.append("Node/corner mask is much denser than line mask; the checkpoint may over-predict node/corner.")

    return {
        "schema": "stroke_ml_graph_metrics_v1",
        "pipeline": "ml_graph_no_skeleton",
        "uses_skeletonisation": False,
        "command_count": len(commands),
        "stroke_count": len(graph.strokes_px),
        "average_points_per_stroke": float(np.mean(stroke_point_counts)) if stroke_point_counts else 0.0,
        "median_points_per_stroke": float(np.median(stroke_point_counts)) if stroke_point_counts else 0.0,
        "closed_loop_stroke_count": sum(1 for stroke in graph.strokes_px if path_closed(stroke)),
        "pen_down_drawing_distance_mm": draw_distance,
        "pen_up_travel_distance_mm": travel_distance,
        "total_movement_distance_mm": draw_distance + travel_distance,
        "estimated_draw_movement_time_s": draw_time_s,
        "estimated_travel_movement_time_s": travel_time_s,
        "estimated_pen_mode_change_settle_time_s": pen_change_time_s,
        "estimated_total_plotting_time_s": (draw_time_s or 0.0) + (travel_time_s or 0.0) + pen_change_time_s,
        "mode_change_count": mode_changes,
        "command_bounds_mm": bounds,
        "bounds_validation_passed": bounds_pass,
        "bounds_validation_error": bounds_error,
        "segmentation_mode_used": "ML graph no skeleton",
        "model_path_used": probabilities.model_path,
        "ml_probability_diagnostics": probabilities.diagnostics,
        "graph": {
            "vertex_count": len(graph.vertices),
            "node_component_vertex_count": len(graph.vertices),
            "candidate_edge_count": len(graph.candidate_edges),
            "accepted_edge_count": len(graph.accepted_edges),
            "rejected_edge_count": len(graph.candidate_edges) - len(graph.accepted_edges),
            "node_mask_pixel_count": int(np.count_nonzero(graph.node_mask)),
            "line_mask_pixel_count": int(np.count_nonzero(graph.line_mask)),
            "line_to_node_pixel_ratio": float(np.count_nonzero(graph.line_mask) / max(np.count_nonzero(graph.node_mask), 1)),
            "edge_score_mean": float(np.mean(edge_scores)) if edge_scores else 0.0,
            "edge_score_max": float(np.max(edge_scores)) if edge_scores else 0.0,
            "candidate_edge_score_mean": float(np.mean(candidate_scores)) if candidate_scores else 0.0,
            "stroke_point_counts": [int(count) for count in stroke_point_counts],
        },
        "thresholds": {
            "node_threshold": args.node_threshold,
            "line_threshold": args.line_threshold,
            "edge_score_threshold": args.edge_score_threshold,
            "support_fraction_threshold": args.support_fraction_threshold,
            "max_edge_distance_px": args.max_edge_distance_px,
            "nearest_neighbors": args.nearest_neighbors,
            "edge_corridor_radius_px": args.edge_corridor_radius_px,
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


def probability_image(prob: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(prob * 255.0, 0, 255).astype(np.uint8), mode="L").convert("RGB")


def mask_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L").convert("RGB")


def save_labeled_grid(items: list[tuple[Image.Image, str]], output_path: Path) -> None:
    padding = 20
    label_h = 24
    thumb_w = max(image.width for image, _ in items)
    thumb_h = max(image.height for image, _ in items)
    canvas = Image.new("RGB", (len(items) * thumb_w + (len(items) + 1) * padding, thumb_h + label_h + 2 * padding), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (image, label) in enumerate(items):
        x = padding + i * (thumb_w + padding)
        y = padding + label_h
        canvas.paste(image.convert("RGB"), (x, y))
        draw.text((x, padding), label, fill=(20, 20, 20))
    canvas.save(output_path)


def save_probability_debug(probabilities: MLProbabilities, graph: MLGraphResult, output_path: Path) -> None:
    gray = Image.fromarray(probabilities.gray, mode="L").convert("RGB")
    save_labeled_grid(
        [
            (gray, "grayscale"),
            (probability_image(probabilities.line_prob), "line probability"),
            (probability_image(probabilities.node_prob), "node/corner probability"),
            (mask_image(graph.line_mask), "line threshold mask"),
            (mask_image(graph.node_mask), "node threshold mask"),
        ],
        output_path,
    )


def save_node_components_debug(probabilities: MLProbabilities, graph: MLGraphResult, output_path: Path) -> None:
    image = Image.fromarray(probabilities.gray, mode="L").convert("RGB")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    node_pixels = np.argwhere(graph.node_mask)
    for y, x in node_pixels:
        draw.point((int(x), int(y)), fill=(255, 0, 180, 160))
    for vertex in graph.vertices:
        r = 5 if vertex.split_from_large_component else 4
        fill = (0, 190, 70, 235) if not vertex.split_from_large_component else (255, 170, 0, 235)
        draw.ellipse((vertex.x - r, vertex.y - r, vertex.x + r, vertex.y + r), fill=fill, outline=(0, 0, 0, 235))
        draw.text((vertex.x + 6, vertex.y + 3), str(vertex.id), fill=(0, 0, 0, 255))
    Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB").save(output_path)


def save_edge_debug(probabilities: MLProbabilities, graph: MLGraphResult, output_path: Path) -> None:
    image = Image.fromarray(probabilities.gray, mode="L").convert("RGB")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for edge in graph.candidate_edges:
        points = [(float(x), float(y)) for x, y in edge.polyline_px]
        color = (30, 150, 90, 220) if edge.accepted else (220, 60, 60, 90)
        width = 2 if edge.accepted else 1
        if len(points) >= 2:
            draw.line(points, fill=color, width=width)
    for vertex in graph.vertices:
        draw.ellipse((vertex.x - 4, vertex.y - 4, vertex.x + 4, vertex.y + 4), fill=(255, 230, 0, 235), outline=(0, 0, 0, 235))
        draw.text((vertex.x + 5, vertex.y + 3), str(vertex.id), fill=(0, 0, 0, 255))
    Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB").save(output_path)


def save_strokes_debug(probabilities: MLProbabilities, graph: MLGraphResult, output_path: Path) -> None:
    image = Image.fromarray(probabilities.gray, mode="L").convert("RGB")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    colors = [
        (230, 57, 70, 225),
        (42, 157, 143, 225),
        (29, 53, 87, 225),
        (244, 162, 97, 225),
        (131, 56, 236, 225),
        (0, 119, 182, 225),
    ]
    for i, stroke in enumerate(graph.strokes_px):
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
    graph = build_ml_graph(probabilities, args)
    paths_mm, transform_info = map_paths_to_gantry_mm(
        graph.strokes_px,
        image_shape=probabilities.gray.shape,
        work_w_mm=args.work_width_mm,
        work_h_mm=args.work_height_mm,
        margin_mm=args.margin_mm,
        centre_on_page=True,
    )
    commands = paths_to_arduino_commands(paths_mm)
    validate_arduino_commands(commands, args.work_width_mm, args.work_height_mm)
    firmware_constants = parse_firmware_constants(repo_root.parent / "Arduino Code" / "ArduinoCode.ino")
    metrics = compute_metrics(commands, graph, probabilities, transform_info, firmware_constants, args)

    save_arduino_commands(commands, output_dir / "arduino_commands.txt")
    save_json(metrics, output_dir / "stroke_metrics.json")
    save_probability_debug(probabilities, graph, output_dir / "ml_probability_debug.png")
    save_node_components_debug(probabilities, graph, output_dir / "node_components_debug.png")
    save_edge_debug(probabilities, graph, output_dir / "candidate_edges_debug.png")
    save_edge_debug(probabilities, graph, output_dir / "graph_debug.png")
    save_strokes_debug(probabilities, graph, output_dir / "stroke_sequence_debug.png")
    save_gantry_preview(paths_mm, output_dir / "gantry_path_preview.png", args.work_width_mm, args.work_height_mm, args.margin_mm)

    print(f"Image: {image_path}")
    print(f"Output directory: {output_dir}")
    print(f"Model path: {model_path}")
    print("Segmentation mode used: ML graph no skeleton")
    print(f"Vertices: {len(graph.vertices)}")
    print(f"Candidate/accepted edges: {len(graph.candidate_edges)} / {len(graph.accepted_edges)}")
    print(f"Strokes: {metrics['stroke_count']}")
    print(f"Commands: {metrics['command_count']}")
    print(f"Bounds valid: {metrics['bounds_validation_passed']}")
    for warning in metrics["warnings"]:
        print(f"WARNING: {warning}")
    print(f"Estimated total plotting time: {metrics['estimated_total_plotting_time_s']:.2f} s")
    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="ML-only Raghav-style graph stroke pipeline without skeletonisation.")
    parser.add_argument("--image", default=str(repo_root / "input_images" / "square.png"))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", default=str(repo_root / "output" / "stroke_ml_graph"))
    parser.add_argument("--node-threshold", type=float, default=0.45)
    parser.add_argument("--line-threshold", type=float, default=0.35)
    parser.add_argument("--edge-score-threshold", type=float, default=0.25)
    parser.add_argument("--support-fraction-threshold", type=float, default=0.20)
    parser.add_argument("--max-edge-distance-px", type=float, default=180.0)
    parser.add_argument("--nearest-neighbors", type=int, default=8)
    parser.add_argument("--edge-corridor-radius-px", type=int, default=2)
    parser.add_argument("--densify-step-px", type=float, default=3.0)
    parser.add_argument("--min-node-area", type=int, default=2)
    parser.add_argument("--large-node-component-area", type=int, default=80)
    parser.add_argument("--max-large-component-vertices", type=int, default=12)
    parser.add_argument("--min-vertex-separation-px", type=float, default=14.0)
    parser.add_argument("--max-vertices", type=int, default=120)
    parser.add_argument("--work-width-mm", type=float, default=150.0)
    parser.add_argument("--work-height-mm", type=float, default=270.0)
    parser.add_argument("--margin-mm", type=float, default=5.0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
