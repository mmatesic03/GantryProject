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
import heapq
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
    straight_polyline_px: np.ndarray
    path_length_px: float
    path_length_ratio: float
    min_line_probability: float
    total_path_cost: float
    search_mode: str
    accepted_before_pruning: bool


@dataclass
class MLGraphResult:
    vertices: list[MLGraphVertex]
    candidate_edges: list[MLGraphEdge]
    accepted_edges: list[MLGraphEdge]
    strokes_px: list[np.ndarray]
    raw_strokes_px: list[np.ndarray]
    node_mask: np.ndarray
    line_mask: np.ndarray
    accepted_edges_before_pruning: list[MLGraphEdge]
    rejected_by_path_score_count: int
    rejected_by_degree_pruning_count: int


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


def path_length(polyline: np.ndarray) -> float:
    if len(polyline) < 2:
        return 0.0
    deltas = np.diff(polyline.astype(np.float32), axis=0)
    return float(np.sum(np.linalg.norm(deltas, axis=1)))


def line_path_cost(probability: float) -> float:
    return 0.05 + (1.0 - float(probability)) ** 2


def astar_probability_path(
    line_prob: np.ndarray,
    p0: tuple[float, float],
    p1: tuple[float, float],
    margin_px: int,
    max_path_to_straight_ratio: float,
    max_expanded_nodes: int,
) -> tuple[np.ndarray, float, bool]:
    height, width = line_prob.shape
    start_x = int(round(p0[0]))
    start_y = int(round(p0[1]))
    goal_x = int(round(p1[0]))
    goal_y = int(round(p1[1]))
    if not (0 <= start_x < width and 0 <= start_y < height and 0 <= goal_x < width and 0 <= goal_y < height):
        return sample_edge_polyline(p0, p1, step_px=1.0), float("inf"), False

    straight = math.hypot(goal_x - start_x, goal_y - start_y)
    if straight <= 1e-9:
        return np.array([[start_x, start_y]], dtype=np.float32), 0.0, True

    min_x = max(0, min(start_x, goal_x) - margin_px)
    max_x = min(width - 1, max(start_x, goal_x) + margin_px)
    min_y = max(0, min(start_y, goal_y) - margin_px)
    max_y = min(height - 1, max(start_y, goal_y) + margin_px)
    max_allowed_cost = straight * max_path_to_straight_ratio * 1.75

    def heuristic(x: int, y: int) -> float:
        return 0.05 * math.hypot(goal_x - x, goal_y - y)

    start = (start_y, start_x)
    goal = (goal_y, goal_x)
    queue: list[tuple[float, float, tuple[int, int]]] = [(heuristic(start_x, start_y), 0.0, start)]
    best_cost: dict[tuple[int, int], float] = {start: 0.0}
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    directions = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (1, 1, math.sqrt(2.0)),
    ]

    found = False
    final_cost = float("inf")
    expanded_nodes = 0
    while queue:
        _, cost, current = heapq.heappop(queue)
        if cost != best_cost.get(current, float("inf")):
            continue
        expanded_nodes += 1
        if expanded_nodes > max_expanded_nodes:
            break
        if current == goal:
            found = True
            final_cost = cost
            break
        if cost > max_allowed_cost:
            continue
        cy, cx = current
        for dy, dx, step_distance in directions:
            ny = cy + dy
            nx = cx + dx
            if nx < min_x or nx > max_x or ny < min_y or ny > max_y:
                continue
            move_cost = line_path_cost(float(line_prob[ny, nx])) * step_distance
            new_cost = cost + move_cost
            if new_cost < best_cost.get((ny, nx), float("inf")):
                best_cost[(ny, nx)] = new_cost
                parent[(ny, nx)] = current
                heapq.heappush(queue, (new_cost + heuristic(nx, ny), new_cost, (ny, nx)))

    if not found:
        return sample_edge_polyline(p0, p1, step_px=1.0), float("inf"), False

    path_yx = [goal]
    current = goal
    while current != start:
        current = parent[current]
        path_yx.append(current)
    path_yx.reverse()
    path_xy = np.array([[float(x), float(y)] for y, x in path_yx], dtype=np.float32)
    return path_xy, final_cost, True


def score_edge_path(
    line_prob: np.ndarray,
    p0: tuple[float, float],
    p1: tuple[float, float],
    args: argparse.Namespace,
) -> dict:
    straight_polyline = sample_edge_polyline(p0, p1, step_px=args.densify_step_px)
    straight_samples, _ = edge_probability_samples(
        line_prob,
        p0,
        p1,
        corridor_radius_px=args.edge_corridor_radius_px,
        step_px=args.densify_step_px,
    )
    straight_distance = float(math.hypot(p1[0] - p0[0], p1[1] - p0[1]))
    straight_score = 0.0
    if len(straight_samples) > 0:
        straight_support = float(np.count_nonzero(straight_samples >= args.line_threshold) / len(straight_samples))
        straight_score = 0.65 * float(np.mean(straight_samples)) + 0.35 * straight_support

    use_path = args.edge_search_mode in {"path", "hybrid"}
    path_found = False
    total_path_cost = float("inf")
    if use_path:
        path_polyline, total_path_cost, path_found = astar_probability_path(
            line_prob,
            p0,
            p1,
            margin_px=args.path_corridor_margin_px,
            max_path_to_straight_ratio=args.max_path_to_straight_ratio,
            max_expanded_nodes=args.path_max_expanded_nodes,
        )
    else:
        path_polyline = straight_polyline
        total_path_cost = straight_distance
        path_found = True

    if args.edge_search_mode == "hybrid" and (not path_found):
        path_polyline = straight_polyline
        total_path_cost = straight_distance
        path_found = True

    pixels = np.rint(path_polyline).astype(np.int32)
    pixels[:, 0] = np.clip(pixels[:, 0], 0, line_prob.shape[1] - 1)
    pixels[:, 1] = np.clip(pixels[:, 1], 0, line_prob.shape[0] - 1)
    values = line_prob[pixels[:, 1], pixels[:, 0]].astype(np.float32)
    mean_prob = float(np.mean(values)) if len(values) else 0.0
    max_prob = float(np.max(values)) if len(values) else 0.0
    min_prob = float(np.min(values)) if len(values) else 0.0
    support_fraction = float(np.count_nonzero(values >= args.line_threshold) / max(len(values), 1))
    length_px = path_length(path_polyline)
    length_ratio = length_px / max(straight_distance, 1e-6)
    path_score = 0.55 * mean_prob + 0.45 * support_fraction
    if args.edge_search_mode == "hybrid":
        score = max(path_score, straight_score)
    elif args.edge_search_mode == "straight":
        score = straight_score
    else:
        score = path_score

    return {
        "polyline": path_polyline,
        "straight_polyline": straight_polyline,
        "mean_prob": mean_prob,
        "max_prob": max_prob,
        "min_prob": min_prob,
        "support_fraction": support_fraction,
        "score": float(score),
        "length_px": length_px,
        "length_ratio": length_ratio,
        "total_path_cost": float(total_path_cost),
        "path_found": path_found or args.edge_search_mode == "straight",
    }


def prune_edges_by_degree(edges: list[MLGraphEdge], vertices: list[MLGraphVertex], max_degree: int) -> tuple[list[MLGraphEdge], int]:
    if max_degree <= 0:
        for i, edge in enumerate(edges):
            edge.id = i
            edge.accepted = True
            edge.reason = "accepted"
        return edges, 0

    degree = {vertex.id: 0 for vertex in vertices}
    accepted: list[MLGraphEdge] = []
    rejected = 0
    for edge in sorted(edges, key=lambda item: (item.score, item.support_fraction, -item.distance_px), reverse=True):
        if degree.get(edge.u, 0) >= max_degree or degree.get(edge.v, 0) >= max_degree:
            edge.accepted = False
            edge.reason = "rejected by degree pruning"
            rejected += 1
            continue
        degree[edge.u] = degree.get(edge.u, 0) + 1
        degree[edge.v] = degree.get(edge.v, 0) + 1
        edge.accepted = True
        edge.reason = "accepted"
        edge.id = len(accepted)
        accepted.append(edge)
    return accepted, rejected


def build_candidate_edges(
    vertices: list[MLGraphVertex],
    line_prob: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[MLGraphEdge], list[MLGraphEdge], list[MLGraphEdge], int, int]:
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
            if distance > args.max_edge_distance_px:
                continue
            pairs.add((min(i, j), max(i, j)))
            added += 1
            if added >= args.nearest_neighbors:
                break

    candidate_edges: list[MLGraphEdge] = []
    accepted_before_pruning: list[MLGraphEdge] = []
    rejected_by_path_score = 0
    for u, v in sorted(pairs):
        p0 = (vertices[u].x, vertices[u].y)
        p1 = (vertices[v].x, vertices[v].y)
        path_metrics = score_edge_path(line_prob, p0, p1, args)
        accepted = bool(
            path_metrics["path_found"]
            and path_metrics["score"] >= args.edge_score_threshold
            and path_metrics["support_fraction"] >= args.path_min_support_fraction
            and path_metrics["support_fraction"] >= args.support_fraction_threshold
            and path_metrics["length_ratio"] <= args.max_path_to_straight_ratio
            and path_metrics["total_path_cost"] <= args.path_cost_threshold
        )
        reason = "accepted" if accepted else "score/support below threshold"
        if not path_metrics["path_found"]:
            reason = "no low-cost probability path found"
        elif path_metrics["length_ratio"] > args.max_path_to_straight_ratio:
            reason = "path too long relative to straight distance"
        elif path_metrics["total_path_cost"] > args.path_cost_threshold:
            reason = "path cost above threshold"
        if not accepted:
            rejected_by_path_score += 1
        edge = MLGraphEdge(
            id=len(candidate_edges),
            u=u,
            v=v,
            distance_px=float(math.hypot(p1[0] - p0[0], p1[1] - p0[1])),
            mean_line_probability=path_metrics["mean_prob"],
            max_line_probability=path_metrics["max_prob"],
            support_fraction=path_metrics["support_fraction"],
            score=path_metrics["score"],
            accepted=accepted,
            reason=reason,
            polyline_px=path_metrics["polyline"],
            straight_polyline_px=path_metrics["straight_polyline"],
            path_length_px=path_metrics["length_px"],
            path_length_ratio=path_metrics["length_ratio"],
            min_line_probability=path_metrics["min_prob"],
            total_path_cost=path_metrics["total_path_cost"],
            search_mode=args.edge_search_mode,
            accepted_before_pruning=accepted,
        )
        candidate_edges.append(edge)
        if accepted:
            accepted_before_pruning.append(edge)
    accepted_edges, rejected_by_degree = prune_edges_by_degree(accepted_before_pruning, vertices, args.max_degree)
    return candidate_edges, accepted_before_pruning, accepted_edges, rejected_by_path_score, rejected_by_degree


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
        previous_direction: np.ndarray | None = None
        while True:
            available = list(adjacency.get(current, set()) & unused)
            if not available:
                break
            if previous_direction is None:
                available.sort(key=lambda edge_id: edge_by_id[edge_id].score, reverse=True)
            else:
                scored_edges = []
                for edge_id in available:
                    candidate = orient_edge(edge_by_id[edge_id], current)
                    if len(candidate) < 2:
                        angle_score = -1.0
                    else:
                        direction = candidate[min(3, len(candidate) - 1)] - candidate[0]
                        norm = float(np.linalg.norm(direction))
                        if norm <= 1e-9:
                            angle_score = -1.0
                        else:
                            unit = direction / norm
                            angle_score = float(np.dot(previous_direction, unit))
                    scored_edges.append((angle_score, edge_by_id[edge_id].score, edge_id))
                scored_edges.sort(reverse=True)
                available = [edge_id for _, _, edge_id in scored_edges]
            edge = edge_by_id[available[0]]
            unused.remove(edge.id)
            oriented = orient_edge(edge, current)
            stroke_parts.append(oriented if not stroke_parts else oriented[1:])
            if len(oriented) >= 2:
                direction = oriented[-1] - oriented[max(0, len(oriented) - 4)]
                norm = float(np.linalg.norm(direction))
                previous_direction = direction / norm if norm > 1e-9 else previous_direction
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
    candidate_edges, accepted_before_pruning, accepted_edges, rejected_by_path_score, rejected_by_degree = build_candidate_edges(
        vertices,
        probabilities.line_prob,
        args,
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
        accepted_edges_before_pruning=accepted_before_pruning,
        rejected_by_path_score_count=rejected_by_path_score,
        rejected_by_degree_pruning_count=rejected_by_degree,
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
    pre_prune_scores = [edge.score for edge in graph.accepted_edges_before_pruning]
    candidate_scores = [edge.score for edge in graph.candidate_edges]
    path_ratios = [edge.path_length_ratio for edge in graph.accepted_edges if math.isfinite(edge.path_length_ratio)]
    candidate_path_ratios = [edge.path_length_ratio for edge in graph.candidate_edges if math.isfinite(edge.path_length_ratio)]
    stroke_point_counts = [len(stroke) for stroke in graph.strokes_px]
    vertex_degree = {vertex.id: 0 for vertex in graph.vertices}
    for edge in graph.accepted_edges:
        vertex_degree[edge.u] = vertex_degree.get(edge.u, 0) + 1
        vertex_degree[edge.v] = vertex_degree.get(edge.v, 0) + 1
    degree_values = list(vertex_degree.values())
    warnings = []
    if not graph.vertices:
        warnings.append("No ML node/corner components became graph vertices.")
    if not graph.accepted_edges:
        warnings.append("No ML graph edges passed the line-probability edge thresholds.")
    if graph.vertices and len(graph.accepted_edges) < max(1, len(graph.vertices) // 3):
        warnings.append("Accepted edge count is low relative to vertex count; edge thresholds may be too strict or line probabilities too weak.")
    if np.count_nonzero(graph.node_mask) > np.count_nonzero(graph.line_mask) * 2:
        warnings.append("Node/corner mask is much denser than line mask; the checkpoint may over-predict node/corner.")
    if degree_values and max(degree_values) > args.max_degree:
        warnings.append("Graph remains over-connected after pruning; max vertex degree exceeds configured max_degree.")
    if graph.accepted_edges_before_pruning and len(graph.accepted_edges) / len(graph.accepted_edges_before_pruning) < 0.35:
        warnings.append("Degree pruning removed most initially accepted edges; candidate graph was over-connected.")

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
            "candidate_edge_count_before_pruning": len(graph.candidate_edges),
            "accepted_edge_count": len(graph.accepted_edges),
            "accepted_edge_count_before_pruning": len(graph.accepted_edges_before_pruning),
            "accepted_edge_count_after_pruning": len(graph.accepted_edges),
            "rejected_edge_count": len(graph.candidate_edges) - len(graph.accepted_edges),
            "rejected_by_path_score_count": graph.rejected_by_path_score_count,
            "rejected_by_degree_pruning_count": graph.rejected_by_degree_pruning_count,
            "node_mask_pixel_count": int(np.count_nonzero(graph.node_mask)),
            "line_mask_pixel_count": int(np.count_nonzero(graph.line_mask)),
            "line_to_node_pixel_ratio": float(np.count_nonzero(graph.line_mask) / max(np.count_nonzero(graph.node_mask), 1)),
            "edge_score_mean": float(np.mean(edge_scores)) if edge_scores else 0.0,
            "edge_score_max": float(np.max(edge_scores)) if edge_scores else 0.0,
            "path_score_mean": float(np.mean(edge_scores)) if edge_scores else 0.0,
            "path_score_max": float(np.max(edge_scores)) if edge_scores else 0.0,
            "pre_prune_path_score_mean": float(np.mean(pre_prune_scores)) if pre_prune_scores else 0.0,
            "candidate_edge_score_mean": float(np.mean(candidate_scores)) if candidate_scores else 0.0,
            "path_length_ratio_mean": float(np.mean(path_ratios)) if path_ratios else 0.0,
            "path_length_ratio_max": float(np.max(path_ratios)) if path_ratios else 0.0,
            "candidate_path_length_ratio_mean": float(np.mean(candidate_path_ratios)) if candidate_path_ratios else 0.0,
            "average_vertex_degree": float(np.mean(degree_values)) if degree_values else 0.0,
            "max_vertex_degree": int(max(degree_values)) if degree_values else 0,
            "vertex_degrees": {str(key): int(value) for key, value in vertex_degree.items()},
            "stroke_point_counts": [int(count) for count in stroke_point_counts],
        },
        "thresholds": {
            "node_threshold": args.node_threshold,
            "line_threshold": args.line_threshold,
            "edge_score_threshold": args.edge_score_threshold,
            "support_fraction_threshold": args.support_fraction_threshold,
            "path_min_support_fraction": args.path_min_support_fraction,
            "max_edge_distance_px": args.max_edge_distance_px,
            "nearest_neighbors": args.nearest_neighbors,
            "edge_corridor_radius_px": args.edge_corridor_radius_px,
            "edge_search_mode": args.edge_search_mode,
            "max_degree": args.max_degree,
            "path_corridor_margin_px": args.path_corridor_margin_px,
            "max_path_to_straight_ratio": args.max_path_to_straight_ratio,
            "path_cost_threshold": args.path_cost_threshold,
            "path_max_expanded_nodes": args.path_max_expanded_nodes,
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
    degree = {vertex.id: 0 for vertex in graph.vertices}
    for edge in graph.accepted_edges:
        degree[edge.u] = degree.get(edge.u, 0) + 1
        degree[edge.v] = degree.get(edge.v, 0) + 1
    for edge in graph.candidate_edges:
        points = [(float(x), float(y)) for x, y in edge.polyline_px]
        straight_points = [(float(x), float(y)) for x, y in edge.straight_polyline_px]
        if len(straight_points) >= 2 and edge.search_mode in {"path", "hybrid"}:
            draw.line(straight_points, fill=(80, 80, 80, 45), width=1)
        color = (30, 150, 90, 230) if edge.accepted else (220, 60, 60, 80)
        width = 2 if edge.accepted else 1
        if len(points) >= 2:
            draw.line(points, fill=color, width=width)
    for vertex in graph.vertices:
        draw.ellipse((vertex.x - 4, vertex.y - 4, vertex.x + 4, vertex.y + 4), fill=(255, 230, 0, 235), outline=(0, 0, 0, 235))
        draw.text((vertex.x + 5, vertex.y + 3), f"{vertex.id}/{degree.get(vertex.id, 0)}", fill=(0, 0, 0, 255))
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
    print(
        "Candidate/pre-prune/post-prune edges: "
        f"{len(graph.candidate_edges)} / {len(graph.accepted_edges_before_pruning)} / {len(graph.accepted_edges)}"
    )
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
    parser.add_argument("--edge-search-mode", choices=["straight", "path", "hybrid"], default="path")
    parser.add_argument("--max-degree", type=int, default=3)
    parser.add_argument("--path-corridor-margin-px", type=int, default=24)
    parser.add_argument("--max-path-to-straight-ratio", type=float, default=2.5)
    parser.add_argument("--path-min-support-fraction", type=float, default=0.35)
    parser.add_argument("--path-cost-threshold", type=float, default=1000000000.0)
    parser.add_argument("--path-max-expanded-nodes", type=int, default=8000)
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
