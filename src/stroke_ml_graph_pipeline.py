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
from dataclasses import dataclass, field
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
    source: str = "node_component"
    parent_node_id: int | None = None
    parent_port_id: int | None = None


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
    supported_pixel_count: int = 0
    coverage_gain_pixels: int = 0
    coverage_overlap_fraction: float = 0.0
    rejected_by_vertex_passthrough: bool = False
    rejected_by_node_topology: bool = False
    node_topology_reason: str = ""
    rejected_by_node_routing: bool = False
    node_routing_reason: str = ""
    uses_node_support: bool = False


@dataclass
class NodePort:
    id: int
    node_id: int
    x: float
    y: float
    direction: tuple[float, float]
    confidence: float
    supporting_line_probability_mean: float
    supporting_line_probability_max: float
    supporting_component_id: int
    distance_from_centroid: float


@dataclass
class NodeRoute:
    node_id: int
    port_a: int
    port_b: int
    allowed: bool
    reason: str
    confidence: float


@dataclass
class NodeBlobTopology:
    id: int
    class_name: str
    confidence: float
    area: int
    centroid_x: float
    centroid_y: float
    weighted_centroid_x: float
    weighted_centroid_y: float
    bbox: tuple[int, int, int, int]
    width: int
    height: int
    aspect_ratio: float
    elongation: float
    node_probability_mean: float
    node_probability_max: float
    local_line_probability_mean: float
    local_line_probability_max: float
    incident_line_component_count: int
    incident_directions: list[tuple[float, float]]
    incident_angles_deg: list[float]
    points_yx: np.ndarray
    ports: list[NodePort] = field(default_factory=list)
    routes: list[NodeRoute] = field(default_factory=list)


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
    rejected_by_coverage_pruning_count: int
    rejected_by_vertex_passthrough_count: int
    node_component_vertex_count: int
    line_anchor_vertex_count: int
    merged_vertex_count_before: int
    line_component_count: int
    line_component_candidate_pair_count: int
    claimed_line_pixel_count: int
    line_coverage_fraction: float
    unclaimed_line_pixel_count: int
    node_topologies: list[NodeBlobTopology]
    rejected_by_node_topology_count: int
    rejected_by_node_routing_count: int
    node_port_count: int
    allowed_port_route_count: int
    rejected_port_route_count: int
    accepted_edges_using_node_support_count: int


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


def merge_nearby_vertices(vertices: list[MLGraphVertex], merge_distance_px: float) -> list[MLGraphVertex]:
    if merge_distance_px <= 0 or len(vertices) <= 1:
        for i, vertex in enumerate(vertices):
            vertex.id = i
        return vertices

    remaining = sorted(vertices, key=lambda vertex: vertex.max_probability, reverse=True)
    clusters: list[list[MLGraphVertex]] = []
    while remaining:
        seed = remaining.pop(0)
        cluster = [seed]
        kept = []
        for vertex in remaining:
            if any(math.hypot(vertex.x - other.x, vertex.y - other.y) <= merge_distance_px for other in cluster):
                cluster.append(vertex)
            else:
                kept.append(vertex)
        remaining = kept
        clusters.append(cluster)

    merged: list[MLGraphVertex] = []
    for cluster in clusters:
        weights = np.array([max(vertex.max_probability, 1e-6) * max(vertex.area, 1) for vertex in cluster], dtype=np.float64)
        total = float(np.sum(weights))
        x = float(sum(vertex.x * weight for vertex, weight in zip(cluster, weights)) / total)
        y = float(sum(vertex.y * weight for vertex, weight in zip(cluster, weights)) / total)
        source_names = {vertex.source for vertex in cluster}
        merged.append(
            MLGraphVertex(
                id=len(merged),
                x=x,
                y=y,
                area=int(sum(vertex.area for vertex in cluster)),
                mean_probability=float(np.mean([vertex.mean_probability for vertex in cluster])),
                max_probability=float(max(vertex.max_probability for vertex in cluster)),
                source_component_id=int(cluster[0].source_component_id),
                split_from_large_component=any(vertex.split_from_large_component for vertex in cluster),
                source="+".join(sorted(source_names)),
            )
        )

    merged.sort(key=lambda vertex: (vertex.y, vertex.x))
    for i, vertex in enumerate(merged):
        vertex.id = i
    return merged


def component_min_distance_px(points_yx: np.ndarray, vertex: MLGraphVertex) -> float:
    if len(points_yx) == 0:
        return float("inf")
    dx = points_yx[:, 1].astype(np.float32) - float(vertex.x)
    dy = points_yx[:, 0].astype(np.float32) - float(vertex.y)
    return float(np.sqrt(np.min(dx * dx + dy * dy)))


def local_probability_stats(probability: np.ndarray, x: float, y: float, radius_px: int = 2) -> tuple[float, float]:
    height, width = probability.shape
    cx = int(round(x))
    cy = int(round(y))
    x0 = max(0, cx - radius_px)
    x1 = min(width, cx + radius_px + 1)
    y0 = max(0, cy - radius_px)
    y1 = min(height, cy + radius_px + 1)
    values = probability[y0:y1, x0:x1]
    if values.size == 0:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.max(values))


def add_line_component_anchor_vertices(
    vertices: list[MLGraphVertex],
    line_components: list[np.ndarray],
    line_prob: np.ndarray,
    min_component_pixels: int,
    min_anchor_distance_px: float,
    max_anchors_per_component: int,
) -> tuple[list[MLGraphVertex], int]:
    """Add sparse virtual anchors where strong line components lack nearby corners.

    Raghav-style graph reconstruction depends on useful graph vertices. For
    smooth curves, a corner model can leave long stretches of line probability
    without anchors, so those curves never become candidate edges. These anchors
    are generated from the line channel only and do not use skeletonisation.
    """
    if max_anchors_per_component <= 0:
        return vertices, 0

    augmented = list(vertices)
    added = 0
    for component_id, points_yx in enumerate(line_components):
        if len(points_yx) < min_component_pixels:
            continue

        candidates: list[tuple[float, float]] = []
        extrema_indices = [
            int(np.argmin(points_yx[:, 1])),
            int(np.argmax(points_yx[:, 1])),
            int(np.argmin(points_yx[:, 0])),
            int(np.argmax(points_yx[:, 0])),
        ]
        for idx in extrema_indices:
            y = float(points_yx[idx, 0])
            x = float(points_yx[idx, 1])
            if all(math.hypot(x - px, y - py) > min_anchor_distance_px for px, py in candidates):
                candidates.append((x, y))

        anchors_for_component = 0
        for x, y in candidates:
            if anchors_for_component >= max_anchors_per_component:
                break
            if any(math.hypot(x - vertex.x, y - vertex.y) < min_anchor_distance_px for vertex in augmented):
                continue
            mean_probability, max_probability = local_probability_stats(line_prob, x, y)
            augmented.append(
                MLGraphVertex(
                    id=len(augmented),
                    x=float(x),
                    y=float(y),
                    area=1,
                    mean_probability=mean_probability,
                    max_probability=max_probability,
                    source_component_id=component_id,
                    source="line_anchor",
                )
            )
            added += 1
            anchors_for_component += 1

    augmented.sort(key=lambda vertex: (vertex.y, vertex.x))
    for i, vertex in enumerate(augmented):
        vertex.id = i
    return augmented, added


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


def angle_between_vectors_deg(a: np.ndarray, b: np.ndarray, undirected: bool = True) -> float:
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a <= 1e-9 or norm_b <= 1e-9:
        return 180.0
    dot = float(np.clip(np.dot(a / norm_a, b / norm_b), -1.0, 1.0))
    angle = math.degrees(math.acos(dot))
    if undirected:
        angle = min(angle, 180.0 - angle)
    return angle


def direction_matches_blob(direction: np.ndarray, blob: NodeBlobTopology, max_angle_deg: float) -> bool:
    if not blob.incident_directions:
        return blob.class_name in {"line_like_node_fragment", "noisy_blob"}
    return any(
        angle_between_vectors_deg(direction, np.array(blob_direction, dtype=np.float32), undirected=True) <= max_angle_deg
        for blob_direction in blob.incident_directions
    )


def classify_node_blob(
    area: int,
    aspect_ratio: float,
    incident_count: int,
    incident_angles: list[float],
    args: argparse.Namespace,
) -> tuple[str, float]:
    if area < args.node_blob_min_area:
        return "noisy_blob", 0.35
    if aspect_ratio >= args.line_like_node_aspect_threshold and incident_count <= 2:
        return "line_like_node_fragment", 0.70
    if incident_count <= 0:
        return "noisy_blob", 0.30
    if incident_count == 1:
        return "endpoint", 0.75
    if incident_count == 2:
        angle = incident_angles[0] if incident_angles else 180.0
        if angle >= args.node_topology_angle_threshold:
            return "smooth_bend", 0.65
        return "sharp_corner", 0.75
    if incident_count == 3:
        return "t_junction", 0.70
    if incident_count >= 4:
        if incident_count > args.max_node_blob_degree:
            return "multi_junction", 0.45
        if incident_angles and max(incident_angles) >= args.node_topology_angle_threshold:
            return "crossing_or_overlap", 0.65
        return "multi_junction", 0.60
    return "noisy_blob", 0.25


def cluster_node_ports(ports: list[NodePort], args: argparse.Namespace) -> list[NodePort]:
    clustered: list[NodePort] = []
    for port in sorted(ports, key=lambda candidate: candidate.confidence, reverse=True):
        duplicate_index: int | None = None
        direction = np.array(port.direction, dtype=np.float32)
        for i, existing in enumerate(clustered):
            existing_direction = np.array(existing.direction, dtype=np.float32)
            angle = angle_between_vectors_deg(direction, existing_direction, undirected=True)
            distance = math.hypot(port.x - existing.x, port.y - existing.y)
            if angle <= args.node_port_angle_bin_degrees or distance <= args.node_port_min_separation_px:
                duplicate_index = i
                break
        if duplicate_index is None:
            port.id = len(clustered)
            clustered.append(port)
            continue

        existing = clustered[duplicate_index]
        weight_a = max(existing.confidence, 1e-6)
        weight_b = max(port.confidence, 1e-6)
        total = weight_a + weight_b
        merged_direction = np.array(existing.direction, dtype=np.float32) * weight_a + direction * weight_b
        norm = float(np.linalg.norm(merged_direction))
        if norm > 1e-9:
            merged_direction = merged_direction / norm
        clustered[duplicate_index] = NodePort(
            id=existing.id,
            node_id=existing.node_id,
            x=float((existing.x * weight_a + port.x * weight_b) / total),
            y=float((existing.y * weight_a + port.y * weight_b) / total),
            direction=(float(merged_direction[0]), float(merged_direction[1])),
            confidence=float(max(existing.confidence, port.confidence)),
            supporting_line_probability_mean=float(max(existing.supporting_line_probability_mean, port.supporting_line_probability_mean)),
            supporting_line_probability_max=float(max(existing.supporting_line_probability_max, port.supporting_line_probability_max)),
            supporting_component_id=existing.supporting_component_id,
            distance_from_centroid=float(max(existing.distance_from_centroid, port.distance_from_centroid)),
        )
    clustered.sort(key=lambda candidate: math.atan2(candidate.direction[1], candidate.direction[0]))
    for i, port in enumerate(clustered):
        port.id = i
    return clustered


def detect_node_ports(
    node_id: int,
    weighted_x: float,
    weighted_y: float,
    bbox: tuple[int, int, int, int],
    line_prob: np.ndarray,
    line_mask: np.ndarray,
    args: argparse.Namespace,
) -> list[NodePort]:
    height, width = line_mask.shape
    x0, y0, x1, y1 = bbox
    radius = int(args.node_port_radius_px)
    ly0 = max(0, y0 - radius)
    ly1 = min(height, y1 + radius + 1)
    lx0 = max(0, x0 - radius)
    lx1 = min(width, x1 + radius + 1)
    local_line_mask = line_mask[ly0:ly1, lx0:lx1]
    ports: list[NodePort] = []
    for component_id, line_component in enumerate(connected_components(local_line_mask, connectivity=8)):
        local_points = np.array(line_component, dtype=np.int32)
        if len(local_points) == 0:
            continue
        global_points = local_points + np.array([ly0, lx0], dtype=np.int32)
        values = line_prob[global_points[:, 0], global_points[:, 1]]
        if float(np.max(values)) < args.node_port_min_line_prob:
            continue

        distances = np.sqrt((global_points[:, 1] - weighted_x) ** 2 + (global_points[:, 0] - weighted_y) ** 2)
        nearby = distances <= args.node_port_radius_px
        if not np.any(nearby):
            continue
        nearby_points = global_points[nearby]
        nearby_values = line_prob[nearby_points[:, 0], nearby_points[:, 1]]
        value_mask = nearby_values >= args.node_port_min_line_prob
        if np.any(value_mask):
            nearby_points = nearby_points[value_mask]
            nearby_values = nearby_values[value_mask]

        near_count = min(10, len(nearby_points))
        nearest_order = np.argsort(np.sqrt((nearby_points[:, 1] - weighted_x) ** 2 + (nearby_points[:, 0] - weighted_y) ** 2))[:near_count]
        support_points = nearby_points[nearest_order]
        support_values = line_prob[support_points[:, 0], support_points[:, 1]]
        port_y = float(np.mean(support_points[:, 0]))
        port_x = float(np.mean(support_points[:, 1]))
        direction = np.array([port_x - weighted_x, port_y - weighted_y], dtype=np.float32)
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-9:
            continue
        direction = direction / norm
        distance = float(math.hypot(port_x - weighted_x, port_y - weighted_y))
        confidence = float(np.clip(0.7 * np.mean(support_values) + 0.3 * np.max(support_values), 0.0, 1.0))
        ports.append(
            NodePort(
                id=len(ports),
                node_id=node_id,
                x=port_x,
                y=port_y,
                direction=(float(direction[0]), float(direction[1])),
                confidence=confidence,
                supporting_line_probability_mean=float(np.mean(support_values)),
                supporting_line_probability_max=float(np.max(support_values)),
                supporting_component_id=component_id,
                distance_from_centroid=distance,
            )
        )
    return cluster_node_ports(ports, args)


def classify_node_blob_from_ports(
    area: int,
    aspect_ratio: float,
    ports: list[NodePort],
    incident_angles: list[float],
    args: argparse.Namespace,
) -> tuple[str, float]:
    port_count = len(ports)
    if port_count <= 0:
        if area < args.node_blob_min_area:
            return "noisy_blob", 0.35
        return "noisy_blob", 0.30
    if port_count == 1:
        return "endpoint", 0.75
    if port_count == 2:
        angle = incident_angles[0] if incident_angles else 180.0
        if aspect_ratio >= args.line_like_node_aspect_threshold and angle >= args.node_topology_angle_threshold:
            return "line_like_node_fragment", 0.72
        if angle >= args.node_topology_angle_threshold:
            return "smooth_bend", 0.72
        return "sharp_corner", 0.78
    if port_count == 3:
        return "t_junction", 0.72
    if port_count >= 4:
        opposite_pairs = 0
        for angle in incident_angles:
            if angle >= args.node_topology_angle_threshold:
                opposite_pairs += 1
        if opposite_pairs >= 2:
            return "crossing_or_overlap", 0.68
        if port_count > args.max_node_blob_degree:
            return "multi_junction", 0.50
        return "multi_junction", 0.62
    return classify_node_blob(area, aspect_ratio, port_count, incident_angles, args)


def build_node_routes(blob: NodeBlobTopology, args: argparse.Namespace) -> list[NodeRoute]:
    ports = blob.ports
    routes: list[NodeRoute] = []
    if len(ports) < 2:
        return routes

    pair_scores: list[tuple[float, float, int, int]] = []
    for i in range(len(ports)):
        for j in range(i + 1, len(ports)):
            angle = angle_between_vectors_deg(
                np.array(ports[i].direction, dtype=np.float32),
                np.array(ports[j].direction, dtype=np.float32),
                undirected=False,
            )
            confidence = float((ports[i].confidence + ports[j].confidence) * 0.5)
            pair_scores.append((angle, confidence, i, j))

    allowed_pairs: set[tuple[int, int]] = set()
    class_name = blob.class_name
    if class_name == "endpoint":
        allowed_pairs = set()
    elif class_name in {"smooth_bend", "line_like_node_fragment"}:
        best = max(pair_scores, key=lambda item: (item[0], item[1]))
        allowed_pairs.add((best[2], best[3]))
    elif class_name == "sharp_corner":
        best = max(pair_scores, key=lambda item: (item[1], -abs(item[0] - 90.0)))
        allowed_pairs.add((best[2], best[3]))
    elif class_name == "t_junction":
        trunk = max(pair_scores, key=lambda item: (item[0], item[1]))
        allowed_pairs.add((trunk[2], trunk[3]))
        branch_pairs = sorted(
            [item for item in pair_scores if item[2] not in trunk[2:4] or item[3] not in trunk[2:4]],
            key=lambda item: (item[1], -abs(item[0] - 90.0)),
            reverse=True,
        )
        for _, _, i, j in branch_pairs[: max(0, args.node_route_max_pairs - 1)]:
            allowed_pairs.add((i, j))
    elif class_name == "crossing_or_overlap":
        used_ports: set[int] = set()
        for _, _, i, j in sorted(pair_scores, key=lambda item: (item[0], item[1]), reverse=True):
            if i in used_ports or j in used_ports:
                continue
            allowed_pairs.add((i, j))
            used_ports.update({i, j})
            if len(allowed_pairs) >= args.node_route_max_pairs:
                break
    elif class_name == "multi_junction":
        for _, _, i, j in sorted(pair_scores, key=lambda item: (item[1], item[0]), reverse=True)[: args.node_route_max_pairs]:
            allowed_pairs.add((i, j))
    elif class_name == "noisy_blob":
        for angle, confidence, i, j in pair_scores:
            if angle >= args.node_route_angle_threshold and confidence >= args.node_port_min_line_prob:
                allowed_pairs.add((i, j))

    for angle, confidence, i, j in pair_scores:
        key = (i, j)
        allowed = key in allowed_pairs or (j, i) in allowed_pairs
        reason = "allowed route" if allowed else f"disallowed {class_name} route"
        routes.append(
            NodeRoute(
                node_id=blob.id,
                port_a=i,
                port_b=j,
                allowed=allowed,
                reason=reason,
                confidence=float(confidence),
            )
        )
    return routes


def analyze_node_topologies(
    node_prob: np.ndarray,
    line_prob: np.ndarray,
    node_mask: np.ndarray,
    line_mask: np.ndarray,
    args: argparse.Namespace,
) -> list[NodeBlobTopology]:
    topologies: list[NodeBlobTopology] = []
    height, width = node_mask.shape
    for component_id, component in enumerate(connected_components(node_mask, connectivity=8)):
        points_yx = np.array(component, dtype=np.int32)
        area = int(len(points_yx))
        if area <= 0:
            continue
        ys = points_yx[:, 0]
        xs = points_yx[:, 1]
        y0 = int(np.min(ys))
        y1 = int(np.max(ys))
        x0 = int(np.min(xs))
        x1 = int(np.max(xs))
        bbox_w = x1 - x0 + 1
        bbox_h = y1 - y0 + 1
        aspect_ratio = max(bbox_w, bbox_h) / max(min(bbox_w, bbox_h), 1)
        values = node_prob[ys, xs]
        centroid_y = float(np.mean(ys))
        centroid_x = float(np.mean(xs))
        weighted_x, weighted_y = weighted_centroid(points_yx, node_prob)

        radius = int(args.node_incident_radius_px)
        ly0 = max(0, y0 - radius)
        ly1 = min(height, y1 + radius + 1)
        lx0 = max(0, x0 - radius)
        lx1 = min(width, x1 + radius + 1)
        local_line_prob = line_prob[ly0:ly1, lx0:lx1]
        local_line_mask = line_mask[ly0:ly1, lx0:lx1]
        incident_directions: list[tuple[float, float]] = []
        incident_count = 0
        for line_component in connected_components(local_line_mask, connectivity=8):
            local_points = np.array(line_component, dtype=np.int32)
            if len(local_points) == 0:
                continue
            global_points = local_points + np.array([ly0, lx0], dtype=np.int32)
            distances = np.sqrt((global_points[:, 1] - weighted_x) ** 2 + (global_points[:, 0] - weighted_y) ** 2)
            if float(np.min(distances)) > args.node_incident_radius_px:
                continue
            incident_count += 1
            near_count = min(8, len(global_points))
            near_points = global_points[np.argsort(distances)[:near_count]]
            mean_y = float(np.mean(near_points[:, 0]))
            mean_x = float(np.mean(near_points[:, 1]))
            direction = np.array([mean_x - weighted_x, mean_y - weighted_y], dtype=np.float32)
            norm = float(np.linalg.norm(direction))
            if norm > 1e-9:
                unit = direction / norm
                # Merge directions that are already represented in this blob.
                if all(angle_between_vectors_deg(unit, np.array(existing), undirected=True) > args.node_direction_bin_degrees for existing in incident_directions):
                    incident_directions.append((float(unit[0]), float(unit[1])))

        ports = detect_node_ports(
            node_id=component_id,
            weighted_x=float(weighted_x),
            weighted_y=float(weighted_y),
            bbox=(x0, y0, x1, y1),
            line_prob=line_prob,
            line_mask=line_mask,
            args=args,
        )
        port_directions = [port.direction for port in ports]
        incident_angles: list[float] = []
        for i in range(len(port_directions)):
            for j in range(i + 1, len(port_directions)):
                incident_angles.append(
                    angle_between_vectors_deg(
                        np.array(port_directions[i], dtype=np.float32),
                        np.array(port_directions[j], dtype=np.float32),
                        undirected=False,
                    )
                )
        if args.enable_node_port_routing:
            class_name, confidence = classify_node_blob_from_ports(area, aspect_ratio, ports, incident_angles, args)
        else:
            class_name, confidence = classify_node_blob(area, aspect_ratio, len(incident_directions), incident_angles, args)
        blob = NodeBlobTopology(
            id=component_id,
            class_name=class_name,
            confidence=confidence,
            area=area,
            centroid_x=centroid_x,
            centroid_y=centroid_y,
            weighted_centroid_x=float(weighted_x),
            weighted_centroid_y=float(weighted_y),
            bbox=(x0, y0, x1, y1),
            width=bbox_w,
            height=bbox_h,
            aspect_ratio=float(aspect_ratio),
            elongation=float(aspect_ratio),
            node_probability_mean=float(np.mean(values)),
            node_probability_max=float(np.max(values)),
            local_line_probability_mean=float(np.mean(local_line_prob)) if local_line_prob.size else 0.0,
            local_line_probability_max=float(np.max(local_line_prob)) if local_line_prob.size else 0.0,
            incident_line_component_count=int(incident_count),
            incident_directions=port_directions if args.enable_node_port_routing else incident_directions,
            incident_angles_deg=[float(angle) for angle in incident_angles],
            points_yx=points_yx,
            ports=ports,
        )
        blob.routes = build_node_routes(blob, args)
        topologies.append(blob)
    return topologies


def node_support_probability_field(
    line_prob: np.ndarray,
    node_prob: np.ndarray,
    node_topologies: list[NodeBlobTopology],
    args: argparse.Namespace,
) -> np.ndarray:
    support_weight = args.node_route_support_weight if args.enable_node_port_routing else args.node_support_weight
    if not (args.enable_node_topology or args.enable_node_port_routing) or support_weight <= 0:
        return line_prob
    support = line_prob.copy()
    local_node_mask = np.zeros_like(line_prob, dtype=bool)
    for blob in node_topologies:
        if blob.class_name == "noisy_blob":
            continue
        local_node_mask[blob.points_yx[:, 0], blob.points_yx[:, 1]] = True
    local_node_mask = expanded_mask(local_node_mask, int(args.node_support_radius_px))
    boosted = np.clip(line_prob + support_weight * node_prob, 0.0, 1.0)
    support[local_node_mask] = np.maximum(support[local_node_mask], boosted[local_node_mask])
    return support


def add_node_port_vertices(
    vertices: list[MLGraphVertex],
    node_topologies: list[NodeBlobTopology],
    line_prob: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[MLGraphVertex], int]:
    if not args.enable_node_port_routing:
        return vertices, 0
    augmented = list(vertices)
    added = 0
    for blob in node_topologies:
        if blob.class_name == "noisy_blob":
            continue
        for port in blob.ports:
            if port.confidence < args.node_port_min_line_prob:
                continue
            if any(
                vertex.parent_node_id == blob.id and math.hypot(vertex.x - port.x, vertex.y - port.y) < args.node_port_min_separation_px
                for vertex in augmented
            ):
                continue
            mean_probability, max_probability = local_probability_stats(line_prob, port.x, port.y)
            augmented.append(
                MLGraphVertex(
                    id=len(augmented),
                    x=float(port.x),
                    y=float(port.y),
                    area=1,
                    mean_probability=float(max(mean_probability, port.supporting_line_probability_mean)),
                    max_probability=float(max(max_probability, port.supporting_line_probability_max)),
                    source_component_id=blob.id,
                    source="node_port",
                    parent_node_id=blob.id,
                    parent_port_id=port.id,
                )
            )
            added += 1
    augmented.sort(key=lambda vertex: (vertex.y, vertex.x))
    for i, vertex in enumerate(augmented):
        vertex.id = i
    return augmented, added


def route_allowed_for_ports(blob: NodeBlobTopology, port_a: int, port_b: int) -> tuple[bool, str]:
    if port_a == port_b:
        return True, "same node port"
    for route in blob.routes:
        if {route.port_a, route.port_b} == {port_a, port_b}:
            return route.allowed, route.reason
    return False, f"no route between node {blob.id} ports {port_a} and {port_b}"


def vertex_port(vertices: list[MLGraphVertex], vertex_id: int) -> tuple[int, int] | None:
    vertex = vertices[vertex_id]
    if vertex.parent_node_id is None or vertex.parent_port_id is None:
        return None
    return int(vertex.parent_node_id), int(vertex.parent_port_id)


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
    path_prob: np.ndarray,
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
            path_prob,
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


def path_passes_near_other_vertex(
    polyline: np.ndarray,
    vertices: list[MLGraphVertex],
    u: int,
    v: int,
    radius_px: float,
) -> bool:
    if radius_px <= 0 or len(polyline) == 0:
        return False
    radius_sq = radius_px * radius_px
    for vertex in vertices:
        if vertex.id in {u, v}:
            continue
        dx = polyline[:, 0] - float(vertex.x)
        dy = polyline[:, 1] - float(vertex.y)
        if float(np.min(dx * dx + dy * dy)) <= radius_sq:
            return True
    return False


def edge_node_topology_conflict(
    polyline: np.ndarray,
    node_topologies: list[NodeBlobTopology],
    vertices: list[MLGraphVertex],
    u: int,
    v: int,
    args: argparse.Namespace,
) -> tuple[bool, str]:
    if not args.enable_node_topology or len(polyline) < 2:
        return False, ""

    start = np.array([vertices[u].x, vertices[u].y], dtype=np.float32)
    end = np.array([vertices[v].x, vertices[v].y], dtype=np.float32)
    for blob in node_topologies:
        center = np.array([blob.weighted_centroid_x, blob.weighted_centroid_y], dtype=np.float32)
        distances = np.linalg.norm(polyline.astype(np.float32) - center[None, :], axis=1)
        nearest_i = int(np.argmin(distances))
        if float(distances[nearest_i]) > args.node_support_radius_px:
            continue

        is_endpoint_blob = (
            float(np.linalg.norm(start - center)) <= args.node_support_radius_px * 1.5
            or float(np.linalg.norm(end - center)) <= args.node_support_radius_px * 1.5
        )
        if blob.class_name == "noisy_blob":
            return True, f"edge crosses noisy node blob {blob.id}"
        if blob.class_name == "endpoint" and not is_endpoint_blob:
            return True, f"edge passes through endpoint blob {blob.id}"

        lo = max(0, nearest_i - 3)
        hi = min(len(polyline) - 1, nearest_i + 3)
        direction = polyline[hi].astype(np.float32) - polyline[lo].astype(np.float32)
        if float(np.linalg.norm(direction)) <= 1e-9:
            continue
        if not direction_matches_blob(direction, blob, args.node_topology_angle_threshold):
            return True, f"edge direction inconsistent with {blob.class_name} blob {blob.id}"

        if blob.class_name in {"line_like_node_fragment", "crossing_or_overlap"} and not is_endpoint_blob:
            # These regions are useful local support, but they should not become
            # broad bridges between unrelated strokes unless an edge terminates
            # at the blob or clearly follows one of its detected directions.
            if blob.confidence < 0.75:
                return True, f"edge cuts through ambiguous {blob.class_name} blob {blob.id}"
    return False, ""


def edge_node_routing_conflict(
    polyline: np.ndarray,
    node_topologies: list[NodeBlobTopology],
    vertices: list[MLGraphVertex],
    u: int,
    v: int,
    args: argparse.Namespace,
) -> tuple[bool, str, bool]:
    if not args.enable_node_port_routing or len(polyline) < 2:
        return False, "", False

    blob_by_id = {blob.id: blob for blob in node_topologies}
    u_port = vertex_port(vertices, u)
    v_port = vertex_port(vertices, v)
    uses_node_support = False

    if u_port and v_port and u_port[0] == v_port[0]:
        blob = blob_by_id.get(u_port[0])
        if blob is None:
            return True, f"missing parent node {u_port[0]}", uses_node_support
        allowed, reason = route_allowed_for_ports(blob, u_port[1], v_port[1])
        return (not allowed), reason, True

    endpoint_ports = {item for item in (u_port, v_port) if item is not None}
    for blob in node_topologies:
        center = np.array([blob.weighted_centroid_x, blob.weighted_centroid_y], dtype=np.float32)
        distances = np.linalg.norm(polyline.astype(np.float32) - center[None, :], axis=1)
        nearest_i = int(np.argmin(distances))
        if float(distances[nearest_i]) > args.node_support_radius_px:
            continue

        uses_node_support = True
        matched_endpoint = next((item for item in endpoint_ports if item[0] == blob.id), None)
        lo = max(0, nearest_i - 3)
        hi = min(len(polyline) - 1, nearest_i + 3)
        direction = polyline[hi].astype(np.float32) - polyline[lo].astype(np.float32)
        norm = float(np.linalg.norm(direction))
        if norm > 1e-9:
            direction = direction / norm

        if matched_endpoint is not None:
            port = next((candidate for candidate in blob.ports if candidate.id == matched_endpoint[1]), None)
            if port is None:
                return True, f"missing endpoint port {matched_endpoint[1]} for node {blob.id}", uses_node_support
            if norm > 1e-9 and angle_between_vectors_deg(direction, np.array(port.direction, dtype=np.float32), undirected=True) > args.node_route_angle_threshold:
                return True, f"edge leaves node {blob.id} through mismatched port {port.id}", uses_node_support
            continue

        if blob.class_name in {"endpoint", "noisy_blob"}:
            return True, f"edge crosses {blob.class_name} without terminating at a port", uses_node_support
        if norm > 1e-9 and not direction_matches_blob(direction, blob, args.node_route_angle_threshold):
            return True, f"edge crosses node {blob.id} without matching an incident port", uses_node_support

    return False, "", uses_node_support


def supported_path_pixels(edge: MLGraphEdge, line_prob: np.ndarray, line_threshold: float, radius_px: int) -> set[tuple[int, int]]:
    height, width = line_prob.shape
    pixels: set[tuple[int, int]] = set()
    rounded = np.rint(edge.polyline_px).astype(np.int32)
    for x, y in rounded:
        for dy in range(-radius_px, radius_px + 1):
            for dx in range(-radius_px, radius_px + 1):
                if dx * dx + dy * dy > radius_px * radius_px:
                    continue
                px = int(x + dx)
                py = int(y + dy)
                if 0 <= px < width and 0 <= py < height and float(line_prob[py, px]) >= line_threshold:
                    pixels.add((py, px))
    return pixels


def prune_edges_by_degree_and_coverage(
    edges: list[MLGraphEdge],
    vertices: list[MLGraphVertex],
    line_prob: np.ndarray,
    line_mask: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[MLGraphEdge], int, int, int, int, float, int]:
    if args.max_degree <= 0:
        for i, edge in enumerate(edges):
            edge.id = i
            edge.accepted = True
            edge.reason = "accepted"
        claimed = set()
        for edge in edges:
            claimed.update(supported_path_pixels(edge, line_prob, args.line_threshold, args.coverage_radius_px))
        line_pixels = int(np.count_nonzero(line_mask))
        claimed_count = len(claimed)
        coverage_fraction = claimed_count / max(line_pixels, 1)
        return edges, 0, 0, 0, claimed_count, coverage_fraction, max(line_pixels - claimed_count, 0)

    degree = {vertex.id: 0 for vertex in vertices}
    accepted: list[MLGraphEdge] = []
    claimed_pixels: set[tuple[int, int]] = set()
    edge_pixels = {edge.id: supported_path_pixels(edge, line_prob, args.line_threshold, args.coverage_radius_px) for edge in edges}
    for edge in edges:
        edge.supported_pixel_count = len(edge_pixels[edge.id])

    rejected_by_degree = 0
    rejected_by_coverage = 0
    line_pixels = int(np.count_nonzero(line_mask))

    def selection_key(edge: MLGraphEdge) -> tuple[float, float, float]:
        supported = max(edge.supported_pixel_count, 1)
        coverage_weight = math.sqrt(float(supported))
        return (edge.score * coverage_weight, edge.support_fraction, float(supported))

    for edge in sorted(edges, key=selection_key, reverse=True):
        pixels = edge_pixels[edge.id]
        new_pixels = pixels - claimed_pixels
        edge.coverage_gain_pixels = len(new_pixels)
        edge.coverage_overlap_fraction = 1.0 - (len(new_pixels) / max(len(pixels), 1))
        enough_new_coverage = (
            len(new_pixels) >= args.min_edge_new_pixels
            and (len(new_pixels) / max(len(pixels), 1)) >= args.min_edge_new_coverage_fraction
        )
        if not enough_new_coverage:
            edge.accepted = False
            edge.reason = "rejected by low new line coverage"
            rejected_by_coverage += 1
            continue
        if degree.get(edge.u, 0) >= args.max_degree or degree.get(edge.v, 0) >= args.max_degree:
            edge.accepted = False
            edge.reason = "rejected by degree pruning"
            rejected_by_degree += 1
            continue
        degree[edge.u] = degree.get(edge.u, 0) + 1
        degree[edge.v] = degree.get(edge.v, 0) + 1
        edge.accepted = True
        edge.reason = "accepted"
        edge.id = len(accepted)
        accepted.append(edge)
        claimed_pixels.update(new_pixels)

    claimed_count = len(claimed_pixels)
    coverage_fraction = claimed_count / max(line_pixels, 1)
    return (
        accepted,
        rejected_by_degree,
        rejected_by_coverage,
        0,
        claimed_count,
        coverage_fraction,
        max(line_pixels - claimed_count, 0),
    )


def build_candidate_edges(
    vertices: list[MLGraphVertex],
    line_prob: np.ndarray,
    path_prob: np.ndarray,
    line_mask: np.ndarray,
    line_components: list[np.ndarray],
    node_topologies: list[NodeBlobTopology],
    args: argparse.Namespace,
) -> tuple[list[MLGraphEdge], list[MLGraphEdge], list[MLGraphEdge], int, int, int, int, int, int, int, int, int, float, int]:
    pairs: set[tuple[int, int]] = set()
    coords = np.array([[vertex.x, vertex.y] for vertex in vertices], dtype=np.float32)

    def add_pair(i: int, j: int) -> None:
        if i == j:
            return
        pairs.add((min(i, j), max(i, j)))

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
            add_pair(i, j)
            added += 1
            if added >= args.nearest_neighbors:
                break

    pair_count_before_components = len(pairs)
    if args.enable_node_port_routing:
        port_vertex_by_key = {
            (int(vertex.parent_node_id), int(vertex.parent_port_id)): vertex.id
            for vertex in vertices
            if vertex.parent_node_id is not None and vertex.parent_port_id is not None
        }
        for blob in node_topologies:
            for route in blob.routes:
                if not route.allowed:
                    continue
                a = port_vertex_by_key.get((blob.id, route.port_a))
                b = port_vertex_by_key.get((blob.id, route.port_b))
                if a is not None and b is not None:
                    add_pair(a, b)

    if args.use_line_component_candidates:
        for points_yx in line_components:
            if len(points_yx) < args.line_component_min_pixels:
                continue
            nearby_vertices = [
                vertex.id
                for vertex in vertices
                if component_min_distance_px(points_yx, vertex) <= args.component_vertex_radius_px
            ]
            if len(nearby_vertices) < 2:
                continue
            for vertex_id in nearby_vertices:
                distances = []
                for other_id in nearby_vertices:
                    if other_id == vertex_id:
                        continue
                    distance = float(np.linalg.norm(coords[vertex_id] - coords[other_id]))
                    if distance <= args.component_max_edge_distance_px:
                        distances.append((distance, other_id))
                distances.sort()
                for _, other_id in distances[: args.component_candidate_neighbors]:
                    add_pair(vertex_id, other_id)
    line_component_pair_count = len(pairs) - pair_count_before_components

    candidate_edges: list[MLGraphEdge] = []
    accepted_before_pruning: list[MLGraphEdge] = []
    rejected_by_path_score = 0
    rejected_by_vertex_passthrough = 0
    rejected_by_node_routing = 0
    for u, v in sorted(pairs):
        p0 = (vertices[u].x, vertices[u].y)
        p1 = (vertices[v].x, vertices[v].y)
        path_metrics = score_edge_path(line_prob, path_prob, p0, p1, args)
        passes_other_vertex = path_passes_near_other_vertex(
            path_metrics["polyline"],
            vertices,
            u,
            v,
            args.vertex_passthrough_radius_px,
        )
        node_conflict, node_conflict_reason = edge_node_topology_conflict(
            path_metrics["polyline"],
            node_topologies,
            vertices,
            u,
            v,
            args,
        )
        routing_conflict, routing_conflict_reason, uses_node_support = edge_node_routing_conflict(
            path_metrics["polyline"],
            node_topologies,
            vertices,
            u,
            v,
            args,
        )
        accepted = bool(
            path_metrics["path_found"]
            and path_metrics["score"] >= args.edge_score_threshold
            and path_metrics["support_fraction"] >= args.path_min_support_fraction
            and path_metrics["support_fraction"] >= args.support_fraction_threshold
            and path_metrics["length_ratio"] <= args.max_path_to_straight_ratio
            and path_metrics["total_path_cost"] <= args.path_cost_threshold
            and not passes_other_vertex
            and not node_conflict
            and not routing_conflict
        )
        reason = "accepted" if accepted else "score/support below threshold"
        if not path_metrics["path_found"]:
            reason = "no low-cost probability path found"
        elif routing_conflict:
            reason = routing_conflict_reason
        elif node_conflict:
            reason = node_conflict_reason
        elif passes_other_vertex:
            reason = "path passes through another graph vertex"
        elif path_metrics["length_ratio"] > args.max_path_to_straight_ratio:
            reason = "path too long relative to straight distance"
        elif path_metrics["total_path_cost"] > args.path_cost_threshold:
            reason = "path cost above threshold"
        if not accepted:
            rejected_by_path_score += 1
            if passes_other_vertex:
                rejected_by_vertex_passthrough += 1
            if routing_conflict:
                rejected_by_node_routing += 1
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
            rejected_by_vertex_passthrough=passes_other_vertex,
            rejected_by_node_topology=node_conflict,
            node_topology_reason=node_conflict_reason,
            rejected_by_node_routing=routing_conflict,
            node_routing_reason=routing_conflict_reason,
            uses_node_support=uses_node_support,
        )
        candidate_edges.append(edge)
        if accepted:
            accepted_before_pruning.append(edge)
    (
        accepted_edges,
        rejected_by_degree,
        rejected_by_coverage,
        _,
        claimed_line_pixels,
        line_coverage_fraction,
        unclaimed_line_pixels,
    ) = prune_edges_by_degree_and_coverage(accepted_before_pruning, vertices, line_prob, line_mask, args)
    return (
        candidate_edges,
        accepted_before_pruning,
        accepted_edges,
        rejected_by_path_score,
        rejected_by_degree,
        rejected_by_coverage,
        rejected_by_vertex_passthrough,
        sum(1 for edge in candidate_edges if edge.rejected_by_node_topology),
        rejected_by_node_routing,
        line_component_pair_count,
        claimed_line_pixels,
        line_coverage_fraction,
        unclaimed_line_pixels,
    )


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
    node_vertices, node_mask = extract_vertices_from_nodes(
        probabilities.node_prob,
        threshold=args.node_threshold,
        min_area=args.min_node_area,
        min_separation_px=args.min_vertex_separation_px,
        large_component_area=args.large_node_component_area,
        max_large_component_vertices=args.max_large_component_vertices,
        max_vertices=args.max_vertices,
    )
    line_mask = probabilities.line_prob >= args.line_threshold
    node_topologies = analyze_node_topologies(
        probabilities.node_prob,
        probabilities.line_prob,
        node_mask,
        line_mask,
        args,
    )
    path_prob = node_support_probability_field(probabilities.line_prob, probabilities.node_prob, node_topologies, args)
    line_components = [
        np.array(component, dtype=np.int32)
        for component in connected_components(line_mask, connectivity=8)
        if len(component) >= args.line_component_min_pixels
    ]
    vertices = list(node_vertices)
    line_anchor_count = 0
    if args.add_line_component_anchors:
        vertices, line_anchor_count = add_line_component_anchor_vertices(
            vertices,
            line_components,
            probabilities.line_prob,
            min_component_pixels=args.line_component_min_pixels,
            min_anchor_distance_px=args.line_anchor_min_distance_px,
            max_anchors_per_component=args.max_line_anchors_per_component,
        )
    merged_vertex_count_before = len(vertices)
    vertices = merge_nearby_vertices(vertices, args.merge_vertex_distance_px)
    port_vertex_count = 0
    if args.enable_node_port_routing:
        vertices, port_vertex_count = add_node_port_vertices(vertices, node_topologies, probabilities.line_prob, args)
    if len(vertices) > args.max_vertices:
        vertices = sorted(vertices, key=lambda vertex: vertex.max_probability, reverse=True)[: args.max_vertices]
        vertices.sort(key=lambda vertex: (vertex.y, vertex.x))
        for i, vertex in enumerate(vertices):
            vertex.id = i

    (
        candidate_edges,
        accepted_before_pruning,
        accepted_edges,
        rejected_by_path_score,
        rejected_by_degree,
        rejected_by_coverage,
        rejected_by_vertex_passthrough,
        rejected_by_node_topology,
        rejected_by_node_routing,
        line_component_pair_count,
        claimed_line_pixels,
        line_coverage_fraction,
        unclaimed_line_pixels,
    ) = build_candidate_edges(
        vertices,
        probabilities.line_prob,
        path_prob,
        line_mask,
        line_components,
        node_topologies,
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
        rejected_by_coverage_pruning_count=rejected_by_coverage,
        rejected_by_vertex_passthrough_count=rejected_by_vertex_passthrough,
        node_component_vertex_count=len(node_vertices),
        line_anchor_vertex_count=line_anchor_count,
        merged_vertex_count_before=merged_vertex_count_before + port_vertex_count,
        line_component_count=len(line_components),
        line_component_candidate_pair_count=line_component_pair_count,
        claimed_line_pixel_count=claimed_line_pixels,
        line_coverage_fraction=line_coverage_fraction,
        unclaimed_line_pixel_count=unclaimed_line_pixels,
        node_topologies=node_topologies,
        rejected_by_node_topology_count=rejected_by_node_topology,
        rejected_by_node_routing_count=rejected_by_node_routing,
        node_port_count=sum(len(blob.ports) for blob in node_topologies),
        allowed_port_route_count=sum(1 for blob in node_topologies for route in blob.routes if route.allowed),
        rejected_port_route_count=sum(1 for blob in node_topologies for route in blob.routes if not route.allowed),
        accepted_edges_using_node_support_count=sum(1 for edge in accepted_edges if edge.uses_node_support),
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
    node_class_counts: dict[str, int] = {}
    incident_counts = []
    port_counts = []
    for blob in graph.node_topologies:
        node_class_counts[blob.class_name] = node_class_counts.get(blob.class_name, 0) + 1
        incident_counts.append(len(blob.incident_directions))
        port_counts.append(len(blob.ports))
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
    if graph.line_mask.any() and graph.line_coverage_fraction < args.line_coverage_warning_fraction:
        warnings.append("Accepted graph paths cover a low fraction of the ML line mask; reconstruction is likely missing visible strokes.")
    if graph.line_component_candidate_pair_count == 0 and graph.line_component_count > 0:
        warnings.append("No extra line-component candidate pairs were added; long curved or long straight edges may be under-connected.")
    if args.enable_node_port_routing and graph.node_topologies:
        zero_port_fraction = sum(1 for count in port_counts if count == 0) / max(len(port_counts), 1)
        if zero_port_fraction > 0.35:
            warnings.append("Many node blobs have no detected ports; node-port routing may be under-connected.")
        if graph.candidate_edges and graph.rejected_by_node_routing_count / max(len(graph.candidate_edges), 1) > 0.45:
            warnings.append("Many candidate edges were rejected by node-port routing; routing thresholds may be too strict.")

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
            "node_component_vertex_count": graph.node_component_vertex_count,
            "line_anchor_vertex_count": graph.line_anchor_vertex_count,
            "merged_vertex_count_before": graph.merged_vertex_count_before,
            "merged_vertex_count_after": len(graph.vertices),
            "line_component_count": graph.line_component_count,
            "line_component_candidate_pair_count": graph.line_component_candidate_pair_count,
            "candidate_edge_count": len(graph.candidate_edges),
            "candidate_edge_count_before_pruning": len(graph.candidate_edges),
            "accepted_edge_count": len(graph.accepted_edges),
            "accepted_edge_count_before_pruning": len(graph.accepted_edges_before_pruning),
            "accepted_edge_count_after_pruning": len(graph.accepted_edges),
            "rejected_edge_count": len(graph.candidate_edges) - len(graph.accepted_edges),
            "rejected_by_path_score_count": graph.rejected_by_path_score_count,
            "rejected_by_degree_pruning_count": graph.rejected_by_degree_pruning_count,
            "rejected_by_coverage_pruning_count": graph.rejected_by_coverage_pruning_count,
            "rejected_by_vertex_passthrough_count": graph.rejected_by_vertex_passthrough_count,
            "rejected_by_node_topology_count": graph.rejected_by_node_topology_count,
            "rejected_by_node_routing_count": graph.rejected_by_node_routing_count,
            "node_mask_pixel_count": int(np.count_nonzero(graph.node_mask)),
            "line_mask_pixel_count": int(np.count_nonzero(graph.line_mask)),
            "claimed_line_pixel_count": graph.claimed_line_pixel_count,
            "unclaimed_line_pixel_count": graph.unclaimed_line_pixel_count,
            "line_coverage_fraction": graph.line_coverage_fraction,
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
            "edge_coverage_gain_pixels": [int(edge.coverage_gain_pixels) for edge in graph.accepted_edges],
            "edge_coverage_overlap_fraction_mean": float(np.mean([edge.coverage_overlap_fraction for edge in graph.accepted_edges]))
            if graph.accepted_edges
            else 0.0,
            "node_blob_count": len(graph.node_topologies),
            "node_blob_class_counts": node_class_counts,
            "endpoint_blob_count": node_class_counts.get("endpoint", 0),
            "sharp_corner_blob_count": node_class_counts.get("sharp_corner", 0),
            "smooth_bend_blob_count": node_class_counts.get("smooth_bend", 0),
            "t_junction_blob_count": node_class_counts.get("t_junction", 0),
            "crossing_or_overlap_blob_count": node_class_counts.get("crossing_or_overlap", 0),
            "multi_junction_blob_count": node_class_counts.get("multi_junction", 0),
            "line_like_node_fragment_count": node_class_counts.get("line_like_node_fragment", 0),
            "noisy_or_ambiguous_blob_count": node_class_counts.get("noisy_blob", 0),
            "average_incident_direction_count": float(np.mean(incident_counts)) if incident_counts else 0.0,
            "max_incident_direction_count": int(max(incident_counts)) if incident_counts else 0,
            "node_port_count": graph.node_port_count,
            "average_ports_per_blob": float(np.mean(port_counts)) if port_counts else 0.0,
            "max_ports_per_blob": int(max(port_counts)) if port_counts else 0,
            "allowed_port_route_count": graph.allowed_port_route_count,
            "rejected_port_route_count": graph.rejected_port_route_count,
            "accepted_edges_using_node_support_count": graph.accepted_edges_using_node_support_count,
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
            "use_line_component_candidates": args.use_line_component_candidates,
            "component_vertex_radius_px": args.component_vertex_radius_px,
            "component_candidate_neighbors": args.component_candidate_neighbors,
            "component_max_edge_distance_px": args.component_max_edge_distance_px,
            "merge_vertex_distance_px": args.merge_vertex_distance_px,
            "vertex_passthrough_radius_px": args.vertex_passthrough_radius_px,
            "coverage_radius_px": args.coverage_radius_px,
            "min_edge_new_pixels": args.min_edge_new_pixels,
            "min_edge_new_coverage_fraction": args.min_edge_new_coverage_fraction,
            "add_line_component_anchors": args.add_line_component_anchors,
            "line_anchor_min_distance_px": args.line_anchor_min_distance_px,
            "max_line_anchors_per_component": args.max_line_anchors_per_component,
            "enable_node_topology": args.enable_node_topology,
            "node_blob_min_area": args.node_blob_min_area,
            "node_incident_radius_px": args.node_incident_radius_px,
            "node_direction_bin_degrees": args.node_direction_bin_degrees,
            "node_support_weight": args.node_support_weight,
            "node_support_radius_px": args.node_support_radius_px,
            "max_node_blob_degree": args.max_node_blob_degree,
            "node_topology_angle_threshold": args.node_topology_angle_threshold,
            "line_like_node_aspect_threshold": args.line_like_node_aspect_threshold,
            "enable_node_port_routing": args.enable_node_port_routing,
            "node_port_radius_px": args.node_port_radius_px,
            "node_port_min_line_prob": args.node_port_min_line_prob,
            "node_port_min_separation_px": args.node_port_min_separation_px,
            "node_port_angle_bin_degrees": args.node_port_angle_bin_degrees,
            "node_route_angle_threshold": args.node_route_angle_threshold,
            "node_route_support_weight": args.node_route_support_weight,
            "node_route_max_pairs": args.node_route_max_pairs,
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
        if "line_anchor" in vertex.source:
            fill = (0, 130, 255, 235)
        else:
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
        if edge.accepted:
            color = (30, 150, 90, 230)
        elif edge.rejected_by_node_routing:
            color = (0, 120, 255, 130)
        elif edge.rejected_by_vertex_passthrough:
            color = (160, 60, 210, 105)
        elif edge.rejected_by_node_topology:
            color = (30, 90, 230, 120)
        elif edge.reason == "rejected by low new line coverage":
            color = (245, 135, 20, 95)
        else:
            color = (220, 60, 60, 80)
        width = 2 if edge.accepted else 1
        if len(points) >= 2:
            draw.line(points, fill=color, width=width)
    for vertex in graph.vertices:
        draw.ellipse((vertex.x - 4, vertex.y - 4, vertex.x + 4, vertex.y + 4), fill=(255, 230, 0, 235), outline=(0, 0, 0, 235))
        draw.text((vertex.x + 5, vertex.y + 3), f"{vertex.id}/{degree.get(vertex.id, 0)}", fill=(0, 0, 0, 255))
    Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB").save(output_path)


def save_node_topology_debug(probabilities: MLProbabilities, graph: MLGraphResult, output_path: Path) -> None:
    image = Image.fromarray(probabilities.gray, mode="L").convert("RGB")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    colors = {
        "endpoint": (40, 120, 255, 190),
        "sharp_corner": (230, 60, 60, 190),
        "smooth_bend": (30, 170, 90, 190),
        "t_junction": (245, 160, 20, 190),
        "crossing_or_overlap": (160, 70, 220, 190),
        "multi_junction": (0, 160, 180, 190),
        "line_like_node_fragment": (220, 80, 170, 190),
        "noisy_blob": (130, 130, 130, 150),
    }
    for blob in graph.node_topologies:
        color = colors.get(blob.class_name, (0, 0, 0, 160))
        for y, x in blob.points_yx:
            draw.point((int(x), int(y)), fill=color)
        cx = float(blob.weighted_centroid_x)
        cy = float(blob.weighted_centroid_y)
        draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=(0, 0, 0, 230))
        for dx, dy in blob.incident_directions:
            draw.line((cx, cy, cx + dx * 16, cy + dy * 16), fill=(0, 0, 0, 210), width=2)
        for port in blob.ports:
            px = float(port.x)
            py = float(port.y)
            dx, dy = port.direction
            draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=(255, 255, 255, 245), outline=(0, 0, 0, 245))
            draw.line((px, py, px + dx * 14, py + dy * 14), fill=(20, 20, 20, 225), width=2)
        draw.text((cx + 5, cy + 3), f"{blob.id}:{blob.class_name}", fill=(0, 0, 0, 255))
    Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB").save(output_path)


def save_node_port_debug(probabilities: MLProbabilities, graph: MLGraphResult, output_path: Path) -> None:
    image = Image.fromarray(probabilities.gray, mode="L").convert("RGB")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for blob in graph.node_topologies:
        cx = float(blob.weighted_centroid_x)
        cy = float(blob.weighted_centroid_y)
        draw.ellipse((cx - 2, cy - 2, cx + 2, cy + 2), fill=(0, 0, 0, 220))
        for port in blob.ports:
            px = float(port.x)
            py = float(port.y)
            dx, dy = port.direction
            confidence_color = int(np.clip(port.confidence * 255, 0, 255))
            fill = (255 - confidence_color, confidence_color, 40, 235)
            draw.line((cx, cy, px, py), fill=(20, 20, 20, 90), width=1)
            draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=fill, outline=(0, 0, 0, 240))
            draw.line((px, py, px + dx * 18, py + dy * 18), fill=(0, 0, 0, 220), width=2)
            draw.text((px + 5, py + 3), f"{blob.id}:{port.id}", fill=(0, 0, 0, 255))
    Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB").save(output_path)


def save_node_routing_debug(probabilities: MLProbabilities, graph: MLGraphResult, output_path: Path) -> None:
    image = Image.fromarray(probabilities.gray, mode="L").convert("RGB")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for blob in graph.node_topologies:
        port_by_id = {port.id: port for port in blob.ports}
        for route in blob.routes:
            port_a = port_by_id.get(route.port_a)
            port_b = port_by_id.get(route.port_b)
            if port_a is None or port_b is None:
                continue
            color = (20, 170, 80, 230) if route.allowed else (220, 60, 60, 90)
            width = 3 if route.allowed else 1
            draw.line((port_a.x, port_a.y, blob.weighted_centroid_x, blob.weighted_centroid_y, port_b.x, port_b.y), fill=color, width=width)
        for port in blob.ports:
            draw.ellipse((port.x - 3, port.y - 3, port.x + 3, port.y + 3), fill=(255, 230, 0, 240), outline=(0, 0, 0, 240))
        draw.text((blob.weighted_centroid_x + 4, blob.weighted_centroid_y + 4), f"{blob.id}:{blob.class_name}", fill=(0, 0, 0, 255))
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
    save_node_topology_debug(probabilities, graph, output_dir / "node_topology_debug.png")
    save_node_port_debug(probabilities, graph, output_dir / "node_port_debug.png")
    save_node_routing_debug(probabilities, graph, output_dir / "node_routing_debug.png")
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
    parser.add_argument("--path-max-expanded-nodes", type=int, default=30000)
    parser.add_argument("--max-edge-distance-px", type=float, default=180.0)
    parser.add_argument("--nearest-neighbors", type=int, default=8)
    parser.add_argument("--edge-corridor-radius-px", type=int, default=2)
    parser.add_argument("--densify-step-px", type=float, default=3.0)
    parser.add_argument("--min-node-area", type=int, default=2)
    parser.add_argument("--large-node-component-area", type=int, default=80)
    parser.add_argument("--max-large-component-vertices", type=int, default=12)
    parser.add_argument("--min-vertex-separation-px", type=float, default=14.0)
    parser.add_argument("--max-vertices", type=int, default=120)
    parser.add_argument("--merge-vertex-distance-px", type=float, default=16.0)
    parser.add_argument("--use-line-component-candidates", action="store_true", default=True)
    parser.add_argument("--disable-line-component-candidates", dest="use_line_component_candidates", action="store_false")
    parser.add_argument("--line-component-min-pixels", type=int, default=12)
    parser.add_argument("--component-vertex-radius-px", type=float, default=18.0)
    parser.add_argument("--component-candidate-neighbors", type=int, default=10)
    parser.add_argument("--component-max-edge-distance-px", type=float, default=1200.0)
    parser.add_argument("--vertex-passthrough-radius-px", type=float, default=18.0)
    parser.add_argument("--coverage-radius-px", type=int, default=2)
    parser.add_argument("--min-edge-new-pixels", type=int, default=12)
    parser.add_argument("--min-edge-new-coverage-fraction", type=float, default=0.30)
    parser.add_argument("--line-coverage-warning-fraction", type=float, default=0.55)
    parser.add_argument("--add-line-component-anchors", action="store_true", default=True)
    parser.add_argument("--disable-line-component-anchors", dest="add_line_component_anchors", action="store_false")
    parser.add_argument("--line-anchor-min-distance-px", type=float, default=35.0)
    parser.add_argument("--max-line-anchors-per-component", type=int, default=4)
    parser.add_argument("--enable-node-topology", action="store_true", help="Use topology-aware node/corner blob decoding.")
    parser.add_argument("--disable-node-topology", dest="enable_node_topology", action="store_false")
    parser.set_defaults(enable_node_topology=False)
    parser.add_argument("--node-blob-min-area", type=int, default=3)
    parser.add_argument("--node-incident-radius-px", type=float, default=18.0)
    parser.add_argument("--node-direction-bin-degrees", type=float, default=22.5)
    parser.add_argument("--node-support-weight", type=float, default=0.35)
    parser.add_argument("--node-support-radius-px", type=float, default=10.0)
    parser.add_argument("--max-node-blob-degree", type=int, default=4)
    parser.add_argument("--node-topology-angle-threshold", type=float, default=135.0)
    parser.add_argument("--line-like-node-aspect-threshold", type=float, default=2.8)
    parser.add_argument("--enable-node-port-routing", action="store_true", help="Use node/corner blob ports as local routing regions.")
    parser.add_argument("--disable-node-port-routing", dest="enable_node_port_routing", action="store_false")
    parser.set_defaults(enable_node_port_routing=False)
    parser.add_argument("--node-port-radius-px", type=float, default=22.0)
    parser.add_argument("--node-port-min-line-prob", type=float, default=0.25)
    parser.add_argument("--node-port-min-separation-px", type=float, default=6.0)
    parser.add_argument("--node-port-angle-bin-degrees", type=float, default=20.0)
    parser.add_argument("--node-route-angle-threshold", type=float, default=55.0)
    parser.add_argument("--node-route-support-weight", type=float, default=0.45)
    parser.add_argument("--node-route-max-pairs", type=int, default=3)
    parser.add_argument("--work-width-mm", type=float, default=150.0)
    parser.add_argument("--work-height-mm", type=float, default=270.0)
    parser.add_argument("--margin-mm", type=float, default=5.0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
