"""
Experimental stroke-based planning pipeline for the XYZ gantry.

This file intentionally stays separate from the contour-based pipeline in
image_to_gantry_coords_serial_ready.py. The canonical gantry mapping, Arduino
command format, command saving, and bounds validation logic are mirrored here
so this module can run even when OpenCV is unavailable.

Conceptual baseline: Raghav et al., "Can I teach a robot to replicate a line
art" - segment raster line art into background/node/line channels, interpret
the channels as a graph, extract strokes, and convert those strokes into robot
drawing commands. This project adapts that idea to Arduino serial commands
instead of G-code.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


Command = tuple[float, float, int]
Pixel = tuple[int, int]  # (y, x)


@dataclass
class SegmentationResult:
    gray: np.ndarray
    line_mask: np.ndarray
    node_mask: np.ndarray
    mode_used: str
    model_path_used: str | None
    notes: list[str]
    diagnostics: dict | None = None


@dataclass
class StrokeGraph:
    vertices: list[dict]
    edges: list[dict]
    raw_strokes_px: list[np.ndarray]
    strokes_px: list[np.ndarray]
    endpoint_mask: np.ndarray
    junction_mask: np.ndarray
    corner_mask: np.ndarray
    rejected_corner_mask: np.ndarray
    skeleton_mask: np.ndarray
    skeleton_component_count: int


class BaseSegmenter(ABC):
    """Segment raster line-art into line and node/corner masks."""

    @abstractmethod
    def segment(self, image_path: Path) -> SegmentationResult:
        raise NotImplementedError


class HeuristicSegmenter(BaseSegmenter):
    """Threshold, clean, skeletonise, and detect structural node candidates."""

    def __init__(self, threshold: int = 127, min_component_area: int = 20):
        self.threshold = threshold
        self.min_component_area = min_component_area

    def segment(self, image_path: Path) -> SegmentationResult:
        gray = load_grayscale(image_path)
        line_mask = gray < self.threshold
        line_mask = close_mask(line_mask, iterations=1)
        line_mask = remove_small_components(line_mask, self.min_component_area)
        skeleton = zhang_suen_thinning(line_mask)
        endpoints, junctions, corners, _ = detect_skeleton_nodes(skeleton)
        node_mask = endpoints | junctions | corners
        return SegmentationResult(
            gray=gray,
            line_mask=line_mask,
            node_mask=node_mask,
            mode_used="heuristic",
            model_path_used=None,
            notes=[
                "Heuristic segmentation used grayscale thresholding, cleanup, "
                "skeletonisation, and endpoint/junction/corner detection."
            ],
            diagnostics={
                "decode_mode": "heuristic_threshold",
                "line_pixel_count": int(np.count_nonzero(line_mask)),
                "node_pixel_count": int(np.count_nonzero(node_mask)),
                "line_pixel_fraction": float(np.count_nonzero(line_mask) / line_mask.size),
            },
        )


class MLSegmenter(BaseSegmenter):
    """
    Lightweight U-Net-style inference wrapper.

    Supported when a local ML framework and a local model are available:
    - PyTorch .pt/.pth, either TorchScript or a state_dict for TinyUNet
    - TensorFlow/Keras .h5/.keras

    Expected model output is either one channel (line probability) or a
    multiclass output. For multiclass checkpoints, class_map metadata is used
    when present; otherwise the default order is background, line, node/corner.
    """

    def __init__(self, model_path: Path, threshold: float = 0.5, ml_probability_threshold: float = 0.35):
        self.model_path = model_path
        self.threshold = threshold
        self.ml_probability_threshold = ml_probability_threshold

    @staticmethod
    def framework_available_for(model_path: Path) -> bool:
        suffix = model_path.suffix.lower()
        if suffix in {".pt", ".pth"}:
            return import_available("torch")
        if suffix in {".h5", ".keras", ".pb"}:
            return import_available("tensorflow")
        return False

    def segment(self, image_path: Path) -> SegmentationResult:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {self.model_path}")

        suffix = self.model_path.suffix.lower()
        if suffix in {".pt", ".pth"}:
            return self._segment_torch(image_path)
        if suffix in {".h5", ".keras"}:
            return self._segment_tensorflow(image_path)

        raise ValueError(f"Unsupported model format: {self.model_path.suffix}")

    def _segment_torch(self, image_path: Path) -> SegmentationResult:
        import torch

        from stroke_ml_model import build_stroke_unet

        gray = load_grayscale(image_path)
        tensor = torch.from_numpy(gray.astype(np.float32) / 255.0)[None, None, :, :]
        class_map = None

        try:
            model = torch.jit.load(str(self.model_path), map_location="cpu")
        except Exception:
            checkpoint = torch.load(str(self.model_path), map_location="cpu")
            model_config = checkpoint.get("model_config", {}) if isinstance(checkpoint, dict) else {}
            class_map = checkpoint.get("class_map") if isinstance(checkpoint, dict) else None
            model = build_stroke_unet(
                num_classes=int(model_config.get("num_classes", 3)),
                base_channels=int(model_config.get("base_channels", 16)),
            )
            if isinstance(checkpoint, dict):
                state_dict = (
                    checkpoint.get("model_state_dict")
                    or checkpoint.get("state_dict")
                    or checkpoint
                )
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict)

        model.eval()
        with torch.no_grad():
            output = model(tensor)
            output_np = output.detach().cpu().numpy()
            probs = output_to_probabilities(output_np)

        line_mask, node_mask, diagnostics = masks_from_model_probabilities(
            probs,
            self.threshold,
            class_map=class_map,
            output_shape=tuple(output_np.shape),
            fallback_probability_threshold=self.ml_probability_threshold,
        )
        return SegmentationResult(
            gray=gray,
            line_mask=line_mask,
            node_mask=node_mask,
            mode_used="ML",
            model_path_used=str(self.model_path),
            notes=["PyTorch U-Net-style segmentation inference completed."],
            diagnostics=diagnostics,
        )

    def _segment_tensorflow(self, image_path: Path) -> SegmentationResult:
        import tensorflow as tf

        gray = load_grayscale(image_path)
        tensor = gray.astype(np.float32)[None, :, :, None] / 255.0
        model = tf.keras.models.load_model(str(self.model_path), compile=False)
        output = model(tensor, training=False).numpy()
        probs = output_to_probabilities(output)
        line_mask, node_mask, diagnostics = masks_from_model_probabilities(
            probs,
            self.threshold,
            class_map=None,
            output_shape=tuple(output.shape),
            fallback_probability_threshold=self.ml_probability_threshold,
        )
        return SegmentationResult(
            gray=gray,
            line_mask=line_mask,
            node_mask=node_mask,
            mode_used="ML",
            model_path_used=str(self.model_path),
            notes=["TensorFlow/Keras segmentation inference completed."],
            diagnostics=diagnostics,
        )


def import_available(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except Exception:
        return False


def output_to_probabilities(output: np.ndarray) -> np.ndarray:
    output = np.asarray(output)
    if output.ndim == 4:
        output = output[0]
    if output.ndim == 3 and output.shape[0] <= 4:
        channels_first = output
    elif output.ndim == 3:
        channels_first = np.moveaxis(output, -1, 0)
    elif output.ndim == 2:
        channels_first = output[None, :, :]
    else:
        raise ValueError(f"Unexpected model output shape: {output.shape}")

    if channels_first.shape[0] > 1:
        shifted = channels_first - np.max(channels_first, axis=0, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.maximum(np.sum(exp, axis=0, keepdims=True), 1e-9)

    return 1.0 / (1.0 + np.exp(-channels_first))


def normalize_class_map(class_map: dict | None) -> dict[int, str]:
    if not class_map:
        return {0: "background", 1: "line", 2: "node_corner"}
    normalized = {}
    for key, value in class_map.items():
        try:
            index = int(key)
        except (TypeError, ValueError):
            index = int(value)
            value = key
        normalized[index] = str(value).lower()
    return normalized


def class_index_for(class_map: dict[int, str], names: tuple[str, ...], default: int) -> int:
    for index, label in class_map.items():
        if any(name in label for name in names):
            return index
    return default


def masks_from_model_probabilities(
    probs: np.ndarray,
    threshold: float,
    class_map: dict | None = None,
    output_shape: tuple[int, ...] | None = None,
    fallback_probability_threshold: float = 0.35,
) -> tuple[np.ndarray, np.ndarray, dict]:
    probability_threshold = threshold / 255.0 if threshold > 1 else threshold
    if probs.shape[0] >= 3:
        normalized_class_map = normalize_class_map(class_map)
        line_class = class_index_for(normalized_class_map, ("line",), default=1)
        node_class = class_index_for(normalized_class_map, ("node", "corner"), default=2)
        predicted = np.argmax(probs, axis=0)
        line_mask = predicted == line_class
        node_mask = predicted == node_class
        decode_mode = "multiclass_argmax"
        if not np.any(line_mask) and line_class < probs.shape[0]:
            probability_line_mask = probs[line_class] >= fallback_probability_threshold
            if np.any(probability_line_mask):
                line_mask = probability_line_mask
                if node_class < probs.shape[0]:
                    node_mask = probs[node_class] >= fallback_probability_threshold
                    node_mask &= ~line_mask
                decode_mode = "multiclass_argmax_empty_line_probability_fallback"
        predicted_counts = {
            str(index): int(np.count_nonzero(predicted == index))
            for index in range(probs.shape[0])
        }
        diagnostics = {
            "decode_mode": decode_mode,
            "output_shape": list(output_shape) if output_shape is not None else None,
            "probability_shape": list(probs.shape),
            "class_map": normalized_class_map,
            "line_class_index": line_class,
            "node_class_index": node_class,
            "argmax_class_pixel_counts": predicted_counts,
            "line_pixel_count": int(np.count_nonzero(line_mask)),
            "node_pixel_count": int(np.count_nonzero(node_mask)),
            "line_pixel_fraction": float(np.count_nonzero(line_mask) / line_mask.size),
            "line_probability_max": float(np.max(probs[line_class])) if line_class < probs.shape[0] else None,
            "line_probability_mean": float(np.mean(probs[line_class])) if line_class < probs.shape[0] else None,
            "node_probability_max": float(np.max(probs[node_class])) if node_class < probs.shape[0] else None,
            "node_probability_mean": float(np.mean(probs[node_class])) if node_class < probs.shape[0] else None,
            "background_probability_mean": float(np.mean(probs[0])) if probs.shape[0] > 0 else None,
            "threshold": threshold,
            "threshold_used_for_decode": None,
            "fallback_probability_threshold": fallback_probability_threshold,
        }
    else:
        line_mask = probs[0] >= probability_threshold
        skeleton = zhang_suen_thinning(line_mask)
        endpoints, junctions, corners, _ = detect_skeleton_nodes(skeleton)
        node_mask = endpoints | junctions | corners
        diagnostics = {
            "decode_mode": "single_channel_threshold",
            "output_shape": list(output_shape) if output_shape is not None else None,
            "probability_shape": list(probs.shape),
            "line_pixel_count": int(np.count_nonzero(line_mask)),
            "node_pixel_count": int(np.count_nonzero(node_mask)),
            "line_pixel_fraction": float(np.count_nonzero(line_mask) / line_mask.size),
            "line_probability_max": float(np.max(probs[0])),
            "threshold": threshold,
            "threshold_used_for_decode": probability_threshold,
        }
    return line_mask, node_mask, diagnostics


def load_grayscale(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8)


def dilate_mask(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    result = mask.astype(bool)
    for _ in range(iterations):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        out = np.zeros_like(result, dtype=bool)
        for dy in range(3):
            for dx in range(3):
                out |= padded[dy : dy + result.shape[0], dx : dx + result.shape[1]]
        result = out
    return result


def erode_mask(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    result = mask.astype(bool)
    for _ in range(iterations):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        out = np.ones_like(result, dtype=bool)
        for dy in range(3):
            for dx in range(3):
                out &= padded[dy : dy + result.shape[0], dx : dx + result.shape[1]]
        result = out
    return result


def close_mask(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    result = mask.astype(bool)
    for _ in range(iterations):
        result = erode_mask(dilate_mask(result, 1), 1)
    return result


def connected_components(mask: np.ndarray, connectivity: int = 8) -> list[list[Pixel]]:
    mask = mask.astype(bool)
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[list[Pixel]] = []
    directions = neighbor_directions(connectivity)

    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if visited[y0, x0]:
            continue
        stack = [(y0, x0)]
        visited[y0, x0] = True
        component: list[Pixel] = []
        while stack:
            y, x = stack.pop()
            component.append((y, x))
            for dy, dx in directions:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        components.append(component)
    return components


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 1:
        return mask.astype(bool)
    cleaned = np.zeros_like(mask, dtype=bool)
    for component in connected_components(mask, connectivity=8):
        if len(component) >= min_area:
            ys = [p[0] for p in component]
            xs = [p[1] for p in component]
            cleaned[ys, xs] = True
    return cleaned


def neighbor_directions(connectivity: int = 8) -> list[Pixel]:
    if connectivity == 4:
        return [(-1, 0), (0, 1), (1, 0), (0, -1)]
    return [
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
    ]


def shifted_neighbors(mask: np.ndarray) -> list[np.ndarray]:
    padded = np.pad(mask.astype(bool), 1, mode="constant", constant_values=False)
    height, width = mask.shape
    return [
        padded[0:height, 1 : width + 1],  # p2 north
        padded[0:height, 2 : width + 2],  # p3 north-east
        padded[1 : height + 1, 2 : width + 2],  # p4 east
        padded[2 : height + 2, 2 : width + 2],  # p5 south-east
        padded[2 : height + 2, 1 : width + 1],  # p6 south
        padded[2 : height + 2, 0:width],  # p7 south-west
        padded[1 : height + 1, 0:width],  # p8 west
        padded[0:height, 0:width],  # p9 north-west
    ]


def transition_count(neighbors: list[np.ndarray]) -> np.ndarray:
    total = np.zeros_like(neighbors[0], dtype=np.uint8)
    for current, nxt in zip(neighbors, neighbors[1:] + neighbors[:1]):
        total += (~current & nxt).astype(np.uint8)
    return total


def zhang_suen_thinning(mask: np.ndarray, max_iterations: int = 1000) -> np.ndarray:
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


def neighbor_count(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(bool), 1, mode="constant", constant_values=False)
    height, width = mask.shape
    count = np.zeros_like(mask, dtype=np.uint8)
    for dy in range(3):
        for dx in range(3):
            if dy == 1 and dx == 1:
                continue
            count += padded[dy : dy + height, dx : dx + width].astype(np.uint8)
    return count


def local_branch_direction(
    skeleton: np.ndarray,
    start: Pixel,
    previous: Pixel,
    max_steps: int,
) -> np.ndarray | None:
    current = start
    last = previous
    for _ in range(max_steps):
        candidates = [pixel for pixel in pixel_neighbors(current, skeleton) if pixel != last]
        if len(candidates) != 1:
            break
        last, current = current, candidates[0]

    vector = np.array([current[1] - previous[1], current[0] - previous[0]], dtype=np.float64)
    norm = np.linalg.norm(vector)
    if norm < 1e-9:
        return None
    return vector / norm


def detect_corner_candidates(
    skeleton: np.ndarray,
    degree: np.ndarray,
    angle_threshold_deg: float,
    lookahead: int,
) -> tuple[np.ndarray, np.ndarray]:
    candidates = np.zeros_like(skeleton, dtype=bool)
    scores = np.full(skeleton.shape, np.inf, dtype=np.float64)

    ys, xs = np.nonzero(skeleton & (degree == 2))
    for y, x in zip(ys.tolist(), xs.tolist()):
        pixel = (y, x)
        neighbors = pixel_neighbors(pixel, skeleton)
        if len(neighbors) != 2:
            continue

        d0 = local_branch_direction(skeleton, neighbors[0], pixel, lookahead)
        d1 = local_branch_direction(skeleton, neighbors[1], pixel, lookahead)
        if d0 is None or d1 is None:
            continue

        cosang = float(np.clip(np.dot(d0, d1), -1.0, 1.0))
        angle = math.degrees(math.acos(cosang))
        if angle <= angle_threshold_deg:
            candidates[y, x] = True
            scores[y, x] = angle

    return candidates, scores


def suppress_nearby_corners(
    candidates: np.ndarray,
    scores: np.ndarray,
    min_separation_px: int,
) -> np.ndarray:
    accepted = np.zeros_like(candidates, dtype=bool)
    candidate_points = np.argwhere(candidates)
    if len(candidate_points) == 0:
        return accepted

    order = sorted(
        [(float(scores[y, x]), int(y), int(x)) for y, x in candidate_points],
        key=lambda item: item[0],
    )
    accepted_points: list[tuple[int, int]] = []
    min_sep2 = float(min_separation_px * min_separation_px)
    for _, y, x in order:
        too_close = any((y - ay) ** 2 + (x - ax) ** 2 < min_sep2 for ay, ax in accepted_points)
        if too_close:
            continue
        accepted[y, x] = True
        accepted_points.append((y, x))

    return accepted


def detect_skeleton_nodes(
    skeleton: np.ndarray,
    corner_angle_threshold_deg: float = 100.0,
    corner_min_separation_px: int = 12,
    corner_lookahead_px: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    skeleton = skeleton.astype(bool)
    count = neighbor_count(skeleton)
    endpoints = skeleton & (count == 1)
    junctions = skeleton & (count >= 3)
    corner_candidates, corner_scores = detect_corner_candidates(
        skeleton,
        degree=count,
        angle_threshold_deg=corner_angle_threshold_deg,
        lookahead=corner_lookahead_px,
    )
    corners = suppress_nearby_corners(corner_candidates, corner_scores, corner_min_separation_px)
    rejected_corners = corner_candidates & ~corners
    return endpoints, junctions, corners, rejected_corners


def cluster_vertices(node_mask: np.ndarray, skeleton: np.ndarray, radius: int = 1) -> tuple[list[dict], np.ndarray]:
    if not np.any(node_mask):
        ys, xs = np.nonzero(skeleton)
        if len(ys) == 0:
            return [], np.full(skeleton.shape, -1, dtype=np.int32)
        node_mask = np.zeros_like(skeleton, dtype=bool)
        idx = int(np.argmin(ys * skeleton.shape[1] + xs))
        node_mask[ys[idx], xs[idx]] = True

    expanded_nodes = dilate_mask(node_mask, iterations=radius)
    label_map = np.full(skeleton.shape, -1, dtype=np.int32)
    vertices: list[dict] = []

    for component in connected_components(expanded_nodes, connectivity=8):
        component_mask = np.zeros_like(skeleton, dtype=bool)
        ys = [p[0] for p in component]
        xs = [p[1] for p in component]
        component_mask[ys, xs] = True
        skeleton_pixels = np.argwhere(component_mask & skeleton)
        node_pixels = np.argwhere(component_mask & node_mask)
        points = node_pixels if len(node_pixels) > 0 else skeleton_pixels
        if len(points) == 0:
            continue

        centroid_yx = np.mean(points, axis=0)
        nearest_idx = int(np.argmin(np.sum((skeleton_pixels - centroid_yx) ** 2, axis=1)))
        representative_y, representative_x = skeleton_pixels[nearest_idx].tolist()
        vertex_id = len(vertices)
        label_pixels = np.argwhere(component_mask & skeleton)
        label_map[label_pixels[:, 0], label_pixels[:, 1]] = vertex_id
        vertices.append(
            {
                "id": vertex_id,
                "x_px": float(representative_x),
                "y_px": float(representative_y),
                "pixel_count": int(len(label_pixels)),
            }
        )

    return vertices, label_map


def pixel_neighbors(pixel: Pixel, skeleton: np.ndarray) -> list[Pixel]:
    y, x = pixel
    height, width = skeleton.shape
    result = []
    for dy, dx in neighbor_directions(8):
        ny, nx = y + dy, x + dx
        if 0 <= ny < height and 0 <= nx < width and skeleton[ny, nx]:
            result.append((ny, nx))
    return result


def edge_key(a: Pixel, b: Pixel) -> tuple[Pixel, Pixel]:
    return tuple(sorted((a, b)))  # type: ignore[return-value]


def trace_graph_edges(skeleton: np.ndarray, label_map: np.ndarray, vertices: list[dict]) -> list[dict]:
    visited_links: set[tuple[Pixel, Pixel]] = set()
    edges: list[dict] = []
    vertex_pixels: dict[int, list[Pixel]] = {v["id"]: [] for v in vertices}

    ys, xs = np.nonzero((label_map >= 0) & skeleton)
    for y, x in zip(ys.tolist(), xs.tolist()):
        vertex_pixels[int(label_map[y, x])].append((y, x))

    for vertex in vertices:
        start_vertex = int(vertex["id"])
        for start_pixel in vertex_pixels.get(start_vertex, []):
            for nxt in pixel_neighbors(start_pixel, skeleton):
                if int(label_map[nxt]) == start_vertex:
                    continue
                first_key = edge_key(start_pixel, nxt)
                if first_key in visited_links:
                    continue

                path = [start_pixel, nxt]
                visited_links.add(first_key)
                previous = start_pixel
                current = nxt
                end_vertex = int(label_map[current]) if label_map[current] >= 0 else None

                while end_vertex is None:
                    candidates = [p for p in pixel_neighbors(current, skeleton) if p != previous]
                    unvisited = [p for p in candidates if edge_key(current, p) not in visited_links]
                    if unvisited:
                        candidates = unvisited
                    if not candidates:
                        break
                    next_pixel = candidates[0]
                    visited_links.add(edge_key(current, next_pixel))
                    previous, current = current, next_pixel
                    path.append(current)
                    label = int(label_map[current])
                    if label >= 0:
                        end_vertex = label

                if len(path) >= 2:
                    edge_id = len(edges)
                    edges.append(
                        {
                            "id": edge_id,
                            "start_vertex": start_vertex,
                            "end_vertex": end_vertex,
                            "path_px": path_to_xy_array(path),
                            "pixel_length": float(path_pixel_length(path)),
                        }
                    )

    return edges


def build_edge_adjacency(edges: list[dict], vertex_count: int) -> dict[int, list[int]]:
    adjacency = {vertex_id: [] for vertex_id in range(vertex_count)}
    for edge in edges:
        start_vertex = edge.get("start_vertex")
        end_vertex = edge.get("end_vertex")
        if start_vertex is not None and start_vertex >= 0:
            adjacency[int(start_vertex)].append(int(edge["id"]))
        if end_vertex is not None and end_vertex >= 0 and end_vertex != start_vertex:
            adjacency[int(end_vertex)].append(int(edge["id"]))
    return adjacency


def oriented_edge_path(edge: dict, from_vertex: int | None) -> tuple[np.ndarray, int | None]:
    path = edge["path_px"]
    start_vertex = edge.get("start_vertex")
    end_vertex = edge.get("end_vertex")
    if from_vertex is not None and end_vertex == from_vertex:
        return path[::-1].copy(), start_vertex
    return path.copy(), end_vertex


def unit_vector(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        return None
    return vector / norm


def path_end_direction(path: np.ndarray) -> np.ndarray | None:
    if len(path) < 2:
        return None
    for i in range(len(path) - 2, -1, -1):
        vector = path[-1] - path[i]
        direction = unit_vector(vector)
        if direction is not None:
            return direction
    return None


def path_start_direction(path: np.ndarray) -> np.ndarray | None:
    if len(path) < 2:
        return None
    for i in range(1, len(path)):
        vector = path[i] - path[0]
        direction = unit_vector(vector)
        if direction is not None:
            return direction
    return None


def choose_smoothest_edge(
    edges: list[dict],
    candidate_edge_ids: list[int],
    current_vertex: int,
    incoming_direction: np.ndarray | None,
    max_join_angle_deg: float,
) -> int | None:
    if not candidate_edge_ids:
        return None
    if incoming_direction is None:
        return candidate_edge_ids[0]

    best_edge_id = None
    best_angle = float("inf")
    for edge_id in candidate_edge_ids:
        candidate_path, _ = oriented_edge_path(edges[edge_id], current_vertex)
        candidate_direction = path_start_direction(candidate_path)
        if candidate_direction is None:
            continue
        cosang = float(np.clip(np.dot(incoming_direction, candidate_direction), -1.0, 1.0))
        angle = math.degrees(math.acos(cosang))
        if angle < best_angle:
            best_angle = angle
            best_edge_id = edge_id

    if best_edge_id is None or best_angle > max_join_angle_deg:
        return None
    return best_edge_id


def append_path_without_duplicate(base: list[list[float]], path: np.ndarray) -> None:
    for point in path.tolist():
        if base and np.allclose(np.array(base[-1]), np.array(point)):
            continue
        base.append(point)


def assemble_continuous_strokes(
    edges: list[dict],
    vertices: list[dict],
    junction_join_angle_deg: float = 35.0,
) -> list[np.ndarray]:
    """
    Merge traced graph edges into longer drawable trails.

    Degree-2 vertices are treated as pass-through points, which joins corners
    and tiny adjacent edge fragments into one ordered polyline. Vertices with
    degree 1 or degree >= 3 remain natural stroke boundaries.
    """
    if not edges:
        return []

    adjacency = build_edge_adjacency(edges, len(vertices))
    degrees = {vertex_id: len(edge_ids) for vertex_id, edge_ids in adjacency.items()}
    used_edges: set[int] = set()
    strokes: list[np.ndarray] = []

    def walk_from(start_vertex: int, first_edge_id: int) -> np.ndarray:
        points: list[list[float]] = []
        current_vertex: int | None = start_vertex
        edge_id = first_edge_id
        incoming_direction: np.ndarray | None = None

        while True:
            used_edges.add(edge_id)
            path, next_vertex = oriented_edge_path(edges[edge_id], current_vertex)
            append_path_without_duplicate(points, path)
            incoming_direction = path_end_direction(path)

            if next_vertex is None:
                break

            candidates = [candidate for candidate in adjacency[next_vertex] if candidate not in used_edges]
            if not candidates:
                break
            if degrees.get(next_vertex, 0) == 2:
                next_edge_id = candidates[0]
            else:
                next_edge_id = choose_smoothest_edge(
                    edges,
                    candidates,
                    current_vertex=next_vertex,
                    incoming_direction=incoming_direction,
                    max_join_angle_deg=junction_join_angle_deg,
                )
                if next_edge_id is None:
                    break
            current_vertex = next_vertex
            edge_id = next_edge_id

        return np.array(points, dtype=np.float64)

    boundary_vertices = [
        vertex_id for vertex_id, degree in degrees.items()
        if degree != 2 and degree > 0
    ]
    for vertex_id in boundary_vertices:
        for edge_id in adjacency[vertex_id]:
            if edge_id not in used_edges:
                stroke = walk_from(vertex_id, edge_id)
                if len(stroke) >= 2:
                    strokes.append(stroke)

    # Remaining unused edges are closed loops or all-degree-2 components.
    for edge in edges:
        edge_id = int(edge["id"])
        if edge_id in used_edges:
            continue
        start_vertex = edge.get("start_vertex")
        if start_vertex is None:
            path = edge["path_px"]
            used_edges.add(edge_id)
            if len(path) >= 2:
                strokes.append(path.copy())
            continue

        points: list[list[float]] = []
        current_vertex = int(start_vertex)
        current_edge_id = edge_id
        while current_edge_id not in used_edges:
            used_edges.add(current_edge_id)
            path, next_vertex = oriented_edge_path(edges[current_edge_id], current_vertex)
            append_path_without_duplicate(points, path)
            if next_vertex is None:
                break
            current_vertex = int(next_vertex)
            candidates = [
                candidate for candidate in adjacency[current_vertex]
                if candidate not in used_edges
            ]
            if not candidates:
                break
            if len(candidates) == 1:
                current_edge_id = candidates[0]
            else:
                incoming_direction = path_end_direction(path)
                next_edge_id = choose_smoothest_edge(
                    edges,
                    candidates,
                    current_vertex=current_vertex,
                    incoming_direction=incoming_direction,
                    max_join_angle_deg=junction_join_angle_deg,
                )
                if next_edge_id is None:
                    break
                current_edge_id = next_edge_id

        if len(points) >= 2:
            first = np.array(points[0])
            last = np.array(points[-1])
            if np.linalg.norm(first - last) <= 2.0:
                points[-1] = points[0]
            strokes.append(np.array(points, dtype=np.float64))

    return strokes


def path_to_xy_array(path: list[Pixel]) -> np.ndarray:
    return np.array([[x, y] for y, x in path], dtype=np.float64)


def path_pixel_length(path: list[Pixel]) -> float:
    if len(path) < 2:
        return 0.0
    total = 0.0
    for (y0, x0), (y1, x1) in zip(path[:-1], path[1:]):
        total += math.hypot(x1 - x0, y1 - y0)
    return total


def path_closed(path: np.ndarray, tolerance_px: float = 6.0) -> bool:
    if len(path) < 3:
        return False
    return float(np.linalg.norm(path[0] - path[-1])) <= tolerance_px


def simplify_stroke_preserving_loop(points: np.ndarray, epsilon: float) -> np.ndarray:
    simplified = simplify_polyline(points, epsilon=epsilon)
    if path_closed(points) and len(simplified) > 2 and not np.allclose(simplified[0], simplified[-1]):
        simplified = np.vstack([simplified, simplified[0]])
    return simplified


def order_strokes_nearest_neighbor(strokes: list[np.ndarray]) -> list[np.ndarray]:
    remaining = [stroke.copy() for stroke in strokes if len(stroke) >= 2]
    if not remaining:
        return []

    ordered = [remaining.pop(0)]
    while remaining:
        current_end = ordered[-1][-1]
        best_index = 0
        best_reverse = False
        best_distance = float("inf")
        for i, stroke in enumerate(remaining):
            distance_start = float(np.linalg.norm(current_end - stroke[0]))
            distance_end = float(np.linalg.norm(current_end - stroke[-1]))
            if distance_start < best_distance:
                best_distance = distance_start
                best_index = i
                best_reverse = False
            if distance_end < best_distance:
                best_distance = distance_end
                best_index = i
                best_reverse = True

        next_stroke = remaining.pop(best_index)
        if best_reverse and not path_closed(next_stroke):
            next_stroke = next_stroke[::-1].copy()
        ordered.append(next_stroke)

    return ordered


def build_stroke_graph(
    line_mask: np.ndarray,
    node_hint_mask: np.ndarray | None = None,
    split_corners: bool = False,
    use_node_hints_as_vertices: bool = False,
    corner_angle_threshold_deg: float = 100.0,
    corner_min_separation_px: int = 12,
    corner_lookahead_px: int = 8,
    simplification_epsilon: float = 0.75,
    junction_join_angle_deg: float = 35.0,
) -> StrokeGraph:
    skeleton = zhang_suen_thinning(line_mask)
    endpoints, junctions, corners, rejected_corners = detect_skeleton_nodes(
        skeleton,
        corner_angle_threshold_deg=corner_angle_threshold_deg,
        corner_min_separation_px=corner_min_separation_px,
        corner_lookahead_px=corner_lookahead_px,
    )
    node_mask = endpoints | junctions
    if split_corners:
        node_mask |= corners
    if use_node_hints_as_vertices and node_hint_mask is not None:
        node_mask |= node_hint_mask.astype(bool) & skeleton

    vertices, label_map = cluster_vertices(node_mask, skeleton, radius=1)
    edges = trace_graph_edges(skeleton, label_map, vertices)
    continuous_strokes = assemble_continuous_strokes(
        edges,
        vertices,
        junction_join_angle_deg=junction_join_angle_deg,
    )
    raw_strokes = order_strokes_nearest_neighbor(continuous_strokes)
    strokes = [
        simplify_stroke_preserving_loop(stroke, epsilon=simplification_epsilon)
        for stroke in continuous_strokes
        if len(stroke) >= 2
    ]
    strokes = order_strokes_nearest_neighbor(strokes)

    return StrokeGraph(
        vertices=vertices,
        edges=edges,
        raw_strokes_px=raw_strokes,
        strokes_px=strokes,
        endpoint_mask=endpoints,
        junction_mask=junctions,
        corner_mask=corners,
        rejected_corner_mask=rejected_corners,
        skeleton_mask=skeleton,
        skeleton_component_count=len(connected_components(skeleton, connectivity=8)),
    )


def simplify_polyline(points: np.ndarray, epsilon: float) -> np.ndarray:
    if len(points) <= 2:
        return points.copy()

    start = points[0]
    end = points[-1]
    line = end - start
    line_len = float(np.linalg.norm(line))
    if line_len < 1e-9:
        distances = np.linalg.norm(points - start, axis=1)
    else:
        relative = start - points
        distances = np.abs((line[0] * relative[:, 1]) - (line[1] * relative[:, 0])) / line_len

    index = int(np.argmax(distances))
    max_distance = float(distances[index])
    if max_distance > epsilon:
        left = simplify_polyline(points[: index + 1], epsilon)
        right = simplify_polyline(points[index:], epsilon)
        return np.vstack([left[:-1], right])
    return np.vstack([start, end])


def map_paths_to_gantry_mm(
    paths_px: list[np.ndarray],
    image_shape: tuple[int, int],
    work_w_mm: float = 150.0,
    work_h_mm: float = 270.0,
    margin_mm: float = 5.0,
    centre_on_page: bool = True,
) -> tuple[list[np.ndarray], dict]:
    """
    Mirrored from the contour pipeline: top-left origin, +X right, +Y down.
    """
    img_h_px, img_w_px = image_shape
    drawable_w_mm = work_w_mm - 2 * margin_mm
    drawable_h_mm = work_h_mm - 2 * margin_mm
    if drawable_w_mm <= 0 or drawable_h_mm <= 0:
        raise ValueError("Margins are too large for the gantry work area.")

    scale = min(drawable_w_mm / img_w_px, drawable_h_mm / img_h_px)
    used_w_mm = img_w_px * scale
    used_h_mm = img_h_px * scale

    if centre_on_page:
        x_offset_mm = margin_mm + (drawable_w_mm - used_w_mm) / 2.0
        y_offset_mm = margin_mm + (drawable_h_mm - used_h_mm) / 2.0
    else:
        x_offset_mm = margin_mm
        y_offset_mm = margin_mm

    mapped_paths = []
    for path_px in paths_px:
        if len(path_px) == 0:
            mapped_paths.append(path_px.copy())
            continue
        x_mm = path_px[:, 0] * scale + x_offset_mm
        y_mm = path_px[:, 1] * scale + y_offset_mm
        mapped_paths.append(np.column_stack([x_mm, y_mm]))

    return mapped_paths, {
        "work_w_mm": work_w_mm,
        "work_h_mm": work_h_mm,
        "margin_mm": margin_mm,
        "drawable_w_mm": drawable_w_mm,
        "drawable_h_mm": drawable_h_mm,
        "img_w_px": img_w_px,
        "img_h_px": img_h_px,
        "scale_mm_per_px": scale,
        "used_w_mm": used_w_mm,
        "used_h_mm": used_h_mm,
        "x_offset_mm": x_offset_mm,
        "y_offset_mm": y_offset_mm,
        "centre_on_page": centre_on_page,
    }


def paths_to_arduino_commands(paths_mm: list[np.ndarray]) -> list[Command]:
    """Mirrored command format: x_mm,y_mm,mode where 0=travel and 1=draw."""
    commands: list[Command] = []
    for path in paths_mm:
        if len(path) == 0:
            continue
        commands.append((float(path[0, 0]), float(path[0, 1]), 0))
        for point in path[1:]:
            commands.append((float(point[0]), float(point[1]), 1))
    return commands


def save_arduino_commands(commands: list[Command], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# x_mm,y_mm,mode\n")
        handle.write("# mode 0 = travel / pen up\n")
        handle.write("# mode 1 = draw / pen down\n")
        for x, y, mode in commands:
            handle.write(f"{x:.2f},{y:.2f},{mode}\n")


def validate_arduino_commands(commands: list[Command], work_w_mm: float, work_h_mm: float) -> None:
    for i, (x, y, mode) in enumerate(commands):
        if mode not in (0, 1):
            raise ValueError(f"Command {i} has invalid mode: {mode}")
        if not (0.0 <= x <= work_w_mm and 0.0 <= y <= work_h_mm):
            raise ValueError(
                f"Command {i} outside gantry bounds: x={x:.2f}, y={y:.2f}, "
                f"allowed X=0..{work_w_mm}, Y=0..{work_h_mm}"
            )


def parse_firmware_constants(firmware_path: Path) -> dict:
    defaults = {
        "BELT_PITCH_MM": 2.0,
        "PULLEY_TEETH": 20.0,
        "FULL_STEPS_PER_REV": 200.0,
        "MICROSTEPS": 16.0,
        "DRAW_SPEED": 1100.0,
        "TRAVEL_SPEED": 1800.0,
        "SERVO_DELAY_MS": 15.0,
        "PEN_SETTLE_MS": 1000.0,
        "PEN_UP_ANGLE": 95.0,
        "PEN_DOWN_ANGLE": 80.0,
    }
    constants = defaults.copy()
    if not firmware_path.exists():
        constants["firmware_parse_warning"] = f"Firmware not found: {firmware_path}"
        return constants

    pattern = re.compile(r"const\s+(?:float|int)\s+([A-Z0-9_]+)\s*=\s*([-+]?\d+(?:\.\d+)?)\s*;")
    for match in pattern.finditer(firmware_path.read_text(encoding="utf-8", errors="ignore")):
        name, value = match.groups()
        if name in constants:
            constants[name] = float(value)
    return constants


def compute_command_metrics(
    commands: list[Command],
    stroke_count: int,
    work_w_mm: float,
    work_h_mm: float,
    firmware_constants: dict,
    segmentation_mode_used: str,
    model_path_used: str | None,
    transform_info: dict,
    graph: StrokeGraph,
    segmentation_notes: list[str],
    segmentation_diagnostics: dict | None = None,
) -> dict:
    draw_distance = 0.0
    travel_distance = 0.0
    previous: Command | None = None
    previous_mode: int | None = None
    mode_changes = 0

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

    if commands:
        xs = [c[0] for c in commands]
        ys = [c[1] for c in commands]
        bounds = {
            "min_x_mm": min(xs),
            "max_x_mm": max(xs),
            "min_y_mm": min(ys),
            "max_y_mm": max(ys),
        }
    else:
        bounds = {"min_x_mm": None, "max_x_mm": None, "min_y_mm": None, "max_y_mm": None}

    try:
        validate_arduino_commands(commands, work_w_mm, work_h_mm)
        bounds_pass = True
        bounds_error = None
    except ValueError as exc:
        bounds_pass = False
        bounds_error = str(exc)

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
    total_time_s = (draw_time_s or 0.0) + (travel_time_s or 0.0) + pen_change_time_s
    stroke_point_counts = [int(len(stroke)) for stroke in graph.strokes_px]
    raw_stroke_point_counts = [int(len(stroke)) for stroke in graph.raw_strokes_px]
    two_point_strokes = sum(1 for count in stroke_point_counts if count == 2)
    two_point_fraction = two_point_strokes / len(stroke_point_counts) if stroke_point_counts else 0.0
    closed_loop_strokes = sum(1 for stroke in graph.strokes_px if path_closed(stroke))
    skeleton_pixel_count = int(np.count_nonzero(graph.skeleton_mask))
    accepted_corner_count = int(np.count_nonzero(graph.corner_mask))
    rejected_corner_count = int(np.count_nonzero(graph.rejected_corner_mask))
    corner_candidate_count = accepted_corner_count + rejected_corner_count
    corner_candidate_fraction = corner_candidate_count / skeleton_pixel_count if skeleton_pixel_count else 0.0
    warnings = []
    if two_point_fraction > 0.5:
        warnings.append(
            "More than 50% of extracted strokes have only two points. "
            "Stroke extraction may still be too fragmented."
        )
    if corner_candidate_fraction > 0.15:
        warnings.append(
            "Corner candidate count is high relative to skeleton pixels. "
            "Consider lowering corner sensitivity or keeping corners debug-only."
        )
    if segmentation_diagnostics and segmentation_mode_used == "ML":
        if segmentation_diagnostics.get("line_pixel_count", 0) == 0:
            warnings.append("ML line_mask is empty; graph extraction will produce no drawing commands.")
        elif segmentation_diagnostics.get("line_pixel_fraction", 0.0) < 0.0005:
            warnings.append("ML line_mask is extremely sparse; thresholding or class decoding may be wrong.")
        if segmentation_diagnostics.get("decode_mode") == "multiclass_argmax_empty_line_probability_fallback":
            warnings.append(
                "ML argmax predicted no line pixels; pipeline used the line probability map fallback. "
                "The checkpoint may still be undertrained or background-biased."
            )

    return {
        "schema": "stroke_pipeline_metrics_v1",
        "command_count": len(commands),
        "stroke_count": stroke_count,
        "average_points_per_stroke": float(np.mean(stroke_point_counts)) if stroke_point_counts else 0.0,
        "median_points_per_stroke": float(np.median(stroke_point_counts)) if stroke_point_counts else 0.0,
        "two_point_stroke_count": two_point_strokes,
        "two_point_stroke_fraction": two_point_fraction,
        "closed_loop_stroke_count": closed_loop_strokes,
        "raw_points_total_before_simplification": int(sum(raw_stroke_point_counts)),
        "points_total_after_simplification": int(sum(stroke_point_counts)),
        "average_raw_points_per_stroke": float(np.mean(raw_stroke_point_counts)) if raw_stroke_point_counts else 0.0,
        "pen_down_drawing_distance_mm": draw_distance,
        "pen_up_travel_distance_mm": travel_distance,
        "total_movement_distance_mm": draw_distance + travel_distance,
        "estimated_draw_movement_time_s": draw_time_s,
        "estimated_travel_movement_time_s": travel_time_s,
        "estimated_pen_mode_change_settle_time_s": pen_change_time_s,
        "estimated_total_plotting_time_s": total_time_s,
        "mode_change_count": mode_changes,
        "command_bounds_mm": bounds,
        "bounds_validation_passed": bounds_pass,
        "bounds_validation_error": bounds_error,
        "segmentation_mode_used": segmentation_mode_used,
        "model_path_used": model_path_used,
        "segmentation_notes": segmentation_notes,
        "segmentation_diagnostics": segmentation_diagnostics or {},
        "warnings": warnings,
        "graph": {
            "vertex_count": len(graph.vertices),
            "edge_count": len(graph.edges),
            "assembled_stroke_count": len(graph.strokes_px),
            "stroke_point_counts": stroke_point_counts,
            "raw_stroke_point_counts": raw_stroke_point_counts,
            "skeleton_connected_component_count": graph.skeleton_component_count,
            "skeleton_pixel_count": skeleton_pixel_count,
            "endpoint_count": int(np.count_nonzero(graph.endpoint_mask)),
            "junction_pixel_count": int(np.count_nonzero(graph.junction_mask)),
            "junction_cluster_count": len(connected_components(graph.junction_mask, connectivity=8)),
            "accepted_corner_count": accepted_corner_count,
            "rejected_corner_candidate_count": rejected_corner_count,
            "corner_candidate_count": corner_candidate_count,
            "corner_candidate_fraction_of_skeleton": corner_candidate_fraction,
        },
        "gantry_mapping": transform_info,
        "firmware_constants": firmware_constants,
        "time_estimation_assumptions": {
            "speed_units": "DRAW_SPEED and TRAVEL_SPEED are treated as AccelStepper steps/s.",
            "steps_per_mm_formula": "(FULL_STEPS_PER_REV * MICROSTEPS) / (PULLEY_TEETH * BELT_PITCH_MM)",
            "steps_per_mm": steps_per_mm,
            "draw_speed_mm_s": draw_speed_mm_s,
            "travel_speed_mm_s": travel_speed_mm_s,
            "pen_change_estimate": "Each mode change adds PEN_SETTLE_MS plus an approximate servo sweep time.",
            "acceleration": "Acceleration and cornering dynamics are not modelled.",
        },
    }


def array_to_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L").convert("RGB")


def save_segmentation_debug(result: SegmentationResult, output_path: Path) -> None:
    gray_rgb = Image.fromarray(result.gray, mode="L").convert("RGB")
    line_rgb = array_to_image(result.line_mask)
    node_rgb = array_to_image(result.node_mask)
    save_labeled_grid(
        [(gray_rgb, "grayscale"), (line_rgb, "line mask"), (node_rgb, "node/corner mask")],
        output_path,
    )


def save_skeleton_debug(graph: StrokeGraph, output_path: Path) -> None:
    skeleton = np.dstack([graph.skeleton_mask * 255] * 3).astype(np.uint8)
    skeleton[graph.endpoint_mask] = [0, 180, 0]
    skeleton[graph.junction_mask] = [220, 40, 40]
    skeleton[graph.corner_mask] = [40, 80, 220]
    save_labeled_grid(
        [(Image.fromarray(skeleton, mode="RGB"), "skeleton: endpoints green, junctions red, corners blue")],
        output_path,
    )


def save_labeled_grid(items: list[tuple[Image.Image, str]], output_path: Path) -> None:
    padding = 20
    label_h = 24
    thumb_w = max(img.width for img, _ in items)
    thumb_h = max(img.height for img, _ in items)
    canvas = Image.new("RGB", (len(items) * thumb_w + (len(items) + 1) * padding, thumb_h + label_h + 2 * padding), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (img, label) in enumerate(items):
        x = padding + i * (thumb_w + padding)
        y = padding + label_h
        canvas.paste(img.convert("RGB"), (x, y))
        draw.text((x, padding), label, fill=(20, 20, 20))
    canvas.save(output_path)


def save_graph_debug(gray: np.ndarray, graph: StrokeGraph, output_path: Path) -> None:
    image = Image.fromarray(gray, mode="L").convert("RGB")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    colors = [
        (230, 57, 70, 220),
        (29, 53, 87, 220),
        (42, 157, 143, 220),
        (244, 162, 97, 220),
        (131, 56, 236, 220),
        (0, 119, 182, 220),
    ]

    for y, x in np.argwhere(graph.rejected_corner_mask):
        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(255, 150, 0, 160))
    for y, x in np.argwhere(graph.corner_mask):
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(160, 50, 220, 220), outline=(0, 0, 0, 220))
    for y, x in np.argwhere(graph.junction_mask):
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(220, 40, 40, 190))
    for y, x in np.argwhere(graph.endpoint_mask):
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(0, 190, 70, 235), outline=(0, 0, 0, 235))

    for i, stroke in enumerate(graph.strokes_px):
        if len(stroke) < 2:
            continue
        points = [(float(x), float(y)) for x, y in stroke]
        is_fragment = len(stroke) == 2
        stroke_color = (255, 80, 0, 235) if is_fragment else colors[i % len(colors)]
        stroke_width = 4 if is_fragment else 2
        draw.line(points, fill=stroke_color, width=stroke_width)
        sx, sy = points[0]
        ex, ey = points[-1]
        draw.ellipse((sx - 4, sy - 4, sx + 4, sy + 4), fill=(0, 190, 70, 245), outline=(0, 0, 0, 245))
        draw.ellipse((ex - 3, ey - 3, ex + 3, ey + 3), fill=(40, 90, 255, 235), outline=(0, 0, 0, 235))
        draw.text((sx + 5, sy + 5), str(i), fill=(0, 100, 0, 255))
        if is_fragment:
            mx = (sx + ex) / 2.0
            my = (sy + ey) / 2.0
            draw.text((mx + 3, my + 3), "2pt", fill=(255, 80, 0, 255))
    for vertex in graph.vertices:
        x = float(vertex["x_px"])
        y = float(vertex["y_px"])
        r = 3
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 230, 0, 235), outline=(0, 0, 0, 235))
    Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB").save(output_path)


def save_stroke_path_debug(
    gray: np.ndarray,
    strokes: list[np.ndarray],
    output_path: Path,
    title: str,
) -> None:
    image = Image.fromarray(gray, mode="L").convert("RGB")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    colors = [
        (230, 57, 70, 225),
        (42, 157, 143, 225),
        (29, 53, 87, 225),
        (244, 162, 97, 225),
        (131, 56, 236, 225),
        (0, 119, 182, 225),
        (80, 150, 40, 225),
    ]
    for i, stroke in enumerate(strokes):
        if len(stroke) < 2:
            continue
        points = [(float(x), float(y)) for x, y in stroke]
        is_fragment = len(stroke) == 2
        draw.line(points, fill=(255, 80, 0, 235) if is_fragment else colors[i % len(colors)], width=4 if is_fragment else 2)
        sx, sy = points[0]
        ex, ey = points[-1]
        draw.ellipse((sx - 4, sy - 4, sx + 4, sy + 4), fill=(0, 190, 70, 245), outline=(0, 0, 0, 245))
        draw.ellipse((ex - 3, ey - 3, ex + 3, ey + 3), fill=(40, 90, 255, 235), outline=(0, 0, 0, 235))
        draw.text((sx + 5, sy + 5), str(i), fill=(0, 0, 0, 255))
    draw.rectangle((0, 0, max(260, len(title) * 7), 18), fill=(255, 255, 255, 210))
    draw.text((4, 3), title, fill=(20, 20, 20, 255))
    Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB").save(output_path)


def save_gantry_preview(
    paths_mm: list[np.ndarray],
    output_path: Path,
    work_w_mm: float,
    work_h_mm: float,
    margin_mm: float,
) -> None:
    scale = 3.0
    pad = 30
    width = int(work_w_mm * scale + 2 * pad)
    height = int(work_h_mm * scale + 2 * pad)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    def to_canvas(point: tuple[float, float]) -> tuple[float, float]:
        x, y = point
        return pad + x * scale, pad + y * scale

    draw.rectangle((*to_canvas((0, 0)), *to_canvas((work_w_mm, work_h_mm))), outline=(20, 20, 20), width=2)
    draw.rectangle(
        (*to_canvas((margin_mm, margin_mm)), *to_canvas((work_w_mm - margin_mm, work_h_mm - margin_mm))),
        outline=(120, 120, 120),
        width=1,
    )

    colors = [(220, 40, 40), (30, 90, 180), (30, 150, 90), (180, 100, 30), (120, 60, 180)]
    for i, path in enumerate(paths_mm):
        if len(path) < 2:
            continue
        points = [to_canvas((float(x), float(y))) for x, y in path]
        draw.line(points, fill=colors[i % len(colors)], width=2)
        sx, sy = points[0]
        draw.ellipse((sx - 3, sy - 3, sx + 3, sy + 3), fill=(0, 0, 0))

    draw.text((pad, 6), "Gantry path preview: top-left origin, +Y down", fill=(20, 20, 20))
    image.save(output_path)


def save_gantry_physical_preview(
    paths_mm: list[np.ndarray],
    output_path: Path,
    work_w_mm: float,
    work_h_mm: float,
    margin_mm: float,
    dpi: float = 600.0,
    pen_width_mm: float = 0.35,
) -> None:
    """Render gantry coordinates at physical scale using a real pen width."""
    px_per_mm = dpi / 25.4
    width_px = max(1, int(round(work_w_mm * px_per_mm)))
    height_px = max(1, int(round(work_h_mm * px_per_mm)))
    line_width_px = max(1, int(round(pen_width_mm * px_per_mm)))
    image = Image.new("RGB", (width_px, height_px), "white")
    draw = ImageDraw.Draw(image)

    def to_canvas(point: tuple[float, float]) -> tuple[float, float]:
        x, y = point
        return x * px_per_mm, y * px_per_mm

    margin_px = max(1, int(round(0.10 * px_per_mm)))
    draw.rectangle(
        (0, 0, width_px - 1, height_px - 1),
        outline=(230, 230, 230),
        width=max(1, margin_px),
    )
    draw.rectangle(
        (
            margin_mm * px_per_mm,
            margin_mm * px_per_mm,
            (work_w_mm - margin_mm) * px_per_mm,
            (work_h_mm - margin_mm) * px_per_mm,
        ),
        outline=(245, 245, 245),
        width=max(1, int(round(0.05 * px_per_mm))),
    )

    radius = line_width_px / 2.0
    for path in paths_mm:
        if len(path) < 2:
            continue
        points = [to_canvas((float(x), float(y))) for x, y in path]
        draw.line(points, fill=(18, 18, 18), width=line_width_px, joint="curve")
        if line_width_px > 1:
            for x, y in points:
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(18, 18, 18))

    image.save(output_path)


def save_json(data: dict, output_path: Path) -> None:
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def find_local_model(repo_root: Path) -> Path | None:
    suffixes = {".pt", ".pth", ".ckpt", ".h5", ".keras", ".pb"}
    for path in repo_root.rglob("*"):
        if ".venv" in path.parts:
            continue
        if path.is_file() and path.suffix.lower() in suffixes:
            return path
    return None


def choose_segmenter(args: argparse.Namespace, repo_root: Path) -> tuple[BaseSegmenter, list[str], str | None]:
    notes: list[str] = []
    mode = args.segmentation_mode.lower()
    model_path = Path(args.model_path).resolve() if args.model_path else find_local_model(repo_root)

    if mode in {"auto", "ml"}:
        if model_path is None:
            message = "ML segmentation attempted, but no local model/checkpoint file was found."
            if mode == "ml":
                raise FileNotFoundError(message)
            notes.append(message)
        elif not MLSegmenter.framework_available_for(model_path):
            message = (
                f"ML segmentation attempted with {model_path}, but no compatible local ML framework is installed."
            )
            if mode == "ml":
                raise RuntimeError(message)
            notes.append(message)
        else:
            notes.append(f"ML segmentation will use model: {model_path}")
            return MLSegmenter(
                model_path=model_path,
                ml_probability_threshold=args.ml_probability_threshold,
            ), notes, str(model_path)

        notes.append("Falling back to heuristic segmentation.")
        return HeuristicSegmenter(threshold=args.threshold), notes, str(model_path) if model_path else None

    if mode == "heuristic":
        notes.append("Heuristic segmentation explicitly selected.")
        return HeuristicSegmenter(threshold=args.threshold), notes, None

    raise ValueError(f"Unknown segmentation mode: {args.segmentation_mode}")


def run_pipeline(args: argparse.Namespace) -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    image_path = (repo_root / args.image).resolve() if not Path(args.image).is_absolute() else Path(args.image)
    output_dir = (repo_root / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    segmenter, setup_notes, selected_model_path = choose_segmenter(args, repo_root)
    try:
        segmentation = segmenter.segment(image_path)
        segmentation.notes = setup_notes + segmentation.notes
    except Exception as exc:
        if isinstance(segmenter, HeuristicSegmenter) or args.segmentation_mode.lower() == "ml":
            raise
        fallback = HeuristicSegmenter(threshold=args.threshold)
        segmentation = fallback.segment(image_path)
        segmentation.mode_used = "ML attempted then fallback"
        segmentation.model_path_used = selected_model_path
        segmentation.notes = setup_notes + [f"ML segmentation failed: {exc}", "Fell back to heuristic segmentation."]

    if args.segmentation_mode.lower() in {"auto", "ml"} and segmentation.mode_used == "heuristic" and setup_notes:
        segmentation.mode_used = "ML attempted then fallback"
        segmentation.model_path_used = selected_model_path

    graph = build_stroke_graph(
        segmentation.line_mask,
        segmentation.node_mask,
        split_corners=args.split_corners,
        use_node_hints_as_vertices=args.use_segmentation_node_hints,
        corner_angle_threshold_deg=args.corner_angle_threshold,
        corner_min_separation_px=args.corner_min_separation,
        corner_lookahead_px=args.corner_lookahead,
        simplification_epsilon=args.simplification_epsilon,
        junction_join_angle_deg=args.junction_join_angle,
    )
    paths_mm, transform_info = map_paths_to_gantry_mm(
        graph.strokes_px,
        image_shape=segmentation.gray.shape,
        work_w_mm=args.work_width_mm,
        work_h_mm=args.work_height_mm,
        margin_mm=args.margin_mm,
        centre_on_page=True,
    )
    commands = paths_to_arduino_commands(paths_mm)
    validate_arduino_commands(commands, args.work_width_mm, args.work_height_mm)

    firmware_path = repo_root.parent / "Arduino Code" / "ArduinoCode.ino"
    firmware_constants = parse_firmware_constants(firmware_path)
    metrics = compute_command_metrics(
        commands=commands,
        stroke_count=len(paths_mm),
        work_w_mm=args.work_width_mm,
        work_h_mm=args.work_height_mm,
        firmware_constants=firmware_constants,
        segmentation_mode_used=segmentation.mode_used,
        model_path_used=segmentation.model_path_used,
        transform_info=transform_info,
        graph=graph,
        segmentation_notes=segmentation.notes,
        segmentation_diagnostics=segmentation.diagnostics,
    )

    save_arduino_commands(commands, output_dir / "arduino_commands.txt")
    save_json(metrics, output_dir / "stroke_metrics.json")
    save_segmentation_debug(segmentation, output_dir / "segmentation_debug.png")
    save_skeleton_debug(graph, output_dir / "skeleton_debug.png")
    save_graph_debug(segmentation.gray, graph, output_dir / "graph_debug.png")
    save_stroke_path_debug(
        segmentation.gray,
        graph.raw_strokes_px,
        output_dir / "raw_traced_strokes_debug.png",
        "Raw traced strokes before simplification",
    )
    save_stroke_path_debug(
        segmentation.gray,
        graph.strokes_px,
        output_dir / "simplified_strokes_debug.png",
        "Simplified strokes after graph assembly",
    )
    save_gantry_preview(paths_mm, output_dir / "gantry_path_preview.png", args.work_width_mm, args.work_height_mm, args.margin_mm)

    print(f"Image: {image_path}")
    print(f"Output directory: {output_dir}")
    print(f"Segmentation mode used: {segmentation.mode_used}")
    if segmentation.model_path_used:
        print(f"Model path: {segmentation.model_path_used}")
    for note in segmentation.notes:
        print(f"- {note}")
    if segmentation.diagnostics:
        print(
            "Segmentation pixels: "
            f"line={segmentation.diagnostics.get('line_pixel_count', 0)}, "
            f"node={segmentation.diagnostics.get('node_pixel_count', 0)}, "
            f"decode={segmentation.diagnostics.get('decode_mode', 'unknown')}"
        )
        if segmentation.diagnostics.get("line_probability_max") is not None:
            print(
                "ML probabilities: "
                f"line_max={segmentation.diagnostics.get('line_probability_max'):.4f}, "
                f"line_mean={segmentation.diagnostics.get('line_probability_mean'):.4f}, "
                f"node_max={segmentation.diagnostics.get('node_probability_max'):.4f}"
            )
    print(f"Strokes: {metrics['stroke_count']}")
    print(f"Average points/stroke: {metrics['average_points_per_stroke']:.2f}")
    print(f"Median points/stroke: {metrics['median_points_per_stroke']:.2f}")
    print(f"2-point strokes: {metrics['two_point_stroke_count']}")
    print(f"Closed-loop strokes: {metrics['closed_loop_stroke_count']}")
    print(
        "Raw/simplified points: "
        f"{metrics['raw_points_total_before_simplification']} -> "
        f"{metrics['points_total_after_simplification']}"
    )
    print(f"Commands: {metrics['command_count']}")
    print(f"Bounds valid: {metrics['bounds_validation_passed']}")
    for warning in metrics["warnings"]:
        print(f"WARNING: {warning}")
    print(f"Estimated total plotting time: {metrics['estimated_total_plotting_time_s']:.2f} s")
    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Experimental stroke-based gantry planning pipeline.")
    parser.add_argument("--image", default=str(repo_root / "input_images" / "square.png"), help="Input raster line-art image.")
    parser.add_argument("--output-dir", default=str(repo_root / "output" / "stroke_based"), help="Directory for stroke outputs.")
    parser.add_argument("--model-path", default=None, help="Optional local ML segmentation model/checkpoint path.")
    parser.add_argument("--segmentation-mode", choices=["ml", "heuristic", "auto"], default="auto")
    parser.add_argument("--threshold", type=int, default=127, help="Heuristic dark-line threshold, 0..255.")
    parser.add_argument(
        "--ml-probability-threshold",
        type=float,
        default=0.35,
        help="Fallback ML foreground probability threshold used only when multiclass argmax predicts no line pixels.",
    )
    parser.add_argument("--work-width-mm", type=float, default=150.0)
    parser.add_argument("--work-height-mm", type=float, default=270.0)
    parser.add_argument("--margin-mm", type=float, default=5.0)
    parser.add_argument("--simplification-epsilon", type=float, default=0.75)
    parser.add_argument("--corner-angle-threshold", type=float, default=100.0)
    parser.add_argument("--corner-min-separation", type=int, default=12)
    parser.add_argument("--corner-lookahead", type=int, default=8)
    parser.add_argument("--split-corners", action="store_true", help="Use accepted corners as graph-splitting vertices.")
    parser.add_argument("--use-segmentation-node-hints", action="store_true", help="Use segmenter node masks as graph vertices.")
    parser.add_argument("--junction-join-angle", type=float, default=35.0, help="Maximum smooth continuation angle through junctions.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
