"""Route-first decoder for rich stroke-structure masks.

This decoder treats the predicted masks as a drawable line network. It builds a
one-pixel skeleton from the support probability, solves each connected skeleton
component as an Euler-style route, and duplicates the shortest supported graph
paths needed to avoid unnecessary pen lifts.

The output format matches the rest of the gantry project:

    x_mm,y_mm,mode

where mode 0 is pen-up travel and mode 1 is pen-down drawing.
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
    close_mask,
    map_paths_to_gantry_mm,
    parse_firmware_constants,
    paths_to_arduino_commands,
    remove_small_components,
    save_arduino_commands,
    save_gantry_preview,
    save_json,
    simplify_polyline,
    validate_arduino_commands,
    zhang_suen_thinning,
)


Pixel = tuple[int, int]
Command = tuple[float, float, int]


@dataclass
class RouteLabels:
    gray: np.ndarray
    support_prob: np.ndarray
    centreline_prob: np.ndarray
    endpoint_prob: np.ndarray
    corner_prob: np.ndarray
    junction_prob: np.ndarray
    tangent_cos: np.ndarray
    tangent_sin: np.ndarray
    support_mask: np.ndarray
    centreline_mask: np.ndarray
    endpoint_mask: np.ndarray
    corner_mask: np.ndarray
    junction_mask: np.ndarray
    tangent_valid_mask: np.ndarray
    model_path: str | None
    label_schema: str | None


@dataclass
class RouteComponent:
    id: int
    mask: np.ndarray
    original_edge_count: int
    duplicated_edge_count: int
    node_count: int
    topology_node_count: int
    odd_before: int
    odd_after: int
    start: Pixel
    end: Pixel
    closed: bool
    raw_route_px: np.ndarray
    simplified_route_px: np.ndarray


@dataclass
class RouteResult:
    strokes_px: list[np.ndarray]
    raw_strokes_px: list[np.ndarray]
    skeleton_mask: np.ndarray
    drawable_mask: np.ndarray
    claimed_support_mask: np.ndarray
    drawn_mask_for_false_positive: np.ndarray
    missed_support_mask: np.ndarray
    false_positive_mask: np.ndarray
    components: list[RouteComponent]
    endpoint_count: int
    corner_count: int
    junction_count: int
    traced_endpoint_count: int
    tangent_consistency_score: float


def load_grayscale(image_path: Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(image_path).convert("L")
    if shape is not None and image.size != (shape[1], shape[0]):
        image = image.resize((shape[1], shape[0]), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.uint8)


def probability_to_mask(prob: np.ndarray, threshold: float) -> np.ndarray:
    return np.asarray(prob, dtype=np.float32) >= threshold


def load_labels_from_arrays(args: argparse.Namespace) -> RouteLabels:
    arrays_path = Path(args.prediction_arrays)
    arrays = np.load(arrays_path)
    support_prob = arrays["support_probability"].astype(np.float32)
    centreline_prob = arrays["centreline_probability"].astype(np.float32)
    endpoint_prob = arrays["endpoint_probability"].astype(np.float32)
    corner_prob = arrays["corner_probability"].astype(np.float32)
    junction_prob = arrays["junction_probability"].astype(np.float32)
    tangent_cos = arrays["tangent_cos"].astype(np.float32)
    tangent_sin = arrays["tangent_sin"].astype(np.float32)
    shape = support_prob.shape
    gray = load_grayscale(Path(args.image), shape=shape)
    support_mask = probability_to_mask(support_prob, args.support_threshold)
    centreline_mask = probability_to_mask(centreline_prob, args.centreline_threshold)
    endpoint_mask = probability_to_mask(endpoint_prob, args.endpoint_threshold)
    corner_mask = probability_to_mask(corner_prob, args.corner_threshold)
    junction_mask = probability_to_mask(junction_prob, args.junction_threshold)
    if "tangent_valid_mask" in arrays:
        tangent_valid = arrays["tangent_valid_mask"].astype(bool)
    else:
        tangent_valid = (support_prob >= args.tangent_valid_threshold) | centreline_mask
    return RouteLabels(
        gray=gray,
        support_prob=support_prob,
        centreline_prob=centreline_prob,
        endpoint_prob=endpoint_prob,
        corner_prob=corner_prob,
        junction_prob=junction_prob,
        tangent_cos=tangent_cos,
        tangent_sin=tangent_sin,
        support_mask=support_mask,
        centreline_mask=centreline_mask,
        endpoint_mask=endpoint_mask,
        corner_mask=corner_mask,
        junction_mask=junction_mask,
        tangent_valid_mask=tangent_valid,
        model_path=None,
        label_schema="saved_structure_prediction_arrays",
    )


def load_labels_from_model(args: argparse.Namespace) -> RouteLabels:
    from stroke_structure_reconstruction_pipeline import infer_structure_labels

    structure_labels, probabilities, checkpoint = infer_structure_labels(Path(args.image), Path(args.model_path), args)
    return RouteLabels(
        gray=structure_labels.gray,
        support_prob=probabilities["support"].astype(np.float32),
        centreline_prob=probabilities["centreline"].astype(np.float32),
        endpoint_prob=probabilities["endpoint"].astype(np.float32),
        corner_prob=probabilities["corner"].astype(np.float32),
        junction_prob=probabilities["junction"].astype(np.float32),
        tangent_cos=probabilities["tangent_cos"].astype(np.float32),
        tangent_sin=probabilities["tangent_sin"].astype(np.float32),
        support_mask=structure_labels.support.astype(bool),
        centreline_mask=structure_labels.centreline.astype(bool),
        endpoint_mask=structure_labels.endpoint >= args.endpoint_threshold,
        corner_mask=structure_labels.corner >= args.corner_threshold,
        junction_mask=structure_labels.junction >= args.junction_threshold,
        tangent_valid_mask=structure_labels.tangent_valid.astype(bool),
        model_path=str(args.model_path),
        label_schema=checkpoint.get("label_schema") if isinstance(checkpoint, dict) else None,
    )


def load_labels(args: argparse.Namespace) -> RouteLabels:
    if args.prediction_arrays:
        return load_labels_from_arrays(args)
    if args.model_path:
        return load_labels_from_model(args)
    raise ValueError("Provide either --prediction-arrays or --model-path.")


def all_neighbors(pixel: Pixel, mask: np.ndarray) -> list[Pixel]:
    y, x = pixel
    height, width = mask.shape
    result: list[Pixel] = []
    for dy, dx in [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]:
        ny, nx = y + dy, x + dx
        if 0 <= ny < height and 0 <= nx < width and mask[ny, nx]:
            result.append((ny, nx))
    return result


def topology_neighbors(pixel: Pixel, mask: np.ndarray) -> list[Pixel]:
    """Skeleton neighbors, avoiding redundant diagonal links in 2x2 blocks."""
    y, x = pixel
    height, width = mask.shape
    result: list[Pixel] = []
    for dy, dx in [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]:
        ny, nx = y + dy, x + dx
        if not (0 <= ny < height and 0 <= nx < width and mask[ny, nx]):
            continue
        if dy != 0 and dx != 0 and (mask[y, nx] or mask[ny, x]):
            continue
        result.append((ny, nx))
    return result


def connected_components(mask: np.ndarray, topology: bool = False) -> list[list[Pixel]]:
    remaining = set(map(tuple, np.argwhere(mask.astype(bool))))
    components: list[list[Pixel]] = []
    neighbor_fn = topology_neighbors if topology else all_neighbors
    while remaining:
        start = remaining.pop()
        stack = [start]
        component = [start]
        while stack:
            pixel = stack.pop()
            for nbr in neighbor_fn(pixel, mask):
                if nbr in remaining:
                    remaining.remove(nbr)
                    stack.append(nbr)
                    component.append(nbr)
        components.append(component)
    return components


def expanded_mask(mask: np.ndarray, radius_px: int) -> np.ndarray:
    if radius_px <= 0:
        return mask.astype(bool).copy()
    height, width = mask.shape
    output = np.zeros_like(mask, dtype=bool)
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys.tolist(), xs.tolist()):
        y0 = max(0, y - radius_px)
        y1 = min(height, y + radius_px + 1)
        x0 = max(0, x - radius_px)
        x1 = min(width, x + radius_px + 1)
        output[y0:y1, x0:x1] = True
    return output


def mask_from_points(points: list[Pixel], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for y, x in points:
        mask[y, x] = True
    return mask


def topology_degree(mask: np.ndarray) -> dict[Pixel, int]:
    return {pixel: len(topology_neighbors(pixel, mask)) for pixel in map(tuple, np.argwhere(mask))}


def heatmap_count(mask: np.ndarray) -> int:
    return len(connected_components(mask.astype(bool), topology=False))


def heatmap_centres(mask: np.ndarray, probability: np.ndarray) -> list[tuple[float, float, float]]:
    centres: list[tuple[float, float, float]] = []
    for component in connected_components(mask.astype(bool), topology=False):
        ys = np.array([p[0] for p in component], dtype=np.float32)
        xs = np.array([p[1] for p in component], dtype=np.float32)
        values = probability[ys.astype(np.int32), xs.astype(np.int32)]
        weights = np.maximum(values, 1e-3)
        centres.append((float(np.average(ys, weights=weights)), float(np.average(xs, weights=weights)), float(np.max(values))))
    return centres


def nearest_pixel(points: list[Pixel], yx: tuple[float, float], max_radius: float | None = None) -> Pixel | None:
    if not points:
        return None
    target_y, target_x = yx
    best: tuple[float, Pixel] | None = None
    for pixel in points:
        y, x = pixel
        d2 = (float(y) - target_y) ** 2 + (float(x) - target_x) ** 2
        if best is None or d2 < best[0]:
            best = (d2, pixel)
    if best is None:
        return None
    distance = math.sqrt(best[0])
    if max_radius is not None and distance > max_radius:
        return None
    return best[1]


def build_drawable_skeleton(labels: RouteLabels, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    strong_support = labels.support_prob >= args.support_threshold
    bridge_support = labels.support_prob >= args.support_bridge_threshold
    centre_seed = labels.centreline_prob >= args.centreline_seed_threshold
    drawable = strong_support | (bridge_support & expanded_mask(centre_seed | strong_support, args.bridge_seed_radius_px))
    if args.close_iterations > 0:
        drawable = close_mask(drawable, iterations=args.close_iterations)
    drawable = remove_small_components(drawable, args.min_component_pixels)
    if not np.any(drawable) and np.any(centre_seed):
        drawable = centre_seed.copy()
    skeleton = zhang_suen_thinning(drawable)

    centre_skeleton = zhang_suen_thinning(centre_seed & expanded_mask(drawable, 1))
    if np.any(centre_skeleton):
        skeleton |= centre_skeleton & expanded_mask(skeleton, args.centreline_repair_radius_px)
        skeleton = zhang_suen_thinning(skeleton)

    protected = expanded_mask(labels.endpoint_mask | labels.corner_mask | labels.junction_mask, args.protected_radius_px)
    skeleton = prune_short_spurs(skeleton, protected & skeleton, args.spur_prune_length_px)
    skeleton = bridge_skeleton_components(skeleton, drawable, labels, args)
    skeleton = remove_small_components(skeleton, args.min_skeleton_component_pixels)
    return drawable.astype(bool), skeleton.astype(bool)


def bridge_skeleton_components(
    skeleton: np.ndarray,
    drawable: np.ndarray,
    labels: RouteLabels,
    args: argparse.Namespace,
) -> np.ndarray:
    """Reconnect skeleton fragments when the underlying support is connected."""
    bridged = skeleton.astype(bool).copy()
    if args.max_skeleton_bridge_px <= 0:
        return bridged
    drawable_components = connected_components(drawable, topology=False)
    for support_points in drawable_components:
        support_mask = mask_from_points(support_points, drawable.shape)
        local_components = sorted(
            connected_components(bridged & support_mask, topology=True),
            key=len,
            reverse=True,
        )
        if len(local_components) <= 1:
            continue
        main_points = set(local_components[0])
        for component in local_components[1:]:
            path = shortest_path_through_mask(
                support_mask,
                component,
                main_points,
                labels,
                max_length=args.max_skeleton_bridge_px,
            )
            if not path:
                continue
            for y, x in path:
                bridged[y, x] = True
            main_points.update(component)
            main_points.update(path)
    return zhang_suen_thinning(bridged)


def shortest_path_through_mask(
    mask: np.ndarray,
    starts: list[Pixel],
    goals: set[Pixel],
    labels: RouteLabels,
    max_length: int,
) -> list[Pixel]:
    if not starts or not goals:
        return []
    queue: list[tuple[float, int, Pixel]] = []
    parent: dict[Pixel, Pixel | None] = {}
    best: dict[Pixel, float] = {}
    order = 0
    for start in starts:
        parent[start] = None
        best[start] = 0.0
        heapq.heappush(queue, (0.0, order, start))
        order += 1
    found: Pixel | None = None
    while queue:
        cost, _, current = heapq.heappop(queue)
        if cost > best.get(current, math.inf) or cost > max_length:
            continue
        if current in goals:
            found = current
            break
        for nbr in all_neighbors(current, mask):
            step = edge_travel_cost(current, nbr, labels)
            new_cost = cost + step
            if new_cost < best.get(nbr, math.inf) and new_cost <= max_length:
                best[nbr] = new_cost
                parent[nbr] = current
                heapq.heappush(queue, (new_cost, order, nbr))
                order += 1
    if found is None:
        return []
    path: list[Pixel] = []
    current: Pixel | None = found
    while current is not None:
        path.append(current)
        current = parent[current]
    path.reverse()
    return path


def prune_short_spurs(mask: np.ndarray, protected: np.ndarray, max_length: int) -> np.ndarray:
    if max_length <= 0:
        return mask.astype(bool).copy()
    pruned = mask.astype(bool).copy()
    changed = True
    while changed:
        changed = False
        degree = topology_degree(pruned)
        endpoints = [pixel for pixel, count in degree.items() if count <= 1 and not protected[pixel]]
        for start in endpoints:
            path = [start]
            previous: Pixel | None = None
            current = start
            while len(path) <= max_length + 1:
                candidates = [p for p in topology_neighbors(current, pruned) if p != previous]
                if len(candidates) != 1:
                    break
                previous, current = current, candidates[0]
                path.append(current)
                if protected[current] or degree.get(current, 0) != 2:
                    break
            if len(path) <= max_length and not any(protected[p] for p in path):
                for y, x in path:
                    pruned[y, x] = False
                changed = True
    return pruned


def component_adjacency(mask: np.ndarray) -> dict[Pixel, list[Pixel]]:
    return {pixel: topology_neighbors(pixel, mask) for pixel in map(tuple, np.argwhere(mask))}


def undirected_edges(adjacency: dict[Pixel, list[Pixel]]) -> list[tuple[Pixel, Pixel]]:
    edges: list[tuple[Pixel, Pixel]] = []
    seen: set[tuple[Pixel, Pixel]] = set()
    for u, neighbors in adjacency.items():
        for v in neighbors:
            key = tuple(sorted([u, v]))
            if key not in seen:
                seen.add(key)
                edges.append((u, v))
    return edges


def pixel_distance(a: Pixel, b: Pixel) -> float:
    return float(math.hypot(a[1] - b[1], a[0] - b[0]))


def choose_component_terminals(
    points: list[Pixel],
    odd_nodes: set[Pixel],
    endpoint_centres: list[tuple[float, float, float]],
    args: argparse.Namespace,
) -> tuple[Pixel, Pixel, bool]:
    if not points:
        raise ValueError("Cannot choose route terminal for empty component.")
    if not odd_nodes:
        start = min(points)
        return start, start, True

    endpoint_snaps: list[Pixel] = []
    for y, x, _ in endpoint_centres:
        snapped = nearest_pixel(points, (y, x), max_radius=args.endpoint_snap_radius_px)
        if snapped is not None and snapped not in endpoint_snaps:
            endpoint_snaps.append(snapped)

    candidates = endpoint_snaps if len(endpoint_snaps) >= 2 else list(odd_nodes)
    best_pair: tuple[float, Pixel, Pixel] | None = None
    for i, first in enumerate(candidates):
        for second in candidates[i + 1 :]:
            score = pixel_distance(first, second)
            if best_pair is None or score > best_pair[0]:
                best_pair = (score, first, second)
    if best_pair is None:
        start = candidates[0] if candidates else min(odd_nodes)
        return start, start, True
    return best_pair[1], best_pair[2], False


def shortest_path(
    adjacency: dict[Pixel, list[Pixel]],
    start: Pixel,
    goals: set[Pixel],
    labels: RouteLabels,
) -> list[Pixel]:
    if start in goals:
        return [start]
    queue: list[tuple[float, int, Pixel]] = [(0.0, 0, start)]
    parent: dict[Pixel, Pixel | None] = {start: None}
    best_cost: dict[Pixel, float] = {start: 0.0}
    order = 1
    found: Pixel | None = None
    while queue:
        cost, _, current = heapq.heappop(queue)
        if cost > best_cost.get(current, math.inf):
            continue
        if current in goals:
            found = current
            break
        for nbr in adjacency.get(current, []):
            step_cost = edge_travel_cost(current, nbr, labels)
            new_cost = cost + step_cost
            if new_cost < best_cost.get(nbr, math.inf):
                best_cost[nbr] = new_cost
                parent[nbr] = current
                heapq.heappush(queue, (new_cost, order, nbr))
                order += 1
    if found is None:
        return []
    path: list[Pixel] = []
    current: Pixel | None = found
    while current is not None:
        path.append(current)
        current = parent[current]
    path.reverse()
    return path


def edge_travel_cost(start: Pixel, end: Pixel, labels: RouteLabels) -> float:
    sy, sx = start
    ey, ex = end
    base = pixel_distance(start, end)
    support = 0.5 * (float(labels.support_prob[sy, sx]) + float(labels.support_prob[ey, ex]))
    centre = 0.5 * (float(labels.centreline_prob[sy, sx]) + float(labels.centreline_prob[ey, ex]))
    return base * (1.25 - 0.20 * support - 0.05 * centre)


def pair_parity_targets(
    adjacency: dict[Pixel, list[Pixel]],
    odd_nodes: set[Pixel],
    start: Pixel,
    end: Pixel,
    closed: bool,
    labels: RouteLabels,
) -> list[list[Pixel]]:
    targets = set(odd_nodes)
    if not closed:
        for terminal in (start, end):
            if terminal in targets:
                targets.remove(terminal)
            else:
                targets.add(terminal)
    paths: list[list[Pixel]] = []
    while len(targets) >= 2:
        source = min(targets)
        targets.remove(source)
        path = shortest_path(adjacency, source, targets, labels)
        if not path:
            break
        target = path[-1]
        if target in targets:
            targets.remove(target)
        paths.append(path)
    return paths


def tangent_step_score(labels: RouteLabels, previous: Pixel | None, current: Pixel, candidate: Pixel) -> float:
    cy, cx = current
    direction = np.array([candidate[1] - cx, candidate[0] - cy], dtype=np.float32)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-6:
        return -10.0
    direction /= norm
    tangent_score = 0.0
    if labels.tangent_valid_mask[cy, cx]:
        tangent = np.array([labels.tangent_cos[cy, cx], labels.tangent_sin[cy, cx]], dtype=np.float32)
        tnorm = float(np.linalg.norm(tangent))
        if tnorm > 1e-6:
            tangent_score = abs(float(np.dot(direction, tangent / tnorm)))
    smooth_score = 0.0
    if previous is not None:
        py, px = previous
        incoming = np.array([cx - px, cy - py], dtype=np.float32)
        in_norm = float(np.linalg.norm(incoming))
        if in_norm > 1e-6:
            smooth_score = float(np.dot(incoming / in_norm, direction))
    support_score = float(labels.support_prob[cy, cx])
    centre_score = float(labels.centreline_prob[cy, cx])
    return 0.45 * smooth_score + 0.35 * tangent_score + 0.15 * support_score + 0.05 * centre_score


def build_multigraph_edges(
    adjacency: dict[Pixel, list[Pixel]],
    duplicate_paths: list[list[Pixel]],
) -> tuple[list[tuple[Pixel, Pixel, bool]], dict[Pixel, list[int]]]:
    edges = [(u, v, False) for u, v in undirected_edges(adjacency)]
    for path in duplicate_paths:
        for u, v in zip(path[:-1], path[1:]):
            edges.append((u, v, True))
    multi_adj: dict[Pixel, list[int]] = {pixel: [] for pixel in adjacency}
    for edge_id, (u, v, _) in enumerate(edges):
        multi_adj.setdefault(u, []).append(edge_id)
        multi_adj.setdefault(v, []).append(edge_id)
    return edges, multi_adj


def odd_count_from_multigraph(edges: list[tuple[Pixel, Pixel, bool]]) -> int:
    degree: dict[Pixel, int] = {}
    for u, v, _ in edges:
        degree[u] = degree.get(u, 0) + 1
        degree[v] = degree.get(v, 0) + 1
    return sum(1 for count in degree.values() if count % 2 == 1)


def euler_route(
    edges: list[tuple[Pixel, Pixel, bool]],
    multi_adj: dict[Pixel, list[int]],
    start: Pixel,
    labels: RouteLabels,
) -> list[Pixel]:
    used = [False] * len(edges)
    stack = [start]
    route: list[Pixel] = []
    while stack:
        current = stack[-1]
        previous = stack[-2] if len(stack) >= 2 else None
        available = [edge_id for edge_id in multi_adj.get(current, []) if not used[edge_id]]
        if not available:
            route.append(stack.pop())
            continue
        scored: list[tuple[float, int, Pixel]] = []
        for edge_id in available:
            u, v, _ = edges[edge_id]
            nxt = v if u == current else u
            scored.append((tangent_step_score(labels, previous, current, nxt), -edge_id, nxt))
        scored.sort(reverse=True)
        _, neg_edge_id, nxt = scored[0]
        used[-neg_edge_id] = True
        stack.append(nxt)
    route.reverse()
    return route


def component_route(
    component_id: int,
    points: list[Pixel],
    skeleton_shape: tuple[int, int],
    endpoint_centres: list[tuple[float, float, float]],
    pin_mask: np.ndarray,
    labels: RouteLabels,
    args: argparse.Namespace,
) -> RouteComponent | None:
    if len(points) < args.min_skeleton_component_pixels:
        return None
    mask = mask_from_points(points, skeleton_shape)
    adjacency = component_adjacency(mask)
    if not adjacency:
        return None
    degree = {pixel: len(neighbors) for pixel, neighbors in adjacency.items()}
    odd_nodes = {pixel for pixel, count in degree.items() if count % 2 == 1}
    start, end, closed = choose_component_terminals(points, odd_nodes, endpoint_centres, args)
    duplicate_paths = pair_parity_targets(adjacency, odd_nodes, start, end, closed, labels)
    edges, multi_adj = build_multigraph_edges(adjacency, duplicate_paths)
    if not closed:
        route_start = start
    else:
        route_start = start if start in adjacency else min(adjacency)
    route = euler_route(edges, multi_adj, route_start, labels)
    if len(route) < 2:
        return None
    raw = np.array([(x, y) for y, x in route], dtype=np.float32)
    simplified = simplify_route_with_pins(raw, route, pin_mask, labels, args)
    topology_node_count = sum(1 for count in degree.values() if count != 2)
    return RouteComponent(
        id=component_id,
        mask=mask,
        original_edge_count=len(undirected_edges(adjacency)),
        duplicated_edge_count=sum(max(0, len(path) - 1) for path in duplicate_paths),
        node_count=len(adjacency),
        topology_node_count=topology_node_count,
        odd_before=len(odd_nodes),
        odd_after=odd_count_from_multigraph(edges),
        start=start,
        end=end,
        closed=closed,
        raw_route_px=raw,
        simplified_route_px=simplified,
    )


def simplify_route_with_pins(
    raw_xy: np.ndarray,
    route_pixels: list[Pixel],
    pin_mask: np.ndarray,
    labels: RouteLabels,
    args: argparse.Namespace,
) -> np.ndarray:
    if len(raw_xy) <= 2:
        return raw_xy
    pin_indexes = {0, len(raw_xy) - 1}
    expanded_pins = expanded_mask(pin_mask, args.pin_snap_radius_px)
    for index, pixel in enumerate(route_pixels):
        if expanded_pins[pixel]:
            pin_indexes.add(index)
    if args.simplify_max_segment_points > 0:
        for index in range(0, len(raw_xy), args.simplify_max_segment_points):
            pin_indexes.add(index)
    ordered = sorted(pin_indexes)
    pieces: list[np.ndarray] = []
    for start, end in zip(ordered[:-1], ordered[1:]):
        if end <= start:
            continue
        segment = raw_xy[start : end + 1]
        simplified = simplify_polyline(segment.astype(np.float32), args.simplification_epsilon)
        if segment_has_unsupported_jump(simplified, labels, args):
            simplified = segment.astype(np.float32)
        if pieces:
            simplified = simplified[1:] if len(simplified) > 1 else simplified
        if len(simplified):
            pieces.append(simplified)
    if not pieces:
        return raw_xy.astype(np.float32)
    return collapse_duplicates(np.vstack(pieces).astype(np.float32))


def collapse_duplicates(points: np.ndarray) -> np.ndarray:
    if len(points) <= 1:
        return points
    kept = [points[0]]
    for point in points[1:]:
        if float(np.linalg.norm(point - kept[-1])) > 1e-6:
            kept.append(point)
    return np.asarray(kept, dtype=np.float32)


def segment_pixels_xy(start_xy: np.ndarray, end_xy: np.ndarray, shape: tuple[int, int]) -> list[Pixel]:
    distance = float(np.linalg.norm(end_xy - start_xy))
    steps = max(2, int(math.ceil(distance)) + 1)
    pixels: list[Pixel] = []
    seen: set[Pixel] = set()
    for t in np.linspace(0.0, 1.0, steps):
        x, y = start_xy * (1.0 - t) + end_xy * t
        pixel = (int(round(float(y))), int(round(float(x))))
        if 0 <= pixel[0] < shape[0] and 0 <= pixel[1] < shape[1] and pixel not in seen:
            seen.add(pixel)
            pixels.append(pixel)
    return pixels


def support_fraction_between(labels: RouteLabels, start_xy: np.ndarray, end_xy: np.ndarray) -> float:
    pixels = segment_pixels_xy(start_xy, end_xy, labels.support_mask.shape)
    if not pixels:
        return 0.0
    supported = 0
    for y, x in pixels:
        if labels.support_mask[y, x] or labels.tangent_valid_mask[y, x] or labels.centreline_mask[y, x]:
            supported += 1
    return supported / len(pixels)


def segment_has_unsupported_jump(points: np.ndarray, labels: RouteLabels, args: argparse.Namespace) -> bool:
    if len(points) < 2:
        return False
    for start, end in zip(points[:-1], points[1:]):
        distance = float(np.linalg.norm(end - start))
        if distance <= args.max_draw_jump_px:
            continue
        if support_fraction_between(labels, start, end) < args.jump_support_fraction:
            return True
    return False


def claimed_pixels_from_strokes(strokes: list[np.ndarray], shape: tuple[int, int], radius: int) -> np.ndarray:
    image = Image.new("L", (shape[1], shape[0]), 0)
    draw = ImageDraw.Draw(image)
    width = max(1, radius * 2 + 1)
    for stroke in strokes:
        if len(stroke) == 1:
            x, y = stroke[0]
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=1)
        elif len(stroke) > 1:
            draw.line([tuple(point) for point in stroke], fill=1, width=width)
    return np.asarray(image, dtype=np.uint8).astype(bool)


def tangent_consistency(strokes: list[np.ndarray], labels: RouteLabels) -> float:
    scores: list[float] = []
    height, width = labels.support_mask.shape
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
            if not (0 <= my < height and 0 <= mx < width and labels.tangent_valid_mask[my, mx]):
                continue
            tangent = np.array([labels.tangent_cos[my, mx], labels.tangent_sin[my, mx]], dtype=np.float32)
            tnorm = float(np.linalg.norm(tangent))
            if tnorm > 1e-6:
                scores.append(abs(float(np.dot(direction, tangent / tnorm))))
    return float(np.mean(scores)) if scores else 0.0


def endpoint_usage(strokes: list[np.ndarray], endpoint_centres: list[tuple[float, float, float]], radius_px: float) -> int:
    used = 0
    endpoints: list[np.ndarray] = []
    for stroke in strokes:
        if len(stroke):
            endpoints.append(stroke[0])
            endpoints.append(stroke[-1])
    for y, x, _ in endpoint_centres:
        centre = np.array([x, y], dtype=np.float32)
        if any(float(np.linalg.norm(point - centre)) <= radius_px for point in endpoints):
            used += 1
    return used


def decode_routes(labels: RouteLabels, args: argparse.Namespace) -> RouteResult:
    drawable, skeleton = build_drawable_skeleton(labels, args)
    endpoint_centres = heatmap_centres(labels.endpoint_mask, labels.endpoint_prob)
    pin_mask = build_route_pin_mask(skeleton, labels, args)
    components: list[RouteComponent] = []
    raw_strokes: list[np.ndarray] = []
    strokes: list[np.ndarray] = []
    for component_id, points in enumerate(sorted(connected_components(skeleton, topology=True), key=len, reverse=True)):
        component = component_route(component_id, points, skeleton.shape, endpoint_centres, pin_mask, labels, args)
        if component is None:
            continue
        components.append(component)
        raw_strokes.append(component.raw_route_px)
        strokes.append(component.simplified_route_px)

    claimed = claimed_pixels_from_strokes(strokes, labels.support_mask.shape, radius=args.coverage_radius_px)
    drawn_for_false_positive = claimed_pixels_from_strokes(
        strokes,
        labels.support_mask.shape,
        radius=args.false_positive_radius_px,
    )
    missed = labels.support_mask & ~claimed
    false_positive = drawn_for_false_positive & ~labels.support_mask
    endpoint_count = len(endpoint_centres)
    traced_endpoint_count = endpoint_usage(strokes, endpoint_centres, args.endpoint_usage_radius_px)
    return RouteResult(
        strokes_px=strokes,
        raw_strokes_px=raw_strokes,
        skeleton_mask=skeleton,
        drawable_mask=drawable,
        claimed_support_mask=claimed & labels.support_mask,
        drawn_mask_for_false_positive=drawn_for_false_positive,
        missed_support_mask=missed,
        false_positive_mask=false_positive,
        components=components,
        endpoint_count=endpoint_count,
        corner_count=heatmap_count(labels.corner_mask),
        junction_count=heatmap_count(labels.junction_mask),
        traced_endpoint_count=traced_endpoint_count,
        tangent_consistency_score=tangent_consistency(strokes, labels),
    )


def build_route_pin_mask(skeleton: np.ndarray, labels: RouteLabels, args: argparse.Namespace) -> np.ndarray:
    """Create compact simplification pins from heatmap centres and topology."""
    pin_mask = np.zeros_like(skeleton, dtype=bool)
    skeleton_points = list(map(tuple, np.argwhere(skeleton)))
    degree = topology_degree(skeleton)
    if args.pin_topology_radius_px >= 0:
        for pixel, count in degree.items():
            if count == 2:
                continue
            y, x = pixel
            y0 = max(0, y - args.pin_topology_radius_px)
            y1 = min(pin_mask.shape[0], y + args.pin_topology_radius_px + 1)
            x0 = max(0, x - args.pin_topology_radius_px)
            x1 = min(pin_mask.shape[1], x + args.pin_topology_radius_px + 1)
            pin_mask[y0:y1, x0:x1] = True
    for mask, prob in [
        (labels.endpoint_mask, labels.endpoint_prob),
        (labels.corner_mask, labels.corner_prob),
        (labels.junction_mask, labels.junction_prob),
    ]:
        for y, x, _ in heatmap_centres(mask, prob):
            snapped = nearest_pixel(skeleton_points, (y, x), max_radius=args.pin_heatmap_snap_radius_px)
            if snapped is None:
                continue
            sy, sx = snapped
            y0 = max(0, sy - args.pin_heatmap_radius_px)
            y1 = min(pin_mask.shape[0], sy + args.pin_heatmap_radius_px + 1)
            x0 = max(0, sx - args.pin_heatmap_radius_px)
            x1 = min(pin_mask.shape[1], sx + args.pin_heatmap_radius_px + 1)
            pin_mask[y0:y1, x0:x1] = True
    return pin_mask


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
        if previous_mode is None or previous_mode != mode:
            mode_changes += 1
        previous = command
        previous_mode = mode
    return draw_distance, travel_distance, mode_changes


def compute_metrics(
    commands: list[Command],
    result: RouteResult,
    labels: RouteLabels,
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

    support_pixels = int(np.count_nonzero(labels.support_mask))
    claimed_support_pixels = int(np.count_nonzero(result.claimed_support_mask))
    false_positive_pixels = int(np.count_nonzero(result.false_positive_mask))
    drawn_pixels = int(np.count_nonzero(result.drawn_mask_for_false_positive))
    counts = [len(stroke) for stroke in result.strokes_px]
    raw_counts = [len(stroke) for stroke in result.raw_strokes_px]
    steps_per_mm = (
        firmware_constants["FULL_STEPS_PER_REV"]
        * firmware_constants["MICROSTEPS"]
        / (firmware_constants["PULLEY_TEETH"] * firmware_constants["BELT_PITCH_MM"])
    )
    draw_speed_mm_s = firmware_constants["DRAW_SPEED"] / steps_per_mm
    travel_speed_mm_s = firmware_constants["TRAVEL_SPEED"] / steps_per_mm
    servo_sweep_ms = abs(firmware_constants["PEN_UP_ANGLE"] - firmware_constants["PEN_DOWN_ANGLE"]) * firmware_constants["SERVO_DELAY_MS"]
    pen_change_time_s = mode_changes * (firmware_constants["PEN_SETTLE_MS"] + servo_sweep_ms) / 1000.0

    return {
        "schema": "stroke_route_decoder_metrics_v1",
        "pipeline": "ml_stroke_route_decoder",
        "command_count": len(commands),
        "stroke_count": len(result.strokes_px),
        "average_points_per_stroke": float(np.mean(counts)) if counts else 0.0,
        "median_points_per_stroke": float(np.median(counts)) if counts else 0.0,
        "raw_points_total": int(sum(raw_counts)),
        "simplified_points_total": int(sum(counts)),
        "pen_up_travel_distance_mm": travel_distance,
        "pen_down_drawing_distance_mm": draw_distance,
        "total_movement_distance_mm": travel_distance + draw_distance,
        "estimated_draw_movement_time_s": draw_distance / draw_speed_mm_s if draw_speed_mm_s > 0 else 0.0,
        "estimated_travel_movement_time_s": travel_distance / travel_speed_mm_s if travel_speed_mm_s > 0 else 0.0,
        "estimated_pen_mode_change_settle_time_s": pen_change_time_s,
        "estimated_plotting_time_s": (
            (draw_distance / draw_speed_mm_s if draw_speed_mm_s > 0 else 0.0)
            + (travel_distance / travel_speed_mm_s if travel_speed_mm_s > 0 else 0.0)
            + pen_change_time_s
        ),
        "mode_change_count": mode_changes,
        "bounds_validation_passed": bounds_valid,
        "bounds_validation_error": bounds_error,
        "command_bounds_mm": bounds,
        "support_coverage": claimed_support_pixels / max(support_pixels, 1),
        "line_support_coverage": claimed_support_pixels / max(support_pixels, 1),
        "false_positive_drawn_fraction": false_positive_pixels / max(drawn_pixels, 1),
        "connected_mask_component_count": len(connected_components(labels.support_mask, topology=False)),
        "skeleton_component_count": len(connected_components(result.skeleton_mask, topology=True)),
        "graph_node_count": int(sum(component.node_count for component in result.components)),
        "graph_edge_count": int(sum(component.original_edge_count for component in result.components)),
        "topology_node_count": int(sum(component.topology_node_count for component in result.components)),
        "odd_degree_nodes_before_routing": int(sum(component.odd_before for component in result.components)),
        "odd_degree_nodes_after_routing": int(sum(component.odd_after for component in result.components)),
        "retraced_graph_edge_count_px": int(sum(component.duplicated_edge_count for component in result.components)),
        "retraced_graph_fraction": (
            sum(component.duplicated_edge_count for component in result.components)
            / max(sum(component.original_edge_count for component in result.components), 1)
        ),
        "endpoint_count": result.endpoint_count,
        "corner_count": result.corner_count,
        "junction_count": result.junction_count,
        "traced_endpoint_count": result.traced_endpoint_count,
        "traced_endpoint_usage_fraction": result.traced_endpoint_count / max(result.endpoint_count, 1),
        "tangent_consistency_score": result.tangent_consistency_score,
        "stroke_point_counts": [int(count) for count in counts],
        "component_summaries": [
            {
                "id": component.id,
                "node_count": component.node_count,
                "topology_node_count": component.topology_node_count,
                "original_edge_count": component.original_edge_count,
                "duplicated_edge_count": component.duplicated_edge_count,
                "odd_before": component.odd_before,
                "odd_after": component.odd_after,
                "closed": component.closed,
                "raw_point_count": int(len(component.raw_route_px)),
                "simplified_point_count": int(len(component.simplified_route_px)),
            }
            for component in result.components
        ],
        "thresholds": {
            "support_threshold": args.support_threshold,
            "support_bridge_threshold": args.support_bridge_threshold,
            "centreline_threshold": args.centreline_threshold,
            "centreline_seed_threshold": args.centreline_seed_threshold,
            "endpoint_threshold": args.endpoint_threshold,
            "corner_threshold": args.corner_threshold,
            "junction_threshold": args.junction_threshold,
            "tangent_valid_threshold": args.tangent_valid_threshold,
            "coverage_radius_px": args.coverage_radius_px,
            "false_positive_radius_px": args.false_positive_radius_px,
            "simplification_epsilon": args.simplification_epsilon,
        },
        "gantry_mapping": transform_info,
        "firmware_constants": firmware_constants,
        "model_path_used": labels.model_path,
        "label_schema_used": labels.label_schema,
    }


def save_reconstruction_debug(labels: RouteLabels, result: RouteResult, output_path: Path) -> None:
    rgb = np.dstack([labels.gray, labels.gray, labels.gray]).astype(np.uint8)
    rgb[labels.support_mask] = [224, 224, 224]
    rgb[result.missed_support_mask] = [235, 70, 60]
    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image)
    palette = [(20, 130, 230), (20, 170, 90), (230, 140, 20), (150, 80, 220), (210, 60, 150)]
    for index, stroke in enumerate(result.strokes_px):
        if len(stroke) < 2:
            continue
        color = palette[index % len(palette)]
        draw.line([tuple(point) for point in stroke], fill=color, width=1)
        x0, y0 = stroke[0]
        x1, y1 = stroke[-1]
        draw.ellipse((x0 - 3, y0 - 3, x0 + 3, y0 + 3), fill=(0, 180, 80))
        draw.ellipse((x1 - 3, y1 - 3, x1 + 3, y1 + 3), fill=(40, 80, 230))
    image.save(output_path)


def save_route_graph_debug(labels: RouteLabels, result: RouteResult, output_path: Path) -> None:
    rgb = np.dstack([labels.gray, labels.gray, labels.gray]).astype(np.uint8)
    rgb[result.drawable_mask] = [230, 230, 230]
    rgb[result.skeleton_mask] = [40, 135, 230]
    degree = topology_degree(result.skeleton_mask)
    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image)
    for pixel, count in degree.items():
        if count != 2:
            y, x = pixel
            color = (230, 50, 60) if count % 2 == 1 else (250, 170, 20)
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
    for y, x, _ in heatmap_centres(labels.endpoint_mask, labels.endpoint_prob):
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline=(0, 160, 70), width=2)
    image.save(output_path)


def save_route_sequence_debug(labels: RouteLabels, result: RouteResult, output_path: Path) -> None:
    image = Image.new("RGB", (labels.gray.shape[1], labels.gray.shape[0]), "white")
    draw = ImageDraw.Draw(image)
    draw.bitmap((0, 0), Image.fromarray(np.where(labels.support_mask, 80, 0).astype(np.uint8), mode="L"), fill=(225, 225, 225))
    palette = [(20, 130, 230), (20, 170, 90), (230, 140, 20), (150, 80, 220), (210, 60, 150)]
    for index, stroke in enumerate(result.strokes_px):
        if len(stroke) < 2:
            continue
        draw.line([tuple(point) for point in stroke], fill=palette[index % len(palette)], width=1)
        for marker_index in np.linspace(0, len(stroke) - 1, min(10, len(stroke))).astype(int):
            x, y = stroke[marker_index]
            draw.ellipse((x - 1.5, y - 1.5, x + 1.5, y + 1.5), fill=(20, 20, 20))
    image.save(output_path)


def save_false_positive_drawn(labels: RouteLabels, result: RouteResult, output_path: Path) -> None:
    drawn = result.drawn_mask_for_false_positive
    rgb = np.dstack([labels.gray, labels.gray, labels.gray]).astype(np.uint8)
    rgb[drawn] = [45, 145, 230]
    rgb[result.false_positive_mask] = [230, 50, 60]
    Image.fromarray(rgb, mode="RGB").save(output_path)


def run_pipeline(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = load_labels(args)
    result = decode_routes(labels, args)
    paths_mm, transform_info = map_paths_to_gantry_mm(
        result.strokes_px,
        labels.support_mask.shape,
        work_w_mm=args.work_width_mm,
        work_h_mm=args.work_height_mm,
        margin_mm=args.margin_mm,
    )
    commands = paths_to_arduino_commands(paths_mm)
    validate_arduino_commands(commands, args.work_width_mm, args.work_height_mm)
    repo_root = Path(__file__).resolve().parents[1]
    firmware_constants = parse_firmware_constants(repo_root.parent / "Arduino Code" / "ArduinoCode.ino")
    metrics = compute_metrics(commands, result, labels, transform_info, firmware_constants, args)
    save_arduino_commands(commands, output_dir / "arduino_commands.txt")
    save_json(metrics, output_dir / "stroke_metrics.json")
    save_gantry_preview(paths_mm, output_dir / "gantry_path_preview.png", args.work_width_mm, args.work_height_mm, args.margin_mm)
    save_reconstruction_debug(labels, result, output_dir / "reconstruction_debug.png")
    save_route_graph_debug(labels, result, output_dir / "route_graph_debug.png")
    save_route_sequence_debug(labels, result, output_dir / "route_sequence_debug.png")
    Image.fromarray(np.where(result.missed_support_mask, 255, 0).astype(np.uint8), mode="L").save(output_dir / "missed_support_pixels.png")
    save_false_positive_drawn(labels, result, output_dir / "false_positive_drawn_pixels.png")
    print(f"Output directory: {output_dir}")
    print(f"Strokes: {metrics['stroke_count']}")
    print(f"Commands: {metrics['command_count']}")
    print(f"Support coverage: {metrics['support_coverage']:.3f}")
    print(f"Pen-up travel mm: {metrics['pen_up_travel_distance_mm']:.1f}")
    print(f"Odd nodes before/after: {metrics['odd_degree_nodes_before_routing']} -> {metrics['odd_degree_nodes_after_routing']}")
    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run route-first reconstruction from rich stroke-structure masks.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--prediction-arrays")
    parser.add_argument("--model-path")
    parser.add_argument("--output-dir", default="output/stroke_route_decoder")
    parser.add_argument("--support-threshold", type=float, default=0.45)
    parser.add_argument("--support-bridge-threshold", type=float, default=0.42)
    parser.add_argument("--centreline-threshold", type=float, default=0.30)
    parser.add_argument("--centreline-seed-threshold", type=float, default=0.25)
    parser.add_argument("--endpoint-threshold", type=float, default=0.35)
    parser.add_argument("--corner-threshold", type=float, default=0.35)
    parser.add_argument("--junction-threshold", type=float, default=0.35)
    parser.add_argument("--tangent-valid-threshold", type=float, default=0.35)
    parser.add_argument("--bridge-seed-radius-px", type=int, default=2)
    parser.add_argument("--centreline-repair-radius-px", type=int, default=1)
    parser.add_argument("--close-iterations", type=int, default=0)
    parser.add_argument("--min-component-pixels", type=int, default=4)
    parser.add_argument("--min-skeleton-component-pixels", type=int, default=4)
    parser.add_argument("--protected-radius-px", type=int, default=2)
    parser.add_argument("--spur-prune-length-px", type=int, default=2)
    parser.add_argument("--max-skeleton-bridge-px", type=int, default=45)
    parser.add_argument("--endpoint-snap-radius-px", type=float, default=18.0)
    parser.add_argument("--endpoint-usage-radius-px", type=float, default=5.0)
    parser.add_argument("--pin-snap-radius-px", type=int, default=3)
    parser.add_argument("--pin-topology-radius-px", type=int, default=-1)
    parser.add_argument("--pin-heatmap-snap-radius-px", type=float, default=10.0)
    parser.add_argument("--pin-heatmap-radius-px", type=int, default=1)
    parser.add_argument("--simplification-epsilon", type=float, default=1.5)
    parser.add_argument("--simplify-max-segment-points", type=int, default=180)
    parser.add_argument("--max-draw-jump-px", type=float, default=80.0)
    parser.add_argument("--jump-support-fraction", type=float, default=0.35)
    parser.add_argument("--coverage-radius-px", type=int, default=3)
    parser.add_argument("--false-positive-radius-px", type=int, default=1)
    parser.add_argument("--work-width-mm", type=float, default=150.0)
    parser.add_argument("--work-height-mm", type=float, default=270.0)
    parser.add_argument("--margin-mm", type=float, default=5.0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
