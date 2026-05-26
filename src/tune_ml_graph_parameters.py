"""Tune ML-only graph reconstruction parameters for gantry stroke planning.

This script runs ML segmentation once, sweeps graph reconstruction parameters,
renders each predicted stroke set back into image space, and ranks candidates
against either the ML line probability mask or a grayscale-derived line mask.

It does not send commands to Arduino and does not use skeletonisation.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from stroke_based_pipeline import (
    map_paths_to_gantry_mm,
    parse_firmware_constants,
    paths_to_arduino_commands,
    save_arduino_commands,
    save_gantry_preview,
    save_json,
)
from stroke_ml_graph_pipeline import (
    MLGraphResult,
    MLProbabilities,
    build_arg_parser as build_graph_arg_parser,
    build_ml_graph,
    compute_metrics,
    load_torch_probabilities,
    save_strokes_debug,
)


Command = tuple[float, float, int]


@dataclass
class RunArtifact:
    index: int
    params: dict[str, Any]
    score: dict[str, float]
    metrics: dict[str, Any]
    graph: MLGraphResult
    commands: list[Command]
    paths_mm: list[np.ndarray]
    prediction_mask: np.ndarray


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def resolve_repo_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (repo_root / path).resolve()


def make_target_mask(probabilities: MLProbabilities, source: str, threshold: float | None) -> np.ndarray:
    if source == "ml_line":
        cutoff = 0.35 if threshold is None else float(threshold)
        return probabilities.line_prob >= cutoff
    cutoff = 180.0 if threshold is None else float(threshold)
    return probabilities.gray <= cutoff


def grayscale_target_mask(probabilities: MLProbabilities, threshold: float | None) -> np.ndarray:
    cutoff = 180.0 if threshold is None else float(threshold)
    return probabilities.gray <= cutoff


def ml_line_target_mask(probabilities: MLProbabilities, threshold: float | None) -> np.ndarray:
    cutoff = 0.35 if threshold is None else float(threshold)
    return probabilities.line_prob >= cutoff


def render_strokes_to_mask(strokes_px: list[np.ndarray], shape: tuple[int, int], line_width_px: int) -> np.ndarray:
    height, width = shape
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    for stroke in strokes_px:
        if len(stroke) == 0:
            continue
        points = [(float(x), float(y)) for x, y in stroke]
        if len(points) == 1:
            x, y = points[0]
            r = max(1, line_width_px // 2)
            draw.ellipse((x - r, y - r, x + r, y + r), fill=255)
        else:
            draw.line(points, fill=255, width=max(1, line_width_px), joint="curve")
    return np.asarray(image, dtype=np.uint8) > 0


def distance_transform(mask: np.ndarray) -> np.ndarray:
    """Approximate Euclidean distance to the nearest true pixel using two passes."""
    height, width = mask.shape
    diagonal = math.hypot(height, width)
    dist = np.where(mask, 0.0, diagonal).astype(np.float32)
    root2 = math.sqrt(2.0)

    for y in range(height):
        for x in range(width):
            current = dist[y, x]
            if y > 0:
                current = min(current, dist[y - 1, x] + 1.0)
                if x > 0:
                    current = min(current, dist[y - 1, x - 1] + root2)
                if x + 1 < width:
                    current = min(current, dist[y - 1, x + 1] + root2)
            if x > 0:
                current = min(current, dist[y, x - 1] + 1.0)
            dist[y, x] = current

    for y in range(height - 1, -1, -1):
        for x in range(width - 1, -1, -1):
            current = dist[y, x]
            if y + 1 < height:
                current = min(current, dist[y + 1, x] + 1.0)
                if x > 0:
                    current = min(current, dist[y + 1, x - 1] + root2)
                if x + 1 < width:
                    current = min(current, dist[y + 1, x + 1] + root2)
            if x + 1 < width:
                current = min(current, dist[y, x + 1] + 1.0)
            dist[y, x] = current
    return dist


def evaluate_prediction_mask(
    prediction_mask: np.ndarray,
    target_mask: np.ndarray,
    tolerance_px: float,
) -> dict[str, float]:
    target_count = int(np.count_nonzero(target_mask))
    prediction_count = int(np.count_nonzero(prediction_mask))
    diagonal = math.hypot(*target_mask.shape)

    if target_count == 0 and prediction_count == 0:
        return {
            "target_coverage": 1.0,
            "false_positive_fraction": 0.0,
            "chamfer_distance_px": 0.0,
            "chamfer_score": 1.0,
        }
    if target_count == 0:
        return {
            "target_coverage": 0.0,
            "false_positive_fraction": 1.0,
            "chamfer_distance_px": diagonal,
            "chamfer_score": 0.0,
        }
    if prediction_count == 0:
        return {
            "target_coverage": 0.0,
            "false_positive_fraction": 0.0,
            "chamfer_distance_px": diagonal,
            "chamfer_score": 0.0,
        }

    dist_to_prediction = distance_transform(prediction_mask)
    dist_to_target = distance_transform(target_mask)
    covered_target = float(np.count_nonzero(target_mask & (dist_to_prediction <= tolerance_px)) / target_count)
    false_positive = float(np.count_nonzero(prediction_mask & (dist_to_target > tolerance_px)) / prediction_count)
    pred_to_target = float(np.mean(dist_to_target[prediction_mask]))
    target_to_pred = float(np.mean(dist_to_prediction[target_mask]))
    chamfer = 0.5 * (pred_to_target + target_to_pred)
    chamfer_score = max(0.0, 1.0 - chamfer / max(diagonal * 0.15, 1e-6))
    return {
        "target_coverage": covered_target,
        "false_positive_fraction": false_positive,
        "chamfer_distance_px": chamfer,
        "chamfer_score": chamfer_score,
    }


def edge_penalty_terms(graph: MLGraphResult, args: argparse.Namespace) -> dict[str, float]:
    accepted = graph.accepted_edges
    if not accepted:
        return {
            "low_support_edge_count": 0.0,
            "suspicious_long_edge_count": 0.0,
            "mean_path_length_ratio": 0.0,
        }
    low_support = sum(1 for edge in accepted if edge.support_fraction < args.suspicious_edge_support)
    suspicious_long = sum(
        1
        for edge in accepted
        if edge.distance_px >= args.suspicious_edge_distance_px
        and edge.support_fraction < args.suspicious_edge_support
    )
    ratios = [edge.path_length_ratio for edge in accepted if math.isfinite(edge.path_length_ratio)]
    return {
        "low_support_edge_count": float(low_support),
        "suspicious_long_edge_count": float(suspicious_long),
        "mean_path_length_ratio": float(np.mean(ratios)) if ratios else 0.0,
    }


def score_run(
    mask_scores: dict[str, float],
    metrics: dict[str, Any],
    graph: MLGraphResult,
    args: argparse.Namespace,
) -> dict[str, float]:
    graph_metrics = metrics.get("graph", {})
    command_count = float(metrics.get("command_count", 0))
    stroke_count = float(metrics.get("stroke_count", 0))
    travel_distance = float(metrics.get("pen_up_travel_distance_mm", 0.0))
    bounds_ok = bool(metrics.get("bounds_validation_passed", False))
    line_coverage_fraction = float(graph_metrics.get("line_coverage_fraction", 0.0))
    edge_terms = edge_penalty_terms(graph, args)

    command_penalty = min(command_count / max(args.command_penalty_scale, 1.0), 1.0)
    travel_penalty = min(travel_distance / max(args.travel_penalty_scale_mm, 1.0), 1.0)
    stroke_penalty = min(stroke_count / max(args.stroke_penalty_scale, 1.0), 1.0)
    suspicious_penalty = min(edge_terms["suspicious_long_edge_count"] / max(args.suspicious_edge_penalty_scale, 1.0), 1.0)
    invalid_penalty = 1.0 if not bounds_ok else 0.0

    total = (
        args.coverage_weight * mask_scores["target_coverage"]
        + args.false_positive_weight * (1.0 - mask_scores["false_positive_fraction"])
        + args.chamfer_weight * mask_scores["chamfer_score"]
        + args.graph_coverage_weight * line_coverage_fraction
        - args.command_penalty_weight * command_penalty
        - args.travel_penalty_weight * travel_penalty
        - args.stroke_penalty_weight * stroke_penalty
        - args.suspicious_edge_penalty_weight * suspicious_penalty
        - args.invalid_bounds_penalty_weight * invalid_penalty
    )

    return {
        "total_score": float(total),
        "line_coverage_score": float(mask_scores["target_coverage"]),
        "false_positive_score": float(1.0 - mask_scores["false_positive_fraction"]),
        "false_positive_fraction": float(mask_scores["false_positive_fraction"]),
        "chamfer_like_score": float(mask_scores["chamfer_score"]),
        "chamfer_distance_px": float(mask_scores["chamfer_distance_px"]),
        "graph_line_coverage_fraction": line_coverage_fraction,
        "command_penalty": float(command_penalty),
        "travel_penalty": float(travel_penalty),
        "stroke_penalty": float(stroke_penalty),
        "suspicious_edge_penalty": float(suspicious_penalty),
        "invalid_bounds_penalty": float(invalid_penalty),
        **edge_terms,
    }


def save_mask_preview(
    gray: np.ndarray,
    target_mask: np.ndarray,
    prediction_mask: np.ndarray,
    output_path: Path,
) -> None:
    base = Image.fromarray(gray, mode="L").convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    target_pixels = np.argwhere(target_mask)
    prediction_pixels = np.argwhere(prediction_mask)
    for y, x in target_pixels:
        draw.point((int(x), int(y)), fill=(40, 120, 255, 100))
    for y, x in prediction_pixels:
        draw.point((int(x), int(y)), fill=(230, 40, 40, 180))
    Image.alpha_composite(base, overlay).convert("RGB").save(output_path)


def diagnostic_counts(
    original_mask: np.ndarray,
    ml_mask: np.ndarray,
    prediction_mask: np.ndarray,
) -> dict[str, int | float]:
    total_original = int(np.count_nonzero(original_mask))
    total_ml = int(np.count_nonzero(ml_mask))
    total_prediction = int(np.count_nonzero(prediction_mask))
    original_and_ml = original_mask & ml_mask
    original_not_ml = original_mask & ~ml_mask
    original_not_prediction = original_mask & ~prediction_mask
    ml_not_prediction = ml_mask & ~prediction_mask
    prediction_not_original = prediction_mask & ~original_mask
    prediction_not_ml = prediction_mask & ~ml_mask
    graph_missed_original_supported_by_ml = original_mask & ml_mask & ~prediction_mask
    segmentation_missed_original = original_not_ml
    graph_drew_ml_false_positive = prediction_mask & ml_mask & ~original_mask
    graph_false_connector_unsupported = prediction_mask & ~ml_mask & ~original_mask
    return {
        "original_line_pixels": total_original,
        "ml_line_pixels": total_ml,
        "prediction_pixels": total_prediction,
        "original_and_ml_pixels": int(np.count_nonzero(original_and_ml)),
        "original_missing_from_ml_pixels": int(np.count_nonzero(segmentation_missed_original)),
        "original_missing_from_prediction_pixels": int(np.count_nonzero(original_not_prediction)),
        "ml_missing_from_prediction_pixels": int(np.count_nonzero(ml_not_prediction)),
        "prediction_not_original_pixels": int(np.count_nonzero(prediction_not_original)),
        "prediction_not_ml_pixels": int(np.count_nonzero(prediction_not_ml)),
        "graph_missed_original_supported_by_ml_pixels": int(np.count_nonzero(graph_missed_original_supported_by_ml)),
        "graph_drew_ml_false_positive_pixels": int(np.count_nonzero(graph_drew_ml_false_positive)),
        "graph_false_connector_unsupported_pixels": int(np.count_nonzero(graph_false_connector_unsupported)),
        "original_supported_by_ml_fraction": float(np.count_nonzero(original_and_ml) / max(total_original, 1)),
        "segmentation_miss_fraction": float(np.count_nonzero(segmentation_missed_original) / max(total_original, 1)),
        "graph_miss_fraction_of_original": float(np.count_nonzero(original_not_prediction) / max(total_original, 1)),
        "graph_miss_fraction_of_ml": float(np.count_nonzero(ml_not_prediction) / max(total_ml, 1)),
    }


def save_three_way_diagnostic(
    gray: np.ndarray,
    original_mask: np.ndarray,
    ml_mask: np.ndarray,
    prediction_mask: np.ndarray,
    output_path: Path,
) -> dict[str, int | float]:
    """Save a mask diagnostic separating segmentation and graph failures.

    Colors:
    - green: original line reconstructed.
    - blue: original line exists and ML line exists, but graph missed it.
    - yellow: original line exists but ML line mask missed it.
    - red: graph drew outside the original and outside the ML mask.
    - purple: graph drew an ML-supported line outside the original target.
    - gray: ML line exists but neither original nor prediction contains it.
    """
    counts = diagnostic_counts(original_mask, ml_mask, prediction_mask)
    height, width = gray.shape
    image = np.full((height, width, 3), 255, dtype=np.uint8)

    ml_only_unclaimed = ml_mask & ~original_mask & ~prediction_mask
    graph_hit_original = original_mask & prediction_mask
    graph_missed_ml_supported = original_mask & ml_mask & ~prediction_mask
    segmentation_missed_original = original_mask & ~ml_mask
    graph_ml_false_positive = prediction_mask & ml_mask & ~original_mask
    graph_unsupported_false_positive = prediction_mask & ~ml_mask & ~original_mask

    image[ml_only_unclaimed] = (180, 180, 180)
    image[graph_hit_original] = (20, 170, 80)
    image[graph_missed_ml_supported] = (35, 90, 235)
    image[segmentation_missed_original] = (245, 190, 30)
    image[graph_ml_false_positive] = (170, 60, 210)
    image[graph_unsupported_false_positive] = (230, 45, 45)

    legend_h = 86
    canvas = Image.new("RGB", (width, height + legend_h), "white")
    canvas.paste(Image.fromarray(image, mode="RGB"), (0, 0))
    draw = ImageDraw.Draw(canvas)
    legend_items = [
        ((20, 170, 80), "green: original reconstructed"),
        ((35, 90, 235), "blue: graph missed original line that ML sees"),
        ((245, 190, 30), "yellow: segmentation missed original line"),
        ((170, 60, 210), "purple: graph drew ML-only line"),
        ((230, 45, 45), "red: graph drew unsupported line"),
        ((180, 180, 180), "gray: ML-only unclaimed line"),
    ]
    x = 8
    y = height + 8
    for color, label in legend_items:
        draw.rectangle((x, y + 3, x + 14, y + 17), fill=color)
        draw.text((x + 20, y), label, fill=(20, 20, 20))
        y += 24
        if y > height + legend_h - 20:
            x += 250
            y = height + 8
    canvas.save(output_path)
    return counts


def make_graph_args(tuner_args: argparse.Namespace, params: dict[str, Any]) -> argparse.Namespace:
    graph_args = build_graph_arg_parser().parse_args(
        [
            "--image",
            str(tuner_args.image_path),
            "--model-path",
            str(tuner_args.model_path),
            "--output-dir",
            str(tuner_args.output_dir),
        ]
    )
    graph_args.work_width_mm = tuner_args.work_width_mm
    graph_args.work_height_mm = tuner_args.work_height_mm
    graph_args.margin_mm = tuner_args.margin_mm
    graph_args.edge_search_mode = "path"
    graph_args.path_max_expanded_nodes = tuner_args.path_max_expanded_nodes
    graph_args.path_corridor_margin_px = tuner_args.path_corridor_margin_px
    graph_args.max_path_to_straight_ratio = tuner_args.max_path_to_straight_ratio
    graph_args.component_max_edge_distance_px = tuner_args.component_max_edge_distance_px
    graph_args.nearest_neighbors = tuner_args.nearest_neighbors
    graph_args.coverage_radius_px = tuner_args.coverage_radius_px
    graph_args.min_edge_new_pixels = tuner_args.min_edge_new_pixels
    graph_args.enable_node_topology = tuner_args.enable_node_topology or tuner_args.enable_node_port_routing
    graph_args.enable_node_port_routing = tuner_args.enable_node_port_routing

    for key, value in params.items():
        setattr(graph_args, key, value)
    return graph_args


def parameter_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    names_and_values = [
        ("node_threshold", parse_float_list(args.node_thresholds)),
        ("line_threshold", parse_float_list(args.line_thresholds)),
        ("edge_score_threshold", parse_float_list(args.edge_score_thresholds)),
        ("support_fraction_threshold", parse_float_list(args.support_fraction_thresholds)),
        ("path_min_support_fraction", parse_float_list(args.path_min_support_fractions)),
        ("vertex_passthrough_radius_px", parse_float_list(args.vertex_passthrough_radii)),
        ("line_anchor_min_distance_px", parse_float_list(args.line_anchor_distances)),
        ("max_line_anchors_per_component", parse_int_list(args.max_line_anchors)),
        ("component_vertex_radius_px", parse_float_list(args.component_vertex_radii)),
        ("min_edge_new_coverage_fraction", parse_float_list(args.min_new_coverage_fractions)),
        ("max_degree", parse_int_list(args.max_degrees)),
        ("node_port_radius_px", parse_float_list(args.node_port_radii)),
        ("node_port_min_line_prob", parse_float_list(args.node_port_min_line_probs)),
        ("node_route_angle_threshold", parse_float_list(args.node_route_angle_thresholds)),
        ("node_route_support_weight", parse_float_list(args.node_route_support_weights)),
    ]
    combos = [
        dict(zip([name for name, _ in names_and_values], values))
        for values in itertools.product(*[values for _, values in names_and_values])
    ]
    if args.grid_sampling == "random" and args.max_runs and len(combos) > args.max_runs:
        rng = np.random.default_rng(args.seed)
        indices = sorted(rng.choice(len(combos), size=args.max_runs, replace=False).tolist())
        combos = [combos[i] for i in indices]
    elif args.max_runs:
        combos = combos[: args.max_runs]
    return combos


def summarize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    graph = metrics.get("graph", {})
    return {
        "command_count": metrics.get("command_count"),
        "stroke_count": metrics.get("stroke_count"),
        "pen_up_travel_distance_mm": metrics.get("pen_up_travel_distance_mm"),
        "pen_down_drawing_distance_mm": metrics.get("pen_down_drawing_distance_mm"),
        "bounds_validation_passed": metrics.get("bounds_validation_passed"),
        "graph_vertex_count": graph.get("vertex_count"),
        "accepted_edge_count": graph.get("accepted_edge_count"),
        "line_coverage_fraction": graph.get("line_coverage_fraction"),
        "rejected_by_vertex_passthrough_count": graph.get("rejected_by_vertex_passthrough_count"),
        "rejected_by_coverage_pruning_count": graph.get("rejected_by_coverage_pruning_count"),
    }


def run_one(
    index: int,
    params: dict[str, Any],
    probabilities: MLProbabilities,
    target_mask: np.ndarray,
    firmware_constants: dict[str, Any],
    tuner_args: argparse.Namespace,
) -> RunArtifact:
    graph_args = make_graph_args(tuner_args, params)
    graph = build_ml_graph(probabilities, graph_args)
    paths_mm, transform_info = map_paths_to_gantry_mm(
        graph.strokes_px,
        image_shape=probabilities.gray.shape,
        work_w_mm=graph_args.work_width_mm,
        work_h_mm=graph_args.work_height_mm,
        margin_mm=graph_args.margin_mm,
        centre_on_page=True,
    )
    commands = paths_to_arduino_commands(paths_mm)
    metrics = compute_metrics(commands, graph, probabilities, transform_info, firmware_constants, graph_args)
    prediction_mask = render_strokes_to_mask(graph.strokes_px, probabilities.gray.shape, tuner_args.render_line_width_px)
    mask_scores = evaluate_prediction_mask(prediction_mask, target_mask, tuner_args.match_tolerance_px)
    score = score_run(mask_scores, metrics, graph, tuner_args)
    return RunArtifact(
        index=index,
        params=params,
        score=score,
        metrics=metrics,
        graph=graph,
        commands=commands,
        paths_mm=paths_mm,
        prediction_mask=prediction_mask,
    )


def write_results_table(results: list[RunArtifact], output_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for rank, artifact in enumerate(results, start=1):
        row = {
            "rank": rank,
            "run_index": artifact.index,
            **artifact.params,
            **artifact.score,
            **summarize_metrics(artifact.metrics),
        }
        rows.append(row)

    save_json(rows, output_dir / "tuning_results.json")
    if rows:
        fieldnames = list(rows[0].keys())
        with (output_dir / "tuning_results.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return rows


def save_top_artifacts(
    results: list[RunArtifact],
    probabilities: MLProbabilities,
    target_mask: np.ndarray,
    original_mask: np.ndarray,
    ml_mask: np.ndarray,
    output_dir: Path,
    top_k: int,
    tuner_args: argparse.Namespace,
) -> None:
    for rank, artifact in enumerate(results[:top_k], start=1):
        prefix = f"top_{rank:03d}"
        preview_path = output_dir / f"{prefix}_preview.png"
        save_mask_preview(probabilities.gray, target_mask, artifact.prediction_mask, preview_path)
        if tuner_args.save_per_run:
            run_dir = output_dir / f"{prefix}_run_{artifact.index:04d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            save_arduino_commands(artifact.commands, run_dir / "arduino_commands.txt")
            save_json(artifact.metrics, run_dir / "stroke_metrics.json")
            save_strokes_debug(probabilities, artifact.graph, run_dir / "stroke_sequence_debug.png")
            save_gantry_preview(
                artifact.paths_mm,
                run_dir / "gantry_path_preview.png",
                tuner_args.work_width_mm,
                tuner_args.work_height_mm,
                tuner_args.margin_mm,
            )

    if not results:
        return
    best = results[0]
    save_arduino_commands(best.commands, output_dir / "best_arduino_commands.txt")
    save_gantry_preview(
        best.paths_mm,
        output_dir / "best_gantry_path_preview.png",
        tuner_args.work_width_mm,
        tuner_args.work_height_mm,
        tuner_args.margin_mm,
    )
    save_strokes_debug(probabilities, best.graph, output_dir / "best_stroke_sequence_debug.png")
    save_mask_preview(probabilities.gray, target_mask, best.prediction_mask, output_dir / "best_mask_comparison.png")
    diagnostic_summary = save_three_way_diagnostic(
        probabilities.gray,
        original_mask,
        ml_mask,
        best.prediction_mask,
        output_dir / "best_three_way_diagnostic.png",
    )
    save_json(diagnostic_summary, output_dir / "best_three_way_diagnostic.json")
    save_json(
        {
            "run_index": best.index,
            "params": best.params,
            "score": best.score,
            "metrics_summary": summarize_metrics(best.metrics),
            "three_way_diagnostic": diagnostic_summary,
            "outputs": {
                "best_arduino_commands": "best_arduino_commands.txt",
                "best_gantry_path_preview": "best_gantry_path_preview.png",
                "best_stroke_sequence_debug": "best_stroke_sequence_debug.png",
                "best_mask_comparison": "best_mask_comparison.png",
                "best_three_way_diagnostic": "best_three_way_diagnostic.png",
            },
        },
        output_dir / "best_params.json",
    )


def run_tuning(args: argparse.Namespace) -> list[RunArtifact]:
    repo_root = Path(__file__).resolve().parents[1]
    args.image_path = resolve_repo_path(repo_root, args.image)
    args.model_path = resolve_repo_path(repo_root, args.model_path)
    args.output_dir = resolve_repo_path(repo_root, args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    probabilities = load_torch_probabilities(args.image_path, args.model_path)
    target_mask = make_target_mask(probabilities, args.target_source, args.target_threshold)
    original_mask = grayscale_target_mask(probabilities, args.grayscale_target_threshold)
    ml_mask = ml_line_target_mask(probabilities, args.ml_line_target_threshold)
    Image.fromarray(np.where(target_mask, 255, 0).astype(np.uint8), mode="L").save(args.output_dir / "target_mask.png")
    Image.fromarray(np.where(original_mask, 255, 0).astype(np.uint8), mode="L").save(args.output_dir / "original_grayscale_mask.png")
    Image.fromarray(np.where(ml_mask, 255, 0).astype(np.uint8), mode="L").save(args.output_dir / "ml_line_mask.png")
    firmware_constants = parse_firmware_constants(repo_root.parent / "Arduino Code" / "ArduinoCode.ino")

    combos = parameter_grid(args)
    if not combos:
        raise ValueError("Parameter grid is empty.")

    print(f"Image: {args.image_path}")
    print(f"Model: {args.model_path}")
    print(f"Output: {args.output_dir}")
    print(f"Target source: {args.target_source}")
    print(f"Runs: {len(combos)}")

    results: list[RunArtifact] = []
    for index, params in enumerate(combos, start=1):
        artifact = run_one(index, params, probabilities, target_mask, firmware_constants, args)
        results.append(artifact)
        print(
            f"[{index:03d}/{len(combos):03d}] "
            f"score={artifact.score['total_score']:.3f} "
            f"coverage={artifact.score['line_coverage_score']:.3f} "
            f"false_pos={artifact.score['false_positive_fraction']:.3f} "
            f"commands={artifact.metrics.get('command_count', 0)} "
            f"strokes={artifact.metrics.get('stroke_count', 0)}"
        )

    results.sort(key=lambda artifact: artifact.score["total_score"], reverse=True)
    write_results_table(results, args.output_dir)
    save_top_artifacts(results, probabilities, target_mask, original_mask, ml_mask, args.output_dir, args.top_k, args)
    print("Best score:", f"{results[0].score['total_score']:.3f}")
    print("Best params:", json.dumps(results[0].params, sort_keys=True))
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Sweep and score ML graph reconstruction parameters.")
    parser.add_argument("--image", default=str(repo_root / "input_images" / "cats.jpg"))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", default=str(repo_root / "output" / "ml_graph_tuning"))
    parser.add_argument("--target-source", choices=["ml_line", "grayscale"], default="ml_line")
    parser.add_argument("--target-threshold", type=float, default=None)
    parser.add_argument("--grayscale-target-threshold", type=float, default=None)
    parser.add_argument("--ml-line-target-threshold", type=float, default=None)
    parser.add_argument("--max-runs", type=int, default=80)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--grid-sampling", choices=["first", "random"], default="random")
    parser.add_argument("--seed", type=int, default=498)
    parser.add_argument("--save-per-run", action="store_true")

    parser.add_argument("--node-thresholds", default="0.30,0.35,0.40")
    parser.add_argument("--line-thresholds", default="0.30,0.35,0.40")
    parser.add_argument("--edge-score-thresholds", default="0.14,0.18,0.22")
    parser.add_argument("--support-fraction-thresholds", default="0.05")
    parser.add_argument("--path-min-support-fractions", default="0.30,0.35")
    parser.add_argument("--vertex-passthrough-radii", default="8,10,14")
    parser.add_argument("--line-anchor-distances", default="16,20,28")
    parser.add_argument("--max-line-anchors", default="4,8")
    parser.add_argument("--component-vertex-radii", default="14,18,24")
    parser.add_argument("--min-new-coverage-fractions", default="0.20,0.30,0.40")
    parser.add_argument("--max-degrees", default="2,3")
    parser.add_argument("--enable-node-topology", action="store_true")
    parser.add_argument("--enable-node-port-routing", action="store_true")
    parser.add_argument("--node-port-radii", default="22")
    parser.add_argument("--node-port-min-line-probs", default="0.25")
    parser.add_argument("--node-route-angle-thresholds", default="55")
    parser.add_argument("--node-route-support-weights", default="0.45")

    parser.add_argument("--path-max-expanded-nodes", type=int, default=30000)
    parser.add_argument("--path-corridor-margin-px", type=int, default=24)
    parser.add_argument("--max-path-to-straight-ratio", type=float, default=2.5)
    parser.add_argument("--component-max-edge-distance-px", type=float, default=1200.0)
    parser.add_argument("--nearest-neighbors", type=int, default=8)
    parser.add_argument("--coverage-radius-px", type=int, default=2)
    parser.add_argument("--min-edge-new-pixels", type=int, default=12)
    parser.add_argument("--render-line-width-px", type=int, default=3)
    parser.add_argument("--match-tolerance-px", type=float, default=4.0)

    parser.add_argument("--coverage-weight", type=float, default=100.0)
    parser.add_argument("--false-positive-weight", type=float, default=45.0)
    parser.add_argument("--chamfer-weight", type=float, default=30.0)
    parser.add_argument("--graph-coverage-weight", type=float, default=25.0)
    parser.add_argument("--command-penalty-weight", type=float, default=6.0)
    parser.add_argument("--travel-penalty-weight", type=float, default=8.0)
    parser.add_argument("--stroke-penalty-weight", type=float, default=3.0)
    parser.add_argument("--suspicious-edge-penalty-weight", type=float, default=12.0)
    parser.add_argument("--invalid-bounds-penalty-weight", type=float, default=1000.0)
    parser.add_argument("--command-penalty-scale", type=float, default=6000.0)
    parser.add_argument("--travel-penalty-scale-mm", type=float, default=1000.0)
    parser.add_argument("--stroke-penalty-scale", type=float, default=40.0)
    parser.add_argument("--suspicious-edge-distance-px", type=float, default=120.0)
    parser.add_argument("--suspicious-edge-support", type=float, default=0.45)
    parser.add_argument("--suspicious-edge-penalty-scale", type=float, default=4.0)

    parser.add_argument("--work-width-mm", type=float, default=150.0)
    parser.add_argument("--work-height-mm", type=float, default=270.0)
    parser.add_argument("--margin-mm", type=float, default=5.0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_tuning(args)


if __name__ == "__main__":
    main()
