"""Tune clean-slate ML support reconstruction parameters.

This tuner is for ``stroke_ml_reconstruction_pipeline.py``. It runs ML
segmentation once, sweeps reconstruction settings, renders each output stroke
set back into image space, compares it with a target mask, and ranks the runs.

It does not send commands to Arduino.
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
    Command,
    map_paths_to_gantry_mm,
    parse_firmware_constants,
    paths_to_arduino_commands,
    save_arduino_commands,
    save_gantry_preview,
    save_json,
    validate_arduino_commands,
)
from stroke_ml_reconstruction_pipeline import (
    MLProbabilities,
    ReconstructionResult,
    build_arg_parser as build_reconstruction_arg_parser,
    claimed_pixels_from_strokes,
    compute_metrics,
    load_torch_probabilities,
    reconstruct,
)


@dataclass
class RunArtifact:
    index: int
    params: dict[str, Any]
    score: dict[str, float]
    metrics: dict[str, Any]
    result: ReconstructionResult
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


def grayscale_target_mask(probabilities: MLProbabilities, threshold: float | None) -> np.ndarray:
    cutoff = 180.0 if threshold is None else float(threshold)
    return probabilities.gray <= cutoff


def ml_line_target_mask(probabilities: MLProbabilities, threshold: float | None) -> np.ndarray:
    cutoff = 0.35 if threshold is None else float(threshold)
    return probabilities.line_prob >= cutoff


def ml_support_target_mask(probabilities: MLProbabilities, threshold: float | None, node_weight: float) -> np.ndarray:
    cutoff = 0.30 if threshold is None else float(threshold)
    support = np.maximum(probabilities.line_prob, node_weight * probabilities.node_prob)
    return support >= cutoff


def make_target_mask(probabilities: MLProbabilities, args: argparse.Namespace) -> np.ndarray:
    if args.target_source == "grayscale":
        return grayscale_target_mask(probabilities, args.target_threshold)
    if args.target_source == "ml_line":
        return ml_line_target_mask(probabilities, args.target_threshold)
    return ml_support_target_mask(probabilities, args.target_threshold, args.target_node_weight)


def sample_points(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if len(points) <= max_points:
        return points.astype(np.float32)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(points), size=max_points, replace=False)
    return points[indices].astype(np.float32)


def mean_nearest_distance(source_points: np.ndarray, target_points: np.ndarray, max_points: int, seed: int) -> float:
    if len(source_points) == 0:
        return 0.0
    if len(target_points) == 0:
        return 999.0
    source = sample_points(source_points, max_points=max_points, seed=seed)
    target = sample_points(target_points, max_points=max_points, seed=seed + 17)
    distances: list[np.ndarray] = []
    chunk = 256
    for i in range(0, len(source), chunk):
        diff = source[i : i + chunk, None, :] - target[None, :, :]
        distances.append(np.sqrt(np.min(np.sum(diff * diff, axis=2), axis=1)))
    return float(np.mean(np.concatenate(distances))) if distances else 0.0


def chamfer_like_distance(prediction_mask: np.ndarray, target_mask: np.ndarray, max_points: int, seed: int) -> float:
    pred_points = np.argwhere(prediction_mask)
    target_points = np.argwhere(target_mask)
    if len(pred_points) == 0 and len(target_points) == 0:
        return 0.0
    if len(pred_points) == 0 or len(target_points) == 0:
        return 999.0
    pred_to_target = mean_nearest_distance(pred_points, target_points, max_points=max_points, seed=seed)
    target_to_pred = mean_nearest_distance(target_points, pred_points, max_points=max_points, seed=seed + 31)
    return 0.5 * (pred_to_target + target_to_pred)


def score_reconstruction(
    prediction_mask: np.ndarray,
    target_mask: np.ndarray,
    metrics: dict[str, Any],
    args: argparse.Namespace,
    seed: int,
) -> dict[str, float]:
    target_pixels = int(np.count_nonzero(target_mask))
    pred_pixels = int(np.count_nonzero(prediction_mask))
    overlap = int(np.count_nonzero(prediction_mask & target_mask))
    coverage = overlap / max(target_pixels, 1)
    false_positive_fraction = int(np.count_nonzero(prediction_mask & ~target_mask)) / max(pred_pixels, 1)
    chamfer = chamfer_like_distance(prediction_mask, target_mask, max_points=args.chamfer_sample_points, seed=seed)
    chamfer_score = math.exp(-chamfer / max(args.chamfer_scale_px, 1e-6))

    stroke_count = float(metrics.get("stroke_count", 0) or 0)
    command_count = float(metrics.get("command_count", 0) or 0)
    travel_mm = float(metrics.get("pen_up_travel_distance_mm", 0.0) or 0.0)
    draw_mm = float(metrics.get("pen_down_drawing_distance_mm", 0.0) or 0.0)
    bounds_valid = bool(metrics.get("bounds_validation_passed", False))
    reconstruction = metrics.get("reconstruction", {})
    support_coverage = float(reconstruction.get("recovered_support_coverage_fraction", 0.0) or 0.0)
    average_points = float(metrics.get("average_points_per_stroke", 0.0) or 0.0)

    command_penalty = min(command_count / max(args.command_penalty_scale, 1.0), 1.0)
    travel_penalty = min(travel_mm / max(args.travel_penalty_scale_mm, 1.0), 1.0)
    stroke_penalty = min(stroke_count / max(args.stroke_penalty_scale, 1.0), 1.0)
    fragmentation_bonus = min(average_points / max(args.avg_points_bonus_scale, 1.0), 1.0)
    zero_output_penalty = 1.0 if command_count <= 0 or draw_mm <= 0 else 0.0
    invalid_penalty = 0.0 if bounds_valid else 1.0

    total = (
        args.coverage_weight * coverage
        + args.support_coverage_weight * support_coverage
        + args.chamfer_weight * chamfer_score
        + args.fragmentation_bonus_weight * fragmentation_bonus
        - args.false_positive_weight * false_positive_fraction
        - args.command_penalty_weight * command_penalty
        - args.travel_penalty_weight * travel_penalty
        - args.stroke_penalty_weight * stroke_penalty
        - args.zero_output_penalty_weight * zero_output_penalty
        - args.invalid_bounds_penalty_weight * invalid_penalty
    )

    return {
        "total_score": float(total),
        "target_coverage_fraction": float(coverage),
        "false_positive_fraction": float(false_positive_fraction),
        "chamfer_like_distance_px": float(chamfer),
        "chamfer_score": float(chamfer_score),
        "support_coverage_fraction": float(support_coverage),
        "fragmentation_bonus": float(fragmentation_bonus),
        "command_penalty": float(command_penalty),
        "travel_penalty": float(travel_penalty),
        "stroke_penalty": float(stroke_penalty),
        "zero_output_penalty": float(zero_output_penalty),
        "invalid_bounds_penalty": float(invalid_penalty),
    }


def make_reconstruction_args(tuner_args: argparse.Namespace, params: dict[str, Any]) -> argparse.Namespace:
    graph_args = build_reconstruction_arg_parser().parse_args(
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
    for key, value in params.items():
        setattr(graph_args, key, value)
    return graph_args


def parameter_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    names_and_values = [
        ("line_threshold", parse_float_list(args.line_thresholds)),
        ("node_threshold", parse_float_list(args.node_thresholds)),
        ("support_threshold", parse_float_list(args.support_thresholds)),
        ("node_support_weight", parse_float_list(args.node_support_weights)),
        ("node_support_radius_px", parse_int_list(args.node_support_radii)),
        ("min_component_pixels", parse_int_list(args.min_component_pixels_values)),
        ("junction_cluster_radius_px", parse_int_list(args.junction_cluster_radii)),
        ("smooth_join_angle_deg", parse_float_list(args.smooth_join_angles)),
        ("corner_join_angle_deg", parse_float_list(args.corner_join_angles)),
        ("simplification_epsilon", parse_float_list(args.simplification_epsilons)),
        ("min_stroke_length_px", parse_float_list(args.min_stroke_lengths)),
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


def render_overlay_preview(gray: np.ndarray, prediction_mask: np.ndarray, target_mask: np.ndarray, output_path: Path) -> None:
    base = Image.fromarray(gray, mode="L").convert("RGB")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    target_only = target_mask & ~prediction_mask
    pred_only = prediction_mask & ~target_mask
    overlap = prediction_mask & target_mask
    for y, x in np.argwhere(target_only):
        draw.point((int(x), int(y)), fill=(40, 90, 255, 180))
    for y, x in np.argwhere(pred_only):
        draw.point((int(x), int(y)), fill=(230, 60, 60, 180))
    for y, x in np.argwhere(overlap):
        draw.point((int(x), int(y)), fill=(30, 170, 90, 210))
    Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB").save(output_path)


def flatten_record(artifact: RunArtifact) -> dict[str, Any]:
    reconstruction = artifact.metrics.get("reconstruction", {})
    record: dict[str, Any] = {
        "run_index": artifact.index,
        **artifact.params,
        **artifact.score,
        "command_count": artifact.metrics.get("command_count"),
        "stroke_count": artifact.metrics.get("stroke_count"),
        "average_points_per_stroke": artifact.metrics.get("average_points_per_stroke"),
        "median_points_per_stroke": artifact.metrics.get("median_points_per_stroke"),
        "pen_up_travel_distance_mm": artifact.metrics.get("pen_up_travel_distance_mm"),
        "pen_down_drawing_distance_mm": artifact.metrics.get("pen_down_drawing_distance_mm"),
        "bounds_validation_passed": artifact.metrics.get("bounds_validation_passed"),
        "support_component_count": reconstruction.get("support_connected_component_count"),
        "centreline_pixel_count": reconstruction.get("centreline_pixel_count"),
        "junction_zone_count": reconstruction.get("junction_zone_count"),
        "centreline_edge_count": reconstruction.get("centreline_edge_count"),
        "joined_stroke_count": reconstruction.get("joined_stroke_count"),
        "unclaimed_support_pixel_count": reconstruction.get("unclaimed_support_pixel_count"),
    }
    return record


def run_one(
    index: int,
    params: dict[str, Any],
    probabilities: MLProbabilities,
    target_mask: np.ndarray,
    firmware_constants: dict[str, Any],
    tuner_args: argparse.Namespace,
) -> RunArtifact:
    reconstruction_args = make_reconstruction_args(tuner_args, params)
    result = reconstruct(probabilities, reconstruction_args)
    paths_mm, transform_info = map_paths_to_gantry_mm(
        result.strokes_px,
        image_shape=probabilities.gray.shape,
        work_w_mm=tuner_args.work_width_mm,
        work_h_mm=tuner_args.work_height_mm,
        margin_mm=tuner_args.margin_mm,
        centre_on_page=True,
    )
    commands = paths_to_arduino_commands(paths_mm)
    try:
        validate_arduino_commands(commands, tuner_args.work_width_mm, tuner_args.work_height_mm)
    except ValueError:
        pass
    metrics = compute_metrics(commands, result, probabilities, transform_info, firmware_constants, reconstruction_args)
    prediction_mask = claimed_pixels_from_strokes(result.strokes_px, probabilities.gray.shape, radius=tuner_args.render_radius_px)
    score = score_reconstruction(prediction_mask, target_mask, metrics, tuner_args, seed=tuner_args.seed + index)
    return RunArtifact(
        index=index,
        params=params,
        score=score,
        metrics=metrics,
        result=result,
        commands=commands,
        paths_mm=paths_mm,
        prediction_mask=prediction_mask,
    )


def save_results(results: list[RunArtifact], target_mask: np.ndarray, probabilities: MLProbabilities, args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = [flatten_record(result) for result in results]
    save_json({"results": records}, args.output_dir / "tuning_results.json")
    if records:
        with (args.output_dir / "tuning_results.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

    if not results:
        return
    best = results[0]
    save_json(best.params, args.output_dir / "best_params.json")
    save_json(best.metrics, args.output_dir / "best_stroke_metrics.json")
    save_arduino_commands(best.commands, args.output_dir / "best_arduino_commands.txt")
    save_gantry_preview(best.paths_mm, args.output_dir / "best_gantry_path_preview.png", args.work_width_mm, args.work_height_mm, args.margin_mm)
    render_overlay_preview(probabilities.gray, best.prediction_mask, target_mask, args.output_dir / "best_overlay_preview.png")

    for rank, artifact in enumerate(results[: args.top_k], start=1):
        prefix = args.output_dir / f"top_{rank:03d}"
        render_overlay_preview(probabilities.gray, artifact.prediction_mask, target_mask, prefix.with_name(prefix.name + "_preview.png"))
        save_json(
            {
                "rank": rank,
                "run_index": artifact.index,
                "params": artifact.params,
                "score": artifact.score,
                "metrics_summary": flatten_record(artifact),
            },
            prefix.with_name(prefix.name + "_summary.json"),
        )


def run_tuning(args: argparse.Namespace) -> list[RunArtifact]:
    repo_root = Path(__file__).resolve().parents[1]
    args.image_path = resolve_repo_path(repo_root, args.image)
    args.model_path = resolve_repo_path(repo_root, args.model_path)
    args.output_dir = resolve_repo_path(repo_root, args.output_dir)

    probabilities = load_torch_probabilities(args.image_path, args.model_path)
    target_mask = make_target_mask(probabilities, args)
    firmware_constants = parse_firmware_constants(repo_root.parent / "Arduino Code" / "ArduinoCode.ino")
    combos = parameter_grid(args)
    results: list[RunArtifact] = []
    for index, params in enumerate(combos, start=1):
        artifact = run_one(index, params, probabilities, target_mask, firmware_constants, args)
        results.append(artifact)
        if args.print_progress:
            print(
                f"[{index}/{len(combos)}] score={artifact.score['total_score']:.3f} "
                f"coverage={artifact.score['target_coverage_fraction']:.3f} "
                f"strokes={artifact.metrics.get('stroke_count')} commands={artifact.metrics.get('command_count')}"
            )
    results.sort(key=lambda artifact: artifact.score["total_score"], reverse=True)
    save_results(results, target_mask, probabilities, args)
    if results:
        best = results[0]
        print(f"Best score: {best.score['total_score']:.3f}")
        print(f"Best params: {json.dumps(best.params, sort_keys=True)}")
        print(f"Best strokes/commands: {best.metrics.get('stroke_count')} / {best.metrics.get('command_count')}")
        print(f"Output directory: {args.output_dir}")
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Tune clean-slate ML reconstruction parameters.")
    parser.add_argument("--image", default=str(repo_root / "input_images" / "cats.jpg"))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", default=str(repo_root / "output" / "ml_reconstruction_tuning"))
    parser.add_argument("--target-source", choices=["grayscale", "ml_line", "ml_support"], default="grayscale")
    parser.add_argument("--target-threshold", type=float, default=None)
    parser.add_argument("--target-node-weight", type=float, default=0.65)
    parser.add_argument("--max-runs", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--grid-sampling", choices=["first", "random"], default="random")
    parser.add_argument("--seed", type=int, default=498)
    parser.add_argument("--print-progress", action="store_true")

    parser.add_argument("--line-thresholds", default="0.30,0.35,0.40")
    parser.add_argument("--node-thresholds", default="0.30,0.35,0.40")
    parser.add_argument("--support-thresholds", default="0.25,0.30,0.35")
    parser.add_argument("--node-support-weights", default="0.50,0.65,0.80")
    parser.add_argument("--node-support-radii", default="2,4,6")
    parser.add_argument("--min-component-pixels-values", default="4,8,12")
    parser.add_argument("--junction-cluster-radii", default="3,5,7,9")
    parser.add_argument("--smooth-join-angles", default="30,38,50,65")
    parser.add_argument("--corner-join-angles", default="100,125,145,165")
    parser.add_argument("--simplification-epsilons", default="0.25,0.5,0.75")
    parser.add_argument("--min-stroke-lengths", default="3,4,8")

    parser.add_argument("--render-radius-px", type=int, default=2)
    parser.add_argument("--chamfer-sample-points", type=int, default=2000)
    parser.add_argument("--chamfer-scale-px", type=float, default=6.0)
    parser.add_argument("--coverage-weight", type=float, default=110.0)
    parser.add_argument("--support-coverage-weight", type=float, default=55.0)
    parser.add_argument("--false-positive-weight", type=float, default=55.0)
    parser.add_argument("--chamfer-weight", type=float, default=35.0)
    parser.add_argument("--fragmentation-bonus-weight", type=float, default=10.0)
    parser.add_argument("--command-penalty-weight", type=float, default=5.0)
    parser.add_argument("--travel-penalty-weight", type=float, default=8.0)
    parser.add_argument("--stroke-penalty-weight", type=float, default=6.0)
    parser.add_argument("--zero-output-penalty-weight", type=float, default=500.0)
    parser.add_argument("--invalid-bounds-penalty-weight", type=float, default=1000.0)
    parser.add_argument("--command-penalty-scale", type=float, default=6000.0)
    parser.add_argument("--travel-penalty-scale-mm", type=float, default=1000.0)
    parser.add_argument("--stroke-penalty-scale", type=float, default=60.0)
    parser.add_argument("--avg-points-bonus-scale", type=float, default=60.0)

    parser.add_argument("--work-width-mm", type=float, default=150.0)
    parser.add_argument("--work-height-mm", type=float, default=270.0)
    parser.add_argument("--margin-mm", type=float, default=5.0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_tuning(args)


if __name__ == "__main__":
    main()
