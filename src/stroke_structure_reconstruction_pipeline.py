"""ML inference and reconstruction for the richer stroke-structure model.

The script predicts support, centreline, endpoint, corner, junction, and
tangent fields, then decodes them with the same structure decoder used by the
oracle label test. It exports Arduino serial command text only; it never sends
commands to hardware.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from oracle_stroke_structure_reconstruction import (
    StructureLabels,
    claimed_pixels_from_strokes,
    compute_structure_metrics,
    decode_structure,
    save_reconstruction_debug,
    save_structure_overlay,
)
from stroke_based_pipeline import (
    map_paths_to_gantry_mm,
    parse_firmware_constants,
    paths_to_arduino_commands,
    save_arduino_commands,
    save_gantry_preview,
    save_json,
    validate_arduino_commands,
)
from stroke_structure_model import build_stroke_structure_unet, require_torch, sigmoid_structure_outputs


torch, _, _ = require_torch()


def load_grayscale(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L")
    return np.asarray(image, dtype=np.uint8)


def load_model(model_path: Path):
    checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)
    config = checkpoint.get("model_config", {}) if isinstance(checkpoint, dict) else {}
    model = build_stroke_structure_unet(base_channels=int(config.get("base_channels", 16)))
    state = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state)
    model.eval()
    return model, checkpoint if isinstance(checkpoint, dict) else {}


def infer_structure_labels(image_path: Path, model_path: Path, args: argparse.Namespace) -> tuple[StructureLabels, dict[str, np.ndarray], dict]:
    gray = load_grayscale(image_path)
    model, checkpoint = load_model(model_path)
    tensor = torch.from_numpy(gray.astype(np.float32) / 255.0)[None, None, :, :]
    with torch.no_grad():
        outputs = model(tensor)
        probs = sigmoid_structure_outputs(outputs)
    arrays = {key: value[0].detach().cpu().numpy() for key, value in probs.items()}
    support_prob = arrays["support"][0]
    centreline_prob = arrays["centreline"][0]
    endpoint_prob = arrays["endpoint"][0]
    corner_prob = arrays["corner"][0]
    junction_prob = arrays["junction"][0]
    tangent = arrays["tangent"]
    labels = StructureLabels(
        gray=gray,
        support=support_prob >= args.support_threshold,
        centreline=centreline_prob >= args.centreline_threshold,
        endpoint=endpoint_prob.astype(np.float32),
        corner=corner_prob.astype(np.float32),
        junction=junction_prob.astype(np.float32),
        tangent_cos=tangent[0].astype(np.float32),
        tangent_sin=tangent[1].astype(np.float32),
        tangent_valid=(support_prob >= args.tangent_valid_threshold) | (centreline_prob >= args.centreline_threshold),
        stroke_id_map=None,
        vector_strokes=None,
        closed_stroke_ids=set(),
        model_path=str(model_path),
        label_schema=checkpoint.get("label_schema"),
        source_record=None,
    )
    return labels, {
        "support": support_prob,
        "centreline": centreline_prob,
        "endpoint": endpoint_prob,
        "corner": corner_prob,
        "junction": junction_prob,
        "tangent_cos": tangent[0],
        "tangent_sin": tangent[1],
    }, checkpoint


def prob_panel(array: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L").convert("RGB")


def mask_panel(mask: np.ndarray) -> Image.Image:
    return Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L").convert("RGB")


def save_prediction_mask_outputs(
    probabilities: dict[str, np.ndarray],
    labels: StructureLabels,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    """Save the raw ML heads and thresholded masks used by reconstruction."""
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

    height, width = labels.gray.shape
    panels = [
        Image.fromarray(labels.gray, mode="L").convert("RGB"),
        mask_panel(masks["support"]),
        mask_panel(masks["centreline"]),
        mask_panel(masks["endpoint"]),
        mask_panel(masks["corner"]),
        mask_panel(masks["junction"]),
        mask_panel(masks["tangent_valid"]),
    ]
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def save_false_positive_drawn(labels: StructureLabels, strokes: list[np.ndarray], output_path: Path, radius: int) -> None:
    drawn = claimed_pixels_from_strokes(strokes, labels.support.shape, radius=radius)
    false_positive = drawn & ~labels.support
    rgb = np.dstack([labels.gray, labels.gray, labels.gray]).astype(np.uint8)
    rgb[drawn] = [40, 140, 230]
    rgb[false_positive] = [230, 50, 60]
    Image.fromarray(rgb, mode="RGB").save(output_path)


def run_pipeline(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels, probabilities, checkpoint = infer_structure_labels(Path(args.image), Path(args.model_path), args)
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
    metrics = compute_structure_metrics(
        commands,
        result,
        labels,
        transform_info,
        firmware_constants,
        args,
        pipeline_name="ml_stroke_structure_reconstruction",
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
    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run stroke-structure ML inference and gantry reconstruction.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", default="output/stroke_structure")
    parser.add_argument("--support-threshold", type=float, default=0.45)
    parser.add_argument("--centreline-threshold", type=float, default=0.45)
    parser.add_argument("--endpoint-threshold", type=float, default=0.35)
    parser.add_argument("--corner-threshold", type=float, default=0.35)
    parser.add_argument("--junction-threshold", type=float, default=0.35)
    parser.add_argument("--tangent-valid-threshold", type=float, default=0.35)
    parser.add_argument("--node-radius-px", type=int, default=2)
    parser.add_argument("--coverage-radius-px", type=int, default=2)
    parser.add_argument("--endpoint-usage-radius-px", type=float, default=5.0)
    parser.add_argument("--min-points", type=int, default=2)
    parser.add_argument("--simplification-epsilon", type=float, default=1.25)
    parser.add_argument("--stitch-gap-px", type=float, default=6.0)
    parser.add_argument("--stitch-support-fraction", type=float, default=0.5)
    parser.add_argument("--disable-stitching", action="store_true")
    parser.add_argument("--work-width-mm", type=float, default=150.0)
    parser.add_argument("--work-height-mm", type=float, default=270.0)
    parser.add_argument("--margin-mm", type=float, default=5.0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
