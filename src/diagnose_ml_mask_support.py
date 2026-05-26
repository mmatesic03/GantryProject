"""Diagnose how ML line and node/corner masks support an input drawing.

This is a lightweight diagnostic helper. It runs ML inference once and compares
the original grayscale line mask against:

- the ML line mask
- the ML node/corner mask
- the union of line OR node/corner support

No graph reconstruction is run, no Arduino commands are generated, and no
skeletonisation is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from stroke_ml_graph_pipeline import load_torch_probabilities


def resolve_repo_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (repo_root / path).resolve()


def save_mask(mask: np.ndarray, output_path: Path) -> None:
    Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L").save(output_path)


def fraction(numerator: int, denominator: int) -> float:
    return float(numerator / max(denominator, 1))


def connected_component_summary(mask: np.ndarray) -> dict:
    # Imported lazily to keep this script focused on diagnostics.
    from stroke_based_pipeline import connected_components

    components = connected_components(mask, connectivity=8)
    areas = [len(component) for component in components]
    return {
        "component_count": len(areas),
        "largest_component_pixels": int(max(areas)) if areas else 0,
        "small_component_count_lt_10px": int(sum(1 for area in areas if area < 10)),
        "median_component_pixels": float(np.median(areas)) if areas else 0.0,
    }


def compute_support_metrics(original_mask: np.ndarray, line_mask: np.ndarray, node_mask: np.ndarray) -> dict:
    support_mask = line_mask | node_mask
    original_pixels = int(np.count_nonzero(original_mask))
    line_pixels = int(np.count_nonzero(line_mask))
    node_pixels = int(np.count_nonzero(node_mask))
    support_pixels = int(np.count_nonzero(support_mask))

    original_in_line = original_mask & line_mask
    original_in_node = original_mask & node_mask
    original_in_both = original_mask & line_mask & node_mask
    original_in_node_only = original_mask & node_mask & ~line_mask
    original_in_line_only = original_mask & line_mask & ~node_mask
    original_in_support = original_mask & support_mask
    original_missing_all_ml = original_mask & ~support_mask
    ml_support_not_original = support_mask & ~original_mask

    return {
        "original_pixels": original_pixels,
        "line_mask_pixels": line_pixels,
        "node_mask_pixels": node_pixels,
        "line_or_node_support_pixels": support_pixels,
        "original_supported_by_line_pixels": int(np.count_nonzero(original_in_line)),
        "original_supported_by_node_pixels": int(np.count_nonzero(original_in_node)),
        "original_supported_by_both_pixels": int(np.count_nonzero(original_in_both)),
        "original_supported_by_line_only_pixels": int(np.count_nonzero(original_in_line_only)),
        "original_supported_by_node_only_pixels": int(np.count_nonzero(original_in_node_only)),
        "original_supported_by_line_or_node_pixels": int(np.count_nonzero(original_in_support)),
        "original_missing_from_all_ml_pixels": int(np.count_nonzero(original_missing_all_ml)),
        "ml_support_not_original_pixels": int(np.count_nonzero(ml_support_not_original)),
        "original_supported_by_line_fraction": fraction(int(np.count_nonzero(original_in_line)), original_pixels),
        "original_supported_by_node_fraction": fraction(int(np.count_nonzero(original_in_node)), original_pixels),
        "original_supported_by_node_only_fraction": fraction(int(np.count_nonzero(original_in_node_only)), original_pixels),
        "original_supported_by_line_or_node_fraction": fraction(int(np.count_nonzero(original_in_support)), original_pixels),
        "true_ml_miss_fraction": fraction(int(np.count_nonzero(original_missing_all_ml)), original_pixels),
        "line_mask_component_summary": connected_component_summary(line_mask),
        "node_mask_component_summary": connected_component_summary(node_mask),
        "support_mask_component_summary": connected_component_summary(support_mask),
    }


def save_support_diagnostic(
    original_mask: np.ndarray,
    line_mask: np.ndarray,
    node_mask: np.ndarray,
    output_path: Path,
) -> None:
    support_mask = line_mask | node_mask
    height, width = original_mask.shape
    image = np.full((height, width, 3), 255, dtype=np.uint8)

    original_line_only = original_mask & line_mask & ~node_mask
    original_node_only = original_mask & node_mask & ~line_mask
    original_both = original_mask & line_mask & node_mask
    original_missing = original_mask & ~support_mask
    ml_support_not_original = support_mask & ~original_mask
    node_not_original = node_mask & ~original_mask

    image[ml_support_not_original] = (190, 190, 190)
    image[node_not_original] = (170, 80, 210)
    image[original_line_only] = (20, 170, 80)
    image[original_node_only] = (245, 140, 20)
    image[original_both] = (35, 110, 235)
    image[original_missing] = (230, 45, 45)

    legend_h = 96
    canvas = Image.new("RGB", (width, height + legend_h), "white")
    canvas.paste(Image.fromarray(image, mode="RGB"), (0, 0))
    draw = ImageDraw.Draw(canvas)
    legend_items = [
        ((20, 170, 80), "green: original supported by line mask only"),
        ((245, 140, 20), "orange: original supported by node mask only"),
        ((35, 110, 235), "blue: original supported by both line and node"),
        ((230, 45, 45), "red: original missing from both ML masks"),
        ((170, 80, 210), "purple: node mask outside original"),
        ((190, 190, 190), "gray: ML support outside original"),
    ]
    x = 8
    y = height + 8
    for color, label in legend_items:
        draw.rectangle((x, y + 3, x + 14, y + 17), fill=color)
        draw.text((x + 20, y), label, fill=(20, 20, 20))
        y += 24
        if y > height + legend_h - 18:
            x += 285
            y = height + 8
    canvas.save(output_path)


def save_probability_grid(gray: np.ndarray, line_prob: np.ndarray, node_prob: np.ndarray, output_path: Path) -> None:
    def prob_image(prob: np.ndarray) -> Image.Image:
        return Image.fromarray(np.clip(prob * 255.0, 0, 255).astype(np.uint8), mode="L").convert("RGB")

    gray_image = Image.fromarray(gray, mode="L").convert("RGB")
    line_image = prob_image(line_prob)
    node_image = prob_image(node_prob)
    padding = 16
    label_h = 24
    width = gray_image.width
    height = gray_image.height
    canvas = Image.new("RGB", (3 * width + 4 * padding, height + label_h + 2 * padding), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (image, label) in enumerate([(gray_image, "grayscale"), (line_image, "line probability"), (node_image, "node/corner probability")]):
        x = padding + i * (width + padding)
        canvas.paste(image, (x, padding + label_h))
        draw.text((x, padding), label, fill=(20, 20, 20))
    canvas.save(output_path)


def run(args: argparse.Namespace) -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    image_path = resolve_repo_path(repo_root, args.image)
    model_path = resolve_repo_path(repo_root, args.model_path)
    output_dir = resolve_repo_path(repo_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    probabilities = load_torch_probabilities(image_path, model_path)
    original_mask = probabilities.gray <= args.grayscale_threshold
    line_mask = probabilities.line_prob >= args.line_threshold
    node_mask = probabilities.node_prob >= args.node_threshold
    support_mask = line_mask | node_mask

    save_mask(original_mask, output_dir / "original_grayscale_mask.png")
    save_mask(line_mask, output_dir / "ml_line_mask.png")
    save_mask(node_mask, output_dir / "ml_node_mask.png")
    save_mask(support_mask, output_dir / "ml_line_or_node_support_mask.png")
    save_support_diagnostic(original_mask, line_mask, node_mask, output_dir / "mask_support_diagnostic.png")
    save_probability_grid(probabilities.gray, probabilities.line_prob, probabilities.node_prob, output_dir / "probability_debug.png")

    metrics = {
        "image": str(image_path),
        "model_path": str(model_path),
        "thresholds": {
            "grayscale_threshold": args.grayscale_threshold,
            "line_threshold": args.line_threshold,
            "node_threshold": args.node_threshold,
        },
        "model_diagnostics": probabilities.diagnostics,
        "support_metrics": compute_support_metrics(original_mask, line_mask, node_mask),
        "outputs": {
            "original_grayscale_mask": "original_grayscale_mask.png",
            "ml_line_mask": "ml_line_mask.png",
            "ml_node_mask": "ml_node_mask.png",
            "ml_line_or_node_support_mask": "ml_line_or_node_support_mask.png",
            "mask_support_diagnostic": "mask_support_diagnostic.png",
            "probability_debug": "probability_debug.png",
        },
    }
    (output_dir / "mask_support_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    support = metrics["support_metrics"]
    print(f"Image: {image_path}")
    print(f"Output: {output_dir}")
    print(f"Original supported by line: {support['original_supported_by_line_fraction']:.3f}")
    print(f"Original supported by node only: {support['original_supported_by_node_only_fraction']:.3f}")
    print(f"Original supported by line OR node: {support['original_supported_by_line_or_node_fraction']:.3f}")
    print(f"True ML miss fraction: {support['true_ml_miss_fraction']:.3f}")
    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Diagnose original-vs-line/node ML mask support without graph reconstruction.")
    parser.add_argument("--image", default=str(repo_root / "input_images" / "cats.jpg"))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", default=str(repo_root / "output" / "ml_mask_support_diagnostic"))
    parser.add_argument("--grayscale-threshold", type=float, default=180.0)
    parser.add_argument("--line-threshold", type=float, default=0.35)
    parser.add_argument("--node-threshold", type=float, default=0.35)
    return parser


def main() -> None:
    run(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
