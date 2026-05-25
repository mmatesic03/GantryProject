"""QuickDraw vector reader and label renderer for stroke segmentation."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw


@dataclass
class QuickDrawSample:
    key_id: str
    category: str
    recognized: bool
    strokes: list[np.ndarray]


def read_quickdraw_ndjson(path: Path, max_drawings: int | None = None, recognized_only: bool = True) -> Iterable[QuickDrawSample]:
    category = path.stem
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if max_drawings is not None and count >= max_drawings:
                break
            if not line.strip():
                continue
            data = json.loads(line)
            if recognized_only and not data.get("recognized", False):
                continue
            strokes = []
            for raw_stroke in data.get("drawing", []):
                if len(raw_stroke) != 2:
                    continue
                xs, ys = raw_stroke
                if len(xs) != len(ys) or len(xs) < 2:
                    continue
                strokes.append(np.column_stack([xs, ys]).astype(np.float32))
            if not strokes:
                continue
            count += 1
            yield QuickDrawSample(
                key_id=str(data.get("key_id", f"{category}_{count}")),
                category=str(data.get("word", category)),
                recognized=bool(data.get("recognized", False)),
                strokes=strokes,
            )


def iter_raw_samples(raw_data_dir: Path, categories: list[str] | None, max_drawings_per_category: int | None) -> Iterable[QuickDrawSample]:
    if categories:
        paths = [raw_data_dir / f"{category}.ndjson" for category in categories]
    else:
        paths = sorted(raw_data_dir.glob("*.ndjson"))
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"QuickDraw file not found: {path}")
        yield from read_quickdraw_ndjson(path, max_drawings=max_drawings_per_category)


def normalize_strokes(strokes: list[np.ndarray], image_size: int, margin: int = 8) -> list[np.ndarray]:
    points = np.vstack(strokes)
    min_xy = points.min(axis=0)
    max_xy = points.max(axis=0)
    size = np.maximum(max_xy - min_xy, 1.0)
    scale = (image_size - 2 * margin) / float(np.max(size))
    normalized = []
    used = size * scale
    offset = np.array([(image_size - used[0]) / 2.0, (image_size - used[1]) / 2.0], dtype=np.float32)
    for stroke in strokes:
        normalized.append((stroke - min_xy) * scale + offset)
    return normalized


def draw_line_mask(strokes: list[np.ndarray], image_size: int, line_width: int) -> np.ndarray:
    image = Image.new("L", (image_size, image_size), 0)
    draw = ImageDraw.Draw(image)
    for stroke in strokes:
        if len(stroke) < 2:
            continue
        draw.line([tuple(point) for point in stroke], fill=1, width=line_width, joint="curve")
    return np.asarray(image, dtype=np.uint8).astype(bool)


def draw_node_disk(mask: np.ndarray, point: np.ndarray, radius: int) -> None:
    y_grid, x_grid = np.ogrid[: mask.shape[0], : mask.shape[1]]
    x, y = float(point[0]), float(point[1])
    disk = (x_grid - x) ** 2 + (y_grid - y) ** 2 <= radius**2
    mask[disk] = True


def angle_degrees(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float | None:
    v1 = a - b
    v2 = c - b
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cosang = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return math.degrees(math.acos(cosang))


def detect_vector_corners(stroke: np.ndarray, angle_threshold_deg: float = 135.0, stride: int = 2) -> list[np.ndarray]:
    corners = []
    if len(stroke) < (2 * stride + 1):
        return corners
    for i in range(stride, len(stroke) - stride):
        angle = angle_degrees(stroke[i - stride], stroke[i], stroke[i + stride])
        if angle is not None and angle <= angle_threshold_deg:
            corners.append(stroke[i])
    return corners


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


def zhang_suen_thinning(mask: np.ndarray, max_iterations: int = 1000) -> np.ndarray:
    skeleton = mask.astype(bool).copy()
    if not np.any(skeleton):
        return skeleton
    for _ in range(max_iterations):
        changed = False
        neighbors = shifted_neighbors(skeleton)
        b = sum(n.astype(np.uint8) for n in neighbors)
        a = transition_count(neighbors)
        p2, _, p4, _, p6, _, p8, _ = neighbors
        remove = skeleton & (b >= 2) & (b <= 6) & (a == 1) & ~(p2 & p4 & p6) & ~(p4 & p6 & p8)
        if np.any(remove):
            skeleton[remove] = False
            changed = True
        neighbors = shifted_neighbors(skeleton)
        b = sum(n.astype(np.uint8) for n in neighbors)
        a = transition_count(neighbors)
        p2, _, p4, _, p6, _, p8, _ = neighbors
        remove = skeleton & (b >= 2) & (b <= 6) & (a == 1) & ~(p2 & p4 & p8) & ~(p2 & p6 & p8)
        if np.any(remove):
            skeleton[remove] = False
            changed = True
        if not changed:
            break
    return skeleton


def shifted_neighbors(mask: np.ndarray) -> list[np.ndarray]:
    padded = np.pad(mask.astype(bool), 1, mode="constant", constant_values=False)
    height, width = mask.shape
    return [
        padded[0:height, 1 : width + 1],
        padded[0:height, 2 : width + 2],
        padded[1 : height + 1, 2 : width + 2],
        padded[2 : height + 2, 2 : width + 2],
        padded[2 : height + 2, 1 : width + 1],
        padded[2 : height + 2, 0:width],
        padded[1 : height + 1, 0:width],
        padded[0:height, 0:width],
    ]


def transition_count(neighbors: list[np.ndarray]) -> np.ndarray:
    total = np.zeros_like(neighbors[0], dtype=np.uint8)
    for current, nxt in zip(neighbors, neighbors[1:] + neighbors[:1]):
        total += (~current & nxt).astype(np.uint8)
    return total


def render_training_pair(
    sample: QuickDrawSample,
    image_size: int = 128,
    line_width: int = 3,
    node_radius: int = 3,
    corner_angle_threshold_deg: float = 135.0,
    corner_stride: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    strokes = normalize_strokes(sample.strokes, image_size=image_size)
    line_mask = draw_line_mask(strokes, image_size=image_size, line_width=line_width)
    node_mask = np.zeros((image_size, image_size), dtype=bool)

    for stroke in strokes:
        draw_node_disk(node_mask, stroke[0], node_radius)
        draw_node_disk(node_mask, stroke[-1], node_radius)
        for corner in detect_vector_corners(stroke, angle_threshold_deg=corner_angle_threshold_deg, stride=corner_stride):
            draw_node_disk(node_mask, corner, node_radius)

    skeleton = zhang_suen_thinning(line_mask)
    counts = neighbor_count(skeleton)
    structural_nodes = skeleton & ((counts == 1) | (counts >= 3))
    ys, xs = np.nonzero(structural_nodes)
    for y, x in zip(ys.tolist(), xs.tolist()):
        draw_node_disk(node_mask, np.array([x, y], dtype=np.float32), node_radius)

    image = np.where(line_mask, 0, 255).astype(np.uint8)
    mask = np.zeros((image_size, image_size), dtype=np.uint8)
    mask[line_mask] = 1
    mask[node_mask] = 2
    return image, mask


def save_preview(image: np.ndarray, mask: np.ndarray, output_path: Path) -> None:
    rgb = np.dstack([image, image, image]).astype(np.uint8)
    rgb[mask == 1] = [20, 20, 20]
    rgb[mask == 2] = [220, 40, 40]
    Image.fromarray(rgb, mode="RGB").save(output_path)


def synthetic_samples() -> list[QuickDrawSample]:
    return [
        QuickDrawSample("synthetic_square", "synthetic", True, [
            np.array([[20, 20], [236, 20], [236, 236], [20, 236], [20, 20]], dtype=np.float32)
        ]),
        QuickDrawSample("synthetic_cross", "synthetic", True, [
            np.array([[30, 128], [226, 128]], dtype=np.float32),
            np.array([[128, 30], [128, 226]], dtype=np.float32),
        ]),
        QuickDrawSample("synthetic_angle", "synthetic", True, [
            np.array([[30, 210], [120, 50], [226, 210]], dtype=np.float32)
        ]),
    ]


def arc_points(
    center: tuple[float, float],
    radius: float,
    start_deg: float,
    end_deg: float,
    count: int,
) -> np.ndarray:
    angles = np.linspace(math.radians(start_deg), math.radians(end_deg), count)
    cx, cy = center
    return np.column_stack([cx + radius * np.cos(angles), cy + radius * np.sin(angles)]).astype(np.float32)


def rectangle_strokes(rng: np.random.Generator) -> list[np.ndarray]:
    x0 = float(rng.uniform(20, 75))
    y0 = float(rng.uniform(20, 75))
    w = float(rng.uniform(95, 185))
    h = float(rng.uniform(95, 185))
    jitter = rng.normal(0, 2.0, size=(5, 2)).astype(np.float32)
    return [np.array([[x0, y0], [x0 + w, y0], [x0 + w, y0 + h], [x0, y0 + h], [x0, y0]], dtype=np.float32) + jitter]


def triangle_strokes(rng: np.random.Generator) -> list[np.ndarray]:
    points = np.array(
        [
            [rng.uniform(40, 90), rng.uniform(175, 225)],
            [rng.uniform(105, 150), rng.uniform(25, 75)],
            [rng.uniform(175, 225), rng.uniform(175, 225)],
        ],
        dtype=np.float32,
    )
    return [np.vstack([points, points[0]])]


def zigzag_strokes(rng: np.random.Generator) -> list[np.ndarray]:
    count = int(rng.integers(4, 8))
    xs = np.linspace(rng.uniform(25, 45), rng.uniform(205, 235), count)
    ys = rng.uniform(45, 215, size=count)
    return [np.column_stack([xs, ys]).astype(np.float32)]


def loop_strokes(rng: np.random.Generator) -> list[np.ndarray]:
    cx = float(rng.uniform(95, 160))
    cy = float(rng.uniform(90, 165))
    rx = float(rng.uniform(45, 80))
    ry = float(rng.uniform(35, 75))
    angles = np.linspace(0, 2 * math.pi, 48)
    points = np.column_stack([cx + rx * np.cos(angles), cy + ry * np.sin(angles)]).astype(np.float32)
    points += rng.normal(0, 1.2, size=points.shape).astype(np.float32)
    return [points]


def intersecting_strokes(rng: np.random.Generator) -> list[np.ndarray]:
    cx = float(rng.uniform(95, 160))
    cy = float(rng.uniform(95, 160))
    span = float(rng.uniform(70, 120))
    return [
        np.array([[cx - span, cy], [cx + span, cy]], dtype=np.float32),
        np.array([[cx, cy - span], [cx, cy + span]], dtype=np.float32),
        np.array([[cx - span * 0.75, cy - span * 0.75], [cx + span * 0.75, cy + span * 0.75]], dtype=np.float32),
    ]


def cat_like_strokes(rng: np.random.Generator) -> list[np.ndarray]:
    x_shift = float(rng.uniform(-15, 15))
    y_shift = float(rng.uniform(-10, 10))
    left_body = arc_points((94 + x_shift, 148 + y_shift), 43, 110, 430, 34)
    right_body = arc_points((158 + x_shift, 148 + y_shift), 45, 105, 430, 34)
    left_ear = np.array([[75, 77], [88, 37], [104, 76]], dtype=np.float32) + [x_shift, y_shift]
    right_ear = np.array([[142, 75], [156, 38], [175, 75]], dtype=np.float32) + [x_shift, y_shift]
    ground = np.array([[20, 185], [232, 185]], dtype=np.float32) + [0, y_shift]
    tail = arc_points((76 + x_shift, 196 + y_shift), 26, 270, 80, 24)
    return [left_ear, left_body, right_ear, right_body, ground, tail]


def synthetic_rich_samples(count: int = 240, seed: int = 7) -> list[QuickDrawSample]:
    rng = np.random.default_rng(seed)
    generators = [rectangle_strokes, triangle_strokes, zigzag_strokes, loop_strokes, intersecting_strokes, cat_like_strokes]
    samples = synthetic_samples()
    for index in range(count):
        generator = generators[index % len(generators)]
        strokes = generator(rng)
        scale = float(rng.uniform(0.85, 1.08))
        offset = np.array([rng.uniform(-8, 8), rng.uniform(-8, 8)], dtype=np.float32)
        transformed = [np.clip((stroke - 128.0) * scale + 128.0 + offset, 0, 255).astype(np.float32) for stroke in strokes]
        samples.append(QuickDrawSample(f"synthetic_rich_{index:05d}", "synthetic_rich", True, transformed))
    return samples


def write_training_pairs(
    samples: Iterable[QuickDrawSample],
    processed_dir: Path,
    image_size: int,
    preview_count: int = 8,
    line_width: int = 3,
    node_radius: int = 3,
    corner_angle_threshold_deg: float = 135.0,
    corner_stride: int = 2,
) -> dict:
    image_dir = processed_dir / "images"
    mask_dir = processed_dir / "masks"
    preview_dir = processed_dir / "previews"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for index, sample in enumerate(samples):
        image, mask = render_training_pair(
            sample,
            image_size=image_size,
            line_width=line_width,
            node_radius=node_radius,
            corner_angle_threshold_deg=corner_angle_threshold_deg,
            corner_stride=corner_stride,
        )
        stem = f"{index:06d}_{sample.category}_{sample.key_id}".replace(" ", "_")
        image_path = image_dir / f"{stem}.npy"
        mask_path = mask_dir / f"{stem}.npy"
        np.save(image_path, image)
        np.save(mask_path, mask)
        if index < preview_count:
            save_preview(image, mask, preview_dir / f"{stem}.png")
        records.append({
            "image": str(image_path.relative_to(processed_dir)),
            "mask": str(mask_path.relative_to(processed_dir)),
            "category": sample.category,
            "key_id": sample.key_id,
        })

    manifest = {
        "schema": "quickdraw_stroke_pairs_v1",
        "image_size": image_size,
        "line_width": line_width,
        "node_radius": node_radius,
        "corner_angle_threshold_deg": corner_angle_threshold_deg,
        "corner_stride": corner_stride,
        "class_map": {"0": "background", "1": "line", "2": "node_corner"},
        "records": records,
    }
    (processed_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate QuickDraw stroke segmentation training pairs.")
    parser.add_argument("--raw-data-dir", default="data/quickdraw/raw")
    parser.add_argument("--processed-dir", default="data/quickdraw/processed")
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--max-drawings-per-category", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--synthetic-rich", action="store_true")
    parser.add_argument("--synthetic-count", type=int, default=240)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--line-width", type=int, default=3)
    parser.add_argument("--node-radius", type=int, default=3)
    parser.add_argument("--corner-angle-threshold", type=float, default=135.0)
    parser.add_argument("--corner-stride", type=int, default=2)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    processed_dir = Path(args.processed_dir)
    if args.synthetic_smoke:
        samples = synthetic_samples()
    elif args.synthetic_rich:
        samples = synthetic_rich_samples(count=args.synthetic_count, seed=args.seed)
    else:
        samples = iter_raw_samples(Path(args.raw_data_dir), args.categories, args.max_drawings_per_category)
    manifest = write_training_pairs(
        samples,
        processed_dir,
        image_size=args.image_size,
        line_width=args.line_width,
        node_radius=args.node_radius,
        corner_angle_threshold_deg=args.corner_angle_threshold,
        corner_stride=args.corner_stride,
    )
    print(f"Wrote {len(manifest['records'])} training pairs to {processed_dir}")


if __name__ == "__main__":
    main()
