"""Generate richer QuickDraw stroke-structure labels.

This is a separate research label path from the existing 3-class
background/line/node-corner dataset. It keeps the raster input simple while
exporting explicit stroke-structure supervision for reconstruction.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw

from quickdraw_dataset import (
    QuickDrawSample,
    arc_points,
    draw_line_mask,
    angle_degrees,
    iter_raw_samples,
    normalize_strokes,
)


LABEL_SCHEMA = {
    "schema": "quickdraw_rich_stroke_structure_v1",
    "label_names": [
        "stroke_support_mask",
        "centreline_mask",
        "endpoint_heatmap",
        "corner_heatmap",
        "junction_heatmap",
        "tangent_cos",
        "tangent_sin",
        "tangent_valid_mask",
        "stroke_id_map",
    ],
    "metadata_fields": ["vector_strokes", "closed_stroke_ids"],
}


def disk_blob(shape: tuple[int, int], point_xy: np.ndarray, radius: float) -> np.ndarray:
    height, width = shape
    x, y = float(point_xy[0]), float(point_xy[1])
    yy, xx = np.ogrid[:height, :width]
    return ((xx - x) ** 2 + (yy - y) ** 2) <= radius**2


def add_gaussian_blob(heatmap: np.ndarray, point_xy: np.ndarray, sigma: float, radius: int | None = None) -> None:
    if radius is None:
        radius = max(1, int(math.ceil(3.0 * sigma)))
    height, width = heatmap.shape
    x, y = float(point_xy[0]), float(point_xy[1])
    x0 = max(0, int(math.floor(x - radius)))
    x1 = min(width, int(math.ceil(x + radius + 1)))
    y0 = max(0, int(math.floor(y - radius)))
    y1 = min(height, int(math.ceil(y + radius + 1)))
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    blob = np.exp(-(((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma**2)))
    heatmap[y0:y1, x0:x1] = np.maximum(heatmap[y0:y1, x0:x1], blob.astype(np.float32))


def draw_centreline_mask(strokes: list[np.ndarray], image_size: int) -> np.ndarray:
    image = Image.new("L", (image_size, image_size), 0)
    draw = ImageDraw.Draw(image)
    for stroke in strokes:
        if len(stroke) >= 2:
            draw.line([tuple(point) for point in stroke], fill=1, width=1)
    return np.asarray(image, dtype=np.uint8).astype(bool)


def is_closed_stroke(stroke: np.ndarray, tolerance_px: float = 2.5) -> bool:
    return len(stroke) >= 3 and float(np.linalg.norm(stroke[0] - stroke[-1])) <= tolerance_px


def unique_closed_points(stroke: np.ndarray) -> np.ndarray:
    if is_closed_stroke(stroke):
        return stroke[:-1]
    return stroke


def add_unique_point(points: list[np.ndarray], point: np.ndarray, min_distance_px: float = 2.0) -> None:
    if not any(float(np.linalg.norm(point - existing)) < min_distance_px for existing in points):
        points.append(point.astype(np.float32))


def detect_structural_corners(
    stroke: np.ndarray,
    angle_threshold_deg: float = 135.0,
    stride: int = 2,
) -> list[np.ndarray]:
    """Detect sharp vector turns, including sparse polygon vertices.

    The older stride-only detector missed triangles/squares because those
    strokes can contain only vertices. This version always evaluates direct
    neighboring vector segments first, then adds a stride-based pass for denser
    hand-drawn strokes.
    """
    if len(stroke) < 3:
        return []

    corners: list[np.ndarray] = []
    closed = is_closed_stroke(stroke)
    points = unique_closed_points(stroke)
    if len(points) < 3:
        return []

    if closed:
        candidate_indices = range(len(points))
    else:
        candidate_indices = range(1, len(points) - 1)

    for index in candidate_indices:
        prev_point = points[(index - 1) % len(points)]
        point = points[index]
        next_point = points[(index + 1) % len(points)]
        angle = angle_degrees(prev_point, point, next_point)
        if angle is not None and angle <= angle_threshold_deg:
            add_unique_point(corners, point)

    if len(points) >= (2 * stride + 1):
        if closed:
            stride_indices = range(len(points))
        else:
            stride_indices = range(stride, len(points) - stride)
        for index in stride_indices:
            prev_point = points[(index - stride) % len(points)]
            point = points[index]
            next_point = points[(index + stride) % len(points)]
            angle = angle_degrees(prev_point, point, next_point)
            if angle is not None and angle <= angle_threshold_deg:
                add_unique_point(corners, point)

    return corners


def dense_stroke_samples(strokes: list[np.ndarray], step_px: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points: list[np.ndarray] = []
    tangents: list[np.ndarray] = []
    stroke_ids: list[int] = []
    for stroke_id, stroke in enumerate(strokes):
        if len(stroke) < 2:
            continue
        for start, end in zip(stroke[:-1], stroke[1:]):
            vector = end - start
            length = float(np.linalg.norm(vector))
            if length < 1e-6:
                continue
            tangent = (vector / length).astype(np.float32)
            count = max(2, int(math.ceil(length / step_px)) + 1)
            ts = np.linspace(0.0, 1.0, count, dtype=np.float32)
            samples = start[None, :] * (1.0 - ts[:, None]) + end[None, :] * ts[:, None]
            points.extend(samples.astype(np.float32))
            tangents.extend([tangent] * len(samples))
            stroke_ids.extend([stroke_id] * len(samples))
    if not points:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )
    return np.vstack(points).astype(np.float32), np.vstack(tangents).astype(np.float32), np.asarray(stroke_ids, dtype=np.int32)


def assign_tangent_fields(
    support_mask: np.ndarray,
    sample_points_xy: np.ndarray,
    sample_tangents_xy: np.ndarray,
    sample_stroke_ids: np.ndarray,
    max_distance_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = support_mask.shape
    cos_field = np.zeros((height, width), dtype=np.float32)
    sin_field = np.zeros((height, width), dtype=np.float32)
    valid = np.zeros((height, width), dtype=bool)
    stroke_id_map = np.full((height, width), -1, dtype=np.int32)
    if len(sample_points_xy) == 0:
        return cos_field, sin_field, valid, stroke_id_map

    ys, xs = np.nonzero(support_mask)
    if len(xs) == 0:
        return cos_field, sin_field, valid, stroke_id_map
    pixels = np.column_stack([xs, ys]).astype(np.float32)
    max_d2 = max_distance_px**2
    for start in range(0, len(pixels), 2048):
        chunk = pixels[start : start + 2048]
        d2 = np.sum((chunk[:, None, :] - sample_points_xy[None, :, :]) ** 2, axis=2)
        nearest = np.argmin(d2, axis=1)
        nearest_d2 = d2[np.arange(len(chunk)), nearest]
        keep = nearest_d2 <= max_d2
        if not np.any(keep):
            continue
        kept_pixels = chunk[keep].astype(np.int32)
        kept_nearest = nearest[keep]
        yy = kept_pixels[:, 1]
        xx = kept_pixels[:, 0]
        tangent = sample_tangents_xy[kept_nearest]
        cos_field[yy, xx] = tangent[:, 0]
        sin_field[yy, xx] = tangent[:, 1]
        valid[yy, xx] = True
        stroke_id_map[yy, xx] = sample_stroke_ids[kept_nearest]
    return cos_field, sin_field, valid, stroke_id_map


def cross_2d(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def segment_intersection(a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray) -> np.ndarray | None:
    r = a1 - a0
    s = b1 - b0
    denom = cross_2d(r, s)
    if abs(denom) < 1e-6:
        return None
    qmp = b0 - a0
    t = cross_2d(qmp, s) / denom
    u = cross_2d(qmp, r) / denom
    eps = 1e-4
    if eps <= t <= 1.0 - eps and eps <= u <= 1.0 - eps:
        return (a0 + t * r).astype(np.float32)
    return None


def detect_intersections(strokes: list[np.ndarray]) -> list[np.ndarray]:
    hits: list[np.ndarray] = []
    segments: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    for stroke_id, stroke in enumerate(strokes):
        for seg_id, (start, end) in enumerate(zip(stroke[:-1], stroke[1:])):
            if np.linalg.norm(end - start) >= 1e-6:
                segments.append((stroke_id, seg_id, start, end))
    for i, (stroke_a, seg_a, a0, a1) in enumerate(segments):
        for stroke_b, seg_b, b0, b1 in segments[i + 1 :]:
            if stroke_a == stroke_b and abs(seg_a - seg_b) <= 1:
                continue
            hit = segment_intersection(a0, a1, b0, b1)
            if hit is None:
                continue
            if not any(float(np.linalg.norm(hit - existing)) < 2.0 for existing in hits):
                hits.append(hit)
    return hits


def render_rich_labels(
    sample: QuickDrawSample,
    image_size: int = 128,
    line_width: int = 3,
    heatmap_sigma: float = 2.0,
    corner_angle_threshold_deg: float = 135.0,
    corner_stride: int = 2,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict]:
    strokes = normalize_strokes(sample.strokes, image_size=image_size)
    support = draw_line_mask(strokes, image_size=image_size, line_width=line_width)
    centreline = draw_centreline_mask(strokes, image_size=image_size)
    image = np.where(support, 0, 255).astype(np.uint8)

    endpoint_heatmap = np.zeros((image_size, image_size), dtype=np.float32)
    corner_heatmap = np.zeros((image_size, image_size), dtype=np.float32)
    junction_heatmap = np.zeros((image_size, image_size), dtype=np.float32)

    endpoint_points: list[np.ndarray] = []
    corner_points: list[np.ndarray] = []
    closed_stroke_ids: list[int] = []
    for stroke_id, stroke in enumerate(strokes):
        if len(stroke) < 2:
            continue
        if is_closed_stroke(stroke):
            closed_stroke_ids.append(stroke_id)
        else:
            endpoint_points.extend([stroke[0], stroke[-1]])
        for corner in detect_structural_corners(stroke, angle_threshold_deg=corner_angle_threshold_deg, stride=corner_stride):
            add_unique_point(corner_points, corner)

    junction_points = detect_intersections(strokes)
    for point in endpoint_points:
        add_gaussian_blob(endpoint_heatmap, point, sigma=heatmap_sigma)
    for point in corner_points:
        add_gaussian_blob(corner_heatmap, point, sigma=heatmap_sigma)
    for point in junction_points:
        add_gaussian_blob(junction_heatmap, point, sigma=heatmap_sigma)

    sample_points, sample_tangents, sample_stroke_ids = dense_stroke_samples(strokes)
    tangent_cos, tangent_sin, tangent_valid, stroke_id_map = assign_tangent_fields(
        support,
        sample_points,
        sample_tangents,
        sample_stroke_ids,
        max_distance_px=max(1.5, line_width * 0.9),
    )
    tangent_valid |= centreline

    labels = {
        "stroke_support_mask": support.astype(np.uint8),
        "centreline_mask": centreline.astype(np.uint8),
        "endpoint_heatmap": endpoint_heatmap,
        "corner_heatmap": corner_heatmap,
        "junction_heatmap": junction_heatmap,
        "tangent_cos": tangent_cos,
        "tangent_sin": tangent_sin,
        "tangent_valid_mask": tangent_valid.astype(np.uint8),
        "stroke_id_map": stroke_id_map,
    }
    meta = {
        "endpoint_count": len(endpoint_points),
        "corner_count": len(corner_points),
        "junction_count": len(junction_points),
        "stroke_count": len(strokes),
        "closed_stroke_count": len(closed_stroke_ids),
        "closed_stroke_ids": closed_stroke_ids,
        "vector_strokes": [[[float(x), float(y)] for x, y in stroke.tolist()] for stroke in strokes],
    }
    return image, labels, meta


def probability_image(values: np.ndarray) -> Image.Image:
    clipped = np.clip(values, 0.0, 1.0)
    return Image.fromarray((clipped * 255.0).astype(np.uint8), mode="L").convert("RGB")


def mask_rgb(mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8) + 255
    rgb[mask.astype(bool)] = color
    return Image.fromarray(rgb, mode="RGB")


def save_rich_preview(image: np.ndarray, labels: dict[str, np.ndarray], output_path: Path) -> None:
    height, width = image.shape
    panels = [
        Image.fromarray(image, mode="L").convert("RGB"),
        mask_rgb(labels["stroke_support_mask"], (20, 20, 20)),
        mask_rgb(labels["centreline_mask"], (30, 120, 230)),
        probability_image(labels["endpoint_heatmap"]),
        probability_image(labels["corner_heatmap"]),
        probability_image(labels["junction_heatmap"]),
    ]
    overlay = Image.fromarray(np.dstack([image, image, image]).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(overlay)
    for name, color in [
        ("endpoint_heatmap", (0, 180, 80)),
        ("corner_heatmap", (230, 145, 0)),
        ("junction_heatmap", (220, 30, 50)),
    ]:
        ys, xs = np.nonzero(labels[name] > 0.35)
        for y, x in zip(ys.tolist()[::4], xs.tolist()[::4]):
            draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=color)
    cos = labels["tangent_cos"]
    sin = labels["tangent_sin"]
    valid = labels["tangent_valid_mask"].astype(bool)
    for y in range(4, height, 12):
        for x in range(4, width, 12):
            if valid[y, x]:
                dx = float(cos[y, x]) * 5.0
                dy = float(sin[y, x]) * 5.0
                draw.line((x - dx, y - dy, x + dx, y + dy), fill=(35, 60, 210), width=1)
    panels.append(overlay)

    canvas = Image.new("RGB", (width * len(panels), height), "white")
    for i, panel in enumerate(panels):
        canvas.paste(panel, (i * width, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def simple_line_sample() -> QuickDrawSample:
    return QuickDrawSample("synthetic_simple_line", "simple_line", True, [np.array([[25, 132], [230, 132]], dtype=np.float32)])


def square_sample() -> QuickDrawSample:
    return QuickDrawSample("synthetic_square", "square", True, [np.array([[45, 45], [210, 45], [210, 210], [45, 210], [45, 45]], dtype=np.float32)])


def triangle_sample() -> QuickDrawSample:
    return QuickDrawSample("synthetic_triangle", "triangle", True, [np.array([[35, 215], [128, 35], [220, 215], [35, 215]], dtype=np.float32)])


def circle_loop_sample() -> QuickDrawSample:
    return QuickDrawSample("synthetic_circle_loop", "circle_loop", True, [arc_points((128, 128), 82, 0, 360, 80)])


def t_junction_sample() -> QuickDrawSample:
    return QuickDrawSample(
        "synthetic_t_junction",
        "t_junction",
        True,
        [
            np.array([[45, 70], [210, 70]], dtype=np.float32),
            np.array([[128, 70], [128, 220]], dtype=np.float32),
        ],
    )


def x_crossing_sample() -> QuickDrawSample:
    return QuickDrawSample(
        "synthetic_x_crossing",
        "x_crossing",
        True,
        [
            np.array([[45, 45], [210, 210]], dtype=np.float32),
            np.array([[210, 45], [45, 210]], dtype=np.float32),
        ],
    )


def smooth_bend_sample() -> QuickDrawSample:
    points = arc_points((130, 135), 88, 205, 15, 48)
    return QuickDrawSample("synthetic_smooth_bend", "smooth_bend", True, [points])


def close_parallel_curves_sample() -> QuickDrawSample:
    base = arc_points((128, 135), 82, 190, 350, 56)
    offset = np.array([0, 14], dtype=np.float32)
    return QuickDrawSample("synthetic_close_parallel", "close_parallel_curves", True, [base, base + offset])


def cat_tail_ground_sample() -> QuickDrawSample:
    ground = np.array([[30, 190], [226, 190]], dtype=np.float32)
    body = arc_points((120, 145), 45, 105, 425, 42)
    tail = arc_points((75, 190), 32, 275, 70, 28)
    return QuickDrawSample("synthetic_cat_tail_ground", "cat_tail_ground", True, [body, tail, ground])


def base_synthetic_rich_samples() -> list[QuickDrawSample]:
    return [
        simple_line_sample(),
        square_sample(),
        triangle_sample(),
        circle_loop_sample(),
        t_junction_sample(),
        x_crossing_sample(),
        smooth_bend_sample(),
        close_parallel_curves_sample(),
        cat_tail_ground_sample(),
    ]


def jittered_sample(sample: QuickDrawSample, rng: np.random.Generator, index: int) -> QuickDrawSample:
    scale = float(rng.uniform(0.86, 1.08))
    rotation = float(rng.uniform(-0.22, 0.22))
    shift = np.array([rng.uniform(-10, 10), rng.uniform(-10, 10)], dtype=np.float32)
    center = np.array([128.0, 128.0], dtype=np.float32)
    rot = np.array(
        [[math.cos(rotation), -math.sin(rotation)], [math.sin(rotation), math.cos(rotation)]],
        dtype=np.float32,
    )
    strokes = []
    for stroke in sample.strokes:
        transformed = (stroke - center) @ rot.T
        transformed = transformed * scale + center + shift
        transformed += rng.normal(0.0, 1.2, size=transformed.shape).astype(np.float32)
        strokes.append(np.clip(transformed, 0, 255).astype(np.float32))
    return QuickDrawSample(f"{sample.key_id}_jitter_{index:05d}", sample.category, True, strokes)


def synthetic_rich_samples(count: int, seed: int = 498) -> list[QuickDrawSample]:
    base = base_synthetic_rich_samples()
    if count <= len(base):
        return base[:count]
    rng = np.random.default_rng(seed)
    samples = list(base)
    for index in range(count - len(base)):
        samples.append(jittered_sample(base[index % len(base)], rng, index))
    return samples


def write_rich_dataset(
    samples: Iterable[QuickDrawSample],
    processed_dir: Path,
    image_size: int,
    line_width: int,
    heatmap_sigma: float,
    corner_angle_threshold_deg: float,
    corner_stride: int,
    preview_count: int,
) -> dict:
    image_dir = processed_dir / "images"
    label_dir = processed_dir / "labels"
    preview_dir = processed_dir / "previews"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    records = []
    totals = {"endpoint_count": 0, "corner_count": 0, "junction_count": 0, "stroke_count": 0}
    for index, sample in enumerate(samples):
        image, labels, meta = render_rich_labels(
            sample,
            image_size=image_size,
            line_width=line_width,
            heatmap_sigma=heatmap_sigma,
            corner_angle_threshold_deg=corner_angle_threshold_deg,
            corner_stride=corner_stride,
        )
        stem = f"{index:06d}_{sample.category}_{sample.key_id}".replace(" ", "_").replace("/", "_")
        image_path = image_dir / f"{stem}.npy"
        np.save(image_path, image)
        label_paths = {}
        for name, array in labels.items():
            path = label_dir / f"{stem}_{name}.npy"
            np.save(path, array)
            label_paths[name] = str(path.relative_to(processed_dir))
        if index < preview_count:
            save_rich_preview(image, labels, preview_dir / f"{stem}_rich_label_preview.png")
        for key in totals:
            totals[key] += int(meta.get(key, 0))
        records.append(
            {
                "image": str(image_path.relative_to(processed_dir)),
                "labels": label_paths,
                "category": sample.category,
                "key_id": sample.key_id,
                "metadata": meta,
            }
        )

    manifest = {
        **LABEL_SCHEMA,
        "image_size": image_size,
        "line_width": line_width,
        "heatmap_sigma": heatmap_sigma,
        "corner_angle_threshold_deg": corner_angle_threshold_deg,
        "corner_stride": corner_stride,
        "record_count": len(records),
        "totals": totals,
        "records": records,
    }
    processed_dir.mkdir(parents=True, exist_ok=True)
    (processed_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate rich QuickDraw stroke-structure labels.")
    parser.add_argument("--raw-data-dir", default="data/quickdraw/raw")
    parser.add_argument("--processed-dir", default="data/quickdraw/rich")
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--max-drawings-per-category", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--line-width", type=int, default=3)
    parser.add_argument("--heatmap-sigma", type=float, default=2.0)
    parser.add_argument("--corner-angle-threshold", type=float, default=135.0)
    parser.add_argument("--corner-stride", type=int, default=2)
    parser.add_argument("--preview-count", type=int, default=12)
    parser.add_argument("--synthetic-rich", action="store_true")
    parser.add_argument("--synthetic-count", type=int, default=240)
    parser.add_argument("--seed", type=int, default=498)
    parser.add_argument("--regenerate-data", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    processed_dir = Path(args.processed_dir)
    if args.synthetic_rich:
        samples = synthetic_rich_samples(args.synthetic_count, seed=args.seed)
    else:
        samples = iter_raw_samples(Path(args.raw_data_dir), args.categories, args.max_drawings_per_category)

    manifest = write_rich_dataset(
        samples=samples,
        processed_dir=processed_dir,
        image_size=args.image_size,
        line_width=args.line_width,
        heatmap_sigma=args.heatmap_sigma,
        corner_angle_threshold_deg=args.corner_angle_threshold,
        corner_stride=args.corner_stride,
        preview_count=args.preview_count,
    )
    print(f"Wrote {manifest['record_count']} rich label records to {processed_dir}")
    print(f"Label schema: {manifest['schema']}")


if __name__ == "__main__":
    main()
