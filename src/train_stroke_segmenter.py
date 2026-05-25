"""Train the experimental QuickDraw-to-stroke-mask segmenter."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image

from quickdraw_dataset import iter_raw_samples, synthetic_rich_samples, synthetic_samples, write_training_pairs
from stroke_ml_model import build_stroke_unet, require_torch


torch, nn, functional = require_torch()


class StrokePairDataset(torch.utils.data.Dataset):
    def __init__(self, processed_dir: Path, augment: bool = False, seed: int = 7):
        self.processed_dir = processed_dir
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        manifest_path = processed_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Training manifest not found: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.records = self.manifest["records"]
        if not self.records:
            raise ValueError(f"No training records found in {manifest_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = np.load(self.processed_dir / record["image"]).astype(np.float32) / 255.0
        mask = np.load(self.processed_dir / record["mask"]).astype(np.int64)
        if self.augment:
            image, mask = augment_pair(image, mask, self.rng)
        return torch.from_numpy(image[None, :, :]), torch.from_numpy(mask)

    def class_counts(self, num_classes: int = 3) -> np.ndarray:
        counts = np.zeros(num_classes, dtype=np.int64)
        for record in self.records:
            mask = np.load(self.processed_dir / record["mask"]).astype(np.int64)
            counts += np.bincount(mask.reshape(-1), minlength=num_classes)[:num_classes]
        return counts


def ensure_smoke_data(processed_dir: Path, image_size: int) -> None:
    write_training_pairs(synthetic_samples(), processed_dir, image_size=image_size, preview_count=3)


def ensure_synthetic_rich_data(args: argparse.Namespace) -> None:
    write_training_pairs(
        synthetic_rich_samples(count=args.synthetic_count, seed=args.seed),
        Path(args.processed_dir),
        image_size=args.image_size,
        preview_count=args.preview_count,
        line_width=args.line_width,
        node_radius=args.node_radius,
        corner_angle_threshold_deg=args.corner_angle_threshold,
        corner_stride=args.corner_stride,
    )


def ensure_quickdraw_data(args: argparse.Namespace) -> None:
    processed_dir = Path(args.processed_dir)
    if (processed_dir / "manifest.json").exists() and not args.regenerate_data:
        return
    samples = iter_raw_samples(
        raw_data_dir=Path(args.raw_data_dir),
        categories=args.categories,
        max_drawings_per_category=args.max_drawings_per_category,
    )
    manifest = write_training_pairs(
        samples,
        processed_dir,
        image_size=args.image_size,
        preview_count=args.preview_count,
        line_width=args.line_width,
        node_radius=args.node_radius,
        corner_angle_threshold_deg=args.corner_angle_threshold,
        corner_stride=args.corner_stride,
    )
    print(f"Generated {len(manifest['records'])} training pairs in {processed_dir}")


def save_prediction_preview(image_tensor, target_tensor, prediction_tensor, output_path: Path) -> None:
    image = (image_tensor.squeeze().detach().cpu().numpy() * 255.0).astype(np.uint8)
    target = target_tensor.detach().cpu().numpy()
    prediction = prediction_tensor.detach().cpu().numpy()
    width = image.shape[1]
    height = image.shape[0]
    canvas = Image.new("RGB", (width * 3, height), "white")

    def colorize(mask: np.ndarray) -> Image.Image:
        rgb = np.zeros((height, width, 3), dtype=np.uint8) + 255
        rgb[mask == 1] = [20, 20, 20]
        rgb[mask == 2] = [220, 40, 40]
        return Image.fromarray(rgb, mode="RGB")

    canvas.paste(Image.fromarray(image, mode="L").convert("RGB"), (0, 0))
    canvas.paste(colorize(target), (width, 0))
    canvas.paste(colorize(prediction), (width * 2, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def augment_pair(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if rng.random() < 0.5:
        image = np.flip(image, axis=1)
        mask = np.flip(mask, axis=1)
    if rng.random() < 0.5:
        image = np.flip(image, axis=0)
        mask = np.flip(mask, axis=0)
    rotations = int(rng.integers(0, 4))
    if rotations:
        image = np.rot90(image, rotations)
        mask = np.rot90(mask, rotations)
    if rng.random() < 0.65:
        image = np.clip(image + rng.normal(0.0, 0.025, size=image.shape).astype(np.float32), 0.0, 1.0)
    if rng.random() < 0.65:
        scale = float(rng.uniform(0.88, 1.12))
        bias = float(rng.uniform(-0.05, 0.05))
        image = np.clip(image * scale + bias, 0.0, 1.0)
    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


def compute_class_weights(
    counts: np.ndarray,
    max_class_weight: float,
    no_class_weights: bool,
):
    if no_class_weights:
        return None, [1.0 for _ in counts.tolist()]
    safe_counts = np.maximum(counts.astype(np.float64), 1.0)
    weights = safe_counts.sum() / (len(safe_counts) * safe_counts)
    weights = weights / np.mean(weights)
    weights = np.clip(weights, 0.05, max_class_weight)
    return torch.tensor(weights, dtype=torch.float32), [float(value) for value in weights.tolist()]


def soft_dice_loss(logits, masks, num_classes: int = 3, include_background: bool = False):
    probs = torch.softmax(logits, dim=1)
    one_hot = functional.one_hot(masks, num_classes=num_classes).permute(0, 3, 1, 2).float()
    start = 0 if include_background else 1
    probs = probs[:, start:, :, :]
    one_hot = one_hot[:, start:, :, :]
    dims = (0, 2, 3)
    intersection = torch.sum(probs * one_hot, dims)
    denominator = torch.sum(probs + one_hot, dims)
    dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
    return 1.0 - dice.mean()


def focal_loss(logits, masks, gamma: float = 2.0, class_weights=None):
    ce = functional.cross_entropy(logits, masks, weight=class_weights, reduction="none")
    pt = torch.exp(-ce)
    return ((1.0 - pt) ** gamma * ce).mean()


def combined_loss(logits, masks, ce_loss_fn, args: argparse.Namespace, class_weights_tensor):
    loss = ce_loss_fn(logits, masks)
    parts = {"cross_entropy": float(loss.detach().cpu())}
    if args.dice_loss_weight > 0:
        dice = soft_dice_loss(logits, masks, num_classes=3, include_background=args.dice_include_background)
        loss = loss + args.dice_loss_weight * dice
        parts["dice"] = float(dice.detach().cpu())
    if args.focal_loss_weight > 0:
        focal = focal_loss(logits, masks, gamma=args.focal_gamma, class_weights=class_weights_tensor)
        loss = loss + args.focal_loss_weight * focal
        parts["focal"] = float(focal.detach().cpu())
    return loss, parts


def split_indices(length: int, validation_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(length))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_count = int(round(length * validation_fraction))
    if length > 1 and validation_fraction > 0:
        val_count = min(max(val_count, 1), length - 1)
    else:
        val_count = 0
    return indices[val_count:], indices[:val_count]


def evaluate(model, loader, device, ce_loss_fn, args: argparse.Namespace, class_weights_tensor) -> dict:
    model.eval()
    total_loss = 0.0
    total_pixels = 0
    correct_pixels = 0
    confusion = np.zeros((3, 3), dtype=np.int64)
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            loss, _ = combined_loss(logits, masks, ce_loss_fn, args, class_weights_tensor)
            total_loss += float(loss.item()) * images.shape[0]
            predictions = torch.argmax(logits, dim=1)
            correct_pixels += int((predictions == masks).sum().item())
            total_pixels += int(masks.numel())
            pred_np = predictions.detach().cpu().numpy().reshape(-1)
            mask_np = masks.detach().cpu().numpy().reshape(-1)
            for target_class in range(3):
                for pred_class in range(3):
                    confusion[target_class, pred_class] += int(np.count_nonzero((mask_np == target_class) & (pred_np == pred_class)))

    ious = []
    for class_index in range(3):
        tp = confusion[class_index, class_index]
        fp = confusion[:, class_index].sum() - tp
        fn = confusion[class_index, :].sum() - tp
        denom = tp + fp + fn
        ious.append(float(tp / denom) if denom else 0.0)
    return {
        "loss": total_loss / max(len(loader.dataset), 1),
        "pixel_accuracy": correct_pixels / max(total_pixels, 1),
        "iou_background": ious[0],
        "iou_line": ious[1],
        "iou_node_corner": ious[2],
        "mean_iou": float(np.mean(ious)),
        "confusion_matrix": confusion.tolist(),
    }


def save_prediction_previews(model, dataset, indices: list[int], device, output_dir: Path, count: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    for preview_i, dataset_i in enumerate(indices[:count]):
        image, mask = dataset[dataset_i]
        with torch.no_grad():
            logits = model(image[None, :, :, :].to(device))
            prediction = torch.argmax(logits, dim=1)[0]
        save_prediction_preview(image, mask, prediction, output_dir / f"prediction_preview_{preview_i:02d}.png")


def train(args: argparse.Namespace) -> dict:
    processed_dir = Path(args.processed_dir)
    if args.smoke_test:
        ensure_smoke_data(processed_dir, image_size=args.image_size)
    elif args.synthetic_rich:
        ensure_synthetic_rich_data(args)
    else:
        ensure_quickdraw_data(args)

    dataset = StrokePairDataset(processed_dir, augment=False, seed=args.seed)
    train_indices, val_indices = split_indices(len(dataset), args.validation_fraction, args.seed)
    if not train_indices:
        train_indices = list(range(len(dataset)))
    train_dataset = torch.utils.data.Subset(StrokePairDataset(processed_dir, augment=args.augment, seed=args.seed), train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices) if val_indices else None
    loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = (
        torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        if val_dataset is not None
        else None
    )
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = build_stroke_unet(num_classes=3, base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    class_counts = dataset.class_counts(num_classes=3)
    class_weights_tensor, class_weights = compute_class_weights(
        class_counts,
        max_class_weight=args.max_class_weight,
        no_class_weights=args.no_class_weights,
    )
    if class_weights_tensor is not None:
        class_weights_tensor = class_weights_tensor.to(device)
    print(f"device: {device}")
    print(f"records: total={len(dataset)} train={len(train_dataset)} validation={len(val_dataset) if val_dataset is not None else 0}")
    print(f"batches per epoch: {len(loader)}")
    print(f"class counts: {class_counts.tolist()}")
    print(f"class weights: {class_weights}")
    loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)

    history = []
    best_score = -1.0
    best_epoch = None
    best_path = Path(args.best_model_out) if args.best_model_out else Path(args.model_out).with_name(Path(args.model_out).stem + "_best.pt")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        part_totals: dict[str, float] = {}
        epoch_started_at = time.time()
        last_progress_at = epoch_started_at
        for batch_index, (images, masks) in enumerate(loader, start=1):
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss, parts = combined_loss(logits, masks, loss_fn, args, class_weights_tensor)
            loss.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()
            total_loss += float(loss.item()) * images.shape[0]
            for key, value in parts.items():
                part_totals[key] = part_totals.get(key, 0.0) + value * images.shape[0]
            if args.progress_every > 0 and (batch_index == 1 or batch_index == len(loader) or batch_index % args.progress_every == 0):
                now = time.time()
                elapsed_s = now - epoch_started_at
                batches_per_s = batch_index / max(elapsed_s, 1e-9)
                remaining_batches = len(loader) - batch_index
                eta_s = remaining_batches / max(batches_per_s, 1e-9)
                running_loss = total_loss / max(batch_index * args.batch_size, 1)
                if now - last_progress_at >= args.progress_min_interval_s or batch_index in {1, len(loader)}:
                    print(
                        f"epoch {epoch + 1}/{args.epochs} "
                        f"batch {batch_index}/{len(loader)} "
                        f"running_loss={running_loss:.4f} "
                        f"elapsed={elapsed_s:.1f}s eta={eta_s:.1f}s",
                        flush=True,
                    )
                    last_progress_at = now
        epoch_loss = total_loss / len(train_dataset)
        entry = {
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "loss_parts": {key: value / len(train_dataset) for key, value in part_totals.items()},
        }
        if val_loader is not None:
            entry["validation"] = evaluate(model, val_loader, device, loss_fn, args, class_weights_tensor)
            score = entry["validation"]["mean_iou"]
            if score > best_score:
                best_score = score
                best_epoch = epoch + 1
                best_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model_config": {"num_classes": 3, "base_channels": args.base_channels},
                        "class_map": {0: "background", 1: "line", 2: "node_corner"},
                        "class_counts": class_counts.tolist(),
                        "class_weights": class_weights,
                        "history": history + [entry],
                        "training_args": vars(args),
                        "best_epoch": best_epoch,
                    },
                    best_path,
                )
        history.append(entry)
        if "validation" in entry:
            print(
                f"epoch {epoch + 1}/{args.epochs} "
                f"train_loss={epoch_loss:.4f} val_loss={entry['validation']['loss']:.4f} "
                f"val_miou={entry['validation']['mean_iou']:.4f}"
            )
        else:
            print(f"epoch {epoch + 1}/{args.epochs} train_loss={epoch_loss:.4f}")

    model.eval()
    preview_indices = val_indices if val_indices else train_indices
    save_prediction_previews(model, dataset, preview_indices, device, Path(args.debug_dir), args.preview_count)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": {"num_classes": 3, "base_channels": args.base_channels},
        "class_map": {0: "background", 1: "line", 2: "node_corner"},
        "class_counts": class_counts.tolist(),
        "class_weights": class_weights,
        "training_args": vars(args),
        "train_records": len(train_dataset),
        "validation_records": len(val_dataset) if val_dataset is not None else 0,
        "best_checkpoint": str(best_path) if best_epoch is not None else None,
        "best_epoch": best_epoch,
        "best_validation_mean_iou": best_score if best_epoch is not None else None,
        "history": history,
    }
    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, model_out)
    summary = {
        "checkpoint": str(model_out),
        "records": len(dataset),
        "train_records": len(train_dataset),
        "validation_records": len(val_dataset) if val_dataset is not None else 0,
        "class_counts": class_counts.tolist(),
        "class_weights": class_weights,
        "best_checkpoint": str(best_path) if best_epoch is not None else None,
        "best_epoch": best_epoch,
        "best_validation_mean_iou": best_score if best_epoch is not None else None,
        "training_args": vars(args),
        "history": history,
    }
    (Path(args.debug_dir) / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved checkpoint: {model_out}")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the stroke segmentation U-Net.")
    parser.add_argument("--raw-data-dir", default="data/quickdraw/raw")
    parser.add_argument("--processed-dir", default="data/quickdraw/processed")
    parser.add_argument("--model-out", default="models/stroke_unet_smoke.pt")
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--max-drawings-per-category", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--debug-dir", default="output/stroke_ml_debug")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--synthetic-rich", action="store_true")
    parser.add_argument("--synthetic-count", type=int, default=240)
    parser.add_argument("--regenerate-data", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--line-width", type=int, default=3)
    parser.add_argument("--node-radius", type=int, default=4)
    parser.add_argument("--corner-angle-threshold", type=float, default=135.0)
    parser.add_argument("--corner-stride", type=int, default=2)
    parser.add_argument("--preview-count", type=int, default=8)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=10, help="Print training progress every N batches. Use 0 to disable.")
    parser.add_argument("--progress-min-interval-s", type=float, default=5.0, help="Minimum seconds between repeated progress prints.")
    parser.add_argument("--no-class-weights", action="store_true", help="Disable foreground-aware class weighting.")
    parser.add_argument("--max-class-weight", type=float, default=20.0, help="Clamp inverse-frequency class weights.")
    parser.add_argument("--dice-loss-weight", type=float, default=0.5)
    parser.add_argument("--dice-include-background", action="store_true")
    parser.add_argument("--focal-loss-weight", type=float, default=0.0)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--best-model-out", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
