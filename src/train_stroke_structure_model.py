"""Train the richer multi-head stroke-structure model."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from quickdraw_rich_labels import LABEL_SCHEMA, synthetic_rich_samples, write_rich_dataset
from quickdraw_dataset import iter_raw_samples
from stroke_structure_model import build_stroke_structure_unet, require_torch, sigmoid_structure_outputs


torch, nn, functional = require_torch()


MASK_HEADS = ("support", "centreline", "endpoint", "corner", "junction")
LABEL_TO_HEAD = {
    "support": "stroke_support_mask",
    "centreline": "centreline_mask",
    "endpoint": "endpoint_heatmap",
    "corner": "corner_heatmap",
    "junction": "junction_heatmap",
}


class RichStrokeDataset(torch.utils.data.Dataset):
    def __init__(self, processed_dir: Path, augment: bool = False, seed: int = 498):
        self.processed_dir = processed_dir
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        manifest_path = processed_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Rich label manifest not found: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema") != LABEL_SCHEMA["schema"]:
            raise ValueError(f"Unexpected rich label schema in {manifest_path}: {self.manifest.get('schema')}")
        self.records = self.manifest["records"]
        if not self.records:
            raise ValueError(f"No rich label records found in {manifest_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = np.load(self.processed_dir / record["image"]).astype(np.float32) / 255.0
        label_paths = record["labels"]
        labels = {
            "support": np.load(self.processed_dir / label_paths["stroke_support_mask"]).astype(np.float32),
            "centreline": np.load(self.processed_dir / label_paths["centreline_mask"]).astype(np.float32),
            "endpoint": np.load(self.processed_dir / label_paths["endpoint_heatmap"]).astype(np.float32),
            "corner": np.load(self.processed_dir / label_paths["corner_heatmap"]).astype(np.float32),
            "junction": np.load(self.processed_dir / label_paths["junction_heatmap"]).astype(np.float32),
            "tangent_cos": np.load(self.processed_dir / label_paths["tangent_cos"]).astype(np.float32),
            "tangent_sin": np.load(self.processed_dir / label_paths["tangent_sin"]).astype(np.float32),
            "tangent_valid": np.load(self.processed_dir / label_paths["tangent_valid_mask"]).astype(np.float32),
        }
        if self.augment:
            image, labels = augment_sample(image, labels, self.rng)
        target = {
            head: torch.from_numpy(np.ascontiguousarray(labels[head][None, :, :]))
            for head in MASK_HEADS
        }
        tangent = np.stack([labels["tangent_cos"], labels["tangent_sin"]], axis=0)
        target["tangent"] = torch.from_numpy(np.ascontiguousarray(tangent))
        target["tangent_valid"] = torch.from_numpy(np.ascontiguousarray(labels["tangent_valid"][None, :, :]))
        return torch.from_numpy(np.ascontiguousarray(image[None, :, :])), target


def augment_sample(image: np.ndarray, labels: dict[str, np.ndarray], rng: np.random.Generator) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    labels = {key: value.copy() for key, value in labels.items()}
    if rng.random() < 0.5:
        image = np.flip(image, axis=1)
        for key in labels:
            labels[key] = np.flip(labels[key], axis=1)
        labels["tangent_cos"] *= -1.0
    if rng.random() < 0.5:
        image = np.flip(image, axis=0)
        for key in labels:
            labels[key] = np.flip(labels[key], axis=0)
        labels["tangent_sin"] *= -1.0
    rotations = int(rng.integers(0, 4))
    for _ in range(rotations):
        image = np.rot90(image, 1)
        for key in labels:
            labels[key] = np.rot90(labels[key], 1)
        old_cos = labels["tangent_cos"].copy()
        old_sin = labels["tangent_sin"].copy()
        labels["tangent_cos"] = -old_sin
        labels["tangent_sin"] = old_cos
    if rng.random() < 0.65:
        image = np.clip(image + rng.normal(0.0, 0.02, size=image.shape).astype(np.float32), 0.0, 1.0)
    if rng.random() < 0.65:
        image = np.clip(image * float(rng.uniform(0.9, 1.1)) + float(rng.uniform(-0.04, 0.04)), 0.0, 1.0)
    return np.ascontiguousarray(image), {key: np.ascontiguousarray(value) for key, value in labels.items()}


def ensure_rich_data(args: argparse.Namespace) -> None:
    processed_dir = Path(args.processed_dir)
    manifest_path = processed_dir / "manifest.json"
    if manifest_path.exists() and not args.regenerate_data:
        return
    if args.smoke_test or args.synthetic_rich:
        samples = synthetic_rich_samples(args.synthetic_count if args.synthetic_rich else min(args.synthetic_count, 24), seed=args.seed)
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
        line_width_min=args.line_width_min,
        line_width_max=args.line_width_max,
        seed=args.seed,
    )
    print(f"Generated {manifest['record_count']} rich label records in {processed_dir}")


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


def dice_loss_from_logits(logits, target):
    probs = torch.sigmoid(logits)
    dims = (0, 2, 3)
    intersection = torch.sum(probs * target, dims)
    denominator = torch.sum(probs + target, dims)
    dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
    return 1.0 - dice.mean()


def structure_loss(outputs: dict, targets: dict, args: argparse.Namespace) -> tuple:
    total = 0.0
    parts: dict[str, float] = {}
    bce = nn.BCEWithLogitsLoss()
    weights = {
        "support": args.support_loss_weight,
        "centreline": args.centreline_loss_weight,
        "endpoint": args.endpoint_loss_weight,
        "corner": args.corner_loss_weight,
        "junction": args.junction_loss_weight,
    }
    for head in MASK_HEADS:
        target = targets[head]
        bce_value = bce(outputs[head], target)
        dice_value = dice_loss_from_logits(outputs[head], target)
        head_loss = bce_value + args.dice_loss_weight * dice_value
        total = total + weights[head] * head_loss
        parts[f"{head}_bce"] = float(bce_value.detach().cpu())
        parts[f"{head}_dice"] = float(dice_value.detach().cpu())

    valid = targets["tangent_valid"]
    target_tangent = targets["tangent"]
    pred_tangent = outputs["tangent"]
    valid2 = valid.repeat(1, 2, 1, 1)
    if torch.count_nonzero(valid2) > 0:
        mse = torch.sum(((pred_tangent - target_tangent) ** 2) * valid2) / torch.clamp(torch.sum(valid2), min=1.0)
        pred_norm = functional.normalize(pred_tangent, dim=1)
        target_norm = functional.normalize(target_tangent, dim=1)
        cosine = 1.0 - torch.sum((pred_norm * target_norm) * valid2) / torch.clamp(torch.sum(valid), min=1.0)
        tangent_loss = mse + args.tangent_cosine_loss_weight * cosine
    else:
        mse = pred_tangent.sum() * 0.0
        cosine = pred_tangent.sum() * 0.0
        tangent_loss = mse
    total = total + args.tangent_loss_weight * tangent_loss
    parts["tangent_mse"] = float(mse.detach().cpu())
    parts["tangent_cosine"] = float(cosine.detach().cpu())
    return total, parts


def move_targets(targets: dict, device):
    return {key: value.to(device) for key, value in targets.items()}


def evaluate(model, loader, device, args: argparse.Namespace) -> dict:
    model.eval()
    total_loss = 0.0
    stats = {
        head: {"intersection": 0.0, "union": 0.0, "predicted": 0.0, "target": 0.0}
        for head in MASK_HEADS
    }
    tangent_scores: list[float] = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = move_targets(targets, device)
            outputs = model(images)
            loss, _ = structure_loss(outputs, targets, args)
            total_loss += float(loss.item()) * images.shape[0]
            probs = sigmoid_structure_outputs(outputs)
            for head in MASK_HEADS:
                pred_threshold = args.eval_threshold if head in {"support", "centreline"} else args.heatmap_eval_threshold
                target_threshold = 0.5 if head in {"support", "centreline"} else args.heatmap_eval_threshold
                pred_mask = probs[head] >= pred_threshold
                target_mask = targets[head] >= target_threshold
                stats[head]["intersection"] += float(torch.count_nonzero(pred_mask & target_mask).item())
                stats[head]["union"] += float(torch.count_nonzero(pred_mask | target_mask).item())
                stats[head]["predicted"] += float(torch.count_nonzero(pred_mask).item())
                stats[head]["target"] += float(torch.count_nonzero(target_mask).item())
            valid = targets["tangent_valid"] > 0.5
            if torch.count_nonzero(valid) > 0:
                pred = functional.normalize(outputs["tangent"], dim=1)
                target = functional.normalize(targets["tangent"], dim=1)
                score = torch.sum(torch.sum(pred * target, dim=1, keepdim=True) * valid) / torch.clamp(torch.sum(valid), min=1.0)
                tangent_scores.append(float(score.item()))
    metrics = {
        "loss": total_loss / max(len(loader.dataset), 1),
        "tangent_consistency": float(np.mean(tangent_scores)) if tangent_scores else 0.0,
        "eval_threshold": args.eval_threshold,
        "heatmap_eval_threshold": args.heatmap_eval_threshold,
    }
    for head, values in stats.items():
        metrics[f"{head}_iou"] = values["intersection"] / max(values["union"], 1.0)
        metrics[f"{head}_precision"] = values["intersection"] / max(values["predicted"], 1.0)
        metrics[f"{head}_recall"] = values["intersection"] / max(values["target"], 1.0)
    return metrics


def to_u8_probability(array: np.ndarray) -> np.ndarray:
    return (np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)


def save_prediction_preview(image_tensor, targets: dict, outputs: dict, output_path: Path) -> None:
    image = (image_tensor.squeeze().detach().cpu().numpy() * 255.0).astype(np.uint8)
    probs = sigmoid_structure_outputs(outputs)
    panels = [Image.fromarray(image, mode="L").convert("RGB")]
    for head in MASK_HEADS:
        panels.append(Image.fromarray(to_u8_probability(targets[head].squeeze().detach().cpu().numpy()), mode="L").convert("RGB"))
        panels.append(Image.fromarray(to_u8_probability(probs[head].squeeze().detach().cpu().numpy()), mode="L").convert("RGB"))
    overlay = Image.fromarray(np.dstack([image, image, image]).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(overlay)
    tangent = outputs["tangent"].squeeze().detach().cpu().numpy()
    support = probs["support"].squeeze().detach().cpu().numpy() > 0.45
    h, w = support.shape
    for y in range(4, h, 12):
        for x in range(4, w, 12):
            if support[y, x]:
                dx = float(tangent[0, y, x]) * 5.0
                dy = float(tangent[1, y, x]) * 5.0
                draw.line((x - dx, y - dy, x + dx, y + dy), fill=(30, 80, 230), width=1)
    panels.append(overlay)
    width, height = image.shape[1], image.shape[0]
    canvas = Image.new("RGB", (width * len(panels), height), "white")
    for i, panel in enumerate(panels):
        canvas.paste(panel, (i * width, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def save_prediction_previews(model, dataset, indices: list[int], device, output_dir: Path, count: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    for preview_i, dataset_i in enumerate(indices[:count]):
        image, targets = dataset[dataset_i]
        with torch.no_grad():
            outputs = model(image[None, :, :, :].to(device))
        cpu_targets = {key: value for key, value in targets.items()}
        cpu_outputs = {key: value[0:1].cpu() for key, value in outputs.items()}
        save_prediction_preview(image, cpu_targets, cpu_outputs, output_dir / f"structure_prediction_preview_{preview_i:02d}.png")


def checkpoint_payload(model, args: argparse.Namespace, dataset: RichStrokeDataset, history: list, best_epoch, best_score) -> dict:
    return {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "architecture": "StrokeStructureUNet",
            "base_channels": args.base_channels,
            "in_channels": 1,
            "head_names": list(MASK_HEADS) + ["tangent"],
        },
        "label_schema": dataset.manifest.get("schema"),
        "label_names": dataset.manifest.get("label_names"),
        "training_args": vars(args),
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_loss": best_score,
    }


def train(args: argparse.Namespace) -> dict:
    ensure_rich_data(args)
    processed_dir = Path(args.processed_dir)
    base_dataset = RichStrokeDataset(processed_dir, augment=False, seed=args.seed)
    train_indices, val_indices = split_indices(len(base_dataset), args.validation_fraction, args.seed)
    if not train_indices:
        train_indices = list(range(len(base_dataset)))
    train_dataset = torch.utils.data.Subset(RichStrokeDataset(processed_dir, augment=args.augment, seed=args.seed), train_indices)
    val_dataset = torch.utils.data.Subset(base_dataset, val_indices) if val_indices else None
    loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = (
        torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        if val_dataset is not None
        else None
    )
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = build_stroke_structure_unet(base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    print(f"device: {device}")
    print(f"records: total={len(base_dataset)} train={len(train_dataset)} validation={len(val_dataset) if val_dataset else 0}")
    print(f"batches per epoch: {len(loader)}")
    print(f"label schema: {base_dataset.manifest.get('schema')}")

    history = []
    best_epoch = None
    best_loss = float("inf")
    best_path = Path(args.best_model_out) if args.best_model_out else Path(args.model_out).with_name(Path(args.model_out).stem + "_best.pt")
    for epoch in range(args.epochs):
        model.train()
        epoch_started = time.time()
        running = 0.0
        part_totals: dict[str, float] = {}
        for batch_index, (images, targets) in enumerate(loader, start=1):
            images = images.to(device)
            targets = move_targets(targets, device)
            optimizer.zero_grad()
            outputs = model(images)
            loss, parts = structure_loss(outputs, targets, args)
            loss.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()
            running += float(loss.item()) * images.shape[0]
            for key, value in parts.items():
                part_totals[key] = part_totals.get(key, 0.0) + value * images.shape[0]
            if args.progress_every > 0 and (batch_index == 1 or batch_index == len(loader) or batch_index % args.progress_every == 0):
                elapsed = time.time() - epoch_started
                print(
                    f"epoch {epoch + 1}/{args.epochs} batch {batch_index}/{len(loader)} "
                    f"running_loss={running / max(batch_index * args.batch_size, 1):.4f} elapsed={elapsed:.1f}s",
                    flush=True,
                )
        entry = {
            "epoch": epoch + 1,
            "train_loss": running / max(len(train_dataset), 1),
            "loss_parts": {key: value / max(len(train_dataset), 1) for key, value in part_totals.items()},
        }
        if val_loader is not None:
            entry["validation"] = evaluate(model, val_loader, device, args)
            val_loss = entry["validation"]["loss"]
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch + 1
                best_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(checkpoint_payload(model, args, base_dataset, history + [entry], best_epoch, best_loss), best_path)
        history.append(entry)
        if "validation" in entry:
            print(
                f"epoch {epoch + 1}/{args.epochs} train_loss={entry['train_loss']:.4f} "
                f"val_loss={entry['validation']['loss']:.4f} "
                f"support_iou={entry['validation']['support_iou']:.4f} "
                f"centreline_iou={entry['validation']['centreline_iou']:.4f} "
                f"corner_iou={entry['validation']['corner_iou']:.4f} "
                f"tangent={entry['validation']['tangent_consistency']:.4f}"
            )
        else:
            print(f"epoch {epoch + 1}/{args.epochs} train_loss={entry['train_loss']:.4f}")

    if best_epoch is None:
        best_epoch = args.epochs
        best_loss = history[-1]["train_loss"] if history else None
        best_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint_payload(model, args, base_dataset, history, best_epoch, best_loss), best_path)

    preview_indices = val_indices if val_indices else train_indices
    save_prediction_previews(model, base_dataset, preview_indices, device, Path(args.debug_dir), args.preview_count)
    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(model, args, base_dataset, history, best_epoch, best_loss), model_out)
    summary = {
        "checkpoint": str(model_out),
        "best_checkpoint": str(best_path),
        "records": len(base_dataset),
        "train_records": len(train_dataset),
        "validation_records": len(val_dataset) if val_dataset else 0,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "label_schema": base_dataset.manifest.get("schema"),
        "history": history,
        "training_args": vars(args),
    }
    Path(args.debug_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.debug_dir) / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved checkpoint: {model_out}")
    print(f"Saved best checkpoint: {best_path}")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the multi-head stroke-structure model.")
    parser.add_argument("--processed-dir", default="data/quickdraw/rich")
    parser.add_argument("--raw-data-dir", default="data/quickdraw/raw")
    parser.add_argument("--model-out", default="models/stroke_structure_last.pt")
    parser.add_argument("--best-model-out", default=None)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--synthetic-rich", action="store_true")
    parser.add_argument("--synthetic-count", type=int, default=240)
    parser.add_argument("--regenerate-data", action="store_true")
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--max-drawings-per-category", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--line-width", type=int, default=3)
    parser.add_argument("--line-width-min", type=int, default=None)
    parser.add_argument("--line-width-max", type=int, default=None)
    parser.add_argument("--heatmap-sigma", type=float, default=2.0)
    parser.add_argument("--corner-angle-threshold", type=float, default=135.0)
    parser.add_argument("--corner-stride", type=int, default=2)
    parser.add_argument("--seed", type=int, default=498)
    parser.add_argument("--preview-count", type=int, default=4)
    parser.add_argument("--debug-dir", default="output/stroke_structure_debug")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--dice-loss-weight", type=float, default=0.5)
    parser.add_argument("--support-loss-weight", type=float, default=0.5)
    parser.add_argument("--centreline-loss-weight", type=float, default=2.0)
    parser.add_argument("--endpoint-loss-weight", type=float, default=1.5)
    parser.add_argument("--corner-loss-weight", type=float, default=2.0)
    parser.add_argument("--junction-loss-weight", type=float, default=1.5)
    parser.add_argument("--tangent-loss-weight", type=float, default=2.0)
    parser.add_argument("--tangent-cosine-loss-weight", type=float, default=1.0)
    parser.add_argument("--eval-threshold", type=float, default=0.5)
    parser.add_argument("--heatmap-eval-threshold", type=float, default=0.35)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
