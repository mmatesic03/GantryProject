"""Train the experimental QuickDraw-to-stroke-mask segmenter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from quickdraw_dataset import iter_raw_samples, synthetic_samples, write_training_pairs
from stroke_ml_model import build_stroke_unet, require_torch


torch, nn, _ = require_torch()


class StrokePairDataset(torch.utils.data.Dataset):
    def __init__(self, processed_dir: Path):
        self.processed_dir = processed_dir
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
        return torch.from_numpy(image[None, :, :]), torch.from_numpy(mask)


def ensure_smoke_data(processed_dir: Path, image_size: int) -> None:
    write_training_pairs(synthetic_samples(), processed_dir, image_size=image_size, preview_count=3)


def ensure_quickdraw_data(args: argparse.Namespace) -> None:
    processed_dir = Path(args.processed_dir)
    if (processed_dir / "manifest.json").exists():
        return
    samples = iter_raw_samples(
        raw_data_dir=Path(args.raw_data_dir),
        categories=args.categories,
        max_drawings_per_category=args.max_drawings_per_category,
    )
    manifest = write_training_pairs(samples, processed_dir, image_size=args.image_size)
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


def train(args: argparse.Namespace) -> dict:
    processed_dir = Path(args.processed_dir)
    if args.smoke_test:
        ensure_smoke_data(processed_dir, image_size=args.image_size)
    else:
        ensure_quickdraw_data(args)

    dataset = StrokePairDataset(processed_dir)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = build_stroke_unet(num_classes=3, base_channels=args.base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    history = []
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, masks)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * images.shape[0]
        epoch_loss = total_loss / len(dataset)
        history.append({"epoch": epoch + 1, "loss": epoch_loss})
        print(f"epoch {epoch + 1}/{args.epochs} loss={epoch_loss:.4f}")

    model.eval()
    sample_image, sample_mask = dataset[0]
    with torch.no_grad():
        logits = model(sample_image[None, :, :, :].to(device))
        prediction = torch.argmax(logits, dim=1)[0]
    save_prediction_preview(sample_image, sample_mask, prediction, Path(args.debug_dir) / "prediction_preview.png")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": {"num_classes": 3, "base_channels": args.base_channels},
        "class_map": {0: "background", 1: "line", 2: "node_corner"},
        "history": history,
    }
    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, model_out)
    summary = {"checkpoint": str(model_out), "records": len(dataset), "history": history}
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
    parser.add_argument("--debug-dir", default="output/stroke_ml_debug")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
