# Stroke ML Pipeline

This project keeps the Arduino serial command interface as the final drawing output:

```text
x_mm,y_mm,mode
```

`mode 0` is travel with the pen up and `mode 1` is drawing with the pen down. The ML path does not produce G-code. It produces segmentation masks that feed the existing stroke graph extraction, gantry coordinate mapping, command export, and bounds validation path.

The gantry coordinate system remains:

- work area: `150 mm x 270 mm`
- origin: top-left
- `+X`: right
- `+Y`: down
- all commands must pass bounds validation before saving

## Relationship To Raghav Et Al.

The intended workflow follows the practical idea from Raghav et al., "Can I teach a robot to replicate a line art":

```text
QuickDraw vector strokes
-> raster line-art image
-> generated background / line / node-corner labels
-> train U-Net-style segmentation model
-> save checkpoint
-> load checkpoint during stroke inference
-> line and node/corner masks
-> graph interpretation
-> stroke extraction
-> gantry coordinate mapping
-> arduino_commands.txt
```

This is a capstone-scale adaptation, not a full research reproduction. The labels are generated automatically from QuickDraw vector strokes using endpoints, sharp corners, and raster skeleton junctions where practical.

## Folders

```text
data/quickdraw/raw/          selected QuickDraw .ndjson files
data/quickdraw/processed/    generated .npy image/mask pairs
models/                      trained local checkpoints
output/stroke_ml_debug/      smoke-training and prediction previews
output/stroke_based/         Arduino commands and stroke debug outputs
```

Large `.ndjson` files, generated training arrays, and model checkpoints are ignored by Git. Keep only the `.gitkeep` placeholders in the repository.

## Optional PyTorch Install

The normal heuristic stroke pipeline needs only `numpy` and `Pillow`. PyTorch is intentionally optional because it is a heavy dependency.

Install it only when training or running a checkpoint:

```bash
python -m pip install -r requirements-ml.txt
```

If PyTorch is not installed, `src/stroke_based_pipeline.py --segmentation-mode auto` will honestly fall back to heuristic mode.

## Download Selected QuickDraw Categories

Do not download the full QuickDraw dataset. Download only selected categories:

```bash
python src/download_quickdraw_categories.py --categories cat flower bicycle
```

This writes selected files to:

```text
data/quickdraw/raw/
```

## Generate Training Pairs

Generate line-art images and 3-class masks:

```bash
python src/generate_quickdraw_training_pairs.py \
  --raw-data-dir data/quickdraw/raw \
  --processed-dir data/quickdraw/processed \
  --categories cat flower bicycle \
  --max-drawings-per-category 100 \
  --image-size 128
```

The class map is:

- `0 = background`
- `1 = line`
- `2 = node/corner`

Preview images are written under:

```text
data/quickdraw/processed/previews/
```

## Smoke Training

Run a tiny synthetic training pass without downloading any data:

```bash
python src/train_stroke_segmenter.py \
  --smoke-test \
  --epochs 1 \
  --batch-size 1 \
  --image-size 64 \
  --model-out models/stroke_unet_smoke.pt
```

This writes:

```text
models/stroke_unet_smoke.pt
output/stroke_ml_debug/prediction_preview.png
output/stroke_ml_debug/training_summary.json
```

## Train From QuickDraw Data

After placing selected `.ndjson` files under `data/quickdraw/raw/`, train with:

```bash
python src/train_stroke_segmenter.py \
  --raw-data-dir data/quickdraw/raw \
  --processed-dir data/quickdraw/processed \
  --categories cat flower bicycle \
  --max-drawings-per-category 100 \
  --epochs 5 \
  --batch-size 8 \
  --image-size 128 \
  --model-out models/stroke_unet_quickdraw.pt
```

If `data/quickdraw/processed/manifest.json` is missing, the training script generates training pairs first.

## Run Stroke Inference With ML

Use a real checkpoint:

```bash
python src/stroke_based_pipeline.py \
  --image input_images/square.png \
  --output-dir output/stroke_based \
  --segmentation-mode ml \
  --model-path models/stroke_unet_smoke.pt
```

Or allow fallback:

```bash
python src/stroke_based_pipeline.py \
  --image input_images/square.png \
  --output-dir output/stroke_based \
  --segmentation-mode auto \
  --model-path models/stroke_unet_smoke.pt
```

The metrics file reports whether ML was actually used:

```text
output/stroke_based/stroke_metrics.json
```

Do not claim ML is working unless this metrics file reports `segmentation_mode_used` as `ML` and includes a non-null `model_path_used`.

## Heuristic Fallback

The heuristic path remains available and should stay end-to-end runnable:

```bash
python src/stroke_based_pipeline.py \
  --image input_images/square.png \
  --output-dir output/stroke_based \
  --segmentation-mode heuristic
```

The fallback path still exports Arduino commands and validates bounds before saving.
