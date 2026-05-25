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
output/stroke_ml_debug/prediction_preview_00.png
output/stroke_ml_debug/training_summary.json
```

## Stronger Local Training

For a stronger no-download checkpoint, generate a richer procedural dataset and train with validation, augmentation, class weighting, and Dice loss:

```bash
python src/train_stroke_segmenter.py \
  --synthetic-rich \
  --synthetic-count 600 \
  --processed-dir data/quickdraw/processed_rich \
  --model-out models/stroke_unet_synthetic_rich.pt \
  --best-model-out models/stroke_unet_synthetic_rich_best.pt \
  --debug-dir output/stroke_ml_debug/rich_train \
  --epochs 60 \
  --batch-size 8 \
  --image-size 128 \
  --base-channels 24 \
  --augment \
  --validation-fraction 0.15 \
  --node-radius 4 \
  --line-width 3 \
  --dice-loss-weight 0.5 \
  --learning-rate 0.001 \
  --cpu
```

This is still synthetic data, but it is much broader than the 3-sample smoke set. It creates rectangles, triangles, loops, intersections, zigzags, and simple cat-like curves so the model sees more line/corner cases before any QuickDraw data is downloaded.

The trainer now writes:

```text
models/stroke_unet_synthetic_rich.pt
models/stroke_unet_synthetic_rich_best.pt
output/stroke_ml_debug/rich_train/training_summary.json
output/stroke_ml_debug/rich_train/prediction_preview_*.png
```

Use the best checkpoint for inference first:

```bash
python src/stroke_based_pipeline.py \
  --image input_images/square.png \
  --output-dir output/stroke_based_ml_rich_square \
  --segmentation-mode ml \
  --model-path models/stroke_unet_synthetic_rich_best.pt
```

Or with the ML-only graph experiment:

```bash
python src/stroke_ml_graph_pipeline.py \
  --image input_images/cats.jpg \
  --model-path models/stroke_unet_synthetic_rich_best.pt \
  --output-dir output/stroke_ml_graph_rich_cats \
  --node-threshold 0.35 \
  --line-threshold 0.35 \
  --edge-score-threshold 0.18 \
  --support-fraction-threshold 0.05 \
  --edge-search-mode path \
  --max-degree 3
```

The ML-only graph path does not use skeletonisation. It now builds extra
candidate edges from connected line-probability components, merges nearby
duplicate corner vertices, rejects paths that pass through other graph
vertices, and greedily keeps edges that explain new line-mask pixels. This is
intended to handle both long square sides and smooth cat curves while avoiding
duplicate/chord edges that reuse the same line probability.

Useful reconstruction tuning flags:

```bash
--merge-vertex-distance-px 16 \
--component-vertex-radius-px 18 \
--component-candidate-neighbors 10 \
--component-max-edge-distance-px 1200 \
--vertex-passthrough-radius-px 18 \
--coverage-radius-px 2 \
--min-edge-new-pixels 12 \
--min-edge-new-coverage-fraction 0.30
```

## Tune ML Graph Parameters

Manual tuning is slow because the best reconstruction settings depend on the
image, model checkpoint, and how dense the node/corner channel is. The tuning
script runs ML inference once, sweeps graph reconstruction parameters, renders
each stroke result back into image space, compares it with a target line mask,
and ranks the candidates.

Example:

```bash
python src/tune_ml_graph_parameters.py \
  --image input_images/cats.jpg \
  --model-path models/stroke_unet_quickdraw_pilot_best.pt \
  --output-dir output/ml_graph_tuning_cats \
  --target-source grayscale \
  --max-runs 80 \
  --top-k 5
```

For a tiny smoke test:

```bash
python src/tune_ml_graph_parameters.py \
  --image input_images/square.png \
  --model-path models/stroke_unet_quickdraw_pilot_best.pt \
  --output-dir output/ml_graph_tuning_square_smoke \
  --target-source grayscale \
  --max-runs 3 \
  --top-k 2
```

Grid values are comma-separated:

```bash
--node-thresholds 0.30,0.35,0.40 \
--line-thresholds 0.30,0.35,0.40 \
--edge-score-thresholds 0.14,0.18,0.22 \
--vertex-passthrough-radii 8,10,14 \
--line-anchor-distances 16,20,28 \
--max-line-anchors 4,8 \
--component-vertex-radii 14,18,24 \
--min-new-coverage-fractions 0.20,0.30,0.40 \
--max-degrees 2,3
```

Outputs include:

- `tuning_results.json` and `tuning_results.csv`: ranked parameter sets and scores.
- `best_params.json`: best settings, score terms, and summary metrics.
- `best_arduino_commands.txt`: Arduino serial commands for the best run.
- `best_gantry_path_preview.png`: best gantry-space preview.
- `best_stroke_sequence_debug.png`: best image-space stroke sequence.
- `best_mask_comparison.png` and `top_###_preview.png`: blue target mask with red rendered strokes.
- `target_mask.png`: the comparison target used by the scorer.

The score rewards target line coverage, low false-positive drawing, low
Chamfer-like line distance, graph line coverage, and valid bounds. It penalizes
excessive command count, pen-up travel, stroke count, suspicious long/low-support
edges, and invalid bounds. A top-ranked result is an empirical reconstruction
setting for the current model and image; it is not proof that the ML model is
perfect.

## Train From QuickDraw Data

After placing selected `.ndjson` files under `data/quickdraw/raw/`, train with:

```bash
python src/train_stroke_segmenter.py \
  --raw-data-dir data/quickdraw/raw \
  --processed-dir data/quickdraw/processed \
  --categories cat flower bicycle \
  --max-drawings-per-category 100 \
  --regenerate-data \
  --epochs 30 \
  --batch-size 8 \
  --image-size 128 \
  --base-channels 24 \
  --augment \
  --node-radius 4 \
  --dice-loss-weight 0.5 \
  --model-out models/stroke_unet_quickdraw.pt \
  --best-model-out models/stroke_unet_quickdraw_best.pt
```

If `data/quickdraw/processed/manifest.json` is missing, the training script generates training pairs first.

## QuickDraw Pilot Training

Before committing to a long CPU Codespace run, use a smaller pilot. This should finish much sooner and is good for checking whether the model starts producing better masks:

```bash
python src/train_stroke_segmenter.py \
  --raw-data-dir data/quickdraw/raw \
  --processed-dir data/quickdraw/processed_quickdraw_pilot \
  --categories cat dog rabbit bird horse \
  --max-drawings-per-category 100 \
  --regenerate-data \
  --model-out models/stroke_unet_quickdraw_pilot.pt \
  --best-model-out models/stroke_unet_quickdraw_pilot_best.pt \
  --debug-dir output/stroke_ml_debug/quickdraw_pilot \
  --epochs 8 \
  --batch-size 8 \
  --image-size 96 \
  --base-channels 16 \
  --augment \
  --validation-fraction 0.15 \
  --node-radius 3 \
  --line-width 3 \
  --dice-loss-weight 0.5 \
  --learning-rate 0.001 \
  --progress-every 5 \
  --cpu
```

Then test the pilot checkpoint:

```bash
python src/stroke_based_pipeline.py \
  --image input_images/cats.jpg \
  --output-dir output/stroke_based_ml_pilot_cats \
  --segmentation-mode ml \
  --model-path models/stroke_unet_quickdraw_pilot_best.pt
```

If the pilot output is meaningfully better than the smoke checkpoint, scale up categories, samples, image size, base channels, and epochs.

The trainer prints in-epoch progress:

```text
epoch 1/8 batch 5/54 running_loss=... elapsed=... eta=...
```

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
