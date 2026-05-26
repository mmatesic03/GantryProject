# Stroke Structure Pipeline

This is the next research path for the ML stroke renderer. The older 3-class
model predicts only `background`, `line`, and `node_corner`; that is useful for
segmentation, but it leaves the graph decoder guessing which pixels are
endpoints, which turns are corners, which crossings are true junctions, and how
same-stroke continuity should pass through ambiguous contact points.

The richer pipeline follows the same broad framing used as background from
Raghav et al.:

QuickDraw vector strokes -> raster labels -> segmentation masks -> graph
interpretation -> stroke extraction.

The practical improvement here is that the model is trained to predict the
structure the decoder needs, not just a foreground mask. The label generator
uses QuickDraw vector strokes to create:

- grayscale line-art input image
- stroke/support mask
- centreline mask
- endpoint heatmap
- corner heatmap
- junction/intersection heatmap
- tangent direction fields, `tangent_cos` and `tangent_sin`
- tangent valid mask
- stroke id map for inspectable same-stroke metadata
- normalized vector stroke metadata and closed-stroke ids for oracle tests

This gives reconstruction explicit hints for Arduino plotting: where to start,
where to split, where to preserve sharp turns, and which local direction to
follow. The final output remains canonical Arduino serial text:

```text
x_mm,y_mm,mode
```

`mode 0` is travel / pen up and `mode 1` is draw / pen down. Commands are mapped
to the 150 mm x 270 mm gantry area with a top-left origin, +X right, +Y down,
and are bounds-validated before saving.

## Generate Rich Labels

Synthetic smoke data, no downloads:

```bash
python src/quickdraw_rich_labels.py \
  --processed-dir data/quickdraw/rich_smoke \
  --synthetic-rich \
  --synthetic-count 20 \
  --image-size 128 \
  --seed 498 \
  --line-width-min 2 \
  --line-width-max 9
```

The synthetic set includes simple line, square, triangle, circle/loop,
T-junction, X-crossing, smooth bend, close parallel curves, and a simplified
cat-tail/ground-contact case. Sparse polygon vertices are labelled as corners,
closed strokes are not labelled as endpoints, and outputs are `.npy` arrays
plus preview PNGs under `previews/`. Use `--line-width-min` and
`--line-width-max` for synthetic training sets so support thickness varies while
the centreline target remains one-pixel stroke structure.

For real QuickDraw data already placed under `data/quickdraw/raw/`:

```bash
python src/quickdraw_rich_labels.py \
  --raw-data-dir data/quickdraw/raw \
  --processed-dir data/quickdraw/rich \
  --categories cat flower bicycle \
  --max-drawings-per-category 100
```

## Smoke Train

The model is a lightweight shared U-Net encoder/decoder with separate heads:

- support probability
- centreline probability
- endpoint probability
- corner probability
- junction probability
- tangent vector regression

Training reports per-head validation metrics: support IoU, centreline IoU,
endpoint/corner/junction heatmap IoU, precision/recall, and tangent
consistency. This matters because support IoU can look excellent while
centreline, corner, or tangent predictions still fail reconstruction.

Run only a smoke pass to verify the pipeline:

```bash
python src/train_stroke_structure_model.py \
  --processed-dir data/quickdraw/rich_smoke \
  --model-out models/stroke_structure_smoke_last.pt \
  --best-model-out models/stroke_structure_smoke_best.pt \
  --epochs 1 \
  --batch-size 2 \
  --base-channels 16 \
  --smoke-test \
  --progress-every 1
```

The smoke checkpoint proves the scripts and tensor shapes work. It is not a
strong model.

## Oracle Reconstruction

The oracle test decodes perfect generated labels before using ML predictions:

```bash
python src/oracle_stroke_structure_reconstruction.py \
  --processed-dir data/quickdraw/rich_smoke \
  --output-dir output/oracle_structure_smoke \
  --max-samples 3
```

By default, oracle mode uses generated same-stroke continuity metadata. This is
the upper-bound test for the label design: simple closed shapes should stay
continuous instead of being split into pen-up fragments. If this fails, the
label/decoder design is the problem. If it works, training a model to predict
the visible heads is justified.

For a stricter raster-only decoder stress test, disable continuity metadata:

```bash
python src/oracle_stroke_structure_reconstruction.py \
  --processed-dir data/quickdraw/rich_smoke \
  --output-dir output/oracle_structure_smoke_raster_only \
  --max-samples 3 \
  --no-oracle-continuity-metadata
```

Use both results together:

- continuity-aware oracle checks whether generated labels preserve true stroke
  order and closed-loop continuity
- raster-only oracle checks how much continuity the graph decoder can recover
  without hidden metadata
- ML reconstruction is expected to underperform both until the model predicts
  stable centreline, endpoint, corner, junction, and tangent fields

Oracle outputs include `arduino_commands.txt`, `stroke_metrics.json`,
`structure_label_overlay.png`, `reconstruction_debug.png`,
`missed_support_pixels.png`, and `gantry_path_preview.png`.

## ML Reconstruction

After a local checkpoint exists:

```bash
python src/stroke_structure_reconstruction_pipeline.py \
  --image input_images/square.png \
  --model-path models/stroke_structure_smoke_best.pt \
  --output-dir output/stroke_structure_smoke_square
```

The ML path writes:

- `arduino_commands.txt`
- `stroke_metrics.json`
- `structure_probability_debug.png`
- `structure_prediction_arrays.npz`
- `structure_mask_debug.png`
- individual `ml_*_probability.png` and `ml_*_mask.png` files for support,
  centreline, endpoint, corner, junction, and tangent-valid predictions
- `endpoint_corner_junction_tangent_overlay.png`
- `reconstruction_debug.png`
- `missed_support_pixels.png`
- `false_positive_drawn_pixels.png`
- `gantry_path_preview.png`

## Metrics

Both oracle and ML reconstruction report command count, stroke count, average
and median points per stroke, pen-up travel distance, pen-down drawing distance,
total movement distance, estimated plotting time, bounds validation status,
support coverage, endpoint/corner/junction counts, traced endpoint usage,
untraced support fraction, tangent consistency, whether oracle continuity
metadata was used, model path, and label schema.

## Recommended Next Run

Once the smoke path passes, use a modest synthetic-rich pilot before any long
QuickDraw training:

```bash
python src/quickdraw_rich_labels.py \
  --processed-dir data/quickdraw/rich_pilot \
  --synthetic-rich \
  --synthetic-count 500 \
  --image-size 128 \
  --seed 498 \
  --line-width-min 2 \
  --line-width-max 9

python src/train_stroke_structure_model.py \
  --processed-dir data/quickdraw/rich_pilot \
  --model-out models/stroke_structure_pilot_last.pt \
  --best-model-out models/stroke_structure_pilot_best.pt \
  --epochs 10 \
  --batch-size 8 \
  --base-channels 24 \
  --augment \
  --progress-every 10 \
  --support-loss-weight 0.5 \
  --centreline-loss-weight 2.0 \
  --corner-loss-weight 2.0 \
  --tangent-loss-weight 2.0
```

Keep the existing contour baseline, heuristic skeleton pipeline, and 3-class ML
scripts as baselines for comparison.
