# GantryProject

XYZ gantry drawing project. The active robot output format is Arduino serial commands:

```text
x_mm,y_mm,mode
```

`mode 0` is travel / pen up and `mode 1` is draw / pen down. The project does not use G-code as its main output.

The stroke-based experimental pipeline lives in `src/stroke_based_pipeline.py` and writes separate outputs under `output/stroke_based/`.

Run the heuristic stroke pipeline:

```bash
python src/stroke_based_pipeline.py --image input_images/square.png --output-dir output/stroke_based --segmentation-mode heuristic
```

Run automatic ML-then-heuristic fallback:

```bash
python src/stroke_based_pipeline.py --image input_images/square.png --output-dir output/stroke_based --segmentation-mode auto
```

The optional QuickDraw/PyTorch ML training pathway is documented in `docs/stroke_ml_pipeline.md`.
