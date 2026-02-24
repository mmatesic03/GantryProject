from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


def close_contour_points(contour: np.ndarray) -> np.ndarray:
    """
    Convert OpenCV contour format (N,1,2) into (N,2) float array and ensure closed loop.
    If first and last points are not the same, append the first point to close the contour.
    """
    pts = contour[:, 0, :].astype(np.float64)  # shape: (N, 2)

    if len(pts) < 2:
        return pts

    if not np.array_equal(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])

    return pts


def resample_closed_polyline(points: np.ndarray, spacing: float) -> np.ndarray:
    """
    Resample a closed polyline at approximately uniform spacing.
    Input: points shape (N,2), expected closed (first == last).
    Output: sampled points shape (M,2), closed loop (first point repeated at end).
    """
    if spacing <= 0:
        raise ValueError("spacing must be > 0")

    if len(points) < 2:
        return points.copy()

    seg_vecs = points[1:] - points[:-1]
    seg_lens = np.linalg.norm(seg_vecs, axis=1)

    total_len = np.sum(seg_lens)
    if total_len == 0:
        return points[:1].copy()

    # Distances where we want points (start at 0, end before total_len to avoid duplicate point at the end)
    sample_dists = np.arange(0, total_len, spacing, dtype=np.float64)

    cumlen = np.concatenate([[0.0], np.cumsum(seg_lens)])

    sampled = []
    seg_idx = 0

    for d in sample_dists:
        while seg_idx < len(seg_lens) - 1 and d > cumlen[seg_idx + 1]:
            seg_idx += 1

        seg_start = points[seg_idx]
        seg_end = points[seg_idx + 1]
        seg_len = seg_lens[seg_idx]

        if seg_len == 0:
            sampled.append(seg_start.copy())
            continue

        local_d = d - cumlen[seg_idx]
        t = local_d / seg_len
        p = seg_start + t * (seg_end - seg_start)
        sampled.append(p)

    sampled = np.array(sampled, dtype=np.float64)

    # Explicitly close loop
    if len(sampled) > 0:
        sampled = np.vstack([sampled, sampled[0]])

    return sampled


def map_paths_to_a4_mm(
    paths_px: list[np.ndarray],
    image_shape: tuple[int, int],
    page_w_mm: float = 210.0,
    page_h_mm: float = 297.0,
    margin_mm: float = 10.0,
    centre_on_page: bool = True,
) -> tuple[list[np.ndarray], dict]:
    """
    Map sampled paths from image pixel coordinates to A4 mm coordinates.

    - Preserves aspect ratio (uniform scaling only)
    - Fits entire image inside drawable area (page - margins)
    - Converts origin from image top-left to page bottom-left

    Returns:
      mapped_paths_mm: list of paths, each shape (N,2) with [x_mm, y_mm]
      info: dict with transform/debug values
    """
    img_h_px, img_w_px = image_shape  # note: image shape is (h, w)

    drawable_w_mm = page_w_mm - 2 * margin_mm
    drawable_h_mm = page_h_mm - 2 * margin_mm

    if drawable_w_mm <= 0 or drawable_h_mm <= 0:
        raise ValueError("Margins are too large for the page size.")

    # Uniform scaling (no warping)
    sx = drawable_w_mm / img_w_px
    sy = drawable_h_mm / img_h_px
    s = min(sx, sy)  # mm per pixel

    used_w_mm = img_w_px * s
    used_h_mm = img_h_px * s

    if centre_on_page:
        x_offset_mm = margin_mm + (drawable_w_mm - used_w_mm) / 2.0
        y_offset_mm = margin_mm + (drawable_h_mm - used_h_mm) / 2.0
    else:
        x_offset_mm = margin_mm
        y_offset_mm = margin_mm

    mapped_paths_mm = []

    for path_px in paths_px:
        # path_px columns: [x_px, y_px]
        x_px = path_px[:, 0]
        y_px = path_px[:, 1]

        # Scale + place on page in a top-left style frame first
        x_mm = x_px * s + x_offset_mm
        y_top_mm = y_px * s + y_offset_mm

        # Flip y so origin becomes bottom-left
        y_mm = page_h_mm - y_top_mm

        path_mm = np.column_stack([x_mm, y_mm])
        mapped_paths_mm.append(path_mm)

    info = {
        "page_w_mm": page_w_mm,
        "page_h_mm": page_h_mm,
        "margin_mm": margin_mm,
        "drawable_w_mm": drawable_w_mm,
        "drawable_h_mm": drawable_h_mm,
        "img_w_px": img_w_px,
        "img_h_px": img_h_px,
        "scale_mm_per_px": s,
        "used_w_mm": used_w_mm,
        "used_h_mm": used_h_mm,
        "x_offset_mm": x_offset_mm,
        "y_offset_mm": y_offset_mm,
        "centre_on_page": centre_on_page,
    }

    return mapped_paths_mm, info

def save_a4_preview(paths_mm: list[np.ndarray], output_path: Path,
                    page_w_mm: float = 210.0, page_h_mm: float = 297.0,
                    margin_mm: float = 10.0) -> None:
    """
    Save a visual preview of the sampled paths on an A4 page (in mm coordinates).
    Assumes input coordinates use bottom-left origin.
    """
    fig, ax = plt.subplots(figsize=(8, 11))  # portrait-like view

    # Page border (A4)
    page_x = [0, page_w_mm, page_w_mm, 0, 0]
    page_y = [0, 0, page_h_mm, page_h_mm, 0]
    ax.plot(page_x, page_y, linewidth=1.5, label="A4 page")

    # Drawable area border (margins)
    draw_x0, draw_y0 = margin_mm, margin_mm
    draw_x1, draw_y1 = page_w_mm - margin_mm, page_h_mm - margin_mm
    draw_x = [draw_x0, draw_x1, draw_x1, draw_x0, draw_x0]
    draw_y = [draw_y0, draw_y0, draw_y1, draw_y1, draw_y0]
    ax.plot(draw_x, draw_y, linestyle="--", linewidth=1.0, label="Drawable area")

    # Plot each contour/path
    for i, path in enumerate(paths_mm):
        if len(path) == 0:
            continue
        ax.plot(path[:, 0], path[:, 1], linewidth=1.2, label=f"Contour {i}")
        # Mark start point
        ax.plot(path[0, 0], path[0, 1], marker="o", markersize=4)
        ax.text(path[0, 0] + 2, path[0, 1] + 2, f"C{i} start", fontsize=8)

    # Axes formatting (important for correctness)
    ax.set_xlim(-5, page_w_mm + 5)
    ax.set_ylim(-5, page_h_mm + 5)
    ax.set_aspect("equal", adjustable="box")  # prevent distortion
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title("A4 Plotter Path Preview (Bottom-left origin)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def detect_sharp_corners_on_closed_contour(
    points: np.ndarray,
    angle_threshold_deg: float = 35.0,
    window: int = 6,
    min_separation: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect sharp corners on a closed contour using local turning angle.

    Parameters:
      points: (N,2) closed contour points (first == last expected)
      angle_threshold_deg: smaller angle = sharper corner (e.g. 30-60 deg)
      window: how many contour samples before/after to use for local direction
      min_separation: suppress nearby duplicate detections (in contour index units)

    Returns:
      corner_indices: indices into `points` for preserved corners (excluding duplicate end)
      corner_points: (K,2) array of corner coordinates
    """
    if len(points) < (2 * window + 3):
        return np.array([], dtype=int), np.empty((0, 2), dtype=np.float64)

    # Work on the unique part only (drop duplicate end point if present)
    if np.allclose(points[0], points[-1]):
        pts = points[:-1]
    else:
        pts = points

    n = len(pts)
    if n < (2 * window + 3):
        return np.array([], dtype=int), np.empty((0, 2), dtype=np.float64)

    candidate_idx = []
    candidate_angle = []

    for i in range(n):
        i_prev = (i - window) % n
        i_next = (i + window) % n

        v1 = pts[i_prev] - pts[i]   # current point to previous point
        v2 = pts[i_next] - pts[i]   # current point to next point

        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-9 or n2 < 1e-9:
            continue

        # angle between the two local directions
        cosang = np.dot(v1, v2) / (n1 * n2)
        cosang = np.clip(cosang, -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cosang))

        # Smaller angle => sharper turn
        if angle_deg <= angle_threshold_deg:
            candidate_idx.append(i)
            candidate_angle.append(angle_deg)

    if not candidate_idx:
        return np.array([], dtype=int), np.empty((0, 2), dtype=np.float64)

    candidate_idx = np.array(candidate_idx, dtype=int)
    candidate_angle = np.array(candidate_angle, dtype=np.float64)

    # we want the SHARPEST point in a neighbourhood, so sort by angle ascending and suppress nearby duplicates
    order = np.argsort(candidate_angle)

    selected = []
    for oi in order:
        idx = int(candidate_idx[oi])

        too_close = False
        for s in selected:
            # Circular contour distance
            d = abs(idx - s)
            d = min(d, n - d)
            if d < min_separation:
                too_close = True
                break

        if not too_close:
            selected.append(idx)

    selected = np.array(sorted(selected), dtype=int)
    corner_points = pts[selected]

    return selected, corner_points

def merge_preserved_points_into_sampled_closed_path(
    raw_closed_points: np.ndarray,
    sampled_closed_points: np.ndarray,
    corner_indices_raw: np.ndarray,
) -> np.ndarray:
    """
    Merge preserved corner points (from raw contour) into sampled path, keeping contour order.

    Strategy:
    - Build a combined path from sampled points + exact corner points
    - Sort all combined points by arc-length position along the raw contour
    - Remove near-duplicates
    - Re-close the loop
    """
    if len(sampled_closed_points) == 0:
        return sampled_closed_points.copy()

    # Raw unique points (drop duplicate close if present)
    if np.allclose(raw_closed_points[0], raw_closed_points[-1]):
        raw_pts = raw_closed_points[:-1]
    else:
        raw_pts = raw_closed_points

    # Sampled unique points (drop duplicate close if present)
    if np.allclose(sampled_closed_points[0], sampled_closed_points[-1]):
        sampled_pts = sampled_closed_points[:-1]
    else:
        sampled_pts = sampled_closed_points

    n_raw = len(raw_pts)
    if n_raw < 2:
        return sampled_closed_points.copy()

    # Build arc length on raw contour
    raw_closed = np.vstack([raw_pts, raw_pts[0]])
    seg_vecs = raw_closed[1:] - raw_closed[:-1]
    seg_lens = np.linalg.norm(seg_vecs, axis=1)
    cumlen = np.concatenate([[0.0], np.cumsum(seg_lens)])  # len = n_raw + 1

    # Map exact raw vertices to arc positions quickly
    raw_key_to_s = {}
    for i in range(n_raw):
        key = (float(raw_pts[i, 0]), float(raw_pts[i, 1]))
        raw_key_to_s[key] = float(cumlen[i])

    def point_to_arc_position(p: np.ndarray) -> float:
        d2 = np.sum((raw_pts - p) ** 2, axis=1)
        i = int(np.argmin(d2))
        return float(cumlen[i])

    combined = []

    # Add sampled points
    for p in sampled_pts:
        s = point_to_arc_position(p)
        combined.append((s, p.copy(), "sampled"))

    # Add exact preserved corners
    for idx in corner_indices_raw:
        idx = int(idx)
        p = raw_pts[idx]
        s = float(cumlen[idx])
        combined.append((s, p.copy(), "corner"))

    # Sort by arc position
    combined.sort(key=lambda t: t[0])

    # keep corners if same location (e.g. if a corner was very close to a sampled point, we want to keep the corner exact)
    dedup = []
    tol = 1e-6
    for item in combined:
        s, p, tag = item
        if not dedup:
            dedup.append(item)
            continue

        s_prev, p_prev, tag_prev = dedup[-1]
        if np.linalg.norm(p - p_prev) < tol:
            # Prefer corner over sampled if duplicate
            if tag == "corner" and tag_prev != "corner":
                dedup[-1] = item
            continue

        dedup.append(item)

    merged_pts = np.array([p for _, p, _ in dedup], dtype=np.float64)

    # Re-close
    if len(merged_pts) > 0:
        merged_pts = np.vstack([merged_pts, merged_pts[0]])

    return merged_pts


def main():
    repo_root = Path(__file__).resolve().parents[1]
    image_path = repo_root / "input_images" / "face test 3.png"
    output_dir = repo_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tunable parameters
    point_spacing_px = 15.0
    page_w_mm = 210.0
    page_h_mm = 297.0
    margin_mm = 10.0
    # Contour retrieval mode / filtering
    contour_mode = "all"   # "external" or "all"
    min_contour_area_px2 = 20.0
    min_contour_perimeter_px = 20.0
    # Corner preservation parameters
    enable_corner_preservation = True
    corner_angle_threshold_deg = 140
    corner_window = 8                   # local neighbourhood size along contour
    corner_min_separation = 17          # suppress duplicate detections near same corner
    print(f"Trying to load: {image_path}")
    print(f"Exists: {image_path.exists()}")

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"OpenCV could not load image: {image_path}")

    print(f"Loaded image with shape: {img.shape}")  # (h, w)

    # Threshold to binary (inverted: contours become white on black background)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # Find external contours only (separate disconnected bodies)
        # Contour retrieval mode:
    # - "external": only outer boundaries
    # - "all": all contours (including holes / nested boundaries)
    if contour_mode == "external":
        retrieval_flag = cv2.RETR_EXTERNAL
    elif contour_mode == "all":
        retrieval_flag = cv2.RETR_TREE
    else:
        raise ValueError(f"Unknown contour_mode: {contour_mode}")

    contours, hierarchy = cv2.findContours(binary, retrieval_flag, cv2.CHAIN_APPROX_NONE)

    print(f"Contours found (raw): {len(contours)}")
    if len(contours) == 0:
        print("No contours found. Check thresholding or input image.")
        return

    # Filter tiny/noisy contours
    filtered_contours = []
    filtered_meta = []  # optional debug info (area/perimeter/index/hierarchy row)

    for ci, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, closed=True)

        if area < min_contour_area_px2:
            continue
        if perimeter < min_contour_perimeter_px:
            continue

        filtered_contours.append(cnt)

        h_row = None
        if hierarchy is not None:
            # hierarchy shape is usually (1, N, 4): [next, prev, first_child, parent]
            h_row = hierarchy[0, ci].copy()
        filtered_meta.append((ci, area, perimeter, h_row))

    contours = filtered_contours

    print(f"Contours kept after filtering: {len(contours)}")
    print(
        f"  Filters -> min_area={min_contour_area_px2:.1f} px^2, "
        f"min_perimeter={min_contour_perimeter_px:.1f} px"
    )

    if len(contours) == 0:
        print("All contours were filtered out. Reduce area/perimeter thresholds.")
        return

    # Sort largest-first for consistency (debug-friendly)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Overlays
    overlay_raw = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(overlay_raw, contours, -1, (0, 0, 255), 2)

    overlay_sampled = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(overlay_sampled, contours, -1, (180, 180, 180), 1)

    all_sampled_paths_px = []

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, closed=True)

        raw_pts = close_contour_points(cnt)

        # Base uniform contour resampling (curve-preserving backbone)
        sampled_pts_uniform = resample_closed_polyline(raw_pts, point_spacing_px)

        # Corner detection and preservation
        if enable_corner_preservation:
            corner_idx_raw, corner_pts = detect_sharp_corners_on_closed_contour(
                raw_pts,
                angle_threshold_deg=corner_angle_threshold_deg,
                window=corner_window,
                min_separation=corner_min_separation,
            )

            sampled_pts = merge_preserved_points_into_sampled_closed_path(
                raw_closed_points=raw_pts,
                sampled_closed_points=sampled_pts_uniform,
                corner_indices_raw=corner_idx_raw,
            )
        else:
            corner_idx_raw = np.array([], dtype=int)
            corner_pts = np.empty((0, 2), dtype=np.float64)
            sampled_pts = sampled_pts_uniform

        all_sampled_paths_px.append(sampled_pts)

        print(f"\nContour {i}")
        print(f"  Area (px^2): {area:.2f}")
        print(f"  Perimeter (px): {perimeter:.2f}")
        print(f"  Raw contour points: {len(raw_pts)} (closed)")
        print(f"  Requested spacing (px): {point_spacing_px}")
        print(f"  Sampled points: {len(sampled_pts)} (includes repeated end point)")
        print(f"  Corners detected: {len(corner_pts)}")
        if len(corner_pts) > 0:
            print(f"  Corner preservation: ENABLED")
        else:
            print(f"  Corner preservation: {'ENABLED' if enable_corner_preservation else 'DISABLED'}")
        if len(sampled_pts) > 1:
            approx_spacing = perimeter / (len(sampled_pts) - 1)
            print(f"  Approx achieved spacing (px): {approx_spacing:.2f}")

        # Draw sampled path on pixel overlay
        for j in range(len(sampled_pts) - 1):
            p1 = tuple(np.round(sampled_pts[j]).astype(int))
            p2 = tuple(np.round(sampled_pts[j + 1]).astype(int))
            cv2.line(overlay_sampled, p1, p2, (255, 0, 0), 1)

        for p in sampled_pts[:-1]:
            x, y = np.round(p).astype(int)
            cv2.circle(overlay_sampled, (x, y), 2, (0, 255, 0), -1)

        if len(sampled_pts) > 0:
            x0, y0 = np.round(sampled_pts[0]).astype(int)
            cv2.putText(
                overlay_sampled,
                f"C{i}",
                (x0 + 5, y0 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
        # Draw detected corners
        for p in corner_pts:
            x, y = np.round(p).astype(int)
            cv2.circle(overlay_sampled, (x, y), 5, (255, 0, 255), 2)
    # Map sampled pixel paths to A4 coordinates
    all_sampled_paths_mm, tf_info = map_paths_to_a4_mm(
        all_sampled_paths_px,
        image_shape=img.shape,  # (h, w)
        page_w_mm=page_w_mm,
        page_h_mm=page_h_mm,
        margin_mm=margin_mm,
        centre_on_page=True,
    )

    print("\nA4 mapping info")
    print(f"  Page size (mm): {tf_info['page_w_mm']} x {tf_info['page_h_mm']}")
    print(f"  Margin (mm): {tf_info['margin_mm']}")
    print(f"  Drawable area (mm): {tf_info['drawable_w_mm']:.2f} x {tf_info['drawable_h_mm']:.2f}")
    print(f"  Image size (px): {tf_info['img_w_px']} x {tf_info['img_h_px']}")
    print(f"  Scale (mm/px): {tf_info['scale_mm_per_px']:.6f}")
    print(f"  Used size on page (mm): {tf_info['used_w_mm']:.2f} x {tf_info['used_h_mm']:.2f}")
    print(f"  Offset on page (mm): x={tf_info['x_offset_mm']:.2f}, y={tf_info['y_offset_mm']:.2f}")

    # Save debug images
    cv2.imwrite(str(output_dir / "binary.png"), binary)
    cv2.imwrite(str(output_dir / "contours_overlay.png"), overlay_raw)
    cv2.imwrite(str(output_dir / "sampled_points_overlay.png"), overlay_sampled)

    # Summary figure (pixel-space verification)
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(binary, cmap="gray")
    axes[1].set_title("Binary")
    axes[1].axis("off")

    axes[2].imshow(cv2.cvtColor(overlay_raw, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Raw Contours")
    axes[2].axis("off")

    axes[3].imshow(cv2.cvtColor(overlay_sampled, cv2.COLOR_BGR2RGB))
    axes[3].set_title(f"Resampled Points ({point_spacing_px}px)")
    axes[3].axis("off")

    fig.tight_layout()
    fig.savefig(output_dir / "resampling_summary.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Save sampled pixel coordinates
    coords_px_out = output_dir / "sampled_points_px.txt"
    with open(coords_px_out, "w", encoding="utf-8") as f:
        f.write("# Sampled contour points in pixel coordinates (x_px, y_px)\n")
        f.write(f"# point_spacing_px = {point_spacing_px}\n")
        for ci, path in enumerate(all_sampled_paths_px):
            f.write(f"\n# Contour {ci}\n")
            for p in path:
                f.write(f"{p[0]:.3f}, {p[1]:.3f}\n")

    print(f"\nSaved sampled point list (px) to: {coords_px_out}")

    # Save sampled A4 mm coordinates
    coords_mm_out = output_dir / "sampled_points_a4_mm.txt"
    with open(coords_mm_out, "w", encoding="utf-8") as f:
        f.write("# Sampled contour points mapped to A4 coordinates (x_mm, y_mm)\n")
        f.write(f"# page_w_mm = {page_w_mm}\n")
        f.write(f"# page_h_mm = {page_h_mm}\n")
        f.write(f"# margin_mm = {margin_mm}\n")
        f.write(f"# scale_mm_per_px = {tf_info['scale_mm_per_px']:.8f}\n")
        f.write(f"# x_offset_mm = {tf_info['x_offset_mm']:.8f}\n")
        f.write(f"# y_offset_mm = {tf_info['y_offset_mm']:.8f}\n")
        f.write(f"# image_size_px = ({tf_info['img_w_px']}, {tf_info['img_h_px']})\n")
        f.write(f"# used_size_mm = ({tf_info['used_w_mm']:.8f}, {tf_info['used_h_mm']:.8f})\n")
        f.write("# Origin convention: bottom-left of A4 page is (0,0)\n")

        for ci, path in enumerate(all_sampled_paths_mm):
            f.write(f"\n# Contour {ci}\n")
            for p in path:
                f.write(f"{p[0]:.3f}, {p[1]:.3f}\n")

    print(f"Saved sampled point list (A4 mm) to: {coords_mm_out}")

    a4_preview_out = output_dir / "a4_path_preview.png"
    save_a4_preview(
        all_sampled_paths_mm,
        a4_preview_out,
        page_w_mm=page_w_mm,
        page_h_mm=page_h_mm,
        margin_mm=margin_mm,
    )
    print(f"Saved A4 path preview to:       {a4_preview_out}")


if __name__ == "__main__":

    main()