from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


def map_paths_to_a4_mm(
    paths_px: list[np.ndarray],
    image_shape: tuple[int, int],
    page_w_mm: float = 210.0,
    page_h_mm: float = 297.0,
    margin_mm: float = 10.0,
    centre_on_page: bool = True,
) -> tuple[list[np.ndarray], dict]:
    img_h_px, img_w_px = image_shape

    drawable_w_mm = page_w_mm - 2 * margin_mm
    drawable_h_mm = page_h_mm - 2 * margin_mm

    if drawable_w_mm <= 0 or drawable_h_mm <= 0:
        raise ValueError("Margins are too large for the page size.")

    sx = drawable_w_mm / img_w_px
    sy = drawable_h_mm / img_h_px
    s = min(sx, sy)

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
        x_px = path_px[:, 0]
        y_px = path_px[:, 1]

        x_mm = x_px * s + x_offset_mm
        y_top_mm = y_px * s + y_offset_mm
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


def save_a4_preview(
    paths_mm: list[np.ndarray],
    output_path: Path,
    page_w_mm: float = 210.0,
    page_h_mm: float = 297.0,
    margin_mm: float = 10.0,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 11))

    page_x = [0, page_w_mm, page_w_mm, 0, 0]
    page_y = [0, 0, page_h_mm, page_h_mm, 0]
    ax.plot(page_x, page_y, linewidth=1.5, label="A4 page")

    draw_x0, draw_y0 = margin_mm, margin_mm
    draw_x1, draw_y1 = page_w_mm - margin_mm, page_h_mm - margin_mm
    draw_x = [draw_x0, draw_x1, draw_x1, draw_x0, draw_x0]
    draw_y = [draw_y0, draw_y0, draw_y1, draw_y1, draw_y0]
    ax.plot(draw_x, draw_y, linestyle="--", linewidth=1.0, label="Drawable area")

    for i, path in enumerate(paths_mm):
        if len(path) == 0:
            continue
        ax.plot(path[:, 0], path[:, 1], linewidth=1.2, label=f"Stroke {i}")
        ax.plot(path[0, 0], path[0, 1], marker="o", markersize=4)

    ax.set_xlim(-5, page_w_mm + 5)
    ax.set_ylim(-5, page_h_mm + 5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title("A4 Stroke Preview")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def load_and_binarise_line_image(
    image_path: Path,
    threshold_value: int = 127,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a line drawing image and generate binary masks.

    Returns:
      gray_img: grayscale image, uint8
      binary_black_lines: black lines on white background
      binary_white_lines: white lines on black background
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img.copy()

    # Standard binary: black lines on white background
    _, binary_black_lines = cv2.threshold(
        gray_img, threshold_value, 255, cv2.THRESH_BINARY
    )

    # Inverted binary: white lines on black background
    binary_white_lines = 255 - binary_black_lines

    return gray_img, binary_black_lines, binary_white_lines


def save_segmentation_debug_figure(
    gray_img: np.ndarray,
    binary_black_lines: np.ndarray,
    binary_white_lines: np.ndarray,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(gray_img, cmap="gray")
    axes[0].set_title("Grayscale input")
    axes[0].axis("off")

    axes[1].imshow(binary_black_lines, cmap="gray")
    axes[1].set_title("Binary mask (black lines)")
    axes[1].axis("off")

    axes[2].imshow(binary_white_lines, cmap="gray")
    axes[2].set_title("Binary mask (white lines)")
    axes[2].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def remove_small_components(binary_mask: np.ndarray, min_area: int = 20) -> np.ndarray:
    """
    Remove tiny connected components from a binary mask.
    Expects white foreground (255) on black background (0).
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    cleaned = np.zeros_like(binary_mask)

    for label in range(1, num_labels):  # skip background
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == label] = 255

    return cleaned


def clean_line_mask(
    binary_white_lines: np.ndarray,
    kernel_size: int = 3,
    close_iterations: int = 1,
    open_iterations: int = 0,
    min_component_area: int = 20,
) -> np.ndarray:
    """
    Clean a white-line binary mask before skeletonisation.
    """
    cleaned = binary_white_lines.copy()
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    if close_iterations > 0:
        cleaned = cv2.morphologyEx(
            cleaned, cv2.MORPH_CLOSE, kernel, iterations=close_iterations
        )

    if open_iterations > 0:
        cleaned = cv2.morphologyEx(
            cleaned, cv2.MORPH_OPEN, kernel, iterations=open_iterations
        )

    if min_component_area > 0:
        cleaned = remove_small_components(cleaned, min_area=min_component_area)

    return cleaned


def skeletonise_mask(binary_white_lines: np.ndarray) -> np.ndarray:
    """
    Morphological skeletonisation.
    Input: white foreground (255), black background (0)
    Output: white skeleton (255), black background (0)
    """
    img = binary_white_lines.copy()
    skeleton = np.zeros_like(img)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        eroded = cv2.erode(img, kernel)
        opened = cv2.dilate(eroded, kernel)
        temp = cv2.subtract(img, opened)
        skeleton = cv2.bitwise_or(skeleton, temp)
        img = eroded.copy()

        if cv2.countNonZero(img) == 0:
            break

    return skeleton


def detect_node_candidate_masks(
    skeleton_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect endpoint and junction candidates on a skeleton.

    Returns:
      endpoint_mask
      junction_mask
      node_candidate_mask
    """
    skel01 = (skeleton_mask > 0).astype(np.uint8)

    neighbour_kernel = np.array(
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ],
        dtype=np.float32,
    )

    # Use float filtering, then convert back to integer neighbour counts
    neighbour_count = cv2.filter2D(
        skel01.astype(np.float32),
        cv2.CV_32F,
        neighbour_kernel,
        borderType=cv2.BORDER_CONSTANT,
    )
    neighbour_count = np.rint(neighbour_count).astype(np.int32)

    endpoint_mask = np.where(
        (skel01 == 1) & (neighbour_count == 1), 255, 0
    ).astype(np.uint8)

    junction_mask = np.where(
        (skel01 == 1) & (neighbour_count >= 3), 255, 0
    ).astype(np.uint8)

    node_candidate_mask = cv2.bitwise_or(endpoint_mask, junction_mask)

    return endpoint_mask, junction_mask, node_candidate_mask


def save_pre_graph_debug_figure(
    gray_img: np.ndarray,
    cleaned_line_mask: np.ndarray,
    skeleton_mask: np.ndarray,
    endpoint_mask: np.ndarray,
    junction_mask: np.ndarray,
    node_candidate_mask: np.ndarray,
    output_path: Path,
) -> None:
    """
    Save a visual checkpoint for everything produced before graph reconstruction.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(gray_img, cmap="gray")
    axes[0, 0].set_title("Grayscale input")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(cleaned_line_mask, cmap="gray")
    axes[0, 1].set_title("Cleaned line mask")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(skeleton_mask, cmap="gray")
    axes[0, 2].set_title("Skeleton mask")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(endpoint_mask, cmap="gray")
    axes[1, 0].set_title("Endpoint mask")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(junction_mask, cmap="gray")
    axes[1, 1].set_title("Junction mask")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(node_candidate_mask, cmap="gray")
    axes[1, 2].set_title("Node candidate mask")
    axes[1, 2].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def main():
    repo_root = Path(__file__).resolve().parents[1]
    input_dir = repo_root / "input_images"
    output_dir = repo_root / "Stroke based output"
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = input_dir / "square.png"
    threshold_value = 127

    print(f"Loading test image: {image_path}")

    gray_img, binary_black_lines, binary_white_lines = load_and_binarise_line_image(
        image_path=image_path,
        threshold_value=threshold_value,
    )

    print(f"Loaded image shape: {gray_img.shape}")
    print(f"Threshold used: {threshold_value}")

    # Existing segmentation-stage outputs
    cv2.imwrite(str(output_dir / "01_gray_input.png"), gray_img)
    cv2.imwrite(str(output_dir / "02_binary_black_lines.png"), binary_black_lines)
    cv2.imwrite(str(output_dir / "03_binary_white_lines.png"), binary_white_lines)

    debug_fig_path = output_dir / "04_segmentation_debug_figure.png"
    save_segmentation_debug_figure(
        gray_img=gray_img,
        binary_black_lines=binary_black_lines,
        binary_white_lines=binary_white_lines,
        output_path=debug_fig_path,
    )

    # New pre-graph structural masks
    cleaned_line_mask = clean_line_mask(
        binary_white_lines,
        kernel_size=3,
        close_iterations=1,
        open_iterations=0,
        min_component_area=20,
    )

    skeleton_mask = skeletonise_mask(cleaned_line_mask)

    endpoint_mask, junction_mask, node_candidate_mask = detect_node_candidate_masks(
        skeleton_mask
    )

    cv2.imwrite(str(output_dir / "05_cleaned_line_mask.png"), cleaned_line_mask)
    cv2.imwrite(str(output_dir / "06_skeleton_mask.png"), skeleton_mask)
    cv2.imwrite(str(output_dir / "07_endpoint_mask.png"), endpoint_mask)
    cv2.imwrite(str(output_dir / "08_junction_mask.png"), junction_mask)
    cv2.imwrite(str(output_dir / "09_node_candidate_mask.png"), node_candidate_mask)

    pre_graph_fig_path = output_dir / "10_pre_graph_debug_figure.png"
    save_pre_graph_debug_figure(
        gray_img=gray_img,
        cleaned_line_mask=cleaned_line_mask,
        skeleton_mask=skeleton_mask,
        endpoint_mask=endpoint_mask,
        junction_mask=junction_mask,
        node_candidate_mask=node_candidate_mask,
        output_path=pre_graph_fig_path,
    )

    print("Saved outputs:")
    print(f"  {output_dir / '01_gray_input.png'}")
    print(f"  {output_dir / '02_binary_black_lines.png'}")
    print(f"  {output_dir / '03_binary_white_lines.png'}")
    print(f"  {debug_fig_path}")
    print(f"  {output_dir / '05_cleaned_line_mask.png'}")
    print(f"  {output_dir / '06_skeleton_mask.png'}")
    print(f"  {output_dir / '07_endpoint_mask.png'}")
    print(f"  {output_dir / '08_junction_mask.png'}")
    print(f"  {output_dir / '09_node_candidate_mask.png'}")
    print(f"  {pre_graph_fig_path}")


if __name__ == "__main__":
    main()