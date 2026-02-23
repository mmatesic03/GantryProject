import cv2
import numpy as np
import matplotlib.pyplot as plt


def main():
    image_path = "input_images/square.png"
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    print(f"Loaded image with shape: {img.shape}")

    plt.imshow(img, cmap="gray")
    plt.title("Loaded Image")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()