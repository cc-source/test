import cv2
import numpy as np
import math
import sys


def compute_focal_length(width, fov_deg):
    """Compute focal length from image width and horizontal FOV."""
    return width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))


def warp_cylindrical_with_yaw(img, f, yaw_rad, output_width):
    """Warp an image onto a cylindrical surface and rotate by yaw."""
    h, w = img.shape[:2]
    # create mesh grid of pixel coordinates
    y_i, x_i = np.indices((h, w))
    x_c = (x_i - w / 2.0) / f
    y_c = (y_i - h / 2.0) / f
    phi = np.arctan(x_c) + yaw_rad
    # cylindrical vertical coordinate
    rho = y_c / np.sqrt(x_c ** 2 + 1)

    xp = f * phi + output_width / 2.0
    yp = f * rho + h / 2.0

    map_x = xp.astype(np.float32)
    map_y = yp.astype(np.float32)
    warped = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    mask = cv2.remap(np.full((h, w), 255, dtype=np.uint8), map_x, map_y, cv2.INTER_NEAREST,
                     borderMode=cv2.BORDER_CONSTANT)
    return warped, mask


def blend(base, overlay, mask):
    """Simple overlay blend using mask."""
    mask3 = mask[:, :, None].astype(bool)
    base[mask3] = overlay[mask3]
    return base


def main():
    if len(sys.argv) != 6:
        print("Usage: python main.py cam1.jpg cam2.jpg cam3.jpg cam4.jpg output.jpg")
        return

    image_paths = sys.argv[1:5]
    images = [cv2.imread(p) for p in image_paths]
    if any(img is None for img in images):
        print("Error reading input images")
        return

    h, w = images[0].shape[:2]
    f = compute_focal_length(w, 120.0)
    panorama_width = int(2 * math.pi * f)

    panorama = np.zeros((h, panorama_width, 3), dtype=images[0].dtype)
    mask_total = np.zeros((h, panorama_width), dtype=np.uint8)

    yaw_angles = [0, 90, 180, 270]
    for img, yaw_deg in zip(images, yaw_angles):
        yaw_rad = math.radians(yaw_deg)
        warped, mask = warp_cylindrical_with_yaw(img, f, yaw_rad, panorama_width)
        panorama = blend(panorama, warped, mask)
        mask_total = cv2.max(mask_total, mask)

    cv2.imwrite(sys.argv[5], panorama)
    print("Saved", sys.argv[5])


if __name__ == "__main__":
    main()
