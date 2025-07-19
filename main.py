import cv2
import numpy as np
import math
import sys


def detect_and_compute(img):
    """Detect keypoints and compute descriptors using ORB."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(5000)
    kps, des = orb.detectAndCompute(gray, None)
    return kps, des


def match_features(des1, des2, ratio=0.75):
    """Match ORB descriptors using Hamming distance and ratio test."""
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def compute_focal_length(width, fov_deg):
    """Compute focal length from image width and horizontal FOV."""
    return width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))


def calibrate_cameras(images, f):
    """Estimate rotation matrices for each camera using feature matching."""
    h, w = images[0].shape[:2]
    K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]])

    rotations = [np.eye(3)]
    for i in range(1, len(images)):
        kp1, des1 = detect_and_compute(images[i - 1])
        kp2, des2 = detect_and_compute(images[i])
        matches = match_features(des1, des2)
        if len(matches) < 8:
            rotations.append(rotations[-1])
            continue

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        E, mask = cv2.findEssentialMat(pts2, pts1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None:
            rotations.append(rotations[-1])
            continue

        _, R, _, _ = cv2.recoverPose(E, pts2, pts1, K)
        rotations.append(rotations[-1] @ R)

    return rotations


def warp_cylindrical(img, f, R, output_width):
    """Warp an image onto a cylindrical surface using rotation R."""
    h, w = img.shape[:2]
    y_i, x_i = np.indices((h, w))
    x_c = (x_i - w / 2.0) / f
    y_c = (y_i - h / 2.0) / f
    dirs = np.stack([x_c, y_c, np.ones_like(x_c)], axis=-1)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    dirs = dirs @ R.T

    phi = np.arctan2(dirs[..., 0], dirs[..., 2])
    rho = dirs[..., 1] / np.sqrt(dirs[..., 0] ** 2 + dirs[..., 2] ** 2)

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

    rotations = calibrate_cameras(images, f)

    panorama = np.zeros((h, panorama_width, 3), dtype=images[0].dtype)
    mask_total = np.zeros((h, panorama_width), dtype=np.uint8)

    for img, R in zip(images, rotations):
        warped, mask = warp_cylindrical(img, f, R, panorama_width)
        panorama = blend(panorama, warped, mask)
        mask_total = cv2.max(mask_total, mask)

    cv2.imwrite(sys.argv[5], panorama)
    print("Saved", sys.argv[5])


if __name__ == "__main__":
    main()
