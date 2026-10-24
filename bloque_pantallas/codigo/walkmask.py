"""Robust mask for the center person while he walks out close to the camera."""
import cv2
import numpy as np

PLATE = cv2.imread('b4/plate.png').astype(np.float32)


def walk_mask(frame, u2):
    d = cv2.GaussianBlur(np.abs(frame - PLATE).max(2), (0, 0), 2)
    near = cv2.dilate((u2 > 90).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81, 81))) > 0
    m = ((d > 13) & near) | (u2 > 90)
    m = cv2.morphologyEx(m.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35)))
    # fill holes
    # pad with an empty border so the flood always starts on background, even when he touches the frame edge
    ff = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0); h, w = ff.shape
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
    m = m | cv2.bitwise_not(ff[1:-1, 1:-1])
    # keep components that touch the detected person
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    keep = np.zeros_like(m)
    for k in range(1, n):
        comp = lab == k
        if (comp & (u2 > 90)).any():
            keep[comp] = 255
    return cv2.GaussianBlur(keep.astype(np.float32) / 255, (0, 0), 1.5)
