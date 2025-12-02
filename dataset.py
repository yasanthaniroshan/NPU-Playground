import numpy as np
import cv2
import os

os.makedirs("dataset", exist_ok=True)

N = 100
H, W = 32, 32

# Generate NORMAL distribution (mean=0, std=1)
data = np.random.normal(0, 1, (N, H, W))


with open("dataset/data.txt", "w") as f:
    for i in range(N):
        img = np.random.normal(0,1,(H,W))
        img_norm = (img - img.min())/(img.max() - img.min())
        img = (img_norm*255).astype(np.uint8)
        img = img[:, :, np.newaxis]
        img = np.repeat(img, 3, axis=2)
        cv2.imwrite(f"dataset/img_{i:04d}.png", img)
        f.write(f"img_{i:04d}.png\n")
