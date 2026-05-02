from PIL import Image
import numpy as np
import os

OUT = r"C:\Users\azama\VS Code\PROJECTS\01 Academic and Tutorial Projects\zeroController"
poses = ["idle", "run", "punch", "kick", "jump"]

def find_p1_foot(im_path):
    im = Image.open(im_path).convert("RGBA")
    arr = np.array(im)
    h, w, _ = arr.shape
    # Find P1 sprite cluster: it has distinctive orange/red highlights with grey armor.
    # Heuristic: pixels where R is significantly higher than B (warm tones) plus sat
    r = arr[..., 0].astype(int)
    g = arr[..., 1].astype(int)
    b = arr[..., 2].astype(int)
    warm = (r - b > 40) & (r > 110)
    # Restrict to playable arena area and left half
    warm[:int(h*0.32)] = False
    warm[int(h*0.85):] = False
    warm[:, w//2:] = False  # exclude right half (P2)
    # Find bottom row containing >5 warm pixels in a single column-cluster
    rows_count = warm.sum(axis=1)
    valid = np.where(rows_count > 4)[0]
    if valid.size == 0:
        return None
    return int(valid.max()), h

print(f"{'POSE':10} {'FOOT_Y':>8} {'NORM_Y':>8}")
for pose in poses:
    src = os.path.join(OUT, f"pivot2_{pose}.png")
    if not os.path.exists(src):
        continue
    res = find_p1_foot(src)
    if res:
        by, h = res
        print(f"{pose:10} {by:8d} {by/h:8.3f}")
