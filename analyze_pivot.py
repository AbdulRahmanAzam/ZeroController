import os
from PIL import Image
import numpy as np

OUT = r"C:\Users\azama\VS Code\PROJECTS\01 Academic and Tutorial Projects\zeroController"
FILES = ["pivot_idle.png","pivot_run.png","pivot_kick.png","pivot_punch.png","pivot_jump.png","pivot_guard.png"]

def to_arr(p):
    return np.asarray(Image.open(p).convert("RGB"))

def sprite_mask(arr):
    r = arr[..., 0].astype(int); g = arr[..., 1].astype(int); b = arr[..., 2].astype(int)
    maxc = np.maximum(np.maximum(r, g), b); minc = np.minimum(np.minimum(r, g), b)
    sat = maxc - minc
    # bright + saturated
    return (sat > 70) & (maxc > 100)

def largest_blob(mask):
    # connected components via flood fill (simple iterative)
    from scipy.ndimage import label
    lab, n = label(mask)
    if n == 0: return None
    sizes = np.bincount(lab.flat)
    sizes[0] = 0
    big = sizes.argmax()
    return lab == big

def analyze(arr, x_lo, x_hi, y_lo, y_hi):
    crop = arr[y_lo:y_hi, x_lo:x_hi]
    m = sprite_mask(crop)
    # remove HUD-like horizontal bars (very wide thin shapes)
    blob = largest_blob(m)
    if blob is None: return None
    ys, xs = np.where(blob)
    if len(xs) == 0: return None
    # feet_y = bottom of mass within blob; pivot center = horizontal centroid
    cx = int(xs.mean()) + x_lo
    feet_y = int(ys.max()) + y_lo
    bx0 = int(xs.min()) + x_lo; bx1 = int(xs.max()) + x_lo
    by0 = int(ys.min()) + y_lo; by1 = int(ys.max()) + y_lo
    return (bx0, by0, bx1, by1, cx, feet_y, blob)

def main():
    # P1 fight floor area:  x in [100..620], y in [330..720]  excludes floor below ~720 trim and HUD above 280
    P1_X = (100, 620)
    P1_Y = (300, 705)  # exclude 705+ to drop floor-line
    print("Search window:", P1_X, P1_Y)
    rows = []
    for f in FILES:
        a = to_arr(os.path.join(OUT, f))
        res = analyze(a, P1_X[0], P1_X[1], P1_Y[0], P1_Y[1])
        if res is None: rows.append((f, None)); continue
        x0,y0,x1,y1,cx,feet,blob = res
        rows.append((f,(x0,y0,x1,y1,cx,feet)))
        print(f"{f:20s} bbox=({x0:4d},{y0:4d},{x1:4d},{y1:4d}) W={x1-x0:3d} H={y1-y0:3d} cx={cx:4d} feet_y={feet:4d}")

    # canonical box from union of all bboxes
    valid = [b for _, b in rows if b]
    if not valid: return
    cx0 = min(b[0] for b in valid)
    cy0 = min(b[1] for b in valid)
    cx1 = max(b[2] for b in valid)
    cy1 = max(b[3] for b in valid)
    pad = 20
    cx0 = max(0, cx0-pad); cy0 = max(0, cy0-pad); cx1 += pad; cy1 += pad
    print("Canonical (union+pad):", (cx0,cy0,cx1,cy1), "W=", cx1-cx0, "H=", cy1-cy0)

    for f, _ in rows:
        im = Image.open(os.path.join(OUT, f)).convert("RGB")
        im.crop((cx0, cy0, cx1, cy1)).save(os.path.join(OUT, "crop_"+f))

    base = rows[0][1]
    print("\n--- Deltas vs idle (P1) ---")
    for f, b in rows:
        if b is None: print(f, "missing"); continue
        d_feet = b[5]-base[5]; d_cx = b[4]-base[4]
        print(f"{f:20s} feet_dy={d_feet:+4d}px  center_dx={d_cx:+4d}px  W={b[2]-b[0]} H={b[3]-b[1]}")

main()
