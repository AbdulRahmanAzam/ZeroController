"""Compute per-frame pivot (feet center) from actual pixel data.

For fighting-game framing the pivot must lock the character's
feet/hip to a consistent screen position across all animations,
otherwise the character jitters or sinks during pose transitions.

Strategy:
- For each frame's pixel rectangle, find the lowest opaque row (feet baseline).
- For that bottom strip (last 6 rows) take the median X of opaque pixels — that
  is the standing/foot center.
- Pivot stored as (px, py) measured from frame top-left in source pixels.
"""
from PIL import Image
import json
import statistics

ATLAS = 'knight.json'
PNG = 'knight.png'
OUT = 'knight.json'  # rewrite in place with added pivots

img = Image.open(PNG).convert('RGBA')
W, H = img.size
pixels = img.load()
ALPHA_THRESHOLD = 24

with open(ATLAS, 'r') as f:
    data = json.load(f)

frames = data['textures'][0]['frames']

def opaque_xs(x0, y0, w, h, y_band):
    """Return x coords (frame-local) where alpha > threshold inside given y band rows."""
    xs = []
    for dy in y_band:
        ay = y0 + dy
        if ay < 0 or ay >= H:
            continue
        for dx in range(w):
            ax = x0 + dx
            if ax < 0 or ax >= W:
                continue
            _, _, _, a = pixels[ax, ay]
            if a > ALPHA_THRESHOLD:
                xs.append(dx)
    return xs

def find_feet_y(x0, y0, w, h):
    """Return frame-local y of lowest opaque pixel."""
    for dy in range(h - 1, -1, -1):
        ay = y0 + dy
        if ay < 0 or ay >= H:
            continue
        for dx in range(w):
            ax = x0 + dx
            if ax < 0 or ax >= W:
                continue
            _, _, _, a = pixels[ax, ay]
            if a > ALPHA_THRESHOLD:
                return dy
    return h - 1

for fr in frames:
    fx, fy = fr['frame']['x'], fr['frame']['y']
    fw, fh = fr['frame']['w'], fr['frame']['h']
    feet_y = find_feet_y(fx, fy, fw, fh)
    # Sample a 6-row band ending at feet_y
    band = list(range(max(0, feet_y - 5), feet_y + 1))
    xs = opaque_xs(fx, fy, fw, fh, band)
    if not xs:
        # fall back to whole-frame midline
        xs = opaque_xs(fx, fy, fw, fh, range(fh))
    pivot_x = int(statistics.median(xs)) if xs else fw // 2
    pivot_y = feet_y + 1  # baseline below last opaque row
    fr['pivot'] = {'x': pivot_x, 'y': pivot_y}

# For stationary animations (idle, guard) the artist's per-frame foot
# variation reads as side-to-side body sway. Lock those animations to a
# single canonical pivot so the character breathes without translating.
STATIC_ANIMS = {'idle', 'guard', 'guard_start', 'guard_end', 'get_hit'}
from collections import defaultdict
by_anim = defaultdict(list)
for fr in frames:
    by_anim[fr['filename'].split('/')[0]].append(fr)
for anim_name, anim_frames in by_anim.items():
    if anim_name not in STATIC_ANIMS:
        continue
    # Use median pivot.x across animation as locked horizontal pivot.
    locked_x = int(statistics.median([f['pivot']['x'] for f in anim_frames]))
    locked_y = int(statistics.median([f['pivot']['y'] for f in anim_frames]))
    for f in anim_frames:
        f['pivot']['x'] = locked_x
        f['pivot']['y'] = locked_y

# Sanity print
from collections import defaultdict
groups = defaultdict(list)
for fr in frames:
    name = fr['filename'].split('/')[0]
    groups[name].append((fr['pivot']['x'], fr['pivot']['y'], fr['sourceSize']['w'], fr['sourceSize']['h']))
for k in sorted(groups):
    pxs = [g[0] for g in groups[k]]
    pys = [g[1] for g in groups[k]]
    print(f'{k}: pivot.x={min(pxs)}-{max(pxs)} pivot.y={min(pys)}-{max(pys)} src.w={[g[2] for g in groups[k]][:3]}...')

with open(OUT, 'w') as f:
    json.dump(data, f, indent='\t')
print('written:', OUT)
