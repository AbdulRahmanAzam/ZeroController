from PIL import Image
import os

OUT = r"C:\Users\azama\VS Code\PROJECTS\01 Academic and Tutorial Projects\zeroController"

poses = ["idle", "run", "punch", "kick", "jump"]

# Wider crop bringing P1 nicely into frame. Game viewport is 1280x800.
# Heroes appear at lower 40% of arena. P1 around x=20%-50% across runs.
for pose in poses:
    src = os.path.join(OUT, f"pivot2_{pose}.png")
    if not os.path.exists(src):
        continue
    im = Image.open(src)
    w, h = im.size
    left = im.crop((int(w*0.15), int(h*0.30), int(w*0.55), int(h*0.95)))
    lw, lh = left.size
    left = left.resize((lw*3, lh*3), Image.NEAREST)
    left.save(os.path.join(OUT, f"pivot2_{pose}_p1zoom.png"))
    print("saved", pose)
