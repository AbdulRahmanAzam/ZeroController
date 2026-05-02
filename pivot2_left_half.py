from PIL import Image
import os

OUT = r"C:\Users\azama\VS Code\PROJECTS\01 Academic and Tutorial Projects\zeroController"

# Take left half of each screenshot (P1 region) at full resolution to count knights on P1 side
poses = ["idle", "run", "punch", "kick", "jump"]
for pose in poses:
    src = os.path.join(OUT, f"pivot2_{pose}.png")
    if not os.path.exists(src):
        continue
    im = Image.open(src)
    w, h = im.size
    # full left half (0 to mid)
    left = im.crop((0, 0, w//2, h))
    left.save(os.path.join(OUT, f"pivot2_{pose}_lefthalf.png"))
    print("saved lefthalf", pose, "->", left.size)
