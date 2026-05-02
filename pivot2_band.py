from PIL import Image
import os

OUT = r"C:\Users\azama\VS Code\PROJECTS\01 Academic and Tutorial Projects\zeroController"

# Crop a tight band around P1 sprite on each image, looking for atlas bleed.
# We need to find the P1 sprite first. Use idle to find the bounding box of bright pixels.
# Simpler: crop a wider band and zoom heavily to spot any adjacent ghost.

poses = ["idle", "run", "punch", "kick", "jump"]
for pose in poses:
    src = os.path.join(OUT, f"pivot2_{pose}.png")
    if not os.path.exists(src):
        continue
    im = Image.open(src).convert("RGBA")
    w, h = im.size
    # Looking at the screenshots, P1 in 1280x800 viewport:
    # idle px ~ x: 380-470, y: 420-540
    # we want a wide band around that to spot adjacent frames
    band = im.crop((300, 360, 620, 600))
    bw, bh = band.size
    band = band.resize((bw*4, bh*4), Image.NEAREST)
    band.save(os.path.join(OUT, f"pivot2_{pose}_band.png"))
    print("saved band", pose)
