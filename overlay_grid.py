import os
from PIL import Image
OUT = r"C:\Users\azama\VS Code\PROJECTS\01 Academic and Tutorial Projects\zeroController"
files = ["pivot_idle.png","pivot_run.png","pivot_kick.png","pivot_punch.png","pivot_jump.png","pivot_guard.png"]
# crop a fixed P1 region from each (full P1 panel including the border)
box = (60, 200, 660, 740)
imgs = [Image.open(os.path.join(OUT, f)).crop(box) for f in files]
W, H = imgs[0].size
grid = Image.new("RGB", (W*3, H*2), (0,0,0))
for i, im in enumerate(imgs):
    grid.paste(im, ((i%3)*W, (i//3)*H))
grid.save(os.path.join(OUT, "p1_grid.png"))
print("saved p1_grid.png")
