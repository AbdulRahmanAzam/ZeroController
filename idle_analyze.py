"""
Capture 6 sequential idle screenshots from the game and analyze horizontal X-center variance.
Uses VS PLAYER mode so AI does not interfere with P1 standing still.
Navigates: VS PLAYER -> KEYBOARD -> FIGHT NOW -> match -> capture quickly.
"""
import time
import os
from playwright.sync_api import sync_playwright
from PIL import Image

OUT_DIR = r"C:\Users\azama\VS Code\PROJECTS\01 Academic and Tutorial Projects\zeroController"
URL = "http://localhost:5176/"

# P1 occupies lower-left of arena. Tight crop on P1 only.
# In 1280x800 viewport, P1 center is roughly x ~ 100-150, y ~ 530-580.
# Use 300x200 crop centered on that area, biased left so P2 isn't included.
CROP_LEFT = 30
CROP_TOP = 430
CROP_W = 300
CROP_H = 200


def find_body_x_center(crop_path: str) -> int | None:
    """
    X-center of P1 body. Filters near-black bg, blue glow rings, green/red UI bars,
    near-white text. Lower 60% of crop only (skip head & UI).
    """
    img = Image.open(crop_path).convert("RGB")
    w, h = img.size
    px = img.load()

    y_start = int(h * 0.40)
    y_end = h

    xs = []
    for y in range(y_start, y_end):
        for x in range(w):
            r, g, b = px[x, y]
            # Skip near-black bg
            if r < 30 and g < 30 and b < 30:
                continue
            # Skip cyan/blue glow rings
            if b > r + 15 and b > g + 5:
                continue
            # Skip pure greens
            if g > r + 25 and g > b + 25:
                continue
            # Skip near-white text
            if r > 230 and g > 230 and b > 230:
                continue
            # Skip pure red UI
            if r > 180 and g < 60 and b < 60:
                continue
            xs.append(x)

    if not xs:
        return None
    return int(round(sum(xs) / len(xs)))


def start_match(page):
    """Click VS PLAYER -> KEYBOARD -> FIGHT NOW so neither side moves."""
    # VS PLAYER (no AI = both idle)
    try:
        page.click("text=VS PLAYER", timeout=4000)
    except Exception:
        page.click("text=VS AI", timeout=4000)
    time.sleep(0.7)

    # If VS AI was clicked, dismiss difficulty by picking EASY
    try:
        page.click("text=EASY", timeout=1500)
        time.sleep(0.7)
    except Exception:
        pass

    # KEYBOARD controller
    try:
        page.click("text=KEYBOARD", timeout=4000)
    except Exception:
        pass
    time.sleep(0.9)

    # FIGHT NOW
    for label in ["FIGHT NOW", "Fight Now", "FIGHT", "Fight", "START", "Start"]:
        try:
            page.click(f"text={label}", timeout=1500)
            time.sleep(0.5)
            break
        except Exception:
            continue


def main():
    shots = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1280, "height": 800})
        page = context.new_page()

        page.goto(URL, wait_until="networkidle")
        time.sleep(3)

        page.reload(wait_until="networkidle")
        time.sleep(1.5)

        start_match(page)
        # Wait for arena countdown to finish (typical 3-2-1-FIGHT)
        time.sleep(5.5)

        page.screenshot(path=os.path.join(OUT_DIR, "idle_debug_prefight.png"), full_page=False)

        # Keep both P1 and P2 idle (VS PLAYER means no AI). Don't press any keys.
        for i in range(6):
            shot_path = os.path.join(OUT_DIR, f"idle_seq_{i}.png")
            page.screenshot(path=shot_path, full_page=False)
            shots.append(shot_path)
            if i < 5:
                time.sleep(0.2)

        browser.close()

    centers = []
    for i, sp in enumerate(shots):
        img = Image.open(sp)
        crop = img.crop((CROP_LEFT, CROP_TOP, CROP_LEFT + CROP_W, CROP_TOP + CROP_H))
        crop_path = os.path.join(OUT_DIR, f"idle_crop_{i}.png")
        crop.save(crop_path)
        cx = find_body_x_center(crop_path)
        centers.append(cx)
        print(f"frame {i}: x_center = {cx}")

    valid = [c for c in centers if c is not None]
    if valid:
        delta = max(valid) - min(valid)
        print(f"\nX-centers: {centers}")
        print(f"min={min(valid)} max={max(valid)} delta={delta}")
        print(f"locked (<=4px): {delta <= 4}")
    else:
        print("No body pixels detected in any frame.")


if __name__ == "__main__":
    main()
