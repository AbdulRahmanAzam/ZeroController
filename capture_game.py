"""Capture screenshots of the 2D fighting game in various states."""
from playwright.sync_api import sync_playwright
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = r"C:\Users\azama\VS Code\PROJECTS\01 Academic and Tutorial Projects\zeroController"

def safe(s, n=1500):
    try:
        return (s or "")[:n].encode("ascii", "replace").decode("ascii")
    except Exception:
        return "<unprintable>"

def dump(page, label):
    txt = page.evaluate("() => document.body.innerText")
    print(f"=== PAGE TEXT [{label}] ===")
    print(safe(txt, 1500))
    print("===========================")
    return txt

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        # Force device_scale_factor=1 so the screenshot is exactly 1280x800
        context = browser.new_context(viewport={"width": 1280, "height": 800}, device_scale_factor=1)
        page = context.new_page()

        print("Navigating to http://localhost:5174/")
        page.goto("http://localhost:5174/", wait_until="networkidle", timeout=30000)
        page.wait_for_timeout(1500)
        dump(page, "main menu")

        # Step 1: click VS PLAYER (use button with that text)
        print("Clicking VS PLAYER button")
        page.locator("button", has_text="VS PLAYER").first.click(timeout=3000)
        page.wait_for_timeout(1200)
        dump(page, "after VS PLAYER")

        # Step 2: Click KEYBOARD button on controller-select screen
        print("Clicking KEYBOARD button")
        page.locator("button", has_text="KEYBOARD").first.click(timeout=3000)
        page.wait_for_timeout(1500)
        dump(page, "after KEYBOARD")

        # Step 3: now we should be on the game arena with a Start / Fight overlay
        # Check buttons
        buttons = page.query_selector_all("button")
        print(f"Buttons after KEYBOARD click: {len(buttons)}")
        for i, b in enumerate(buttons):
            try:
                t = b.inner_text().strip()
                v = b.is_visible()
                print(f"  Btn {i}: visible={v} text='{safe(t,100)}'")
            except Exception as e:
                print(f"  Btn {i}: err {e}")

        # Try various start/fight button patterns
        start_candidates = ["START GAME", "Start Game", "START", "Start", "FIGHT!", "FIGHT", "BEGIN", "PLAY"]
        for txt in start_candidates:
            try:
                loc = page.locator("button", has_text=txt).first
                if loc.count() > 0 and loc.is_visible():
                    print(f"Clicking start button: '{txt}'")
                    loc.click(timeout=2000)
                    page.wait_for_timeout(2500)
                    break
            except Exception:
                continue

        # If still no game started, the WAITING state may need ENTER or SPACE on overlay
        # The EnhancedGameUI component likely has handleStartGame triggered by a "READY" / "FIGHT" overlay button
        # Wait extra for FIGHT! announcement to clear
        page.wait_for_timeout(2500)
        dump(page, "after start attempt")

        # Click into game area to ensure focus
        try:
            page.locator("body").click(position={"x": 640, "y": 400}, force=True)
        except Exception:
            pass
        page.wait_for_timeout(400)

        # Idle screenshot
        idle_path = os.path.join(OUT_DIR, "sprite_idle.png")
        page.screenshot(path=idle_path, full_page=False)
        print(f"Saved: {idle_path}")

        # Kick: 'c' = right_kick (Player 1)
        print("Pressing 'c' for right_kick")
        page.keyboard.down("c")
        page.wait_for_timeout(140)
        kick_path = os.path.join(OUT_DIR, "sprite_kick.png")
        page.screenshot(path=kick_path, full_page=False)
        page.keyboard.up("c")
        print(f"Saved: {kick_path}")
        page.wait_for_timeout(700)

        # Punch: 'e' = right_punch
        print("Pressing 'e' for right_punch")
        page.keyboard.down("e")
        page.wait_for_timeout(120)
        punch_path = os.path.join(OUT_DIR, "sprite_punch.png")
        page.screenshot(path=punch_path, full_page=False)
        page.keyboard.up("e")
        print(f"Saved: {punch_path}")
        page.wait_for_timeout(700)

        # Jump: 'w'
        print("Pressing 'w' for jump")
        page.keyboard.down("w")
        page.wait_for_timeout(220)
        jump_path = os.path.join(OUT_DIR, "sprite_jump.png")
        page.screenshot(path=jump_path, full_page=False)
        page.keyboard.up("w")
        print(f"Saved: {jump_path}")

        browser.close()
        print("DONE")

if __name__ == "__main__":
    main()
