from playwright.sync_api import sync_playwright
import os, time, sys

OUT = r"C:\Users\azama\VS Code\PROJECTS\01 Academic and Tutorial Projects\zeroController"
URL = "http://localhost:5175/"

def safeprint(*args):
    try:
        msg = " ".join(str(a) for a in args)
        sys.stdout.buffer.write((msg + "\n").encode("utf-8", errors="replace"))
        sys.stdout.flush()
    except Exception:
        pass

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1280, "height": 800})
        page = ctx.new_page()

        page.goto(URL, wait_until="networkidle")
        time.sleep(2.0)
        page.reload(wait_until="networkidle")
        time.sleep(3.5)

        # Step 1 (mode select): click VS PLAYER - text contains "VS PLAYER"
        try:
            page.locator("button", has_text="VS PLAYER").first.click(force=True, timeout=8000)
            safeprint("VS PLAYER click OK")
        except Exception as e:
            safeprint("VS PLAYER err:", e)
        time.sleep(2.0)
        page.screenshot(path=os.path.join(OUT, "pivot2_after_mode.png"), full_page=False)

        # Step 2 (controller select): click "KEYBOARD" - need exact match because "ZERO CONTROLLER" doesn't include the word KEYBOARD but "← BACK" doesn't either
        # Use the icon prefix: locator with KEYBOARD but excluding TWO PLAYERS (no longer present here anyway).
        try:
            page.locator("button", has_text="KEYBOARD").first.click(force=True, timeout=8000)
            safeprint("KEYBOARD click OK")
        except Exception as e:
            safeprint("KEYBOARD err:", e)
        time.sleep(3.5)
        page.screenshot(path=os.path.join(OUT, "pivot2_after_kb.png"), full_page=False)

        # Now we should be in the game / Battle Arena pre-fight overlay. Click FIGHT NOW if present.
        try:
            page.locator("button", has_text="FIGHT NOW").first.click(force=True, timeout=10000)
            safeprint("FIGHT NOW click OK")
        except Exception as e:
            safeprint("FIGHT NOW err (may be auto-started):", str(e)[:200])
        time.sleep(2.5)
        page.screenshot(path=os.path.join(OUT, "pivot2_after_fight.png"), full_page=False)

        # Wait for any countdown
        time.sleep(3.0)
        page.screenshot(path=os.path.join(OUT, "pivot2_debug_post.png"), full_page=False)

        # Diagnostic page text
        try:
            txt = page.evaluate("() => document.body.innerText || ''")
            safeprint("FINAL_PAGE_TEXT_LEN:", len(txt))
            safeprint(txt[:800])
        except Exception:
            pass

        # Focus
        try:
            page.click("body")
        except Exception:
            pass
        time.sleep(0.3)

        page.screenshot(path=os.path.join(OUT, "pivot2_idle.png"), full_page=False)
        safeprint("saved idle")

        # Run: hold D ~600ms, capture mid (~300ms in)
        page.keyboard.down("d")
        time.sleep(0.3)
        page.screenshot(path=os.path.join(OUT, "pivot2_run.png"), full_page=False)
        time.sleep(0.3)
        page.keyboard.up("d")
        safeprint("saved run")
        time.sleep(0.6)

        # Punch: press Q
        page.keyboard.down("q")
        time.sleep(0.05)
        page.screenshot(path=os.path.join(OUT, "pivot2_punch.png"), full_page=False)
        page.keyboard.up("q")
        safeprint("saved punch")
        time.sleep(0.8)

        # Kick: press C
        page.keyboard.down("c")
        time.sleep(0.05)
        page.screenshot(path=os.path.join(OUT, "pivot2_kick.png"), full_page=False)
        page.keyboard.up("c")
        safeprint("saved kick")
        time.sleep(0.8)

        # Jump: press W, wait 200ms peak
        page.keyboard.down("w")
        time.sleep(0.02)
        page.keyboard.up("w")
        time.sleep(0.2)
        page.screenshot(path=os.path.join(OUT, "pivot2_jump.png"), full_page=False)
        safeprint("saved jump")

        browser.close()

if __name__ == "__main__":
    main()
