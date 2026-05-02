import asyncio
import os
from playwright.async_api import async_playwright

OUT = r"C:\Users\azama\VS Code\PROJECTS\01 Academic and Tutorial Projects\zeroController"
URL = "http://localhost:5175/"

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(viewport={"width": 1280, "height": 800})
        page = await ctx.new_page()
        await page.goto(URL, wait_until="networkidle", timeout=20000)
        await page.wait_for_timeout(1500)

        async def click_text(label):
            for sel in [f"text={label}"]:
                try:
                    btn = await page.query_selector(sel)
                    if btn:
                        await btn.click()
                        print("clicked", label)
                        return True
                except Exception:
                    pass
            return False

        await click_text("Play")
        await page.wait_for_timeout(700)
        await click_text("KEYBOARD")
        await page.wait_for_timeout(900)
        # FIGHT NOW
        for sel in ["text=FIGHT NOW", "text=Fight Now", "button:has-text('FIGHT')"]:
            try:
                btn = await page.query_selector(sel)
                if btn:
                    await btn.click()
                    print("clicked FIGHT NOW")
                    break
            except Exception:
                pass
        await page.wait_for_timeout(2500)

        # focus canvas
        try:
            cv = await page.query_selector("canvas")
            if cv:
                box = await cv.bounding_box()
                print("canvas:", box)
                await page.mouse.click(box["x"] + box["width"]/2, box["y"] + box["height"]/2)
        except Exception:
            pass
        await page.wait_for_timeout(700)

        # idle
        await page.wait_for_timeout(1800)
        await page.screenshot(path=os.path.join(OUT, "pivot_idle.png"))
        print("idle")

        # run: hold D for ~1s, capture mid
        await page.keyboard.down("KeyD")
        await page.wait_for_timeout(500)
        await page.screenshot(path=os.path.join(OUT, "pivot_run.png"))
        await page.wait_for_timeout(300)
        await page.keyboard.up("KeyD")
        await page.wait_for_timeout(700)
        print("run")

        # kick: C (heavy kick) - take one frame quickly
        await page.keyboard.down("KeyC")
        await page.wait_for_timeout(70)
        await page.keyboard.up("KeyC")
        await page.wait_for_timeout(110)
        await page.screenshot(path=os.path.join(OUT, "pivot_kick.png"))
        await page.wait_for_timeout(700)
        print("kick")

        # punch: E (heavy punch)
        await page.keyboard.down("KeyE")
        await page.wait_for_timeout(70)
        await page.keyboard.up("KeyE")
        await page.wait_for_timeout(110)
        await page.screenshot(path=os.path.join(OUT, "pivot_punch.png"))
        await page.wait_for_timeout(700)
        print("punch")

        # jump: W
        await page.keyboard.down("KeyW")
        await page.wait_for_timeout(60)
        await page.keyboard.up("KeyW")
        await page.wait_for_timeout(220)
        await page.screenshot(path=os.path.join(OUT, "pivot_jump.png"))
        await page.wait_for_timeout(900)
        print("jump")

        # guard: S (block)
        await page.keyboard.down("KeyS")
        await page.wait_for_timeout(180)
        await page.screenshot(path=os.path.join(OUT, "pivot_guard.png"))
        await page.keyboard.up("KeyS")
        print("guard")

        await browser.close()

asyncio.run(main())
