"""Shared utilities used by collect_data.py and run_model.py.

Exports:
    POSE_CONNECTIONS  – list of (a, b) index pairs for MediaPipe's 33-landmark skeleton.
    ensure_pose_model – download the .task model file if it is not already present.
"""

import os
import urllib.request

# ── MediaPipe 33-landmark skeleton connections ────────────────────────────────
# Mirrors mediapipe.solutions.pose.POSE_CONNECTIONS (Tasks API no longer
# exposes this constant directly).
POSE_CONNECTIONS = [
    # Face
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    # Torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Left arm
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19),
    # Right arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (18, 20),
    # Left leg
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    # Right leg
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]


# ── Model downloader ──────────────────────────────────────────────────────────

def ensure_pose_model(model_path: str, model_url: str) -> None:
    """Download the MediaPipe .task model file if it does not already exist.

    Args:
        model_path: Local destination path (e.g. ``"models/pose_landmarker_full.task"``).
        model_url:  Remote URL to download from when the file is absent.
    """
    if os.path.exists(model_path):
        return

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    print(f"[MODEL] Downloading pose model → {model_path}")
    print(f"[MODEL] Source: {model_url}")

    try:
        urllib.request.urlretrieve(model_url, model_path, _download_progress)
        print()  # newline after progress dots
        print(f"[MODEL] Saved to {model_path}")
    except Exception as exc:
        # Clean up a partial download so the next run retries cleanly.
        if os.path.exists(model_path):
            os.remove(model_path)
        raise RuntimeError(
            f"Failed to download pose model from {model_url}: {exc}"
        ) from exc


def _download_progress(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        bar_len = 40
        filled = int(bar_len * pct / 100)
        bar = "#" * filled + "-" * (bar_len - filled)
        print(f"\r  [{bar}] {pct:5.1f}%", end="", flush=True)
    else:
        mb = downloaded / (1024 * 1024)
        print(f"\r  {mb:.1f} MB downloaded…", end="", flush=True)
