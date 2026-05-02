"""Camera utilities for reliable webcam startup on Windows."""

import cv2

from config import (
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    CAMERA_TARGET_FPS,
    CAMERA_BUFFER_SIZE,
    CAMERA_READ_WARMUP_FRAMES,
    CAMERA_FOURCC,
)


def _backend_flag(name):
    backend = (name or "").lower()
    if backend == "dshow":
        return cv2.CAP_DSHOW
    if backend == "msmf":
        return cv2.CAP_MSMF
    return None


def open_camera(index, backend_name):
    """Open a camera using one backend and apply common capture properties."""
    flag = _backend_flag(backend_name)
    if flag is None:
        cap = cv2.VideoCapture(index)
    else:
        cap = cv2.VideoCapture(index, flag)

    if CAMERA_FOURCC:
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*CAMERA_FOURCC))
        except Exception:
            pass
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_TARGET_FPS)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFER_SIZE)
    except Exception:
        pass
    return cap


def open_camera_with_fallback(index, preferred_backend):
    """Try preferred backend first, then common fallbacks, returning the first working camera."""
    candidates = [preferred_backend, "msmf", "dshow", "auto"]
    seen = set()
    ordered = []
    for name in candidates:
        key = (name or "auto").lower()
        if key not in seen:
            seen.add(key)
            ordered.append(key)

    for backend in ordered:
        cap = open_camera(index, backend)
        if not cap.isOpened():
            cap.release()
            continue

        valid = False
        for _ in range(CAMERA_READ_WARMUP_FRAMES):
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                valid = True
                break

        if valid:
            return cap, backend

        cap.release()

    return None, None
