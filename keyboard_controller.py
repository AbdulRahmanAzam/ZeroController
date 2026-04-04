"""
Keyboard Controller Module
==========================
Translates detected move names into keyboard key presses using pynput.
Supports press-and-release or hold-based key actions.
"""

import time
import threading
from pynput.keyboard import Controller, Key


# Map special key names to pynput Key objects
SPECIAL_KEYS = {
    "up": Key.up,
    "down": Key.down,
    "left": Key.left,
    "right": Key.right,
    "space": Key.space,
    "enter": Key.enter,
    "shift": Key.shift,
    "ctrl": Key.ctrl,
    "alt": Key.alt,
    "esc": Key.esc,
    "tab": Key.tab,
    "num0": Key.insert,  # Numpad mapping varies; adjust as needed
    "num1": Key.end,
    "num2": Key.down,
    "num4": Key.left,
    "num5": Key.clear if hasattr(Key, 'clear') else Key.down,
    "num6": Key.right,
}


class KeyboardController:
    """
    Handles pressing and releasing keyboard keys based on detected moves.
    
    Args:
        key_map: dict mapping move names to key strings
        hold_duration: how long to hold a key press (seconds)
    """

    def __init__(self, key_map, hold_duration=0.08):
        self.keyboard = Controller()
        self.key_map = key_map
        self.hold_duration = hold_duration
        self._active_keys = set()
        self._lock = threading.Lock()

    def _resolve_key(self, key_str):
        """Convert a key string to a pynput key object."""
        if key_str in SPECIAL_KEYS:
            return SPECIAL_KEYS[key_str]
        if len(key_str) == 1:
            return key_str  # Single character key
        # Try as attribute of Key enum
        try:
            return getattr(Key, key_str)
        except AttributeError:
            return key_str[0]  # Fallback to first character

    def press_move(self, move_name):
        """
        Press the key mapped to a move, hold briefly, then release.
        Runs in a thread to avoid blocking the main loop.
        """
        if move_name not in self.key_map:
            return

        key_str = self.key_map[move_name]
        key = self._resolve_key(key_str)

        def _press_and_release():
            with self._lock:
                if key in self._active_keys:
                    return  # Already being pressed
                self._active_keys.add(key)

            try:
                self.keyboard.press(key)
                time.sleep(self.hold_duration)
                self.keyboard.release(key)
            finally:
                with self._lock:
                    self._active_keys.discard(key)

        thread = threading.Thread(target=_press_and_release, daemon=True)
        thread.start()

    def press_moves(self, move_list):
        """Press keys for all moves in the list."""
        for move in move_list:
            self.press_move(move)

    def release_all(self):
        """Release all currently held keys."""
        with self._lock:
            for key in list(self._active_keys):
                try:
                    self.keyboard.release(key)
                except Exception:
                    pass
            self._active_keys.clear()
