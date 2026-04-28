import { useEffect, useRef, useCallback } from 'react';
import type { ActionType, PlayerInput } from '../types/game';

// Key mappings for both players - make them immutable
const PLAYER1_KEYS: Readonly<Record<string, ActionType>> = Object.freeze({
  'w': 'jump', 'W': 'jump',
  'a': 'move_backward', 'A': 'move_backward',
  'd': 'move_forward', 'D': 'move_forward',
  's': 'block', 'S': 'block',
  'q': 'left_punch', 'Q': 'left_punch',
  'e': 'right_punch', 'E': 'right_punch',
  'z': 'left_kick', 'Z': 'left_kick',
  'c': 'right_kick', 'C': 'right_kick',
});

const PLAYER2_KEYS: Readonly<Record<string, ActionType>> = Object.freeze({
  'ArrowUp': 'jump', 'ArrowLeft': 'move_backward',
  'ArrowRight': 'move_forward', 'ArrowDown': 'block',
  'Numpad4': 'left_punch', 'Numpad6': 'right_punch',
  'Numpad1': 'left_kick', 'Numpad3': 'right_kick',
  // Alternative keys for player 2 (useful when no numpad)
  'i': 'jump', 'I': 'jump',
  'j': 'move_backward', 'J': 'move_backward',
  'l': 'move_forward', 'L': 'move_forward',
  'k': 'block', 'K': 'block',
  'u': 'left_punch', 'U': 'left_punch',
  'o': 'right_punch', 'O': 'right_punch',
  'n': 'left_kick', 'N': 'left_kick',
  'm': 'right_kick', 'M': 'right_kick',
});

interface UseKeyboardInputOptions {
  onInput: (input: PlayerInput) => void;
  enabled?: boolean;
  player1Enabled?: boolean;
  player2Enabled?: boolean;
}

// Optimize key resolution with early return
function resolveAction(keys: string[], map: Readonly<Record<string, ActionType>>): ActionType {
  for (let i = 0; i < keys.length; i++) {
    const action = map[keys[i]];
    if (action) return action;
  }
  return 'idle';
}

export function useKeyboardInput({
  onInput,
  enabled = true,
  player1Enabled = true,
  player2Enabled = true,
}: UseKeyboardInputOptions) {
  const pressedKeysRef = useRef<Set<string>>(new Set());
  const animationFrameRef = useRef<number | null>(null);
  const lastActionsRef = useRef<{ player1: ActionType; player2: ActionType }>({
    player1: 'idle',
    player2: 'idle',
  });
  const enabledPlayersRef = useRef({ player1: player1Enabled, player2: player2Enabled });
  const onInputRef = useRef(onInput);
  
  // Update ref when callback changes
  useEffect(() => {
    onInputRef.current = onInput;
  }, [onInput]);

  useEffect(() => {
    enabledPlayersRef.current = { player1: player1Enabled, player2: player2Enabled };

    if (!player1Enabled && lastActionsRef.current.player1 !== 'idle') {
      onInputRef.current({ playerId: 1, action: 'idle' });
      lastActionsRef.current.player1 = 'idle';
    }

    if (!player2Enabled && lastActionsRef.current.player2 !== 'idle') {
      onInputRef.current({ playerId: 2, action: 'idle' });
      lastActionsRef.current.player2 = 'idle';
    }
  }, [player1Enabled, player2Enabled]);
  
  const processKeys = useCallback(() => {
    const orderedKeys = Array.from(pressedKeysRef.current).reverse();

    if (enabledPlayersRef.current.player1) {
      const p1Action = resolveAction(orderedKeys, PLAYER1_KEYS);
      if (p1Action !== lastActionsRef.current.player1) {
        onInputRef.current({ playerId: 1, action: p1Action });
        lastActionsRef.current.player1 = p1Action;
      }
    }

    if (enabledPlayersRef.current.player2) {
      const p2Action = resolveAction(orderedKeys, PLAYER2_KEYS);
      if (p2Action !== lastActionsRef.current.player2) {
        onInputRef.current({ playerId: 2, action: p2Action });
        lastActionsRef.current.player2 = p2Action;
      }
    }
  }, []);
  
  useEffect(() => {
    if (!enabled) {
      pressedKeysRef.current.clear();
      lastActionsRef.current.player1 = 'idle';
      lastActionsRef.current.player2 = 'idle';
      return;
    }
    
    const handleKeyDown = (e: KeyboardEvent) => {
      const key = e.key;
      const isPlayer1Key = player1Enabled && PLAYER1_KEYS[key];
      const isPlayer2Key = player2Enabled && PLAYER2_KEYS[key];
      if (isPlayer1Key || isPlayer2Key) {
        e.preventDefault();
        pressedKeysRef.current.add(key);
      }
    };
    
    const handleKeyUp = (e: KeyboardEvent) => {
      const key = e.key;
      pressedKeysRef.current.delete(key);
      // Also delete the opposite case
      pressedKeysRef.current.delete(key.toLowerCase());
      pressedKeysRef.current.delete(key.toUpperCase());
    };

    const handleWindowBlur = () => {
      pressedKeysRef.current.clear();
    };
    
    // Continuous input processing loop - throttled
    let lastFrameTime = 0;
    const inputLoop = (currentTime: number) => {
      // Throttle to 120fps for input processing
      if (currentTime - lastFrameTime >= 8.33) {
        processKeys();
        lastFrameTime = currentTime;
      }
      animationFrameRef.current = requestAnimationFrame(inputLoop);
    };
    
    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    window.addEventListener('blur', handleWindowBlur);
    
    // Start input processing loop
    animationFrameRef.current = requestAnimationFrame(inputLoop);
    
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
      window.removeEventListener('blur', handleWindowBlur);
      if (animationFrameRef.current !== null) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [enabled, player1Enabled, player2Enabled, processKeys]);
}

// Export key mappings for display
export const KEY_MAPPINGS = {
  player1: {
    jump: 'W',
    move_forward: 'D',
    move_backward: 'A',
    block: 'S',
    left_punch: 'Q',
    right_punch: 'E',
    left_kick: 'Z',
    right_kick: 'C',
  },
  player2: {
    jump: 'I / ↑',
    move_forward: 'L / →',
    move_backward: 'J / ←',
    block: 'K / ↓',
    left_punch: 'U',
    right_punch: 'O',
    left_kick: 'N',
    right_kick: 'M',
  },
};
