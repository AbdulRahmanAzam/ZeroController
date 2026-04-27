import { useEffect, useRef } from 'react';
import { useGameStore } from '../store/gameStore';
import { getAIAction, DIFFICULTY_TICK_MS } from '../game/aiController';
import type { AIDifficulty } from '../types/game';
import type { PlayerInput } from '../types/game';

interface UseAIInputOptions {
  onInput: (input: PlayerInput) => void;
  enabled: boolean;
  difficulty: AIDifficulty;
}

/**
 * Drives Player 2 with AI decisions.
 *
 * Produces exactly the same PlayerInput objects that useKeyboardInput produces,
 * injected at a difficulty-dependent interval. No stat modifications — purely
 * a decision-making layer on top of the identical physics engine.
 */
export function useAIInput({ onInput, enabled, difficulty }: UseAIInputOptions) {
  // Refs that persist across ticks without causing re-renders
  const comboIndexRef      = useRef(0);
  const prevP1AttackingRef = useRef(false);
  const lastActionRef      = useRef<string>('idle');
  const onInputRef         = useRef(onInput);

  // Keep callback ref fresh without re-triggering the interval effect
  useEffect(() => {
    onInputRef.current = onInput;
  }, [onInput]);

  // Reset AI state at the start of every new round
  useEffect(() => {
    if (enabled) {
      comboIndexRef.current      = 0;
      prevP1AttackingRef.current = false;
      lastActionRef.current      = 'idle';
    }
  }, [enabled]);

  useEffect(() => {
    if (!enabled) return;

    const tickMs = DIFFICULTY_TICK_MS[difficulty];

    const intervalId = window.setInterval(() => {
      // Always read fresh state — avoids stale closure trap
      const state = useGameStore.getState();

      if (state.gameStatus !== 'playing') return;

      const refs = {
        comboIndex:      comboIndexRef.current,
        prevP1Attacking: prevP1AttackingRef.current,
      };

      const action = getAIAction(state, difficulty, refs);

      // Write back mutated refs
      comboIndexRef.current      = refs.comboIndex;
      prevP1AttackingRef.current = refs.prevP1Attacking;

      // Only emit when the action actually changes (delta-based, same pattern
      // as useKeyboardInput) to avoid spamming setPlayerAction
      if (action !== lastActionRef.current) {
        lastActionRef.current = action;
        onInputRef.current({ playerId: 2, action });
      }
    }, tickMs);

    return () => window.clearInterval(intervalId);
  }, [enabled, difficulty]);
}
