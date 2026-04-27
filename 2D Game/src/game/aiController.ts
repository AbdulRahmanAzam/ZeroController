/**
 * AI Controller for Player 2
 *
 * The AI has IDENTICAL stats, speed, damage, and cooldowns to Player 1.
 * The ONLY difference between difficulty levels is decision-making quality
 * and reaction speed (tick interval). No stat advantages are given to any
 * difficulty level — the AI wins or loses purely based on how well it reads
 * and reacts to the game state.
 *
 * Easy   (600ms tick): slow, random, barely reacts
 * Medium (150ms tick): reactive, blocks ~50% of attacks, punishes sometimes
 * Hard    (50ms tick): near-instant reads, blocks 95% of attacks, chains combos
 */

import type { GameState, ActionType, AIDifficulty } from '../types/game';
import { GAME_CONFIG } from './config';

export type { AIDifficulty } from '../types/game';

/** How often the AI makes a new decision per difficulty */
export const DIFFICULTY_TICK_MS: Record<AIDifficulty, number> = {
  easy: 600,
  medium: 150,
  hard: 50,
};

/** Mutable refs passed in from the hook to carry state between ticks */
export interface AIRefs {
  comboIndex: number;
  prevP1Attacking: boolean;
}

// Attack-only actions (no movement or blocking)
const ATTACK_ACTIONS: ActionType[] = [
  'left_punch', 'right_punch', 'left_kick', 'right_kick',
];

// All possible actions
const ALL_ACTIONS: ActionType[] = [
  'idle', 'move_forward', 'move_backward', 'jump',
  'block', 'left_punch', 'right_punch', 'left_kick', 'right_kick',
];

/**
 * Hard AI combo cycle — the AI loops through this sequence when in melee
 * range and not blocking/approaching. Pure skill expression; same attacks
 * any human could perform.
 */
const HARD_COMBO_SEQUENCE: ActionType[] = [
  'left_punch', 'right_punch', 'left_punch',
  'left_kick',  'right_punch', 'right_kick',
];

function randomFrom<T>(arr: T[]): T {
  return arr[Math.floor(Math.random() * arr.length)];
}

function randomAttack(): ActionType {
  return randomFrom(ATTACK_ACTIONS);
}

/**
 * Returns true if P2 is within melee range of P1.
 * Uses the same attackRange calculation the game engine uses.
 */
function isInRange(state: GameState): boolean {
  const { player1: p1, player2: p2 } = state;
  const centerP1 = p1.x + GAME_CONFIG.playerWidth / 2;
  const centerP2 = p2.x + GAME_CONFIG.playerWidth / 2;
  const distance = Math.abs(centerP1 - centerP2);
  // Slightly generous — same heuristic a human would use
  return distance <= GAME_CONFIG.attackRange + GAME_CONFIG.playerWidth * 0.9;
}

/** Returns the action P2 should take to move toward P1.
 *
 * move_forward  = always moves RIGHT (+X)
 * move_backward = always moves LEFT  (-X)
 * P2 starts on the right, so to approach P1 on the left → move_backward.
 */
function approachP1(state: GameState): ActionType {
  const { player1: p1, player2: p2 } = state;
  return p2.x > p1.x ? 'move_backward' : 'move_forward';
}

// ---------------------------------------------------------------------------
// Easy AI
// ---------------------------------------------------------------------------
function getEasyAction(
  state: GameState,
  _refs: AIRefs,
): ActionType {
  const { player2: p2 } = state;
  const inRange = isInRange(state);

  // 35%: completely random — produces silly, unpredictable behaviour
  if (Math.random() < 0.35) return randomFrom(ALL_ACTIONS);

  // Can't attack right now (cooldown)
  const canAttack = p2.attackCooldown <= 0 && p2.hitCooldown <= 0;
  if (!canAttack) return 'idle';

  if (inRange) {
    const r = Math.random();
    if (r < 0.50) return randomAttack(); // 50% random attack
    if (r < 0.65) return 'block';        // 15% block
    return 'idle';                        // 35% idle (slow reaction)
  } else {
    const r = Math.random();
    if (r < 0.70) return approachP1(state); // slowly walk toward P1
    if (r < 0.75) return 'jump';
    return 'idle';
  }
}

// ---------------------------------------------------------------------------
// Medium AI
// ---------------------------------------------------------------------------
function getMediumAction(
  state: GameState,
  refs: AIRefs,
): ActionType {
  const { player1: p1, player2: p2 } = state;
  const inRange = isInRange(state);

  // 15%: random (still makes mistakes)
  if (Math.random() < 0.15) return randomFrom(ALL_ACTIONS);

  const canAttack = p2.attackCooldown <= 0 && p2.hitCooldown <= 0;

  // Can't attack — may still block
  if (!canAttack) {
    if (p1.isAttacking && inRange && Math.random() < 0.50) return 'block';
    return 'idle';
  }

  // React to P1 attacking
  if (p1.isAttacking && inRange) {
    const r = Math.random();
    if (r < 0.50) return 'block';       // block half the time
    return randomAttack();               // trade blows the other half
  }

  // Punish: P1 just finished an attack animation
  const justFinished = refs.prevP1Attacking && !p1.isAttacking;
  if (justFinished && inRange) {
    if (Math.random() < 0.70) return randomAttack();
  }

  if (inRange) {
    const r = Math.random();
    if (r < 0.65) return randomAttack();
    if (r < 0.85) return 'block';
    return 'idle';
  } else {
    const r = Math.random();
    if (r < 0.85) return approachP1(state);
    if (r < 0.92) return 'jump';
    return 'idle';
  }
}

// ---------------------------------------------------------------------------
// Hard AI
// ---------------------------------------------------------------------------
function getHardAction(
  state: GameState,
  refs: AIRefs,
): ActionType {
  const { player1: p1, player2: p2 } = state;
  const inRange = isInRange(state);

  // 3%: tiny noise — makes it feel elite, not robotic-impossible
  if (Math.random() < 0.03) return randomFrom(ALL_ACTIONS);

  const canAttack = p2.attackCooldown <= 0 && p2.hitCooldown <= 0;

  // Respect P2's own cooldowns first
  if (!canAttack) {
    // Can still block even while in cooldown
    if (p1.isAttacking && inRange && p2.hitCooldown <= 0) return 'block';
    return 'idle';
  }

  // PRIORITY 1 — Block incoming attack (95% reaction rate)
  // This is the core "impossibility" of Hard: the AI almost never eats a hit
  if (p1.isAttacking && inRange) {
    if (Math.random() < 0.95) return 'block';
  }

  // PRIORITY 2 — Punish: P1 just stopped attacking → immediate counter combo
  const justFinished = refs.prevP1Attacking && !p1.isAttacking;
  if (justFinished && inRange) {
    const comboAction = HARD_COMBO_SEQUENCE[refs.comboIndex % HARD_COMBO_SEQUENCE.length];
    refs.comboIndex++;
    return comboAction;
  }

  // PRIORITY 3 — Anti-corner: escape if P2 is pinned against a wall
  const edgeThreshold = GAME_CONFIG.canvasWidth * 0.06;
  const nearRightEdge = p2.x + GAME_CONFIG.playerWidth >= GAME_CONFIG.canvasWidth - edgeThreshold;
  const nearLeftEdge  = p2.x <= edgeThreshold;
  if ((nearRightEdge || nearLeftEdge) && !p1.isAttacking) {
    return 'jump'; // hop over / repositions
  }

  // PRIORITY 4 — Approach P1 if out of range
  if (!inRange) {
    return approachP1(state);
  }

  // PRIORITY 5 — In-range aggression: cycle combo sequence
  const comboAction = HARD_COMBO_SEQUENCE[refs.comboIndex % HARD_COMBO_SEQUENCE.length];
  refs.comboIndex++;
  return comboAction;
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/**
 * Returns the action Player 2 (AI) should perform this tick.
 *
 * @param state      Current game state (fresh from store)
 * @param difficulty AI difficulty level
 * @param refs       Mutable ref object shared across ticks by useAIInput
 */
export function getAIAction(
  state: GameState,
  difficulty: AIDifficulty,
  refs: AIRefs,
): ActionType {
  let action: ActionType;

  switch (difficulty) {
    case 'easy':
      action = getEasyAction(state, refs);
      break;
    case 'medium':
      action = getMediumAction(state, refs);
      break;
    case 'hard':
      action = getHardAction(state, refs);
      break;
  }

  // Update prevP1Attacking AFTER the decision so the next tick can detect
  // the transition from attacking → not attacking (punish window).
  refs.prevP1Attacking = state.player1.isAttacking;

  return action;
}
