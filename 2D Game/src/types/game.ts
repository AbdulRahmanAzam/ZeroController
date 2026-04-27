// AI difficulty and game mode
export type AIDifficulty = 'easy' | 'medium' | 'hard';
export type GameMode = 'vs_player' | 'vs_ai';

// Game action types - the 9 movements
export type ActionType =
  | 'idle'
  | 'move_forward'
  | 'move_backward'
  | 'jump'
  | 'block'
  | 'left_punch'
  | 'right_punch'
  | 'left_kick'
  | 'right_kick';

// Player state interface
export interface PlayerState {
  id: 1 | 2;
  x: number;
  y: number;
  velocityX: number;
  velocityY: number;
  health: number;
  maxHealth: number;
  action: ActionType;
  facingRight: boolean;
  isGrounded: boolean;
  isBlocking: boolean;
  isAttacking: boolean;
  attackCooldown: number;
  attackTimer: number; // Time elapsed in current attack animation
  hitCooldown: number;
  comboCount: number;
  comboTimer: number;
  specialMeter: number;
}

// Game state interface
export interface GameState {
  player1: PlayerState;
  player2: PlayerState;
  gameStatus: 'waiting' | 'playing' | 'paused' | 'round_end' | 'match_end';
  winner: 1 | 2 | 'draw' | null;
  roundTime: number;
  round: number;
  maxRounds: number;
  player1RoundsWon: number;
  player2RoundsWon: number;
  roundWinner: 1 | 2 | 'draw' | null;
}

// Input commands from external source (camera/keyboard)
export interface PlayerInput {
  playerId: 1 | 2;
  action: ActionType;
}

// Game configuration
export interface GameConfig {
  canvasWidth: number;
  canvasHeight: number;
  groundY: number;
  gravity: number;
  moveSpeed: number;
  jumpForce: number;
  playerWidth: number;
  playerHeight: number;
  attackDamage: {
    punch: number;
    kick: number;
  };
  attackRange: number;
  attackCooldownTime: number;
  hitCooldownTime: number;
  roundDuration: number;
}

// Animation frame data
export interface AnimationFrame {
  name: ActionType;
  frameCount: number;
  frameDuration: number;
  loop: boolean;
}

// --- Recommendation / Analytics types ---

/**
 * Snapshot of the full game state at the moment an attack action is initiated.
 * All raw values are stored here; the analytics server normalises them into
 * ML-ready features (0–1 range) before writing to the training_events collection.
 */
export interface StateSnapshot {
  p1_x: number;              p1_y: number;
  p2_x: number;              p2_y: number;
  p1_health: number;         p2_health: number;
  p1_isAttacking: boolean;   p2_isAttacking: boolean;
  p1_isBlocking: boolean;    p2_isBlocking: boolean;
  p1_attackCooldown: number; p2_attackCooldown: number;
  p1_hitCooldown: number;    p2_hitCooldown: number;
  roundTimeRemaining: number;
  round: number;
  canvasWidth: number;
  canvasHeight: number;
}

/** A single logged attack attempt within a round */
export interface ActionEvent {
  timestamp: number;        // ms since round start
  player: 1 | 2;
  action: ActionType;
  succeeded: boolean;       // true if the hit landed (filled in retroactively)
  damageDealt: number;
  wasBlocked: boolean;
  /** Game state when the attack was initiated — the core feature vector for ML training */
  stateSnapshot?: StateSnapshot;
}

/** Full statistics captured at the end of every round */
export interface RoundStatistics {
  roundNumber: number;
  winner: 1 | 2 | 'draw';
  durationMs: number;
  finalHealth: { p1: number; p2: number };
  totalDamageDealt: { p1: number; p2: number };
  attacksAttempted: { p1: number; p2: number };
  attacksLanded: { p1: number; p2: number };
  blocksPerformed: { p1: number; p2: number };
  highestCombo: { p1: number; p2: number };
  actionLog: ActionEvent[];
}

/** History of all rounds in the current match */
export interface MatchHistory {
  rounds: RoundStatistics[];
  matchWinner: 1 | 2 | 'draw' | null;
}
