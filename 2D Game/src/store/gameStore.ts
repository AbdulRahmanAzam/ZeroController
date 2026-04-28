import { create } from 'zustand';
import type { GameState, PlayerState, ActionType, ActionEvent, RoundStatistics, MatchHistory, AIDifficulty, GameMode, PlayerOneControlMode } from '../types/game';
import { GAME_CONFIG, ANIMATIONS } from '../game/config';
import { SoundManager } from '../audio/SoundManager';

// Create initial player state
function createInitialPlayerState(playerId: 1 | 2): PlayerState {
  // Dynamic positioning based on viewport
  const canvasWidth = GAME_CONFIG.canvasWidth;
  const groundY = GAME_CONFIG.groundY;
  const playerHeight = GAME_CONFIG.playerHeight;
  
  const startX = playerId === 1 
    ? canvasWidth * 0.2  // 20% from left
    : canvasWidth * 0.8 - GAME_CONFIG.playerWidth; // 80% from left minus player width
    
  return {
    id: playerId,
    x: startX,
    y: groundY - playerHeight,
    velocityX: 0,
    velocityY: 0,
    health: 100,
    maxHealth: 100,
    action: 'idle',
    facingRight: playerId === 1,
    isGrounded: true,
    isBlocking: false,
    isAttacking: false,
    attackCooldown: 0,
    attackTimer: 0,
    hitCooldown: 0,
    comboCount: 0,
    comboTimer: 0,
    specialMeter: 0,
  };
}

interface GameStore extends GameState {
  // AI / game mode settings (persist across rounds and resets)
  gameMode: GameMode;
  aiDifficulty: AIDifficulty;
  player1ControlMode: PlayerOneControlMode;
  setGameMode: (mode: GameMode) => void;
  setAIDifficulty: (difficulty: AIDifficulty) => void;
  setPlayer1ControlMode: (mode: PlayerOneControlMode) => void;
  /** Unique ID for the current match — persists across rounds, cleared on reset */
  sessionId: string;

  // Actions
  startGame: () => void;
  pauseGame: () => void;
  resumeGame: () => void;
  resetGame: () => void;
  startNextRound: () => void;
  setPlayerAction: (playerId: 1 | 2, action: ActionType) => void;
  updateGame: (deltaTime: number) => void;
  applyDamage: (targetId: 1 | 2, damage: number) => void;
  
  // UI State
  showHitEffect: { player: 1 | 2; type: 'hit' | 'block' } | null;
  setHitEffect: (effect: { player: 1 | 2; type: 'hit' | 'block' } | null) => void;
  
  // Screen shake
  screenShake: number;
  triggerScreenShake: (intensity: number) => void;

  // Analytics / Recommendation data layer
  matchHistory: MatchHistory;
  currentRoundLog: ActionEvent[];
  roundStartTime: number;
  peakCombo: { p1: number; p2: number };
  blocksPerformed: { p1: number; p2: number };
}

export const useGameStore = create<GameStore>((set, get) => ({
  // Initial state
  player1: createInitialPlayerState(1),
  player2: createInitialPlayerState(2),
  gameStatus: 'waiting',
  winner: null,
  roundTime: GAME_CONFIG.roundDuration,
  round: 1,
  maxRounds: 3,
  player1RoundsWon: 0,
  player2RoundsWon: 0,
  roundWinner: null,
  showHitEffect: null,
  screenShake: 0,
  matchHistory: { rounds: [], matchWinner: null },
  currentRoundLog: [],
  roundStartTime: 0,
  peakCombo: { p1: 0, p2: 0 },
  blocksPerformed: { p1: 0, p2: 0 },

  // AI settings — default to vs_player / medium
  gameMode: 'vs_player' as GameMode,
  aiDifficulty: 'medium' as AIDifficulty,
  player1ControlMode: 'keyboard' as PlayerOneControlMode,
  setGameMode: (mode) => set({ gameMode: mode }),
  setAIDifficulty: (difficulty) => set({ aiDifficulty: difficulty }),
  setPlayer1ControlMode: (mode) => set({ player1ControlMode: mode }),
  sessionId: '',

  // Actions
  startGame: () => set((s) => ({
    gameStatus: 'playing',
    roundStartTime: Date.now(),
    // Generate a fresh session ID only when starting a brand-new match (not on pause resume)
    sessionId: s.sessionId || (typeof window !== 'undefined' ? window.crypto.randomUUID() : Date.now().toString()),
  })),
  
  pauseGame: () => set({ gameStatus: 'paused' }),
  
  resumeGame: () => set({ gameStatus: 'playing' }),
  
  resetGame: () => set((s) => ({
    player1: createInitialPlayerState(1),
    player2: createInitialPlayerState(2),
    gameStatus: 'waiting',
    winner: null,
    roundTime: GAME_CONFIG.roundDuration,
    round: 1,
    player1RoundsWon: 0,
    player2RoundsWon: 0,
    roundWinner: null,
    showHitEffect: null,
    screenShake: 0,
    matchHistory: { rounds: [], matchWinner: null },
    currentRoundLog: [],
    roundStartTime: 0,
    peakCombo: { p1: 0, p2: 0 },
    blocksPerformed: { p1: 0, p2: 0 },
    sessionId: '',
    // Preserve AI mode settings across resets
    gameMode: s.gameMode,
    aiDifficulty: s.aiDifficulty,
    player1ControlMode: s.player1ControlMode,
  })),

  // Start next round (reset positions/health, keep scores)
  startNextRound: () => {
    const state = get();

    // --- Snapshot this round's stats before resetting ---
    const log = state.currentRoundLog;
    const p1Logs = log.filter(e => e.player === 1);
    const p2Logs = log.filter(e => e.player === 2);

    const roundStats: RoundStatistics = {
      roundNumber: state.round,
      winner: state.roundWinner ?? 'draw',
      durationMs: state.roundStartTime > 0 ? Date.now() - state.roundStartTime : 0,
      finalHealth: { p1: state.player1.health, p2: state.player2.health },
      totalDamageDealt: {
        p1: p1Logs.reduce((s, e) => s + e.damageDealt, 0),
        p2: p2Logs.reduce((s, e) => s + e.damageDealt, 0),
      },
      attacksAttempted: { p1: p1Logs.length, p2: p2Logs.length },
      attacksLanded: {
        p1: p1Logs.filter(e => e.succeeded).length,
        p2: p2Logs.filter(e => e.succeeded).length,
      },
      blocksPerformed: { p1: state.blocksPerformed.p1, p2: state.blocksPerformed.p2 },
      highestCombo: { p1: state.peakCombo.p1, p2: state.peakCombo.p2 },
      actionLog: log,
    };

    const updatedHistory: MatchHistory = {
      rounds: [...state.matchHistory.rounds, roundStats],
      matchWinner: state.matchHistory.matchWinner,
    };

    set({
      player1: createInitialPlayerState(1),
      player2: createInitialPlayerState(2),
      gameStatus: 'playing',
      roundTime: GAME_CONFIG.roundDuration,
      round: state.round + 1,
      roundWinner: null,
      showHitEffect: null,
      screenShake: 0,
      matchHistory: updatedHistory,
      currentRoundLog: [],
      roundStartTime: Date.now(),
      peakCombo: { p1: 0, p2: 0 },
      blocksPerformed: { p1: 0, p2: 0 },
    });
  },

  setPlayerAction: (playerId, action) => {
    const playerKey = playerId === 1 ? 'player1' : 'player2';
    const player = get()[playerKey];
    
    // Only block attack actions during hit cooldown, allow movement
    const isAttackAction = ['left_punch', 'right_punch', 'left_kick', 'right_kick'].includes(action);
    if (isAttackAction && player.hitCooldown > 0) return;

    if (player.action === action && player.isBlocking === (action === 'block')) {
      return;
    }

    const state = get();

    // Log new attack attempts — include full state snapshot for ML training
    if (isAttackAction && player.action !== action) {
      const event: ActionEvent = {
        timestamp: state.roundStartTime > 0 ? Date.now() - state.roundStartTime : 0,
        player: playerId,
        action,
        succeeded: false,
        damageDealt: 0,
        wasBlocked: false,
        stateSnapshot: {
          p1_x: state.player1.x,              p1_y: state.player1.y,
          p2_x: state.player2.x,              p2_y: state.player2.y,
          p1_health: state.player1.health,    p2_health: state.player2.health,
          p1_isAttacking: state.player1.isAttacking, p2_isAttacking: state.player2.isAttacking,
          p1_isBlocking:  state.player1.isBlocking,  p2_isBlocking:  state.player2.isBlocking,
          p1_attackCooldown: state.player1.attackCooldown, p2_attackCooldown: state.player2.attackCooldown,
          p1_hitCooldown:    state.player1.hitCooldown,    p2_hitCooldown:    state.player2.hitCooldown,
          roundTimeRemaining: state.roundTime,
          round: state.round,
          canvasWidth:  GAME_CONFIG.canvasWidth,
          canvasHeight: GAME_CONFIG.canvasHeight,
        },
      };
      set({ currentRoundLog: [...state.currentRoundLog, event] });
    }

    // Track block inputs for stats
    if (action === 'block' && player.action !== 'block') {
      const pk = playerId === 1 ? 'p1' : 'p2';
      set({ blocksPerformed: { ...state.blocksPerformed, [pk]: state.blocksPerformed[pk] + 1 } });
    }
    
    set((s) => ({
      [playerKey]: {
        ...s[playerKey],
        action,
        isBlocking: action === 'block',
        attackTimer: isAttackAction ? 0 : s[playerKey].attackTimer,
      },
    }));
  },

  updateGame: (deltaTime) => {
    const state = get();
    if (state.gameStatus !== 'playing') return;

    const updatePlayer = (player: PlayerState, opponent: PlayerState): PlayerState => {
      const newPlayer = { ...player };
      
      // Apply gravity
      if (!newPlayer.isGrounded) {
        newPlayer.velocityY += GAME_CONFIG.gravity;
      }
      
      // Handle movement based on action
      // Screen-relative: forward = right on screen, backward = left on screen
      // Both players use same direction mapping for intuitive arrow-key control
      switch (newPlayer.action) {
        case 'move_forward':
          // Always moves RIGHT on screen (positive X)
          newPlayer.velocityX = GAME_CONFIG.moveSpeed;
          break;
        case 'move_backward':
          // Always moves LEFT on screen (negative X)
          newPlayer.velocityX = -GAME_CONFIG.moveSpeed;
          break;
        case 'jump':
          if (newPlayer.isGrounded) {
            newPlayer.velocityY = GAME_CONFIG.jumpForce;
            newPlayer.isGrounded = false;
            SoundManager.play('jump');
          }
          break;
        case 'block':
        case 'idle':
        case 'left_punch':
        case 'right_punch':
        case 'left_kick':
        case 'right_kick':
        default:
          // Stop horizontal movement immediately on non-movement actions
          if (newPlayer.hitCooldown > 0) {
            newPlayer.velocityX *= 0.9; // Slide slightly when stunned
          } else {
            newPlayer.velocityX = 0;
          }
      }

      // Update position
      newPlayer.x += newPlayer.velocityX;
      newPlayer.y += newPlayer.velocityY;

      // Ground collision
      const groundLevel = GAME_CONFIG.groundY - GAME_CONFIG.playerHeight;
      if (newPlayer.y >= groundLevel) {
        newPlayer.y = groundLevel;
        newPlayer.velocityY = 0;
        newPlayer.isGrounded = true;
      }

      // Wall boundaries
      newPlayer.x = Math.max(20, Math.min(newPlayer.x, GAME_CONFIG.canvasWidth - GAME_CONFIG.playerWidth - 20));

      // Update cooldowns
      if (newPlayer.attackCooldown > 0) {
        newPlayer.attackCooldown -= deltaTime;
        if (newPlayer.attackCooldown <= 0) {
          newPlayer.isAttacking = false;
        }
      }
      
      // Update attack timer for timing hit detection
      const isAttackAction = ['left_punch', 'right_punch', 'left_kick', 'right_kick'].includes(newPlayer.action);
      if (isAttackAction) {
        newPlayer.attackTimer += deltaTime;
      } else {
        newPlayer.attackTimer = 0;
      }
      
      if (newPlayer.hitCooldown > 0) {
        newPlayer.hitCooldown = Math.max(0, newPlayer.hitCooldown - deltaTime);
      }

      // Combo decay timer
      if (newPlayer.comboTimer > 0) {
        newPlayer.comboTimer -= deltaTime;
        if (newPlayer.comboTimer <= 0) {
          newPlayer.comboCount = 0;
          newPlayer.comboTimer = 0;
        }
      }

      // Face opponent
      newPlayer.facingRight = newPlayer.x < opponent.x;

      return newPlayer;
    };

    // Check for attacks
    const checkAttack = (attacker: PlayerState, defender: PlayerState): { hit: boolean; damage: number; isPunch: boolean } => {
      const isAttackAction = ['left_punch', 'right_punch', 'left_kick', 'right_kick'].includes(attacker.action);
      
      if (!isAttackAction || attacker.attackCooldown > 0 || defender.hitCooldown > 0) {
        return { hit: false, damage: 0, isPunch: false };
      }

      // Hit window spans the middle-to-end of the animation so it's at least 2 frames
      // wide and can never be skipped by a single 16ms frame step.
      // Punch: 4 frames × 10ms = 40ms total  → window 12-42ms  (≈3 frames)
      // Kick : 5 frames × 25ms = 125ms total → window 60-130ms (≈4 frames)
      const isPunch = attacker.action.includes('punch');
      const hitWindowStart = isPunch ? 12 : 60;
      const hitWindowEnd   = isPunch ? 42 : 130;

      if (attacker.attackTimer < hitWindowStart || attacker.attackTimer > hitWindowEnd) {
        return { hit: false, damage: 0, isPunch: false };
      }

      const attackerCenter = attacker.x + GAME_CONFIG.playerWidth / 2;
      const defenderCenter = defender.x + GAME_CONFIG.playerWidth / 2;
      const distance = Math.abs(attackerCenter - defenderCenter);
      
      const isFacingDefender = attacker.facingRight
        ? defenderCenter > attackerCenter
        : defenderCenter < attackerCenter;

      if (distance < GAME_CONFIG.attackRange + GAME_CONFIG.playerWidth && isFacingDefender) {
        const baseDamage = isPunch ? GAME_CONFIG.attackDamage.punch : GAME_CONFIG.attackDamage.kick;
        const damage = defender.isBlocking ? Math.floor(baseDamage * 0.2) : baseDamage;
        return { hit: true, damage, isPunch };
      }

      return { hit: false, damage: 0, isPunch: false };
    };

    const newPlayer1 = updatePlayer(state.player1, state.player2);
    const newPlayer2 = updatePlayer(state.player2, state.player1);

    // Process attacks — use the already-updated player states so attackTimer,
    // hitCooldown and attackCooldown are all current-frame values.
    const attack1 = checkAttack(newPlayer1, newPlayer2);
    const attack2 = checkAttack(newPlayer2, newPlayer1);

    if (attack1.hit) {
      newPlayer2.health = Math.max(0, newPlayer2.health - attack1.damage);
      newPlayer2.hitCooldown = GAME_CONFIG.hitCooldownTime;
      // Knockback pushes defender AWAY from attacker
      const knockbackDir1 = newPlayer1.x < newPlayer2.x ? 1 : -1;
      newPlayer2.velocityX = knockbackDir1 * (state.player2.isBlocking ? 3 : 8);
      
      // Set attack cooldown to animation duration to ensure one hit per animation
      const animDuration = ANIMATIONS[state.player1.action].frameCount * ANIMATIONS[state.player1.action].frameDuration;
      newPlayer1.attackCooldown = animDuration;
      newPlayer1.isAttacking = true;
      newPlayer1.comboCount += 1;
      newPlayer1.comboTimer = 1500; // Reset combo decay timer (1.5s)
      newPlayer1.specialMeter = Math.min(100, newPlayer1.specialMeter + (state.player2.isBlocking ? 3 : 8));
      
      // Play attack and hit sounds
      SoundManager.playAttack(attack1.isPunch ? 'punch' : 'kick');
      SoundManager.playHit(state.player2.isBlocking);
      
      get().setHitEffect({ player: 2, type: state.player2.isBlocking ? 'block' : 'hit' });
      if (!state.player2.isBlocking) get().triggerScreenShake(5);
      setTimeout(() => get().setHitEffect(null), 200);

      // Mark the most recent P1 attack log entry as succeeded
      const updatedLog1 = [...get().currentRoundLog];
      for (let i = updatedLog1.length - 1; i >= 0; i--) {
        if (updatedLog1[i].player === 1 && !updatedLog1[i].succeeded) {
          updatedLog1[i] = { ...updatedLog1[i], succeeded: true, damageDealt: attack1.damage, wasBlocked: state.player2.isBlocking };
          break;
        }
      }
      // Track peak combo
      const curPeak1 = get().peakCombo;
      set({
        currentRoundLog: updatedLog1,
        peakCombo: { ...curPeak1, p1: Math.max(curPeak1.p1, newPlayer1.comboCount) },
      });
    }

    // Only process P2's attack if P2 is still alive this frame.
    // Prevents a player who was KO'd by attack1 from landing a simultaneous counter.
    if (attack2.hit && newPlayer2.health > 0) {
      newPlayer1.health = Math.max(0, newPlayer1.health - attack2.damage);
      newPlayer1.hitCooldown = GAME_CONFIG.hitCooldownTime;
      // Knockback pushes defender AWAY from attacker
      const knockbackDir2 = newPlayer2.x < newPlayer1.x ? 1 : -1;
      newPlayer1.velocityX = knockbackDir2 * (state.player1.isBlocking ? 3 : 8);
      
      // Set attack cooldown to animation duration to ensure one hit per animation
      const animDuration = ANIMATIONS[state.player2.action].frameCount * ANIMATIONS[state.player2.action].frameDuration;
      newPlayer2.attackCooldown = animDuration;
      newPlayer2.isAttacking = true;
      newPlayer2.comboCount += 1;
      newPlayer2.comboTimer = 1500; // Reset combo decay timer (1.5s)
      newPlayer2.specialMeter = Math.min(100, newPlayer2.specialMeter + (state.player1.isBlocking ? 3 : 8));
      
      // Play attack and hit sounds
      SoundManager.playAttack(attack2.isPunch ? 'punch' : 'kick');
      SoundManager.playHit(state.player1.isBlocking);
      
      get().setHitEffect({ player: 1, type: state.player1.isBlocking ? 'block' : 'hit' });
      if (!state.player1.isBlocking) get().triggerScreenShake(5);
      setTimeout(() => get().setHitEffect(null), 200);

      // Mark the most recent P2 attack log entry as succeeded
      const updatedLog2 = [...get().currentRoundLog];
      for (let i = updatedLog2.length - 1; i >= 0; i--) {
        if (updatedLog2[i].player === 2 && !updatedLog2[i].succeeded) {
          updatedLog2[i] = { ...updatedLog2[i], succeeded: true, damageDealt: attack2.damage, wasBlocked: state.player1.isBlocking };
          break;
        }
      }
      // Track peak combo
      const curPeak2 = get().peakCombo;
      set({
        currentRoundLog: updatedLog2,
        peakCombo: { ...curPeak2, p2: Math.max(curPeak2.p2, newPlayer2.comboCount) },
      });
    }

    // Check for round winner
    let roundWinner: 1 | 2 | 'draw' | null = state.roundWinner;
    let winner = state.winner;
    let gameStatus: 'waiting' | 'playing' | 'paused' | 'round_end' | 'match_end' = state.gameStatus as 'waiting' | 'playing' | 'paused' | 'round_end' | 'match_end';
    let player1RoundsWon = state.player1RoundsWon;
    let player2RoundsWon = state.player2RoundsWon;
    
    // Determine round winner if health drops to 0
    if (roundWinner === null && gameStatus === 'playing') {
      if (newPlayer1.health <= 0 && newPlayer2.health <= 0) {
        roundWinner = 'draw';
      } else if (newPlayer1.health <= 0) {
        roundWinner = 2;
        player2RoundsWon += 1;
      } else if (newPlayer2.health <= 0) {
        roundWinner = 1;
        player1RoundsWon += 1;
      }
    }

    // Decay screen shake
    const screenShake = Math.max(0, state.screenShake - 0.5);

    // Decrement round timer (deltaTime is in ms, roundTime is in seconds)
    let roundTime = state.roundTime - (deltaTime / 1000);
    
    // Time out - determine winner by health
    if (roundTime <= 0 && gameStatus === 'playing' && roundWinner === null) {
      roundTime = 0;
      if (newPlayer1.health > newPlayer2.health) {
        roundWinner = 1;
        player1RoundsWon += 1;
      } else if (newPlayer2.health > newPlayer1.health) {
        roundWinner = 2;
        player2RoundsWon += 1;
      } else {
        // True draw - both players have equal health
        roundWinner = 'draw';
      }
    }

    // If round just ended, determine game status
    if (roundWinner !== null && gameStatus === 'playing') {
      const winsNeeded = Math.ceil(state.maxRounds / 2); // e.g., 2 wins for best of 3
      if (player1RoundsWon >= winsNeeded) {
        winner = 1;
        gameStatus = 'match_end';
        SoundManager.play('victory');
      } else if (player2RoundsWon >= winsNeeded) {
        winner = 2;
        gameStatus = 'match_end';
        SoundManager.play('victory');
      } else if (state.round >= state.maxRounds && roundWinner === 'draw') {
        // Final round draw - sudden death or tie match
        winner = 'draw';
        gameStatus = 'match_end';
      } else {
        gameStatus = 'round_end';
        SoundManager.play('ko'); // Use KO sound for round end
      }

      // Snapshot this round into match history when game/round ends
      const finalLog = get().currentRoundLog;
      const fp1 = finalLog.filter(e => e.player === 1);
      const fp2 = finalLog.filter(e => e.player === 2);
      const finalPeak = get().peakCombo;
      const finalBlocks = get().blocksPerformed;

      const finalRoundStats: RoundStatistics = {
        roundNumber: state.round,
        winner: roundWinner,
        durationMs: state.roundStartTime > 0 ? Date.now() - state.roundStartTime : 0,
        finalHealth: { p1: newPlayer1.health, p2: newPlayer2.health },
        totalDamageDealt: {
          p1: fp1.reduce((s, e) => s + e.damageDealt, 0),
          p2: fp2.reduce((s, e) => s + e.damageDealt, 0),
        },
        attacksAttempted: { p1: fp1.length, p2: fp2.length },
        attacksLanded: {
          p1: fp1.filter(e => e.succeeded).length,
          p2: fp2.filter(e => e.succeeded).length,
        },
        blocksPerformed: { p1: finalBlocks.p1, p2: finalBlocks.p2 },
        highestCombo: { p1: finalPeak.p1, p2: finalPeak.p2 },
        actionLog: finalLog,
      };

      const updatedMatchHistory: MatchHistory = {
        rounds: [...state.matchHistory.rounds, finalRoundStats],
        matchWinner: gameStatus === 'match_end' ? (winner ?? null) : null,
      };

      set({
        player1: newPlayer1,
        player2: newPlayer2,
        winner,
        roundWinner,
        player1RoundsWon,
        player2RoundsWon,
        gameStatus,
        screenShake,
        roundTime,
        matchHistory: updatedMatchHistory,
        currentRoundLog: [],
        peakCombo: { p1: 0, p2: 0 },
        blocksPerformed: { p1: 0, p2: 0 },
      });
      return;
    }

    // AABB (Axis-Aligned Bounding Box) collision detection based on sprite sizes
    // Define sprite bounding boxes
    const p1_left = newPlayer1.x;
    const p1_right = newPlayer1.x + GAME_CONFIG.playerWidth;
    const p1_top = newPlayer1.y;
    const p1_bottom = newPlayer1.y + GAME_CONFIG.playerHeight;
    
    const p2_left = newPlayer2.x;
    const p2_right = newPlayer2.x + GAME_CONFIG.playerWidth;
    const p2_top = newPlayer2.y;
    const p2_bottom = newPlayer2.y + GAME_CONFIG.playerHeight;
    
    // Check if bounding boxes overlap
    const overlappingX = p1_left < p2_right && p1_right > p2_left;
    const overlappingY = p1_top < p2_bottom && p1_bottom > p2_top;
    
    if (overlappingX && overlappingY) {
      // Calculate overlap depths on each axis
      const overlapLeft = p1_right - p2_left;
      const overlapRight = p2_right - p1_left;
      const overlapTop = p1_bottom - p2_top;
      const overlapBottom = p2_bottom - p1_top;
      
      // Find minimum overlap (shortest distance to push apart)
      const minOverlapX = Math.min(overlapLeft, overlapRight);
      const minOverlapY = Math.min(overlapTop, overlapBottom);
      
      // Resolve collision on axis with least overlap (most efficient push)
      if (minOverlapX < minOverlapY) {
        // Resolve horizontally
        const pushForce = minOverlapX / 2 + 1;
        if (overlapLeft < overlapRight) {
          // P1 is to the left, push both away
          newPlayer1.x -= pushForce;
          newPlayer2.x += pushForce;
        } else {
          // P2 is to the left, push both away
          newPlayer1.x += pushForce;
          newPlayer2.x -= pushForce;
        }
      } else {
        // Resolve vertically
        const pushForce = minOverlapY / 2 + 1;
        if (overlapTop < overlapBottom) {
          // P1 is on top, push both away
          newPlayer1.y -= pushForce;
          newPlayer2.y += pushForce;
        } else {
          // P2 is on top, push both away
          newPlayer1.y += pushForce;
          newPlayer2.y -= pushForce;
        }
      }
      
      // Keep players within bounds
      newPlayer1.x = Math.max(20, Math.min(newPlayer1.x, GAME_CONFIG.canvasWidth - GAME_CONFIG.playerWidth - 20));
      newPlayer2.x = Math.max(20, Math.min(newPlayer2.x, GAME_CONFIG.canvasWidth - GAME_CONFIG.playerWidth - 20));
      newPlayer1.y = Math.max(0, newPlayer1.y);
      newPlayer2.y = Math.max(0, newPlayer2.y);
    }

    set({
      player1: newPlayer1,
      player2: newPlayer2,
      winner,
      roundWinner,
      player1RoundsWon,
      player2RoundsWon,
      gameStatus,
      screenShake,
      roundTime,
    });
  },

  applyDamage: (targetId, damage) => {
    const playerKey = targetId === 1 ? 'player1' : 'player2';
    set((state) => ({
      [playerKey]: {
        ...state[playerKey],
        health: Math.max(0, state[playerKey].health - damage),
      },
    }));
  },

  setHitEffect: (effect) => set({ showHitEffect: effect }),
  
  triggerScreenShake: (intensity) => set({ screenShake: intensity }),
}));
