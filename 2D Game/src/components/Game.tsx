import React, { useCallback, useRef, useEffect, useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { EnhancedArena } from './EnhancedArena';
import { EnhancedFighter } from './EnhancedFighter';
import { EnhancedHealthBar } from './EnhancedHealthBar';
import { EnhancedGameUI } from './EnhancedGameUI';
import { HitEffects } from './HitEffects';
import { ComboCounter } from './ComboCounter';
import { SpecialMoveIndicator } from './SpecialMoveIndicator';
import { RoundSummary } from './RoundSummary';
import { useGameLoop } from '../hooks/useGameLoop';
import { useKeyboardInput } from '../hooks/useKeyboardInput';
import { usePoseInput } from '../hooks/usePoseInput';
import { useAIInput } from '../hooks/useAIInput';
import { useGameStore } from '../store/gameStore';
import { SoundManager } from '../audio/SoundManager';
import { sendSession } from '../services/analyticsService';
import type { ActionType, RoundStatistics } from '../types/game';
import type { PoseInputStatus } from '../hooks/usePoseInput';

export const Game: React.FC<{ onGoToMenu?: () => void }> = ({ onGoToMenu }) => {
  const {
    player1,
    player2,
    gameStatus,
    winner,
    roundTime,
    round,
    maxRounds,
    player1RoundsWon,
    player2RoundsWon,
    roundWinner,
    screenShake,
    showHitEffect,
    matchHistory,
    gameMode,
    aiDifficulty,
    player1ControlMode,
    startGame,
    pauseGame,
    resetGame,
    startNextRound,
    setPlayerAction,
    updateGame,
  } = useGameStore();

  // Viewport dimensions and responsive controls
  const [viewportDimensions, setViewportDimensions] = useState({
    width: window.innerWidth,
    height: window.innerHeight
  });
  const [isCompactControls, setIsCompactControls] = useState(false);

  // Update viewport dimensions and responsive settings
  useEffect(() => {
    const updateDimensions = () => {
      const newDimensions = {
        width: window.innerWidth,
        height: window.innerHeight
      };
      setViewportDimensions(newDimensions);
      setIsCompactControls(window.innerWidth < 900);
    };

    updateDimensions();
    window.addEventListener('resize', updateDimensions);
    return () => window.removeEventListener('resize', updateDimensions);
  }, []);

  const roundTimerRef = useRef<number>(0);
  const [showFightAnnouncement, setShowFightAnnouncement] = useState(false);
  const [announcementText, setAnnouncementText] = useState('');
  const [poseStatus, setPoseStatus] = useState<PoseInputStatus>({
    connection: 'idle',
    action: 'idle',
    confidence: 0,
  });
  const prevGameStatus = useRef(gameStatus);
  const isZeroControllerMode = player1ControlMode === 'zero_controller';

  // Initialize SoundManager on mount
  useEffect(() => {
    SoundManager.init();
    return () => {
      SoundManager.stopMusic();
    };
  }, []);

  // Handle game status changes for sounds + analytics persistence
  useEffect(() => {
    // Match finished — play victory sounds and send session to MongoDB
    if (prevGameStatus.current === 'playing' && gameStatus === 'match_end') {
      SoundManager.stopMusic();
      SoundManager.playVictory();

      // Read fresh state — the final round is still in currentRoundLog
      // (startNextRound was never called for it)
      const s = useGameStore.getState();
      const log   = s.currentRoundLog;
      const p1Log = log.filter(e => e.player === 1);
      const p2Log = log.filter(e => e.player === 2);

      const finalRound: RoundStatistics = {
        roundNumber:      s.round,
        winner:           s.roundWinner ?? 'draw',
        durationMs:       s.roundStartTime > 0 ? Date.now() - s.roundStartTime : 0,
        finalHealth:      { p1: s.player1.health, p2: s.player2.health },
        totalDamageDealt: {
          p1: p1Log.reduce((acc, e) => acc + e.damageDealt, 0),
          p2: p2Log.reduce((acc, e) => acc + e.damageDealt, 0),
        },
        attacksAttempted: { p1: p1Log.length, p2: p2Log.length },
        attacksLanded: {
          p1: p1Log.filter(e => e.succeeded).length,
          p2: p2Log.filter(e => e.succeeded).length,
        },
        blocksPerformed: { p1: s.blocksPerformed.p1, p2: s.blocksPerformed.p2 },
        highestCombo:    { p1: s.peakCombo.p1,       p2: s.peakCombo.p2 },
        actionLog:       log,
      };

      sendSession({
        sessionId:    s.sessionId,
        gameMode:     s.gameMode,
        aiDifficulty: s.gameMode === 'vs_ai' ? s.aiDifficulty : null,
        matchWinner:  s.winner,
        rounds:       [...s.matchHistory.rounds, finalRound],
      });
    }

    // Game reset
    if ((prevGameStatus.current === 'match_end' || prevGameStatus.current === 'round_end') && gameStatus === 'waiting') {
      SoundManager.stopMusic();
    }

    prevGameStatus.current = gameStatus;
  }, [gameStatus]);

  const showRoundAnnouncement = useCallback(() => {
    setAnnouncementText('READY?');
    setShowFightAnnouncement(true);
    window.setTimeout(() => setAnnouncementText('FIGHT!'), 800);
    window.setTimeout(() => setShowFightAnnouncement(false), 1800);
  }, []);

  // Handle input from keyboard (P2 key inputs are ignored in AI mode)
  const handleInput = useCallback((input: { playerId: 1 | 2; action: ActionType }) => {
    if (input.playerId === 2 && gameMode === 'vs_ai') return;
    setPlayerAction(input.playerId, input.action);
  }, [setPlayerAction, gameMode]);

  const handlePoseInput = useCallback((input: { playerId: 1 | 2; action: ActionType }) => {
    if (input.playerId !== 1 || gameStatus !== 'playing') return;
    setPlayerAction(1, input.action);
  }, [setPlayerAction, gameStatus]);

  // Handle input from AI (only for Player 2)
  const handleAIInput = useCallback((input: { playerId: 1 | 2; action: ActionType }) => {
    setPlayerAction(input.playerId, input.action);
  }, [setPlayerAction]);

  // Keyboard input hook (P1 always, P2 only in vs_player mode)
  useKeyboardInput({
    onInput: handleInput,
    enabled: gameStatus === 'playing',
    player1Enabled: !isZeroControllerMode,
    player2Enabled: gameMode === 'vs_player',
  });

  // ZeroController camera input hook (Player 1 only)
  usePoseInput({
    onInput: handlePoseInput,
    enabled: isZeroControllerMode && (gameStatus === 'waiting' || gameStatus === 'playing'),
    playerId: 1,
    onStatus: setPoseStatus,
  });

  // AI input hook (P2 only in vs_ai mode)
  useAIInput({
    onInput: handleAIInput,
    enabled: gameStatus === 'playing' && gameMode === 'vs_ai',
    difficulty: aiDifficulty,
  });

  // Game update function
  const onUpdate = useCallback((deltaTime: number) => {
    if (gameStatus !== 'playing') return;
    
    updateGame(deltaTime);
    
    // Update round timer
    roundTimerRef.current += deltaTime;
    if (roundTimerRef.current >= 1000) {
      roundTimerRef.current -= 1000;
      // Timer logic handled in store
    }
  }, [gameStatus, updateGame]);

  // Render function
  const onRender = useCallback(() => {}, []);

  // Game loop hook
  const { start, stop } = useGameLoop({
    onUpdate,
    onRender,
    targetFPS: 60,
  });

  // Start/resume game
  const handleStartGame = useCallback(() => {
    if (gameStatus === 'waiting') {
      showRoundAnnouncement();
      SoundManager.playRoundStart();
      SoundManager.startMusic();
    } else if (gameStatus === 'paused') {
      SoundManager.resumeMusic();
    }

    startGame();
    start();
  }, [gameStatus, showRoundAnnouncement, startGame, start]);

  // Pause game
  const handlePauseGame = useCallback(() => {
    SoundManager.pauseMusic();
    pauseGame();
    stop();
  }, [pauseGame, stop]);

  // Go to main menu — stop game, reset, navigate up
  const handleGoToMenu = useCallback(() => {
    SoundManager.stopMusic();
    stop();
    roundTimerRef.current = 0;
    setShowFightAnnouncement(false);
    setAnnouncementText('');
    resetGame();
    onGoToMenu?.();
  }, [stop, resetGame, onGoToMenu]);

  // Reset game
  const handleResetGame = useCallback(() => {
    SoundManager.stopMusic();
    stop();
    roundTimerRef.current = 0;
    setShowFightAnnouncement(false);
    setAnnouncementText('');
    resetGame();
  }, [stop, resetGame]);

  // Cleanup on unmount
  useEffect(() => {
    return () => stop();
  }, [stop]);

  // ESC key to toggle pause
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        if (gameStatus === 'playing') {
          handlePauseGame();
        } else if (gameStatus === 'paused') {
          handleStartGame();
        }
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [gameStatus, handlePauseGame, handleStartGame]);

  // Memoized screen shake calculation - stable reference
  const shakeOffset = useMemo(() => {
    if (screenShake <= 0) return { x: 0, y: 0 };
    const seed = (player1.x + player2.x + roundTime) * 0.001;
    return {
      x: Math.sin(seed * 70) * screenShake,
      y: Math.cos(seed * 110) * screenShake * 0.65
    };
  }, [screenShake, player1.x, player2.x, roundTime]);

  // Calculate responsive font sizes based on viewport
  const responsiveSizes = useMemo(() => {
    const baseScale = Math.min(viewportDimensions.width / 1200, viewportDimensions.height / 600);
    return {
      controlFont: Math.max(10, 11 * baseScale),
    };
  }, [viewportDimensions]);

  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100vh',
        width: '100vw',
        background: 'linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #0a0a0a 100%)',
        padding: 0,
        overflow: 'hidden',
        position: 'relative',
      }}
    >
      {/* Game container with shake effect - TRUE FULLSCREEN */}
      <motion.div
        animate={{
          x: shakeOffset.x,
          y: shakeOffset.y,
        }}
        transition={{ duration: 0.05 }}
        style={{
          position: 'relative',
          width: '100vw',
          height: '100vh',
          overflow: 'hidden',
          background: 'transparent',
        }}
      >
        <EnhancedArena>
          {/* Players */}
          <EnhancedFighter player={player1} />
          <EnhancedFighter player={player2} />

          {/* Hit effects */}
          <AnimatePresence>
            {showHitEffect && (
              <HitEffects
                player={showHitEffect.player}
                type={showHitEffect.type}
                position={{
                  x: showHitEffect.player === 1 ? player1.x + 50 : player2.x + 50,
                  y: showHitEffect.player === 1 ? player1.y + 50 : player2.y + 50,
                }}
              />
            )}
          </AnimatePresence>

          {/* Health bars */}
          <EnhancedHealthBar player={player1} position="left" />
          <EnhancedHealthBar player={player2} position="right" />

          {/* Combo counters - new enhanced version */}
          <ComboCounter comboCount={player1.comboCount} position="left" damageDealt={player1.specialMeter > 0 ? player1.comboCount * 8 : 0} />
          <ComboCounter comboCount={player2.comboCount} position="right" damageDealt={player2.specialMeter > 0 ? player2.comboCount * 8 : 0} />

          {/* Special move indicators */}
          <SpecialMoveIndicator player={player1} position="left" />
          <SpecialMoveIndicator player={player2} position="right" />

          {/* Round indicator - Tekken style */}
          {gameStatus === 'playing' && (
            <motion.div
              initial={{ scale: 0, y: -50 }}
              animate={{ scale: 1, y: 0 }}
              transition={{ type: 'spring', damping: 12 }}
              style={{
                position: 'absolute',
                top: 20,
                left: '50%',
                transform: 'translateX(-50%)',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                zIndex: 50,
              }}
            >
              {/* Round badge */}
              <div
                style={{
                  background: 'linear-gradient(180deg, #0d0303 0%, #1a0505 100%)',
                  padding: '5px 28px',
                  clipPath: 'polygon(12px 0, 100% 0, calc(100% - 12px) 100%, 0 100%)',
                  border: '2px solid rgba(245, 166, 35, 0.6)',
                  boxShadow: '0 0 16px rgba(245, 166, 35, 0.25)',
                  marginBottom: 8,
                }}
              >
                <span style={{
                  color: '#f5a623',
                  fontSize: 14,
                  fontWeight: 900,
                  fontFamily: 'Bebas Neue, Orbitron, sans-serif',
                  letterSpacing: 5,
                  textShadow: '0 0 12px #f5a62380',
                }}>
                  ROUND {round}
                </span>
              </div>

              {/* Timer - aggressive angular SF6/Tekken style */}
              <motion.div
                animate={roundTime <= 10 ? {
                  scale: [1, 1.08, 1],
                  boxShadow: [
                    '0 0 20px rgba(231, 76, 60, 0.5)',
                    '0 0 50px rgba(231, 76, 60, 0.9)',
                    '0 0 20px rgba(231, 76, 60, 0.5)',
                  ],
                } : {}}
                transition={{ duration: 0.5, repeat: Infinity }}
                style={{
                  width: 84,
                  height: 84,
                  clipPath: 'polygon(0 14px, 14px 0, calc(100% - 14px) 0, 100% 14px, 100% calc(100% - 14px), calc(100% - 14px) 100%, 14px 100%, 0 calc(100% - 14px))',
                  background: roundTime <= 10
                    ? 'linear-gradient(180deg, #8b0000 0%, #c0392b 50%, #8b0000 100%)'
                    : 'linear-gradient(180deg, #0d0303 0%, #1a0808 50%, #0d0303 100%)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  border: `3px solid ${roundTime <= 10 ? '#ff4444' : 'rgba(245,166,35,0.5)'}`,
                  boxShadow: roundTime <= 10
                    ? '0 0 30px rgba(231,76,60,0.6), inset 0 0 20px rgba(0,0,0,0.5)'
                    : '0 0 18px rgba(245,166,35,0.2), inset 0 0 20px rgba(0,0,0,0.5)',
                }}
              >
                <motion.span
                  key={roundTime}
                  initial={{ scale: 1.3, opacity: 0.5 }}
                  animate={{ scale: 1, opacity: 1 }}
                  transition={{ duration: 0.15 }}
                  style={{
                    color: '#fff',
                    fontSize: 44,
                    fontWeight: 900,
                    fontFamily: 'Bebas Neue, Orbitron, sans-serif',
                    textShadow: `
                      0 0 20px ${roundTime <= 10 ? '#ff2222' : '#f5a62380'},
                      2px 2px 4px rgba(0,0,0,0.9)
                    `,
                  }}
                >
                  {Math.ceil(roundTime)}
                </motion.span>
              </motion.div>

              {/* Time's up warning */}
              {roundTime <= 5 && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: [0.5, 1, 0.5], y: 0 }}
                  transition={{ duration: 0.3, repeat: Infinity }}
                  style={{
                    marginTop: 8,
                    color: '#ff4757',
                    fontSize: 12,
                    fontWeight: 900,
                    fontFamily: 'Orbitron, sans-serif',
                    letterSpacing: 2,
                    textShadow: '0 0 10px #ff0000',
                  }}
                >
                  HURRY UP!
                </motion.div>
              )}
            </motion.div>
          )}

          {/* FIGHT! Announcement Overlay */}
          <AnimatePresence>
            {showFightAnnouncement && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  right: 0,
                  bottom: 0,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  zIndex: 200,
                  background: 'radial-gradient(ellipse at center, rgba(0,0,0,0.3) 0%, transparent 70%)',
                  pointerEvents: 'none',
                }}
              >
                <motion.div
                  key={announcementText}
                  initial={{ scale: 3, opacity: 0, rotate: -10 }}
                  animate={{ scale: 1, opacity: 1, rotate: 0 }}
                  exit={{ scale: 0.5, opacity: 0 }}
                  transition={{ type: 'spring', damping: 10, stiffness: 200 }}
                  style={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                  }}
                >
                  <motion.span
                    animate={{
                      textShadow: [
                        '0 0 30px #f1c40f, 0 0 60px #e74c3c, 0 0 90px #ff6b6b',
                        '0 0 50px #ff6b6b, 0 0 100px #f1c40f, 0 0 150px #e74c3c',
                        '0 0 30px #f1c40f, 0 0 60px #e74c3c, 0 0 90px #ff6b6b',
                      ],
                    }}
                    transition={{ duration: 0.3, repeat: Infinity }}
                    style={{
                      fontSize: announcementText === 'FIGHT!' ? 120 : 80,
                      fontWeight: 900,
                      fontFamily: 'Bebas Neue, Orbitron, sans-serif',
                      color: announcementText === 'FIGHT!' ? '#ff4757' : '#f1c40f',
                      letterSpacing: 15,
                      WebkitTextStroke: '3px rgba(0,0,0,0.8)',
                    }}
                  >
                    {announcementText}
                  </motion.span>
                  {announcementText === 'FIGHT!' && (
                    <>
                      {/* Energy burst lines */}
                      {Array.from({ length: 12 }).map((_, i) => (
                        <motion.div
                          key={i}
                          initial={{ scale: 0, rotate: i * 30 }}
                          animate={{ scale: [0, 3, 0], rotate: i * 30 }}
                          transition={{ duration: 0.5, delay: 0.1 }}
                          style={{
                            position: 'absolute',
                            width: 4,
                            height: 100,
                            background: `linear-gradient(180deg, #ff4757 0%, transparent 100%)`,
                            transformOrigin: 'center center',
                          }}
                        />
                      ))}
                    </>
                  )}
                </motion.div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Round summary with recommendations (shown between rounds) */}
          {gameStatus === 'round_end' && matchHistory.rounds.length > 0 && (() => {
            const lastRound = matchHistory.rounds[matchHistory.rounds.length - 1];
            return (
              <RoundSummary
                roundStats={lastRound}
                onNextRound={startNextRound}
                isLastRound={round >= maxRounds}
              />
            );
          })()}

          {/* Game UI overlay */}
          <EnhancedGameUI
            gameStatus={gameStatus}
            winner={winner}
            round={round}
            roundWinner={roundWinner}
            player1RoundsWon={player1RoundsWon}
            player2RoundsWon={player2RoundsWon}
            onStartGame={handleStartGame}
            onPauseGame={handlePauseGame}
            onResetGame={handleResetGame}
            onGoToMenu={handleGoToMenu}
            onNextRound={startNextRound}
            matchHistory={matchHistory}
            player1ControlMode={player1ControlMode}
            poseStatus={poseStatus}
          />
        </EnhancedArena>
      </motion.div>

      {/* Control hints - responsive */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
        style={{
          position: 'absolute',
          bottom: 10,
          left: '50%',
          transform: 'translateX(-50%)',
          zIndex: 95,
          background: 'rgba(0,0,0,0.45)',
          border: '1px solid rgba(255,255,255,0.2)',
          borderRadius: 10,
          padding: '6px 12px',
          backdropFilter: 'blur(4px)',
          display: 'flex',
          flexDirection: isCompactControls ? 'column' : 'row',
          gap: isCompactControls ? 12 : 26,
          color: '#7f8c8d',
          fontSize: responsiveSizes.controlFont,
          maxWidth: '90vw',
          textAlign: 'center',
        }}
      >
        <div style={{ textAlign: 'center' }}>
          <span style={{ color: '#3498db', fontWeight: 'bold' }}>PLAYER 1</span>
          <div style={{ marginTop: 5, fontFamily: 'monospace' }}>
            {isZeroControllerMode
              ? `ZeroController: ${poseStatus.connection.toUpperCase()} · ${poseStatus.action.replace('_', ' ')}`
              : 'W/A/S/D + Q/E (punch) + Z/C (kick)'}
          </div>
        </div>
        <div style={{ textAlign: 'center' }}>
          <span style={{ color: '#e74c3c', fontWeight: 'bold' }}>PLAYER 2</span>
          <div style={{ marginTop: 5, fontFamily: 'monospace' }}>
            I/J/K/L + U/O (punch) + N/M (kick)
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default Game;
