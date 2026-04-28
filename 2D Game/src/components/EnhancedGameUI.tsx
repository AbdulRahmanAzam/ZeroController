import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { SoundManager } from '../audio/SoundManager';
import { VictoryScreen } from './VictoryScreen';
import type { MatchHistory, PlayerOneControlMode } from '../types/game';
import type { PoseInputStatus } from '../hooks/usePoseInput';

interface EnhancedGameUIProps {
  gameStatus: 'waiting' | 'playing' | 'paused' | 'round_end' | 'match_end';
  winner: 1 | 2 | 'draw' | null;
  round?: number;
  roundWinner?: 1 | 2 | 'draw' | null;
  player1RoundsWon?: number;
  player2RoundsWon?: number;
  onStartGame: () => void;
  onPauseGame: () => void;
  onResetGame: () => void;
  onGoToMenu: () => void;
  onNextRound?: () => void;
  matchHistory?: MatchHistory;
  player1ControlMode?: PlayerOneControlMode;
  poseStatus?: PoseInputStatus;
}

// Wrapper for button click with sound
const playSelectSound = () => {
  SoundManager.play('menuSelect');
};

const playHoverSound = () => {
  SoundManager.play('menuHover');
};

export const EnhancedGameUI: React.FC<EnhancedGameUIProps> = ({
  gameStatus,
  winner,
  onStartGame,
  onPauseGame,
  onResetGame,
  onGoToMenu,
  matchHistory,
  player1ControlMode = 'keyboard',
  poseStatus,
}) => {
  const zeroControllerSelected = player1ControlMode === 'zero_controller';
  const poseConnected = poseStatus?.connection === 'connected';
  const poseProgress = poseStatus?.sequenceLength
    ? Math.min(1, (poseStatus.bufferFill ?? 0) / poseStatus.sequenceLength)
    : 0;
  const poseStatusText = poseStatus?.message
    ?? (poseConnected ? 'ZeroController bridge connected.' : 'Run python run_model.py before starting.');
  const canStart = !zeroControllerSelected || poseConnected;

  const renderOverlay = () => {
    if (gameStatus === 'waiting') {
      return (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          style={overlayStyle}
        >
          {/* Animated background grid */}
          <motion.div
            animate={{ opacity: [0.1, 0.3, 0.1] }}
            transition={{ duration: 4, repeat: Infinity }}
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              backgroundImage: 'linear-gradient(0deg, rgba(52, 152, 219, 0.15) 1px, transparent 1px), linear-gradient(90deg, rgba(52, 152, 219, 0.15) 1px, transparent 1px)',
              backgroundSize: '30px 30px',
              pointerEvents: 'none',
            }}
          />

          {/* Corner accent triangles */}
          <motion.div
            animate={{ opacity: [0.3, 0.6, 0.3] }}
            transition={{ duration: 3, repeat: Infinity }}
            style={{
              position: 'absolute',
              top: 20,
              left: 20,
              width: 0,
              height: 0,
              borderLeft: '30px solid transparent',
              borderRight: '0px solid transparent',
              borderTop: '30px solid rgba(52, 152, 219, 0.5)',
            }}
          />
          <motion.div
            animate={{ opacity: [0.3, 0.6, 0.3] }}
            transition={{ duration: 3, repeat: Infinity, delay: 0.5 }}
            style={{
              position: 'absolute',
              bottom: 20,
              right: 20,
              width: 0,
              height: 0,
              borderLeft: '0px solid transparent',
              borderRight: '30px solid transparent',
              borderBottom: '30px solid rgba(231, 76, 60, 0.5)',
            }}
          />

          <motion.div
            initial={{ scale: 0.8, y: 50 }}
            animate={{ scale: 1, y: 0 }}
            transition={{ type: 'spring', damping: 12, stiffness: 100 }}
            style={modalStyle}
          >
            {/* Top accent bar */}
            <motion.div
              animate={{ width: ['0%', '100%'] }}
              transition={{ duration: 0.8 }}
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                height: 3,
                background: 'linear-gradient(90deg, #3498db 0%, #e74c3c 50%, #3498db 100%)',
              }}
            />

            {/* Left accent beam */}
            <motion.div
              animate={{ height: ['0%', '100%'] }}
              transition={{ duration: 0.8, delay: 0.1 }}
              style={{
                position: 'absolute',
                left: 0,
                top: 0,
                width: 3,
                background: 'linear-gradient(180deg, #3498db 0%, #e74c3c 50%, #3498db 100%)',
              }}
            />

            {/* Right accent beam */}
            <motion.div
              animate={{ height: ['0%', '100%'] }}
              transition={{ duration: 0.8, delay: 0.2 }}
              style={{
                position: 'absolute',
                right: 0,
                top: 0,
                width: 3,
                background: 'linear-gradient(180deg, #3498db 0%, #e74c3c 50%, #3498db 100%)',
              }}
            />

            {/* Bottom accent bar */}
            <motion.div
              animate={{ width: ['0%', '100%'] }}
              transition={{ duration: 0.8, delay: 0.3 }}
              style={{
                position: 'absolute',
                bottom: 0,
                left: 0,
                height: 3,
                background: 'linear-gradient(90deg, #3498db 0%, #e74c3c 50%, #3498db 100%)',
              }}
            />

            <div style={{ position: 'relative', zIndex: 1, width: '100%' }}>
              {/* Main title */}
              <motion.div
                animate={{ opacity: [0.8, 1, 0.8] }}
                transition={{ duration: 2, repeat: Infinity }}
                style={{
                  fontSize: 'clamp(9px, 1.2vh, 14px)',
                  fontWeight: 900,
                  color: '#e74c3c',
                  letterSpacing: 4,
                  marginBottom: 'clamp(6px, 1vh, 15px)',
                  textTransform: 'uppercase',
                  fontFamily: 'Bebas Neue, Orbitron, sans-serif',
                }}
              >
                ⚡ PREPARE FOR BATTLE ⚡
              </motion.div>

              <motion.h1
                animate={{
                  textShadow: [
                    '0 0 20px rgba(231, 76, 60, 0.8), 0 0 40px rgba(52, 152, 219, 0.4)',
                    '0 0 40px rgba(231, 76, 60, 1), 0 0 60px rgba(52, 152, 219, 0.6)',
                    '0 0 20px rgba(231, 76, 60, 0.8), 0 0 40px rgba(52, 152, 219, 0.4)',
                  ],
                }}
                transition={{ duration: 2, repeat: Infinity }}
                style={{
                  fontSize: 'clamp(26px, 5.5vh, 56px)',
                  marginBottom: 'clamp(4px, 0.8vh, 8px)',
                  color: '#fff',
                  fontWeight: 900,
                  fontFamily: 'Bebas Neue, Orbitron, sans-serif',
                  letterSpacing: 3,
                  WebkitTextStroke: '2px rgba(0, 0, 0, 0.6)',
                  textTransform: 'uppercase',
                  lineHeight: 1,
                }}
              >
                ⚔️ BATTLE ARENA ⚔️
              </motion.h1>

              <motion.p
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.3 }}
                style={{
                  fontSize: 'clamp(10px, 1.4vh, 16px)',
                  marginBottom: 'clamp(10px, 2vh, 35px)',
                  color: '#ecf0f1',
                  letterSpacing: 4,
                  fontFamily: 'Bebas Neue, Orbitron, sans-serif',
                  textTransform: 'uppercase',
                  background: 'linear-gradient(90deg, #e74c3c, #ffd700)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                }}
              >
                2-PLAYER FIGHTING GAME
              </motion.p>

              {zeroControllerSelected && (
                <motion.div
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.35 }}
                  style={zeroControllerPanelStyle}
                >
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 18, minWidth: 0 }}>
                    <div style={{ textAlign: 'left', minWidth: 0, flex: 1 }}>
                      <div style={{ color: '#4a9eff', fontFamily: 'Orbitron, sans-serif', fontSize: 11, letterSpacing: 3, marginBottom: 8, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                        ZERO CONTROLLER INPUT
                      </div>
                      <div style={{ color: '#f5f5f5', fontSize: 13, fontFamily: 'monospace', letterSpacing: 1, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                        python run_model.py
                      </div>
                      <div style={{ color: poseConnected ? '#8ff0b2' : '#ffb36b', fontSize: 12, marginTop: 9, lineHeight: 1.5, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                        {poseStatusText}
                      </div>
                    </div>
                    <div style={{ minWidth: 118, flexShrink: 0, textAlign: 'right' }}>
                      <div style={{ color: poseConnected ? '#2ecc71' : '#f39c12', fontWeight: 900, fontFamily: 'Bebas Neue, sans-serif', fontSize: 24, letterSpacing: 3, whiteSpace: 'nowrap' }}>
                        {poseConnected ? 'READY' : 'WAIT'}
                      </div>
                      <div style={{ color: '#777', fontSize: 11, fontFamily: 'monospace', marginTop: 4, whiteSpace: 'nowrap' }}>
                        {(poseStatus?.action ?? 'idle').replace('_', ' ')} · {Math.round((poseStatus?.confidence ?? 0) * 100)}%
                      </div>
                    </div>
                  </div>
                  <div style={{ height: 5, marginTop: 14, background: 'rgba(255,255,255,0.08)', overflow: 'hidden' }}>
                    <motion.div
                      animate={{ width: `${Math.round(poseProgress * 100)}%` }}
                      style={{ height: '100%', background: poseConnected ? 'linear-gradient(90deg, #3498db, #2ecc71)' : 'linear-gradient(90deg, #7f8c8d, #f39c12)' }}
                    />
                  </div>
                </motion.div>
              )}

              {/* Start button with epic styling */}
              <motion.button
                whileHover={canStart ? { scale: 1.08, boxShadow: '0 0 60px rgba(231, 76, 60, 0.8), inset 0 0 20px rgba(255, 255, 255, 0.1)' } : {}}
                whileTap={canStart ? { scale: 0.95 } : {}}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4 }}
                onClick={() => {
                  if (!canStart) return;
                  playSelectSound();
                  onStartGame();
                }}
                onHoverStart={canStart ? playHoverSound : undefined}
                disabled={!canStart}
                style={{
                  ...startButtonStyle,
                  opacity: canStart ? 1 : 0.55,
                  cursor: canStart ? 'pointer' : 'not-allowed',
                  filter: canStart ? 'none' : 'grayscale(0.45)',
                }}
              >
                <motion.span
                  animate={{ opacity: [1, 0.8, 1] }}
                  transition={{ duration: 1, repeat: Infinity }}
                  style={{ display: 'inline-block', marginRight: 12 }}
                >
                  ▶️
                </motion.span>
                {canStart ? 'FIGHT NOW' : 'RUN MODEL FIRST'}
                <motion.span
                  animate={{ opacity: [1, 0.8, 1] }}
                  transition={{ duration: 1, repeat: Infinity, delay: 0.2 }}
                  style={{ display: 'inline-block', marginLeft: 12 }}
                >
                  ⚡
                </motion.span>
              </motion.button>

              {/* Player cards */}
              <div style={{ marginTop: 'clamp(10px, 2.2vh, 45px)', display: 'flex', gap: 'clamp(14px, 3vw, 50px)', width: '100%', minWidth: 0 }}>
                <motion.div
                  initial={{ opacity: 0, x: -30 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.5 }}
                  style={{ ...playerCardStyle, borderTopColor: '#4a9eff' }}
                >
                  <div style={{ fontSize: 'clamp(16px, 2.2vh, 24px)', marginBottom: 'clamp(6px, 1vh, 12px)' }}>🔵</div>
                  <h3 style={{ color: '#4a9eff', marginBottom: 'clamp(8px, 1.2vh, 18px)', fontSize: 'clamp(13px, 1.8vh, 20px)', fontWeight: 900, letterSpacing: 2 }}>
                    PLAYER 1
                  </h3>
                  {zeroControllerSelected ? (
                    <div style={{ color: '#b9d9ff', fontSize: 12, lineHeight: 1.6, textAlign: 'left' }}>
                      <div style={{ fontFamily: 'Orbitron, sans-serif', letterSpacing: 2, color: poseConnected ? '#2ecc71' : '#f39c12', marginBottom: 8 }}>
                        {poseConnected ? 'CAMERA ONLINE' : 'BRIDGE OFFLINE'}
                      </div>
                      <div style={{ color: '#777' }}>Live actions drive Player 1.</div>
                      <div style={{ color: '#999', marginTop: 8, fontFamily: 'monospace' }}>
                        {poseStatus?.bridgeStatus ?? 'waiting'}
                      </div>
                    </div>
                  ) : (
                    <div style={controlGridStyle}>
                      <ControlKey label="Jump" keys={['W']} />
                      <ControlKey label="Move" keys={['A', 'D']} />
                      <ControlKey label="Block" keys={['S']} />
                      <ControlKey label="Punch" keys={['Q', 'E']} />
                      <ControlKey label="Kick" keys={['Z', 'C']} />
                    </div>
                  )}
                </motion.div>

                <motion.div
                  initial={{ opacity: 0, x: 30 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.6 }}
                  style={{ ...playerCardStyle, borderTopColor: '#e74c3c' }}
                >
                  <div style={{ fontSize: 'clamp(16px, 2.2vh, 24px)', marginBottom: 'clamp(6px, 1vh, 12px)' }}>🔴</div>
                  <h3 style={{ color: '#e74c3c', marginBottom: 'clamp(8px, 1.2vh, 18px)', fontSize: 'clamp(13px, 1.8vh, 20px)', fontWeight: 900, letterSpacing: 2 }}>
                    PLAYER 2
                  </h3>
                  <div style={controlGridStyle}>
                    <ControlKey label="Jump" keys={['I', '↑']} />
                    <ControlKey label="Move" keys={['J', 'L']} />
                    <ControlKey label="Block" keys={['K', '↓']} />
                    <ControlKey label="Punch" keys={['U', 'O']} />
                    <ControlKey label="Kick" keys={['N', 'M']} />
                  </div>
                </motion.div>
              </div>
            </div>
          </motion.div>
        </motion.div>
      );
    }

    if (gameStatus === 'paused') {
      return (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          style={overlayStyle}
        >
          {/* Scanlines */}
          <div style={{ position: 'absolute', inset: 0, background: 'repeating-linear-gradient(0deg, transparent 0px, transparent 2px, rgba(0,0,0,0.28) 2px, rgba(0,0,0,0.28) 3px)', pointerEvents: 'none', zIndex: 0 }} />
          {/* Arena fire glow from below */}
          <div style={{ position: 'absolute', inset: 0, background: 'radial-gradient(ellipse 80% 40% at 50% 110%, rgba(180,20,0,0.4) 0%, transparent 65%)', pointerEvents: 'none' }} />
          <motion.div
            initial={{ y: -24, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ type: 'spring', damping: 16 }}
            style={pauseModalStyle}
          >
            {/* Top accent bar */}
            <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: 3, background: 'linear-gradient(90deg, transparent, #e74c3c, #ffd700, #e74c3c, transparent)' }} />
            {/* Bottom accent bar */}
            <div style={{ position: 'absolute', bottom: 0, left: 0, right: 0, height: 3, background: 'linear-gradient(90deg, transparent, #e74c3c, #ffd700, #e74c3c, transparent)' }} />
            {/* Left bar */}
            <div style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: 3, background: 'linear-gradient(180deg, transparent, #e74c3c 30%, #e74c3c 70%, transparent)' }} />
            {/* Right bar */}
            <div style={{ position: 'absolute', right: 0, top: 0, bottom: 0, width: 3, background: 'linear-gradient(180deg, transparent, #e74c3c 30%, #e74c3c 70%, transparent)' }} />
            <div style={{ position: 'relative', zIndex: 1 }}>
              <div style={{ fontSize: 11, letterSpacing: 8, color: '#e74c3c', fontFamily: "'Orbitron', sans-serif", marginBottom: 4, opacity: 0.85 }}>— GAME —</div>
              <motion.h2
                animate={{ textShadow: ['0 0 18px rgba(241,196,15,0.8), 4px 4px 0 #000', '0 0 38px rgba(241,196,15,1), 4px 4px 0 #000', '0 0 18px rgba(241,196,15,0.8), 4px 4px 0 #000'] }}
                transition={{ duration: 1.8, repeat: Infinity }}
                style={{
                  fontSize: 80,
                  fontWeight: 900,
                  fontFamily: "'Bebas Neue', sans-serif",
                  color: '#f1c40f',
                  letterSpacing: 14,
                  margin: '0 0 8px',
                  lineHeight: 1,
                  WebkitTextStroke: '2px rgba(0,0,0,0.5)',
                }}
              >
                PAUSED
              </motion.h2>
              <div style={{ height: 2, background: 'linear-gradient(90deg, transparent, #444, transparent)', margin: '14px 0 26px' }} />
              <div style={{ display: 'flex', flexDirection: 'column', gap: 12, width: 300, margin: '0 auto' }}>
                <PauseBtn label="RESUME"    icon="▶" color="#27ae60" onClick={() => { playSelectSound(); onStartGame(); }} />
                <PauseBtn label="RESTART"   icon="↺" color="#7f8c8d" onClick={() => { playSelectSound(); onResetGame(); }} />
                <PauseBtn label="MAIN MENU" icon="⌂" color="#e74c3c" onClick={() => { playSelectSound(); onGoToMenu(); }} />
              </div>
            </div>
          </motion.div>
        </motion.div>
      );
    }

    // Round end - handled by RoundSummary component in Game.tsx
    if (gameStatus === 'round_end') {
      return null;
    }

    // Match end - show victory screen
    if (gameStatus === 'match_end') {
      return (
        <VictoryScreen
          winner={winner!}
          onPlayAgain={() => { playSelectSound(); onResetGame(); }}
          matchHistory={matchHistory}
        />
      );
    }

    return null;
  };

  return (
    <>
      {/* Pause button during gameplay */}
      {gameStatus === 'playing' && (
        <motion.button
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
          onClick={() => { playSelectSound(); onPauseGame(); }}
          onHoverStart={playHoverSound}
          style={pauseButtonStyle}
        >
          ⏸️
        </motion.button>
      )}

      <AnimatePresence>
        {renderOverlay()}
      </AnimatePresence>
    </>
  );
};

// Control key display component
const ControlKey: React.FC<{ label: string; keys: string[] }> = ({ label, keys }) => (
  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
    <span style={{ color: '#95a5a6', fontSize: 12, width: 50 }}>{label}</span>
    <div style={{ display: 'flex', gap: 4 }}>
      {keys.map((key, i) => (
        <span
          key={i}
          style={{
            background: 'rgba(255,255,255,0.1)',
            padding: '4px 10px',
            borderRadius: 5,
            fontSize: 13,
            fontFamily: 'monospace',
            color: '#fff',
            border: '1px solid rgba(255,255,255,0.2)',
            minWidth: 20,
            textAlign: 'center',
          }}
        >
          {key}
        </span>
      ))}
    </div>
  </div>
);

// Styles
const overlayStyle: React.CSSProperties = {
  position: 'absolute',
  top: 0,
  left: 0,
  right: 0,
  bottom: 0,
  background: 'rgba(0, 0, 0, 0.92)',
  backdropFilter: 'blur(8px)',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  zIndex: 100,
};

const pauseModalStyle: React.CSSProperties = {
  background: 'linear-gradient(160deg, #080808 0%, #100808 50%, #0a0505 100%)',
  padding: '48px 64px',
  textAlign: 'center',
  color: '#fff',
  position: 'relative',
  overflow: 'hidden',
  minWidth: 420,
};

const modalStyle: React.CSSProperties = {
  background: 'linear-gradient(180deg, rgba(20, 8, 8, 0.99) 0%, rgba(12, 6, 6, 0.99) 50%, rgba(8, 4, 4, 0.99) 100%)',
  padding: 'clamp(14px, 3vh, 60px) clamp(18px, 4vw, 70px)',
  borderRadius: 0,
  textAlign: 'center',
  color: '#fff',
  border: '3px solid rgba(231, 76, 60, 0.5)',
  boxShadow: `
    0 30px 100px rgba(0, 0, 0, 0.9),
    0 0 60px rgba(231, 76, 60, 0.15),
    inset 0 1px 20px rgba(255, 255, 255, 0.03)
  `,
  position: 'relative',
  overflow: 'hidden',
  width: '90vw',
  maxWidth: 900,
  minWidth: 480,
  maxHeight: '94vh',
  display: 'flex',
  flexDirection: 'column' as const,
  alignItems: 'center',
};

const startButtonStyle: React.CSSProperties = {
  padding: 'clamp(10px, 1.8vh, 20px) clamp(28px, 6vw, 80px)',
  fontSize: 'clamp(16px, 2.5vh, 28px)',
  fontWeight: 900,
  background: 'linear-gradient(180deg, #e74c3c 0%, #c0392b 50%, #a93226 100%)',
  color: '#fff',
  border: '3px solid #ff6b6b',
  borderRadius: 20,
  cursor: 'pointer',
  textTransform: 'uppercase',
  letterSpacing: 4,
  boxShadow: `
    0 15px 50px rgba(231, 76, 60, 0.6),
    0 0 30px rgba(255, 107, 107, 0.4),
    inset 0 1px 0 rgba(255, 255, 255, 0.2)
  `,
  transition: 'all 0.3s',
  fontFamily: 'Bebas Neue, Orbitron, sans-serif',
  textShadow: '0 4px 10px rgba(0, 0, 0, 0.8)',
  WebkitTextStroke: '1px rgba(0, 0, 0, 0.4)',
};

const zeroControllerPanelStyle: React.CSSProperties = {
  margin: '0 auto clamp(10px, 1.6vh, 24px)',
  width: 'min(560px, 100%)',
  flexShrink: 0,
  padding: 'clamp(10px, 1.5vh, 18px) 20px',
  background: 'linear-gradient(108deg, rgba(11, 24, 34, 0.94) 0%, rgba(11, 8, 8, 0.94) 100%)',
  border: '2px solid rgba(74, 158, 255, 0.45)',
  boxShadow: '0 12px 32px rgba(0,0,0,0.45), inset 0 0 24px rgba(74,158,255,0.07)',
  clipPath: 'polygon(0 0, calc(100% - 16px) 0, 100% 16px, 100% 100%, 16px 100%, 0 calc(100% - 16px))',
};

const playerCardStyle: React.CSSProperties = {
  flex: 1,
  minWidth: 0,
  textAlign: 'center',
  padding: 'clamp(10px, 1.8vh, 25px) clamp(10px, 1.5vw, 20px)',
  background: 'linear-gradient(180deg, rgba(20, 10, 5, 0.9) 0%, rgba(0, 0, 0, 0.5) 100%)',
  borderRadius: 0,
  border: '2px solid rgba(180, 60, 20, 0.25)',
  borderTop: '4px solid #e74c3c',
  boxShadow: '0 10px 30px rgba(0, 0, 0, 0.6), inset 0 1px 10px rgba(255, 255, 255, 0.02)',
};

const controlGridStyle: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
};

const pauseButtonStyle: React.CSSProperties = {
  position: 'absolute',
  top: 15,
  right: 15,
  width: 48,
  height: 48,
  fontSize: 18,
  background: 'rgba(10, 0, 0, 0.75)',
  border: '2px solid rgba(231, 76, 60, 0.5)',
  clipPath: 'polygon(0 0, calc(100% - 10px) 0, 100% 10px, 100% 100%, 10px 100%, 0 calc(100% - 10px))',
  cursor: 'pointer',
  zIndex: 50,
  backdropFilter: 'blur(6px)',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  color: '#e74c3c',
  fontFamily: "'Bebas Neue', sans-serif",
  letterSpacing: 1,
};

/* ── PauseBtn: clip-path action button used inside pause screen ── */
const PauseBtn: React.FC<{ label: string; icon: string; color: string; onClick: () => void }> = ({ label, icon, color, onClick }) => (
  <motion.button
    whileHover={{ x: 6, boxShadow: `0 0 28px ${color}66` }}
    whileTap={{ scale: 0.97 }}
    onClick={onClick}
    style={{
      position: 'relative',
      background: `linear-gradient(108deg, ${color}18 0%, ${color}08 100%)`,
      border: `2px solid ${color}`,
      clipPath: 'polygon(0 0, calc(100% - 14px) 0, 100% 14px, 100% 100%, 14px 100%, 0 calc(100% - 14px))',
      padding: '14px 24px',
      color: '#fff',
      cursor: 'pointer',
      display: 'flex',
      alignItems: 'center',
      gap: 16,
      outline: 'none',
      textAlign: 'left',
      boxShadow: '0 4px 16px rgba(0,0,0,0.7)',
      transition: 'box-shadow 0.2s',
    }}
  >
    <div style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: 4, background: color, boxShadow: `0 0 8px ${color}` }} />
    <span style={{ fontSize: 18, marginLeft: 8, opacity: 0.9 }}>{icon}</span>
    <span style={{ flex: 1, fontSize: 22, fontWeight: 900, fontFamily: "'Bebas Neue', sans-serif", letterSpacing: 5 }}>{label}</span>
    <span style={{ fontSize: 10, color, marginRight: 4, opacity: 0.55 }}>▶</span>
  </motion.button>
);

export default EnhancedGameUI;
