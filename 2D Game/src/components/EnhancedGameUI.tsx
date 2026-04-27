import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { SoundManager } from '../audio/SoundManager';
import { VictoryScreen } from './VictoryScreen';
import type { MatchHistory } from '../types/game';

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
}) => {

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

            <div style={{ position: 'relative', zIndex: 1 }}>
              {/* Main title */}
              <motion.div
                animate={{ opacity: [0.8, 1, 0.8] }}
                transition={{ duration: 2, repeat: Infinity }}
                style={{
                  fontSize: 14,
                  fontWeight: 900,
                  color: '#e74c3c',
                  letterSpacing: 4,
                  marginBottom: 15,
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
                  fontSize: 56,
                  marginBottom: 8,
                  color: '#fff',
                  fontWeight: 900,
                  fontFamily: 'Bebas Neue, Orbitron, sans-serif',
                  letterSpacing: 3,
                  WebkitTextStroke: '2px rgba(0, 0, 0, 0.6)',
                  textTransform: 'uppercase',
                }}
              >
                ⚔️ BATTLE ARENA ⚔️
              </motion.h1>

              <motion.p
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.3 }}
                style={{
                  fontSize: 16,
                  marginBottom: 35,
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

              {/* Start button with epic styling */}
              <motion.button
                whileHover={{ scale: 1.08, boxShadow: '0 0 60px rgba(231, 76, 60, 0.8), inset 0 0 20px rgba(255, 255, 255, 0.1)' }}
                whileTap={{ scale: 0.95 }}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4 }}
                onClick={() => { playSelectSound(); onStartGame(); }}
                onHoverStart={playHoverSound}
                style={startButtonStyle}
              >
                <motion.span
                  animate={{ opacity: [1, 0.8, 1] }}
                  transition={{ duration: 1, repeat: Infinity }}
                  style={{ display: 'inline-block', marginRight: 12 }}
                >
                  ▶️
                </motion.span>
                FIGHT NOW
                <motion.span
                  animate={{ opacity: [1, 0.8, 1] }}
                  transition={{ duration: 1, repeat: Infinity, delay: 0.2 }}
                  style={{ display: 'inline-block', marginLeft: 12 }}
                >
                  ⚡
                </motion.span>
              </motion.button>

              {/* Player cards */}
              <div style={{ marginTop: 45, display: 'flex', gap: 50 }}>
                <motion.div
                  initial={{ opacity: 0, x: -30 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.5 }}
                  style={{ ...playerCardStyle, borderTopColor: '#4a9eff' }}
                >
                  <div style={{ fontSize: 24, marginBottom: 12 }}>🔵</div>
                  <h3 style={{ color: '#4a9eff', marginBottom: 18, fontSize: 20, fontWeight: 900, letterSpacing: 2 }}>
                    PLAYER 1
                  </h3>
                  <div style={controlGridStyle}>
                    <ControlKey label="Jump" keys={['W']} />
                    <ControlKey label="Move" keys={['A', 'D']} />
                    <ControlKey label="Block" keys={['S']} />
                    <ControlKey label="Punch" keys={['Q', 'E']} />
                    <ControlKey label="Kick" keys={['Z', 'C']} />
                  </div>
                </motion.div>

                <motion.div
                  initial={{ opacity: 0, x: 30 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.6 }}
                  style={{ ...playerCardStyle, borderTopColor: '#e74c3c' }}
                >
                  <div style={{ fontSize: 24, marginBottom: 12 }}>🔴</div>
                  <h3 style={{ color: '#e74c3c', marginBottom: 18, fontSize: 20, fontWeight: 900, letterSpacing: 2 }}>
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
  padding: '60px 70px',
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
  maxWidth: '90vw',
  maxHeight: '90vh',
  overflowY: 'auto' as const,
};

const startButtonStyle: React.CSSProperties = {
  padding: '20px 80px',
  fontSize: 28,
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

const playerCardStyle: React.CSSProperties = {
  flex: 1,
  textAlign: 'center',
  padding: '25px 20px',
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
