import React, { useMemo, useEffect } from 'react';
import { motion } from 'framer-motion';
import type { MatchHistory } from '../types/game';

interface VictoryScreenProps {
  winner: 1 | 2 | 'draw';
  onPlayAgain: () => void;
  matchHistory?: MatchHistory;
}

// Deterministic value generation - stable across renders
const getDeterministicValue = (seed: number): number => {
  const value = Math.sin(seed * 12.9898) * 43758.5453;
  return value - Math.floor(value);
};

// Pre-calculated confetti particles - stable generation
const CONFETTI_PARTICLES = Array.from({ length: 40 }, (_, i) => {
  const seed = i + 500;
  return {
    x: (getDeterministicValue(seed + 1) - 0.5) * 800,
    y: getDeterministicValue(seed + 2) * 600 + 200,
    rotate: getDeterministicValue(seed + 3) * 720,
    duration: 2 + getDeterministicValue(seed + 4) * 2,
    size: getDeterministicValue(seed + 5) * 10 + 5,
    isRound: getDeterministicValue(seed + 6) > 0.5,
    color: ['#f1c40f', '#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#ff6b6b'][i % 6],
  };
});

export const VictoryScreen: React.FC<VictoryScreenProps> = ({ winner, onPlayAgain, matchHistory }) => {
  const isDraw = winner === 'draw';
  const winnerColor = isDraw ? '#f1c40f' : winner === 1 ? '#00d4ff' : '#ff4757';
  const winnerName = isDraw ? 'DRAW' : winner === 1 ? 'BLUE WARRIOR' : 'RED FIGHTER';

  // ESC key to return to menu
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onPlayAgain();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onPlayAgain]);

  // Memoize expensive style calculations
  const victoryStyles = useMemo(() => ({
    koText: {
      fontSize: 140,
      fontWeight: 900 as const,
      fontFamily: 'Bebas Neue, Impact, sans-serif',
      color: '#ff3333',
      textShadow: `
        0 0 40px #ff0000,
        0 0 80px #ff3333,
        6px 6px 12px rgba(0,0,0,0.9),
        -3px -3px 0 rgba(0,0,0,0.8)
      `,
      WebkitTextStroke: '4px rgba(0,0,0,0.9)',
      letterSpacing: 20,
      marginBottom: -20,
    },
    winnerBanner: {
      background: `linear-gradient(90deg, transparent 0%, ${winnerColor}40 20%, ${winnerColor}60 50%, ${winnerColor}40 80%, transparent 100%)`,
      borderTop: `3px solid ${winnerColor}`,
      borderBottom: `3px solid ${winnerColor}`,
    },
    playerBox: {
      background: 'rgba(0,0,0,0.7)',
      border: `3px solid ${winnerColor}`,
      boxShadow: `
        0 0 40px ${winnerColor}60,
        inset 0 0 30px rgba(0,0,0,0.5)
      `,
    },
    avatar: {
      background: `linear-gradient(135deg, ${winnerColor} 0%, ${winnerColor}cc 100%)`,
      border: `4px solid #fff`,
    },
    playButton: {
      background: `linear-gradient(180deg, ${winnerColor} 0%, ${winnerColor}cc 100%)`,
      boxShadow: `0 10px 35px ${winnerColor}40`,
    }
  }), [winnerColor]);

  return (
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
        background: 'linear-gradient(180deg, rgba(0,0,0,0.95) 0%, rgba(20,20,40,0.98) 100%)',
        backdropFilter: 'blur(15px)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 200,
        overflow: 'hidden',
      }}
    >
      {/* Radial glow effect */}
      <motion.div
        animate={{
          scale: [1, 1.2, 1],
          opacity: [0.3, 0.5, 0.3],
        }}
        transition={{ duration: 2, repeat: Infinity }}
        style={{
          position: 'absolute',
          width: 600,
          height: 600,
          borderRadius: '50%',
          background: `radial-gradient(circle, ${winnerColor}40 0%, transparent 70%)`,
        }}
      />

      {/* Confetti particles - optimized */}
      {CONFETTI_PARTICLES.map((particle, i) => (
        <motion.div
          key={i}
          initial={{
            x: 0,
            y: 0,
            opacity: 1,
            rotate: 0,
          }}
          animate={{
            x: particle.x,
            y: particle.y,
            opacity: 0,
            rotate: particle.rotate,
          }}
          transition={{
            duration: particle.duration,
            delay: i * 0.03,
            ease: 'easeOut',
          }}
          style={{
            position: 'absolute',
            width: particle.size,
            height: particle.size,
            background: particle.color,
            borderRadius: particle.isRound ? '50%' : '0',
            boxShadow: '0 0 10px rgba(255,255,255,0.5)',
          }}
        />
      ))}

      <motion.div
        initial={{ scale: 0, rotate: -20 }}
        animate={{ scale: 1, rotate: 0 }}
        transition={{ type: 'spring', damping: 12, stiffness: 150 }}
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: 30,
          zIndex: 1,
        }}
      >
        {/* K.O. Text - memoized */}
        <motion.div
          initial={{ y: -100, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.2, type: 'spring', damping: 10 }}
          style={victoryStyles.koText}
        >
          K.O.
        </motion.div>

        {/* Winner banner */}
        <motion.div
          initial={{ scaleX: 0 }}
          animate={{ scaleX: 1 }}
          transition={{ delay: 0.5, duration: 0.5, type: 'spring' }}
          style={{
            ...victoryStyles.winnerBanner,
            padding: '20px 80px',
            position: 'relative',
            overflow: 'hidden',
          }}
        >
          {/* Animated shine */}
          <motion.div
            animate={{ x: [-200, 600] }}
            transition={{ duration: 1.5, repeat: Infinity, repeatDelay: 1 }}
            style={{
              position: 'absolute',
              top: 0,
              width: 100,
              height: '100%',
              background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent)',
              transform: 'skewX(-20deg)',
            }}
          />

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.7 }}
            style={{
              fontSize: 32,
              fontWeight: 900,
              fontFamily: 'Orbitron, sans-serif',
              color: '#fff',
              textShadow: '2px 2px 4px rgba(0,0,0,0.9)',
              letterSpacing: 8,
              textAlign: 'center',
            }}
          >
            WINNER
          </motion.div>
        </motion.div>

        {/* Player info */}
        <motion.div
          initial={{ scale: 0, rotate: 180 }}
          animate={{ scale: 1, rotate: 0 }}
          transition={{ delay: 0.9, type: 'spring', damping: 12 }}
          style={{
            ...victoryStyles.playerBox,
            display: 'flex',
            alignItems: 'center',
            gap: 25,
            padding: '25px 50px',
            borderRadius: 20,
          }}
        >
          {/* Winner avatar */}
          <motion.div
            animate={{
              boxShadow: [
                `0 0 30px ${winnerColor}80`,
                `0 0 50px ${winnerColor}`,
                `0 0 30px ${winnerColor}80`,
              ],
            }}
            transition={{ duration: 1, repeat: Infinity }}
            style={{
              ...victoryStyles.avatar,
              width: 100,
              height: 100,
              borderRadius: 15,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              position: 'relative',
              overflow: 'hidden',
            }}
          >
            <span
              style={{
                color: '#fff',
                fontWeight: 900,
                fontSize: 48,
                fontFamily: 'Bebas Neue, sans-serif',
                textShadow: '3px 3px 6px rgba(0,0,0,0.8)',
              }}
            >
              {isDraw ? '⚔️' : `P${winner}`}
            </span>

            {/* Rotating shine */}
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 3, repeat: Infinity, ease: 'linear' }}
              style={{
                position: 'absolute',
                width: '150%',
                height: 2,
                background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.8), transparent)',
                top: '50%',
              }}
            />
          </motion.div>

          {/* Winner name */}
          <div style={{ textAlign: 'left' }}>
            <motion.div
              animate={{
                textShadow: [
                  `0 0 20px ${winnerColor}`,
                  `0 0 40px ${winnerColor}`,
                  `0 0 20px ${winnerColor}`,
                ],
              }}
              transition={{ duration: 1, repeat: Infinity }}
              style={{
                fontSize: 52,
                fontWeight: 900,
                fontFamily: 'Bebas Neue, sans-serif',
                color: winnerColor,
                letterSpacing: 6,
                lineHeight: 1,
              }}
            >
              {isDraw ? 'DRAW' : `PLAYER ${winner}`}
            </motion.div>
            <div
              style={{
                fontSize: 18,
                fontFamily: 'Orbitron, sans-serif',
                color: '#aaa',
                letterSpacing: 3,
                marginTop: 8,
              }}
            >
              {winnerName}
            </div>
          </div>
        </motion.div>

        {/* Victory text */}
        <motion.div
          initial={{ y: 50, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 1.2 }}
          style={{
            fontSize: 72,
            fontWeight: 900,
            fontFamily: 'Bebas Neue, Impact, sans-serif',
            background: 'linear-gradient(135deg, #f1c40f 0%, #ff6b6b 50%, #f1c40f 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            backgroundClip: 'text',
            letterSpacing: 10,
            filter: 'drop-shadow(0 0 20px rgba(241, 196, 15, 0.6))',
          }}
        >
          🏆 VICTORY! 🏆
        </motion.div>

        {/* Match history scoreboard */}
        {matchHistory && matchHistory.rounds.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.3 }}
            style={{
              background: 'rgba(255,255,255,0.04)',
              border: '1px solid rgba(255,255,255,0.1)',
              borderRadius: 12,
              padding: '16px 28px',
              minWidth: 340,
            }}
          >
            {/* Round-by-round */}
            <div style={{ fontSize: 10, letterSpacing: 4, color: '#9b59b6', fontFamily: 'Orbitron, sans-serif', marginBottom: 10, textAlign: 'center' }}>MATCH SUMMARY</div>
            <div style={{ display: 'flex', gap: 8, justifyContent: 'center', marginBottom: 14 }}>
              {matchHistory.rounds.map((r, i) => (
                <div key={i} style={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  background: 'rgba(0,0,0,0.4)',
                  borderRadius: 8,
                  padding: '8px 12px',
                  border: `1px solid ${ r.winner === winner ? winnerColor + '60' : 'rgba(255,255,255,0.08)'}`,
                }}>
                  <span style={{ fontSize: 10, color: '#666', fontFamily: 'Orbitron, sans-serif', marginBottom: 4 }}>R{r.roundNumber}</span>
                  <span style={{ fontSize: 16, fontWeight: 900, fontFamily: 'Bebas Neue, sans-serif', color: r.winner === 'draw' ? '#f1c40f' : r.winner === 1 ? '#00d4ff' : '#ff4757' }}>
                    {r.winner === 'draw' ? 'DRAW' : `P${r.winner}`}
                  </span>
                </div>
              ))}
            </div>
            {/* Aggregate totals */}
            {(() => {
              const totDmgP1 = matchHistory.rounds.reduce((s, r) => s + r.totalDamageDealt.p1, 0);
              const totDmgP2 = matchHistory.rounds.reduce((s, r) => s + r.totalDamageDealt.p2, 0);
              const bestComboP1 = Math.max(...matchHistory.rounds.map(r => r.highestCombo.p1));
              const bestComboP2 = Math.max(...matchHistory.rounds.map(r => r.highestCombo.p2));
              const row = (label: string, v1: string | number, v2: string | number) => (
                <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
                  <span style={{ flex: 1, textAlign: 'right', color: '#00d4ff', fontSize: 12, fontFamily: 'Orbitron, sans-serif', fontWeight: 700 }}>{v1}</span>
                  <span style={{ width: 120, textAlign: 'center', color: '#555', fontSize: 10, letterSpacing: 1, textTransform: 'uppercase' }}>{label}</span>
                  <span style={{ flex: 1, textAlign: 'left', color: '#ff4757', fontSize: 12, fontFamily: 'Orbitron, sans-serif', fontWeight: 700 }}>{v2}</span>
                </div>
              );
              return (
                <div>
                  {row('Total damage', totDmgP1, totDmgP2)}
                  {row('Best combo', bestComboP1, bestComboP2)}
                </div>
              );
            })()}
          </motion.div>
        )}

        {/* Play again button */}
        <motion.button
          initial={{ y: 30, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 1.5 }}
          whileHover={{
            scale: 1.08,
            boxShadow: `0 15px 50px ${winnerColor}60`,
          }}
          whileTap={{ scale: 0.95 }}
          onClick={onPlayAgain}
          style={{
            ...victoryStyles.playButton,
            padding: '20px 70px',
            fontSize: 26,
            fontWeight: 900,
            fontFamily: 'Bebas Neue, sans-serif',
            color: '#fff',
            border: 'none',
            borderRadius: 15,
            cursor: 'pointer',
            textTransform: 'uppercase',
            letterSpacing: 4,
            transition: 'all 0.3s ease',
          }}
        >
          ⚔️ FIGHT AGAIN
        </motion.button>

        {/* Streak display (optional - can be connected to state later) */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.8 }}
          style={{
            fontSize: 14,
            fontFamily: 'Orbitron, sans-serif',
            color: '#666',
            letterSpacing: 2,
          }}
        >
          Press ESC to return to menu
        </motion.div>
      </motion.div>
    </motion.div>
  );
};

export default VictoryScreen;
