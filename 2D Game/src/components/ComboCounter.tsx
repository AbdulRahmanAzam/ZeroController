import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface ComboCounterProps {
  comboCount: number;
  position: 'left' | 'right';
  damageDealt?: number;
}

// Deterministic value generation - stable across renders
const getDeterministicValue = (seed: number): number => {
  const value = Math.sin(seed * 12.9898) * 43758.5453;
  return value - Math.floor(value);
};

// Pre-calculated stable particle positions
const ENERGY_PARTICLES = Array.from({ length: 8 }, (_, i) => ({
  xOffset: (getDeterministicValue(i + 50) - 0.5) * 100,
  yOffset: -(20 + getDeterministicValue(i + 80) * 80),
}));

export const ComboCounter: React.FC<ComboCounterProps> = ({
  comboCount,
  position,
  damageDealt = 0,
}) => {
  const getComboText = () => {
    if (comboCount >= 10) return 'AMAZING!';
    if (comboCount >= 7) return 'GREAT!';
    if (comboCount >= 5) return 'NICE!';
    return 'COMBO!';
  };

  const getComboColor = () => {
    if (comboCount >= 10) return { main: '#ff0066', glow: '#ff3399' };
    if (comboCount >= 7) return { main: '#ff6600', glow: '#ff9933' };
    if (comboCount >= 5) return { main: '#ffcc00', glow: '#ffdd44' };
    return { main: '#ffaa00', glow: '#ffcc33' };
  };

  const colors = getComboColor();
  const comboText = getComboText();
  const comboStyles = {
    comboNumber: {
      fontSize: comboCount >= 10 ? 90 : 72,
      fontWeight: 900 as const,
      fontFamily: 'Bebas Neue, Impact, sans-serif',
      color: colors.main,
      textShadow: `
        0 0 30px ${colors.glow},
        0 0 60px ${colors.glow},
        4px 4px 8px rgba(0,0,0,0.9),
        -2px -2px 0 rgba(0,0,0,0.5)
      `,
      WebkitTextStroke: '2px rgba(0,0,0,0.8)',
      letterSpacing: -2,
    },
    comboText: {
      fontSize: 24,
      fontWeight: 900 as const,
      fontFamily: 'Bebas Neue, sans-serif',
      color: colors.glow,
      textShadow: `0 0 20px ${colors.main}, 2px 2px 4px rgba(0,0,0,0.9)`,
      letterSpacing: 4,
    },
  };

  if (comboCount < 2) return null;

  return (
    <AnimatePresence>
      <motion.div
        key={`combo-${comboCount}`}
        initial={{ scale: 0, rotate: -180, opacity: 0 }}
        animate={{ scale: 1, rotate: 0, opacity: 1 }}
        exit={{ scale: 0, opacity: 0 }}
        transition={{ type: 'spring', damping: 10, stiffness: 200 }}
        style={{
          position: 'absolute',
          top: '35%',
          [position]: 80,
          display: 'flex',
          flexDirection: 'column',
          alignItems: position === 'left' ? 'flex-start' : 'flex-end',
          gap: 5,
          zIndex: 60,
        }}
      >
        {/* Combo Number - Skull Girls Style */}
        <motion.div
          animate={{
            scale: [1, 1.15, 1],
            rotate: [0, 5, -5, 0],
          }}
          transition={{ duration: 0.4, repeat: Infinity }}
          style={{
            display: 'flex',
            alignItems: 'baseline',
            gap: 10,
          }}
        >
          <motion.span
            style={comboStyles.comboNumber}
          >
            {comboCount}
          </motion.span>
          
          <motion.span
            style={{
              fontSize: 28,
              fontWeight: 900,
              fontFamily: 'Orbitron, sans-serif',
              color: '#fff',
              textShadow: '2px 2px 4px rgba(0,0,0,0.9)',
              letterSpacing: 1,
            }}
          >
            HITS
          </motion.span>
        </motion.div>

        {/* Combo Text - memoized */}
        <motion.div
          key={comboText}
          initial={{ x: position === 'left' ? -20 : 20, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          style={comboStyles.comboText}
        >
          {comboText}
        </motion.div>

        {/* Damage Display */}
        {damageDealt > 0 && (
          <motion.div
            initial={{ y: 10, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            style={{
              fontSize: 16,
              fontWeight: 700,
              fontFamily: 'Orbitron, monospace',
              color: '#ff3333',
              textShadow: '0 0 10px #ff0000, 1px 1px 2px rgba(0,0,0,0.8)',
              background: 'rgba(0,0,0,0.6)',
              padding: '4px 12px',
              borderRadius: 8,
              border: '2px solid rgba(255,51,51,0.5)',
            }}
          >
            -{damageDealt} DMG
          </motion.div>
        )}

        {/* Energy particles - optimized */}
        {ENERGY_PARTICLES.map((particle, i) => (
          <motion.div
            key={i}
            initial={{
              x: 0,
              y: 0,
              opacity: 1,
              scale: 1,
            }}
            animate={{
              x: particle.xOffset,
              y: particle.yOffset,
              opacity: 0,
              scale: 0,
            }}
            transition={{
              duration: 0.8,
              delay: i * 0.05,
              repeat: Infinity,
              repeatDelay: 0.5,
            }}
            style={{
              position: 'absolute',
              width: 8,
              height: 8,
              borderRadius: '50%',
              background: colors.main,
              boxShadow: `0 0 10px ${colors.glow}`,
              top: 40,
              [position]: position === 'left' ? 0 : 'auto',
              [position === 'left' ? 'right' : 'left']: position === 'left' ? 'auto' : 0,
            }}
          />
        ))}
      </motion.div>
    </AnimatePresence>
  );
};

export default ComboCounter;
