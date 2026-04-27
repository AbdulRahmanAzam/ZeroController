import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { PlayerState } from '../types/game';

interface CharacterPortraitProps {
  player: PlayerState;
  position: 'left' | 'right';
}

export const CharacterPortrait: React.FC<CharacterPortraitProps> = ({ player, position }) => {
  const colors = player.id === 1
    ? { primary: '#f5a623', secondary: '#c07012', accent: '#ffd060', name: 'KNIGHT' }
    : { primary: '#e83030', secondary: '#9b1010', accent: '#ff6060', name: 'FIGHTER' };

  const healthPercent = (player.health / player.maxHealth) * 100;
  const isLowHealth = healthPercent <= 25;
  const isDamaged = player.hitCooldown > 0;

  return (
    <motion.div
      initial={{ x: position === 'left' ? -100 : 100, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      transition={{ type: 'spring', damping: 15 }}
      style={{
        position: 'absolute',
        top: 15,
        [position]: 15,
        width: 180,
        display: 'flex',
        flexDirection: 'column',
        gap: 8,
        zIndex: 45,
      }}
    >
      {/* Main portrait container */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 12,
          flexDirection: position === 'left' ? 'row' : 'row-reverse',
        }}
      >
        {/* Animated portrait frame - Tekken/Guilty Gear inspired */}
        <motion.div
          animate={
            isDamaged
              ? { scale: [1, 0.95, 1], rotate: [0, -2, 2, 0] }
              : isLowHealth
              ? { scale: [1, 1.05, 1] }
              : {}
          }
          transition={{ duration: 0.3 }}
          style={{
            position: 'relative',
            width: 70,
            height: 70,
            flexShrink: 0,
          }}
        >
          {/* Outer glow ring */}
          <motion.div
            animate={
              isLowHealth
                ? {
                    boxShadow: [
                      `0 0 20px ${colors.primary}80, inset 0 0 20px rgba(0,0,0,0.5)`,
                      `0 0 35px ${colors.primary}, inset 0 0 25px rgba(0,0,0,0.5)`,
                      `0 0 20px ${colors.primary}80, inset 0 0 20px rgba(0,0,0,0.5)`,
                    ],
                  }
                : {}
            }
            transition={{ duration: 0.5, repeat: Infinity }}
            style={{
              position: 'absolute',
              width: '100%',
              height: '100%',
              borderRadius: 12,
              background: `linear-gradient(135deg, ${colors.primary} 0%, ${colors.secondary} 100%)`,
              boxShadow: `0 0 25px ${colors.primary}60, inset 0 0 20px rgba(255,255,255,0.15)`,
              border: `3px solid ${colors.accent}`,
              clipPath: 'polygon(15% 0, 100% 0, 100% 85%, 85% 100%, 0 100%, 0 15%)',
            }}
          >
            {/* Diagonal accent lines */}
            <div
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                right: 0,
                bottom: 0,
                background: `repeating-linear-gradient(
                  45deg,
                  transparent,
                  transparent 8px,
                  rgba(255,255,255,0.05) 8px,
                  rgba(255,255,255,0.05) 10px
                )`,
              }}
            />

            {/* Animated shine effect */}
            <motion.div
              animate={{ x: [-80, 100], opacity: [0, 1, 0] }}
              transition={{ duration: 2.5, repeat: Infinity, repeatDelay: 1.5 }}
              style={{
                position: 'absolute',
                top: 0,
                width: 30,
                height: '100%',
                background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent)',
                transform: 'skewX(-20deg)',
              }}
            />
          </motion.div>

          {/* Character avatar/icon */}
          <div
            style={{
              position: 'absolute',
              top: '50%',
              left: '50%',
              transform: 'translate(-50%, -50%)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              zIndex: 2,
            }}
          >
            <motion.span
              animate={isDamaged ? { opacity: [1, 0.3, 1] } : {}}
              transition={{ duration: 0.2 }}
              style={{
                color: '#fff',
                fontWeight: 900,
                fontSize: 32,
                fontFamily: 'Bebas Neue, Orbitron, sans-serif',
                textShadow: `
                  0 0 10px ${colors.primary},
                  3px 3px 6px rgba(0,0,0,0.9)
                `,
              }}
            >
              P{player.id}
            </motion.span>
          </div>

          {/* Corner decorations - fighting game style */}
          <div
            style={{
              position: 'absolute',
              top: -2,
              left: -2,
              width: 12,
              height: 12,
              border: `3px solid ${colors.accent}`,
              borderRight: 'none',
              borderBottom: 'none',
              borderRadius: '3px 0 0 0',
            }}
          />
          <div
            style={{
              position: 'absolute',
              bottom: -2,
              right: -2,
              width: 12,
              height: 12,
              border: `3px solid ${colors.accent}`,
              borderLeft: 'none',
              borderTop: 'none',
              borderRadius: '0 0 3px 0',
            }}
          />

          {/* Damage flash overlay */}
          <AnimatePresence>
            {isDamaged && (
              <motion.div
                initial={{ opacity: 0.8 }}
                animate={{ opacity: 0 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.2 }}
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  right: 0,
                  bottom: 0,
                  background: 'rgba(255, 0, 0, 0.5)',
                  borderRadius: 12,
                  clipPath: 'polygon(15% 0, 100% 0, 100% 85%, 85% 100%, 0 100%, 0 15%)',
                  zIndex: 3,
                }}
              />
            )}
          </AnimatePresence>

          {/* Health indicator ring */}
          <svg
            style={{
              position: 'absolute',
              top: -5,
              left: -5,
              width: 80,
              height: 80,
              transform: 'rotate(-90deg)',
              zIndex: 1,
            }}
          >
            <circle
              cx="40"
              cy="40"
              r="37"
              fill="none"
              stroke="rgba(0,0,0,0.5)"
              strokeWidth="3"
            />
            <motion.circle
              cx="40"
              cy="40"
              r="37"
              fill="none"
              stroke={isLowHealth ? '#ff3333' : colors.accent}
              strokeWidth="3"
              strokeDasharray={2 * Math.PI * 37}
              strokeDashoffset={2 * Math.PI * 37 * (1 - healthPercent / 100)}
              strokeLinecap="round"
              initial={{ strokeDashoffset: 2 * Math.PI * 37 }}
              animate={{
                strokeDashoffset: 2 * Math.PI * 37 * (1 - healthPercent / 100),
                opacity: isLowHealth ? [1, 0.5, 1] : 1,
              }}
              transition={{
                strokeDashoffset: { type: 'spring', stiffness: 80, damping: 15 },
                opacity: { duration: 0.5, repeat: Infinity },
              }}
            />
          </svg>
        </motion.div>

        {/* Player info panel */}
        <div
          style={{
            flex: 1,
            display: 'flex',
            flexDirection: 'column',
            gap: 4,
            textAlign: position === 'left' ? 'left' : 'right',
          }}
        >
          {/* Player name */}
          <motion.div
            animate={
              isLowHealth
                ? {
                    textShadow: [
                      `0 0 10px ${colors.primary}`,
                      `0 0 20px ${colors.primary}`,
                      `0 0 10px ${colors.primary}`,
                    ],
                  }
                : {}
            }
            transition={{ duration: 0.5, repeat: Infinity }}
            style={{
              color: colors.primary,
              fontSize: 20,
              fontWeight: 900,
              fontFamily: 'Bebas Neue, Orbitron, sans-serif',
              letterSpacing: 3,
              textShadow: `0 0 15px ${colors.primary}80, 2px 2px 4px rgba(0,0,0,0.9)`,
              lineHeight: 1,
            }}
          >
            PLAYER {player.id}
          </motion.div>

          {/* Character class/type */}
          <div
            style={{
              color: '#888',
              fontSize: 10,
              fontFamily: 'Orbitron, sans-serif',
              letterSpacing: 1.5,
              textTransform: 'uppercase',
            }}
          >
            {colors.name}
          </div>

          {/* Status indicators */}
          <div
            style={{
              display: 'flex',
              gap: 4,
              marginTop: 2,
              justifyContent: position === 'left' ? 'flex-start' : 'flex-end',
            }}
          >
            {/* Action indicator */}
            {player.action !== 'idle' && (
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                exit={{ scale: 0 }}
                style={{
                  padding: '2px 6px',
                  fontSize: 8,
                  fontWeight: 700,
                  fontFamily: 'Orbitron, sans-serif',
                  color: '#fff',
                  background: colors.primary,
                  borderRadius: 4,
                  textTransform: 'uppercase',
                  letterSpacing: 0.5,
                }}
              >
                {player.action}
              </motion.div>
            )}

            {/* Blocking indicator */}
            {player.isBlocking && (
              <motion.div
                animate={{ opacity: [0.7, 1, 0.7] }}
                transition={{ duration: 0.5, repeat: Infinity }}
                style={{
                  padding: '2px 6px',
                  fontSize: 8,
                  fontWeight: 700,
                  fontFamily: 'Orbitron, sans-serif',
                  color: '#fff',
                  background: '#3498db',
                  borderRadius: 4,
                  textTransform: 'uppercase',
                  letterSpacing: 0.5,
                }}
              >
                🛡️ BLOCK
              </motion.div>
            )}
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default CharacterPortrait;
