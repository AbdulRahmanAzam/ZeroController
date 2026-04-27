import React from 'react';
import { motion } from 'framer-motion';
import type { PlayerState } from '../types/game';

interface EnhancedHealthBarProps {
  player: PlayerState;
  position: 'left' | 'right';
}

// P1: amber-gold war knight  |  P2: blood-crimson fighter
const COLORS = {
  1: { primary: '#f5a623', secondary: '#c07012', accent: '#ffd060', name: 'P1 · KNIGHT' },
  2: { primary: '#e83030', secondary: '#9b1010', accent: '#ff6060', name: 'P2 · FIGHTER' },
};

export const EnhancedHealthBar: React.FC<EnhancedHealthBarProps> = ({ player, position }) => {
  const colors = COLORS[player.id];
  const healthPercent = (player.health / player.maxHealth) * 100;
  const isLowHealth = healthPercent <= 25;
  
  // Damage decay animation - Skull Girls style
  const [damageDecayPercent, setDamageDecayPercent] = React.useState(healthPercent);
  
  React.useEffect(() => {
    if (healthPercent < damageDecayPercent) {
      // Delay before the "red bar" catches up to actual health
      const timer = setTimeout(() => {
        setDamageDecayPercent(healthPercent);
      }, 300);
      return () => clearTimeout(timer);
    } else {
      setDamageDecayPercent(healthPercent);
    }
  }, [healthPercent, damageDecayPercent]);

  const getHealthColor = () => {
    if (healthPercent > 50) return { main: '#00ff88', glow: '#00cc66' };
    if (healthPercent > 25) return { main: '#ffcc00', glow: '#ff9900' };
    return { main: '#ff3333', glow: '#cc0000' };
  };

  const healthColor = getHealthColor();

  return (
    <div
      style={{
        position: 'absolute',
        top: 30,
        [position]: 20,
        width: '42%',
        maxWidth: 480,
      }}
    >
      {/* Player info row */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: position === 'left' ? 'flex-start' : 'flex-end',
          marginBottom: 10,
          gap: 12,
          flexDirection: position === 'right' ? 'row-reverse' : 'row',
        }}
      >
        {/* Player avatar */}
        <motion.div
          animate={isLowHealth ? { scale: [1, 1.1, 1] } : {}}
          transition={{ duration: 0.5, repeat: Infinity }}
          style={{
            width: 50,
            height: 50,
            clipPath: 'polygon(15% 0, 100% 0, 100% 85%, 85% 100%, 0 100%, 0 15%)',
            background: `linear-gradient(135deg, ${colors.primary} 0%, ${colors.secondary} 100%)`,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: `0 0 18px ${colors.primary}70`,
            position: 'relative',
            overflow: 'hidden',
          }}
        >
          <span style={{ 
            color: '#fff', 
            fontWeight: 900, 
            fontSize: 22,
            fontFamily: 'Bebas Neue, Orbitron, sans-serif',
            textShadow: '2px 2px 4px rgba(0,0,0,0.5)',
          }}>
            P{player.id}
          </span>
          {/* Shine effect */}
          <motion.div
            animate={{ x: [-100, 100] }}
            transition={{ duration: 2, repeat: Infinity, repeatDelay: 1 }}
            style={{
              position: 'absolute',
              top: 0,
              width: 30,
              height: '100%',
              background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent)',
              transform: 'skewX(-20deg)',
            }}
          />
        </motion.div>

        {/* Player name */}
        <div style={{ textAlign: position }}>
          <div
            style={{
              color: colors.primary,
              fontSize: 24,
              fontWeight: 900,
              fontFamily: 'Bebas Neue, Orbitron, sans-serif',
              letterSpacing: 4,
              textShadow: `0 0 20px ${colors.primary}80, 2px 2px 4px rgba(0,0,0,0.8)`,
            }}
          >
            PLAYER {player.id}
          </div>
          <div
            style={{
              color: '#888',
              fontSize: 10,
              fontFamily: 'Orbitron, sans-serif',
              letterSpacing: 2,
              marginTop: 2,
            }}
          >
            {colors.name}
          </div>
        </div>
      </div>

      {/* Health bar container - Tekken style angular */}
      <div
        style={{
          position: 'relative',
          height: 36,
          background: 'rgba(0,0,0,0.95)',
          clipPath: position === 'left' 
            ? 'polygon(0 0, 100% 0, 95% 100%, 0 100%)'
            : 'polygon(5% 0, 100% 0, 100% 100%, 0 100%)',
          border: `2px solid ${colors.primary}99`,
          boxShadow: `
            0 0 16px rgba(0,0,0,0.9),
            inset 0 0 24px rgba(0,0,0,0.6),
            0 0 22px ${colors.primary}22
          `,
        }}
      >
        {/* Background pattern */}
        <div
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: `repeating-linear-gradient(
              ${position === 'left' ? '90deg' : '270deg'},
              transparent 0px,
              transparent 8px,
              rgba(255,255,255,0.03) 8px,
              rgba(255,255,255,0.03) 10px
            )`,
          }}
        />

        {/* Damage decay bar - shows behind actual health */}
        <motion.div
          animate={{ width: `${damageDecayPercent}%` }}
          transition={{ duration: 0.8, ease: 'easeOut' }}
          style={{
            position: 'absolute',
            top: 2,
            bottom: 2,
            [position]: 2,
            background: 'linear-gradient(90deg, #ff9900 0%, #ffaa22 50%, #ff8800 100%)',
            boxShadow: '0 0 15px rgba(255, 152, 0, 0.6)',
            zIndex: 1,
          }}
        />

        {/* Health fill */}
        <motion.div
          initial={{ width: '100%' }}
          animate={{ width: `${healthPercent}%` }}
          transition={{ type: 'spring', stiffness: 100, damping: 15 }}
          style={{
            position: 'absolute',
            top: 2,
            bottom: 2,
            [position]: 2,
            background: `linear-gradient(${position === 'left' ? '90deg' : '270deg'}, 
              ${healthColor.main} 0%, 
              ${healthColor.glow} 80%,
              ${healthColor.main} 100%
            )`,
            boxShadow: `
              inset 0 -8px 20px rgba(0,0,0,0.3),
              inset 0 4px 10px rgba(255,255,255,0.4),
              0 0 20px ${healthColor.glow}80
            `,
            zIndex: 2,
          }}
        >
          {/* Animated shine */}
          <motion.div
            animate={{ x: position === 'left' ? [-200, 500] : [500, -200] }}
            transition={{ duration: 2, repeat: Infinity, repeatDelay: 2 }}
            style={{
              position: 'absolute',
              top: 0,
              width: 100,
              height: '100%',
              background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent)',
              transform: 'skewX(-20deg)',
            }}
          />
        </motion.div>

        {/* Damage flash */}
        {player.hitCooldown > 0 && (
          <motion.div
            initial={{ opacity: 1 }}
            animate={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              background: 'rgba(255, 0, 0, 0.6)',
            }}
          />
        )}

        {/* Health segments with color coding */}
        <div
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            display: 'flex',
            zIndex: 3,
          }}
        >
          {Array.from({ length: 20 }).map((_, i) => (
            <div
              key={i}
              style={{
                flex: 1,
                borderRight: i < 19 ? '2px solid rgba(0,0,0,0.5)' : 'none',
              }}
            />
          ))}
        </div>

        {/* Health percentage text */}
        <div
          style={{
            position: 'absolute',
            top: '50%',
            [position]: 15,
            transform: 'translateY(-50%)',
            color: '#fff',
            fontSize: 16,
            fontWeight: 900,
            fontFamily: 'Bebas Neue, Orbitron, sans-serif',
            textShadow: '2px 2px 4px rgba(0,0,0,0.9)',
            letterSpacing: 1,
            zIndex: 4,
          }}
        >
          {player.health}
        </div>

        {/* Damage number indicator */}
        {player.hitCooldown > 0 && (
          <motion.div
            initial={{ y: 0, opacity: 1 }}
            animate={{ y: -30, opacity: 0 }}
            transition={{ duration: 0.8 }}
            style={{
              position: 'absolute',
              top: -10,
              [position === 'left' ? 'left' : 'right']: 20,
              fontSize: 20,
              fontWeight: 900,
              fontFamily: 'Bebas Neue, sans-serif',
              color: '#ff3333',
              textShadow: '0 0 10px #ff0000, 2px 2px 4px rgba(0,0,0,0.9)',
              zIndex: 5,
            }}
          >
            -{Math.max(0, player.maxHealth - player.health)}
          </motion.div>
        )}
      </div>

      {/* Danger warning */}
      {isLowHealth && (
        <motion.div
          animate={{ 
            opacity: [0.5, 1, 0.5],
            x: position === 'left' ? [0, 5, 0] : [0, -5, 0],
          }}
          transition={{ duration: 0.3, repeat: Infinity }}
          style={{
            position: 'absolute',
            top: 100,
            [position === 'left' ? 'right' : 'left']: 0,
            color: '#ff3333',
            fontSize: 14,
            fontWeight: 900,
            fontFamily: 'Orbitron, sans-serif',
            textShadow: '0 0 15px #ff0000',
            letterSpacing: 2,
          }}
        >
          ⚠ DANGER ⚠
        </motion.div>
      )}
    </div>
  );
};

export default EnhancedHealthBar;
