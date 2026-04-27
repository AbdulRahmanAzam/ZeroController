import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { PlayerState } from '../types/game';

interface SpecialMoveIndicatorProps {
  player: PlayerState;
  position: 'left' | 'right';
}

export const SpecialMoveIndicator: React.FC<SpecialMoveIndicatorProps> = ({
  player,
  position,
}) => {
  const colors = player.id === 1
    ? { primary: '#f5a623', secondary: '#c07012', accent: '#ffd060' }
    : { primary: '#e83030', secondary: '#9b1010', accent: '#ff6060' };

  // Calculate special move meter from game state
  const specialMeterPercent = player.specialMeter ?? 0;

  // Check if player is performing a special move (punch or kick actions)
  const isPerformingSpecial = 
    player.action === 'left_punch' || 
    player.action === 'right_punch' || 
    player.action === 'left_kick' || 
    player.action === 'right_kick';

  return (
    <div
      style={{
        position: 'absolute',
        bottom: 100,
        [position]: 30,
        display: 'flex',
        flexDirection: 'column',
        alignItems: position === 'left' ? 'flex-start' : 'flex-end',
        gap: 10,
      }}
    >
      {/* Special Move Meter - Guilty Gear Style */}
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          gap: 6,
        }}
      >
        <div
          style={{
            fontSize: 11,
            fontWeight: 700,
            fontFamily: 'Orbitron, sans-serif',
            color: colors.primary,
            letterSpacing: 2,
            textShadow: `0 0 10px ${colors.primary}`,
          }}
        >
          SPECIAL
        </div>

        {/* Meter segments */}
        <div
          style={{
            display: 'flex',
            gap: 4,
            flexDirection: position === 'left' ? 'row' : 'row-reverse',
          }}
        >
          {Array.from({ length: 3 }).map((_, i) => {
            const segmentFilled = specialMeterPercent > (i * 33.33);
            return (
              <motion.div
                key={i}
                animate={
                  segmentFilled
                    ? {
                        boxShadow: [
                          `0 0 10px ${colors.primary}80`,
                          `0 0 20px ${colors.primary}`,
                          `0 0 10px ${colors.primary}80`,
                        ],
                      }
                    : {}
                }
                transition={{ duration: 0.8, repeat: Infinity }}
                style={{
                  width: 50,
                  height: 14,
                  background: segmentFilled
                    ? `linear-gradient(135deg, ${colors.primary} 0%, ${colors.accent} 100%)`
                    : 'rgba(0,0,0,0.6)',
                  border: `2px solid ${segmentFilled ? colors.primary : 'rgba(255,255,255,0.2)'}`,
                  borderRadius: 4,
                  position: 'relative',
                  overflow: 'hidden',
                  clipPath: 'polygon(10% 0, 100% 0, 90% 100%, 0 100%)',
                }}
              >
                {segmentFilled && (
                  <motion.div
                    animate={{ x: [-50, 60] }}
                    transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                    style={{
                      position: 'absolute',
                      width: 20,
                      height: '100%',
                      background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.5), transparent)',
                      transform: 'skewX(-20deg)',
                    }}
                  />
                )}
              </motion.div>
            );
          })}
        </div>
      </div>

      {/* Move Input Display - Street Fighter Style */}
      <AnimatePresence>
        {isPerformingSpecial && (
          <motion.div
            initial={{ scale: 0, opacity: 0, y: 20 }}
            animate={{ scale: 1, opacity: 1, y: 0 }}
            exit={{ scale: 0, opacity: 0 }}
            transition={{ type: 'spring', damping: 12 }}
            style={{
              background: 'rgba(0,0,0,0.9)',
              padding: '10px 15px',
              borderRadius: 10,
              border: `2px solid ${colors.primary}`,
              boxShadow: `0 0 20px ${colors.primary}60`,
            }}
          >
            <div
              style={{
                display: 'flex',
                gap: 6,
                alignItems: 'center',
                flexDirection: position === 'left' ? 'row' : 'row-reverse',
              }}
            >
              {/* Input icons */}
              {['→', '↓', '↘', 'P'].map((input, i) => (
                <motion.div
                  key={i}
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ delay: i * 0.1 }}
                  style={{
                    width: 28,
                    height: 28,
                    background: `linear-gradient(135deg, ${colors.primary}40, ${colors.secondary}40)`,
                    border: `2px solid ${colors.primary}`,
                    borderRadius: 6,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: 14,
                    fontWeight: 900,
                    color: '#fff',
                    textShadow: `0 0 10px ${colors.primary}`,
                  }}
                >
                  {input}
                </motion.div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Move name display */}
      <AnimatePresence>
        {isPerformingSpecial && (
          <motion.div
            initial={{ x: position === 'left' ? -30 : 30, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: position === 'left' ? -30 : 30, opacity: 0 }}
            style={{
              fontSize: 18,
              fontWeight: 900,
              fontFamily: 'Bebas Neue, sans-serif',
              color: colors.accent,
              textShadow: `0 0 20px ${colors.primary}, 2px 2px 4px rgba(0,0,0,0.9)`,
              letterSpacing: 2,
              background: 'rgba(0,0,0,0.7)',
              padding: '6px 14px',
              borderRadius: 8,
              border: `2px solid ${colors.primary}40`,
            }}
          >
            {player.action.includes('punch') ? '💥 POWER STRIKE!' : '⚡ HEAVY KICK!'}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Quick move reference - Tekken style command list */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 0.4 }}
        whileHover={{ opacity: 1 }}
        transition={{ duration: 0.3 }}
        style={{
          marginTop: 10,
          fontSize: 10,
          fontFamily: 'monospace',
          color: '#666',
          background: 'rgba(0,0,0,0.8)',
          padding: '8px 10px',
          borderRadius: 6,
          border: '1px solid rgba(255,255,255,0.1)',
          maxWidth: 140,
        }}
      >
        <div style={{ marginBottom: 4, fontWeight: 700, color: '#999' }}>MOVES</div>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
          <span>Light</span>
          <span style={{ color: colors.primary }}>{player.id === 1 ? 'Q/E' : 'U/O'}</span>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span>Heavy</span>
          <span style={{ color: colors.primary }}>{player.id === 1 ? 'Z/C' : 'N/M'}</span>
        </div>
      </motion.div>
    </div>
  );
};

export default SpecialMoveIndicator;
