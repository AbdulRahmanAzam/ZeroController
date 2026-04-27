import React, { useState, useMemo } from 'react';
import { motion } from 'framer-motion';

interface HitEffectsProps {
  player: 1 | 2;
  type: 'hit' | 'block';
  position: { x: number; y: number };
}

// Deterministic value generation - stable across renders
const getDeterministicValue = (seed: number): number => {
  const value = Math.sin(seed * 12.9898) * 43758.5453;
  return value - Math.floor(value);
};

// Pre-calculated stable values for particles
const STAR_PARTICLES = Array.from({ length: 6 }, (_, i) => {
  const angle = (i / 6) * Math.PI * 2;
  const distance = 40 + getDeterministicValue(i + 1) * 20;
  const hue = getDeterministicValue(i + 20) * 360;
  return { angle, distance, hue };
});

const SPARKLE_PARTICLES = Array.from({ length: 4 }, (_, i) => {
  const angle = (i / 4) * Math.PI * 2 + Math.PI / 4;
  const distance = 25 + getDeterministicValue(i + 50) * 15;
  return { angle, distance };
});

export const HitEffects: React.FC<HitEffectsProps> = ({ type, position }) => {
  const color = type === 'hit' ? '#ff6b6b' : '#74b9ff';
  const [useSprites] = useState(true);
  
  // Memoize expensive style calculations
  const effectStyles = useMemo(() => ({
    centralBurst: {
      position: 'absolute' as const,
      width: 60,
      height: 60,
      left: -30,
      top: -30,
      background: `radial-gradient(circle, ${color} 0%, transparent 70%)`,
      borderRadius: '50%',
    },
    impactText: {
      position: 'absolute' as const,
      left: -30,
      top: -20,
      fontSize: 24,
      fontWeight: 'bold' as const,
      color: '#fff',
      textShadow: `0 0 10px ${color}, 0 0 20px ${color}`,
      whiteSpace: 'nowrap' as const,
    }
  }), [color]);
  
  return (
    <motion.div
      initial={{ opacity: 1, scale: 0.5 }}
      animate={{ opacity: 0, scale: 2 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.3, ease: 'easeOut' }}
      style={{
        position: 'absolute',
        left: position.x,
        top: position.y,
        pointerEvents: 'none',
        zIndex: 60,
      }}
    >
      {type === 'hit' ? (
        // Hit spark effect
        <>
          {useSprites ? (
            // Sprite-based hit effect
            <>
              {/* Explosion sprite */}
              <motion.div
                initial={{ scale: 0, rotate: 0 }}
                animate={{ scale: [0, 1.5, 1], rotate: 360 }}
                transition={{ duration: 0.3 }}
                style={{
                  position: 'absolute',
                  left: -40,
                  top: -40,
                  width: 80,
                  height: 80,
                }}
              >
                <img
                  src="/sprites/effects/blue-flare.png"
                  alt="hit"
                  style={{
                    width: '100%',
                    height: '100%',
                    filter: `hue-rotate(${type === 'hit' ? '300deg' : '0deg'}) brightness(1.5)`,
                    imageRendering: 'pixelated',
                  }}
                />
              </motion.div>

              {/* Star particles - optimized */}
              {STAR_PARTICLES.map((particle, i) => (
                <motion.div
                  key={`star-${i}`}
                  initial={{ x: 0, y: 0, opacity: 1, scale: 0 }}
                  animate={{ 
                    x: Math.cos(particle.angle) * particle.distance,
                    y: Math.sin(particle.angle) * particle.distance,
                    opacity: 0,
                    scale: 1,
                    rotate: 360,
                  }}
                  transition={{ duration: 0.4, delay: i * 0.02 }}
                  style={{
                    position: 'absolute',
                    left: -8,
                    top: -8,
                    width: 16,
                    height: 16,
                  }}
                >
                  <img
                      src="/sprites/effects/star.png"
                      alt="star"
                      style={{
                        width: '100%',
                        height: '100%',
                        filter: `hue-rotate(${particle.hue}deg)`,
                        imageRendering: 'pixelated',
                      }}
                    />
                </motion.div>
              ))}

              {/* Sparkles - optimized */}
              {SPARKLE_PARTICLES.map((particle, i) => (
                <motion.div
                  key={`sparkle-${i}`}
                  initial={{ x: 0, y: 0, opacity: 1, scale: 0.5 }}
                  animate={{ 
                    x: Math.cos(particle.angle) * particle.distance,
                    y: Math.sin(particle.angle) * particle.distance,
                    opacity: 0,
                    scale: 1.2,
                  }}
                  transition={{ duration: 0.3, delay: 0.05 }}
                  style={{
                    position: 'absolute',
                    left: -10,
                    top: -10,
                    width: 20,
                    height: 20,
                  }}
                >
                  <img
                    src="/sprites/effects/sparkle1.png"
                    alt="sparkle"
                    style={{
                      width: '100%',
                      height: '100%',
                      imageRendering: 'pixelated',
                    }}
                  />
                </motion.div>
              ))}
            </>
          ) : (
            // Original SVG-based hit effect
            <>
          {/* Central burst - memoized */}
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: [0, 1.5, 0] }}
            transition={{ duration: 0.2 }}
            style={effectStyles.centralBurst}
          />
          
          {/* Star burst lines */}
          {Array.from({ length: 8 }).map((_, i) => (
            <motion.div
              key={i}
              initial={{ scale: 0, rotate: i * 45 }}
              animate={{ scale: [0, 1, 0], rotate: i * 45 }}
              transition={{ duration: 0.25, delay: 0.05 }}
              style={{
                position: 'absolute',
                width: 4,
                height: 40,
                left: -2,
                top: -20,
                background: `linear-gradient(180deg, ${color} 0%, transparent 100%)`,
                transformOrigin: 'center center',
              }}
            />
          ))}
          </> 
          )}
          
          {/* Impact text - memoized */}
          <motion.div
            initial={{ y: 0, opacity: 1, scale: 0.5 }}
            animate={{ y: -50, opacity: 0, scale: 1.5 }}
            transition={{ duration: 0.4 }}
            style={effectStyles.impactText}
          >
            HIT!
          </motion.div>
        </>
      ) : (
        // Block effect
        <>
          {useSprites && (
            // Sprite-based shield effect  
            <motion.div
              initial={{ scale: 0, opacity: 1 }}
              animate={{ scale: 1.5, opacity: 0 }}
              transition={{ duration: 0.3 }}
              style={{
                position: 'absolute',
                left: -50,
                top: -50,
                width: 100,
                height: 100,
              }}
            >
              <img
                src="/sprites/effects/white-flare.png"
                alt="shield"
                style={{
                  width: '100%',
                  height: '100%',
                  filter: 'hue-rotate(200deg) brightness(1.2)',
                  imageRendering: 'pixelated',
                }}
              />
            </motion.div>
          )}

          {/* Shield ripple */}
          <motion.div
            initial={{ scale: 0.5, opacity: 0.8 }}
            animate={{ scale: 2, opacity: 0 }}
            transition={{ duration: 0.3 }}
            style={{
              position: 'absolute',
              width: 80,
              height: 80,
              left: -40,
              top: -40,
              border: `4px solid ${color}`,
              borderRadius: '50%',
              boxShadow: `0 0 20px ${color}`,
            }}
          />
          
          {/* Inner shield */}
          <motion.div
            initial={{ scale: 0.8, opacity: 1 }}
            animate={{ scale: 1.5, opacity: 0 }}
            transition={{ duration: 0.2, delay: 0.1 }}
            style={{
              position: 'absolute',
              width: 50,
              height: 50,
              left: -25,
              top: -25,
              background: `radial-gradient(circle, ${color}40 0%, transparent 70%)`,
              borderRadius: '50%',
            }}
          />
          
          {/* Block text */}
          <motion.div
            initial={{ y: 0, opacity: 1, scale: 0.8 }}
            animate={{ y: -40, opacity: 0, scale: 1.2 }}
            transition={{ duration: 0.35 }}
            style={{
              position: 'absolute',
              left: -35,
              top: -15,
              fontSize: 20,
              fontWeight: 'bold',
              color: '#74b9ff',
              textShadow: '0 0 10px #0984e3',
              whiteSpace: 'nowrap',
            }}
          >
            BLOCKED!
          </motion.div>
        </>
      )}
    </motion.div>
  );
};

export default HitEffects;
