import React, { useState, useMemo } from 'react';
import { motion } from 'framer-motion';

interface EnhancedArenaProps {
  children: React.ReactNode;
  backgroundImage?: string;
}

// Deterministic particle generation - stable between renders
const createStableParticles = () => {
  const particles = [];
  for (let i = 0; i < 20; i++) { // Reduced from 30 to 20
    const seed = i * 13.7; // Deterministic seed
    const normalizedSeed = (Math.sin(seed) + 1) / 2;
    particles.push({
      left: normalizedSeed * 100,
      size: 2 + (normalizedSeed * 3),
      color: i % 3 === 0 ? '#ff4422' : i % 3 === 1 ? '#ff8800' : '#ffcc44',
      duration: 6 + (normalizedSeed * 4),
      delay: normalizedSeed * 5,
      xOffset: Math.sin(i) * 50,
    });
  }
  return particles;
};

const STABLE_PARTICLES = createStableParticles();

export const EnhancedArena: React.FC<EnhancedArenaProps> = ({ 
  children,
  backgroundImage = '/backgrounds/darkstone.png'
}) => {
  const [bgLoaded, setBgLoaded] = useState(false);
  
  // Memoize expensive gradient calculations
  const gradientStyles = useMemo(() => ({
    sky: bgLoaded 
      ? 'linear-gradient(180deg, rgba(5,0,0,0.7) 0%, rgba(18,4,4,0.5) 50%, transparent 100%)'
      : `linear-gradient(180deg, 
          #050000 0%, 
          #120404 20%,
          #200808 40%,
          #180404 70%,
          #050000 100%
        )`,
    floor: `linear-gradient(180deg, 
      #120808 0%,
      #080404 30%,
      #020202 100%
    )`,
    spotlight: `
      radial-gradient(ellipse 80% 50% at 20% 100%, rgba(255, 100, 0, 0.12) 0%, transparent 50%),
      radial-gradient(ellipse 80% 50% at 80% 100%, rgba(220, 30, 30, 0.12) 0%, transparent 50%),
      radial-gradient(ellipse 100% 60% at 50% 110%, rgba(200, 60, 0, 0.18) 0%, transparent 50%)
    `,
    vignette: `radial-gradient(ellipse at center, transparent 30%, rgba(0,0,0,0.75) 100%)`
  }), [bgLoaded]);

  return (
    <div
      style={{
        position: 'relative',
        width: '100%',
        height: '100%',
        overflow: 'hidden',
        background: '#000',
      }}
    >
      {/* Background Image Layer */}
      <div
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundImage: `url(${backgroundImage})`,
          backgroundSize: 'cover',
          backgroundPosition: 'center',
          backgroundRepeat: 'no-repeat',
          opacity: bgLoaded ? 0.7 : 0,
          transition: 'opacity 0.5s',
        }}
      />
      <img
        src={backgroundImage}
        alt="background"
        onLoad={() => setBgLoaded(true)}
        onError={() => setBgLoaded(false)}
        style={{ display: 'none' }}
      />

      {/* Dramatic gradient sky overlay */}
      <div
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          height: '88%', // Match groundY
          background: gradientStyles.sky,
          pointerEvents: 'none',
        }}
      />

      {/* Animated energy waves in background */}
      <div
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          height: '88%', // Match groundY
          overflow: 'hidden',
          opacity: 0.3,
        }}
      >
        {[...Array(5)].map((_, i) => (
          <motion.div
            key={i}
            animate={{
              x: ['-100%', '200%'],
            }}
            transition={{
              duration: 8 + i * 2,
              repeat: Infinity,
              ease: 'linear',
              delay: i * 1.5,
            }}
            style={{
              position: 'absolute',
              top: 50 + i * 80,
              width: '50%',
              height: 3,
              background: `linear-gradient(90deg, transparent, ${i % 2 === 0 ? '#ff4400' : '#ff8822'}, transparent)`,
              filter: 'blur(2px)',
            }}
          />
        ))}
      </div>

      {/* Dramatic spotlight effects */}
      <div
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: gradientStyles.spotlight,
          pointerEvents: 'none',
        }}
      />

      {/* Particle dust floating - optimized */}
      <div style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, overflow: 'hidden', pointerEvents: 'none' }}>
        {STABLE_PARTICLES.map((particle, i) => (
          <motion.div
            key={i}
            animate={{
              y: ['100vh', '-20px'], // Use viewport height for responsiveness
              x: [0, particle.xOffset],
              opacity: [0, 0.6, 0],
            }}
            transition={{
              duration: particle.duration,
              repeat: Infinity,
              delay: particle.delay,
              ease: 'linear',
            }}
            style={{
              position: 'absolute',
              left: `${particle.left}%`,
              width: particle.size,
              height: particle.size,
              background: particle.color,
              borderRadius: '50%',
              filter: 'blur(1px)',
            }}
          />
        ))}
      </div>

      {/* Epic arena floor */}
      <div
        style={{
          position: 'absolute',
          top: '88%', // Match groundY
          left: 0,
          right: 0,
          bottom: 0,
          background: gradientStyles.floor,
          borderTop: '4px solid',
          borderImage: 'linear-gradient(90deg, #8b0000, #e74c3c, #ff8c00, #e74c3c, #8b0000) 1',
        }}
      >
        {/* Glowing floor line */}
        <motion.div
          animate={{
            boxShadow: [
              '0 0 20px #c0200a, 0 0 40px #c0200a',
              '0 0 40px #ff5500, 0 0 80px #ff8800',
              '0 0 20px #c0200a, 0 0 40px #e74c3c',
              '0 0 20px #c0200a, 0 0 40px #c0200a',
            ],
          }}
          transition={{ duration: 4, repeat: Infinity }}
          style={{
            position: 'absolute',
            top: 0,
            left: '10%',
            right: '10%',
            height: 2,
            background: 'linear-gradient(90deg, transparent, #ff8844, transparent)',
          }}
        />

        {/* Grid pattern on floor */}
        <svg
          width="100%"
          height="100%"
          style={{ position: 'absolute', top: 10, opacity: 0.15 }}
        >
          <defs>
            <pattern id="floorGrid" width="80" height="40" patternUnits="userSpaceOnUse">
              <path d="M 80 0 L 40 40 L 0 0" stroke="#8b2010" strokeWidth="1" fill="none" />
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#floorGrid)" />
        </svg>

        {/* Reflection effect */}
        <div
          style={{
            position: 'absolute',
            top: 20,
            left: 0,
            right: 0,
            height: 60,
            background: 'linear-gradient(180deg, rgba(255,255,255,0.05) 0%, transparent 100%)',
            pointerEvents: 'none',
          }}
        />
      </div>

      {/* Fire/ember glow rising from floor */}
      <div
        style={{
          position: 'absolute',
          bottom: 0,
          left: 0,
          right: 0,
          height: '35%',
          background: `
            radial-gradient(ellipse 90% 60% at 50% 110%, rgba(200, 50, 0, 0.22) 0%, transparent 65%),
            radial-gradient(ellipse 50% 40% at 20% 110%, rgba(245, 130, 0, 0.12) 0%, transparent 60%),
            radial-gradient(ellipse 50% 40% at 80% 110%, rgba(220, 40, 0, 0.12) 0%, transparent 60%)
          `,
          pointerEvents: 'none',
        }}
      />

      {/* Player side indicators - dramatic lighting */}
      <motion.div
        animate={{ opacity: [0.3, 0.6, 0.3] }}
        transition={{ duration: 2, repeat: Infinity }}
        style={{
          position: 'absolute',
          left: 0,
          top: 0,
          width: 8,
          height: '100%',
          background: 'linear-gradient(180deg, #f5a623 0%, transparent 30%, transparent 70%, #f5a623 100%)',
          boxShadow: '0 0 30px #f5a623, 0 0 60px #c07012',
        }}
      />
      <motion.div
        animate={{ opacity: [0.3, 0.6, 0.3] }}
        transition={{ duration: 2, repeat: Infinity, delay: 1 }}
        style={{
          position: 'absolute',
          right: 0,
          top: 0,
          width: 8,
          height: '100%',
          background: 'linear-gradient(180deg, #e74c3c 0%, transparent 30%, transparent 70%, #e74c3c 100%)',
          boxShadow: '0 0 30px #e74c3c, 0 0 60px #e74c3c',
        }}
      />

      {/* Center VS marker */}
      <div
        style={{
          position: 'absolute',
          left: '50%',
          top: '88%', // Match groundY
          transform: 'translateX(-50%) translateY(-5px)',
          width: 60,
          height: 60,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 20, repeat: Infinity, ease: 'linear' }}
          style={{
            width: 50,
            height: 50,
            border: '2px solid rgba(255,255,255,0.1)',
            borderRadius: '0',
            clipPath: 'polygon(30% 0, 100% 0, 70% 100%, 0 100%)',
            position: 'absolute',
          }}
        />
        <motion.div
          animate={{ rotate: -360 }}
          transition={{ duration: 15, repeat: Infinity, ease: 'linear' }}
          style={{
            width: 35,
            height: 35,
            border: '2px solid rgba(231,76,60,0.5)',
            borderRadius: '0',
            clipPath: 'polygon(30% 0, 100% 0, 70% 100%, 0 100%)',
            position: 'absolute',
          }}
        />
      </div>

      {/* Epic vignette */}
      <div
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: gradientStyles.vignette,
          pointerEvents: 'none',
        }}
      />

      {/* Corner decorations */}
      <svg style={{ position: 'absolute', top: 0, left: 0, width: 100, height: 100, opacity: 0.5 }}>
        <path d="M 0 80 L 0 0 L 80 0" stroke="#f5a623" strokeWidth="3" fill="none" />
        <path d="M 0 60 L 0 0 L 60 0" stroke="#f5a623" strokeWidth="1" fill="none" opacity="0.5" />
      </svg>
      <svg style={{ position: 'absolute', top: 0, right: 0, width: 100, height: 100, opacity: 0.5, transform: 'scaleX(-1)' }}>
        <path d="M 0 80 L 0 0 L 80 0" stroke="#e74c3c" strokeWidth="3" fill="none" />
        <path d="M 0 60 L 0 0 L 60 0" stroke="#e74c3c" strokeWidth="1" fill="none" opacity="0.5" />
      </svg>

      {/* Game elements */}
      {children}
    </div>
  );
};

export default EnhancedArena;
