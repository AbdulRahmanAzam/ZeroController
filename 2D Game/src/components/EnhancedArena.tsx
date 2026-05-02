import React, { useState, useMemo } from 'react';
import { motion } from 'framer-motion';

interface EnhancedArenaProps {
  children: React.ReactNode;
  backgroundImage?: string;
  performanceMode?: boolean;
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
      color: i % 3 === 0 ? '#a855f7' : i % 3 === 1 ? '#38bdf8' : '#f5a623',
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
  backgroundImage = '/backgrounds/cavern1.png',
  performanceMode = false,
}) => {
  const [bgLoaded, setBgLoaded] = useState(false);
  
  // Memoize expensive gradient calculations
  const gradientStyles = useMemo(() => ({
    sky: bgLoaded
      ? 'linear-gradient(180deg, rgba(0,0,8,0.75) 0%, rgba(10,4,20,0.55) 40%, transparent 100%)'
      : `linear-gradient(180deg,
          #000008 0%,
          #080418 20%,
          #120830 40%,
          #0a0420 70%,
          #000008 100%
        )`,
    floor: `linear-gradient(180deg,
      #1a1028 0%,
      #0d0818 30%,
      #050308 100%
    )`,
    spotlight: `
      radial-gradient(ellipse 60% 40% at 15% 90%, rgba(80, 30, 200, 0.18) 0%, transparent 55%),
      radial-gradient(ellipse 60% 40% at 85% 90%, rgba(200, 30, 80, 0.18) 0%, transparent 55%),
      radial-gradient(ellipse 80% 50% at 50% 100%, rgba(255, 180, 0, 0.14) 0%, transparent 60%),
      radial-gradient(ellipse 100% 30% at 50% 0%,  rgba(30, 0, 80, 0.35) 0%, transparent 100%)
    `,
    vignette: `radial-gradient(ellipse at center, transparent 25%, rgba(0,0,0,0.82) 100%)`
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
          opacity: bgLoaded ? 0.88 : 0,
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

      {!performanceMode && (
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
                background: `linear-gradient(90deg, transparent, ${i % 2 === 0 ? '#a855f7' : '#38bdf8'}, transparent)`,
                filter: 'blur(2px)',
              }}
            />
          ))}
        </div>
      )}

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

      {!performanceMode && (
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
      )}

      {/* Arena floor */}
      <div
        style={{
          position: 'absolute',
          top: '88%',
          left: 0,
          right: 0,
          bottom: 0,
          background: gradientStyles.floor,
          borderTop: '4px solid',
          borderImage: 'linear-gradient(90deg, #2a006a, #9b30ff, #f5a623, #9b30ff, #2a006a) 1',
        }}
      >
        {/* Glowing floor line */}
        <div
          style={{
            position: 'absolute',
            top: 0,
            left: '8%',
            right: '8%',
            height: 3,
            background: 'linear-gradient(90deg, transparent, #c084fc, #f5a623, #c084fc, transparent)',
            boxShadow: performanceMode
              ? '0 0 18px #7c3aed, 0 0 30px #7c3aed'
              : '0 0 25px #a855f7, 0 0 55px #7c3aed, 0 0 80px #6d28d9',
          }}
        />

        {/* Stone tile grid */}
        <svg
          width="100%"
          height="100%"
          style={{ position: 'absolute', top: 8, opacity: 0.2 }}
        >
          <defs>
            <pattern id="stoneTile" width="60" height="30" patternUnits="userSpaceOnUse">
              <rect width="60" height="30" fill="none" stroke="#6d28d9" strokeWidth="0.8" />
              <rect x="30" y="0" width="30" height="15" fill="none" stroke="#6d28d9" strokeWidth="0.5" opacity="0.5" />
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#stoneTile)" />
        </svg>

        {/* Sheen reflection */}
        <div
          style={{
            position: 'absolute',
            top: 10,
            left: 0,
            right: 0,
            height: 50,
            background: 'linear-gradient(180deg, rgba(168,85,247,0.08) 0%, transparent 100%)',
            pointerEvents: 'none',
          }}
        />
      </div>

      {/* Magical energy glow rising from floor */}
      <div
        style={{
          position: 'absolute',
          bottom: 0,
          left: 0,
          right: 0,
          height: '35%',
          background: `
            radial-gradient(ellipse 90% 60% at 50% 110%, rgba(120, 40, 220, 0.25) 0%, transparent 65%),
            radial-gradient(ellipse 50% 40% at 15% 110%, rgba(245, 130, 0, 0.14) 0%, transparent 60%),
            radial-gradient(ellipse 50% 40% at 85% 110%, rgba(200, 30, 100, 0.14) 0%, transparent 60%)
          `,
          pointerEvents: 'none',
        }}
      />

      {/* Player side indicators — P1 blue, P2 red */}
      <div
        style={{
          position: 'absolute',
          left: 0,
          top: 0,
          width: 8,
          height: '100%',
          background: 'linear-gradient(180deg, #00d4ff 0%, transparent 30%, transparent 70%, #00d4ff 100%)',
          boxShadow: '0 0 30px #00d4ff, 0 0 60px #0088aa',
        }}
      />
      <div
        style={{
          position: 'absolute',
          right: 0,
          top: 0,
          width: 8,
          height: '100%',
          background: 'linear-gradient(180deg, #ff4757 0%, transparent 30%, transparent 70%, #ff4757 100%)',
          boxShadow: '0 0 30px #ff4757, 0 0 60px #cc0022',
        }}
      />

      {/* Center VS marker */}
      <div
        style={{
          position: 'absolute',
          left: '50%',
          top: '88%',
          transform: 'translateX(-50%) translateY(-5px)',
          width: 60,
          height: 60,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <div
          style={{
            width: 50,
            height: 50,
            border: '2px solid rgba(168,85,247,0.25)',
            clipPath: 'polygon(30% 0, 100% 0, 70% 100%, 0 100%)',
            position: 'absolute',
          }}
        />
        <div
          style={{
            width: 35,
            height: 35,
            border: '2px solid rgba(245,166,35,0.5)',
            clipPath: 'polygon(30% 0, 100% 0, 70% 100%, 0 100%)',
            position: 'absolute',
          }}
        />
      </div>

      {/* Vignette */}
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
      <svg style={{ position: 'absolute', top: 0, left: 0, width: 100, height: 100, opacity: 0.6 }}>
        <path d="M 0 80 L 0 0 L 80 0" stroke="#00d4ff" strokeWidth="3" fill="none" />
        <path d="M 0 60 L 0 0 L 60 0" stroke="#00d4ff" strokeWidth="1" fill="none" opacity="0.4" />
      </svg>
      <svg style={{ position: 'absolute', top: 0, right: 0, width: 100, height: 100, opacity: 0.6, transform: 'scaleX(-1)' }}>
        <path d="M 0 80 L 0 0 L 80 0" stroke="#ff4757" strokeWidth="3" fill="none" />
        <path d="M 0 60 L 0 0 L 60 0" stroke="#ff4757" strokeWidth="1" fill="none" opacity="0.4" />
      </svg>

      {/* Game elements */}
      {children}
    </div>
  );
};

export default EnhancedArena;
