import React from 'react';
import { motion } from 'framer-motion';
import { SoundManager } from '../audio/SoundManager';

interface HowToPlayProps {
  onClose: () => void;
}

export const HowToPlay: React.FC<HowToPlayProps> = ({ onClose }) => {
  const playSelectSound = () => SoundManager.play('menuSelect');
  const playHoverSound = () => SoundManager.play('menuHover');

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100vw',
        height: '100vh',
        background: 'rgba(0, 0, 0, 0.95)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 10000,
        padding: 20,
        overflow: 'auto',
      }}
    >
      <motion.div
        initial={{ scale: 0.8, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ type: 'spring', damping: 15 }}
        style={{
          background: 'linear-gradient(135deg, #2c3e50 0%, #34495e 100%)',
          border: '3px solid #3498db',
          borderRadius: 20,
          padding: 40,
          maxWidth: 800,
          maxHeight: '90vh',
          overflow: 'auto',
          boxShadow: '0 20px 60px rgba(0, 0, 0, 0.5)',
        }}
      >
        {/* Header */}
        <div style={{ marginBottom: 30, textAlign: 'center' }}>
          <h2
            style={{
              fontSize: 36,
              fontWeight: 900,
              fontFamily: 'Bebas Neue, sans-serif',
              color: '#3498db',
              letterSpacing: 4,
              margin: 0,
            }}
          >
            📖 HOW TO PLAY
          </h2>
        </div>

        {/* Game Objective */}
        <Section title="🎯 OBJECTIVE">
          <p>
            Defeat your opponent by reducing their health to zero! Win 2 out of 3 rounds to claim
            victory in the match.
          </p>
        </Section>

        {/* Controls */}
        <Section title="🎮 CONTROLS">
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 15 }}>
            <ControlCard
              title="PLAYER 1"
              controls={[
                { action: 'Move Left', key: '← or A' },
                { action: 'Move Right', key: '→ or D' },
                { action: 'Jump', key: '↑ or W' },
                { action: 'Block', key: '↓ or S' },
                { action: 'Left Punch', key: 'J' },
                { action: 'Right Punch', key: 'K' },
                { action: 'Left Kick', key: 'U' },
                { action: 'Right Kick', key: 'I' },
              ]}
            />
            <ControlCard
              title="PLAYER 2"
              controls={[
                { action: 'Move Left', key: 'Numpad 4' },
                { action: 'Move Right', key: 'Numpad 6' },
                { action: 'Jump', key: 'Numpad 8' },
                { action: 'Block', key: 'Numpad 5' },
                { action: 'Punch', key: 'Numpad 1/2' },
                { action: 'Kick', key: 'Numpad 7/9' },
              ]}
            />
          </div>
        </Section>

        {/* Game Mechanics */}
        <Section title="⚔️ COMBAT MECHANICS">
          <ul style={{ paddingLeft: 20, lineHeight: 1.8 }}>
            <li>
              <strong>Punches:</strong> Fast attacks with moderate damage. Chain them together for
              combos!
            </li>
            <li>
              <strong>Kicks:</strong> Slower but deal more damage. Perfect for finishing moves.
            </li>
            <li>
              <strong>Blocking:</strong> Hold block to reduce incoming damage. Timing is
              everything!
            </li>
            <li>
              <strong>Combos:</strong> Land consecutive hits without being hit to build your combo
              counter.
            </li>
            <li>
              <strong>Special Meter:</strong> Fills as you deal damage. (Future special moves!)
            </li>
          </ul>
        </Section>

        {/* Camera Mode Info */}
        <Section title="📷 CAMERA MODE (DLP/ANN)">
          <p>
            <strong style={{ color: '#e74c3c' }}>Coming Soon!</strong> This game is designed to
            work with camera-based pose detection using Deep Learning and Artificial Neural
            Networks.
          </p>
          <p>
            Stand in front of your camera and control your fighter with real body movements - punch,
            kick, jump, and block in real life to control your character!
          </p>
          <p style={{ fontSize: 14, color: '#95a5a6', marginTop: 10 }}>
            Required: Pose detection API running on <code>ws://localhost:8000/ws/pose/</code>
          </p>
        </Section>

        {/* Tips */}
        <Section title="💡 TIPS">
          <ul style={{ paddingLeft: 20, lineHeight: 1.8 }}>
            <li>Mix up your attacks - don't be predictable!</li>
            <li>Use blocking strategically to reduce damage and find openings</li>
            <li>Aerial attacks are powerful but leave you vulnerable</li>
            <li>Watch the round timer - play defensively if you're ahead on health</li>
            <li>Build combos to fill your special meter faster</li>
          </ul>
        </Section>

        {/* Close Button */}
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={() => {
            playSelectSound();
            onClose();
          }}
          onHoverStart={playHoverSound}
          style={{
            width: '100%',
            marginTop: 20,
            background: 'linear-gradient(135deg, #27ae60 0%, #2ecc71 100%)',
            border: '2px solid #27ae60',
            borderRadius: 10,
            padding: '15px 25px',
            color: '#fff',
            fontSize: 18,
            fontWeight: 700,
            fontFamily: 'Bebas Neue, sans-serif',
            letterSpacing: 2,
            cursor: 'pointer',
            boxShadow: '0 6px 20px rgba(39, 174, 96, 0.4)',
          }}
        >
          GOT IT!
        </motion.button>
      </motion.div>
    </motion.div>
  );
};

interface SectionProps {
  title: string;
  children: React.ReactNode;
}

const Section: React.FC<SectionProps> = ({ title, children }) => {
  return (
    <div
      style={{
        marginBottom: 25,
        background: 'rgba(0, 0, 0, 0.3)',
        borderRadius: 12,
        padding: 20,
      }}
    >
      <h3
        style={{
          fontSize: 20,
          color: '#3498db',
          marginBottom: 15,
          fontFamily: 'Bebas Neue, sans-serif',
          letterSpacing: 2,
        }}
      >
        {title}
      </h3>
      <div style={{ fontSize: 15, color: '#ecf0f1', lineHeight: 1.6 }}>{children}</div>
    </div>
  );
};

interface ControlCardProps {
  title: string;
  controls: Array<{ action: string; key: string }>;
}

const ControlCard: React.FC<ControlCardProps> = ({ title, controls }) => {
  return (
    <div
      style={{
        background: 'rgba(0, 0, 0, 0.2)',
        borderRadius: 8,
        padding: 15,
      }}
    >
      <h4
        style={{
          fontSize: 16,
          color: '#e74c3c',
          marginBottom: 10,
          fontFamily: 'Bebas Neue, sans-serif',
          letterSpacing: 1,
        }}
      >
        {title}
      </h4>
      <div style={{ fontSize: 13 }}>
        {controls.map((control, i) => (
          <div
            key={i}
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              marginBottom: 6,
              color: '#ecf0f1',
            }}
          >
            <span style={{ color: '#95a5a6' }}>{control.action}:</span>
            <span style={{ fontWeight: 700, fontFamily: 'monospace' }}>{control.key}</span>
          </div>
        ))}
      </div>
    </div>
  );
};
