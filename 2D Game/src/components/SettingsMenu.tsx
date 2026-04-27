import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { SoundManager } from '../audio/SoundManager';

const STORAGE_KEYS = {
  master: 'fighterArena_masterVolume',
  sfx: 'fighterArena_sfxVolume',
  music: 'fighterArena_musicVolume',
} as const;

const getSavedVolume = (key: string, fallback: number): number => {
  const saved = localStorage.getItem(key);
  if (saved !== null) {
    const parsed = Number(saved);
    if (!isNaN(parsed)) return parsed;
  }
  return fallback;
};

interface SettingsMenuProps {
  onClose: () => void;
}

export const SettingsMenu: React.FC<SettingsMenuProps> = ({ onClose }) => {
  const [masterVolume, setMasterVolume] = useState(() => getSavedVolume(STORAGE_KEYS.master, 100));
  const [sfxVolume, setSfxVolume] = useState(() => getSavedVolume(STORAGE_KEYS.sfx, 100));
  const [musicVolume, setMusicVolume] = useState(() => getSavedVolume(STORAGE_KEYS.music, 80));

  const playSelectSound = () => SoundManager.play('menuSelect');
  const playHoverSound = () => SoundManager.play('menuHover');

  const handleMasterVolumeChange = (value: number) => {
    setMasterVolume(value);
    SoundManager.setMasterVolume(value / 100);
    localStorage.setItem(STORAGE_KEYS.master, String(value));
  };

  const handleSfxVolumeChange = (value: number) => {
    setSfxVolume(value);
    SoundManager.setSfxVolume(value / 100);
    localStorage.setItem(STORAGE_KEYS.sfx, String(value));
  };

  const handleMusicVolumeChange = (value: number) => {
    setMusicVolume(value);
    SoundManager.setMusicVolume(value / 100);
    localStorage.setItem(STORAGE_KEYS.music, String(value));
  };

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
          minWidth: 500,
          maxWidth: '90vw',
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
            ⚙️ SETTINGS
          </h2>
        </div>

        {/* Volume Controls */}
        <div style={{ marginBottom: 30 }}>
          <VolumeSlider
            label="Master Volume"
            value={masterVolume}
            onChange={handleMasterVolumeChange}
          />
          <VolumeSlider
            label="Sound Effects"
            value={sfxVolume}
            onChange={handleSfxVolumeChange}
          />
          <VolumeSlider
            label="Music"
            value={musicVolume}
            onChange={handleMusicVolumeChange}
          />
        </div>

        {/* Controls Info */}
        <div
          style={{
            background: 'rgba(0, 0, 0, 0.3)',
            borderRadius: 12,
            padding: 20,
            marginBottom: 20,
          }}
        >
          <h3
            style={{
              fontSize: 18,
              color: '#3498db',
              marginBottom: 15,
              fontFamily: 'Bebas Neue, sans-serif',
              letterSpacing: 2,
            }}
          >
            🎮 KEYBOARD CONTROLS
          </h3>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, fontSize: 14 }}>
            <ControlRow label="Move" value="← → / A D" />
            <ControlRow label="Jump" value="↑ / W" />
            <ControlRow label="Block" value="↓ / S" />
            <ControlRow label="Punch" value="J / K" />
            <ControlRow label="Kick" value="U / I" />
            <ControlRow label="Pause" value="ESC" />
          </div>
        </div>

        {/* Buttons */}
        <div style={{ display: 'flex', gap: 15 }}>
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => {
              playSelectSound();
              onClose();
            }}
            onHoverStart={playHoverSound}
            style={{
              flex: 1,
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
            BACK
          </motion.button>
        </div>
      </motion.div>
    </motion.div>
  );
};

interface VolumeSliderProps {
  label: string;
  value: number;
  onChange: (value: number) => void;
}

const VolumeSlider: React.FC<VolumeSliderProps> = ({ label, value, onChange }) => {
  return (
    <div style={{ marginBottom: 20 }}>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          marginBottom: 8,
          fontSize: 14,
          color: '#ecf0f1',
        }}
      >
        <span>{label}</span>
        <span style={{ color: '#3498db', fontWeight: 700 }}>{value}%</span>
      </div>
      <input
        type="range"
        min="0"
        max="100"
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{
          width: '100%',
          height: 8,
          borderRadius: 4,
          background: `linear-gradient(to right, #3498db 0%, #3498db ${value}%, #34495e ${value}%, #34495e 100%)`,
          outline: 'none',
          cursor: 'pointer',
          appearance: 'none',
        }}
      />
    </div>
  );
};

interface ControlRowProps {
  label: string;
  value: string;
}

const ControlRow: React.FC<ControlRowProps> = ({ label, value }) => {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', color: '#ecf0f1' }}>
      <span style={{ color: '#95a5a6' }}>{label}:</span>
      <span style={{ fontWeight: 700, fontFamily: 'monospace' }}>{value}</span>
    </div>
  );
};
