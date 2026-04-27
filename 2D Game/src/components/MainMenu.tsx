import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { SoundManager } from '../audio/SoundManager';
import { useGameStore } from '../store/gameStore';
import type { AIDifficulty } from '../types/game';

interface MainMenuProps {
  onStartGame: () => void;
  onOpenSettings: () => void;
  onOpenHowToPlay: () => void;
}

export const MainMenu: React.FC<MainMenuProps> = ({
  onStartGame,
  onOpenSettings,
  onOpenHowToPlay,
}) => {
  const { setGameMode, setAIDifficulty } = useGameStore();

  // 'mode'       → pick VS PLAYER or VS AI
  // 'difficulty' → pick Easy / Medium / Hard (VS AI only)
  type MenuStep = 'mode' | 'difficulty';
  const [step, setStep] = useState<MenuStep>('mode');

  const playSelectSound = () => SoundManager.play('menuSelect');
  const playHoverSound  = () => SoundManager.play('menuHover');

  const handleVsPlayer = () => {
    playSelectSound();
    setGameMode('vs_player');
    onStartGame();
  };

  const handleVsAI = () => {
    playSelectSound();
    setGameMode('vs_ai');
    setStep('difficulty');
  };

  const handleDifficulty = (diff: AIDifficulty) => {
    playSelectSound();
    setAIDifficulty(diff);
    onStartGame();
  };

  const handleBack = () => {
    playSelectSound();
    setStep('mode');
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      style={{
        position: 'fixed',
        inset: 0,
        background: '#060606',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 9999,
        overflow: 'hidden',
      }}
    >
      {/* Ground fire glow — arena floor lighting */}
      <div style={{ position: 'absolute', inset: 0, pointerEvents: 'none', background: 'radial-gradient(ellipse 100% 55% at 50% 115%, rgba(200,30,0,0.55) 0%, rgba(100,10,0,0.3) 35%, transparent 65%)' }} />
      {/* Ceiling spotlight */}
      <div style={{ position: 'absolute', inset: 0, pointerEvents: 'none', background: 'radial-gradient(ellipse 70% 45% at 50% -5%, rgba(255,100,0,0.1) 0%, transparent 60%)' }} />
      {/* Scanlines */}
      <div style={{ position: 'absolute', inset: 0, pointerEvents: 'none', zIndex: 1, background: 'repeating-linear-gradient(0deg, transparent 0px, transparent 2px, rgba(0,0,0,0.22) 2px, rgba(0,0,0,0.22) 3px)' }} />
      {/* Side vignette */}
      <div style={{ position: 'absolute', inset: 0, pointerEvents: 'none', background: 'linear-gradient(90deg, rgba(0,0,0,0.72) 0%, transparent 22%, transparent 78%, rgba(0,0,0,0.72) 100%)' }} />
      {/* Horizontal accent lines */}
      <div style={{ position: 'absolute', top: '11%', left: 0, right: 0, height: 1, background: 'linear-gradient(90deg, transparent, rgba(200,30,0,0.55), transparent)', pointerEvents: 'none' }} />
      <div style={{ position: 'absolute', bottom: '10%', left: 0, right: 0, height: 1, background: 'linear-gradient(90deg, transparent, rgba(200,30,0,0.38), transparent)', pointerEvents: 'none' }} />

      {/* ── All content (above scanlines) ── */}
      <div style={{ position: 'relative', zIndex: 2, display: 'flex', flexDirection: 'column', alignItems: 'center', width: '100%' }}>

        {/* Title block */}
        <motion.div
          initial={{ y: -50, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.1, type: 'spring', damping: 14 }}
          style={{ textAlign: 'center', marginBottom: 0 }}
        >
          <div style={{ fontSize: 12, letterSpacing: 8, color: '#e74c3c', fontFamily: "'Orbitron', sans-serif", marginBottom: 6, opacity: 0.9 }}>
            ⚡ ZERO CONTROLLER ⚡
          </div>
          <motion.h1
            animate={{
              textShadow: [
                '0 0 25px rgba(220,30,0,0.9), 5px 5px 0px #000',
                '0 0 50px rgba(220,30,0,1.0), 5px 5px 0px #000, 0 0 90px rgba(200,30,0,0.4)',
                '0 0 25px rgba(220,30,0,0.9), 5px 5px 0px #000',
              ],
            }}
            transition={{ duration: 2.5, repeat: Infinity }}
            style={{
              fontSize: 'clamp(54px, 9vw, 104px)',
              fontWeight: 900,
              fontFamily: "'Bebas Neue', sans-serif",
              color: '#fff',
              letterSpacing: 14,
              margin: 0,
              lineHeight: 1,
              WebkitTextStroke: '2px rgba(255,40,0,0.35)',
            }}
          >
            FIGHTER ARENA
          </motion.h1>
          {/* Gold-to-red underline */}
          <motion.div
            initial={{ scaleX: 0 }}
            animate={{ scaleX: 1 }}
            transition={{ delay: 0.4, duration: 0.7 }}
            style={{ height: 3, background: 'linear-gradient(90deg, transparent, #e74c3c, #ffd700, #e74c3c, transparent)', marginTop: 8, borderRadius: 2 }}
          />
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.6 }}
            style={{ fontSize: 12, color: '#666', fontFamily: "'Orbitron', sans-serif", letterSpacing: 5, marginTop: 8 }}
          >
            MOTION-CONTROLLED COMBAT
          </motion.p>
        </motion.div>

        {/* Section label */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
          style={{ display: 'flex', alignItems: 'center', gap: 14, margin: '28px 0 20px' }}
        >
          <div style={{ width: 80, height: 1, background: 'linear-gradient(90deg, transparent, rgba(231,76,60,0.6))' }} />
          <span style={{ color: '#e74c3c', fontFamily: "'Orbitron', sans-serif", fontSize: 10, letterSpacing: 5 }}>
            {step === 'mode' ? 'SELECT MODE' : 'SELECT DIFFICULTY'}
          </span>
          <div style={{ width: 80, height: 1, background: 'linear-gradient(90deg, rgba(231,76,60,0.6), transparent)' }} />
        </motion.div>

        {/* Step screens */}
        <AnimatePresence mode="wait">
          {step === 'mode' ? (
            <motion.div
              key="mode"
              initial={{ opacity: 0, x: -40 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 40 }}
              transition={{ duration: 0.18 }}
              style={{ display: 'flex', flexDirection: 'column', gap: 10, width: 400 }}
            >
              <FightButton label="VS AI"       sublabel="FIGHT AN AI OPPONENT"       icon="🤖" onClick={handleVsAI}     onHover={playHoverSound} primary />
              <FightButton label="VS PLAYER"   sublabel="TWO PLAYERS — ONE KEYBOARD" icon="👤" onClick={handleVsPlayer} onHover={playHoverSound} />
              <div style={{ height: 6 }} />
              <FightButton label="HOW TO PLAY" sublabel="" icon="📖" onClick={() => { playSelectSound(); onOpenHowToPlay(); }} onHover={playHoverSound} small />
              <FightButton label="SETTINGS"    sublabel="" icon="⚙"  onClick={() => { playSelectSound(); onOpenSettings(); }}  onHover={playHoverSound} small />
            </motion.div>
          ) : (
            <motion.div
              key="difficulty"
              initial={{ opacity: 0, x: 40 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -40 }}
              transition={{ duration: 0.18 }}
              style={{ display: 'flex', flexDirection: 'column', gap: 10, width: 440 }}
            >
              <TierButton label="EASY"   sub="SLOW REACTIONS · MOSTLY RANDOM"        tier="easy"   onClick={() => handleDifficulty('easy')}   onHover={playHoverSound} />
              <TierButton label="MEDIUM" sub="REACTS TO ATTACKS · PUNISHES MISTAKES"  tier="medium" onClick={() => handleDifficulty('medium')} onHover={playHoverSound} />
              <TierButton label="HARD"   sub="NEARLY IMPOSSIBLE · BLOCKS EVERYTHING"  tier="hard"   onClick={() => handleDifficulty('hard')}   onHover={playHoverSound} />
              <motion.button
                whileHover={{ x: -5 }}
                whileTap={{ scale: 0.97 }}
                onClick={handleBack}
                style={{
                  background: 'transparent', border: 'none', color: '#555',
                  fontSize: 12, fontFamily: "'Orbitron', sans-serif",
                  letterSpacing: 3, cursor: 'pointer', marginTop: 6,
                  padding: '8px 0', outline: 'none', textAlign: 'left',
                }}
              >
                ← BACK
              </motion.button>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Version */}
      <div style={{ position: 'absolute', bottom: 14, color: '#282828', fontSize: 11, fontFamily: 'monospace', letterSpacing: 2, zIndex: 2 }}>
        v1.0.0 · DLP/ANN PROJECT
      </div>
    </motion.div>
  );
};

/* ── FightButton: hexagonal-clipped fighting game button ── */
interface FightButtonProps {
  label: string; sublabel: string; icon: string;
  onClick: () => void; onHover?: () => void;
  primary?: boolean; small?: boolean;
}

const FightButton: React.FC<FightButtonProps> = ({ label, sublabel, icon, onClick, onHover, primary = false, small = false }) => (
  <motion.button
    whileHover={{ x: 8, boxShadow: primary ? '0 0 40px rgba(231,76,60,0.7), 0 0 80px rgba(231,76,60,0.2)' : '0 0 18px rgba(255,255,255,0.06)' }}
    whileTap={{ scale: 0.97 }}
    onHoverStart={onHover}
    onClick={onClick}
    style={{
      position: 'relative',
      background: primary
        ? 'linear-gradient(108deg, #7a0000 0%, #c0392b 50%, #e74c3c 100%)'
        : small
          ? 'linear-gradient(108deg, #111 0%, #1a1a1a 100%)'
          : 'linear-gradient(108deg, #0d0d0d 0%, #1c1c1c 100%)',
      border: primary ? '2px solid #e74c3c' : `2px solid ${small ? '#222' : '#2a2a2a'}`,
      clipPath: 'polygon(0 0, calc(100% - 18px) 0, 100% 18px, 100% 100%, 18px 100%, 0 calc(100% - 18px))',
      padding: small ? '11px 22px' : '18px 26px',
      color: '#fff',
      cursor: 'pointer',
      display: 'flex',
      alignItems: 'center',
      gap: 14,
      outline: 'none',
      textAlign: 'left',
      boxShadow: primary ? '0 8px 30px rgba(231,76,60,0.45)' : '0 4px 12px rgba(0,0,0,0.7)',
      transition: 'box-shadow 0.2s',
    }}
  >
    {/* Left accent bar */}
    <div style={{
      position: 'absolute', left: 0, top: 0, bottom: 0, width: 4,
      background: primary ? 'linear-gradient(180deg, #ff8888, #e74c3c)' : (small ? '#1e1e1e' : '#2a2a2a'),
      boxShadow: primary ? '0 0 10px #e74c3c' : 'none',
    }} />
    <span style={{ fontSize: small ? 16 : 20, marginLeft: 8 }}>{icon}</span>
    <span style={{ flex: 1 }}>
      <span style={{
        display: 'block',
        fontSize: small ? 15 : 22,
        fontWeight: 900,
        fontFamily: "'Bebas Neue', sans-serif",
        letterSpacing: 4,
        color: primary ? '#fff' : '#bbb',
        textShadow: primary ? '0 2px 8px rgba(0,0,0,0.9)' : 'none',
      }}>{label}</span>
      {sublabel && (
        <span style={{
          display: 'block', fontSize: 9,
          fontFamily: "'Orbitron', sans-serif",
          letterSpacing: 2,
          color: primary ? '#ffbbbb' : '#444',
          marginTop: 2,
        }}>{sublabel}</span>
      )}
    </span>
    <span style={{ fontSize: 11, color: primary ? '#ff9999' : '#383838', marginRight: 8 }}>▶</span>
  </motion.button>
);

/* ── TierButton: difficulty tier selector ── */
const TIER_COLORS = {
  easy:   { main: '#27ae60', dark: '#0a2e1a' },
  medium: { main: '#f39c12', dark: '#2e1e00' },
  hard:   { main: '#e74c3c', dark: '#2e0808' },
};

interface TierButtonProps {
  label: string; sub: string; tier: 'easy' | 'medium' | 'hard';
  onClick: () => void; onHover?: () => void;
}

const TierButton: React.FC<TierButtonProps> = ({ label, sub, tier, onClick, onHover }) => {
  const c = TIER_COLORS[tier];
  return (
    <motion.button
      whileHover={{ x: 10, boxShadow: `0 0 40px ${c.main}55, inset 0 0 24px ${c.main}18` }}
      whileTap={{ scale: 0.97 }}
      onHoverStart={onHover}
      onClick={onClick}
      style={{
        position: 'relative',
        background: `linear-gradient(108deg, ${c.dark} 0%, ${c.main}16 100%)`,
        border: `2px solid ${c.main}`,
        clipPath: 'polygon(0 0, calc(100% - 18px) 0, 100% 18px, 100% 100%, 18px 100%, 0 calc(100% - 18px))',
        padding: '17px 24px',
        color: '#fff',
        cursor: 'pointer',
        display: 'flex',
        alignItems: 'center',
        gap: 16,
        outline: 'none',
        textAlign: 'left',
        boxShadow: `0 6px 22px rgba(0,0,0,0.7)`,
        transition: 'box-shadow 0.2s',
      }}
    >
      <div style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: 5, background: c.main, boxShadow: `0 0 10px ${c.main}` }} />
      <div style={{ marginLeft: 10, flex: 1 }}>
        <div style={{ fontSize: 26, fontWeight: 900, fontFamily: "'Bebas Neue', sans-serif", letterSpacing: 5, color: c.main, textShadow: `0 0 14px ${c.main}88` }}>
          {label}
          {tier === 'hard' && (
            <span style={{ fontSize: 10, marginLeft: 10, fontFamily: "'Orbitron', sans-serif", color: '#ff6b6b', letterSpacing: 1 }}>
              ⚠ NEARLY IMPOSSIBLE
            </span>
          )}
        </div>
        <div style={{ fontSize: 9, fontFamily: "'Orbitron', sans-serif", color: '#666', letterSpacing: 1.5, marginTop: 3 }}>{sub}</div>
      </div>
      <span style={{ fontSize: 14, color: c.main, marginRight: 8, opacity: 0.7 }}>▶</span>
    </motion.button>
  );
};
