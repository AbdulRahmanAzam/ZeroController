import React, { useMemo } from 'react';
import { motion } from 'framer-motion';
import type { PlayerState, ActionType } from '../types/game';
import { GAME_CONFIG, ANIMATIONS } from '../game/config';
import { SpriteSheet } from './SpriteSheet';

interface EnhancedFighterProps {
  player: PlayerState;
  useSprites?: boolean;
}

// Player color schemes - more dramatic
const FIGHTER_COLORS = {
  1: {
    primary: '#00d4ff',
    secondary: '#0099cc',
    accent: '#00ffff',
    dark: '#006688',
    skin: '#ffd5c8',
    hair: '#1a1a2e',
    energy: '#00ffff',
  },
  2: {
    primary: '#ff4757',
    secondary: '#cc0022',
    accent: '#ff6b81',
    dark: '#880011',
    skin: '#e8c4a0',
    hair: '#8b4513',
    energy: '#ff4757',
  },
};

export const EnhancedFighter: React.FC<EnhancedFighterProps> = ({ player, useSprites = true }) => {
  const colors = FIGHTER_COLORS[player.id];
  const width = GAME_CONFIG.playerWidth;
  const height = GAME_CONFIG.playerHeight;
  const facingScale = player.facingRight ? 1 : -1;
  
  const isHurt = player.hitCooldown > 0;
  const isAttacking = player.isAttacking;
  const isPunching = player.action === 'left_punch' || player.action === 'right_punch';
  const isKicking = player.action === 'left_kick' || player.action === 'right_kick';

  // Map game actions to sprite animations - use all 3 attack animations for variety
  const spriteAnimation = useMemo(() => {
    if (isHurt) return 'get_hit';
    if (player.isBlocking) return 'guard';
    
    // Separate mapping for each attack type to distinguish left/right
    if (player.action === 'left_punch') return 'attack_A';
    if (player.action === 'right_punch') return 'attack_B';
    if (player.action === 'left_kick') return 'attack_C';
    if (player.action === 'right_kick') return 'attack_C';
    
    if (!player.isGrounded) {
      // Jump sequence: jump_start -> jump_loop -> fall_loop
      return player.velocityY < 0 ? 'jump_start' : 'fall_loop';
    }
    if (player.action === 'move_forward' || player.action === 'move_backward' || Math.abs(player.velocityX) > 0.2) {
      return 'run';
    }
    return 'idle';
  }, [isHurt, player.isBlocking, player.action, player.isGrounded, player.velocityX, player.velocityY]);

  // Calculate frame rate based on action's frameDuration
  const frameRate = useMemo(() => {
    const action = player.action as ActionType;
    const animConfig = ANIMATIONS[action];
    if (animConfig && animConfig.frameDuration > 0) {
      return Math.round(1000 / animConfig.frameDuration);
    }
    return 12; // Default fallback
  }, [player.action]);

  // Use a different slice of attack_C for right_kick to keep it distinct without sprite flipping
  const frameOffset = useMemo(() => {
    if (player.action === 'right_kick') {
      return 5;
    }
    return 0;
  }, [player.action]);

  // Get animation based on action
  const getBodyRotation = () => {
    if (isPunching) return player.facingRight ? 15 : -15;
    if (isKicking) return player.facingRight ? 20 : -20;
    return 0;
  };

  // Sprite paths - both players use knight sprite, P2 gets hue-shifted via style filter
  const spritePaths = {
    1: {
      atlas: '/sprites/player1/knight.json',
      image: '/sprites/player1/knight.png',
    },
    2: {
      atlas: '/sprites/player1/knight.json',
      image: '/sprites/player1/knight.png',
    },
  };

  return (
    <motion.div
      style={{
        position: 'absolute',
        left: player.x,
        top: player.y,
        width: width,
        height: height,
        transformOrigin: 'center bottom',
        filter: isHurt ? 'brightness(2) saturate(0.3)' : 'none',
      }}
      animate={{
        y: player.action === 'idle' ? [0, -2, 0] : 0,
      }}
      transition={{
        y: { duration: 1.2, repeat: Infinity, ease: 'easeInOut' },
      }}
    >
      {/* Energy aura when attacking */}
      {isAttacking && (
        <motion.div
          initial={{ scale: 0.5, opacity: 0.8 }}
          animate={{ scale: 2, opacity: 0 }}
          transition={{ duration: 0.4 }}
          style={{
            position: 'absolute',
            top: '10%',
            left: '-20%',
            right: '-20%',
            bottom: '10%',
            borderRadius: '50%',
            background: `radial-gradient(circle, ${colors.energy}60 0%, ${colors.energy}20 40%, transparent 70%)`,
            pointerEvents: 'none',
            zIndex: -1,
          }}
        />
      )}

      {/* Sprite-based Fighter or SVG fallback */}
      {useSprites ? (
        <div
          style={{
            position: 'absolute',
            bottom: 0,
            left: '50%',
            transform: `translateX(-50%) scaleX(${facingScale})`,
            transformOrigin: 'center bottom',
          }}
        >
          <SpriteSheet
            key={`${player.id}`}
            atlasPath={spritePaths[player.id].atlas}
            imagePath={spritePaths[player.id].image}
            animation={spriteAnimation}
            frameRate={frameRate}
            loop={!isAttacking && !isHurt}
            scale={3}
            flipX={false}
            frameOffset={frameOffset}
            style={{
              filter: `drop-shadow(0 4px 8px ${colors.primary}40) hue-rotate(${player.id === 2 ? '180deg' : '0deg'})`,
            }}
          />
          
          {/* Ground shadow for sprite */}
          <div
            style={{
              position: 'absolute',
              bottom: -10,
              left: '50%',
              transform: 'translateX(-50%)',
              width: player.isGrounded ? 60 : 40,
              height: 10,
              background: 'radial-gradient(ellipse, rgba(0,0,0,0.5) 0%, transparent 70%)',
              borderRadius: '50%',
              transition: 'width 0.2s',
            }}
          />
        </div>
      ) : (
        <div
          style={{
            transform: `scaleX(${facingScale})`,
            transformOrigin: 'center bottom',
          }}
        >
          {/* Original Fighter SVG */}
      <motion.svg 
        width={width} 
        height={height} 
        viewBox="0 0 100 150"
        animate={{ rotate: getBodyRotation() }}
        transition={{ duration: 0.1 }}
        style={{ transformOrigin: 'center bottom' }}
      >
        <defs>
          {/* Gradients */}
          <linearGradient id={`bodyGrad${player.id}`} x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor={colors.primary} />
            <stop offset="50%" stopColor={colors.secondary} />
            <stop offset="100%" stopColor={colors.dark} />
          </linearGradient>
          <linearGradient id={`skinGrad${player.id}`} x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor={colors.skin} />
            <stop offset="100%" stopColor="#d4a574" />
          </linearGradient>
          <linearGradient id={`energyGrad${player.id}`} x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor={colors.energy} stopOpacity="0" />
            <stop offset="50%" stopColor={colors.energy} stopOpacity="1" />
            <stop offset="100%" stopColor={colors.energy} stopOpacity="0" />
          </linearGradient>
          
          {/* Glow filter */}
          <filter id={`glow${player.id}`} x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="3" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          
          {/* Shadow */}
          <filter id="fighterShadow">
            <feDropShadow dx="2" dy="3" stdDeviation="2" floodColor="#000" floodOpacity="0.5" />
          </filter>
        </defs>

        {/* Ground shadow */}
        <ellipse
          cx="50"
          cy="148"
          rx={player.isGrounded ? 30 : 20}
          ry="5"
          fill="rgba(0,0,0,0.5)"
        />

        {/* LEGS */}
        <g filter="url(#fighterShadow)">
          {/* Left leg */}
          <motion.rect
            x="30"
            y="95"
            width="14"
            height="48"
            rx="6"
            fill={colors.secondary}
            animate={{
              x: player.action === 'left_kick' ? 5 : 30,
              rotate: player.action === 'left_kick' ? -70 : 0,
            }}
            style={{ transformOrigin: '37px 95px' }}
          />
          {/* Right leg */}
          <motion.rect
            x="56"
            y="95"
            width="14"
            height="48"
            rx="6"
            fill={colors.secondary}
            animate={{
              x: player.action === 'right_kick' ? 80 : 56,
              rotate: player.action === 'right_kick' ? 70 : 0,
            }}
            style={{ transformOrigin: '63px 95px' }}
          />
          {/* Boots */}
          <ellipse cx="37" cy="142" rx="10" ry="6" fill={colors.dark} />
          <ellipse cx="63" cy="142" rx="10" ry="6" fill={colors.dark} />
        </g>

        {/* BODY - Athletic build */}
        <g filter="url(#fighterShadow)">
          <path
            d="M 25 45 
               C 20 55, 20 85, 28 95 
               L 72 95 
               C 80 85, 80 55, 75 45 
               Q 75 40, 50 38 
               Q 25 40, 25 45"
            fill={`url(#bodyGrad${player.id})`}
          />
          {/* Chest highlight */}
          <ellipse cx="50" cy="60" rx="15" ry="12" fill="rgba(255,255,255,0.1)" />
          {/* Belt */}
          <rect x="26" y="88" width="48" height="8" fill={colors.accent} rx="2" />
          {/* Player number on chest */}
          <text 
            x="50" 
            y="72" 
            textAnchor="middle" 
            fill="#fff" 
            fontSize="16" 
            fontWeight="bold"
            fontFamily="Orbitron, sans-serif"
            filter={`url(#glow${player.id})`}
          >
            {player.id}
          </text>
        </g>

        {/* ARMS */}
        <g filter="url(#fighterShadow)">
          {/* Left arm */}
          <motion.g
            animate={{
              rotate: player.action === 'left_punch' ? -45 : 0,
              x: player.action === 'left_punch' ? -25 : 0,
            }}
            style={{ transformOrigin: '25px 55px' }}
          >
            <rect
              x="5"
              y="48"
              width={player.action === 'left_punch' ? 45 : 24}
              height="12"
              rx="6"
              fill={`url(#skinGrad${player.id})`}
            />
            {/* Fist glow on punch */}
            {player.action === 'left_punch' && (
              <circle cx="-5" cy="54" r="12" fill={colors.energy} filter={`url(#glow${player.id})`} opacity="0.8" />
            )}
          </motion.g>
          
          {/* Right arm */}
          <motion.g
            animate={{
              rotate: player.action === 'right_punch' ? 45 : 0,
              x: player.action === 'right_punch' ? 25 : 0,
            }}
            style={{ transformOrigin: '75px 55px' }}
          >
            <rect
              x={player.action === 'right_punch' ? 51 : 71}
              y="48"
              width={player.action === 'right_punch' ? 45 : 24}
              height="12"
              rx="6"
              fill={`url(#skinGrad${player.id})`}
            />
            {/* Fist glow on punch */}
            {player.action === 'right_punch' && (
              <circle cx="105" cy="54" r="12" fill={colors.energy} filter={`url(#glow${player.id})`} opacity="0.8" />
            )}
          </motion.g>
        </g>

        {/* HEAD */}
        <g filter="url(#fighterShadow)">
          {/* Face */}
          <circle cx="50" cy="24" r="20" fill={`url(#skinGrad${player.id})`} />
          
          {/* Hair - Anime style spiky */}
          <path
            d={`M 30 24 
                Q 30 5, 45 8 
                Q 50 2, 55 8 
                Q 70 5, 70 24
                Q 65 15, 50 18
                Q 35 15, 30 24`}
            fill={colors.hair}
          />
          {/* Hair spikes */}
          <path d="M 35 10 L 32 0 L 40 8" fill={colors.hair} />
          <path d="M 50 8 L 50 -2 L 55 6" fill={colors.hair} />
          <path d="M 65 10 L 68 0 L 60 8" fill={colors.hair} />
          
          {/* Eyes - Anime style */}
          <g>
            {/* Eye whites */}
            <ellipse cx="42" cy="24" rx="6" ry="7" fill="#fff" />
            <ellipse cx="58" cy="24" rx="6" ry="7" fill="#fff" />
            {/* Irises */}
            <circle cx="43" cy="25" r="4" fill={player.id === 1 ? '#3498db' : '#e74c3c'} />
            <circle cx="59" cy="25" r="4" fill={player.id === 1 ? '#3498db' : '#e74c3c'} />
            {/* Pupils */}
            <circle cx="44" cy="25" r="2" fill="#000" />
            <circle cx="60" cy="25" r="2" fill="#000" />
            {/* Eye shine */}
            <circle cx="41" cy="23" r="1.5" fill="#fff" />
            <circle cx="57" cy="23" r="1.5" fill="#fff" />
          </g>
          
          {/* Eyebrows - Expression based */}
          <motion.path
            d={isAttacking ? "M 36 16 L 48 19" : isHurt ? "M 36 19 L 48 16" : "M 36 17 L 48 17"}
            stroke={colors.hair}
            strokeWidth="3"
            strokeLinecap="round"
            fill="none"
          />
          <motion.path
            d={isAttacking ? "M 64 16 L 52 19" : isHurt ? "M 64 19 L 52 16" : "M 64 17 L 52 17"}
            stroke={colors.hair}
            strokeWidth="3"
            strokeLinecap="round"
            fill="none"
          />
          
          {/* Mouth */}
          <path
            d={isAttacking ? "M 44 33 Q 50 40 56 33" : isHurt ? "M 44 35 Q 50 30 56 35" : "M 46 33 Q 50 35 54 33"}
            stroke="#333"
            strokeWidth="2"
            fill="none"
            strokeLinecap="round"
          />
        </g>

        {/* BLOCKING SHIELD */}
        {player.isBlocking && (
          <g>
            <motion.ellipse
              cx="50"
              cy="65"
              rx="55"
              ry="70"
              fill="none"
              stroke={colors.energy}
              strokeWidth="3"
              strokeDasharray="20,10"
              filter={`url(#glow${player.id})`}
              animate={{ strokeDashoffset: [0, 60] }}
              transition={{ duration: 0.5, repeat: Infinity, ease: 'linear' }}
            />
            <ellipse
              cx="50"
              cy="65"
              rx="50"
              ry="65"
              fill={`${colors.energy}15`}
            />
            <text
              x="50"
              y="10"
              textAnchor="middle"
              fill={colors.energy}
              fontSize="10"
              fontWeight="bold"
              fontFamily="Orbitron"
            >
              GUARD
            </text>
          </g>
        )}

        {/* KICK TRAIL EFFECT */}
        {isKicking && (
          <motion.path
            d={player.action === 'left_kick' 
              ? "M 30 95 Q 0 70, -20 95" 
              : "M 70 95 Q 100 70, 120 95"}
            stroke={`url(#energyGrad${player.id})`}
            strokeWidth="8"
            fill="none"
            initial={{ pathLength: 0, opacity: 0 }}
            animate={{ pathLength: 1, opacity: [0, 1, 0] }}
            transition={{ duration: 0.2 }}
          />
        )}

        {/* PUNCH TRAIL EFFECT */}
        {isPunching && (
          <motion.path
            d={player.action === 'left_punch'
              ? "M 25 54 L -20 50"
              : "M 75 54 L 120 50"}
            stroke={`url(#energyGrad${player.id})`}
            strokeWidth="6"
            fill="none"
            strokeLinecap="round"
            initial={{ pathLength: 0, opacity: 0 }}
            animate={{ pathLength: 1, opacity: [0, 1, 0] }}
            transition={{ duration: 0.15 }}
          />
        )}
        </motion.svg>
        </div>
      )}

      {/* Player name badge - Text never mirrored */}
      <motion.div
        initial={{ y: -20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        style={{
          position: 'absolute',
          top: -35,
          left: '50%',
          transform: 'translateX(-50%)', // No scaleX - text always readable
          padding: '4px 16px',
          background: `linear-gradient(180deg, ${colors.primary} 0%, ${colors.dark} 100%)`,
          borderRadius: 20,
          color: '#fff',
          fontWeight: 'bold',
          fontSize: 11,
          fontFamily: 'Orbitron, sans-serif',
          boxShadow: `0 0 15px ${colors.energy}80`,
          border: `2px solid ${colors.accent}`,
          letterSpacing: 2,
          whiteSpace: 'nowrap',
        }}
      >
        PLAYER {player.id}
      </motion.div>

      {/* Action indicator - Text never mirrored */}
      <motion.div
        key={player.action}
        initial={{ opacity: 0, scale: 0.5 }}
        animate={{ opacity: 1, scale: 1 }}
        style={{
          position: 'absolute',
          bottom: -28,
          left: '50%',
          transform: 'translateX(-50%)', // No scaleX - text always readable
          color: colors.energy,
          fontSize: 10,
          fontWeight: 'bold',
          fontFamily: 'Orbitron, sans-serif',
          textTransform: 'uppercase',
          textShadow: `0 0 10px ${colors.energy}`,
          letterSpacing: 1,
          whiteSpace: 'nowrap',
        }}
      >
        {player.action.replace('_', ' ')}
      </motion.div>
    </motion.div>
  );
};

export default EnhancedFighter;
