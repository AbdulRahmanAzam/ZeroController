import type { GameConfig, AnimationFrame, ActionType } from '../types/game';

// Dynamic canvas dimensions based on viewport
const getCanvasDimensions = () => {
  const viewportWidth = window.innerWidth;
  const viewportHeight = window.innerHeight;
  
  // Use full viewport dimensions
  return {
    width: viewportWidth,
    height: viewportHeight
  };
};

// Default dimensions for SSR/initial render
const DEFAULT_WIDTH = 1200;
const DEFAULT_HEIGHT = 600;

export const GAME_CONFIG: GameConfig = {
  get canvasWidth() {
    if (typeof window !== 'undefined') {
      return getCanvasDimensions().width;
    }
    return DEFAULT_WIDTH;
  },
  get canvasHeight() {
    if (typeof window !== 'undefined') {
      return getCanvasDimensions().height;
    }
    return DEFAULT_HEIGHT;
  },
  get groundY() {
    // Ground should be 88% down from the top of the screen
    return this.canvasHeight * 0.88;
  },
  get gravity() {
    // Scale gravity with screen height for consistent jump feel
    const scale = this.canvasHeight / DEFAULT_HEIGHT;
    return 0.8 * scale;
  },
  get moveSpeed() {
    // Scale movement speed with screen width
    const scale = this.canvasWidth / DEFAULT_WIDTH;
    return Math.max(4, 6 * scale);
  },
  get jumpForce() {
    // Scale jump force with screen height
    const scale = this.canvasHeight / DEFAULT_HEIGHT;
    return -18 * scale;
  },
  get playerWidth() {
    // Scale player size relative to screen width
    return Math.max(60, this.canvasWidth * 0.08);
  },
  get playerHeight() {
    // Scale player height relative to screen height
    return Math.max(80, this.canvasHeight * 0.18);
  },
  attackDamage: {
    punch: 8,
    kick: 12,
  },
  get attackRange() {
    // Scale attack range with player size
    return this.playerWidth * 0.8;
  },
  attackCooldownTime: 50,
  hitCooldownTime: 200,
  roundDuration: 99,
};

export const ANIMATIONS: Record<ActionType, AnimationFrame> = {
  idle: { name: 'idle', frameCount: 4, frameDuration: 150, loop: true },
  move_forward: { name: 'move_forward', frameCount: 6, frameDuration: 100, loop: true },
  move_backward: { name: 'move_backward', frameCount: 6, frameDuration: 100, loop: true },
  jump: { name: 'jump', frameCount: 4, frameDuration: 100, loop: false },
  block: { name: 'block', frameCount: 2, frameDuration: 100, loop: false },
  left_punch: { name: 'left_punch', frameCount: 4, frameDuration: 10, loop: false },
  right_punch: { name: 'right_punch', frameCount: 4, frameDuration: 10, loop: false },
  left_kick: { name: 'left_kick', frameCount: 5, frameDuration: 25, loop: false },
  right_kick: { name: 'right_kick', frameCount: 5, frameDuration: 25, loop: false },
};

// Color schemes for players (used for placeholder sprites)
export const PLAYER_COLORS = {
  1: {
    primary: '#3498db',
    secondary: '#2980b9',
    accent: '#e74c3c',
    skin: '#f5d5c8',
  },
  2: {
    primary: '#e74c3c',
    secondary: '#c0392b',
    accent: '#3498db',
    skin: '#deb887',
  },
};
