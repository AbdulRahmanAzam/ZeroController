# Fighter Arena - 2D Fighting Game 🎮

> A production-ready browser-based 2D fighting game with **camera-based pose detection** control system. Built for an ANN/Deep Learning academic project.

[![TypeScript](https://img.shields.io/badge/TypeScript-5.7-blue)](https://www.typescriptlang.org/)
[![React](https://img.shields.io/badge/React-19-61dafb)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-6-646cff)](https://vitejs.dev/)

## 🎯 Project Overview

This is a full-featured 2D fighting game (inspired by Tekken and Skullgirls) designed to be controlled through **real-time body movements** using camera-based pose detection with Artificial Neural Networks and Deep Learning. Players can use traditional keyboard controls or stand in front of a camera and physically perform punches, kicks, blocks, and jumps to control their character!

### Key Features
✅ **9 Action Movements** - Full fighting game move set  
✅ **Multi-Round System** - Best-of-3 rounds with score tracking  
✅ **Dual Input Modes** - Keyboard controls + Camera pose detection  
✅ **Sprite-Based Animation** - Smooth character animations  
✅ **Complete Sound System** - SFX and background music  
✅ **Production UI** - Main menu, settings, victory screens  
✅ **WebSocket Pose API** - Ready for DLP/ANN integration  

---

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ and npm
- Modern web browser (Chrome, Firefox, Edge)
- *Optional*: Webcam and pose detection backend for camera mode

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd 2D\ Game

# Install dependencies
npm install

# Start development server
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser.

### Build for Production

```bash
npm run build
npm run preview  # Preview production build
```

---

## 🎮 How to Play

### Game Modes
1. **Local Multiplayer** - Two players on one keyboard
2. **Camera Mode** - Control with body movements (requires pose detection backend)

### Controls

#### Keyboard - Player 1
| Action | Keys |
|--------|------|
| Move | A / D or ← / → |
| Jump | W or ↑ |
| Block | S or ↓ |
| Left Punch | J |
| Right Punch | K |
| Left Kick | U |
| Right Kick | I |

#### Keyboard - Player 2
| Action | Keys |
|--------|------|
| Move | Numpad 4 / 6 |
| Jump | Numpad 8 |
| Block | Numpad 5 |
| Punch | Numpad 1 / 2 |
| Kick | Numpad 7 / 9 |

#### Camera Mode (Pose Detection)
- **Punch**: Extend arm forward
- **Kick**: Raise and extend leg
- **Block**: Arms crossed in front
- **Jump**: Arms raised, knees bent
- **Move**: Lean left/right or step

See [POSE_API_SPEC.md](./POSE_API_SPEC.md) for full camera control documentation.

### Combat Mechanics
- **Punches**: Fast attacks (40ms animation, 8 damage)
- **Kicks**: Slower but stronger (125ms animation, 12 damage)  
- **Blocking**: Reduces damage by 75%
- **Combos**: Chain hits to build combo counter
- **Rounds**: Win 2 out of 3 rounds to win the match

---

## 📸 Camera Pose Detection Setup

This game is designed to work with a camera-based pose detection system for the DLP/ANN project.

### Required Backend

The game expects a WebSocket server at:
```
ws://localhost:8000/ws/pose/{player_id}
```

See [POSE_API_SPEC.md](./POSE_API_SPEC.md) for complete API documentation.

### Recommended Tech Stack
- **Pose Detection**: MediaPipe Pose, OpenPose, or PoseNet
- **Backend**: FastAPI (Python) or Node.js
- **ML Framework**: TensorFlow, PyTorch, or ONNX

### Quick Backend Setup (Python Example)

```bash
pip install fastapi uvicorn opencv-python mediapipe websockets
```

See API spec for example implementation.

---

## 🏗️ Architecture

### Tech Stack
- **Frontend**: React 19 + TypeScript 5.7
- **Build Tool**: Vite 6
- **Styling**: Inline styles + Framer Motion
- **State Management**: Zustand 5
- **Audio**: Howler.js 2
- **Sprites**: JSON atlas with Aseprite

### Project Structure
```
src/
├── components/       # React components
│   ├── Game.tsx           # Main game component
│   ├── EnhancedFighter.tsx    # Player sprite
│   ├── EnhancedArena.tsx      # Game arena
│   ├── MainMenu.tsx           # Main menu
│   ├── SettingsMenu.tsx       # Settings screen
│   └── VictoryScreen.tsx      # Victory display
├── store/
│   └── gameStore.ts   # Zustand game state
├── hooks/
│   ├── useGameLoop.ts       # 60 FPS game loop
│   ├── useKeyboardInput.ts  # Keyboard controls
│   └── usePoseInput.ts      # Pose detection WebSocket
├── game/
│   └── config.ts      # Game configuration
├── types/
│   └── game.ts        # TypeScript interfaces
└── audio/
    └── SoundManager.ts     # Sound system

public/
├── sprites/          # Character sprite sheets
└── sounds/           # Audio files
```

### Game State Flow
```
Menu → Waiting → Playing → Round End → (Next Round | Match End) → Menu
```

---

## 🎨 Customization

### Adjust Game Balance

Edit `src/game/config.ts`:

```typescript
export const GAME_CONFIG = {
  canvasWidth: 1200,
  canvasHeight: 600,
  gravity: 0.8,
  jumpForce: -15,
  moveSpeed: 5,
  playerWidth: 80,
  playerHeight: 120,
  punchDamage: 8,      // Normal punch damage
  kickDamage: 12,      // Normal kick damage
  blockDamageReduction: 0.75,  // 75% damage reduction
  hitCooldownTime: 300,  // Hit stun duration (ms)
  roundDuration: 99,   // Round time (seconds)
};
```

### Add New Sprites

1. Export sprite sheet as PNG + JSON atlas (Aseprite format)
2. Place in `public/sprites/`
3. Update `EnhancedFighter.tsx` with new sprite paths

### Modify Sounds

Replace audio files in `public/sounds/` or update `SoundManager.ts` to load new files.

---

## 🧪 Testing

```bash
# Run linter
npm run lint

# Type checking
npm run type-check

# Build test
npm run build
```

---

## 📚 Documentation

- [POSE_API_SPEC.md](./POSE_API_SPEC.md) - Complete pose detection API specification
- [QUICK_START_SOUNDS.md](./QUICK_START_SOUNDS.md) - Audio setup guide
- [SOUND_RESOURCES.md](./SOUND_RESOURCES.md) - Sound effect resources

---

## 🗺️ Roadmap

### ✅ Completed (Phase 1-2)
- [x] Core game mechanics
- [x] Multi-round system (best-of-3)
- [x] Main menu and settings
- [x] Keyboard controls for 2 players
- [x] Sound effects and music
- [x] Sprite animations
- [x] WebSocket pose detection hook

### 🚧 In Progress (Phase 3)
- [ ] Camera overlay UI
- [ ] Pose confidence indicator
- [ ] Player calibration system
- [ ] Gesture tutorial mode
- [ ] Input mode toggle (keyboard/camera)

### 📋 Planned (Phase 4-6)
- [ ] AI opponent (CPU)
- [ ] Special moves system
- [ ] Character selection
- [ ] Training mode
- [ ] Mobile touch controls
- [ ] Unit and integration tests

---

## 🤝 Contributing

This is an academic project, but contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

## 📝 License

MIT License - feel free to use this project for learning and development.

---

## 🎓 Academic Project Info

**Course**: Deep Learning & ANN  
**Concept**: Motion-controlled 2D fighter with real-time pose detection  
**Features**: 7-9 action movements controlled by camera + body movements  
**Goal**: Make gaming more interactive and physically engaging  

**Technical Highlights**:
- Real-time pose estimation (30-60 FPS)
- Low-latency action recognition (<100ms)
- WebSocket-based client-server architecture
- Confidence-based action filtering

---

## 🐛 Troubleshooting

### Game won't start
- Check browser console for errors
- Verify all assets loaded (sprites, sounds)
- Try clearing browser cache

### Camera mode not working
- Ensure pose detection server is running
- Check WebSocket URL in `usePoseInput.ts`
- Verify camera permissions in browser
- Test with POSE_API_SPEC test messages

### Performance issues
- Close other browser tabs
- Disable browser extensions
- Check FPS in game (target: 60 FPS)
- Reduce sprite scale if needed

---

## 📧 Contact

For questions about this project, please open an issue on GitHub.

---

**Built with** ❤️ **using React, TypeScript, and Deep Learning**
| Left Kick | N |
| Right Kick | M |

## Getting Started

### Prerequisites
- Node.js 18+
- npm or yarn

### Installation

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build
```

### Running the Game
1. Open http://localhost:5173 in your browser
2. Click "START GAME" to begin
3. Use keyboard controls to play (or connect camera pose detection)

## Architecture

```
src/
├── components/       # React UI components
│   ├── Game.tsx      # Main game component
│   ├── Fighter.tsx   # Player character rendering
│   ├── Arena.tsx     # Background and stage
│   ├── HealthBar.tsx # Health bar UI
│   └── GameUI.tsx    # Menus and overlays
├── game/             # Game logic
│   ├── config.ts     # Game configuration
│   └── gameEngine.ts # Physics and state management
├── hooks/            # React hooks
│   ├── useGameLoop.ts      # 60 FPS game loop
│   ├── useKeyboardInput.ts # Keyboard controls
│   └── usePoseInput.ts     # Camera/API input
└── types/            # TypeScript definitions
    └── game.ts       # Game types
```

## FastAPI Integration

The game is designed to receive input from a Python FastAPI backend that processes camera pose detection. The `usePoseInput` hook connects via WebSocket to receive real-time action commands.

### Expected API Endpoints

**WebSocket**: `ws://localhost:8000/ws/pose/{player_id}`

**Response format**:
```json
{
  "action": "left_punch",
  "confidence": 0.95,
  "timestamp": 1234567890
}
```

**REST fallback**: `GET /pose/{player_id}`

## Tech Stack

- **React 19** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool
- **Custom game engine** - Physics and state management

## License

MIT
