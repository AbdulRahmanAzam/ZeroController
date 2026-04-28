import { useState, useEffect } from 'react';
import { AnimatePresence } from 'framer-motion';
import { Game } from './components/Game';
import { MainMenu } from './components/MainMenu';
import { SettingsMenu } from './components/SettingsMenu';
import { HowToPlay } from './components/HowToPlay';
import { preloadSprites } from './components/spriteAssets';
import './App.css';

type AppScreen = 'menu' | 'game' | 'settings' | 'howToPlay';

function App() {
  const [currentScreen, setCurrentScreen] = useState<AppScreen>('menu');
  const [assetsReady, setAssetsReady] = useState(false);

  // Preload all sprite assets before showing any screen.
  // This fills the module-level caches in SpriteSheet.tsx so fighters
  // render immediately on the very first mount — no reload needed.
  useEffect(() => {
    preloadSprites().finally(() => setAssetsReady(true));
  }, []);

  if (!assetsReady) {
    return (
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        height: '100vh', background: '#0a0a1a',
        color: '#00d4ff', fontFamily: 'sans-serif', fontSize: '1.4rem',
        letterSpacing: '0.1em',
      }}>
        Loading assets…
      </div>
    );
  }

  return (
    <AnimatePresence mode="wait">
      {currentScreen === 'menu' && (
        <MainMenu
          key="menu"
          onStartGame={() => setCurrentScreen('game')}
          onOpenSettings={() => setCurrentScreen('settings')}
          onOpenHowToPlay={() => setCurrentScreen('howToPlay')}
        />
      )}
      {currentScreen === 'game' && <Game key="game" onGoToMenu={() => setCurrentScreen('menu')} />}
      {currentScreen === 'settings' && (
        <SettingsMenu key="settings" onClose={() => setCurrentScreen('menu')} />
      )}
      {currentScreen === 'howToPlay' && (
        <HowToPlay key="howToPlay" onClose={() => setCurrentScreen('menu')} />
      )}
    </AnimatePresence>
  );
}

export default App;
