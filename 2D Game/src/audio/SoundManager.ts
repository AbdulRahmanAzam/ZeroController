import { Howl, Howler } from 'howler';

// Sound categories and their configurations
export type SoundCategory = 
  | 'punch'
  | 'kick'
  | 'block'
  | 'hit'
  | 'jump'
  | 'whoosh'
  | 'fight'
  | 'victory'
  | 'ko'
  | 'countdown'
  | 'menuSelect'
  | 'menuHover'
  | 'bgMusic';

interface SoundConfig {
  src: string[];
  volume?: number;
  loop?: boolean;
  rate?: number;
}

interface SoundVariant {
  sound: Howl;
  config: SoundConfig;
}

/**
 * SoundManager - Singleton class for managing all game audio
 * Uses Howler.js for cross-browser audio playback
 */
class SoundManagerClass {
  private static instance: SoundManagerClass;
  private sounds: Map<SoundCategory, SoundVariant[]> = new Map();
  private bgMusic: Howl | null = null;
  private isMuted: boolean = false;
  private masterVolume: number = 1.0;
  private sfxVolume: number = 0.7;
  private musicVolume: number = 0.4;
  private isInitialized: boolean = false;

  private constructor() {
    // Private constructor for singleton
  }

  public static getInstance(): SoundManagerClass {
    if (!SoundManagerClass.instance) {
      SoundManagerClass.instance = new SoundManagerClass();
    }
    return SoundManagerClass.instance;
  }

  /**
   * Initialize all game sounds - call this once at game startup
   */
  public init(): void {
    if (this.isInitialized) return;

    // Configure Howler global settings
    Howler.volume(this.masterVolume);

    // Register all sound effects
    this.registerSounds();
    this.isInitialized = true;
  }

  private registerSounds(): void {
    // Punch sounds (variations for variety)
    this.registerSound('punch', [
      { src: ['/sounds/punch1.mp3', '/sounds/punch1.ogg'], volume: 0.6 },
      { src: ['/sounds/punch2.mp3', '/sounds/punch2.ogg'], volume: 0.6 },
      { src: ['/sounds/punch3.mp3', '/sounds/punch3.ogg'], volume: 0.6 },
    ]);

    // Kick sounds (heavier impact)
    this.registerSound('kick', [
      { src: ['/sounds/kick1.mp3', '/sounds/kick1.ogg'], volume: 0.7 },
      { src: ['/sounds/kick2.mp3', '/sounds/kick2.ogg'], volume: 0.7 },
      { src: ['/sounds/kick3.mp3', '/sounds/kick3.ogg'], volume: 0.7 },
    ]);

    // Block/guard sounds
    this.registerSound('block', [
      { src: ['/sounds/block1.mp3', '/sounds/block1.ogg'], volume: 0.5 },
      { src: ['/sounds/block2.mp3', '/sounds/block2.ogg'], volume: 0.5 },
    ]);

    // Hit impact sounds (when taking damage)
    this.registerSound('hit', [
      { src: ['/sounds/hit1.mp3', '/sounds/hit1.ogg'], volume: 0.7 },
      { src: ['/sounds/hit2.mp3', '/sounds/hit2.ogg'], volume: 0.7 },
      { src: ['/sounds/hit3.mp3', '/sounds/hit3.ogg'], volume: 0.7 },
    ]);

    // Jump sound
    this.registerSound('jump', [
      { src: ['/sounds/jump.mp3', '/sounds/jump.ogg'], volume: 0.4 },
    ]);

    // Whoosh/swing sounds for attacks
    this.registerSound('whoosh', [
      { src: ['/sounds/whoosh1.mp3', '/sounds/whoosh1.ogg'], volume: 0.3 },
      { src: ['/sounds/whoosh2.mp3', '/sounds/whoosh2.ogg'], volume: 0.3 },
    ]);

    // Announcer sounds
    this.registerSound('fight', [
      { src: ['/sounds/fight.mp3', '/sounds/fight.ogg'], volume: 0.9 },
    ]);

    this.registerSound('victory', [
      { src: ['/sounds/victory.mp3', '/sounds/victory.ogg'], volume: 0.8 },
    ]);

    this.registerSound('ko', [
      { src: ['/sounds/ko.mp3', '/sounds/ko.ogg'], volume: 0.9 },
    ]);

    this.registerSound('countdown', [
      { src: ['/sounds/countdown.mp3', '/sounds/countdown.ogg'], volume: 0.6 },
    ]);

    // Menu sounds
    this.registerSound('menuSelect', [
      { src: ['/sounds/menu_select.mp3', '/sounds/menu_select.ogg'], volume: 0.5 },
    ]);

    this.registerSound('menuHover', [
      { src: ['/sounds/menu_hover.mp3', '/sounds/menu_hover.ogg'], volume: 0.3 },
    ]);

    // Background music
    this.registerSound('bgMusic', [
      { src: ['/sounds/bg_music.mp3', '/sounds/bg_music.ogg'], volume: 0.4, loop: true },
    ]);
  }

  private registerSound(category: SoundCategory, configs: SoundConfig[]): void {
    const variants: SoundVariant[] = configs.map(config => ({
      sound: new Howl({
        src: config.src,
        volume: (config.volume ?? 0.5) * this.sfxVolume,
        loop: config.loop ?? false,
        rate: config.rate ?? 1.0,
        preload: false,
        onloaderror: () => undefined,
      }),
      config,
    }));
    this.sounds.set(category, variants);
  }

  /**
   * Play a sound effect with random variation
   */
  public play(category: SoundCategory, options?: { volume?: number; rate?: number }): number | null {
    if (this.isMuted) return null;
    if (!this.isInitialized) this.init();

    const variants = this.sounds.get(category);
    if (!variants || variants.length === 0) {
      return null;
    }

    // Pick a random variant for variety
    const variant = variants[Math.floor(Math.random() * variants.length)];
    const sound = variant.sound;

    // Apply optional overrides
    if (options?.volume !== undefined) {
      sound.volume(options.volume * this.sfxVolume);
    }
    if (options?.rate !== undefined) {
      sound.rate(options.rate);
    }

    try {
      return sound.play();
    } catch {
      return null;
    }
  }

  /**
   * Play attack sound (punch or kick with whoosh)
   */
  public playAttack(type: 'punch' | 'kick'): void {
    // Play whoosh first (attack swing)
    this.play('whoosh', { volume: 0.3 });
    
    // Slight delay for impact sound
    setTimeout(() => {
      this.play(type);
    }, 50);
  }

  /**
   * Play hit sound based on whether blocked or not
   */
  public playHit(blocked: boolean): void {
    if (blocked) {
      this.play('block');
    } else {
      this.play('hit');
    }
  }

  /**
   * Start background music
   */
  public startMusic(): void {
    if (this.isMuted) return;

    const musicVariants = this.sounds.get('bgMusic');
    if (musicVariants && musicVariants.length > 0) {
      this.bgMusic = musicVariants[0].sound;
      this.bgMusic.volume(0);
      this.bgMusic.play();
      this.bgMusic.fade(0, this.musicVolume, 1000);
    }
  }

  /**
   * Stop background music with fade out
   */
  public stopMusic(): void {
    if (this.bgMusic) {
      this.bgMusic.fade(this.musicVolume, 0, 500);
      setTimeout(() => {
        this.bgMusic?.stop();
      }, 500);
    }
  }

  /**
   * Pause background music
   */
  public pauseMusic(): void {
    if (this.bgMusic) {
      this.bgMusic.pause();
    }
  }

  /**
   * Resume background music
   */
  public resumeMusic(): void {
    if (this.bgMusic && !this.isMuted) {
      this.bgMusic.play();
    }
  }

  /**
   * Play round start sequence
   */
  public playRoundStart(): void {
    this.play('countdown');
    setTimeout(() => {
      this.play('fight');
    }, 800);
  }

  /**
   * Play victory fanfare
   */
  public playVictory(): void {
    this.play('ko');
    setTimeout(() => {
      this.play('victory');
    }, 500);
  }

  /**
   * Toggle mute state
   */
  public toggleMute(): boolean {
    this.isMuted = !this.isMuted;
    
    if (this.isMuted) {
      Howler.mute(true);
    } else {
      Howler.mute(false);
    }
    
    return this.isMuted;
  }

  /**
   * Set mute state
   */
  public setMuted(muted: boolean): void {
    this.isMuted = muted;
    Howler.mute(muted);
  }

  /**
   * Get mute state
   */
  public getMuted(): boolean {
    return this.isMuted;
  }

  /**
   * Set master volume (0.0 - 1.0)
   */
  public setMasterVolume(volume: number): void {
    this.masterVolume = Math.max(0, Math.min(1, volume));
    Howler.volume(this.masterVolume);
  }

  /**
   * Set SFX volume (0.0 - 1.0)
   */
  public setSfxVolume(volume: number): void {
    this.sfxVolume = Math.max(0, Math.min(1, volume));
    // Update all registered sound volumes
    this.sounds.forEach((variants, category) => {
      if (category !== 'bgMusic') {
        variants.forEach(variant => {
          variant.sound.volume((variant.config.volume ?? 0.5) * this.sfxVolume);
        });
      }
    });
  }

  /**
   * Set music volume (0.0 - 1.0)
   */
  public setMusicVolume(volume: number): void {
    this.musicVolume = Math.max(0, Math.min(1, volume));
    if (this.bgMusic) {
      this.bgMusic.volume(this.musicVolume);
    }
  }

  /**
   * Get current volume levels
   */
  public getVolumes(): { master: number; sfx: number; music: number } {
    return {
      master: this.masterVolume,
      sfx: this.sfxVolume,
      music: this.musicVolume,
    };
  }

  /**
   * Stop all sounds
   */
  public stopAll(): void {
    Howler.stop();
  }

  /**
   * Cleanup - unload all sounds
   */
  public destroy(): void {
    this.stopAll();
    this.sounds.forEach(variants => {
      variants.forEach(variant => variant.sound.unload());
    });
    this.sounds.clear();
    this.bgMusic = null;
    this.isInitialized = false;
  }
}

// Export singleton instance
export const SoundManager = SoundManagerClass.getInstance();
export default SoundManager;
