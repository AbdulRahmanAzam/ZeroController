import { useState, useEffect, useMemo } from 'react';
import { atlasCache, imageReadyCache, type Frame, type SpriteAtlas } from './spriteAssets';

interface SpriteSheetProps {
  atlasPath: string;
  imagePath: string;
  animation: string;
  frameRate?: number;
  loop?: boolean;
  onComplete?: () => void;
  scale?: number;
  flipX?: boolean;
  frameOffset?: number;
  style?: React.CSSProperties;
}

export const SpriteSheet: React.FC<SpriteSheetProps> = ({
  atlasPath,
  imagePath,
  animation,
  frameRate = 12,
  loop = true,
  onComplete,
  scale = 1,
  flipX = false,
  frameOffset = 0,
  style = {},
}) => {
  // Lazy-initialize from module-level cache so remounts are instant
  const [atlas, setAtlas] = useState<SpriteAtlas | null>(() => atlasCache.get(atlasPath) ?? null);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [imageLoaded, setImageLoaded] = useState(() => imageReadyCache.has(imagePath));

  // Load atlas JSON — skips the fetch if already in cache
  useEffect(() => {
    let cancelled = false;
    const cached = atlasCache.get(atlasPath);
    if (cached) {
      queueMicrotask(() => {
        if (!cancelled) setAtlas(cached);
      });
      return () => { cancelled = true; };
    }
    fetch(atlasPath)
      .then(res => res.json())
      .then((data: SpriteAtlas) => {
        atlasCache.set(atlasPath, data);
        if (!cancelled) setAtlas(data);
      })
      .catch(err => {
        if (!cancelled) console.error('[SpriteSheet] Failed to load atlas:', err);
      });

    return () => { cancelled = true; };
  }, [atlasPath]);

  // Get frames for current animation with improved fallback logic
  const frames = useMemo(() => {
    if (!atlas) return [];

    const allFrames = atlas.textures[0]?.frames ?? [];
    const sortFrames = (input: Frame[]) =>
      [...input].sort((a, b) => {
        const aNum = Number.parseInt(a.filename.split('frame')[1] ?? '0', 10);
        const bNum = Number.parseInt(b.filename.split('frame')[1] ?? '0', 10);
        return aNum - bNum;
      });

    // Try to find exact animation match
    let animFrames = sortFrames(allFrames.filter((f) => f.filename.startsWith(`${animation}/`)));
    if (animFrames.length > 0) {
      return animFrames;
    }

    // Fallback mappings for missing animations
    const fallbackMap: Record<string, string> = {
      'attack_A': 'attack',
      'attack_B': 'attack', 
      'jump_start': 'jump',
      'fall_loop': 'jump',
      'get_hit': 'idle', // If no get_hit animation, use idle
    };

    const fallbackAnimation = fallbackMap[animation];
    if (fallbackAnimation) {
      animFrames = sortFrames(allFrames.filter((f) => f.filename.startsWith(`${fallbackAnimation}/`)));
      if (animFrames.length > 0) {
        return animFrames;
      }
    }

    // Ultimate fallback to idle animation
    return sortFrames(allFrames.filter((f) => f.filename.startsWith('idle/')));
  }, [atlas, animation]);

  // Animate frames using interval - resets when animation changes
  useEffect(() => {
    if (frames.length === 0) return;

    // Start from frame 0 for new animation
    let localFrame = 0;
    
    // Set initial frame immediately via timeout (callback-based)
    const initTimeout = setTimeout(() => {
      setCurrentFrame(0);
    }, 0);

    const interval = setInterval(() => {
      localFrame += 1;
      if (localFrame >= frames.length) {
        if (!loop) {
          clearInterval(interval);
          onComplete?.();
          return;
        }
        localFrame = 0;
      }
      setCurrentFrame(localFrame);
    }, 1000 / frameRate);

    return () => {
      clearTimeout(initTimeout);
      clearInterval(interval);
    };
  }, [animation, frames.length, frameRate, loop, onComplete, frameOffset]);

  // --- Render ---
  // We NEVER return null. When atlas/frames are still loading we render an
  // invisible placeholder with a sensible default size so the layout is
  // preserved and the sprite fades in as soon as data is available.
  const isReady = atlas !== null && frames.length > 0;

  if (!isReady) {
    // Placeholder — same rough size as a typical knight frame so the
    // fighter bounding-box doesn't collapse while the atlas is fetching.
    return (
      <div
        style={{
          width: 72 * scale,
          height: 63 * scale,
          opacity: 0,
          ...style,
        }}
      />
    );
  }

  const normalizedOffset = ((frameOffset % frames.length) + frames.length) % frames.length;
  const frameIndex = (Math.min(currentFrame, frames.length - 1) + normalizedOffset) % frames.length;
  const frame = frames[frameIndex];
  const texture = atlas.textures[0];
  const { w: atlasWidth, h: atlasHeight } = texture.size;

  // Position the atlas image inside the sourceSize window so only the
  // correct frame is visible (the div clips with overflow: hidden).
  const spriteX = frame.spriteSourceSize?.x ?? 0;
  const spriteY = frame.spriteSourceSize?.y ?? 0;
  const offsetX = (spriteX - frame.frame.x) * scale;
  const offsetY = (spriteY - frame.frame.y) * scale;

  return (
    <div
      style={{
        position: 'relative',
        width: frame.sourceSize.w * scale,
        height: frame.sourceSize.h * scale,
        overflow: 'hidden',
        transform: flipX ? 'scaleX(-1)' : undefined,
        transformOrigin: 'center center',
        ...style,
      }}
    >
      <img
        src={imagePath}
        alt="sprite"
        onLoad={() => { imageReadyCache.add(imagePath); setImageLoaded(true); }}
        style={{
          position: 'absolute',
          left: offsetX,
          top: offsetY,
          width: atlasWidth * scale,
          height: atlasHeight * scale,
          imageRendering: 'pixelated',
          // Only show once the browser has actually decoded the image.
          // If imageLoaded was pre-set from cache this is already 1 on first paint.
          opacity: imageLoaded ? 1 : 0,
          transition: 'opacity 0.1s ease',
          pointerEvents: 'none',
        }}
      />
    </div>
  );
};

interface SimpleSpriteProps {
  imagePath: string;
  frameWidth: number;
  frameHeight: number;
  frameCount: number;
  frameRate?: number;
  loop?: boolean;
  scale?: number;
  flipX?: boolean;
  style?: React.CSSProperties;
}

export const SimpleSprite: React.FC<SimpleSpriteProps> = ({
  imagePath,
  frameWidth,
  frameHeight,
  frameCount,
  frameRate = 12,
  loop = true,
  scale = 1,
  flipX = false,
  style = {},
}) => {
  const [currentFrame, setCurrentFrame] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentFrame(prev => {
        const next = prev + 1;
        if (next >= frameCount) {
          return loop ? 0 : prev;
        }
        return next;
      });
    }, 1000 / frameRate);

    return () => clearInterval(interval);
  }, [frameCount, frameRate, loop]);

  return (
    <div
      style={{
        width: frameWidth * scale,
        height: frameHeight * scale,
        overflow: 'hidden',
        ...style,
      }}
    >
      <img
        src={imagePath}
        alt="sprite"
        style={{
          position: 'relative',
          left: -currentFrame * frameWidth * scale,
          width: frameWidth * frameCount * scale,
          height: frameHeight * scale,
          imageRendering: 'pixelated',
          transform: flipX ? 'scaleX(-1)' : 'none',
        }}
      />
    </div>
  );
};

export default SpriteSheet;
