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
  const safeFrameRate = Math.max(1, Math.min(frameRate, 30));

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
      'attack_C': 'attack_B',
      'attack_B': 'attack_A',
      'jump_start': 'jump',
      'jump_loop': 'jump',
      'fall_loop': 'jump_loop',
      'guard_start': 'guard',
      'guard_end': 'guard',
      'get_hit': 'idle',
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

  // Pivot-aware canonical box (symmetric, bottom-anchored).
  //
  // Locks character feet/center to a fixed screen position across every frame
  // so attack/kick poses don't swim. Box is symmetric horizontally so the
  // parent wrapper can keep using bottom:0 + translateX(-50%) without knowing
  // about pivots.
  //
  // Across ALL atlas frames:
  //   maxHalf  = max(pivot.x, frame.w - pivot.x)   horizontal half-width
  //   maxAbove = max(pivot.y)                       distance pivot ↔ frame-top
  //   maxBelow = max(frame.h - pivot.y)             distance pivot ↔ frame-bot
  // Canonical box w = 2*maxHalf, h = maxAbove + maxBelow.
  // Canonical pivot = (maxHalf, maxAbove). Bottom of box = pivot + maxBelow.
  const canonical = useMemo(() => {
    const all = atlas?.textures[0]?.frames ?? [];
    let maxHalf = 0, maxAbove = 0, maxBelow = 0;
    for (const f of all) {
      const px = f.pivot?.x ?? f.sourceSize.w / 2;
      const py = f.pivot?.y ?? f.sourceSize.h;
      maxHalf = Math.max(maxHalf, px, f.sourceSize.w - px);
      maxAbove = Math.max(maxAbove, py);
      maxBelow = Math.max(maxBelow, f.sourceSize.h - py);
    }
    if (maxHalf === 0) return { w: 72, h: 70, pivotX: 36, pivotY: 70, below: 0 };
    return {
      w: maxHalf * 2,
      h: maxAbove + maxBelow,
      pivotX: maxHalf,
      pivotY: maxAbove,
      below: maxBelow,
    };
  }, [atlas]);

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
    }, 1000 / safeFrameRate);

    return () => {
      clearTimeout(initTimeout);
      clearInterval(interval);
    };
  }, [animation, frames.length, safeFrameRate, loop, onComplete, frameOffset]);

  const isReady = atlas !== null && frames.length > 0;

  if (!isReady) {
    return (
      <div
        style={{
          width: canonical.w * scale,
          height: canonical.h * scale,
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

  // Pivot alignment.
  // padX/padY = offset from canonical-box top-left to frame top-left
  //             so that frame's own pivot lines up with canonical pivot.
  const framePivotX = frame.pivot?.x ?? frame.sourceSize.w / 2;
  const framePivotY = frame.pivot?.y ?? frame.sourceSize.h;
  const padX = canonical.pivotX - framePivotX;
  const padY = canonical.pivotY - framePivotY;
  const spriteX = frame.spriteSourceSize?.x ?? 0;
  const spriteY = frame.spriteSourceSize?.y ?? 0;
  // Inner img shifted so frame sits at top-left of inner clip box.
  const imgLeft = (spriteX - frame.frame.x) * scale;
  const imgTop = (spriteY - frame.frame.y) * scale;

  return (
    <div
      style={{
        position: 'relative',
        width: canonical.w * scale,
        height: canonical.h * scale,
        overflow: 'hidden',
        transform: flipX ? 'scaleX(-1)' : undefined,
        transformOrigin: `${canonical.pivotX * scale}px ${canonical.pivotY * scale}px`,
        ...style,
      }}
    >
      {/* Inner clip box sized to current frame's sourceSize, positioned so
          the frame's pivot lands on the canonical pivot. Prevents adjacent
          atlas frames from bleeding into the canonical box. */}
      <div
        style={{
          position: 'absolute',
          left: padX * scale,
          top: padY * scale,
          width: frame.sourceSize.w * scale,
          height: frame.sourceSize.h * scale,
          overflow: 'hidden',
        }}
      >
        <img
          src={imagePath}
          alt="sprite"
          onLoad={() => { imageReadyCache.add(imagePath); setImageLoaded(true); }}
          style={{
            position: 'absolute',
            left: imgLeft,
            top: imgTop,
            width: atlasWidth * scale,
            height: atlasHeight * scale,
            imageRendering: 'pixelated',
            opacity: imageLoaded ? 1 : 0,
            transition: 'opacity 0.1s ease',
            pointerEvents: 'none',
          }}
        />
      </div>
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
  const safeFrameRate = Math.max(1, Math.min(frameRate, 30));

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentFrame(prev => {
        const next = prev + 1;
        if (next >= frameCount) {
          return loop ? 0 : prev;
        }
        return next;
      });
    }, 1000 / safeFrameRate);

    return () => clearInterval(interval);
  }, [frameCount, safeFrameRate, loop]);

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
