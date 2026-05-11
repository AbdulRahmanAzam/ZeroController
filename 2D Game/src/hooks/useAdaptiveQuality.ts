import { useEffect, useRef, useState } from 'react';
import type { QualityMode } from '../types/game';

interface UseAdaptiveQualityOptions {
  enabled: boolean;
}

const LOW_FPS_THRESHOLD = 55;
const RECOVER_FPS_THRESHOLD = 58;
const SAMPLE_MS = 1000;
const LOW_STREAK_LIMIT = 2;
const RECOVER_STREAK_LIMIT = 4;

export function useAdaptiveQuality({ enabled }: UseAdaptiveQualityOptions): QualityMode {
  const [quality, setQuality] = useState<QualityMode>(enabled ? 'medium' : 'high');
  const qualityRef = useRef(quality);

  useEffect(() => {
    qualityRef.current = quality;
  }, [quality]);

  useEffect(() => {
    let qualityResetFrame = 0;

    if (!enabled) {
      qualityResetFrame = requestAnimationFrame(() => setQuality('high'));
      return () => cancelAnimationFrame(qualityResetFrame);
    }

    qualityResetFrame = requestAnimationFrame(() => setQuality('medium'));

    let rafId = 0;
    let frames = 0;
    let sampleStart = performance.now();
    let lowStreak = 0;
    let recoverStreak = 0;

    const tick = (now: number) => {
      frames += 1;
      const elapsed = now - sampleStart;

      if (elapsed >= SAMPLE_MS) {
        const fps = (frames * 1000) / elapsed;
        const current = qualityRef.current;

        if (fps < LOW_FPS_THRESHOLD) {
          lowStreak += 1;
          recoverStreak = 0;
        } else if (fps > RECOVER_FPS_THRESHOLD) {
          recoverStreak += 1;
          lowStreak = 0;
        } else {
          lowStreak = 0;
          recoverStreak = 0;
        }

        if (lowStreak >= LOW_STREAK_LIMIT && current !== 'low') {
          setQuality(current === 'high' ? 'medium' : 'low');
          lowStreak = 0;
        } else if (recoverStreak >= RECOVER_STREAK_LIMIT && current !== 'high') {
          setQuality(current === 'low' ? 'medium' : 'high');
          recoverStreak = 0;
        }

        frames = 0;
        sampleStart = now;
      }

      rafId = requestAnimationFrame(tick);
    };

    rafId = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(qualityResetFrame);
      cancelAnimationFrame(rafId);
    };
  }, [enabled]);

  return quality;
}
