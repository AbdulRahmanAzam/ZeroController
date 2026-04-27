import { useEffect, useRef, useCallback } from 'react';

interface GameLoopOptions {
  onUpdate: (deltaTime: number) => void;
  onRender: () => void;
  targetFPS?: number;
}

export function useGameLoop({ onUpdate, onRender, targetFPS = 60 }: GameLoopOptions) {
  const frameRef = useRef<number>(0);
  const lastTimeRef = useRef<number>(0);
  const accumulatorRef = useRef<number>(0);
  const isRunningRef = useRef<boolean>(false);
  const tickRef = useRef<((currentTime: number) => void) | null>(null);
  const onUpdateRef = useRef(onUpdate);
  const onRenderRef = useRef(onRender);
  const frameTime = 1000 / targetFPS;
  const maxDeltaTime = frameTime * 5; // Prevent spiral of death

  useEffect(() => {
    onUpdateRef.current = onUpdate;
  }, [onUpdate]);

  useEffect(() => {
    onRenderRef.current = onRender;
  }, [onRender]);

  useEffect(() => {
    tickRef.current = (currentTime: number) => {
      if (!isRunningRef.current) return;

      if (lastTimeRef.current === 0) {
        lastTimeRef.current = currentTime;
      }

      let deltaTime = currentTime - lastTimeRef.current;
      // Clamp delta time to prevent large jumps
      deltaTime = Math.min(deltaTime, maxDeltaTime);
      
      lastTimeRef.current = currentTime;
      accumulatorRef.current += deltaTime;

      // Fixed timestep updates with accumulator cleanup
      let updateCount = 0;
      while (accumulatorRef.current >= frameTime && updateCount < 3) {
        onUpdateRef.current(frameTime);
        accumulatorRef.current -= frameTime;
        updateCount++;
      }

      // If we hit the update limit, reset accumulator to prevent spiral
      if (updateCount >= 3) {
        accumulatorRef.current = 0;
      }

      onRenderRef.current();
      frameRef.current = requestAnimationFrame(tickRef.current!);
    };
  }, [frameTime, maxDeltaTime]);

  const start = useCallback(() => {
    if (isRunningRef.current || !tickRef.current) return;
    isRunningRef.current = true;
    lastTimeRef.current = 0;
    accumulatorRef.current = 0;
    frameRef.current = requestAnimationFrame(tickRef.current);
  }, []);
  
  const stop = useCallback(() => {
    isRunningRef.current = false;
    if (frameRef.current) {
      cancelAnimationFrame(frameRef.current);
    }
  }, []);
  
  useEffect(() => {
    return () => {
      stop();
    };
  }, [stop]);
  
  return { start, stop };
}
