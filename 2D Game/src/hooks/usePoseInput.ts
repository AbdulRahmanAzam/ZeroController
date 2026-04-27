import { useEffect, useRef } from 'react';
import type { PlayerInput, ActionType } from '../types/game';

interface UsePoseInputOptions {
  onInput: (input: PlayerInput) => void;
  enabled?: boolean;
  apiUrl?: string;
  playerId: 1 | 2;
}

// API response from FastAPI pose detection
interface PoseResponse {
  action: ActionType;
  confidence: number;
  timestamp: number;
}

export function usePoseInput({
  onInput,
  enabled = true,
  apiUrl = 'http://localhost:8000',
  playerId,
}: UsePoseInputOptions) {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const retryCountRef = useRef(0);
  const MAX_RETRIES = 5;

  useEffect(() => {
    if (!enabled) return;

    let cancelled = false;
    retryCountRef.current = 0;

    const disconnect = () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };

    const connect = () => {
      if (cancelled) return;
      if (retryCountRef.current >= MAX_RETRIES) {
        console.warn(`[Player ${playerId}] Max reconnection attempts (${MAX_RETRIES}) reached. Pose input disabled.`);
        return;
      }

      try {
        const ws = new WebSocket(`${apiUrl.replace('http', 'ws')}/ws/pose/${playerId}`);

        ws.onopen = () => {
          console.log(`[Player ${playerId}] Connected to pose detection`);
          retryCountRef.current = 0; // Reset on successful connection
        };

        ws.onmessage = (event) => {
          try {
            const data: PoseResponse = JSON.parse(event.data);
            if (data.confidence > 0.6) {
              onInput({
                playerId,
                action: data.action,
              });
            }
          } catch (error) {
            console.error('Failed to parse pose data:', error);
          }
        };

        ws.onerror = (error) => {
          console.error(`[Player ${playerId}] WebSocket error:`, error);
        };

        ws.onclose = () => {
          if (cancelled) return;
          retryCountRef.current += 1;
          const backoffMs = Math.min(2000 * Math.pow(2, retryCountRef.current - 1), 32000);
          console.log(`[Player ${playerId}] Reconnecting in ${backoffMs}ms (attempt ${retryCountRef.current}/${MAX_RETRIES})`);
          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, backoffMs);
        };

        wsRef.current = ws;
      } catch (error) {
        console.error('Failed to connect:', error);
        retryCountRef.current += 1;
        const backoffMs = Math.min(2000 * Math.pow(2, retryCountRef.current - 1), 32000);
        reconnectTimeoutRef.current = setTimeout(() => {
          connect();
        }, backoffMs);
      }
    };

    connect();

    return () => {
      cancelled = true;
      disconnect();
    };
  }, [enabled, apiUrl, playerId, onInput]);
}

// Alternative: Polling-based input (fallback if WebSocket not available)
export function usePoseInputPolling({
  onInput,
  enabled = true,
  apiUrl = 'http://localhost:8000',
  playerId,
  pollInterval = 50, // 20 FPS
}: UsePoseInputOptions & { pollInterval?: number }) {
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  
  useEffect(() => {
    if (!enabled) return;
    
    const poll = async () => {
      try {
        const response = await fetch(`${apiUrl}/pose/${playerId}`);
        if (response.ok) {
          const data: PoseResponse = await response.json();
          if (data.confidence > 0.6) {
            onInput({
              playerId,
              action: data.action,
            });
          }
        }
      } catch {
        // Silent fail for polling
      }
    };
    
    intervalRef.current = setInterval(poll, pollInterval);
    
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [enabled, apiUrl, playerId, pollInterval, onInput]);
}
