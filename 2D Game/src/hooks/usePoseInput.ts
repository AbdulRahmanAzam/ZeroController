import { useEffect, useRef } from 'react';
import type { PlayerInput, ActionType } from '../types/game';

interface UsePoseInputOptions {
  onInput: (input: PlayerInput) => void;
  enabled?: boolean;
  apiUrl?: string;
  playerId: 1 | 2;
  onStatus?: (status: PoseInputStatus) => void;
}

// API response from FastAPI pose detection
interface PoseResponse {
  action: ActionType;
  confidence: number;
  timestamp: number;
  status?: string;
  message?: string;
  bufferFill?: number;
  sequenceLength?: number;
  fps?: number;
  triggered?: boolean;
}

export interface PoseInputStatus {
  connection: 'idle' | 'connecting' | 'connected' | 'disconnected' | 'error' | 'unavailable';
  bridgeStatus?: string;
  message?: string;
  action: ActionType;
  confidence: number;
  bufferFill?: number;
  sequenceLength?: number;
  fps?: number;
  timestamp?: number;
}

const ACTION_ALIASES: Readonly<Record<string, ActionType>> = Object.freeze({
  idle: 'idle',
  forward: 'move_forward',
  backward: 'move_backward',
  move_forward: 'move_forward',
  move_backward: 'move_backward',
  jump: 'jump',
  block: 'block',
  left_punch: 'left_punch',
  right_punch: 'right_punch',
  left_kick: 'left_kick',
  right_kick: 'right_kick',
});

function normalizeAction(action: string | undefined): ActionType {
  return action ? ACTION_ALIASES[action] ?? 'idle' : 'idle';
}

export function usePoseInput({
  onInput,
  enabled = true,
  apiUrl = 'http://localhost:8000',
  playerId,
  onStatus,
}: UsePoseInputOptions) {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const HEALTH_RETRY_MS = 2000;
  const WS_RETRY_MS = 2000;

  useEffect(() => {
    if (!enabled) {
      onStatus?.({ connection: 'idle', action: 'idle', confidence: 0 });
      return;
    }

    let cancelled = false;

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

    const scheduleReconnect = (connection: PoseInputStatus['connection'], message: string, delayMs: number) => {
      if (cancelled) return;
      onStatus?.({
        connection,
        action: 'idle',
        confidence: 0,
        message,
      });
      reconnectTimeoutRef.current = setTimeout(() => {
        void connect();
      }, delayMs);
    };

    const checkBridgeHealth = async () => {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 1200);
      try {
        const response = await fetch(`${apiUrl}/health`, {
          cache: 'no-store',
          signal: controller.signal,
        });
        return response.ok;
      } catch {
        return false;
      } finally {
        clearTimeout(timeout);
      }
    };

    const connect = async () => {
      if (cancelled) return;

      onStatus?.({ connection: 'connecting', action: 'idle', confidence: 0 });
      const bridgeReady = await checkBridgeHealth();
      if (!bridgeReady) {
        scheduleReconnect(
          'unavailable',
          'ZeroController bridge is unavailable. Run python run_model.py.',
          HEALTH_RETRY_MS,
        );
        return;
      }

      try {
        const ws = new WebSocket(`${apiUrl.replace('http', 'ws')}/ws/pose/${playerId}`);

        ws.onopen = () => {
          onStatus?.({ connection: 'connected', action: 'idle', confidence: 0 });
        };

        ws.onmessage = (event) => {
          try {
            const data: PoseResponse = JSON.parse(event.data);
            const action = normalizeAction(data.action);
            onStatus?.({
              connection: 'connected',
              bridgeStatus: data.status,
              message: data.message,
              action,
              confidence: data.confidence ?? 0,
              bufferFill: data.bufferFill,
              sequenceLength: data.sequenceLength,
              fps: data.fps,
              timestamp: data.timestamp,
            });

            if (action === 'idle' || data.triggered || data.confidence > 0.6) {
              onInput({
                playerId,
                action,
              });
            }
          } catch (error) {
            console.error('Failed to parse pose data:', error);
          }
        };

        ws.onerror = () => {
          onStatus?.({ connection: 'error', action: 'idle', confidence: 0 });
        };

        ws.onclose = () => {
          if (cancelled) return;
          scheduleReconnect('disconnected', 'Reconnecting to ZeroController...', WS_RETRY_MS);
        };

        wsRef.current = ws;
      } catch {
        scheduleReconnect('error', 'Reconnecting to ZeroController...', WS_RETRY_MS);
      }
    };

    void connect();

    return () => {
      cancelled = true;
      disconnect();
    };
  }, [enabled, apiUrl, playerId, onInput, onStatus]);
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
