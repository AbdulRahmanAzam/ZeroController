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
  eventId?: number;
  triggerSource?: 'early' | 'classifier' | null;
  captureAgeMs?: number;
  pipelineLatencyMs?: number;
  latencyStats?: PoseLatencyStats;
}

interface PoseLatencyBucket {
  lastMs: number;
  p50Ms: number;
  p95Ms: number;
}

type PoseLatencyStats = Record<string, PoseLatencyBucket>;

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
  eventId?: number;
  triggerSource?: 'early' | 'classifier' | null;
  captureAgeMs?: number;
  pipelineLatencyMs?: number;
  latencyStats?: PoseLatencyStats;
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

const ONE_SHOT_ACTIONS = new Set<ActionType>([
  'jump',
  'left_punch',
  'right_punch',
  'left_kick',
  'right_kick',
]);
const STATUS_UPDATE_INTERVAL_MS = 100;

function shouldEmitPoseInput(
  action: ActionType,
  data: PoseResponse,
  lastEventId: number | null,
): { emit: boolean; eventId: number | null } {
  if (!ONE_SHOT_ACTIONS.has(action)) {
    return { emit: true, eventId: lastEventId };
  }

  const eventIsNew = typeof data.eventId === 'number' && data.eventId > 0 && data.eventId !== lastEventId;
  const hasTriggerSignal = data.triggered === true || data.triggerSource != null;

  if (!hasTriggerSignal && !eventIsNew) {
    return { emit: false, eventId: lastEventId };
  }

  if (typeof data.eventId !== 'number') {
    return { emit: hasTriggerSignal, eventId: lastEventId };
  }

  if (data.eventId === lastEventId) {
    return { emit: false, eventId: lastEventId };
  }

  return { emit: true, eventId: data.eventId };
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
  const lastEventIdRef = useRef<number | null>(null);
  const lastStatusUpdateRef = useRef(0);
  const lastStatusActionRef = useRef<ActionType>('idle');
  const lastBridgeStatusRef = useRef<string | undefined>(undefined);
  const HEALTH_RETRY_MS = 2000;
  const WS_RETRY_MS = 2000;

  useEffect(() => {
    if (!enabled) {
      onStatus?.({ connection: 'idle', action: 'idle', confidence: 0 });
      return;
    }

    let cancelled = false;

    const disconnect = () => {
      lastEventIdRef.current = null;
      lastStatusUpdateRef.current = 0;
      lastStatusActionRef.current = 'idle';
      lastBridgeStatusRef.current = undefined;
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
          lastEventIdRef.current = null;
          lastStatusUpdateRef.current = 0;
          lastStatusActionRef.current = 'idle';
          lastBridgeStatusRef.current = undefined;
          onStatus?.({ connection: 'connected', action: 'idle', confidence: 0 });
        };

        ws.onmessage = (event) => {
          try {
            const data: PoseResponse = JSON.parse(event.data);
            const action = normalizeAction(data.action);
            const decision = shouldEmitPoseInput(action, data, lastEventIdRef.current);
            lastEventIdRef.current = decision.eventId;
            if (decision.emit) {
              onInput({
                playerId,
                action,
              });
            }

            const now = performance.now();
            const shouldUpdateStatus =
              data.triggered === true ||
              action !== lastStatusActionRef.current ||
              data.status !== lastBridgeStatusRef.current ||
              now - lastStatusUpdateRef.current >= STATUS_UPDATE_INTERVAL_MS;

            if (shouldUpdateStatus) {
              lastStatusUpdateRef.current = now;
              lastStatusActionRef.current = action;
              lastBridgeStatusRef.current = data.status;
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
                eventId: data.eventId,
                triggerSource: data.triggerSource,
                captureAgeMs: data.captureAgeMs,
                pipelineLatencyMs: data.pipelineLatencyMs,
                latencyStats: data.latencyStats,
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
  const lastEventIdRef = useRef<number | null>(null);
  
  useEffect(() => {
    if (!enabled) return;
    
    const poll = async () => {
      try {
        const response = await fetch(`${apiUrl}/pose/${playerId}`);
        if (response.ok) {
          const data: PoseResponse = await response.json();
          const action = normalizeAction(data.action);
          const decision = shouldEmitPoseInput(action, data, lastEventIdRef.current);
          lastEventIdRef.current = decision.eventId;
          if (decision.emit) {
            onInput({
              playerId,
              action,
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
