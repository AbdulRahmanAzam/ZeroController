/**
 * Analytics Service
 *
 * Sends completed match data to the local analytics server (server/server.js)
 * which writes it to MongoDB Atlas.
 *
 * Fire-and-forget: the game works normally even if the server is unreachable.
 * The stored data is shaped for future ML / recommendation-model training.
 */

import type { RoundStatistics, GameMode, AIDifficulty } from '../types/game';

/** URL of the local Express analytics server */
const SERVER_URL = 'http://localhost:3001';

export interface SessionPayload {
  sessionId: string;
  gameMode: GameMode;
  aiDifficulty: AIDifficulty | null;
  matchWinner: 1 | 2 | 'draw' | null;
  /** All rounds including the final one (compiled by the caller) */
  rounds: RoundStatistics[];
}

/**
 * Persist one full match session to MongoDB.
 * Catches all network errors so the game is never blocked.
 */
export async function sendSession(payload: SessionPayload): Promise<void> {
  try {
    const res = await fetch(`${SERVER_URL}/api/session`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (res.ok) {
      const data = await res.json() as { eventsStored: number; sessionId: string };
      console.info(
        `[Analytics] Session ${data.sessionId} saved — ${data.eventsStored} training events stored.`,
      );
    }
  } catch {
    console.warn(
      '[Analytics] Server unreachable — session not stored.\n' +
      '  Start the server with: cd "2D Game/server" && npm start',
    );
  }
}
