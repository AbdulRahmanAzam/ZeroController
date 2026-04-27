'use strict';

/**
 * ZeroController Analytics Server
 *
 * Stores game sessions and ML training data in MongoDB Atlas.
 *
 * MongoDB Atlas URI: zerocontroller cluster
 * Database:         zerocontroller
 * Collections:
 *   sessions        — one document per match (summary + round stats)
 *   training_events — one document per attack action with full state context
 *
 * The training_events collection is the ML dataset.
 * Each document contains:
 *   - state:      raw game state when the attack was initiated
 *   - features:   normalised (0–1) feature vector ready to feed into a model
 *   - actionIndex: numeric class label (0–8) for the action taken
 *   - outcome:    whether the attack hit, damage dealt, blocked flag
 *   - playerWonRound / playerWonMatch: quality labels for imitation learning
 *
 * Endpoints:
 *   POST /api/session  — called once per match by the game
 *   GET  /api/stats    — count documents in both collections
 *   GET  /api/export   — export training_events as JSON for model training
 *                         ?gameMode=vs_ai&aiDifficulty=hard&winnersOnly=true&limit=5000
 */

const express  = require('express');
const cors     = require('cors');
const { MongoClient } = require('mongodb');
const { randomUUID }  = require('crypto');

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
const MONGODB_URI = 'mongodb+srv://k230061_db_user:nmKsiCT2TgIsYIzL@zerocontroller.l3ahizu.mongodb.net/?appName=zerocontroller';
const DB_NAME     = 'zerocontroller';
const PORT        = process.env.PORT || 3001;

// ---------------------------------------------------------------------------
// Action label → numeric index
// (0-based, matches the 9 ActionType values in the game)
// ---------------------------------------------------------------------------
const ACTION_INDEX = {
  idle: 0, move_forward: 1, move_backward: 2, jump: 3,
  block: 4, left_punch: 5, right_punch: 6, left_kick: 7, right_kick: 8,
};

// ---------------------------------------------------------------------------
// Build a normalised ML feature vector from a raw StateSnapshot.
// All values are in the 0–1 range (or small integers for round number).
// This is what you'd pass to sklearn / PyTorch / etc.
// ---------------------------------------------------------------------------
function buildFeatures(snap) {
  if (!snap) return null;
  const cw         = snap.canvasWidth  || 1200;
  const maxCooldown = 200; // ms — maximum cooldown duration in the game config

  return {
    // Health (1 = full, 0 = dead)
    p1_health_norm:   snap.p1_health  / 100,
    p2_health_norm:   snap.p2_health  / 100,
    // Health advantage from P1's perspective (+1 = P1 has full advantage, -1 = P2 does)
    health_diff_norm: (snap.p1_health - snap.p2_health) / 100,

    // Screen positions (0 = left edge, 1 = right edge)
    p1_x_norm:       snap.p1_x / cw,
    p2_x_norm:       snap.p2_x / cw,
    // Normalised distance between players (0 = touching, 1 = opposite edges)
    distance_norm:   Math.abs(snap.p1_x - snap.p2_x) / cw,

    // Binary combat state flags
    p1_is_attacking: snap.p1_isAttacking ? 1 : 0,
    p2_is_attacking: snap.p2_isAttacking ? 1 : 0,
    p1_is_blocking:  snap.p1_isBlocking  ? 1 : 0,
    p2_is_blocking:  snap.p2_isBlocking  ? 1 : 0,

    // Cooldowns (0 = free to act, 1 = fully locked)
    p1_atk_cd_norm:  Math.min(snap.p1_attackCooldown / maxCooldown, 1),
    p2_atk_cd_norm:  Math.min(snap.p2_attackCooldown / maxCooldown, 1),
    p1_hit_cd_norm:  Math.min(snap.p1_hitCooldown    / maxCooldown, 1),
    p2_hit_cd_norm:  Math.min(snap.p2_hitCooldown    / maxCooldown, 1),

    // Round context
    round_time_norm: Math.max(0, snap.roundTimeRemaining / 99),
    round_num:       snap.round,  // 1, 2, or 3 (small integer, fine as-is)
  };
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
async function main() {
  const client = new MongoClient(MONGODB_URI, {
    maxPoolSize:                10,
    serverSelectionTimeoutMS:   5000,
    connectTimeoutMS:          10000,
  });

  console.log('[Analytics] Connecting to MongoDB Atlas...');
  await client.connect();
  console.log('[Analytics] Connected.');

  const db = client.db(DB_NAME);

  // Ensure indexes (idempotent — safe to run on every startup)
  await db.collection('sessions').createIndex({ createdAt: -1 });
  await db.collection('sessions').createIndex({ gameMode: 1, createdAt: -1 });

  await db.collection('training_events').createIndex({ sessionId: 1 });
  await db.collection('training_events').createIndex({ action: 1, playerWonRound: 1 });
  await db.collection('training_events').createIndex({ gameMode: 1, aiDifficulty: 1 });
  await db.collection('training_events').createIndex({ playerId: 1, playerWonMatch: 1 });
  console.log('[Analytics] Indexes ready.');

  const app = express();
  app.use(cors());  // Local academic project — allow all origins
  app.use(express.json({ limit: '2mb' }));

  // -------------------------------------------------------------------------
  // POST /api/session
  //
  // Stores one complete match:
  //   1. A session summary document  → sessions collection
  //   2. Per-attack training events  → training_events collection
  //
  // Payload shape (sent by analyticsService.ts):
  //   { sessionId, gameMode, aiDifficulty, matchWinner, rounds: RoundStatistics[] }
  // -------------------------------------------------------------------------
  app.post('/api/session', async (req, res) => {
    try {
      const payload    = req.body;
      const sessionId  = payload.sessionId || randomUUID();
      const createdAt  = new Date();

      // --- 1. Session summary ---
      const sessionDoc = {
        sessionId,
        createdAt,
        gameMode:    payload.gameMode    ?? 'vs_player',
        aiDifficulty: payload.aiDifficulty ?? null,
        matchWinner: payload.matchWinner  ?? null,
        totalRounds: (payload.rounds ?? []).length,
        rounds: (payload.rounds ?? []).map(r => ({
          roundNumber:      r.roundNumber,
          winner:           r.winner,
          durationMs:       r.durationMs,
          finalHealth:      r.finalHealth,
          totalDamageDealt: r.totalDamageDealt,
          attacksAttempted: r.attacksAttempted,
          attacksLanded:    r.attacksLanded,
          accuracy: {
            p1: r.attacksAttempted?.p1 > 0
              ? +(r.attacksLanded.p1 / r.attacksAttempted.p1).toFixed(3) : 0,
            p2: r.attacksAttempted?.p2 > 0
              ? +(r.attacksLanded.p2 / r.attacksAttempted.p2).toFixed(3) : 0,
          },
          blocksPerformed: r.blocksPerformed,
          highestCombo:    r.highestCombo,
        })),
      };
      await db.collection('sessions').insertOne(sessionDoc);

      // --- 2. Per-action ML training events ---
      const trainingDocs = [];
      for (const round of (payload.rounds ?? [])) {
        // Map player ID → whether they won this round (for imitation-learning filter)
        const wonRound = { 1: round.winner === 1, 2: round.winner === 2 };

        for (const event of (round.actionLog ?? [])) {
          // Skip events without a state snapshot — they can't be used for ML
          if (!event.stateSnapshot) continue;

          const features = buildFeatures(event.stateSnapshot);

          trainingDocs.push({
            // ---- Identifiers ----
            sessionId,
            createdAt,
            gameMode:     payload.gameMode    ?? 'vs_player',
            aiDifficulty: payload.aiDifficulty ?? null,
            matchWinner:  payload.matchWinner  ?? null,
            roundNumber:  round.roundNumber,
            timestamp_ms: event.timestamp,

            // ---- Action (supervised classification label) ----
            playerId:    event.player,
            action:      event.action,
            actionIndex: ACTION_INDEX[event.action] ?? -1,

            // ---- Raw state (for inspection / custom feature engineering) ----
            state: event.stateSnapshot,

            // ---- Normalised ML features (ready to pass to a model directly) ----
            features,

            // ---- Outcome (reward / quality signal for RL or weighted training) ----
            outcome: {
              hit:         event.succeeded,
              damageDealt: event.damageDealt,
              wasBlocked:  event.wasBlocked,
            },

            // ---- Training-quality labels ----
            // playerWonRound = true → this player's decisions led to a round win.
            // Use this filter for imitation learning (learn from the winner).
            playerWonRound:  wonRound[event.player] ?? false,
            playerWonMatch:  payload.matchWinner === event.player,
          });
        }
      }

      if (trainingDocs.length > 0) {
        await db.collection('training_events').insertMany(trainingDocs);
      }

      console.log(`[Analytics] Saved session ${sessionId} — ${trainingDocs.length} training events`);
      res.json({ ok: true, sessionId, eventsStored: trainingDocs.length });

    } catch (err) {
      console.error('[/api/session] Error:', err);
      res.status(500).json({ ok: false, error: err.message });
    }
  });

  // -------------------------------------------------------------------------
  // GET /api/stats — document counts (health check / progress view)
  // -------------------------------------------------------------------------
  app.get('/api/stats', async (_req, res) => {
    try {
      const [sessions, events] = await Promise.all([
        db.collection('sessions').countDocuments(),
        db.collection('training_events').countDocuments(),
      ]);
      res.json({ ok: true, sessions, trainingEvents: events });
    } catch (err) {
      res.status(500).json({ ok: false, error: err.message });
    }
  });

  // -------------------------------------------------------------------------
  // GET /api/export — export training_events as JSON for model training
  //
  // Query params (all optional):
  //   gameMode      = 'vs_ai' | 'vs_player'
  //   aiDifficulty  = 'easy' | 'medium' | 'hard'
  //   winnersOnly   = 'true'  → only events where playerWonMatch = true
  //   playerId      = '1' | '2'
  //   limit         = number  (default 10000, max 50000)
  //
  // Returns only the fields needed for training:
  //   { features, actionIndex, action, outcome, playerWonRound, playerId }
  //
  // Example usage in Python:
  //   import requests, pandas as pd
  //   data = requests.get('http://localhost:3001/api/export?winnersOnly=true&limit=20000').json()
  //   df = pd.DataFrame(data['data'])
  //   X = pd.DataFrame(df['features'].tolist())
  //   y = df['actionIndex']
  // -------------------------------------------------------------------------
  app.get('/api/export', async (req, res) => {
    try {
      const filter = {};
      if (req.query.gameMode)           filter.gameMode    = req.query.gameMode;
      if (req.query.aiDifficulty)       filter.aiDifficulty = req.query.aiDifficulty;
      if (req.query.winnersOnly === 'true') filter.playerWonMatch = true;
      if (req.query.playerId)           filter.playerId    = parseInt(req.query.playerId);

      const limit = Math.min(parseInt(req.query.limit ?? '10000'), 50000);

      const events = await db.collection('training_events')
        .find(filter, {
          projection: {
            _id: 0,
            sessionId: 1, roundNumber: 1, timestamp_ms: 1,
            playerId: 1, action: 1, actionIndex: 1,
            features: 1, outcome: 1,
            playerWonRound: 1, playerWonMatch: 1,
            gameMode: 1, aiDifficulty: 1,
          },
        })
        .limit(limit)
        .toArray();

      res.json({ ok: true, count: events.length, data: events });
    } catch (err) {
      res.status(500).json({ ok: false, error: err.message });
    }
  });

  app.listen(PORT, () => {
    console.log(`\n[Analytics Server] Running → http://localhost:${PORT}`);
    console.log(`[Analytics Server] Database → ${DB_NAME}`);
    console.log(`\n  POST /api/session          store a match session`);
    console.log(`  GET  /api/stats            count sessions & training events`);
    console.log(`  GET  /api/export           export ML training data as JSON\n`);
  });

  // Graceful shutdown
  process.on('SIGINT', async () => {
    console.log('\n[Analytics] Shutting down...');
    await client.close();
    process.exit(0);
  });
}

main().catch(err => {
  console.error('[Analytics] Startup failed:', err.message);
  process.exit(1);
});
