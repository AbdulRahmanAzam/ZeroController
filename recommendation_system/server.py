"""
ZeroController Recommendation Server — FastAPI on port 8001.

Endpoints:
  POST /recommend      — receive RoundStatistics + loser, return recommendations
  GET  /health         — server status, model readiness, round count
  POST /train          — manually trigger model update from MongoDB data

The server reads training data from the Node.js analytics server (port 3001)
and uses four ML models to generate per-round coaching recommendations.
"""

from __future__ import annotations
import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .data_store           import DataStore
from .payoff_matrix        import PayoffMatrix
from .archetype_model      import ArchetypeModel
from .q_function           import QFunction
from .recommendation_engine import RecommendationEngine

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(name)s — %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

MIN_ROUNDS_FOR_ARCHETYPE_UPDATE = 20
MIN_ROUNDS_FOR_Q_TRAINING       = 50

# ── Singletons (initialised in lifespan) ────────────────────────────────────
_data_store: Optional[DataStore]             = None
_payoff:     Optional[PayoffMatrix]          = None
_archetype:  Optional[ArchetypeModel]        = None
_q:          Optional[QFunction]             = None
_engine:     Optional[RecommendationEngine]  = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _data_store, _payoff, _archetype, _q, _engine

    log.info('Initialising recommendation models...')
    _data_store = DataStore()
    _payoff     = PayoffMatrix()
    _archetype  = ArchetypeModel()
    _q          = QFunction()
    _engine     = RecommendationEngine(_payoff, _archetype, _q)

    await _data_store.start()

    # Initial model update from any existing MongoDB data
    asyncio.create_task(_update_models())

    log.info('Recommendation server ready.')
    yield

    await _data_store.stop()
    log.info('Recommendation server shut down.')


app = FastAPI(
    title='ZeroController Recommendation API',
    version='1.0.0',
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['GET', 'POST'],
    allow_headers=['*'],
)


# ── Pydantic models ──────────────────────────────────────────────────────────

class StateSnapshot(BaseModel):
    p1_x: float = 0; p1_y: float = 0
    p2_x: float = 0; p2_y: float = 0
    p1_health: float = 100; p2_health: float = 100
    p1_isAttacking: bool = False; p2_isAttacking: bool = False
    p1_isBlocking:  bool = False; p2_isBlocking:  bool = False
    p1_attackCooldown: float = 0; p2_attackCooldown: float = 0
    p1_hitCooldown: float = 0; p2_hitCooldown: float = 0
    roundTimeRemaining: float = 99
    round: int = 1
    canvasWidth: float = 1200; canvasHeight: float = 800


class ActionEvent(BaseModel):
    timestamp: float
    player: int
    action: str
    succeeded: bool = False
    damageDealt: float = 0
    wasBlocked: bool = False
    stateSnapshot: Optional[StateSnapshot] = None


class RoundStatistics(BaseModel):
    roundNumber: int = 1
    winner: Any  # 1 | 2 | 'draw'
    durationMs: float = 0
    finalHealth: Dict[str, float]       = Field(default_factory=lambda: {'p1': 0, 'p2': 0})
    totalDamageDealt: Dict[str, float]  = Field(default_factory=lambda: {'p1': 0, 'p2': 0})
    attacksAttempted: Dict[str, int]    = Field(default_factory=lambda: {'p1': 0, 'p2': 0})
    attacksLanded: Dict[str, int]       = Field(default_factory=lambda: {'p1': 0, 'p2': 0})
    blocksPerformed: Dict[str, int]     = Field(default_factory=lambda: {'p1': 0, 'p2': 0})
    highestCombo: Dict[str, int]        = Field(default_factory=lambda: {'p1': 0, 'p2': 0})
    actionLog: List[ActionEvent]        = Field(default_factory=list)


class RecommendRequest(BaseModel):
    round_stats: RoundStatistics
    loser: int                # 1 or 2
    generate_llm: bool = False


class RecommendResponse(BaseModel):
    recommendations: List[Dict[str, Any]]
    insights: Dict[str, Any]


# ── Routes ───────────────────────────────────────────────────────────────────

@app.post('/recommend', response_model=RecommendResponse)
async def recommend(req: RecommendRequest):
    if _engine is None:
        raise HTTPException(status_code=503, detail='Engine not ready')

    if req.loser not in (1, 2):
        raise HTTPException(status_code=400, detail='loser must be 1 or 2')

    # Convert Pydantic model to plain dict for the engine
    stats_dict = _round_stats_to_dict(req.round_stats)

    try:
        result = _engine.recommend(stats_dict, req.loser, req.generate_llm)
    except Exception as e:
        log.exception('recommend() failed: %s', e)
        raise HTTPException(status_code=500, detail=str(e))

    return result


@app.get('/health')
async def health():
    rounds = _data_store.round_count if _data_store else 0
    return {
        'ok': True,
        'rounds_in_db':       rounds,
        'q_model_active':     _q.is_active() if _q else False,
        'payoff_loaded':      _payoff is not None,
        'archetype_loaded':   _archetype is not None,
        'milestones': {
            'archetype_update': rounds >= MIN_ROUNDS_FOR_ARCHETYPE_UPDATE,
            'q_training':       rounds >= MIN_ROUNDS_FOR_Q_TRAINING,
        },
    }


@app.post('/train')
async def trigger_training():
    """Manually trigger model update from MongoDB data."""
    asyncio.create_task(_update_models(force=True))
    return {'ok': True, 'message': 'Training scheduled in background'}


# ── Background model update ──────────────────────────────────────────────────

async def _update_models(force: bool = False) -> None:
    """
    Fetch latest data from MongoDB and update models that have hit their
    data milestone threshold.
    """
    if _data_store is None:
        return

    if force:
        await _data_store.force_refresh()
    else:
        await _data_store.refresh()

    rounds = _data_store.round_count
    events = _data_store.training_events

    if not events:
        log.info('No training data available yet (play some rounds first).')
        return

    log.info('Updating models from %d training events (%d sessions)',
             len(events), rounds)

    # Payoff matrix — always update from new events
    try:
        _payoff.update_from_events(events)
        log.info('Payoff matrix updated.')
    except Exception as e:
        log.warning('Payoff update failed: %s', e)

    # Archetype K-means — only when enough round data
    if rounds >= MIN_ROUNDS_FOR_ARCHETYPE_UPDATE:
        try:
            freq_vectors = _data_store.get_player_freq_vectors()
            if freq_vectors:
                _archetype.update_from_round_data(freq_vectors)
                log.info('Archetype model updated with %d vectors.', len(freq_vectors))
        except Exception as e:
            log.warning('Archetype update failed: %s', e)

    # Q-function FQI — only when enough round data
    if rounds >= MIN_ROUNDS_FOR_Q_TRAINING:
        try:
            n = await asyncio.to_thread(_q.train_from_events, events)
            _q._rounds_trained_on = rounds
            log.info('Q-function trained on %d samples.', n)
        except Exception as e:
            log.warning('Q-function training failed: %s', e)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _round_stats_to_dict(rs: RoundStatistics) -> Dict[str, Any]:
    """Convert Pydantic model to plain dict. ActionEvents include stateSnapshot."""
    action_log = []
    for ev in rs.actionLog:
        ev_dict: Dict[str, Any] = {
            'timestamp':   ev.timestamp,
            'player':      ev.player,
            'action':      ev.action,
            'succeeded':   ev.succeeded,
            'damageDealt': ev.damageDealt,
            'wasBlocked':  ev.wasBlocked,
        }
        if ev.stateSnapshot:
            ev_dict['stateSnapshot'] = ev.stateSnapshot.model_dump()
        action_log.append(ev_dict)

    return {
        'roundNumber':      rs.roundNumber,
        'winner':           rs.winner,
        'durationMs':       rs.durationMs,
        'finalHealth':      rs.finalHealth,
        'totalDamageDealt': rs.totalDamageDealt,
        'attacksAttempted': rs.attacksAttempted,
        'attacksLanded':    rs.attacksLanded,
        'blocksPerformed':  rs.blocksPerformed,
        'highestCombo':     rs.highestCombo,
        'actionLog':        action_log,
    }


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    uvicorn.run(
        'recommendation_system.server:app',
        host='0.0.0.0',
        port=8001,
        reload=False,
        log_level='info',
    )
