"""
Data Store — fetches training data from the Node.js analytics server.

The MongoDB Atlas analytics server (port 3001, server.js) is the single
source of truth for round history and ML training events. This module reads
from it via HTTP rather than duplicating data into a second database.

Two caches are maintained in memory:
  - round_count   : total number of sessions stored (for milestone checks)
  - training_data : the last-fetched batch of training_events

update_training_data() is called on server startup and after each saved round.
"""

from __future__ import annotations
import os
import json
import time
import logging
import asyncio
from typing import List, Dict, Any, Optional

import httpx

log = logging.getLogger(__name__)

# Analytics server base URL (Node.js, port 3001)
ANALYTICS_BASE = os.environ.get('ANALYTICS_URL', 'http://localhost:3001')

# How many training events to fetch per update
MAX_EVENTS = 20_000

# Minimum interval between full re-fetches (seconds)
REFETCH_INTERVAL = 60


class DataStore:
    def __init__(self) -> None:
        self._training_events: List[Dict[str, Any]] = []
        self._round_count: int = 0
        self._last_fetch_time: float = 0.0
        self._client: Optional[httpx.AsyncClient] = None

    async def start(self) -> None:
        self._client = httpx.AsyncClient(timeout=10.0)
        await self.refresh()

    async def stop(self) -> None:
        if self._client:
            await self._client.aclose()

    @property
    def training_events(self) -> List[Dict[str, Any]]:
        return self._training_events

    @property
    def round_count(self) -> int:
        return self._round_count

    async def refresh(self) -> None:
        """Fetch stats and training data from the analytics server."""
        now = time.monotonic()
        if now - self._last_fetch_time < REFETCH_INTERVAL:
            return

        await self._fetch_stats()
        await self._fetch_training_events()
        self._last_fetch_time = now

    async def force_refresh(self) -> None:
        """Force immediate re-fetch regardless of interval."""
        self._last_fetch_time = 0.0
        await self.refresh()

    async def _fetch_stats(self) -> None:
        if not self._client:
            return
        try:
            r = await self._client.get(f'{ANALYTICS_BASE}/api/stats')
            if r.status_code == 200:
                data = r.json()
                self._round_count = data.get('sessions', 0)
        except Exception as e:
            log.debug('Could not fetch analytics stats: %s', e)

    async def _fetch_training_events(self) -> None:
        """
        Fetch the most recent training_events from MongoDB via /api/export.
        Fetches winner-only events for imitation learning (higher quality data).
        """
        if not self._client:
            return
        try:
            r = await self._client.get(
                f'{ANALYTICS_BASE}/api/export',
                params={'limit': MAX_EVENTS},
            )
            if r.status_code == 200:
                data = r.json()
                self._training_events = data.get('data', [])
                log.info('Fetched %d training events from analytics server',
                         len(self._training_events))
        except Exception as e:
            log.debug('Could not fetch training events: %s', e)

    def get_player_freq_vectors(self) -> List:
        """
        Extract per-round action frequency vectors from training events.
        Used to update archetype centroids.
        Returns list of numpy arrays, shape (9,).
        """
        import numpy as np
        from .feature_extractor import ACTION_NAMES, N_ACTIONS

        # Group events by (sessionId, roundNumber, playerId)
        from collections import defaultdict
        groups: Dict = defaultdict(list)
        for ev in self._training_events:
            key = (ev.get('sessionId', ''), ev.get('roundNumber', 0), ev.get('playerId', 1))
            groups[key].append(ev.get('actionIndex', 0))

        vectors = []
        for action_indices in groups.values():
            freq = np.zeros(N_ACTIONS)
            for ai in action_indices:
                if 0 <= ai < N_ACTIONS:
                    freq[ai] += 1
            if freq.sum() > 0:
                vectors.append(freq / freq.sum())

        return vectors
