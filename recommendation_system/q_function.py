"""
Q-Function via Fitted Q Iteration (FQI).

A small PyTorch MLP that estimates Q(state, action) — the expected health
advantage at round-end if the player takes action `a` in game state `s`.

State features (15 dims, normalized — mirrors buildFeatures in server.js):
  p1_health_norm, p2_health_norm, health_diff_norm,
  p1_x_norm, p2_x_norm, distance_norm,
  p1_is_attacking, p2_is_attacking, p1_is_blocking, p2_is_blocking,
  p1_atk_cd_norm, p2_atk_cd_norm, p1_hit_cd_norm, p2_hit_cd_norm,
  round_time_norm

Output: Q-values for 9 actions. Q > 0 → action leads to health advantage.

Training: Fitted Q Iteration (FQI) on offline data from MongoDB training_events.
  reward = damageDealt (positive for hits, 0 for misses/blocked)
  γ = 0.95 (discount factor; rounds last ~90 frames after downsampling)

The model is only active once ≥ 50 rounds of data exist. Until then,
regret_score() returns None and the recommendation engine skips model D.
"""

from __future__ import annotations
import os
import json
import numpy as np
from typing import List, Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.optim as optim

from .feature_extractor import ACTION_NAMES, N_ACTIONS

_HERE       = os.path.dirname(__file__)
MODEL_PATH  = os.path.join(_HERE, 'models', 'q_network.pth')
META_PATH   = os.path.join(_HERE, 'models', 'q_meta.json')

GAMMA       = 0.95
LR          = 1e-3
BATCH_SIZE  = 256
N_EPOCHS    = 5      # FQI iterations (each uses full offline dataset)
MIN_ROUNDS  = 50     # Minimum rounds before Q-training activates
STATE_DIM   = 15


class _QNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, N_ACTIONS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QFunction:
    """
    Wraps _QNet with FQI training and per-action regret scoring.
    """

    def __init__(self) -> None:
        os.makedirs(os.path.join(_HERE, 'models'), exist_ok=True)
        self._model   = _QNet()
        self._trained = False
        self._rounds_trained_on = 0
        self._load_if_exists()

    def _load_if_exists(self) -> None:
        if os.path.exists(MODEL_PATH) and os.path.exists(META_PATH):
            try:
                self._model.load_state_dict(
                    torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
                )
                self._model.eval()
                with open(META_PATH) as f:
                    meta = json.load(f)
                self._rounds_trained_on = meta.get('rounds_trained_on', 0)
                self._trained = self._rounds_trained_on >= MIN_ROUNDS
            except Exception:
                pass

    def is_active(self) -> bool:
        return self._trained

    def train_from_events(self, training_events: List[Dict[str, Any]]) -> int:
        """
        Run FQI on MongoDB training_events.
        Each event needs: features (dict), outcome.damageDealt, actionIndex.
        Returns number of training samples used.
        """
        # Build (state, action, reward, next_state) tuples
        # We treat consecutive events from the same round as transitions.
        # Group events by (sessionId, roundNumber, playerId)
        from collections import defaultdict
        groups: Dict[Tuple, List] = defaultdict(list)
        for ev in training_events:
            key = (ev.get('sessionId', ''), ev.get('roundNumber', 0), ev.get('playerId', 1))
            groups[key].append(ev)

        states, actions, rewards, next_states = [], [], [], []
        for evs in groups.values():
            evs_sorted = sorted(evs, key=lambda e: e.get('timestamp_ms', 0))
            for i, ev in enumerate(evs_sorted):
                feats = ev.get('features')
                if feats is None:
                    continue
                state = self._features_dict_to_array(feats)
                if state is None:
                    continue

                action_idx = int(ev.get('actionIndex', 0))
                if action_idx < 0 or action_idx >= N_ACTIONS:
                    continue

                outcome = ev.get('outcome', {})
                reward  = float(outcome.get('damageDealt', 0))
                # Penalize blocked attacks slightly (wasted commitment)
                if outcome.get('wasBlocked', False):
                    reward = reward * 0.5

                # Next state: next event's features, or zeros at end of sequence
                if i + 1 < len(evs_sorted):
                    next_feats = evs_sorted[i + 1].get('features')
                    next_state = self._features_dict_to_array(next_feats) if next_feats else np.zeros(STATE_DIM)
                else:
                    next_state = np.zeros(STATE_DIM)

                states.append(state)
                actions.append(action_idx)
                rewards.append(reward)
                next_states.append(next_state)

        n = len(states)
        if n < BATCH_SIZE:
            return 0  # Not enough data

        S  = torch.tensor(np.array(states,      dtype=np.float32))
        A  = torch.tensor(np.array(actions),     dtype=torch.long)
        R  = torch.tensor(np.array(rewards,      dtype=np.float32))
        S2 = torch.tensor(np.array(next_states,  dtype=np.float32))

        optimizer = optim.Adam(self._model.parameters(), lr=LR)
        loss_fn   = nn.MSELoss()

        self._model.train()
        for epoch in range(N_EPOCHS):
            # Fitted Q: compute targets using current model
            with torch.no_grad():
                q_next = self._model(S2)
                targets = R + GAMMA * q_next.max(dim=1).values

            # Mini-batch SGD
            perm = torch.randperm(n)
            epoch_loss = 0.0
            for start in range(0, n, BATCH_SIZE):
                idx   = perm[start:start + BATCH_SIZE]
                s_b   = S[idx]; a_b = A[idx]; t_b = targets[idx]
                q_all = self._model(s_b)
                q_pred= q_all.gather(1, a_b.unsqueeze(1)).squeeze(1)
                loss  = loss_fn(q_pred, t_b)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

        self._model.eval()
        self._trained = True
        torch.save(self._model.state_dict(), MODEL_PATH)
        with open(META_PATH, 'w') as f:
            json.dump({'rounds_trained_on': self._rounds_trained_on}, f)

        return n

    def q_values(self, state: np.ndarray) -> np.ndarray:
        """Return Q-values for all 9 actions given a state vector."""
        with torch.no_grad():
            x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            return self._model(x).squeeze(0).numpy()

    def regret_score(
        self,
        action_log: List[Dict[str, Any]],
        player_id: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Compute per-action regret for the losing player's action log.

        regret(t) = max_a Q(s_t, a) - Q(s_t, a_taken)

        Returns:
          {
            top_regret_action  : str,    # action the player SHOULD have used most
            top_regret_moment_ms: float, # timestamp of highest single regret
            avg_regret         : float,  # mean regret across all events
            action_regrets     : dict,   # avg regret per action taken
            missed_action      : str,    # most underused high-value action
          }
        or None if model is not yet trained / no stateSnapshot data.
        """
        if not self._trained:
            return None

        from .feature_extractor import state_snapshot_to_features
        events = [e for e in action_log if e.get('player') == player_id and e.get('stateSnapshot')]
        if len(events) < 3:
            return None

        regrets   = []
        action_regret_totals: Dict[str, float] = {}
        action_regret_counts: Dict[str, int]   = {}
        missed_q: Dict[str, float]             = {}

        for e in events:
            snap       = e.get('stateSnapshot', {})
            state      = state_snapshot_to_features(snap, player_id)
            action_str = e.get('action', 'idle')
            a_idx      = ACTION_NAMES.index(action_str) if action_str in ACTION_NAMES else 0

            qv    = self.q_values(state)
            best  = float(qv.max())
            taken = float(qv[a_idx])
            reg   = max(0.0, best - taken)

            regrets.append((reg, e.get('timestamp', 0), action_str))

            action_regret_totals[action_str] = action_regret_totals.get(action_str, 0) + reg
            action_regret_counts[action_str] = action_regret_counts.get(action_str, 0) + 1

            # Track which action would have been best at this state
            best_action = ACTION_NAMES[int(np.argmax(qv))]
            missed_q[best_action] = missed_q.get(best_action, 0) + reg

        if not regrets:
            return None

        regrets.sort(key=lambda x: x[0], reverse=True)
        avg_regret = float(np.mean([r[0] for r in regrets]))

        # Action the player most should have used instead
        missed_action = max(missed_q, key=missed_q.get) if missed_q else 'block'

        action_regrets = {
            a: action_regret_totals[a] / action_regret_counts[a]
            for a in action_regret_totals
        }

        return {
            'top_regret_action':    regrets[0][2],
            'top_regret_moment_ms': regrets[0][1],
            'avg_regret':           avg_regret,
            'action_regrets':       action_regrets,
            'missed_action':        missed_action,
            'top_regret_value':     regrets[0][0],
        }

    @staticmethod
    def _features_dict_to_array(feats: Dict[str, Any]) -> Optional[np.ndarray]:
        """Convert MongoDB features dict to numpy array (15 dims)."""
        keys = [
            'p1_health_norm', 'p2_health_norm', 'health_diff_norm',
            'p1_x_norm', 'p2_x_norm', 'distance_norm',
            'p1_is_attacking', 'p2_is_attacking',
            'p1_is_blocking', 'p2_is_blocking',
            'p1_atk_cd_norm', 'p2_atk_cd_norm',
            'p1_hit_cd_norm', 'p2_hit_cd_norm',
            'round_time_norm',
        ]
        try:
            return np.array([float(feats.get(k, 0)) for k in keys], dtype=np.float64)
        except Exception:
            return None
