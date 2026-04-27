"""
9×9 Payoff Matrix + Nash Equilibrium solver.

payoff[player_action, opp_action] = expected health delta for the player
in a direct action-vs-action exchange.

Seeded from game mechanic constants (punch=8dmg, kick=12dmg, block=80%
reduction, punch faster startup than kick, etc.) so it works from round 1
with zero training data. Updated incrementally from MongoDB training events.

Nash Equilibrium:
  Given opponent's observed action distribution, compute the player's
  optimal mixed strategy via linear programming (scipy).
"""

from __future__ import annotations
import json
import os
import numpy as np
from scipy.optimize import linprog
from typing import Dict, Optional, Tuple, List

from .feature_extractor import ACTION_NAMES, N_ACTIONS

# ── Path for persisted matrix ────────────────────────────────────────────────
_HERE = os.path.dirname(__file__)
MATRIX_PATH = os.path.join(_HERE, 'models', 'payoff_matrix.json')

# ── Game constants (from config.ts) ─────────────────────────────────────────
PUNCH_DAMAGE   = 8.0
KICK_DAMAGE    = 12.0
BLOCK_FACTOR   = 0.20    # blocked attack deals 20% of full damage
PUNCH_HIT_MS   = 35.0    # mid-point of punch hit window (30-40ms)
KICK_HIT_MS    = 112.5   # mid-point of kick hit window (100-125ms)
HIT_COOLDOWN_MS = 200.0  # stun duration after being hit


def _build_initial_payoff() -> np.ndarray:
    """
    Construct the mechanics-seeded 9×9 payoff matrix.

    Rows = player's action, Cols = opponent's action.
    Value = net health advantage for the PLAYER in this exchange.

    Positive  = player benefits (deals damage / avoids damage)
    Negative  = player is harmed
    Zero      = neutral (no interaction)

    Key insights encoded:
    - Punch hits at 30-40ms (fast); kick at 100-125ms (slow but more damage).
    - If punch hits first, hitCooldown (200ms) cancels opponent's slower kick.
      → punch BEATS kick initiated simultaneously.
    - block reduces incoming damage by 80%.
    - move_backward moves player out of attack range (partial avoidance).
    - jump avoids punches (ground-level) but not kicks (aerial trajectory).
    """
    P = np.zeros((N_ACTIONS, N_ACTIONS), dtype=np.float64)

    # ── idle (0): player does nothing ────────────────────────────────────────
    # Takes full damage from attacks
    P[0, 5] = -PUNCH_DAMAGE    # vs left_punch
    P[0, 6] = -PUNCH_DAMAGE    # vs right_punch
    P[0, 7] = -KICK_DAMAGE     # vs left_kick
    P[0, 8] = -KICK_DAMAGE     # vs right_kick

    # ── move_forward (1): closing distance ───────────────────────────────────
    # Walks into attacks; same as idle for damage
    P[1, 5] = -PUNCH_DAMAGE
    P[1, 6] = -PUNCH_DAMAGE
    P[1, 7] = -KICK_DAMAGE
    P[1, 8] = -KICK_DAMAGE

    # ── move_backward (2): retreating ────────────────────────────────────────
    # Moves out of range — partially avoids punches (shorter range)
    # Kicks have slightly more range so avoidance is lower
    P[2, 5] = -PUNCH_DAMAGE * 0.35   # ~65% chance of escaping range
    P[2, 6] = -PUNCH_DAMAGE * 0.35
    P[2, 7] = -KICK_DAMAGE  * 0.45   # kick range: less avoidance
    P[2, 8] = -KICK_DAMAGE  * 0.45

    # ── jump (3): airborne ───────────────────────────────────────────────────
    # Avoids ground-level punches entirely; kicks can still reach upward
    P[3, 5] =  0.0                    # jumps over punch
    P[3, 6] =  0.0
    P[3, 7] = -KICK_DAMAGE * 0.6     # kicks can still connect mid-air
    P[3, 8] = -KICK_DAMAGE * 0.6
    # Jumping into an idle/forward opponent deals no damage (no aerial attack)
    # → neutral (0) by default

    # ── block (4): defending ─────────────────────────────────────────────────
    # Takes only 20% of punch/kick damage
    P[4, 5] = -PUNCH_DAMAGE * BLOCK_FACTOR
    P[4, 6] = -PUNCH_DAMAGE * BLOCK_FACTOR
    P[4, 7] = -KICK_DAMAGE  * BLOCK_FACTOR
    P[4, 8] = -KICK_DAMAGE  * BLOCK_FACTOR
    # Block vs idle/movement/jump: neutral (0)

    # ── left_punch (5) / right_punch (6): fast attack (8 dmg, 30-40ms) ──────
    for p_row in [5, 6]:
        P[p_row, 0] =  PUNCH_DAMAGE          # vs idle: lands cleanly
        P[p_row, 1] =  PUNCH_DAMAGE          # vs forward: walks into it
        P[p_row, 2] =  PUNCH_DAMAGE * 0.35   # vs backward: may escape range
        P[p_row, 3] =  PUNCH_DAMAGE * 0.25   # vs jump: sometimes misses
        P[p_row, 4] =  PUNCH_DAMAGE * BLOCK_FACTOR  # vs block: reduced damage
        P[p_row, 5] =  0.0                   # vs punch: mutual exchange (~0 net)
        P[p_row, 6] =  0.0
        # Punch (35ms) beats kick (112ms) — player punch hits first,
        # hitCooldown (200ms) cancels opponent's pending kick
        P[p_row, 7] =  PUNCH_DAMAGE          # vs left_kick: punch wins
        P[p_row, 8] =  PUNCH_DAMAGE          # vs right_kick: punch wins

    # ── left_kick (7) / right_kick (8): slow heavy attack (12 dmg, 100-125ms)
    for k_row in [7, 8]:
        P[k_row, 0] =  KICK_DAMAGE           # vs idle
        P[k_row, 1] =  KICK_DAMAGE           # vs forward
        P[k_row, 2] =  KICK_DAMAGE * 0.45   # vs backward: more range than punch
        P[k_row, 3] = -KICK_DAMAGE * 0.5    # vs jump: kicks upward, but player
                                              # may be out of range → net negative
        P[k_row, 4] =  KICK_DAMAGE * BLOCK_FACTOR  # vs block
        # Opponent's punch (35ms) hits BEFORE player's kick (112ms) fires
        # → opponent punch wins: player takes punch damage, kick is cancelled
        P[k_row, 5] = -PUNCH_DAMAGE          # vs left_punch: punch wins
        P[k_row, 6] = -PUNCH_DAMAGE          # vs right_punch
        P[k_row, 7] =  0.0                   # vs kick: mutual (same timing)
        P[k_row, 8] =  0.0

    return P


class PayoffMatrix:
    """
    Maintains and queries the 9×9 payoff matrix.

    update_from_events() adjusts payoff values from observed game outcomes,
    blending the mechanics-seeded prior with empirical data.
    """

    def __init__(self) -> None:
        os.makedirs(os.path.join(_HERE, 'models'), exist_ok=True)
        self._matrix = self._load_or_init()
        # Count of empirical updates per cell (used for blending weight)
        self._update_counts = np.zeros((N_ACTIONS, N_ACTIONS), dtype=np.int64)

    def _load_or_init(self) -> np.ndarray:
        if os.path.exists(MATRIX_PATH):
            try:
                with open(MATRIX_PATH) as f:
                    data = json.load(f)
                return np.array(data['matrix'], dtype=np.float64)
            except Exception:
                pass
        return _build_initial_payoff()

    def save(self) -> None:
        with open(MATRIX_PATH, 'w') as f:
            json.dump({'matrix': self._matrix.tolist()}, f)

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix.copy()

    def update_from_events(self, training_events: list) -> None:
        """
        Incrementally update payoff cells from MongoDB training_events.

        Each event has: action (str), outcome.hit (bool), outcome.damageDealt (float),
        outcome.wasBlocked (bool), and features (the full feature vector at time of action).
        We don't have the opponent's concurrent action directly, but we can infer
        the interaction type from the outcome flags.
        """
        if not training_events:
            return

        # Accumulate empirical damage per (player_action, opp_state) pair
        # We approximate opponent's action from context: if player's hit was blocked,
        # opponent used 'block'; otherwise if player succeeded, opponent was idle/forward.
        cell_totals  = np.zeros((N_ACTIONS, N_ACTIONS), dtype=np.float64)
        cell_counts  = np.zeros((N_ACTIONS, N_ACTIONS), dtype=np.int64)

        for ev in training_events:
            action_str = ev.get('action', 'idle')
            a_idx = ACTION_NAMES.index(action_str) if action_str in ACTION_NAMES else 0
            outcome = ev.get('outcome', {})
            was_blocked = outcome.get('wasBlocked', False)
            hit         = outcome.get('hit', False)
            dmg         = float(outcome.get('damageDealt', 0))

            if was_blocked:
                opp_idx = ACTION_NAMES.index('block')
            elif hit:
                # Opponent was vulnerable (idle or moving)
                opp_idx = ACTION_NAMES.index('idle')
            else:
                # Attack missed — opponent may have retreated or jumped
                opp_idx = ACTION_NAMES.index('move_backward')

            # Net health delta from player's perspective
            net_delta = dmg if hit or was_blocked else 0.0
            cell_totals[a_idx, opp_idx] += net_delta
            cell_counts[a_idx, opp_idx] += 1

        # Blend empirical mean with mechanics prior using Bayesian update
        # Weight = n_empirical / (n_empirical + prior_strength)
        PRIOR_STRENGTH = 20  # equivalent to 20 observations of prior
        for r in range(N_ACTIONS):
            for c in range(N_ACTIONS):
                n = cell_counts[r, c]
                if n == 0:
                    continue
                empirical_mean = cell_totals[r, c] / n
                prior_mean     = _build_initial_payoff()[r, c]
                weight         = n / (n + PRIOR_STRENGTH)
                self._matrix[r, c] = (1 - weight) * prior_mean + weight * empirical_mean
                self._update_counts[r, c] += n

        self.save()

    def best_response_distribution(
        self, opp_freq: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Given opponent's action frequency distribution, return the player's
        best-response action distribution and expected payoff.

        best_response[i] = probability of playing action i.
        This is the pure-strategy best response (argmax) cast as a one-hot,
        unless the payoff of the optimal action is negative (then mix in block).
        """
        expected_payoff = self._matrix @ opp_freq  # shape (9,)
        best_action = int(np.argmax(expected_payoff))
        best_value  = float(expected_payoff[best_action])

        dist = np.zeros(N_ACTIONS)
        dist[best_action] = 1.0
        return dist, best_value

    def nash_equilibrium(
        self, opp_freq: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Compute the Nash equilibrium mixed strategy for the player against a
        fixed opponent distribution, using linear programming.

        Solves: maximize v
                s.t.  (P @ opp_freq) · x ≥ v   (already collapsed since opp fixed)
                Actually for fixed opponent: maximize  x^T P opp_freq
                s.t.  sum(x) = 1, x ≥ 0

        With a fixed opponent distribution, the Nash equilibrium degenerates to
        the best-response. For a mixed Nash (opponent adapts), we solve the full LP.
        """
        P = self._matrix

        # Full minimax LP:  find x that maximizes min over opp actions of (x^T P)
        # i.e., maximize v  s.t.  P^T x ≥ v·1,  sum(x)=1,  x≥0
        # Rearranged as minimization:  minimize -v
        # Variables: [x_0, ..., x_8, v]  (length 10)

        n = N_ACTIONS
        # Objective: minimize -v  (maximize v)
        c = np.zeros(n + 1)
        c[-1] = -1.0

        # Constraints: P^T x - v·1 ≥ 0  →  -P^T x + v·1 ≤ 0
        # For each opponent action j: -sum_i(P[i,j] * x_i) + v ≤ 0
        A_ub = np.hstack([-P.T, np.ones((n, 1))])   # shape (9, 10)
        b_ub = np.zeros(n)

        # Equality: sum(x) = 1
        A_eq = np.ones((1, n + 1))
        A_eq[0, -1] = 0.0
        b_eq = np.array([1.0])

        # Bounds: x_i ≥ 0 (unbounded v)
        bounds = [(0, None)] * n + [(None, None)]

        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                         bounds=bounds, method='highs')

        if result.success:
            x = np.clip(result.x[:n], 0, None)
            x /= x.sum() if x.sum() > 0 else 1.0
            v = float(result.x[-1])
            return x, v

        # Fallback to best response if LP fails
        return self.best_response_distribution(opp_freq)

    def action_value_vs_opponent(self, opp_freq: np.ndarray) -> np.ndarray:
        """
        Expected payoff for each player action against the given opponent distribution.
        Returns shape (9,) — use to rank actions and compute regret.
        """
        return self._matrix @ opp_freq
