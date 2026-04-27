"""
Markov Chain Opponent Model.

Builds a first-order Markov chain from the opponent's action sequence in the
current round. Uses it to:
  1. Predict the opponent's stationary distribution (long-run tendencies).
  2. Identify the opponent's most common action sequences (2-grams, 3-grams).
  3. Compute the best counter-strategy from the payoff matrix against those
     tendencies.

Works from a single round's data — no training history required.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

from .feature_extractor import ACTION_NAMES, ACTION_INDEX, N_ACTIONS, ATTACK_ACTIONS


class MarkovOpponentModel:
    """
    First-order Markov chain over the 9 action types.
    Updated from a single round's action log for the opponent.
    """

    def __init__(self, transition_matrix: np.ndarray, action_freq: np.ndarray) -> None:
        """
        transition_matrix : shape (9, 9), row-normalized.
        action_freq       : shape (9,), normalized frequency of each action.
        """
        self.T    = transition_matrix    # P(next=j | current=i)
        self.freq = action_freq          # marginal distribution

    @classmethod
    def from_action_log(cls, action_log: List[Dict], player_id: int) -> 'MarkovOpponentModel':
        events = [e for e in action_log if e.get('player') == player_id]
        if len(events) < 2:
            # Fallback: uniform model
            return cls(
                np.ones((N_ACTIONS, N_ACTIONS)) / N_ACTIONS,
                np.ones(N_ACTIONS) / N_ACTIONS,
            )

        freq = np.zeros(N_ACTIONS)
        trans = np.zeros((N_ACTIONS, N_ACTIONS))
        for i, e in enumerate(events):
            ai = ACTION_INDEX.get(e.get('action', 'idle'), 0)
            freq[ai] += 1
            if i < len(events) - 1:
                aj = ACTION_INDEX.get(events[i + 1].get('action', 'idle'), 0)
                trans[ai, aj] += 1

        freq /= freq.sum()

        row_sums   = trans.sum(axis=1, keepdims=True)
        uniform    = np.ones((1, N_ACTIONS)) / N_ACTIONS
        safe_sums  = np.where(row_sums > 0, row_sums, 1.0)
        trans      = np.where(row_sums > 0, trans / safe_sums, uniform)

        return cls(trans, freq)

    def stationary_distribution(self) -> np.ndarray:
        """
        Compute the stationary distribution π where π T = π (left eigenvector).
        This represents the opponent's long-run action tendencies.
        """
        # Find left eigenvector for eigenvalue 1
        eigvals, eigvecs = np.linalg.eig(self.T.T)
        # Eigenvector corresponding to eigenvalue closest to 1
        idx = int(np.argmin(np.abs(eigvals - 1.0)))
        pi  = np.real(eigvecs[:, idx])
        pi  = np.abs(pi)
        if pi.sum() > 0:
            return pi / pi.sum()
        return self.freq.copy()

    def predict_next_action(self, last_action: int) -> np.ndarray:
        """Given opponent's last action, return distribution over next action."""
        return self.T[last_action].copy()

    def top_sequences(self, n: int = 3) -> List[Tuple[str, str, float]]:
        """
        Return the top-n most probable 2-action sequences for the opponent.
        Each entry: (from_action_name, to_action_name, probability)
        """
        pairs = []
        for i in range(N_ACTIONS):
            for j in range(N_ACTIONS):
                prob = float(self.freq[i] * self.T[i, j])
                pairs.append((ACTION_NAMES[i], ACTION_NAMES[j], prob))
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs[:n]

    def dominant_pattern(self) -> Tuple[str, float]:
        """
        Return the opponent's single most-used action and its frequency.
        """
        top_idx   = int(np.argmax(self.freq))
        return ACTION_NAMES[top_idx], float(self.freq[top_idx])

    def attack_heavy(self, threshold: float = 0.45) -> bool:
        """True if opponent's attacks make up more than threshold of actions."""
        attack_sum = sum(self.freq[ACTION_INDEX[a]] for a in ATTACK_ACTIONS)
        return bool(attack_sum > threshold)  # cast numpy.bool_ → Python bool for JSON serialization

    def defence_heavy(self, threshold: float = 0.30) -> bool:
        """True if opponent's block+backward > threshold."""
        d = self.freq[ACTION_INDEX['block']] + self.freq[ACTION_INDEX['move_backward']]
        return bool(d > threshold)  # cast numpy.bool_ → Python bool for JSON serialization

    def counter_strategy_given_payoff(
        self, payoff_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Given opponent's stationary distribution, compute:
          1. Expected payoff for each player action (shape 9).
          2. Recommended mixed strategy (one-hot for best single action).

        Returns (expected_payoffs, recommended_distribution)
        """
        pi = self.stationary_distribution()
        expected = payoff_matrix @ pi            # shape (9,)
        best_idx = int(np.argmax(expected))
        dist     = np.zeros(N_ACTIONS)
        dist[best_idx] = 1.0
        return expected, dist


def top_k_counter_actions(
    expected_payoffs: np.ndarray,
    k: int = 3,
    exclude_negative: bool = True,
) -> List[Tuple[str, float]]:
    """
    Return the top-k actions by expected payoff.
    If exclude_negative=True, omits actions with negative payoff.
    """
    indexed = sorted(
        enumerate(expected_payoffs.tolist()),
        key=lambda x: x[1],
        reverse=True
    )
    result = [(ACTION_NAMES[i], v) for i, v in indexed
              if not exclude_negative or v >= 0]
    return result[:k]
