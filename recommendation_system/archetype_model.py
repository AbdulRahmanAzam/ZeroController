"""
Player Archetype Clustering.

Classifies players into one of 5 playstyle archetypes based on their action
frequency vector. Uses K-means with domain-knowledge-seeded initial centroids.

Archetypes:
  Rushdown  — High pressure: forward + punches + kicks, minimal defense
  Defensive — Block and retreat, waits for opponent to commit
  Zoner     — Keep distance: jump + backward + kicks, avoid close combat
  Balanced  — Even distribution across all actions
  Mix-Up    — High entropy: constantly varies actions to be unpredictable

Counter-archetype lookup: given (player archetype, opponent archetype),
what playstyle shift is recommended?
"""

from __future__ import annotations
import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional

from .feature_extractor import ACTION_NAMES, ACTION_INDEX, N_ACTIONS

_HERE = os.path.dirname(__file__)
KMEANS_PATH = os.path.join(_HERE, 'models', 'archetype_kmeans.json')

# ── Archetype definitions ────────────────────────────────────────────────────
# Each centroid is a 9-dim action frequency vector that characterizes the style.
# Index order: idle, forward, backward, jump, block, lpunch, rpunch, lkick, rkick

ARCHETYPE_NAMES = ['Rushdown', 'Defensive', 'Zoner', 'Balanced', 'Mix-Up']

_INITIAL_CENTROIDS = np.array([
    # Rushdown: aggressive, forward pressure, lots of attacks
    [0.03, 0.30, 0.02, 0.03, 0.04, 0.18, 0.18, 0.11, 0.11],
    # Defensive: high block and backward, minimal attacks
    [0.08, 0.03, 0.25, 0.04, 0.38, 0.06, 0.06, 0.05, 0.05],
    # Zoner: jump + backward + kicks, keeps distance
    [0.04, 0.03, 0.22, 0.22, 0.08, 0.05, 0.05, 0.16, 0.15],
    # Balanced: roughly uniform
    [0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.12, 0.11],
    # Mix-Up: high entropy — varied but active
    [0.04, 0.14, 0.14, 0.14, 0.05, 0.14, 0.05, 0.15, 0.15],
], dtype=np.float64)

# ── Counter-strategy lookup ──────────────────────────────────────────────────
# (player_archetype, opponent_archetype) → recommended_archetype + rationale

_COUNTER_TABLE: Dict[Tuple[str, str], Tuple[str, str]] = {
    ('Rushdown',  'Rushdown'):  ('Defensive',
        'When both players rush in, the one who blocks first wins. Mix in more blocks to punish their aggression.'),
    ('Rushdown',  'Defensive'): ('Zoner',
        'A defensive opponent waits for your attacks. Use jump and kicks to apply pressure from a safe distance.'),
    ('Rushdown',  'Zoner'):     ('Rushdown',
        'Close the gap immediately — a zoner loses when you are right on them. Keep pressing forward.'),
    ('Rushdown',  'Balanced'):  ('Mix-Up',
        'A balanced opponent adapts to patterns. Vary your actions more to stay unpredictable.'),
    ('Rushdown',  'Mix-Up'):    ('Defensive',
        'A mix-up opponent is unpredictable. Play defensively, block often, and punish their errors.'),

    ('Defensive', 'Rushdown'):  ('Defensive',
        'Your blocking is on the right track. Stay defensive but counter-punch or counter-kick right after blocking — do not just hold block.'),
    ('Defensive', 'Defensive'): ('Rushdown',
        'Both players are passive. Take the initiative — throw punches and move forward to deal damage.'),
    ('Defensive', 'Zoner'):     ('Rushdown',
        'A zoner wants distance. Charge forward aggressively to deny their preferred range.'),
    ('Defensive', 'Balanced'):  ('Mix-Up',
        'Vary your actions more. Pure defense gives the opponent free pressure time.'),
    ('Defensive', 'Mix-Up'):    ('Defensive',
        'Keep blocking — their mix-up is designed to confuse. Wait for predictable patterns and punish.'),

    ('Zoner',     'Rushdown'):  ('Balanced',
        'The rusher is closing your gap. Mix blocking with counter-kicks when they get close.'),
    ('Zoner',     'Defensive'): ('Rushdown',
        'A defensive opponent is safe at distance. Get close and punch — they cannot block forever.'),
    ('Zoner',     'Zoner'):     ('Rushdown',
        'Two zoners stall. Be the first to close distance and throw a combo.'),
    ('Zoner',     'Balanced'):  ('Mix-Up',
        'Add more variety to your zoning — alternating jump and kick timing surprises a balanced player.'),
    ('Zoner',     'Mix-Up'):    ('Defensive',
        'Their mix-up removes your zoning advantage. Play defensively and counter their patterns.'),

    ('Balanced',  'Rushdown'):  ('Defensive',
        'Lean more defensive against a rusher — block their onslaught then punish with kicks.'),
    ('Balanced',  'Defensive'): ('Rushdown',
        'Become more aggressive — a passive opponent cannot stop a sustained push.'),
    ('Balanced',  'Zoner'):     ('Rushdown',
        'Close the gap — a zoner needs distance. Punches at close range beat kicks at distance.'),
    ('Balanced',  'Balanced'):  ('Mix-Up',
        'Add unpredictability. A pure mix-up style beats a balanced mirror-match.'),
    ('Balanced',  'Mix-Up'):    ('Defensive',
        'Block more against a mix-up player — wait for mistakes and punish.'),

    ('Mix-Up',    'Rushdown'):  ('Defensive',
        'A rusher will overwhelm mix-up eventually. Anchor your defense and only attack after blocking.'),
    ('Mix-Up',    'Defensive'): ('Rushdown',
        'A defensive player blocks your mix-up. Pile on sustained pressure — punish their passive stance.'),
    ('Mix-Up',    'Zoner'):     ('Rushdown',
        'Close the gap immediately. A zoner cannot handle close-range mix-up pressure.'),
    ('Mix-Up',    'Balanced'):  ('Mix-Up',
        'Your mix-up is working — keep it up and increase attack frequency.'),
    ('Mix-Up',    'Mix-Up'):    ('Defensive',
        'Both players are varied. The one who blocks better wins. Prioritize defense.'),
}


class ArchetypeModel:
    """
    Classifies players into one of 5 archetypes via cosine distance to centroids.
    Optionally updates centroids via K-means from accumulated round data.
    """

    def __init__(self) -> None:
        os.makedirs(os.path.join(_HERE, 'models'), exist_ok=True)
        self._centroids = self._load_or_init()

    def _load_or_init(self) -> np.ndarray:
        if os.path.exists(KMEANS_PATH):
            try:
                with open(KMEANS_PATH) as f:
                    data = json.load(f)
                C = np.array(data['centroids'], dtype=np.float64)
                if C.shape == (len(ARCHETYPE_NAMES), N_ACTIONS):
                    return C
            except Exception:
                pass
        return _INITIAL_CENTROIDS.copy()

    def save(self) -> None:
        with open(KMEANS_PATH, 'w') as f:
            json.dump({'centroids': self._centroids.tolist(),
                       'archetype_names': ARCHETYPE_NAMES}, f)

    def assign(self, action_freq: np.ndarray) -> Tuple[str, float]:
        """
        Assign a player to the closest archetype using cosine similarity.
        Returns (archetype_name, confidence_score 0-1).
        """
        freq = action_freq / (np.linalg.norm(action_freq) + 1e-10)
        sims = np.array([
            float(np.dot(freq, c / (np.linalg.norm(c) + 1e-10)))
            for c in self._centroids
        ])
        best_idx   = int(np.argmax(sims))
        confidence = float(sims[best_idx])
        # Normalize confidence to 0-1 range (cosine similarity is already -1 to 1)
        confidence = (confidence + 1) / 2
        return ARCHETYPE_NAMES[best_idx], confidence

    def counter_strategy(
        self, player_arch: str, opp_arch: str
    ) -> Tuple[str, str]:
        """
        Look up the recommended archetype shift and its rationale.
        Returns (recommended_archetype, explanation).
        """
        return _COUNTER_TABLE.get(
            (player_arch, opp_arch),
            ('Balanced',
             'Vary your strategy and focus on blocking more to disrupt opponent patterns.')
        )

    def update_from_round_data(self, freq_vectors: List[np.ndarray]) -> None:
        """
        Update centroids from a list of action frequency vectors using K-means.
        Only called when there are enough data points (≥ 20 rounds).
        """
        if len(freq_vectors) < len(ARCHETYPE_NAMES):
            return

        X = np.stack(freq_vectors, axis=0)
        centroids = self._centroids.copy()

        # K-means: 10 iterations (sufficient for small dataset)
        for _ in range(10):
            # Assign each vector to nearest centroid
            dists = np.array([
                [np.linalg.norm(x - c) for c in centroids]
                for x in X
            ])
            labels = np.argmin(dists, axis=1)

            # Update centroids
            new_centroids = centroids.copy()
            for k in range(len(ARCHETYPE_NAMES)):
                members = X[labels == k]
                if len(members) > 0:
                    new_centroids[k] = members.mean(axis=0)

            # Check convergence
            if np.max(np.abs(new_centroids - centroids)) < 1e-6:
                break
            centroids = new_centroids

        # Blend with domain-knowledge priors (50/50) to prevent drift
        self._centroids = 0.5 * _INITIAL_CENTROIDS + 0.5 * centroids
        # Renormalize each centroid
        for k in range(len(ARCHETYPE_NAMES)):
            s = self._centroids[k].sum()
            if s > 0:
                self._centroids[k] /= s

        self.save()

    def archetype_description(self, name: str) -> str:
        desc = {
            'Rushdown':  'Aggressive, forward pressure with many attacks',
            'Defensive': 'Block-heavy, retreats and waits for opponent mistakes',
            'Zoner':     'Keeps distance using jumps and kicks',
            'Balanced':  'Even mix of all actions',
            'Mix-Up':    'Highly varied and unpredictable action choices',
        }
        return desc.get(name, 'Unknown playstyle')
