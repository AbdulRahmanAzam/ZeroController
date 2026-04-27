"""
Feature extraction from a round's ActionEvent log.

Produces per-player feature vectors used by every downstream model:
  - action_freq       : normalized 9-dim histogram of actions used
  - transition_matrix : 9×9 empirical Markov chain (row = from, col = to)
  - aggression_ratio  : fraction of events that are attacks
  - block_rate        : fraction of events that are blocks
  - accuracy          : attacks landed / attacks attempted
  - action_entropy    : Shannon entropy of action distribution (higher = more varied)
  - dominant_action   : index of most-used action
  - n_events          : total action events logged
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict, Any, Optional

# Canonical action order — must match ACTION_INDEX in the Node analytics server
ACTION_NAMES: List[str] = [
    'idle',          # 0
    'move_forward',  # 1
    'move_backward', # 2
    'jump',          # 3
    'block',         # 4
    'left_punch',    # 5
    'right_punch',   # 6
    'left_kick',     # 7
    'right_kick',    # 8
]
ACTION_INDEX: Dict[str, int] = {a: i for i, a in enumerate(ACTION_NAMES)}
N_ACTIONS = 9

ATTACK_ACTIONS = {'left_punch', 'right_punch', 'left_kick', 'right_kick'}
MOVE_ACTIONS   = {'move_forward', 'move_backward', 'jump'}


def extract_player_features(action_log: List[Dict[str, Any]], player_id: int) -> Dict[str, Any]:
    """
    Extract features for one player from the round's full action log.

    action_log: list of ActionEvent dicts (all players)
    player_id:  1 or 2
    """
    events = [e for e in action_log if e.get('player') == player_id]

    if not events:
        return _empty_features()

    n = len(events)

    # ── Action frequency vector ──────────────────────────────────────────────
    freq = np.zeros(N_ACTIONS, dtype=np.float64)
    for e in events:
        freq[ACTION_INDEX.get(e.get('action', 'idle'), 0)] += 1
    freq_norm = freq / freq.sum() if freq.sum() > 0 else np.ones(N_ACTIONS) / N_ACTIONS

    # ── Transition matrix (empirical Markov chain) ───────────────────────────
    trans = np.zeros((N_ACTIONS, N_ACTIONS), dtype=np.float64)
    for i in range(len(events) - 1):
        from_i = ACTION_INDEX.get(events[i].get('action', 'idle'), 0)
        to_i   = ACTION_INDEX.get(events[i + 1].get('action', 'idle'), 0)
        trans[from_i, to_i] += 1
    row_sums = trans.sum(axis=1, keepdims=True)
    # Rows with no transitions → uniform distribution (safe divide avoids RuntimeWarning)
    uniform_row = np.ones((1, N_ACTIONS)) / N_ACTIONS
    safe_sums = np.where(row_sums > 0, row_sums, 1.0)
    trans_norm = np.where(row_sums > 0, trans / safe_sums, uniform_row)

    # ── Derived stats ────────────────────────────────────────────────────────
    attack_events = [e for e in events if e.get('action') in ATTACK_ACTIONS]
    aggression_ratio = len(attack_events) / n
    block_rate       = sum(1 for e in events if e.get('action') == 'block') / n

    if attack_events:
        landed   = sum(1 for e in attack_events if e.get('succeeded', False))
        accuracy = landed / len(attack_events)
    else:
        accuracy = 0.0

    # Shannon entropy: higher = more varied / less predictable
    eps = 1e-10
    action_entropy = float(-np.sum(freq_norm * np.log(freq_norm + eps)))

    # Damage dealt and received (from actionLog fields)
    damage_dealt    = sum(e.get('damageDealt', 0)    for e in events)
    blocks_absorbed = sum(1 for e in events if e.get('wasBlocked', False))

    return {
        'action_freq':       freq_norm,
        'transition_matrix': trans_norm,
        'aggression_ratio':  float(aggression_ratio),
        'block_rate':        float(block_rate),
        'accuracy':          float(accuracy),
        'action_entropy':    float(action_entropy),
        'dominant_action':   int(np.argmax(freq_norm)),
        'dominant_action_name': ACTION_NAMES[int(np.argmax(freq_norm))],
        'n_events':          n,
        'damage_dealt':      float(damage_dealt),
        'blocks_absorbed':   int(blocks_absorbed),
        'attack_count':      len(attack_events),
    }


def extract_round_features(round_stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract features for both players from a full RoundStatistics dict.
    Returns a dict with keys 'p1', 'p2', and derived 'matchup' features.
    """
    action_log = round_stats.get('actionLog', [])
    p1 = extract_player_features(action_log, 1)
    p2 = extract_player_features(action_log, 2)

    # Relative features
    p1_dmg = round_stats.get('totalDamageDealt', {}).get('p1', 0)
    p2_dmg = round_stats.get('totalDamageDealt', {}).get('p2', 0)

    matchup = {
        'damage_ratio_p1': p1_dmg / max(p1_dmg + p2_dmg, 1),
        'health_remaining_p1': round_stats.get('finalHealth', {}).get('p1', 0) / 100,
        'health_remaining_p2': round_stats.get('finalHealth', {}).get('p2', 0) / 100,
        'duration_s': round_stats.get('durationMs', 0) / 1000,
        'round_number': round_stats.get('roundNumber', 1),
        'winner': round_stats.get('winner'),
    }

    return {'p1': p1, 'p2': p2, 'matchup': matchup}


def state_snapshot_to_features(snap: Dict[str, Any], acting_player: int) -> np.ndarray:
    """
    Convert a StateSnapshot dict (from ActionEvent.stateSnapshot) to a normalized
    feature vector for Q-function inference.

    Mirrors the buildFeatures() function in the Node analytics server so that
    training data features and inference features are identical.

    acting_player: 1 or 2 — the player whose Q-values we want
    Returns: float64 array of shape (15,)
    """
    if not snap:
        return np.zeros(15, dtype=np.float64)

    cw          = snap.get('canvasWidth', 1200) or 1200
    max_cd      = 200.0
    max_time    = 99.0

    feats = np.array([
        snap.get('p1_health', 100) / 100,
        snap.get('p2_health', 100) / 100,
        (snap.get('p1_health', 100) - snap.get('p2_health', 100)) / 100,
        snap.get('p1_x', 0) / cw,
        snap.get('p2_x', 0) / cw,
        abs(snap.get('p1_x', 0) - snap.get('p2_x', 0)) / cw,
        float(snap.get('p1_isAttacking', False)),
        float(snap.get('p2_isAttacking', False)),
        float(snap.get('p1_isBlocking', False)),
        float(snap.get('p2_isBlocking', False)),
        min(snap.get('p1_attackCooldown', 0) / max_cd, 1.0),
        min(snap.get('p2_attackCooldown', 0) / max_cd, 1.0),
        min(snap.get('p1_hitCooldown', 0) / max_cd, 1.0),
        min(snap.get('p2_hitCooldown', 0) / max_cd, 1.0),
        max(snap.get('roundTimeRemaining', max_time) / max_time, 0.0),
    ], dtype=np.float64)

    # Flip perspective for player 2 so the model always reasons from "my" perspective
    if acting_player == 2:
        feats[0], feats[1] = feats[1], feats[0]        # swap healths
        feats[2] = -feats[2]                             # flip health diff
        feats[3], feats[4] = feats[4], feats[3]          # swap x positions
        feats[6], feats[7] = feats[7], feats[6]          # swap attacking flags
        feats[8], feats[9] = feats[9], feats[8]          # swap blocking flags
        feats[10], feats[11] = feats[11], feats[10]      # swap attack cooldowns
        feats[12], feats[13] = feats[13], feats[12]      # swap hit cooldowns

    return feats


def _empty_features() -> Dict[str, Any]:
    return {
        'action_freq':        np.ones(N_ACTIONS) / N_ACTIONS,
        'transition_matrix':  np.ones((N_ACTIONS, N_ACTIONS)) / N_ACTIONS,
        'aggression_ratio':   0.0,
        'block_rate':         0.0,
        'accuracy':           0.0,
        'action_entropy':     float(np.log(N_ACTIONS)),
        'dominant_action':    0,
        'dominant_action_name': 'idle',
        'n_events':           0,
        'damage_dealt':       0.0,
        'blocks_absorbed':    0,
        'attack_count':       0,
    }
