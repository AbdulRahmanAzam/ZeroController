"""
Recommendation Engine — aggregates all 4 ML models into ranked coaching tips.

Pipeline for each round:
  1. Extract features for both players from the action log.
  2. Model A: Payoff Matrix + Nash Equilibrium
     → Optimal strategy distribution vs this opponent.
     → Compare to loser's actual distribution → identify underused high-value actions.
  3. Model B: Markov Opponent Model
     → Opponent's dominant sequences.
     → Best counter action given those patterns.
  4. Model C: Archetype Clustering
     → Classify both players → counter-archetype recommendation.
  5. Model D: Q-Function Regret Analysis (only when model trained)
     → Per-timestep regret → most impactful missed action.
  6. Aggregate candidates → rank by confidence × priority → top 3.
  7. (Optional) LLM one-liner.

Each candidate recommendation has:
  { title, detail, priority, confidence (0-1), source }
The source is stripped before returning to the frontend (internal only).
"""

from __future__ import annotations
import math
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from .feature_extractor import (
    extract_round_features,
    ACTION_NAMES, N_ACTIONS, ATTACK_ACTIONS
)
from .payoff_matrix  import PayoffMatrix
from .markov_model   import MarkovOpponentModel, top_k_counter_actions
from .archetype_model import ArchetypeModel
from .q_function     import QFunction
from . import llm_explainer

# Priority ordering for frontend display
PRIORITY_ORDER = {'high': 0, 'medium': 1, 'low': 2}

_ACTION_DISPLAY = {
    'idle':          'resting (idle)',
    'move_forward':  'moving forward',
    'move_backward': 'retreating',
    'jump':          'jumping',
    'block':         'blocking',
    'left_punch':    'left punch',
    'right_punch':   'right punch',
    'left_kick':     'left kick',
    'right_kick':    'right kick',
}


def _disp(action: str) -> str:
    return _ACTION_DISPLAY.get(action, action.replace('_', ' '))


def _pct(v: float) -> str:
    return f'{round(v * 100)}%'


class RecommendationEngine:
    def __init__(
        self,
        payoff_matrix: PayoffMatrix,
        archetype_model: ArchetypeModel,
        q_function: QFunction,
    ) -> None:
        self._payoff    = payoff_matrix
        self._archetype = archetype_model
        self._q         = q_function

    def recommend(
        self,
        round_stats: Dict[str, Any],
        loser: int,
        generate_llm: bool = False,
    ) -> Dict[str, Any]:
        """
        Main entry point.

        round_stats: RoundStatistics dict from the frontend (camelCase keys mapped to snake).
        loser:       1 or 2
        Returns:
          {
            recommendations: [{ title, detail, priority }],  ← top 3
            insights: { ... }                                 ← extra panel data
          }
        """
        p_key   = 'p1' if loser == 1 else 'p2'
        opp_key = 'p2' if loser == 1 else 'p1'
        opp_id  = 2 if loser == 1 else 1

        features = extract_round_features(round_stats)
        p_feat   = features[p_key]
        opp_feat = features[opp_key]

        action_log = round_stats.get('actionLog', [])

        candidates: List[Dict[str, Any]] = []

        # ── Model B: Markov Opponent Model ───────────────────────────────────
        opp_markov = MarkovOpponentModel.from_action_log(action_log, opp_id)
        opp_stationary = opp_markov.stationary_distribution()
        expected_payoffs, _ = opp_markov.counter_strategy_given_payoff(self._payoff.matrix)
        top_counters = top_k_counter_actions(expected_payoffs, k=3)

        # What did the opponent actually do most?
        dom_opp_action, dom_opp_freq = opp_markov.dominant_pattern()

        # Compare player's actual distribution vs the best-response
        p_freq   = p_feat['action_freq']
        best_act = ACTION_NAMES[int(np.argmax(expected_payoffs))]
        best_val = float(expected_payoffs[int(np.argmax(expected_payoffs))])
        p_best_usage = float(p_freq[ACTION_NAMES.index(best_act)])

        if best_val > 0 and p_best_usage < 0.20:
            candidates.append({
                'title':      f'Counter their {_disp(dom_opp_action)}',
                'detail':     (
                    f'Opponent used {_disp(dom_opp_action)} in {_pct(dom_opp_freq)} of their actions. '
                    f'Your best counter is {_disp(best_act)} — but you only used it {_pct(p_best_usage)} of the time. '
                    f'Increase {_disp(best_act)} when you see them start their pattern.'
                ),
                'priority':   'high',
                'confidence': min(0.95, 0.55 + dom_opp_freq * 0.6),
                'source':     'markov',
            })

        # Opponent is attack-heavy but player barely blocked
        p_block_rate = float(p_feat['block_rate'])
        opp_is_attack_heavy = opp_markov.attack_heavy(threshold=0.40)
        dmg_taken = float(round_stats.get('totalDamageDealt', {}).get(opp_key, 0))
        if opp_is_attack_heavy and p_block_rate < 0.12 and dmg_taken > 25:
            opp_attack_freq = sum(
                opp_stationary[ACTION_NAMES.index(a)] for a in ATTACK_ACTIONS
            )
            candidates.append({
                'title':      'Block their attack patterns',
                'detail':     (
                    f'Opponent spent {_pct(opp_attack_freq)} of their time attacking. '
                    f'You only blocked {_pct(p_block_rate)} of your actions and took {dmg_taken:.0f} damage. '
                    f'Blocking reduces damage by 80% — use it when they close the gap.'
                ),
                'priority':   'high',
                'confidence': 0.88,
                'source':     'markov_defense',
            })

        # ── Model A: Payoff Matrix — best-response & Nash ────────────────────
        # best-response: what action is most valuable against THIS opponent's observed pattern
        action_values = self._payoff.action_value_vs_opponent(opp_stationary)

        # Build best-response distribution (normalised top-3 positive actions)
        av_sorted    = sorted(enumerate(action_values.tolist()), key=lambda x: x[1], reverse=True)
        pos_actions  = [(i, v) for i, v in av_sorted if v > 0]
        if pos_actions:
            total = sum(v for _, v in pos_actions[:3])
            best_response_dist = np.zeros(N_ACTIONS)
            for i, v in pos_actions[:3]:
                best_response_dist[i] = v / total if total > 0 else 1 / 3
        else:
            best_response_dist = np.ones(N_ACTIONS) / N_ACTIONS

        # Nash (minimax) — used only for insight panel display
        nash_dist, nash_val = self._payoff.nash_equilibrium(opp_stationary)

        # Which actions does the player underuse vs best-response?
        br_diff = best_response_dist - p_freq  # positive = underused

        underused_value_pairs = [
            (ACTION_NAMES[i], float(br_diff[i]), float(action_values[i]))
            for i in range(N_ACTIONS)
            if br_diff[i] > 0.10 and action_values[i] > 1.0
        ]
        underused_value_pairs.sort(key=lambda x: x[2], reverse=True)

        if underused_value_pairs:
            best_miss, miss_gap, miss_val = underused_value_pairs[0]
            br_pct = _pct(float(best_response_dist[ACTION_NAMES.index(best_miss)]))
            p_pct  = _pct(float(p_freq[ACTION_NAMES.index(best_miss)]))
            candidates.append({
                'title':      f'Use {_disp(best_miss)} more often',
                'detail':     (
                    f'Against this opponent\'s pattern, {_disp(best_miss)} yields '
                    f'+{miss_val:.1f} expected HP per exchange. '
                    f'Optimal usage is around {br_pct} of your actions, '
                    f'but you only used it {p_pct} of the time. '
                    f'Increase {_disp(best_miss)} especially when the opponent is moving in.'
                ),
                'priority':   'medium' if miss_val < 6 else 'high',
                'confidence': min(0.90, 0.60 + miss_gap * 0.5),
                'source':     'nash',
            })

        # Dominant action has very negative expected value → player keeps doing the worst thing
        dom_p_action_idx = int(p_feat['dominant_action'])
        dom_p_action     = ACTION_NAMES[dom_p_action_idx]
        dom_p_val        = float(action_values[dom_p_action_idx])
        dom_p_freq_val   = float(p_freq[dom_p_action_idx])

        if dom_p_val < -2.0 and dom_p_freq_val > 0.20:
            best_alt = top_counters[0][0] if top_counters else 'block'
            candidates.append({
                'title':      f'Stop relying on {_disp(dom_p_action)}',
                'detail':     (
                    f'You used {_disp(dom_p_action)} {_pct(dom_p_freq_val)} of the time, '
                    f'but against this opponent it yields {dom_p_val:.1f} HP per exchange. '
                    f'Switch to {_disp(best_alt)} which has a positive expected outcome.'
                ),
                'priority':   'high',
                'confidence': 0.85,
                'source':     'payoff_dominant',
            })

        # ── Model C: Archetype Clustering ────────────────────────────────────
        p_arch,   p_conf   = self._archetype.assign(p_freq)
        opp_arch, opp_conf = self._archetype.assign(opp_feat['action_freq'])
        rec_arch, rationale = self._archetype.counter_strategy(p_arch, opp_arch)
        arch_confidence     = (p_conf + opp_conf) / 2

        if p_arch != rec_arch:
            candidates.append({
                'title':      f'Switch to {rec_arch} style',
                'detail':     (
                    f'You played as {p_arch} ({self._archetype.archetype_description(p_arch)}) '
                    f'against a {opp_arch} opponent. {rationale}'
                ),
                'priority':   'medium',
                'confidence': float(arch_confidence),
                'source':     'archetype',
            })

        # ── Model D: Q-Function Regret ───────────────────────────────────────
        regret_info = self._q.regret_score(action_log, loser)
        if regret_info is not None:
            missed = regret_info['missed_action']
            missed_disp = _disp(missed)
            avg_reg = regret_info['avg_regret']
            top_ts  = regret_info['top_regret_moment_ms']
            top_reg = regret_info['top_regret_value']

            if avg_reg > 1.0:
                ts_sec = f'{top_ts / 1000:.1f}s' if top_ts else 'multiple moments'
                candidates.append({
                    'title':      f'Key moments: use {missed_disp}',
                    'detail':     (
                        f'Across your action choices, the average missed opportunity was {avg_reg:.1f} HP. '
                        f'Your biggest single mistake ({top_reg:.1f} HP lost opportunity) was at {ts_sec}. '
                        f'{missed_disp.capitalize()} would have been {top_reg / max(avg_reg, 0.1):.1f}× more effective at that moment.'
                    ),
                    'priority':   'high' if top_reg > 8 else 'medium',
                    'confidence': 0.80,
                    'source':     'q_regret',
                })

        # ── Accuracy / timing tip ────────────────────────────────────────────
        att = round_stats.get('attacksAttempted', {}).get(p_key, 0)
        lnd = round_stats.get('attacksLanded',    {}).get(p_key, 0)
        accuracy = lnd / att if att > 0 else 0
        if att >= 4 and accuracy < 0.35:
            candidates.append({
                'title':      'Improve attack timing',
                'detail':     (
                    f'You landed only {lnd} of {att} attacks ({_pct(accuracy)} accuracy). '
                    f'Wait until you are at close range before throwing punches. '
                    f'Kicks have longer range but slower startup — punches are safer to throw first.'
                ),
                'priority':   'medium',
                'confidence': 0.70,
                'source':     'accuracy',
            })

        # ── Combo tip ────────────────────────────────────────────────────────
        combo = round_stats.get('highestCombo', {}).get(p_key, 0)
        if combo < 2 and att >= 5 and accuracy > 0.35:
            candidates.append({
                'title':      'Chain your attacks into combos',
                'detail':     (
                    f'Your best combo was just {combo} hit(s). '
                    f'Chain a punch immediately into a kick while the opponent is still in hitstun (within 200ms). '
                    f'Combos deal bonus damage without giving the opponent time to recover.'
                ),
                'priority':   'low',
                'confidence': 0.60,
                'source':     'combo',
            })

        # ── Passivity tip ────────────────────────────────────────────────────
        if att < 3:
            candidates.append({
                'title':      'Be more aggressive',
                'detail':     (
                    f'You only attempted {att} attack(s) this round. '
                    f'Punches have a fast startup (35ms) — use them to apply pressure. '
                    f'An opponent that is never attacked can move freely and set up their own combos.'
                ),
                'priority':   'low',
                'confidence': 0.65,
                'source':     'passivity',
            })

        # ── Rank and deduplicate ─────────────────────────────────────────────
        # Sort by confidence × priority_weight
        priority_weight = {'high': 3.0, 'medium': 2.0, 'low': 1.0}
        candidates.sort(
            key=lambda c: c['confidence'] * priority_weight[c['priority']],
            reverse=True
        )

        # Deduplicate by source category (keep highest-ranked per source family)
        seen_sources: set = set()
        deduped: List[Dict] = []
        for c in candidates:
            src_family = c['source'].split('_')[0]   # 'markov', 'nash', 'archetype', etc.
            if src_family not in seen_sources:
                seen_sources.add(src_family)
                deduped.append(c)
            if len(deduped) >= 3:
                break

        # Clean output for frontend (remove internal fields)
        recs = [
            {'title': c['title'], 'detail': c['detail'], 'priority': c['priority']}
            for c in deduped[:3]
        ]

        # ── Build insights dict (for the insights panel) ─────────────────────
        # Show best-response distribution (what to use vs THIS opponent)
        # rather than Nash minimax (which is opponent-agnostic)
        nash_strategy_display = {
            ACTION_NAMES[i]: round(float(best_response_dist[i]) * 100)
            for i in range(N_ACTIONS)
            if best_response_dist[i] > 0.05
        }

        insights: Dict[str, Any] = {
            'player_archetype':    p_arch,
            'opponent_archetype':  opp_arch,
            'counter_strategy':    rec_arch,
            'nash_strategy':       nash_strategy_display,
            'dominant_opp_action': dom_opp_action,
            'opp_attack_heavy':    opp_markov.attack_heavy(),
            'opp_defence_heavy':   opp_markov.defence_heavy(),
            'missed_action':       regret_info['missed_action'] if regret_info else None,
            'top_regret_moment_ms': regret_info['top_regret_moment_ms'] if regret_info else None,
            'q_active':            self._q.is_active(),
            'llm_summary':         None,
        }

        # ── Optional LLM one-liner ───────────────────────────────────────────
        if generate_llm:
            insights['llm_summary'] = llm_explainer.generate_one_liner(
                insights, loser, round_stats
            )

        return {'recommendations': recs, 'insights': insights}
