"""
LLM one-line explainer (optional, Claude Haiku).

Converts the structured model output into a single natural-language coaching
sentence. Falls back gracefully if the API key is missing or the call fails.
The recommendation engine always works without this module.
"""

from __future__ import annotations
import os
import logging
from typing import Optional, Dict, Any

log = logging.getLogger(__name__)

_API_KEY = os.environ.get('ANTHROPIC_API_KEY', '')
_CLIENT  = None


def _get_client():
    global _CLIENT
    if _CLIENT is None and _API_KEY:
        try:
            import anthropic
            _CLIENT = anthropic.Anthropic(api_key=_API_KEY)
        except ImportError:
            log.debug('anthropic SDK not installed; LLM explainer disabled')
    return _CLIENT


def generate_one_liner(
    insights: Dict[str, Any],
    loser: int,
    round_stats: Dict[str, Any],
) -> Optional[str]:
    """
    Generate a single coaching sentence from structured model insights.

    insights keys (all optional):
      player_archetype, opponent_archetype, nash_strategy,
      counter_strategy_name, top_regret_action, missed_action,
      dominant_opp_action, opp_attack_heavy, opp_defence_heavy

    Returns None if the API is unavailable or the call fails.
    """
    client = _get_client()
    if client is None:
        return None

    p = 'p1' if loser == 1 else 'p2'
    opp = 'p2' if loser == 1 else 'p1'

    damage_dealt  = round_stats.get('totalDamageDealt', {}).get(p, 0)
    damage_taken  = round_stats.get('totalDamageDealt', {}).get(opp, 0)
    blocks_used   = round_stats.get('blocksPerformed', {}).get(p, 0)
    accuracy_pct  = 0
    att = round_stats.get('attacksAttempted', {}).get(p, 0)
    lnd = round_stats.get('attacksLanded',    {}).get(p, 0)
    if att > 0:
        accuracy_pct = round(lnd / att * 100)

    prompt = (
        f"You are a concise fighting game coach. Give Player {loser} one specific, "
        f"actionable tip in one sentence (max 20 words) based on this round data:\n"
        f"- Player archetype: {insights.get('player_archetype', 'Unknown')}\n"
        f"- Opponent archetype: {insights.get('opponent_archetype', 'Unknown')}\n"
        f"- Recommended switch to: {insights.get('counter_strategy_name', 'Balanced')}\n"
        f"- Damage dealt: {damage_dealt}, taken: {damage_taken}\n"
        f"- Blocks used: {blocks_used}, attack accuracy: {accuracy_pct}%\n"
        f"- Most impactful missed action: {insights.get('missed_action', 'block')}\n"
        f"- Opponent dominant action: {insights.get('dominant_opp_action', 'idle')}\n"
        "Reply with the coaching sentence ONLY. No labels, no preamble."
    )

    try:
        resp = client.messages.create(
            model='claude-haiku-4-5-20251001',
            max_tokens=60,
            messages=[{'role': 'user', 'content': prompt}],
        )
        text = resp.content[0].text.strip()
        # Hard cap to keep it one sentence
        if '.' in text:
            text = text[:text.index('.') + 1]
        return text if len(text) > 10 else None
    except Exception as e:
        log.debug('LLM explainer failed: %s', e)
        return None
