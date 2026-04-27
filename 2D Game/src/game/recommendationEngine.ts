import type { RoundStatistics } from '../types/game';

export interface Recommendation {
  title: string;
  detail: string;
  priority: 'high' | 'medium' | 'low';
}

export interface RecommendationInsights {
  player_archetype: string;
  opponent_archetype: string;
  counter_strategy: string;
  nash_strategy: Record<string, number>;
  dominant_opp_action: string;
  opp_attack_heavy: boolean;
  opp_defence_heavy: boolean;
  missed_action: string | null;
  top_regret_moment_ms: number | null;
  q_active: boolean;
  llm_summary: string | null;
}

export interface RecommendationResult {
  recommendations: Recommendation[];
  insights: RecommendationInsights | null;
}

const ML_SERVER = 'http://localhost:8001';
const TIMEOUT_MS = 1500;

/**
 * Fetch ML-powered recommendations from the Python recommendation server.
 * Falls back to hardcoded rules if the server is unavailable.
 */
export async function getRecommendationsWithFallback(
  stats: RoundStatistics,
  loser: 1 | 2,
): Promise<RecommendationResult> {
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);

    const response = await fetch(`${ML_SERVER}/recommend`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ round_stats: stats, loser, generate_llm: false }),
      signal: controller.signal,
    });
    clearTimeout(timer);

    if (response.ok) {
      const data = await response.json();
      return {
        recommendations: data.recommendations ?? [],
        insights: data.insights ?? null,
      };
    }
  } catch {
    // Server unavailable or timeout — use fallback silently
  }

  return { recommendations: getFallbackRecommendations(stats, loser), insights: null };
}

/**
 * Hardcoded rule-based recommendations — used as fallback when ML server is
 * unavailable. Preserved exactly as they were before.
 */
export function getFallbackRecommendations(
  stats: RoundStatistics,
  loser: 1 | 2
): Recommendation[] {
  const p   = loser === 1 ? 'p1' : 'p2';
  const opp = loser === 1 ? 'p2' : 'p1';

  const attempted = stats.attacksAttempted[p];
  const landed    = stats.attacksLanded[p];
  const accuracy  = attempted > 0 ? landed / attempted : 0;
  const blocks    = stats.blocksPerformed[p];
  const dmgTaken  = stats.totalDamageDealt[opp];
  const dmgDealt  = stats.totalDamageDealt[p];
  const combo     = stats.highestCombo[p];

  const tips: Recommendation[] = [];

  if (attempted >= 3 && accuracy < 0.35) {
    tips.push({
      title:    'Improve your timing',
      detail:   `You landed only ${landed} of ${attempted} attacks (${Math.round(accuracy * 100)}% accuracy). Wait for your opponent to commit before striking.`,
      priority: 'high',
    });
  }

  if (dmgTaken > 40 && blocks < 3) {
    tips.push({
      title:    'Defend more',
      detail:   `You took ${dmgTaken} damage with only ${blocks} block(s). Hold the block button when your opponent attacks to reduce incoming damage by 80%.`,
      priority: 'high',
    });
  }

  if (dmgDealt < 20) {
    tips.push({
      title:    'Be more aggressive',
      detail:   `You only dealt ${dmgDealt} damage. Try mixing punches and kicks, and use your special move when your meter is full.`,
      priority: 'medium',
    });
  }

  if (combo < 2 && attempted >= 4) {
    tips.push({
      title:    'Chain your attacks',
      detail:   'Your highest combo was just 1 hit. Chain a punch into a kick for combo bonus damage.',
      priority: 'medium',
    });
  }

  if (blocks === 0 && dmgTaken > 20) {
    tips.push({
      title:    'Use your block',
      detail:   'You never blocked! Holding the block key reduces damage by 80%.',
      priority: 'high',
    });
  }

  if (attempted < 3) {
    tips.push({
      title:    'Attack more often',
      detail:   'You barely attacked this round. Punches are fast and safe — try pressing attack keys more frequently.',
      priority: 'low',
    });
  }

  const order: Record<Recommendation['priority'], number> = { high: 0, medium: 1, low: 2 };
  return tips.sort((a, b) => order[a.priority] - order[b.priority]).slice(0, 3);
}

/** @deprecated Use getRecommendationsWithFallback instead. Kept for reference. */
export const getRecommendations = getFallbackRecommendations;
