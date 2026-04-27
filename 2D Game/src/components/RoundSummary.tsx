import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { RoundStatistics } from '../types/game';
import {
  getRecommendationsWithFallback,
  type Recommendation,
  type RecommendationInsights,
} from '../game/recommendationEngine';

interface RoundSummaryProps {
  roundStats: RoundStatistics;
  onNextRound: () => void;
  isLastRound: boolean;
}

const PRIORITY_COLOR: Record<string, string> = {
  high:   '#ff4757',
  medium: '#f1c40f',
  low:    '#2ecc71',
};

const ACTION_LABELS: Record<string, string> = {
  idle:          'Idle',
  move_forward:  'Forward',
  move_backward: 'Backward',
  jump:          'Jump',
  block:         'Block',
  left_punch:    'L.Punch',
  right_punch:   'R.Punch',
  left_kick:     'L.Kick',
  right_kick:    'R.Kick',
};

export const RoundSummary: React.FC<RoundSummaryProps> = ({ roundStats, onNextRound, isLastRound }) => {
  const loser = roundStats.winner === 'draw' ? null : roundStats.winner === 1 ? 2 : 1;

  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [insights, setInsights] = useState<RecommendationInsights | null>(null);
  const [loading, setLoading] = useState(false);
  const [mlActive, setMlActive] = useState(false);

  useEffect(() => {
    if (!loser) return;
    let cancelled = false;
    setLoading(true);

    getRecommendationsWithFallback(roundStats, loser).then(result => {
      if (cancelled) return;
      setRecommendations(result.recommendations);
      setInsights(result.insights);
      setMlActive(result.insights !== null);
      setLoading(false);
    });

    return () => { cancelled = true; };
  }, [roundStats, loser]);

  // Allow Enter/Space to advance
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Enter' || e.key === ' ') onNextRound();
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onNextRound]);

  const p1Accuracy = roundStats.attacksAttempted.p1 > 0
    ? Math.round((roundStats.attacksLanded.p1 / roundStats.attacksAttempted.p1) * 100)
    : 0;
  const p2Accuracy = roundStats.attacksAttempted.p2 > 0
    ? Math.round((roundStats.attacksLanded.p2 / roundStats.attacksAttempted.p2) * 100)
    : 0;

  const statRow = (label: string, v1: string | number, v2: string | number) => (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
      <span style={{ flex: 1, textAlign: 'right', color: '#00d4ff', fontSize: 13, fontFamily: 'Orbitron, sans-serif', fontWeight: 700 }}>{v1}</span>
      <span style={{ width: 130, textAlign: 'center', color: '#aaa', fontSize: 11, letterSpacing: 1, textTransform: 'uppercase' }}>{label}</span>
      <span style={{ flex: 1, textAlign: 'left', color: '#ff4757', fontSize: 13, fontFamily: 'Orbitron, sans-serif', fontWeight: 700 }}>{v2}</span>
    </div>
  );

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        style={{
          position: 'absolute',
          inset: 0,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 300,
          background: 'radial-gradient(ellipse at center, rgba(10,10,30,0.97) 0%, rgba(0,0,0,0.98) 100%)',
          overflowY: 'auto',
        }}
      >
        <motion.div
          initial={{ scale: 0.85, y: 40, opacity: 0 }}
          animate={{ scale: 1, y: 0, opacity: 1 }}
          transition={{ type: 'spring', damping: 18, stiffness: 220, delay: 0.1 }}
          style={{
            width: '92%',
            maxWidth: 720,
            background: 'linear-gradient(180deg, #0d0d1f 0%, #11112a 100%)',
            border: '2px solid rgba(155,89,182,0.5)',
            borderRadius: 16,
            boxShadow: '0 0 60px rgba(155,89,182,0.3), inset 0 0 40px rgba(0,0,0,0.6)',
            padding: '28px 32px',
            display: 'flex',
            flexDirection: 'column',
            gap: 18,
            margin: '24px 0',
          }}
        >
          {/* ── Header ── */}
          <div style={{ textAlign: 'center' }}>
            <motion.div
              initial={{ scale: 1.4, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ delay: 0.2 }}
              style={{ fontSize: 11, letterSpacing: 5, color: '#9b59b6', fontFamily: 'Orbitron, sans-serif', marginBottom: 4, textTransform: 'uppercase' }}
            >
              Round {roundStats.roundNumber} Complete
            </motion.div>
            <motion.div
              initial={{ y: -20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 0.25 }}
              style={{
                fontSize: 36, fontWeight: 900, fontFamily: 'Bebas Neue, Orbitron, sans-serif',
                color: roundStats.winner === 'draw' ? '#f1c40f' : roundStats.winner === 1 ? '#00d4ff' : '#ff4757',
                textShadow: '0 0 20px currentColor', letterSpacing: 6,
              }}
            >
              {roundStats.winner === 'draw' ? 'DRAW' : `PLAYER ${roundStats.winner} WINS`}
            </motion.div>
          </div>

          {/* ── Stats grid ── */}
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.35 }}
            style={{ background: 'rgba(255,255,255,0.03)', borderRadius: 10, border: '1px solid rgba(255,255,255,0.08)', padding: '14px 20px' }}
          >
            <div style={{ display: 'flex', marginBottom: 10 }}>
              <span style={{ flex: 1, textAlign: 'right', color: '#00d4ff', fontSize: 12, fontFamily: 'Orbitron, sans-serif', fontWeight: 900, letterSpacing: 2 }}>P1</span>
              <span style={{ width: 130 }} />
              <span style={{ flex: 1, textAlign: 'left', color: '#ff4757', fontSize: 12, fontFamily: 'Orbitron, sans-serif', fontWeight: 900, letterSpacing: 2 }}>P2</span>
            </div>
            {statRow('Health left',     `${Math.round(roundStats.finalHealth.p1)}`,   `${Math.round(roundStats.finalHealth.p2)}`)}
            {statRow('Damage dealt',    roundStats.totalDamageDealt.p1,               roundStats.totalDamageDealt.p2)}
            {statRow('Accuracy',        `${p1Accuracy}%`,                              `${p2Accuracy}%`)}
            {statRow('Attacks landed',  `${roundStats.attacksLanded.p1}/${roundStats.attacksAttempted.p1}`, `${roundStats.attacksLanded.p2}/${roundStats.attacksAttempted.p2}`)}
            {statRow('Best combo',      roundStats.highestCombo.p1,                   roundStats.highestCombo.p2)}
            {statRow('Blocks',          roundStats.blocksPerformed.p1,                roundStats.blocksPerformed.p2)}
          </motion.div>

          {/* ── ML Insights panel ── */}
          {insights && loser && (
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.45 }}
              style={{
                background: 'rgba(0,100,255,0.06)',
                border: '1px solid rgba(0,150,255,0.25)',
                borderRadius: 10,
                padding: '12px 18px',
              }}
            >
              <div style={{
                fontSize: 9, letterSpacing: 4, color: '#3498db',
                fontFamily: 'Orbitron, sans-serif', marginBottom: 10,
                textTransform: 'uppercase', fontWeight: 900,
              }}>
                🤖 AI Analysis
              </div>
              <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
                <InsightChip label="Your Style" value={insights.player_archetype} color="#9b59b6" />
                <InsightChip label="Opponent Style" value={insights.opponent_archetype} color="#e67e22" />
                <InsightChip label="Recommended" value={insights.counter_strategy} color="#27ae60" />
                {insights.missed_action && (
                  <InsightChip label="Key Action" value={insights.missed_action.replace('_', ' ')} color="#e74c3c" />
                )}
              </div>

              {/* Nash strategy bar */}
              {Object.keys(insights.nash_strategy).length > 0 && (
                <div style={{ marginTop: 10 }}>
                  <div style={{ fontSize: 9, color: '#888', letterSpacing: 2, textTransform: 'uppercase', marginBottom: 6 }}>
                    Optimal Action Mix vs This Opponent
                  </div>
                  <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                    {Object.entries(insights.nash_strategy)
                      .sort((a, b) => b[1] - a[1])
                      .map(([action, pct]) => (
                        <div key={action} style={{ textAlign: 'center' }}>
                          <div style={{
                            width: Math.max(pct * 1.2, 20),
                            height: 4,
                            background: `hsl(${Object.keys(insights.nash_strategy).indexOf(action) * 40}, 70%, 55%)`,
                            borderRadius: 2,
                            marginBottom: 2,
                          }} />
                          <div style={{ fontSize: 8, color: '#888', whiteSpace: 'nowrap' }}>
                            {ACTION_LABELS[action] ?? action} {pct}%
                          </div>
                        </div>
                      ))}
                  </div>
                </div>
              )}

              {/* LLM summary */}
              {insights.llm_summary && (
                <div style={{ marginTop: 8, fontSize: 11, color: '#aaa', fontStyle: 'italic', lineHeight: 1.5 }}>
                  "{insights.llm_summary}"
                </div>
              )}
            </motion.div>
          )}

          {/* ── Recommendation panel ── */}
          {loser && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
              style={{
                background: 'rgba(231,76,60,0.07)',
                border: '1px solid rgba(231,76,60,0.3)',
                borderRadius: 10,
                padding: '14px 20px',
                minHeight: 80,
              }}
            >
              <div style={{
                fontSize: 10, letterSpacing: 4, color: '#e74c3c',
                fontFamily: 'Orbitron, sans-serif', marginBottom: 12,
                textTransform: 'uppercase', fontWeight: 900,
                display: 'flex', alignItems: 'center', gap: 8,
              }}>
                💡 Advice for Player {loser}
                {mlActive && (
                  <span style={{ fontSize: 8, background: 'rgba(46,204,113,0.2)', color: '#2ecc71', padding: '2px 6px', borderRadius: 4, letterSpacing: 1 }}>
                    ML
                  </span>
                )}
              </div>

              {loading ? (
                <div style={{ color: '#555', fontSize: 11, fontFamily: 'monospace' }}>
                  Analysing round data...
                </div>
              ) : recommendations.length === 0 ? (
                <div style={{ color: '#555', fontSize: 11 }}>No recommendations available.</div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  {recommendations.map((rec, i) => (
                    <motion.div
                      key={i}
                      initial={{ opacity: 0, x: -12 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: 0.55 + i * 0.1 }}
                      style={{ display: 'flex', gap: 10, alignItems: 'flex-start' }}
                    >
                      <span style={{
                        width: 8, height: 8, borderRadius: '50%',
                        background: PRIORITY_COLOR[rec.priority], flexShrink: 0, marginTop: 5,
                        boxShadow: `0 0 6px ${PRIORITY_COLOR[rec.priority]}`,
                      }} />
                      <div>
                        <div style={{ color: '#fff', fontSize: 12, fontWeight: 700, marginBottom: 2, fontFamily: 'Orbitron, sans-serif' }}>
                          {rec.title}
                        </div>
                        <div style={{ color: '#aaa', fontSize: 11, lineHeight: 1.5 }}>
                          {rec.detail}
                        </div>
                      </div>
                    </motion.div>
                  ))}
                </div>
              )}
            </motion.div>
          )}

          {/* ── Next round button ── */}
          <motion.button
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.7 }}
            whileHover={{ scale: 1.03, boxShadow: '0 0 30px rgba(155,89,182,0.6)' }}
            whileTap={{ scale: 0.97 }}
            onClick={onNextRound}
            style={{
              alignSelf: 'center', padding: '14px 48px', fontSize: 16, fontWeight: 900,
              fontFamily: 'Bebas Neue, Orbitron, sans-serif', letterSpacing: 4, color: '#fff',
              background: 'linear-gradient(180deg, #8e44ad 0%, #6c3483 100%)',
              border: '2px solid rgba(255,255,255,0.2)', borderRadius: 8, cursor: 'pointer',
              textTransform: 'uppercase', boxShadow: '0 0 20px rgba(155,89,182,0.4)',
            }}
          >
            {isLastRound ? 'See Results' : 'Next Round →'}
          </motion.button>

          <div style={{ textAlign: 'center', color: '#444', fontSize: 11, fontFamily: 'monospace' }}>
            Press Enter or Space to continue
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
};

// ── Small helper component ───────────────────────────────────────────────────

const InsightChip: React.FC<{ label: string; value: string; color: string }> = ({ label, value, color }) => (
  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', minWidth: 70 }}>
    <div style={{ fontSize: 8, color: '#666', letterSpacing: 1, textTransform: 'uppercase', marginBottom: 3 }}>
      {label}
    </div>
    <div style={{
      fontSize: 10, fontWeight: 700, color, fontFamily: 'Orbitron, sans-serif',
      background: `${color}22`, padding: '3px 8px', borderRadius: 4,
      border: `1px solid ${color}44`, textAlign: 'center',
    }}>
      {value}
    </div>
  </div>
);
