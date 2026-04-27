# Recommendation System Architecture for 2D Fighting Game

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Problem Statement](#problem-statement)
3. [Research Literature Review](#research-literature-review)
4. [Architecture Overview](#architecture-overview)
5. [Model A: Payoff Matrix & Nash Equilibrium](#model-a-payoff-matrix--nash-equilibrium)
6. [Model B: Markov Chain Opponent Modeling](#model-b-markov-chain-opponent-modeling)
7. [Model C: Archetype Clustering](#model-c-archetype-clustering)
8. [Model D: Q-Function via Fitted Q Iteration](#model-d-q-function-via-fitted-q-iteration)
9. [Dataset and Feature Engineering](#dataset-and-feature-engineering)
10. [Cold Start Strategy](#cold-start-strategy)
11. [Progressive Learning Milestones](#progressive-learning-milestones)
12. [API Design](#api-design)
13. [Frontend Integration](#frontend-integration)
14. [Training Recipes](#training-recipes)
15. [Why NOT Pure LLM / Pure Deep RL](#why-not-pure-llm--pure-deep-rl)
16. [Implementation Roadmap](#implementation-roadmap)

---

## Executive Summary

This document describes a **hybrid multi-model recommendation system** for post-round strategy coaching in a 2D fighting game. The system is NOT real-time game AI; it is a **human-facing analytics and coaching tool** that generates actionable strategy recommendations after each round concludes.

### Key Features

- **Four specialized models** in a weighted ensemble:
  - **Payoff Matrix** (game-theoretic): Nash equilibrium solver + damage mechanics
  - **Markov Chain** (opponent modeling): transition probabilities from action history
  - **Archetype Clustering** (player profiling): 5 behavioral archetypes with counter-strategies
  - **Q-Function (FQI)** (offline RL): fitted Q iteration on accumulated game data

- **Progressive learning**: Cold-start from seeded domain knowledge → full hybrid system at 50+ rounds
- **Confidence-weighted ranking**: Each recommendation scored by model confidence × strategic priority
- **MongoDB analytics backend**: Persistent training data pipeline (3001, Node.js)
- **FastAPI server**: Model serving and recommendation generation (8001, Python)
- **9 actions**: idle, move_forward, move_backward, jump, block, left_punch, right_punch, left_kick, right_kick

### Design Philosophy

1. **Interpretability over pure accuracy**: Players need to *understand* why we suggest action X
2. **Graceful degradation**: Works with zero data (cold start) → improves continuously
3. **Game-theory grounded**: Payoff matrix + Nash equilibrium provide principled foundation
4. **Empirical validation**: Real gameplay data updates models incrementally
5. **Modular**: Each model independent; ensemble aggregation via confidence scoring

---

## Problem Statement

### Context

A 2D fighting game player loses a round. The current system has just collected:
- Both players' action sequences (with timestamps)
- Health trajectories throughout the round
- Cooldown information
- Position data (if available)

The question is: **What should the losing player practice or change for the next round?**

### Not Real-Time AI

This recommendation system is **post-round coaching**, not:
- Real-time next-action prediction during gameplay
- Live optimal play computation (too slow)
- Game-tree search or MCTS (impractical in real-time)

### Requirements

1. **Cold-start capability**: Recommendations from round 1 (no training data yet)
2. **Interpretability**: Each recommendation includes title, detail, and priority rationale
3. **Opponent modeling**: Recognize opponent patterns and counter them
4. **Robustness**: Work even with small sample sizes
5. **Scalability**: Support thousands of training events in MongoDB
6. **Explainability**: Optional Claude Haiku summaries for non-technical players

---

## Research Literature Review

### Model A: Payoff Matrix + Nash Equilibrium

**Key References:**
- **[arXiv 1904.03821]** Blade & Soul self-play curriculum (Reward shaping for fighting games; optimal strategy from payoff matrices)
- **[NIPS 2007]** Counterfactual Regret Minimization (CFR): Theoretical foundation for game-theoretic solution concepts
- **[arXiv 2402.15923]** LSTM outcome prediction: Health trajectory models (0.94 AUC) show action-to-health mapping is learnable

**Why Payoff Matrices?**
Fighting games have **deterministic outcome mechanics**: punch at range X deals Y damage, block reduces damage by Z%. The payoff matrix encodes these mechanics and is seeded with domain knowledge, then updated empirically.

**Algorithm**: Minimax linear programming (scipy.optimize.linprog)
- Input: 9×9 payoff matrix (player_action × opponent_action)
- Output: Mixed strategy (probability distribution over 9 actions)
- Solution concept: Nash equilibrium (no player can improve by unilateral deviation)

**Incremental Updates**: Bayesian blend of prior (game mechanics seed) with likelihood (observed outcomes).

---

### Model B: Markov Chain Opponent Modeling

**Key References:**
- **[IEEE 8080432]** Online Markov action table for MCTS fighting game AI (Real-time opponent state tracking via Markov chains)
- **[arXiv 2406.02081]** FightLadder benchmark and exploitability metrics (Opponent pattern detection)

**Why Markov Chains?**
- **Computationally efficient**: O(n²) transition matrix for n=9 actions
- **Works with single round**: No warm-up period; usable immediately
- **Interpretable**: Transition probabilities are direct action frequencies
- **First-order assumption**: Sufficient for most fighting game patterns (opponent's next action depends mainly on current action, not history)

**Algorithm**: 
1. Build transition matrix from opponent's action log
2. Compute stationary distribution (long-run action probabilities) via eigendecomposition
3. Counter-strategy = argmax(payoff_matrix @ stationary_distribution)

**Limitation**: Assumes stationarity (opponent doesn't adapt round-to-round); mitigated by Model C (archetypes) and model ensemble.

---

### Model C: Archetype Clustering

**Key References:**
- **[arXiv 2404.04234]** player2vec — player behavior embeddings from event logs (Behavioral clustering and player classification)
- **[arXiv 1805.02070]** A3C+ Recurrent Info Network for fighting games (Combat style profiling)

**Why Archetypes?**
- **Semantic understanding**: "Rushdown" vs "Zoner" are human-interpretable play styles
- **Domain knowledge**: Counter-strategy lookup can be pre-computed and verified by domain experts
- **Robustness**: Works with limited data; seeded centroids improve cold-start
- **Explainability**: Recommendations include archetype rationale

**5 Archetypes**:
1. **Rushdown**: High attack frequency, high damage output, low defense. Counter: Spacing + zoning
2. **Defensive**: High block, low damage, reactive. Counter: Mix-ups + continuous pressure
3. **Zoner**: High ranged action (kicks), medium spacing. Counter: Closing distance + reads
4. **Balanced**: Equal distribution across actions. Counter: Pattern exploitation
5. **Mix-Up**: High action variety, unpredictable. Counter: Adaptation + conditioning reads

---

### Model D: Q-Function via Fitted Q Iteration

**Key References:**
- **[arXiv 2402.15923]** LSTM outcome prediction (Neural networks on fight state)
- **[arXiv 1904.03821]** Blade & Soul curriculum learning (Offline RL in fighting games)
- **[arXiv 2406.02081]** FightLadder exploitability metrics (Offline evaluation)

**Why Fitted Q Iteration?**
- **Offline learning**: No live interaction; learns from MongoDB replay data
- **Function approximation**: PyTorch MLP captures non-linear state-action value
- **Regret quantification**: Measures how suboptimal observed actions were
- **Delayed activation**: Only useful after 50+ rounds (need sufficient data)

**Algorithm**:
1. State: [health, opponent_health, distance, cooldowns, round_time] → normalize
2. Q-function: MLP(state) → [Q(s, a₁), ..., Q(s, a₉)]
3. Reward: damageDealt (punches/kicks that hit)
4. FQI iteration:
   - Sample mini-batch from MongoDB
   - Compute target Q̂ = r + γ max_a' Q(s', a')
   - Update via MSE loss
   - Repeat 5 iterations

**Regret Analysis**:
```
regret(transition t) = max_a Q(s_t, a) - Q(s_t, a_taken)
```
High regret → action was significantly suboptimal → recommend alternative.

---

## Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                   2D FIGHTING GAME (React/TS)               │
│  - Collision detection, action execution, health tracking   │
│  - Sends round-end event to Analytics Server               │
└─────────────────────────────────────────────────────────────┘
                            │
                    (POST /rounds)
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│        ANALYTICS SERVER (Node.js, MongoDB Atlas)             │
│                    Port 3001                                │
│  - Stores training_events with features                    │
│  - Real-time validation & normalization                    │
│  - Trigger FQI training when 50+ rounds accumulated         │
└─────────────────────────────────────────────────────────────┘
                            │
                (GET /recommendations)
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│          ML SERVER (FastAPI, Python)                        │
│                    Port 8001                                │
│                                                              │
│  ┌────────────────────────────────────────────────────┐   │
│  │ Model A: Payoff Matrix + Nash Equilibrium         │   │
│  │  - Seeded from game mechanics                      │   │
│  │  - Updated via Bayesian prior-likelihood blend    │   │
│  └────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌────────────────────────────────────────────────────┐   │
│  │ Model B: Markov Chain Opponent Modeling           │   │
│  │  - Transition matrix from action log              │   │
│  │  - Stationary distribution & counter-strategy     │   │
│  └────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌────────────────────────────────────────────────────┐   │
│  │ Model C: Archetype Clustering (K-means)           │   │
│  │  - 5 seeded archetypes                            │   │
│  │  - Updated centroids from training data           │   │
│  │  - Counter-strategy lookup table                  │   │
│  └────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌────────────────────────────────────────────────────┐   │
│  │ Model D: Q-Function (PyTorch FQI)                 │   │
│  │  - 15-dim state → 9 Q-values                      │   │
│  │  - Activated after 50+ rounds                     │   │
│  │  - Regret-based recommendation ranking            │   │
│  └────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌────────────────────────────────────────────────────┐   │
│  │ Ensemble Aggregator                               │   │
│  │  - Confidence-weighted scoring                    │   │
│  │  - Deduplication by source family                │   │
│  │  - Top-3 ranking                                  │   │
│  └────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌────────────────────────────────────────────────────┐   │
│  │ Optional: Claude Haiku Explainer                  │   │
│  │  - One-liner summary per recommendation           │   │
│  │  - Uses Anthropic API (low cost, fast)           │   │
│  └────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
              (JSON: recommendations)
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  GAME CLIENT (React/TS)                     │
│  - Displays recommendations in UI                          │
│  - "Learn from this" button → logs feedback                │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
ROUND ENDS
  ├─ Collect action_log, health_trajectory, cooldowns, positions
  ├─ Normalize features (min-max scaling to [0,1])
  ├─ POST to /api/analytics/training_events
  │   └─ MongoDB: training_events collection
  │
  └─ POST to /api/ml/recommendations
      ├─ Model A: Solve payoff matrix
      ├─ Model B: Build Markov chain
      ├─ Model C: Assign archetype + lookup counter
      ├─ Model D: Q-forward passes (if 50+ rounds)
      ├─ Ensemble: Combine & rank
      ├─ Optional LLM: Summarize
      └─ Return [recommendation₁, recommendation₂, recommendation₃]
```

---

## Model A: Payoff Matrix & Nash Equilibrium

### Theoretical Foundation

A **payoff matrix** for fighting games is a 9×9 matrix where:
```
payoff[player_action][opponent_action] = expected_health_delta
```

**Example**:
```
         opponent:    idle    punch    kick    block
player:
  idle       [0      -8      -12      0   ]
  punch      [+4     -2      -5       +2  ]
  kick       [+6     +4      -3       +1  ]
  block      [0      +2      +3       0   ]
```

Interpretation:
- Player punch vs Opponent idle: +4 health (hit lands)
- Player punch vs Opponent punch (simultaneous): -2 (both take damage)
- Player kick vs Opponent block: +1 (block reduces damage from 12 to 11)

### Nash Equilibrium

A **mixed strategy** is a probability distribution over actions. At Nash equilibrium:
- Neither player can improve expected payoff by unilaterally deviating
- Solved via **minimax linear programming**:

```
max  v
subject to:
  payoff[opponent_action] @ strategy ≥ v  for all opponent actions
  sum(strategy) = 1
  strategy[i] ≥ 0  for all i
```

**Interpretation**: The optimal mixed strategy guarantees expected payoff ≥ v regardless of opponent's action.

### Implementation Design

#### Seeding from Game Mechanics

Initial payoff matrix is constructed from known game mechanics:

| Action | Damage | Startup (ms) | Block Reduction |
|--------|--------|--------------|-----------------|
| Punch  | 8      | 30-40        | 80%             |
| Kick   | 12     | 100-125      | 80%             |
| Block  | N/A    | Instant      | 80% damage      |
| Move   | 0      | Instant      | N/A             |

**Assumptions**:
- Punch hits 30-40ms after execution (detection in frame window)
- Kick hits 100-125ms after execution
- Block reduces all incoming damage by 80%
- Simultaneous actions deal reduced damage (both partially hit)

#### Bayesian Incremental Update

After each round, update payoff matrix beliefs:

```
payoff_posterior = (1 - α) × payoff_prior + α × payoff_observed

where:
  α = learning_rate (e.g., 0.1)
  payoff_observed = empirical damage from this round
  payoff_prior = previous belief (seeded from mechanics)
```

**Effect**: Early rounds heavily weighted toward game mechanics seed; as more data accumulates, empirical outcomes dominate.

#### Confidence Scoring

Confidence for Model A recommendation:

```
confidence_A = min(1.0, data_sample_size / N_threshold)

where:
  N_threshold = 20 (number of observed transitions per cell)
  data_sample_size = count of (action, opponent_action) pairs observed
```

If few data points, confidence lower; recommendations still valid but more cautious.

---

## Model B: Markov Chain Opponent Modeling

### Theoretical Foundation

A **first-order Markov chain** over actions assumes:

```
P(action_t | history) = P(action_t | action_{t-1})
```

The **transition matrix** is:

```
transition[from_action][to_action] = P(next = to_action | current = from_action)
```

**Stationary distribution** π satisfies:

```
π = π @ transition_matrix

Interpretation: π[i] = long-run proportion of time opponent uses action i
```

Found via **eigendecomposition**: π is the left eigenvector with eigenvalue 1.

### Counter-Strategy

Once stationary distribution is known, compute counter-strategy:

```
counter_action = argmax_a ( payoff[a] @ stationary_distribution )

Interpretation: Take action a that maximizes expected damage against opponent's steady-state distribution
```

### Implementation Design

#### Transition Matrix Construction

From opponent's action log `[a₁, a₂, ..., aₙ]`:

```
transition[i][j] = count(a_t = i and a_{t+1} = j) / count(a_t = i)
```

Handle **zero counts** via Laplace smoothing:

```
transition[i][j] = (count[i][j] + 1) / (sum_j count[i][j] + 9)
```

Ensures all transitions have non-zero probability (avoids degenerate matrices).

#### Cold-Start Behavior

With only **one action** from opponent: Can't build Markov chain from a single transition.

**Fallback**: Use **stationary distribution from uniform prior** (each action 1/9 probability) until 5+ actions observed. After 5+ actions, use empirical Markov chain.

#### Confidence Scoring

```
confidence_B = min(1.0, action_count / N_threshold)

where:
  N_threshold = 9 (need ~9 transitions for robust estimates)
```

With fewer actions, confidence is reduced but recommendation still issued.

### Limitations & Mitigations

**Assumption 1: Stationarity** (opponent doesn't adapt round-to-round)
- Mitigation: Model C (archetypes) captures higher-level opponent types
- Mitigation: Model ensemble detects conflicting signals

**Assumption 2: First-order dependence**
- Reality: Some combos have memory (player chains 2-3 actions)
- Mitigation: Acceptable for post-round coaching (not real-time)
- Mitigation: Model D (Q-function) learns higher-order patterns

---

## Model C: Archetype Clustering

### Five Archetypes

Fighting game players fall into behavioral clusters. We define **5 archetypes** with domain-knowledge-seeded centroids:

#### 1. Rushdown
- **Characteristics**: High attack frequency, low block ratio, position forward, short distances
- **Counter-strategy**: Create space via backward movement + kicks; use block when close
- **Feature signature**: 
  - action_frequency[punch] + action_frequency[kick] > 50%
  - block_ratio < 10%
  - avg_distance < threshold

#### 2. Defensive
- **Characteristics**: High block ratio, low attack frequency, reactive, reactive movement
- **Counter-strategy**: Mix-ups (overhead vs low), continuous pressure, grab/throw if enabled
- **Feature signature**:
  - block_ratio > 30%
  - action_frequency[attack] < 30%
  - reaction_time (time between opponent action and response) < 200ms

#### 3. Zoner
- **Characteristics**: High kick frequency (ranged), medium distance, spacing-oriented
- **Counter-strategy**: Close distance quickly via move_forward + jump; punish recovery
- **Feature signature**:
  - action_frequency[kick] > 40%
  - avg_distance > threshold
  - kick_landing_rate > 40%

#### 4. Balanced
- **Characteristics**: Even distribution across actions, no strong preference
- **Counter-strategy**: Exploit patterns in specific situations (corner, low health)
- **Feature signature**:
  - All action frequencies within [10%, 20%] range
  - No clustering on block or attack

#### 5. Mix-Up
- **Characteristics**: High action entropy, unpredictable, frequent action changes
- **Counter-strategy**: Conditioning reads (repeated blocks then punish); focus on health lead
- **Feature signature**:
  - Shannon entropy of action distribution > threshold
  - Low action repeat rate

### K-Means Clustering

**Algorithm**:
1. Define seeded centroids for each archetype (domain knowledge)
2. Feature vector: 15-dim normalized features (see Dataset section)
3. Each round's data → assign to nearest centroid via **cosine similarity**
4. Update centroids via K-means iteration when 20+ rounds accumulated
5. Assign archetype via nearest centroid

**Cosine similarity**:
```
similarity(x, centroid) = (x · centroid) / (||x|| ||centroid||)
```

Why cosine? Fighting game features are **directional**: high punch frequency + low kick frequency is distinct from (low punch + high kick), even if Euclidean distance is same.

### Counter-Strategy Lookup

Each archetype has a pre-computed **counter-strategy recommendation**:

| Archetype | Counter-Strategy | Rationale |
|-----------|-----------------|-----------|
| Rushdown  | Create spacing, use kick zoning, block when close | Deny approach, keep distance |
| Defensive | Mix-ups, continuous pressure, conditioning reads | Force decisions, prevent reactions |
| Zoner     | Forward movement + jump, punish recovery | Close distance, avoid projectiles |
| Balanced  | Exploit phase reads, focus health lead | No obvious pattern; bet on gamestate |
| Mix-Up    | Conditioning, block and punish, health trading | Adapt to their adaptation |

### Cold-Start Seeding

Before 20 rounds, use **hand-crafted seeded centroids** (based on domain expert knowledge). After 20 rounds, K-means updates centroids from real data.

**Confidence Scoring**:
```
confidence_C = (1 - min_distance / max_possible_distance)

where:
  min_distance = cosine distance to nearest centroid
  max_possible_distance = max observed distance to any centroid
```

Very close to centroid → high confidence; equidistant from 2 archetypes → lower confidence.

---

## Model D: Q-Function via Fitted Q Iteration

### Theoretical Foundation

**Q-function** in RL:
```
Q(s, a) = expected cumulative reward from state s, taking action a
```

In fighting games:
```
Q(s, a) = expected damage output from state s if we execute action a
```

**Fitted Q Iteration (FQI)** learns Q via **offline batch updates** (no live rollouts):

```
For iteration k = 1, 2, ..., K:
  For each (s, a, r, s') in training_batch:
    target_q = r + γ × max_a' Q_{k-1}(s', a')
    Update Q_k to minimize: (Q_k(s, a) - target_q)²
```

**Offline RL context**: We have historical (s, a, r, s') tuples from MongoDB; we update Q iteratively without live interaction.

### State Representation

15-dimensional normalized state vector:

| Index | Feature | Range |
|-------|---------|-------|
| 0-1   | Player health, Opponent health | [0, 1] (normalized) |
| 2-3   | Player x, Player y | [0, 1] (arena bounds) |
| 4-5   | Opponent x, Opponent y | [0, 1] (arena bounds) |
| 6-7   | Player punch cooldown, kick cooldown | [0, 1] (0=ready) |
| 8-9   | Opponent punch cooldown, kick cooldown | [0, 1] |
| 10-11 | Round time elapsed, round time remaining | [0, 1] |
| 12-13 | Distance (euclidean), directional (facing) | [0, 1], {-1, 0, +1} |
| 14    | (Reserved for future: combo counter, hitstun, etc.) | [0, 1] |

**Normalization**:
```
feature_norm = (feature - feature_min) / (feature_max - feature_min)
Clamp to [0, 1] to handle outliers
```

### Reward Function

**Reward per transition**:
```
r(s, a, s') = damage_dealt - damage_taken

where:
  damage_dealt = opponent_health[s] - opponent_health[s']
  damage_taken = player_health[s] - player_health[s']
```

**Interpretation**: Positive reward for landing attacks; negative for taking damage. Encourages damage-positive trades.

### Q-Network Architecture

**PyTorch MLP**:
```
Input (15) → 64 ReLU → 64 ReLU → 32 ReLU → 9 (Q-values)
```

Shallow network (3 hidden layers) sufficient for fighting game state space.

### FQI Training Pipeline

**Activation condition**: Only train when 50+ rounds in MongoDB

**Training data**:
- Sample 32 rounds randomly from MongoDB
- Unroll each round into (s, a, r, s') transitions
- Total: ~32 × 20 = 640 transitions per mini-batch (estimate ~20 actions/round)

**5 iterations**:
```
For iteration k = 1 to 5:
  For mini_batch in training_data:
    Compute target: target_q = r + 0.99 × max_a' Q_old(s', a')
    Update Q via MSE: loss = (Q_new(s, a) - target_q)²
    Optimize with Adam, lr=0.001
```

**Update frequency**: Retrain every 10 new rounds (or when 100+ new transitions accumulated)

### Regret Analysis

**Offline RL regret** quantifies action optimality:

```
regret(s, a) = max_a' Q(s, a') - Q(s, a)

Interpretation: 
  If max Q-value is 5 and observed action had Q=1, regret=4
  High regret → action was suboptimal
```

**Recommendation heuristic**:
```
If regret(s, a_taken) > regret_threshold (e.g., 3):
  Recommend: alternate_action = argmax_a Q(s, a)
  Confidence = min(1.0, Q(s, alternate) / max_possible_reward)
```

### Limitations & When to Use

**Limitations**:
1. **Data-hungry**: Needs 50+ rounds to activate (cold start not possible)
2. **Distribution shift**: Q-values assume observed states are representative; novel game states untrained
3. **Offline bias**: Can't correct for actions not in training data (off-policy learning without importance weighting)
4. **Assumption of stationarity**: Environment doesn't change (opponents always play similarly)

**When to use**:
- Long-term player progression (100+ rounds accumulated)
- Recurring matchups (same opponent repeatedly)
- NOT for first encounter with new opponent

**Mitigation strategies**:
- Combine with Model B (Markov) and Model C (Archetypes) for opponent-specific reads
- Weight regret-based recommendations lower in early stage (0-50 rounds)
- Use models A, B, C until 50+ rounds, then add D

---

## Dataset and Feature Engineering

### Training Data Collection

**Source**: MongoDB `training_events` collection

Each event is a complete **round**:

```json
{
  "round_id": "uuid",
  "timestamp": "2025-04-27T...",
  "player": {
    "id": "player_uuid",
    "actions": [
      {"time": 100, "action": "punch", "hit": true, "damage": 8},
      {"time": 200, "action": "move_forward", "hit": false, "damage": 0},
      ...
    ],
    "health_trajectory": [100, 98, 96, ...],
    "position_trajectory": [[0, 0], [0.5, 0], [1.0, 0], ...],
    "cooldowns": {
      "punch": [30, 0, 0, ...],
      "kick": [100, 90, 80, ...]
    }
  },
  "opponent": {
    "id": "opp_uuid",
    "actions": [...],
    "health_trajectory": [...],
    ...
  },
  "round_duration_ms": 8000,
  "winner": "player_id"
}
```

### Feature Extraction

**Per-round aggregate features** (15-dim for Q-function state):

1. **Health features** (2):
   - player_health_at_round_end / 100
   - opponent_health_at_round_end / 100

2. **Position features** (4):
   - mean(player_x) / arena_width
   - mean(player_y) / arena_height
   - mean(opponent_x) / arena_width
   - mean(opponent_y) / arena_height

3. **Cooldown features** (4):
   - mean(player_punch_cooldown) / max_cooldown
   - mean(player_kick_cooldown) / max_cooldown
   - mean(opponent_punch_cooldown) / max_cooldown
   - mean(opponent_kick_cooldown) / max_cooldown

4. **Temporal features** (2):
   - round_time_elapsed / round_duration
   - round_time_remaining / round_duration

5. **Distance features** (2):
   - euclidean_distance / max_distance
   - (relative_x > 0) ? 1 : -1  (which direction)

6. **Archetype features** (15 for clustering):
   - action_frequency[i] for i in 0-8 (9 actions)
   - block_ratio
   - damage_dealt_ratio
   - damage_taken_ratio

### Normalization

**Min-max scaling** per feature:

```
feature_norm = (feature - dataset_min) / (dataset_max - dataset_min)
Clamp to [0, 1]
```

**Why**: Neural networks (Model D) converge faster with normalized inputs; comparison across features is fair.

### Data Quality

**Validation rules**:
1. **Sanity checks**: All features in [0, 1] (after normalization)
2. **No nulls**: Missing cooldown → assume full value
3. **Temporal ordering**: action_time strictly increasing
4. **Health bounds**: health in [0, max_health]

**Outlier handling**:
- Clamp health to [0, max_health]
- Clamp distance to [0, max_distance]
- If feature exceeds 3σ from mean: flag for review, clamp to mean ± 3σ

---

## Cold Start Strategy

### The Cold Start Problem

On **round 1** with **no training data**, how do we generate recommendations?

Naive approaches fail:
- **Model B (Markov)**: Single opponent action → can't build chain
- **Model D (FQI)**: Needs 50+ rounds
- **Model C (Archetypes)**: Seeded centroids help, but no signal from this opponent yet

### Solution: Hybrid Cold Start

**Round 1-5: Mechanics-Seeded Models A & B + Seeded Archetype C**

1. **Model A (Payoff Matrix)**: Use **game mechanics seed** (no empirical data yet)
   - Punch deals 8 damage, kick deals 12, block reduces 80%
   - Solve Nash equilibrium on seeded payoff matrix
   - Confidence: 0.3 (very low; mechanics may not reflect actual balance)

2. **Model B (Markov)**: Use **uniform stationary distribution**
   - Opponent equally likely to do any action
   - Counter-strategy = most damaging action (likely kick)
   - Confidence: 0.1 (single action observed; high uncertainty)

3. **Model C (Archetype)**: Assign to **nearest seeded centroid**
   - Cosine similarity between observed features and 5 archetypes
   - Return archetype + its counter-strategy
   - Confidence: min(1.0, features_observed / features_required)
     - With 1 action, maybe confidence 0.2; more actions → higher

### Ensemble Aggregation at Cold Start

**Scoring**:
```
score = confidence × priority_weight

priority_weights:
  Model A: 0.4 (game theory foundation)
  Model B: 0.2 (high uncertainty; low priority)
  Model C: 0.3 (archetype intuition)
  Model D: 0 (not active yet)
```

**Example cold-start recommendation**:
```json
{
  "recommendations": [
    {
      "title": "Exploit Opponent's Damage Gap",
      "detail": "Nash equilibrium suggests kick (12 dmg) vs opponent's expected idle. Use kick to build health lead.",
      "priority": "high",
      "source_models": ["ModelA"],
      "confidence": 0.3
    },
    {
      "title": "Assign Opponent Archetype",
      "detail": "Opponent appears Balanced (equal action distribution). Counter with pattern exploitation in specific game phases.",
      "priority": "medium",
      "source_models": ["ModelC"],
      "confidence": 0.25
    }
  ]
}
```

### Graceful Degradation

As more data accumulates:

| Rounds | Active Models | Dominant Signal |
|--------|---------------|-----------------|
| 1-5    | A (seeded), B (uniform), C (seeded) | Game mechanics + player style |
| 5-20   | A (empirical), B (empirical), C (seeded) | Emerging patterns in opponent + archetype |
| 20-50  | A, B, C (all empirical) + D warming up | Opponent-specific models + behavior shift |
| 50+    | A, B, C, D (all active) | Hybrid ensemble |

---

## Progressive Learning Milestones

The system evolves from cold-start (mechanics-based) to data-driven (ensemble of all models).

| Rounds | Model A | Model B | Model C | Model D | Dominant Use Case |
|--------|---------|---------|---------|---------|-------------------|
| 0 (cold start) | Game mechanics seed | Uniform distribution | Seeded centroids | Inactive (0.0 weight) | New opponent encounter |
| 1-4 | Mechanics seed | Single action observed | Assign nearest centroid | Inactive | Initial opponent read |
| 5-19 | Empirical payoff update starts | Markov matrix from 5+ actions | Centroids locked (no K-means update yet) | Inactive | Opponent pattern emergence |
| 20-49 | Bayesian blend (40% mechanics, 60% empirical) | Empirical Markov with 20+ actions observed | K-means centroids updated, high confidence | Inactive | Opponent archetype refinement |
| 50-99 | Empirical dominant (80% empirical, 20% mechanics) | Empirical Markov (robust) | Archetype finalized | FQI training active, low confidence | All models learning |
| 100+ | Fully empirical (mechanics seed historical only) | Empirical Markov (stable) | Archetype stable | FQI confidence high | Mature opponent model |

### Detailed Milestone Descriptions

#### Milestone 1: Cold Start (0 rounds)
- **What happens**: Game sends first round data
- **Model A**: Solve Nash on mechanics-seeded payoff (no empirical data)
- **Model B**: Uniform stationary distribution (1 action observed)
- **Model C**: Assign to nearest seeded centroid
- **Model D**: Inactive (0.0 weight)
- **Recommendation**: "Based on game mechanics, use kick (highest damage output)"

#### Milestone 2: Pattern Emergence (5 rounds)
- **What happens**: 5 rounds accumulated; opponent actions start showing pattern
- **Model A**: Begin Bayesian blend; payoff matrix shifts toward empirical
- **Model B**: Build transition matrix from 5 actions; stationary distribution more robust
- **Model C**: Confidence in archetype increases; may drift to different centroid
- **Model D**: Still inactive (< 50 rounds)
- **Recommendation**: "Opponent favors punch (70% of actions). Counter with block + kick."

#### Milestone 3: Opponent Type Identified (20 rounds)
- **What happens**: Clear opponent behavioral signature
- **Model A**: Payoff matrix 50/50 mechanics/empirical blend
- **Model B**: Markov chain stable across 20 transitions
- **Model C**: K-means refit; centroids updated; archetype locked (e.g., "Rushdown")
- **Model D**: Still inactive; accumulating MongoDB training data
- **Recommendation**: "Opponent is Rushdown archetype (high attack frequency). Space with backward movement."

#### Milestone 4: FQI Activation (50 rounds)
- **What happens**: Sufficient data for Q-network training
- **Model A**: Empirical payoff dominant (70% weight)
- **Model B**: Stable Markov matrix
- **Model C**: Confident archetype assignment
- **Model D**: FQI training begins; Q-network initialized and updated
- **Recommendation**: "Regret analysis shows your punch (Q=2) vs their block is suboptimal. Use kick instead (Q=5)."

#### Milestone 5: Mature System (100+ rounds)
- **What happens**: Full ensemble at capacity
- **Model A**: Fully empirical payoff matrix
- **Model B**: Markov chain stable and interpretable
- **Model C**: Archetype locked; no further updates
- **Model D**: High-confidence Q-function; used for fine-tuning
- **Recommendation**: "Hybrid recommendation: payoff matrix suggests kick (A), Markov confirms opponent dodges kicks (B), but Q-function shows kick punishes their recovery (D). Use kick in round 2."

---

## API Design

### Analytics Server (Node.js, Port 3001)

#### Endpoint: POST /api/analytics/training_events

**Purpose**: Store round data in MongoDB for ML training

**Request**:
```json
{
  "round_id": "string (uuid)",
  "timestamp": "ISO 8601",
  "match_id": "string",
  "player_id": "string",
  "opponent_id": "string",
  "actions": [
    {
      "actor": "player" | "opponent",
      "time_ms": "number",
      "action": "idle" | "move_forward" | "move_backward" | "jump" | "block" | "left_punch" | "right_punch" | "left_kick" | "right_kick",
      "landed": "boolean",
      "damage": "number"
    }
  ],
  "health_trajectory": {
    "player": [100, 98, 96, ...],
    "opponent": [100, 100, 98, ...]
  },
  "position_trajectory": {
    "player": [[0, 0], [0.5, 0], ...],
    "opponent": [[10, 0], [9.5, 0], ...]
  },
  "cooldowns": {
    "player": { "punch": [...], "kick": [...] },
    "opponent": { "punch": [...], "kick": [...] }
  },
  "round_duration_ms": "number",
  "winner_id": "string"
}
```

**Response**:
```json
{
  "success": true,
  "event_id": "string (uuid)",
  "message": "Training event stored",
  "features_extracted": {
    "player": [0.95, 0.2, ...],
    "opponent": [0.45, 0.8, ...]
  }
}
```

**Validation**:
- All health values in [0, max_health]
- All actions valid enum values
- Timestamps monotonic increasing
- If any validation fails: return 400 with error details

#### Endpoint: GET /api/analytics/player_stats/:player_id

**Purpose**: Aggregate stats for a player (for frontend dashboard)

**Query params**:
- `opponent_id` (optional): Filter by specific opponent
- `limit` (optional, default 50): Number of recent rounds

**Response**:
```json
{
  "player_id": "string",
  "total_rounds": 42,
  "win_rate": 0.64,
  "opponent_archetypes": {
    "rushdown": 15,
    "defensive": 10,
    "zoner": 8,
    "balanced": 5,
    "mix_up": 4
  },
  "average_damage_per_round": 45.3,
  "action_frequencies": {
    "punch": 0.35,
    "kick": 0.25,
    "block": 0.20,
    ...
  }
}
```

---

### ML Server (FastAPI, Port 8001)

#### Endpoint: POST /api/ml/recommendations

**Purpose**: Generate post-round strategy recommendations

**Request**:
```json
{
  "player_id": "string",
  "opponent_id": "string",
  "round_id": "string",
  "action_log": [
    {
      "actor": "player" | "opponent",
      "action": "string",
      "time_ms": "number"
    }
  ],
  "features": {
    "player_health": 0.5,
    "opponent_health": 0.8,
    ...
  },
  "include_llm_summary": true
}
```

**Response**:
```json
{
  "recommendations": [
    {
      "rank": 1,
      "title": "Exploit Health Gap",
      "detail": "You are at 50% health, opponent at 80%. Aggress with kick (Model A: payoff +6 vs their likely block).",
      "priority": "high",
      "confidence": 0.72,
      "source_models": ["ModelA", "ModelB"],
      "action_suggestion": "right_kick",
      "rationale": {
        "model_a": {
          "payoff_matrix_q_value": 6.0,
          "nash_equilibrium_probability": 0.45
        },
        "model_b": {
          "counter_action": "right_kick",
          "stationary_distribution": {
            "punch": 0.3,
            "block": 0.5,
            "kick": 0.2
          }
        }
      }
    },
    {
      "rank": 2,
      "title": "Opponent is Defensive",
      "detail": "Model C identifies Defensive archetype (high block ratio). Mix-ups recommended: vary between high/low attacks.",
      "priority": "medium",
      "confidence": 0.58,
      "source_models": ["ModelC"],
      "action_suggestion": "left_punch",
      "rationale": {
        "model_c": {
          "assigned_archetype": "defensive",
          "cosine_similarity": 0.82,
          "counter_strategy": "Mix-ups + continuous pressure"
        }
      }
    },
    {
      "rank": 3,
      "title": "Q-Function Suggests Alternative",
      "detail": "Historical data shows block (Q=2.1) underperforms vs. their pattern; move_forward (Q=3.8) better positions for next round.",
      "priority": "low",
      "confidence": 0.41,
      "source_models": ["ModelD"],
      "action_suggestion": "move_forward",
      "rationale": {
        "model_d": {
          "q_value_action_taken": 2.1,
          "q_value_best_alternative": 3.8,
          "regret": 1.7,
          "training_rounds_used": 67
        }
      }
    }
  ],
  "summary": "Focus on aggressive play (Model A), vary your attacks (Model C), and reposition (Model D).",
  "system_status": {
    "model_a_active": true,
    "model_b_active": true,
    "model_c_active": true,
    "model_d_active": true,
    "rounds_accumulated": 142
  }
}
```

**Error Handling**:
- `400 Bad Request`: Invalid action_log or missing required fields
- `503 Service Unavailable`: Models not yet trained (< 50 rounds); fallback to cold-start A+B+C

#### Endpoint: GET /api/ml/model_status

**Purpose**: Check which models are active and their training status

**Response**:
```json
{
  "model_a": {
    "active": true,
    "name": "Payoff Matrix + Nash Equilibrium",
    "status": "empirical_dominant",
    "confidence": 0.85,
    "description": "Seeded from game mechanics, now 80% empirical"
  },
  "model_b": {
    "active": true,
    "name": "Markov Chain Opponent Model",
    "status": "stable",
    "confidence": 0.92
  },
  "model_c": {
    "active": true,
    "name": "Archetype Clustering",
    "status": "learned",
    "confidence": 0.78
  },
  "model_d": {
    "active": true,
    "name": "Q-Function FQI",
    "status": "training",
    "confidence": 0.64,
    "training_rounds": 142,
    "training_iterations": 5
  },
  "total_rounds_accumulated": 142,
  "system_status": "mature"
}
```

---

## Frontend Integration

### React/TypeScript Game Component

#### Recommendation Modal

After each round ends, display a modal:

```
┌─────────────────────────────────────────────────┐
│  ROUND OVER                                      │
│  You lost. Learn from this round.               │
├─────────────────────────────────────────────────┤
│                                                  │
│  📊 AI COACH RECOMMENDATIONS (3 suggestions)   │
│                                                  │
│  [1] ⭐⭐⭐ HIGH PRIORITY                       │
│      "Exploit Health Gap"                       │
│      You at 50%, opponent at 80%. Use kick      │
│      [Learn] [Dismiss]                          │
│                                                  │
│  [2] ⭐⭐ MEDIUM PRIORITY                       │
│      "Opponent is Defensive"                    │
│      Try mix-ups vs. their high block ratio    │
│      [Learn] [Dismiss]                          │
│                                                  │
│  [3] ⭐ LOW PRIORITY                            │
│      "Q-Function Suggests Move Forward"         │
│      Reposition for next round                  │
│      [Learn] [Dismiss]                          │
│                                                  │
├─────────────────────────────────────────────────┤
│  💡 Summary: Focus on aggression (Model A),    │
│  vary attacks (Model C), reposition (Model D)  │
│                                                  │
│  [Next Round] [Stats] [View Details]            │
└─────────────────────────────────────────────────┘
```

#### "Learn" Button Feedback

When user clicks [Learn]:
1. Log feedback event to MongoDB (user_id, recommendation_id, accepted=true)
2. Increment model confidence for that model
3. Optional: Use feedback to fine-tune models (e.g., Bayesian update)

#### Stats Dashboard

Accessible from main menu or post-round modal:

```
┌─────────────────────────────────────────────────┐
│  PLAYER STATS                                    │
├─────────────────────────────────────────────────┤
│  Total Rounds: 142 | Win Rate: 64%              │
│                                                  │
│  Opponent Breakdown:                            │
│    Rushdown: 35% (50 rounds) → 70% WR          │
│    Defensive: 25% (35 rounds) → 60% WR         │
│    Zoner: 20% (28 rounds) → 50% WR             │
│    Balanced: 12% (17 rounds) → 65% WR          │
│    Mix-Up: 8% (12 rounds) → 42% WR             │
│                                                  │
│  Model Performance:                             │
│    Model A (Payoff): 0.85 confidence           │
│    Model B (Markov): 0.92 confidence           │
│    Model C (Archetype): 0.78 confidence        │
│    Model D (Q-Function): 0.64 confidence       │
│                                                  │
│  Top Recommended Actions:                       │
│    1. Right Kick (18% recommendation rate)     │
│    2. Move Forward (15%)                       │
│    3. Block (14%)                              │
└─────────────────────────────────────────────────┘
```

#### Confidence Visualization

Each recommendation shows confidence as visual indicator:

```
Model A (Payoff Matrix): ████████░░ 0.85
Model B (Markov Chain):  █████████░ 0.92
Model C (Archetype):     ████████░░ 0.78
Model D (Q-Function):    ██████░░░░ 0.64
```

---

## Training Recipes

### Recipe 1: Bootstrap Model A (Payoff Matrix)

**When**: Before any rounds played

**Steps**:
1. Define 9×9 payoff matrix from game mechanics (punch=8 dmg, kick=12 dmg, block=80% reduction)
2. Solve Nash equilibrium via `scipy.optimize.linprog()`
3. Extract mixed strategy probabilities
4. Test: With no opponent history, recommend highest-damage action (kick, ~12 dmg)

**Validation**: Sanity check that no strategy is strictly dominated (all actions should have non-zero Nash probability)

### Recipe 2: Populate Model C Archetypes

**When**: Before first round or as domain expert task

**Steps**:
1. Define 5 archetype centroids (manually, domain knowledge)
   - Example: Rushdown = [0.5, 0.3, 0.0, 0.15, 0.05, ...] (high attack, low block, etc.)
2. For each archetype, define counter-strategy rationale
   - Rushdown counter: "Create space via kick zoning"
3. Store centroids in persistent config file
4. Test: New opponent → should assign to nearest centroid

### Recipe 3: Incremental Learning Loop (Online)

**When**: Each round plays

**Steps**:
1. Round ends; Game sends action log to `/api/analytics/training_events`
2. Analytics server stores in MongoDB
3. Every 5 rounds (or on demand), trigger `/api/ml/recommendations`
4. Model A: Bayesian update payoff matrix (α=0.1 blend)
   ```
   payoff_new = 0.9 × payoff_old + 0.1 × empirical_damage_observed
   ```
5. Model B: Rebuild Markov transition matrix from all opponent actions
6. Model C: If 20+ rounds, run K-means 1 iteration (update centroids)
7. Return recommendations via ensemble aggregation

### Recipe 4: FQI Training (Batch)

**When**: 50+ rounds accumulated; run every 10 new rounds

**Steps**:
1. Load all (state, action, reward, next_state) tuples from MongoDB
2. Initialize PyTorch MLP: 15→64→64→32→9
3. For iteration k = 1 to 5:
   a. Mini-batch sample from MongoDB
   b. Forward: Q_old(s, a)
   c. Target: r + 0.99 × max_a' Q_old(s', a')
   d. Backward: MSE loss, Adam optimizer
   e. Update Q weights
4. Save Q-network to disk
5. Compute regret per transition for next round's recommendations

### Recipe 5: Offline Validation (Batch)

**When**: Weekly or after significant data accumulation

**Steps**:
1. Split MongoDB data: 80% train, 20% test
2. Train Models A, B, C, D on train set
3. On test set:
   a. For each round, generate recommendations
   b. Compare Model D Q-value against actual outcome reward
   c. Compute mean absolute error (MAE)
   d. Report coverage: % rounds where top recommendation was actually optimal?
4. If MAE > threshold or coverage < threshold: review model assumptions

---

## Why NOT Pure LLM / Pure Deep RL

### Why Not Pure LLM?

#### Potential Approach
```
Given round data (action log, health trajectory):
  → Prompt Claude Haiku with game description
  → LLM generates strategy recommendation
  → Return to player
```

#### Why It Falls Short

1. **No action grounding**: LLM can describe strategy ("be more aggressive"), but can't quantify "aggressive" or explain *which action* (punch vs kick?)
2. **Latency**: LLM API call ~200-500ms per recommendation; game session expects sub-100ms response
3. **Cost**: Every recommendation = API call; at $0.80 per million input tokens, 100 recommendations/day = $0.08/day (small but non-zero). Model A-D are free once deployed.
4. **Hallucination risk**: LLM might invent plausible-sounding but incorrect game mechanics ("block reduces 50% damage" when actually 80%)
5. **No learning**: LLM doesn't improve with more game data; humans must re-prompt it

#### Our Approach
- Use Claude Haiku for **optional narrative summary** of recommendations (e.g., "Focus on aggression (Model A), vary attacks (Model C)")
- Models A-D generate **ground-truth recommendations** based on game mechanics and data
- LLM is **decorative**, not foundational

### Why Not Pure Deep RL?

#### Potential Approach
```
Train an end-to-end deep RL agent (e.g., PPO, SAC):
  state (15-dim) → neural network → [Q-values for 9 actions]
  Optimize on MongoDB replay data
```

#### Why It Falls Short

1. **Cold start impossible**: Pure RL needs exploration to learn; offline RL from batch data alone is hard
   - Model D (FQI) with seeding from Models A, B, C avoids this
2. **Interpretability**: Neural network outputs [0.1, 0.8, 0.05, ...] for Q-values. *Why* is action 1 best? No explanation.
   - Models A, B, C provide explicit rationales (payoff matrix, Markov chain, archetype)
3. **Sample efficiency**: Fighting games have sparse rewards (damage = reward). Pure RL needs many interactions.
   - Hybrid approach: Models A, B, C give strong priors; Model D refines
4. **Non-stationary opponent**: Opponent adapts round-to-round; pure RL assumes fixed reward function
   - Models B + C detect adaptation; re-cluster, recompute counter
5. **Overfit to training distribution**: If training data is vs. Defensive opponents, policy overfits; fails on Rushdown
   - Archetype clustering (Model C) encourages generalization

#### Our Approach
- **Model D (FQI)** is deep RL, but:
  - Seeded by Models A, B, C (warm start)
  - Only activated at 50+ rounds (enough data)
  - Ensemble-weighted with game-theoretic models (not dominant signal)
  - Offline training (no live interaction required)

---

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)

- [ ] Define payoff matrix from game mechanics spec
- [ ] Implement Model A (payoff + Nash via scipy)
- [ ] Create MongoDB schema for training_events
- [ ] Set up FastAPI server skeleton (Port 8001)
- [ ] Test: cold-start recommendation on toy data

### Phase 2: Opponent Modeling (Weeks 2-3)

- [ ] Implement Model B (Markov chain with Laplace smoothing)
- [ ] Implement Model C (K-means + seeded archetypes)
- [ ] Ensemble aggregation (confidence-weighted scoring)
- [ ] Test: recommendations on 10+ real game rounds

### Phase 3: RL Backend (Weeks 3-4)

- [ ] Define state representation (15-dim normalized features)
- [ ] Implement Model D (PyTorch MLP + FQI)
- [ ] Implement regret analysis
- [ ] Conditional activation (only at 50+ rounds)
- [ ] Test: Q-network on offline batch from MongoDB

### Phase 4: Integration & Polish (Weeks 4-5)

- [ ] Connect React frontend to `/api/ml/recommendations`
- [ ] Build recommendation modal UI
- [ ] Add stats dashboard
- [ ] Optional: Claude Haiku summarization layer
- [ ] Testing: 50+ real game rounds, user feedback
- [ ] Performance tuning: caching, batch optimization

### Phase 5: Validation & Deployment (Week 6)

- [ ] Offline validation: train/test split on accumulated data
- [ ] A/B test: recommendations vs. baseline (no coaching)
- [ ] Measure: user engagement, win-rate improvement
- [ ] Deploy to production (AWS/GCP or local server)
- [ ] Monitor: model drift, data quality

---

## Conclusion

The **four-model ensemble approach** balances interpretability, robustness, and learning:

1. **Game theory (Model A)**: Principled foundation from payoff matrices
2. **Opponent modeling (Model B)**: Real-time adaptability via Markov chains
3. **Behavioral clustering (Model C)**: Semantic understanding via archetypes
4. **Offline RL (Model D)**: Fine-grained optimization via fitted Q iteration

**Progressive learning** ensures cold-start capability while scaling to mature systems. **Ensemble aggregation** via confidence weighting prevents over-reliance on any single model.

This architecture is **interpretable to players**, **robust to small data**, and **continuously learning** from real gameplay.

---

## Appendix: Notation Reference

| Symbol | Meaning |
|--------|---------|
| s | State (15-dim vector) |
| a | Action (one of 9) |
| Q(s, a) | Expected cumulative reward from state s, action a |
| payoff[a₁][a₂] | Expected damage delta when P1 plays a₁ vs P2 plays a₂ |
| π | Mixed strategy (probability distribution over actions) |
| π_s | Stationary distribution of Markov chain |
| transition[a₁][a₂] | P(next action = a₂ \| current action = a₁) |
| α | Learning rate (Bayesian blend parameter) |
| γ | Discount factor (typically 0.99) |
| regret(s, a) | max_a' Q(s, a') - Q(s, a) |
| confidence | Model-specific confidence score [0, 1] |

