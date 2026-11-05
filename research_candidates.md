# Candidate Research Questions for Next Insights

## 12+ Candidate Research Questions

### 1. Corner Efficiency and Set-Piece Profiles
**Question:** Do corners contain predictive information beyond shots? Are there systematic "corner specialists" whose set-piece efficiency persists?
**Why interesting:** Corners are available but completely unexplored. Set pieces are a distinct tactical phase. If corner efficiency (goals per corner, shots per corner) persists differently from open-play efficiency, it reveals a separate skill dimension.
**Data:** HC, AC, goals, shots, shots on target. Can compute corners per game, shots per corner, goals per corner.
**Observational unit:** Team-season or team-match.
**Feasibility:** High - data fully available across all leagues/seasons.

### 2. Game State Effects on Match Statistics
**Question:** How do shot volume, accuracy, fouls, and corners change conditional on game state (leading/drawing/trailing) and time remaining?
**Why interesting:** Existing HT-FT analysis looks at results, not *how* teams play in different states. Score effects (trailing teams shoot more but less accurately) are well-known but not quantified in this dataset with shot/foul/corner data.
**Data:** Need to infer game state from HT result + match progression. Can approximate using HT score and FT score changes.
**Observational unit:** Match-period (HT-FT segments) or team-match with HT result as proxy.
**Feasibility:** Medium - no minute-by-minute data, but HT result gives crude game state.

### 3. Draw Specialists vs. Win/Loss Polarity
**Question:** Do some teams systematically draw more than expected given their strength? Is "draw tendency" a persistent team characteristic or noise?
**Why interesting:** Draws are distinct from wins/losses. A team with 15 draws, 10 wins, 13 losses (43 pts) vs 5 draws, 15 wins, 18 losses (50 pts) - same GD but different profiles. Draw tendency might reflect risk aversion, defensive solidity, or inability to close games.
**Data:** FTR, HTR, team strength (PPG).
**Observational unit:** Team-season.
**Feasibility:** High - simple to compute draw%, expected draws from strength.

### 4. Goal Difference Efficiency (Pythagorean Residuals)
**Question:** Which teams over/underperform their goal difference? Is "clutch" GD-to-points conversion persistent?
**Why interesting:** Football Pythagorean expectation (points from GF/GA) is standard. Residuals = "luck" or "clutch". Existing close-match analysis (new insight 3) touches this but from 1-goal win angle. GD efficiency is broader.
**Data:** GF, GA, Points per season.
**Observational unit:** Team-season.
**Feasibility:** High - standard calculation.

### 5. Home/Away Asymmetry in Playing Style (Beyond Results)
**Question:** Do teams systematically change their *style* (shots, corners, fouls, discipline) home vs away, beyond just winning more at home?
**Why interesting:** HA components analysis (new insight 2) shows shot generation/suppression differences. But what about fouls, corners, discipline, shot *accuracy*? Some teams may play possession football at home, counter-attack away.
**Data:** All team-match stats split by venue.
**Observational unit:** Team-season home/away splits.
**Feasibility:** High - already have home/away splits in HA components.

### 6. Competitive Balance Evolution
**Question:** Is the gap between elite and bottom teams widening across seasons? Are points becoming more concentrated?
**Why interesting:** Longitudinal competitive balance is a major football economics question. Herfindahl-Hirschman Index (HHI) of points, Gini coefficient of PPG, top-4 vs bottom-4 gaps.
**Data:** Team-season PPG across 7 seasons × 8 leagues.
**Observational unit:** League-season.
**Feasibility:** High - straightforward aggregation.

### 7. Shot Quality Proxy: Selective vs Volume Shooting Profiles
**Question:** Do teams with high shot accuracy but low volume (selective) perform differently from high volume/low accuracy (volume) teams? Do these profiles persist?
**Why interesting:** Shot decomposition (core fig4) shows volume > accuracy > finishing for PPG. But the *interaction* and *profile stability* is unexplored. A "selective shooting" profile might be more sustainable.
**Data:** Shots_PG, Shot_Accuracy, Goals_Per_Shot_OT.
**Observational unit:** Team-season.
**Feasibility:** High - all metrics exist.

### 8. Fouls Drawn as Attacking Threat Proxy
**Question:** Are fouls drawn (Fouls_Against) a better indicator of dangerous attacking play than shots? Does "winning fouls in dangerous areas" persist and predict goals?
**Why interesting:** Foul differential analysis (core fig11) shows fouls drawn - fouls committed correlates with PPG (r=0.28). But fouls drawn *alone* as attacking metric is unexplored. Fouls often occur in final third during dangerous attacks.
**Data:** Fouls, Fouls_Against, GF, Shots.
**Observational unit:** Team-season or team-match.
**Feasibility:** High - data available.

### 9. Red Card Timing and Asymmetric Impact
**Question:** When in matches do red cards occur? Does early red card (1st half) have different impact than late (2nd half)? Does impact differ by score state at time of dismissal?
**Why interesting:** Core fig10 shows red cards flip win probability ~20-25pp. But timing matters - early red forces tactical reshuffle for longer; late red may just see out game. No timing data, but can proxy via HT vs FT red card presence.
**Data:** HR, AR, HTR, FTR.
**Observational unit:** Match with red card.
**Feasibility:** Medium - limited by no exact timing, but HT red vs FT red proxy exists.

### 10. League Style Fingerprints
**Question:** Do leagues have distinct, persistent statistical "fingerprints" (shots/corners/fouls ratios)? Are these converging or diverging?
**Why interesting:** Tactical convergence debate. Premier League vs La Liga vs Bundesliga styles. Do shot/foul/corner ratios distinguish leagues persistently?
**Data:** All team-match stats aggregated by league-season.
**Observational unit:** League-season.
**Feasibility:** High - simple aggregation.

### 11. Points from Winning/Losing Positions ("Character" Metric)
**Question:** Which teams gain most points from losing positions and drop fewest from winning positions? Is this persistent?
**Why interesting:** "Mental strength" narrative. Comeback analysis (core fig7) looks at HT trailing → FT not lose. This extends to any point in match (approximated via HT/FT). Points won from behind vs points dropped when ahead.
**Data:** HTR, FTR → can infer winning/losing/drawing at HT and FT.
**Observational unit:** Team-season.
**Feasibility:** High - HTR/FTR available.

### 12. Match Tempo (Total Shot Volume) and Predictability
**Question:** Do high-tempo matches (many shots both teams) produce more predictable or less predictable outcomes? Is tempo a team characteristic?
**Why interesting:** Some matches are open (30+ shots), some cagey (15 shots). Does tempo correlate with result variance? Do some teams consistently play in high/low tempo matches?
**Data:** HS + AS = total shots per match.
**Observational unit:** Match or team-season (avg tempo).
**Feasibility:** High - simple calculation.

### 13. Seasonal Trajectory Shapes (Beyond Linear)
**Question:** Do team performances follow non-linear trajectories within seasons (U-shape: slow start, improvement, fade; steady improvement; steady decline; early peak)?
**Why interesting:** Existing analyses use early vs late split or rolling 5-game. But the *shape* of the trajectory might be informative. Newly promoted teams might improve steadily (new insight 6); relegated teams might decline non-linearly.
**Data:** Rolling PPG, Shot_Dominance by match number.
**Observational unit:** Team-season trajectory.
**Feasibility:** Medium - needs curve fitting per team-season.

### 14. Disciplinary Asymmetry in Game States
**Question:** Do trailing teams commit more fouls (tactical fouling)? Do leading teams draw more fouls? How do yellow/red cards distribute across game states?
**Why interesting:** Tactical fouling is a known strategy. Foul differential by game state unexplored. Could reveal "streetwise" vs "naive" teams.
**Data:** Fouls, Yellow, Red, HTR (as game state proxy), FTR.
**Observational unit:** Team-match with HT result as game state.
**Feasibility:** Medium - HTR is crude game state proxy.

### 15. Corner-to-Shot-to-Goal Chain (Set-Piece Efficiency Funnel)
**Question:** What's the conversion funnel: Corners → Shots → Shots on Target → Goals? Which teams excel at each stage? Does set-piece efficiency persist independently of open-play?
**Why interesting:** Decomposes set pieces like core fig4 decomposes open play. Corners → shots (delivery quality), shots → SoT (attacking header quality), SoT → goals (finishing). Each stage might have different persistence.
**Data:** HC, AC, Shots, Shots_OT, GF (but can't separate set-piece vs open-play shots perfectly).
**Observational unit:** Team-season.
**Feasibility:** Medium - can't perfectly separate set-piece shots from open play, but can use corners as proxy.

### 16. Yellow Card Accumulation and Suspension Effects
**Question:** Do teams suffer performance drops after key players accumulate 5/10/15 yellows (suspension thresholds)? Is squad depth visible in resilience to suspensions?
**Why interesting:** Suspension rules create natural experiments. No player-level data, but team-level yellow accumulation patterns might show effects.
**Data:** Yellow cards per team-match, rolling yellow totals.
**Observational unit:** Team-season around suspension thresholds.
**Feasibility:** Low - no player-level data, can't identify which players suspended.

### 17. Match Outcome Entropy / Predictability
**Question:** Are some leagues/seasons inherently more unpredictable? Entropy of result distribution (H/D/A probabilities).
**Why interesting:** Competitive balance vs unpredictability. High entropy = more random outcomes. Could relate to parity, variance, or genuine unpredictability.
**Data:** FTR by league-season.
**Observational unit:** League-season.
**Feasibility:** High - simple entropy calculation.

### 18. Rest and Performance Decay Curves (Non-linear)
**Question:** Is the rest-performance relationship linear or non-linear? Diminishing returns beyond 3-4 days? Cliff edge at 1-2 days?
**Why interesting:** New insight 4 (rest asymmetry) shows linear rest diff effect. But the *shape* of own-rest effect (not just differential) might be non-linear.
**Data:** Rest_Days (from fixture gaps).
**Observational unit:** Team-match.
**Feasibility:** Medium - Rest_Days has noise (unobserved cup matches).

---

## Evaluation Criteria (weights from mission)
| Criterion | Weight |
|-----------|--------|
| Novelty vs existing research | 25% |
| Potential football insight | 20% |
| Statistical feasibility | 20% |
| Data availability/quality | 15% |
| Visualization potential | 10% |
| Robustness/interpretability | 10% |

---

## Scoring Table (1-5 each)

| # | Question | Novelty | Insight | Feasibility | Data | Viz | Robustness | Weighted Score |
|---|----------|---------|---------|-------------|------|-----|------------|----------------|
| 1 | Corner Efficiency | 5 | 4 | 4 | 5 | 4 | 3 | **4.20** |
| 2 | Game State Effects | 4 | 5 | 3 | 3 | 5 | 3 | **3.85** |
| 3 | Draw Specialists | 4 | 4 | 5 | 5 | 4 | 4 | **4.10** |
| 4 | GD Efficiency (Pythag) | 3 | 4 | 5 | 5 | 4 | 4 | **3.95** |
| 5 | Home/Away Style Asymmetry | 4 | 4 | 4 | 5 | 4 | 3 | **3.95** |
| 6 | Competitive Balance | 4 | 5 | 5 | 5 | 4 | 4 | **4.45** |
| 7 | Selective vs Volume Shooting | 4 | 4 | 5 | 5 | 4 | 4 | **4.15** |
| 8 | Fouls Drawn as Attack Proxy | 4 | 4 | 4 | 5 | 3 | 3 | **3.85** |
| 9 | Red Card Timing | 3 | 3 | 3 | 3 | 3 | 2 | **2.85** |
| 10 | League Style Fingerprints | 4 | 4 | 5 | 5 | 5 | 4 | **4.30** |
| 11 | Points from Win/Loss Positions | 4 | 5 | 4 | 5 | 4 | 3 | **4.10** |
| 12 | Match Tempo & Predictability | 4 | 4 | 5 | 5 | 4 | 3 | **4.05** |
| 13 | Seasonal Trajectory Shapes | 4 | 4 | 3 | 4 | 5 | 3 | **3.75** |
| 14 | Disciplinary Game States | 4 | 4 | 3 | 3 | 4 | 3 | **3.50** |
| 15 | Corner-Shot-Goal Funnel | 4 | 4 | 3 | 4 | 4 | 2 | **3.50** |
| 16 | Yellow Accumulation/Suspensions | 2 | 3 | 2 | 2 | 3 | 2 | **2.35** |
| 17 | Match Outcome Entropy | 4 | 3 | 5 | 5 | 3 | 4 | **3.85** |
| 18 | Non-linear Rest Effects | 3 | 3 | 3 | 3 | 3 | 2 | **2.90** |

---

## Top 5 Selected for Implementation

### 1. **Competitive Balance Evolution** (Score: 4.45)
- **Novelty:** Not covered at all. Longitudinal league structure question.
- **Insight:** Addresses "is football becoming less competitive?" - major debate.
- **Feasibility:** Very high. Simple HHI, Gini, top/bottom gaps per league-season.
- **Data:** Complete team-season PPG for all 7×8=56 league-seasons.
- **Viz:** Small multiples by league, trend lines, HHI evolution.
- **Robustness:** Clear metrics, league-season unit avoids pseudoreplication.

### 2. **League Style Fingerprints** (Score: 4.30)
- **Novelty:** Not covered. Advanced B does team profiles, E does pooled vs within-league correlations, but not league-level style characterization.
- **Insight:** Tactical convergence/divergence debate. Distinctive league identities?
- **Feasibility:** High. Aggregate ratios (shots/fouls/corners per goal, per match).
- **Data:** Complete across all leagues/seasons.
- **Viz:** Radar charts, trajectories, clustering of league-seasons.
- **Robustness:** League-season unit (n=56) appropriate.

### 3. **Corner Efficiency and Set-Piece Profiles** (Score: 4.20)
- **Novelty:** Corners completely unexplored in 23 existing figures.
- **Insight:** Set pieces = distinct tactical phase. Specialists exist?
- **Feasibility:** High. Corners per game, shots per corner, goals per corner.
- **Data:** HC/AC available for all leagues/seasons.
- **Viz:** Funnel plots, persistence scatter, team profiles.
- **Robustness:** Exposure thresholds needed (min corners).

### 4. **Selective vs Volume Shooting Profiles** (Score: 4.15)
- **Novelty:** Core fig4 shows volume>accuracy>finishing, but not profile *interaction* or *persistence of profiles*.
- **Insight:** Is "selective shooting" (high accuracy, low volume) a sustainable alternative to volume?
- **Feasibility:** High. Quadrant analysis: Shots_PG × Shot_Accuracy.
- **Data:** Complete.
- **Viz:** Quadrant scatter with persistence arrows, profile stability.
- **Robustness:** Team-season unit, exposure thresholds.

### 5. **Points from Winning/Losing Positions** (Score: 4.10)
- **Novelty:** Core fig7 has HT comeback rates. This extends to "character" metric across all game states.
- **Insight:** "Mental strength" quantification. Persistence of clutch/anti-clutch.
- **Feasibility:** High. HTR→FTR transitions already computed.
- **Data:** Complete HTR/FTR.
- **Viz:** Points won from behind vs dropped when ahead, persistence.
- **Robustness:** Crude game state (HT only) but valid for all matches.

---

## Honorable Mentions (Strong but Not Selected)

- **Draw Specialists** (4.10) - similar to close-match dependence, could be combined
- **Home/Away Style Asymmetry** (3.95) - partially covered by HA components (new insight 2)
- **Match Tempo** (4.05) - interesting but might correlate strongly with league/team strength
- **GD Efficiency** (3.95) - classic Pythagorean, well-studied elsewhere

---

## Rejected / Deferred

- **Red Card Timing** (2.85) - insufficient timing data
- **Yellow Accumulation** (2.35) - no player-level data
- **Non-linear Rest** (2.90) - rest data too noisy (unobserved cups)
- **Corner Funnel** (3.50) - can't separate set-piece vs open-play shots cleanly
- **Disciplinary Game States** (3.50) - HTR too crude
- **Seasonal Trajectory Shapes** (3.75) - low n per team-season for curve fitting