# Fresh Research Candidates for Football Analytics Study

## 15 Candidate Research Questions

### 1. Referee Characteristics and Potential Biases
**Question:** Do individual referees exhibit measurable, persistent differences in officiating patterns (cards, fouls, home-away asymmetry) after controlling for league, season, team tendencies, and match context?
**Why interesting:** Referees are the only non-team actors with systematic influence on matches. If referee effects exist beyond noise, they reveal structural biases or stylistic differences. English leagues (E0-E3) have full referee coverage across 7 seasons (~14,000 matches, 400+ referees).
**Data:** Referee, HY/AY, HR/AR, HF/AF, FTR, HTR, Div, Season, teams. 100% coverage in E0-E3.
**Observational unit:** Referee-season or referee-match.
**Statistical approach:** Mixed-effects models with referee random effects; league-season fixed effects; team tendency controls; bootstrap CIs for referee-specific estimates.
**Confounders:** Team disciplinary tendencies, match importance, score state, league/season trends, home advantage.
**Visualization:** Referee "fingerprint" scatterplots (cards vs fouls vs home bias); caterpillar plots of referee random effects; home-away asymmetry by referee.
**Limitations:** Only English leagues; no player-level data; observational not causal.
**Feasibility:** High (data complete for E0-E3).

### 2. Draw Specialists and Match-Outcome Structure
**Question:** Do some teams systematically draw more (or fewer) matches than their underlying strength predicts? Is "draw propensity" a persistent team characteristic distinct from win/loss ability?
**Why interesting:** Draws are a distinct outcome from wins/losses. A team with 15 draws/10 wins/13 losses (43 pts) has a fundamentally different profile than 5 draws/15 wins/18 losses (50 pts) despite similar GD. Draw tendency may reflect risk aversion, defensive solidity, or inability to close games.
**Data:** FTR, HTR, team strength (PPG, GD, shots). All leagues/seasons complete.
**Observational unit:** Team-season.
**Statistical approach:** Expected draws from strength model (quadratic in PPG); residuals = draw propensity; season-to-season persistence; league comparisons; profile characterization (defensive/attacking stats of draw specialists).
**Confounders:** Team strength (non-linear), league draw rates, schedule strength, game state.
**Visualization:** Observed vs expected draws scatter; draw propensity persistence; defensive/attacking profiles of high/low draw teams.
**Limitations:** Crude strength model; draws may be strength artifact.
**Feasibility:** High (simple calculation, complete data).

### 3. Expected Points vs Actual Points (Pythagorean Residuals)
**Question:** Which teams consistently over/underperform their goal-difference-based expected points? Is "clutch" GD-to-points conversion persistent or noise? Do shot dominance and GD tell different stories?
**Why interesting:** Football Pythagorean expectation is standard. Residuals capture "luck," "clutch," or tactical efficiency in converting dominance to points. Existing close-match analysis (New Insight 3) covers 1-goal wins; this is broader.
**Data:** GF, GA, Points per team-season. Shot dominance available for comparison.
**Observational unit:** Team-season.
**Statistical approach:** Pythagorean exponent optimization; residual calculation; persistence (half-season, season-to-season); incremental value of shot dominance over GD; outlier identification with uncertainty.
**Confounders:** Schedule strength, blowout vs close games, league scoring environment.
**Visualization:** Pythagorean residual vs PPG; residual persistence; shot dominance vs GD residual comparison; team trajectories.
**Limitations:** Exponent choice arbitrary; 7 seasons limits persistence power.
**Feasibility:** High (standard calculation).

### 4. Match Scheduling and Performance (Kickoff Time Effects)
**Question:** Does kickoff timing correspond to meaningful differences in match statistics and outcomes? Are there weekday vs weekend effects? Does performance vary across schedule periods?
**Why interesting:** Broadcasting drives kickoff times. If early/late kickoffs systematically affect play (tempo, goals, home advantage), it reveals scheduling as a competitive factor. All leagues have 100% Time coverage.
**Data:** Time (kickoff), Date, all match stats. 23,787 matches with times.
**Observational unit:** Match.
**Statistical approach:** Hour-of-day effects on goals, shots, cards, home win rate; weekday vs weekend; month/period effects; control for league, season, team strength; interaction with venue.
**Confounders:** Team quality (big teams get prime slots), competition importance, TV selection bias, weather/season.
**Visualization:** Hourly profiles of goals, shots, home win%; weekday/weekend comparison; heatmap of hour x venue.
**Limitations:** Selection bias (strong teams → prime time); no attendance data; confounding by team quality.
**Feasibility:** High (Time 100% coverage).

### 5. Match Tempo and Statistical Structure
**Question:** Do high-tempo matches (many total shots) produce more or less predictable outcomes? Is tempo a persistent team characteristic? How does tempo relate to goal frequency and result distributions?
**Why interesting:** Some matches are open (30+ shots), some cagey (15 shots). Tempo may correlate with variance, unpredictability, or specific team styles. Total shots = HS+AS available for all matches.
**Data:** HS, AS, FTHG, FTAG, FTR. All leagues/seasons.
**Observational unit:** Match (tempo) and team-season (tempo profile).
**Statistical approach:** Tempo bins → outcome distributions, goal variance; team-season avg tempo → style profile; tempo persistence; interaction with team strength.
**Confounders:** Team quality (strong teams create more shots), game state, league style.
**Visualization:** Tempo vs goal variance; tempo vs result entropy; team tempo profiles; high/low tempo match examples.
**Limitations:** Shots ≠ possession; tempo conflates both teams' styles.
**Feasibility:** High (simple calculation).

### 6. Tactical Matchup Effects (Style Interactions)
**Question:** Do particular combinations of team characteristics produce systematically different outcomes beyond what main effects predict? E.g., high-volume attacks vs strong defenses; shot suppression vs accurate shooting.
**Why interesting:** Football is interactive. A high-volume attack may struggle against a low-block defense but thrive against open teams. Existing analyses treat teams independently; matchups reveal tactical rock-paper-scissors.
**Data:** Team-season styles (Shots_PG, Shot_Accuracy, GA_Per_Opp_Shot, etc.); match-level pairings.
**Observational unit:** Match (paired team styles) or team-season vs opponent-style distribution.
**Statistical approach:** Interaction terms in outcome models (Shot_Dom × Opp_Def_Style); matched pair analysis; quadrant analysis of style combinations; expected vs actual results by matchup type.
**Confounders:** Team quality correlation (good teams have good attack AND defense), league, venue, schedule.
**Visualization:** Style matchup heatmaps (attack style × defense style → PPG/win%); interaction coefficient plots; matchup-specific residuals.
**Limitations:** Matchup n small per combination; style measures noisy; ecological fallacy risk.
**Feasibility:** Medium (needs careful pairing, sufficient n per cell).

### 7. League-Specific Statistical-Result Relationships
**Question:** Do the relationships between underlying statistics (shots, accuracy, finishing) and results (PPG) differ structurally across leagues? Are some leagues more "efficient" at converting shots to points?
**Why interesting:** Existing pooled vs within-league analysis (Advanced E) compares correlations. This goes deeper: different functional forms, different rate-limiting factors, different competitive structures.
**Data:** Team-season stats and PPG by league. 8 leagues × 7 seasons.
**Statistical approach:** League-specific regression slopes (Shots→PPG, Acc→PPG, GPSOT→PPG); heterogeneity tests; league "conversion efficiency" rankings; interaction with competitive balance.
**Confounders:** League quality, sample size per league-season, measurement differences (Serie A shot accuracy artifact).
**Visualization:** League-specific coefficient forest plots; conversion efficiency rankings; slope vs league characteristics.
**Limitations:** Only 7 seasons per league; Serie A data artifact confounds.
**Feasibility:** High (league-season aggregation straightforward).

### 8. Game State Effects on Match Statistics
**Question:** How do shot volume, accuracy, fouls, and corners change conditional on game state (leading/drawing/trailing) and time remaining? Do "score effects" differ by league, team style, or venue?
**Why interesting:** Existing HT-FT analysis looks at results, not *how* teams play in different states. Score effects (trailing teams shoot more but less accurately) are well-known but not quantified with shot/foul/corner data in this dataset.
**Data:** HTR as game state proxy; match stats; venue. HTR available for all matches.
**Observational unit:** Team-match with HT result as state.
**Statistical approach:** Compare stats by HT_Result (W/D/L) and venue; control for team strength; league differences; interaction with deficit size (HT_GD).
**Confounders:** Team quality (strong teams lead more), HT score reflects first-half play, not just state.
**Visualization:** Score effect profiles (shots, fouls, corners by state); venue × state interactions; league differences; deficit-size gradients.
**Limitations:** HTR is crude proxy (no minute-level state changes); first-half play determines HT state (endogeneity).
**Feasibility:** High (HTR 100% coverage).

### 9. Seasonal Trajectory Shapes (Non-Linear Within-Season Patterns)
**Question:** Do team performances follow non-linear trajectories within seasons (U-shape, inverted-U, steady improvement, early peak)? Are shapes associated with promotion/relegation, European competition, or managerial changes?
**Why interesting:** Existing analyses use early vs late split or rolling 5-game. The *shape* of the trajectory might be informative. Newly promoted teams might improve steadily; relegated teams might decline non-linearly.
**Data:** Match-order PPG, Shot_Dominance by team-season. Rolling features available.
**Observational unit:** Team-season trajectory.
**Statistical approach:** Fit polynomial/spline trajectories per team-season; classify shapes (k-means on coefficients); associate shapes with context (promotion, Europe, manager); test shape persistence.
**Confounders:** Small n per team-season (38 matches max); noise in match results; schedule strength variation.
**Visualization:** Trajectory shape taxonomy; average trajectories by cluster; context associations.
**Limitations:** Low n per team-season for reliable curve fitting; schedule strength confounds.
**Feasibility:** Medium (noisy individual trajectories).

### 10. Fouls Drawn as Attacking Threat Proxy
**Question:** Are fouls drawn (Fouls_Against) a better indicator of dangerous attacking play than shots? Does "winning fouls in dangerous areas" persist and predict goals/points?
**Why interesting:** Foul differential analysis (Core Fig 11) shows fouls drawn - fouls committed correlates with PPG (r=0.28). But fouls drawn *alone* as attacking metric is unexplored. Fouls often occur in final third during dangerous attacks.
**Data:** Fouls, Fouls_Against, GF, Shots, team-season.
**Observational unit:** Team-season or team-match.
**Statistical approach:** Fouls_Drawn_PG vs GF/Points; persistence; incremental over shots; foul location proxy (fouls when trailing vs leading); set-piece fouls via corners.
**Confounders:** Playing style (possession teams draw more fouls), refereeing, simulation/diving, tactical fouling by opponents.
**Visualization:** Fouls drawn vs goals/points; persistence scatter; attacking profiles of high-foul-drawn teams.
**Limitations:** No foul location data; fouls conflate dangerous attacks with tactical fouling by opponents; simulation.
**Feasibility:** High (data complete).

### 11. Match Outcome Entropy and Unpredictability
**Question:** Are some leagues/seasons inherently more unpredictable? Entropy of result distribution (H/D/A probabilities) as unpredictability measure.
**Why interesting:** Competitive balance vs unpredictability. High entropy = more random outcomes. Could relate to parity, variance, or genuine unpredictability. Simple to compute.
**Data:** FTR by league-season. Complete.
**Observational unit:** League-season.
**Statistical approach:** Shannon entropy of result distribution; trends over time; correlation with balance metrics (HHI, Gini); relationship to betting market efficiency (excluded but conceptually).
**Confounders:** Number of teams (fixed at 20/24), league structure, draw rate.
**Visualization:** Entropy trends by league; entropy vs balance metrics; high/low entropy seasons.
**Limitations:** Only 56 league-seasons; entropy driven heavily by draw rate; no betting data for validation.
**Feasibility:** High (simple calculation).

### 12. Home/Away Style Asymmetry Beyond Results
**Question:** Do teams systematically change their *style* (shots, corners, fouls, discipline, shot accuracy) home vs away, beyond just winning more at home? Some teams may play possession football at home, counter-attack away.
**Why interesting:** HA components analysis (New Insight 2) shows shot generation/suppression differences. But what about fouls, corners, discipline, shot *accuracy*? Style asymmetry reveals tactical flexibility.
**Data:** All team-match stats split by venue. Already computed in HA components.
**Observational unit:** Team-season home/away splits.
**Statistical approach:** Paired home-away differences per metric; persistence of asymmetry; league differences; cluster teams by asymmetry profile; relation to overall HA.
**Confounders:** Opponent quality differs home vs away (schedule), score state effects, referee bias.
**Visualization:** Home vs away style scatterplots; asymmetry profiles; league differences; team examples.
**Limitations:** Partially covered by HA components (New Insight 2); schedule confounding.
**Feasibility:** High (data ready).

### 13. Red Card Timing and Asymmetric Impact
**Question:** When in matches do red cards occur? Does early red card (1st half) have different impact than late (2nd half)? Does impact differ by score state at time of dismissal?
**Why interesting:** Core Fig 10 shows red cards flip win probability ~20-25pp. But timing matters - early red forces tactical reshuffle for longer; late red may just see out game. No timing data, but HT vs FT red card presence is a proxy.
**Data:** HR, AR, HTR, FTR. Can infer if red occurred by HT (if team had red at HT) vs only at FT.
**Observational unit:** Match with red card.
**Statistical approach:** Compare HT-red vs FT-only-red impact on outcome; stratify by HT score state; home vs away asymmetry.
**Confounders:** Red card *causes* score change (endogeneity); teams playing with 10 men change style.
**Visualization:** Red card timing impact by score state; home/away asymmetry; HT vs FT red comparison.
**Limitations:** No exact timing; HT red proxy crude; endogeneity severe.
**Feasibility:** Medium (limited by proxy quality).

### 14. Yellow Card Accumulation and Suspension Effects
**Question:** Do teams suffer performance drops after key players accumulate 5/10/15 yellows (suspension thresholds)? Is squad depth visible in resilience to suspensions?
**Why interesting:** Suspension rules create natural experiments. No player-level data, but team-level yellow accumulation patterns might show effects.
**Data:** Yellow cards per team-match, rolling yellow totals.
**Observational unit:** Team-season around suspension thresholds.
**Statistical approach:** Event study around 5th/10th yellow milestones; rolling yellow count vs subsequent performance; squad depth proxy (squad size not available but consistency metrics might proxy).
**Confounders:** No player-level data; yellow accumulation correlates with team style/quality; fixture congestion.
**Visualization:** Performance before/after yellow milestones; team resilience ranking.
**Limitations:** No player data = fatal flaw; cannot identify which players suspended.
**Feasibility:** Low (no player-level data).

### 15. Non-Linear Rest Effects and Performance Decay Curves
**Question:** Is the rest-performance relationship linear or non-linear? Diminishing returns beyond 3-4 days? Cliff edge at 1-2 days?
**Why interesting:** New Insight 4 (Rest Asymmetry) shows linear rest diff effect. But the *shape* of own-rest effect (not just differential) might be non-linear.
**Data:** Rest_Days from fixture gaps (noisy due to unobserved cups).
**Observational unit:** Team-match.
**Statistical approach:** Spline/polynomial of Rest_Days on Points/Shot_Dom; threshold effects; interaction with venue and fixture congestion.
**Confounders:** Rest_Days noisy (unobserved cup matches, international breaks); team quality; schedule strength.
**Visualization:** Rest-performance curve with confidence bands; threshold detection; venue interaction.
**Limitations:** Rest data too noisy (unobserved cups); selection bias (cup teams rotate squads).
**Feasibility:** Low/Medium (data quality issues).

---

## Evaluation Scorecard

| # | Question | Novelty (25%) | Insight (20%) | Feasibility (20%) | Data (15%) | Viz (10%) | Robustness (10%) | **Weighted Score** |
|---|----------|---------------|---------------|-------------------|------------|-----------|------------------|-------------------|
| 1 | Referee Characteristics | 5 | 4 | 4 | 3 | 4 | 3 | **4.00** |
| 2 | Draw Specialists | 4 | 4 | 5 | 5 | 4 | 4 | **4.25** |
| 3 | Pythagorean Residuals | 3 | 4 | 5 | 5 | 4 | 4 | **3.95** |
| 4 | Match Scheduling | 4 | 4 | 4 | 5 | 4 | 3 | **3.95** |
| 5 | Match Tempo | 4 | 4 | 5 | 5 | 4 | 3 | **4.10** |
| 6 | Tactical Matchups | 5 | 5 | 3 | 4 | 4 | 2 | **3.85** |
| 7 | League-Specific Relationships | 4 | 4 | 4 | 5 | 4 | 3 | **3.90** |
| 8 | Game State Effects | 4 | 5 | 4 | 4 | 5 | 3 | **4.10** |
| 9 | Seasonal Trajectories | 4 | 4 | 3 | 4 | 5 | 3 | **3.75** |
| 10 | Fouls Drawn as Attack Proxy | 4 | 4 | 4 | 5 | 3 | 3 | **3.85** |
| 11 | Outcome Entropy | 4 | 3 | 5 | 5 | 3 | 4 | **3.85** |
| 12 | Home/Away Style Asymmetry | 3 | 4 | 4 | 5 | 4 | 3 | **3.70** |
| 13 | Red Card Timing | 3 | 3 | 3 | 3 | 3 | 2 | **2.85** |
| 14 | Yellow Accumulation | 2 | 3 | 2 | 2 | 3 | 2 | **2.30** |
| 15 | Non-Linear Rest | 3 | 3 | 3 | 3 | 3 | 2 | **2.85** |

---

## Top 5 Selected for Implementation

### 1. **Draw Specialists and Match-Outcome Structure** (Score: 4.25)
- **Novelty:** Not covered at all. Draw propensity as distinct from win/loss ability is unexplored.
- **Insight:** Reveals "risk profile" dimension of team performance. Are draws a skill?
- **Feasibility:** Very high. Simple expected-draw model from PPG; residuals = draw propensity.
- **Data:** Complete across all leagues/seasons.
- **Viz:** Observed vs expected draws; persistence; defensive profiles of draw specialists.
- **Robustness:** Clear metrics; team-season unit avoids pseudoreplication.

### 2. **Match Tempo and Statistical Structure** (Score: 4.10)
- **Novelty:** Total shots as tempo measure unexplored. Tempo-predictability link novel.
- **Insight:** Does open play increase variance? Do some teams control tempo?
- **Feasibility:** High. HS+AS = tempo. Match and team-season levels.
- **Data:** Complete.
- **Viz:** Tempo vs goal variance; tempo vs result entropy; team tempo profiles.
- **Robustness:** Simple metric; match-level n=23,787.

### 3. **Game State Effects on Match Statistics** (Score: 4.10)
- **Novelty:** Score effects on *statistics* (not results) unexplored in this dataset.
- **Insight:** Quantifies "score effects" - how trailing teams shoot more/foul more.
- **Feasibility:** High. HTR as game state proxy for all matches.
- **Data:** Complete HTR coverage.
- **Viz:** Score effect profiles by metric; venue × state; deficit size gradients.
- **Robustness:** Crude proxy but valid for all matches; control for team strength.

### 4. **Referee Characteristics and Potential Biases** (Score: 4.00)
- **Novelty:** Referee-level analysis completely new. Only English leagues but 14k matches.
- **Insight:** Structural officiating differences? Home bias variation by referee?
- **Feasibility:** High for E0-E3. Mixed-effects models with controls.
- **Data:** 100% referee coverage in E0-E3 (4 leagues × 7 seasons).
- **Viz:** Referee fingerprints; caterpillar plots; home-away asymmetry by referee.
- **Robustness:** Control for team tendencies, league, season, match context.

### 5. **Expected Points vs Actual Points (Pythagorean Residuals)** (Score: 3.95)
- **Novelty:** Broader than close-match dependence. Shot dominance vs GD comparison novel.
- **Insight:** "Clutch" conversion persistence? Shot dominance vs GD residuals?
- **Feasibility:** Very high. Standard Pythagorean calculation.
- **Data:** Complete GF, GA, Points, Shot_Dominance.
- **Viz:** Residual vs PPG; persistence; shot dominance vs GD residual comparison.
- **Robustness:** Exponent optimization; bootstrap CIs; incremental value test.

---

## Honorable Mentions (Strong but Not Selected)

- **Tactical Matchups** (3.85) - highest insight but feasibility concerns (small n per combination)
- **Fouls Drawn as Attack Proxy** (3.85) - interesting but fouls confounded by opposition tactics
- **Outcome Entropy** (3.85) - simple but may not add beyond competitive balance
- **League-Specific Relationships** (3.90) - partially covered by Pooled vs Within-League (Advanced E)

---

## Rejected / Deferred

- **Red Card Timing** (2.85) - insufficient timing data, endogeneity
- **Yellow Accumulation** (2.30) - no player-level data
- **Non-Linear Rest** (2.85) - rest data too noisy (unobserved cups)
- **Seasonal Trajectories** (3.75) - low n per team-season for curve fitting
- **Home/Away Style Asymmetry** (3.70) - partially covered by HA Components (New Insight 2)

---

## Implementation Plan

Create new modules:
- `src/analysis/fresh_insights.py` — 5 new analysis functions
- `src/visualization/fresh_insight_plots.py` — 5 new figure functions
- `docs/fresh_insights_report.md` — Research report

Integrate into `run_analysis.py` with `--skip-fresh` flag.

Each analysis must:
- Follow existing safeguards (no leakage, consecutive seasons, valid denominators)
- Produce machine-readable CSV for each figure
- Include uncertainty quantification (bootstrap CIs, standard errors)
- Test sensitivity to thresholds and specifications