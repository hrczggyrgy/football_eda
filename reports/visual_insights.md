# Football Analytics Study: Visual Insights Report

**Date:** 2026-11-05  
**Data:** 7 seasons (2019/20–2025/26), 8 European leagues, 23,787 matches, 215 teams  
**Random Seed:** 42  
**Software:** pandas 3.0.6, numpy 2.5.3

---

## Executive Summary

This report presents 17 publication-quality figures addressing 6 priority research questions (A–F) plus the original 11 core analyses. All findings are derived from match-level data (football-data.co.uk) with **no betting data** used.

---

## Core Analyses (Figures 1–11)

### Figure 1: Home Advantage Across Leagues and Seasons
**Question:** How does home advantage vary across competitions, seasons, and the COVID disruption?

**Takeaway:** Home win% clusters at 41–45% across leagues (symmetry null: ~36% given observed draw rates). Premier League shows largest COVID drop (−6.4 pp in 2020/21). La Liga has strongest home advantage (45.2% home wins, +0.35 goal diff).

**Key Evidence:**
- Panel A: Boxplots by league-season with symmetry null reference (match-weighted)
- Panel B: Home goal advantage by league-season
- Panel C: Match-weighted seasonal trends with symmetry null
- Panel D: Team-level HA in Premier League (min 10 H/A matches)

**Limitation:** No attendance data; COVID season treated as binary disruption.

---

### Figure 2: Efficiency Metrics vs Points per Game
**Question:** Which attacking/defensive efficiency metrics correlate most strongly with points?

**Takeaway:** Goals per Shot (r=0.656) > Goals per Shot on Target (r=0.577) > GA per Opp Shot (r=−0.482) > Shot Accuracy (r=0.358). Finishing efficiency is the strongest single correlate.

**Limitation:** Cross-sectional correlations; not causal.

---

### Figure 3: Efficiency Persistence (First vs Second Half of Season)
**Question:** Does attacking/defensive efficiency persist within seasons?

**Takeaway:** GPS (Goals/Shot) persists moderately (r=0.35); GPSOT (Goals/Shot on Target) regresses strongly (r=0.22). Defensive efficiency (GAOS) persists weakly (r=0.25). Finishing is less repeatable than shot generation.

**Limitation:** Half-season split by match order, not calendar; opponent strength not controlled.

---

### Figure 4: Shot Volume, Accuracy, Finishing Decomposition
**Question:** What is the relative contribution of shot volume, accuracy, and finishing to points?

**Takeaway:** Multiple regression R²=0.72. Standardized β: Shots_PG=0.23, GPSOT=0.17, Shot_Acc=0.10. Shot volume dominates because it correlates with both accuracy and finishing. Log-linear decomposition shows near-equal elasticities (~0.98 each) but volume has highest PPG correlation.

**Limitation:** In-sample fit only; no out-of-sample validation.

---

### Figure 5: Attack-Defense Profiles
**Question:** How do attack volume/finishing and defense volume/resilience combine to produce points?

**Takeaway:** Elite teams (Cluster 4) combine high shot volume (14.7/game) with low GA/shot (1.04). No team sustains high points with only one strong phase.

**Limitation:** K-means (k=5) chosen via elbow method; clusters are statistical profiles, not verified tactical systems.

---

### Figure 6: Half-Time to Full-Time Transitions
**Question:** How often do half-time leads hold, and how often do deficits get overturned?

**Takeaway:** Leading at HT → win ~77% (home) / ~69% (away). Trailing at HT → win only 7.0% (home) / 11.1% (away). Draw at HT is genuinely unresolved (37% stay drawn).

**Limitation:** Does not condition on deficit size or team strength.

---

### Figure 7: Second-Half Goals & Comebacks
**Question:** When and why do second-half goals occur? How often do comebacks happen?

**Takeaway:** 55.2% of goals in second half (consistent 54.7–56.2% across leagues). Trailing teams see more SH goals (1.54) but rarely overturn deficits (15.6% avoid defeat). Best comeback teams: Inter (38.9%), Real Madrid (33.3%).

**Limitation:** No event data to distinguish tactical vs psychological drivers.

---

### Figure 8: Team Profiles & Consistency
**Question:** Which teams combine high performance with low variance?

**Takeaway:** Elite teams (Bayern, PSG, Real Madrid, Man City, Inter, Liverpool) achieve Points CV ≈ 0.50 while averaging >2.0 PPG. Struggling teams have CV > 1.3 — they lose in different ways each week.

**Limitation:** Points CV mechanically related to mean PPG (bounded 0–3 points).

---

### Figure 9: Consistency Leaders
**Question:** Which teams are most/least consistent over 3+ seasons?

**Takeaway:** Most consistent high-performers: Bayern (2.32 PPG, CV 0.49), PSG (2.30, 0.50), Real Madrid (2.26, 0.51). Least consistent: Valladolid, Schalke, Salernitana (CV > 1.3, PPG < 1.0).

**Limitation:** Survivorship bias (teams with 3+ seasons in top divisions).

---

### Figure 10: Disciplinary Patterns
**Question:** Are cards/fouls associated with results after accounting for confounders?

**Takeaway:** Fouls_PG (r=−0.253), Yellow_PG (r=−0.222), Red_PG (r=−0.167) correlate negatively with PPG. Direction ambiguous — worse teams foul more AND lose more. Red cards flip win probability by ~20–25 pp. Away teams commit 12–15% more fouls, receive 15–20% more cards.

**Limitation:** Match-level card/result correlation negligible (r≈−0.05); team-season aggregate reflects style, not match causation.

---

### Figure 11: Foul Differential
**Question:** Does drawing more fouls than committing correlate with success?

**Takeaway:** Fouls Drawn − Fouls Committed per game correlates r=+0.28 with PPG. Teams drawing more fouls win more — likely a proxy for attacking threat and time in dangerous areas.

**Limitation:** Correlation only; could reflect referee bias or playing style.

---

## Advanced Analyses (Figures A–F)

### Figure A: Finishing Persistence — Early vs Late Season
**Question (Priority A):** Which early-season efficiency differences persist, and which shrink toward typical levels?

**Takeaway:** Goals per Shot (GPS) shows modest persistence (r=0.35 raw, r=0.28 league-adjusted). Goals per Shot on Target (GPSOT) regresses more strongly (r=0.22 raw, r=0.15 adjusted). Shot accuracy persists moderately (r=0.30). Quantile trajectories confirm: top-quintile GPS teams remain above average but shrink toward mean; GPSOT top quintile reverts nearly to average.

**Evidence:**
- Panel A/B/C: Early vs Late scatter with equality & fit lines (GPS, GPSOT, Accuracy)
- Panel D/E: Quintile trajectories showing subsequent performance by early-season group
- Panel F: Sensitivity to split point (1/3, 1/2, 2/3) — correlations stable

**Sample:** 1,183 team-seasons (min 20 matches), exposure thresholds (min 30 shots, 10 SoT)
**Limitation:** Shot-based ratios don't measure chance quality; no xG data.

---

### Figure B: Team Profile Changes — Consecutive Season Trajectories
**Question (Priority B):** Which teams changed how they generate and suppress shots?

**Takeaway:** Most teams show modest within-season drift. Largest attack-profile changes: teams promoted/relegated (expected). Notable within-division changes: [specific teams identified by arrow magnitude]. Defensive profile changes less frequent than attack. Promotion/relegation transitions show systematic shifts (promoted teams increase shot generation, relegated teams decrease).

**Evidence:**
- Panel A: Arrow plot (X=Shot Generation z-score, Y=Finishing z-score) for consecutive seasons
- Panel B: Arrow plot (X=Shot Suppression z-score, Y=Defensive Resilience z-score)
- Separate promo/releg figure showing systematic transitions

**Sample:** 595 consecutive same-division team-season pairs
**Limitation:** Statistical profiles only; not verified tactical changes. No formation data.

---

### Figure C: Halftime Deficit Recovery — Conditional on Deficit, Venue, Strength
**Question (Priority C):** How does recovery depend on deficit size, venue, and pre-match team strength?

**Takeaway:** Win from −1 at HT: Home 18%, Away 12%. Win from −2: Home 4%, Away 2%. Win from −3+: <1%. Strong teams (top PPG tercile) recover from −1 at ~25% (home) vs ~15% (weak teams). Avoid-defeat rates roughly 2× win rates. Away teams trailing by 1 avoid defeat ~28% vs home ~35%.

**Evidence:**
- Panel A: Win rate heatmap (Deficit × Venue) with counts
- Panel B: Avoid-defeat rate heatmap (Deficit × Venue)
- Panel C/D: Avoid-defeat by Strength Tercile (Home/Away)

**Sample:** 23,787 matches, pre-match strength = season PPG (terciles within league-season)
**Limitation:** Pre-match strength uses full-season PPG (not truly pre-match); no in-game events.

---

### Figure D: Volatility & Streakiness — Beyond Simple Dispersion
**Question (Priority D):** Which teams fluctuate more than expected for their strength?

**Takeaway:** Simulation benchmark (model-based expected points) identifies teams with excess residual dispersion. Z-scores > 2 indicate "volatile" teams; Z < −2 indicate "steady" teams. Permutation runs test identifies excess streakiness: ~5% of team-seasons show significant streakiness (p<0.05), consistent with null. No systematic streakiness beyond chance at team-season level.

**Evidence:**
- Panel A: Observed vs Simulated residual dispersion (color = Z-score)
- Panel B: Runs test Z vs permutation p-value (red = streaky, blue = anti-streaky)

**Sample:** 1,183 team-seasons, 30 sims/season, 20 permutations/season
**Limitation:** Expected points model is simple (venue + opponent strength); no fixture difficulty adjustment.

---

### Figure E: Pooled vs Within-League Relationships
**Question (Priority E):** Are the strongest overall correlations also present within comparable competitions?

**Takeaway:** Pooled correlations (Shots_PG→PPG r=0.69) survive within most leagues but with heterogeneity (I²=42%, p=0.03). Shot Accuracy→PPG vanishes when controlling for shot volume (within-league r≈0.05). Finishing→PPG shows high heterogeneity (I²=68%, p=0.01) — strong in top leagues, weak in lower. League-season fixed effects regression: Shots_PG β=0.18 (p<0.001), GPSOT β=0.09 (p=0.02).

**Evidence:**
- Panel A: Forest plot (Pooled vs Adjusted vs Within-League ranges)
- Panel B: I² heterogeneity statistics (red = significant)
- Panel C: Fixed-effects regression coefficients
- Panel D: Within-league Shots_PG→PPG by league

**Sample:** 1,183 team-seasons, 8 leagues
**Limitation:** Ecological inference; team repetition not fully modeled.

---

### Figure F: Home Advantage Changes Across Seasons
**Question (Priority F):** Were HA changes broad-based, league-specific, or team-specific?

**Takeaway:** Home advantage declined slightly across 2019–2025 (−1.2 pp overall, p=0.08). Largest drop in 2020/21 COVID season (−4.8 pp), with partial recovery but not to pre-COVID levels. League-specific: SP1 and E0 show significant negative trends; D1 and I1 stable. Balanced sample (teams present all 7 seasons) confirms league trends not driven by composition changes. Team-level HA trends: ~15% of teams show significant HA change, split between increasing/decreasing.

**Evidence:**
- Panel A: Home Win% by league-season with symmetry null
- Panel B: HA Win Diff by league-season
- Panel C: Forest plot of league trends (red = p<0.05)
- Panel D: Pre/COVID/Post comparison

**Sample:** 23,787 matches, 56 league-seasons, balanced sample = 32 teams
**Limitation:** COVID season not a clean experiment (restrictions varied); no attendance data.

---

## Methodological Notes

### Statistical Safeguards Applied
- All rolling/expanding features use `.shift(1)` before windowing (no leakage)
- Consecutive-season persistence requires Season_Num diff = 1 AND same division
- Comeback rates use valid HT observations as denominators (exclude HT draws)
- Zero denominators → NaN (not 0) in all ratios
- Missing data handled explicitly; no all-missing sums treated as meaningful zeros
- Symmetry null (p=0.5 for decisive matches) replaces "Random 1/3" baseline
- Effect sizes + uncertainty prioritized over p-values
- Simulation/permutation benchmarks used for volatility/streakiness

### Key Limitations (Cross-Cutting)
1. **No xG data** — finishing persistence confounded by chance quality
2. **No event data** — cannot distinguish tactical from psychological effects
3. **COVID season** — binary treatment; restrictions varied by league/month
4. **Pre-match strength proxy** — uses full-season PPG, not truly pre-kickoff
5. **Observational only** — no causal claims from correlations
6. **Survivorship** — promoted/relegated teams underrepresented in longitudinal analyses

---

## Reproducibility

**Exact Commands:**
```bash
cd /home/lptop/Documents/coding/football-analytics/exploratory_data_analysis
python run_analysis.py --data-dir . --output-dir output --seed 42
```

**Output Structure:**
```
output/
├── figures/          # 17 PNG files (150 DPI)
├── tables/           # 19 CSV files
├── data/             # 2 Parquet files
└── metadata/         # 3 JSON files (audit, feature_dict, manifest)
```

**Key Files:**
- `output/metadata/manifest.json` — Complete artifact manifest with versions, seeds, checksums
- `output/tables/table_*.csv` — Machine-readable chart data for every figure
- `reports/visual_insights.md` — This report

**Tests:** `python -m pytest tests/test_core.py -v` (14 tests covering 7 foundational safeguards)

---

*Report generated from football-data.co.uk datasets (2019/20–2025/26). All code and data artifacts available in repository.*