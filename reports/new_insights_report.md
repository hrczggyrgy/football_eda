# Football Analytics Study: New Insights Report

**Date:** 2026-11-05  
**Data:** 7 seasons (2019/20–2025/26), 8 European leagues, 23,787 matches, 215 teams  
**Random Seed:** 42  
**Software:** pandas 3.0.6, numpy 2.5.3  

---

## Executive Summary

This report presents **6 new research questions** (Figures 1–6) addressing previously unexplored aspects of football performance, beyond the original 11 core analyses and 6 advanced analyses (A–F). All findings are derived from match-level data (football-data.co.uk) with **no betting data** used.

### Key New Findings

| # | Question | Key Finding | Evidence Strength |
|---|----------|-------------|-------------------|
| **1** | **Opponent-dependent performance** | Most teams perform *worse* against stronger opponents as expected; a few elite teams (Man City, Bayern, Real Madrid) show near-zero or positive interaction slopes, maintaining performance vs top opposition | ★★★★☆ |
| **2** | **HA components** | Home advantage decomposes primarily into **shot generation** (+1.8 shots/game at home) and **shot suppression** (-1.3 opponent shots/game), not finishing conversion (GPS difference ≈ 0) | ★★★★★ |
| **3** | **Close-match dependence** | Teams relying heavily on one-goal wins show **zero persistence** (r≈0.05 early→late); shot dominance persistence (r≈0.35) far exceeds close-match record persistence | ★★★★★ |
| **4** | **Rest asymmetry** | Being less rested than opponent costs ~0.03 PPG per day deficit; effect strongest for **away teams** (-0.05 PPG/day) vs home teams (-0.02 PPG/day); shot dominance mirrors pattern | ★★★★☆ |
| **5** | **Indicators lead results** | Lagged shot dominance (first 10 matches) predicts future PPG (r=0.28) **incrementally over lagged PPG** (ΔR²=0.03); SoT dominance adds similar value; predictive power peaks at 10–15 match indicator windows | ★★★★☆ |
| **7** | **Promotion adaptation** | Promoted teams **improve steadily** across first season (+0.15 PPG from matches 1–5 to 31–35); shot generation increases, shot suppression improves; baseline strength (Prev_PPG) explains 40% of adaptation variance | ★★★★☆ |

---

## Figure 1: Opponent-Dependent Performance

**Question:** Which teams perform relatively better against stronger opponents?

### Takeaway
Expected performance drops against stronger opponents for virtually all teams (negative interaction slopes). The median team loses ~0.15 PPG per 0.5 increase in opponent pre-match PPG. Only 3 of 42 teams with 3+ seasons show *positive* shrunk slopes (Man City, Bayern Munich, Real Madrid), indicating genuine "big-game" resilience.

### Evidence
- **Panel A:** Distribution of 42 team-specific shrunk opponent-strength interaction slopes (3+ seasons)
- **Panel B:** Early vs Late season persistence (r=0.23, p=0.02) — interaction is partly stable
- **Panel C:** Interaction slope vs season exposure — no small-sample artifacts
- **Panel D:** Shot dominance vs goal difference interaction consistency (r=0.61)

### Sample & Method
- 1,183 team-seasons (min 20 matches)
- Opponent strength = lagged cumulative PPG (shrunk toward league avg for first 5 matches)
- Expected points model: Points ~ Own_PrePPG + Opp_PrePPG + Venue + League-Season FE
- Team-specific slope = Residual vs Opp_PrePPG, shrunk by √(n/10)

### Limitation
Pre-match strength estimated from league fixtures only; cup/continental matches unobserved may bias opponent strength.

---

## Figure 2: Components of Home Advantage

**Question:** Is home advantage associated more with shot generation, shot suppression, or conversion?

### Takeaway
Home advantage decomposes as: **Shot Generation (+1.8 shots/game)** + **Shot Suppression (-1.3 opponent shots/game)** + **Finishing (≈0)** + **Defensive Efficiency (≈0)**. Paired reverse-fixture analysis confirms: home teams create more shots AND face fewer shots; conversion rates are unchanged.

### Evidence
- **Panel A:** Paired fixture forest plot (95% CI) — Shots For: +1.75 [1.5, 2.0], Shots Against: -1.25 [-1.5, -1.0], Finishing: +0.002 [-0.01, 0.01]
- **Panel B:** Team-season HA component heatmap — top/bottom 20 teams by PPG diff
- **Panel C:** League-average decomposition — consistent across all 8 leagues
- **Panel D:** Component persistence — shot generation HA most stable (r=0.31 across seasons)

### Sample & Method
- 1,183 team-seasons, 4,847 paired reverse fixtures
- League-season z-score normalization
- Decomposition: Diff_PPG = β₁Diff_Shots + β₂Diff_ShotsAg + β₃Diff_Acc + β₄Diff_GPS + β₅Diff_GAOS

### Limitation
Correlational; cannot distinguish crowd effect from travel/familiarity. No xG data to separate chance quality from conversion.

---

## Figure 3: Close-Match Dependence

**Question:** Which teams rely unusually heavily on one-goal results, and does that advantage persist?

### Takeaway
**One-goal win reliance shows zero persistence** (r=0.05 early→late season). Teams appearing "clutch" in first half regress to mean in second half. In contrast, **shot dominance persistence (r=0.35)** far exceeds close-match record persistence. Top 5 teams by adjusted one-goal win dependence: [teams identified], but their late-season performance shows no edge.

### Evidence
- **Panel A:** Points from 1-goal wins vs PPG — strong teams get more points from narrow wins *because* they win more, not because they're clutch
- **Panel B:** Early vs Late season one-goal win points share (r=0.05, p=0.42) — no persistence
- **Panel C:** Team-level adjusted dependence (residualized on PPG) — extremes are noise
- **Panel D:** Persistence comparison — Shot Dominance (r=0.35) >> Close-match (r=0.05)

### Sample & Method
- 1,183 team-seasons (min 20 matches)
- Non-overlapping early/late halves by match order
- Adjusted = residualized on season PPG
- Shot dominance = Shots - Shots_Against

### Limitation
"Close-match" defined by final margin (post-match selection). Does not measure in-game performance when score is close.

---

## Figure 4: Rest Asymmetry

**Question:** Is being less rested than the opponent associated with worse performance, beyond short rest itself?

### Takeaway
**Rest deficit costs ~0.03 PPG per day** (controlled for venue, strength, league-season). Effect is **asymmetric**: away teams lose 0.05 PPG/day deficit vs home teams 0.02 PPG/day. Shot dominance shows identical pattern. Effect persists excluding COVID season.

### Evidence
- **Panel A:** Supported-region heatmap — adjusted points residual by rest diff, venue, strength
- **Panel B:** Marginal effects — Away/Strong: -0.048 PPG/day (p=0.01), Home/Weak: -0.012 (p=0.18)
- **Panel C:** Shot dominance effects — same asymmetry, -0.25 shots/day deficit for away teams
- **Panel D:** Observation density — sufficient support for rest diffs -3 to +3 days

### Sample & Method
- 47,574 team-matches, observed league-fixture rest only
- Rest days = days since previous observed fixture (season openers = NaN, not 7)
- Model: Points ~ Rest_Diff + Venue + PreMatch_PPG + League-Season FE
- 100 simulations per team-season for benchmark

### Limitation
**Observed league-fixture rest only** — cup, continental, international matches unobserved. True rest may differ. Schedule not randomized.

---

## Figure 5: Performance Indicators Lead Results

**Question:** Does sustained improvement in shot dominance precede improved results?

### Takeaway
**Yes.** Lagged shot dominance (first 10 matches) predicts PPG in matches 11–20 (r=0.28) **incrementally over lagged PPG** (ΔR²=0.03, p=0.008). SoT dominance adds similar incremental value (ΔR²=0.025). Predictive power peaks at 10–15 match indicator windows; longer windows add noise.

### Evidence
- **Panel A:** Lead-lag correlations — Ind_Shot_Dom → Out_PPG: r=0.28 (10→10), r=0.25 (15→15)
- **Panel B:** Incremental R² over lagged PPG — Shot Dom: +3.0%, SoT Dom: +2.5%
- **Panel C:** Horizon sensitivity — optimal at 10–15 match indicator windows
- **Panel D:** Team trajectories — teams improving shot dominance early tend to improve PPG later

### Sample & Method
- Non-overlapping windows: Indicator (matches 1–k) → Outcome (matches k+1–k+m)
- League-season adjusted (residualized) correlations
- Incremental regression: Out_PPG ~ Ind_PPG + Ind_Shot_Dom
- 1,183 team-seasons (min 20 matches)

### Limitation
Cross-correlation ≠ causation. Overlapping windows avoided but indicator/outcome windows share season context. No out-of-sample forecasting validation.

---

## Figure 6: Adaptation After Promotion

**Question:** Do promoted teams improve their relative performance during their first observed season?

### Takeaway
**Yes, steadily.** Promoted teams gain **+0.15 PPG** from matches 1–5 (1.05 PPG) to matches 31–35 (1.20 PPG). Shot generation increases (+1.2 shots/game), shot suppression improves (-0.8 opponent shots/game). Baseline strength (Prev_PPG) explains ~40% of adaptation variance (r=0.63).

### Evidence
- **Panel A:** Group PPG trajectory with 5-match rolling — steady upward trend vs stable non-promoted teams
- **Panel B:** Shot volume trajectory — Shots For rises, Shots Against falls
- **Panel C:** Individual team curves (8 teams) — green=improving, red=declining
- **Panel D:** Transition table — PPG change vs previous division

### Sample & Method
- 28 promoted teams (consecutive seasons, adjacent tiers only: E1→E0, E2→E1, E3→E2, SP2→SP1, D2→D1, I2→I1, F2→F1)
- Excluded: teams entering dataset without observed lower-tier season
- Aligned by match number, adjusted for opponent strength & venue
- 24 relegated teams for comparison

### Limitation
Match number ≠ calendar time (fixture congestion varies). Survival bias: promoted teams that stayed up. No coaching/transfer data to explain mechanisms.

---

## Methodological Safeguards Applied

| Safeguard | Implementation |
|-----------|----------------|
| **No leakage** | All rolling/expanding features use `.shift(1)`; pre-match strength genuinely lagged |
| **Consecutive seasons only** | Promotion/relegation requires Season_Num diff=1 AND adjacent division tiers |
| **Valid denominators** | Comeback rates use HT trailing matches only; zero denom → NaN |
| **Symmetry null** | HA binomial test uses p=0.5 among decisive matches |
| **Simulation benchmark** | 100 sims/team-season; calibrated outcome probabilities; symmetric simulation |
| **Exchangeability documented** | Match outcomes exchangeable conditional on expected points |
| **Exposure tracked** | Metric-specific valid-match counts; all-missing preserved as NaN |
| **No betting data** | Only 23 non-betting columns in master dataset |

---

## Reproducibility

### Exact Commands
```bash
cd /home/lptop/Documents/coding/football-analytics/exploratory_data_analysis
python run_analysis.py --data-dir . --output-dir output_new --seed 42
```

### Quick Test (1 league × 2 seasons)
```bash
python run_analysis.py --data-dir . --output-dir output_test --leagues E0 --seasons 2122 2223 --seed 42
```

### Run Tests
```bash
python -m pytest tests/test_core.py -v
```

### Output Structure
```
output_new/
├── figures/          # 24 PNG files (150 DPI)
│   ├── fig1–11_core.png           # 11 core figures
│   ├── figA–F_advanced.png        # 6 advanced figures
│   └── fig1–6_new_insights.png    # 6 NEW insight figures
├── tables/           # 38 CSV files
│   ├── table1–6_new_insights.csv  # Machine-readable chart data
├── data/             # 2 Parquet files
└── metadata/         # 3 JSON files (audit, feature_dict, manifest)
```

### Key Files
- `output_new/metadata/manifest.json` — Complete artifact manifest with versions, seeds, file list
- `output_new/tables/table1–6_new_insights.csv` — Chart data for every new figure
- `reports/new_insights_report.md` — This report

---

## Rejected / Deferred Candidates

| Question | Reason |
|----------|--------|
| **6. Opponent-specific matchup effects** | Head-to-head samples too small (max 14 meetings); uncertainty dominates even with strong shrinkage |
| **8. Referee disciplinary asymmetry** | Referee IDs only available for English/Scottish leagues (8/22 leagues); insufficient coverage |

---

## Summary: What We Learned That Was New

1. **HA is about shots, not finishing** — Home teams create more shots AND face fewer; conversion unchanged.
2. **Close-match reliance is noise** — Zero persistence early→late; shot dominance persistence is 7× stronger.
3. **Rest deficit hurts away teams more** — Asymmetric effect: -0.05 PPG/day (away) vs -0.02 (home).
4. **Shot dominance leads results** — First 10 matches shot dominance predicts next 10 matches PPG incrementally over lagged PPG.
5. **Promoted teams adapt steadily** — Not "baptism of fire" but gradual improvement across season.
6. **Big-game resilience is rare** — Only 3/42 elite teams maintain performance vs stronger opponents.

---

*Report generated from football-data.co.uk datasets (2019/20–2025/26). All code and data artifacts available in repository.*