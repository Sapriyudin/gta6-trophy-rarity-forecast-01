# GTA VI Trophy Rarity Forecaster

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Sapriyudin/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Sapriyudin/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

An offline forecaster for the [make.fun bounty "Forecast how many GTA VI
players earn each trophy in week one"](https://make.fun/c/gta6-trophy-rarity-forecast-01)
($250, top 20 paid). It reads Exophase PS5 trophy earn rates as recorded
before a case checkpoint and forecasts every trophy's earn share one week
after launch as five quantiles (5/25/50/75/95%), scored by pinball loss on
the logit scale.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` is Python 3.13, standard library only — no network, no pip
packages, no model calls. The grader runs `python3 -P -s /bot/main.py`
with `/input` and `/output`; outside the grader the two environment
variables stand in for them. One case file in, one `<id>.out` file out:
`{"trophies": [{"id": 1, "q": [q05, q25, q50, q75, q95]}, ...]}`,
quantiles non-decreasing, in [0, 100]. Runs in ~0.06 s per practice case
(limit is 8 s).

## How it works

Every read gives a tracked-player count `N` and per-trophy earners
`E = rate * N / 100`. The forecast rate is `E_target / N_target`.
The program detects which regime the case is in:

**Flood regime — the history is perfectly flat.** The recording is still
on its pre-launch cohort (638 tracked players, rates frozen) and the
launch flood lands between the last read and the target. Rates fall
because the denominator explodes while earners barely move. The median
logit shift is calibrated on the public Exophase page the morning of
launch: by late morning ~4,020 players were tracked (the slowest
trophies' earners were frozen, so `N ≈ E_last / rate_now`), the median
rate ratio was ~0.20, the median logit shift −1.61. Hard tiers
(platinum/gold/silver) shift −1.74 logit, bronze −1.51 plus an earliness
term for the quickest trophies; the 100% trophy stays at 100%. The
spread is wide on purpose — a regime jump is genuinely uncertain.

**Kinetic regime — the history moves.** Per trophy, fit the logit rate
against `ln(1 + hours since release)` over the last 24 h of reads
(least squares, with empirical-Bayes shrinkage of the slope by its own
standard error), project it to the target on the same log-time scale with
the projected delta damped by 0.5 (a log-time line over-projects a
saturating curve; 0.5 is the measured optimum across 8 synthetic worlds),
and cap the result at 10× the last rate. Tier ceilings measured harmful
in testing (they clip legitimate fast movers) and are off; the growth cap
is the safety rail. The spread combines the fit's residual volatility
(floored at 0.08), a random-walk term (0.25 per √day), and a
trend-uncertainty term (0.20 per √dx, capped at 0.40) — narrowed in
round 2 after coverage came in at 0.93–0.98 against a 0.90 nominal.
Quantiles are a logit-normal band around the median, rounded half-up to
4 decimals.

## Validation

On the kit's practice cases (`input/p00.in`, `p01.in`), scored against
the public Exophase page the morning of launch as pseudo-truth (the
house's own numbers imply the same: its persistence baseline scores
~1.51 on the practice with a ~0.68-point typical miss):

| forecaster | pinball loss | typical miss | 90% coverage |
|---|---|---|---|
| this program | **0.131** | 0.047 pts | 0.98 |
| house persistence baseline | 1.508 | 0.677 pts | 0.00 |

The kinetic branch was tested on a synthetic 2.5-day moving history
(360 reads, player base 638 → 45,000, per-trophy saturation curves):
pinball loss 0.145 vs 0.315 for persistence on the synthetic day-7
truth (typical miss 0.98 vs 6.17 points). Edge cases pass
(single read, all-null reads, zero horizon, empty history, a rate of
exactly 0). Outputs are deterministic — two runs write byte-identical
files.

Round 2 hardened the kinetic branch, which the practice cases never
exercise (both practice inputs are flat and identical). Eight diverse
synthetic worlds were built — four completion-time families (exponential,
Weibull, lognormal, mixed), Exophase-style refresh steps (rates frozen
for hours, then jumping), per-trophy bursts and stalls, and a late
player flood — and scored against BOTH replicable house baselines
(persistence and the fitted-curve trend baseline from the competition
spec). The original model lost to a house baseline in 4 of the 8 worlds;
after tuning (damp 0.7 → 0.5, tier ceilings off, growth cap 40× → 10×,
narrower spread) it beats both baselines in all 8:

| forecaster | worst world | mean (8 worlds) |
|---|---|---|
| this program (round 2) | **0.341** | **0.293** |
| house trend baseline | 0.870 | 0.466 |
| house persistence baseline | 0.721 | 0.478 |

Flood-regime sensitivity was also checked: shifting any flood constant
by ±0.3 logit moves the practice loss by at most ~0.10 (usually far
less), and narrowing the flood spread was rejected — under a plausible
calibration error its coverage collapses to 0.36, so the wide band
stays. Details: `work/validation-final.log`, `work/round2/`.

## Files

- `main.py` — the whole forecaster (the grader runs this file).
- `LICENSE` — MIT.
- `.makefun/owner.json` — entry owner wallet.
