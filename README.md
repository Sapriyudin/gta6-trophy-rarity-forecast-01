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
standard error), project it to the target on the same log-time scale so
growth saturates as time passes, and cap the result at a loose tier
ceiling (platinum 3%, gold 45%, silver 60%, bronze 90%) plus a 40×
growth cap. The spread combines the fit's residual volatility, a
random-walk term growing with the horizon, and a trend-uncertainty term
growing with the extrapolation distance. Quantiles are a logit-normal
band around the median, rounded half-up to 4 decimals.

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
pinball loss 0.20 vs 0.32 for persistence on the synthetic day-7
truth, 0.20 vs 0.44 in a slower-growth variant. Edge cases pass
(single read, all-null reads, zero horizon, empty history, a rate of
exactly 0). Outputs are deterministic — two runs write byte-identical
files.

## Files

- `main.py` — the whole forecaster (the grader runs this file).
- `LICENSE` — MIT.
- `.makefun/owner.json` — entry owner wallet.
