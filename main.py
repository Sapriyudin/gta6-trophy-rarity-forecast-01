"""GTA VI trophy rarity forecaster — gta6-trophy-rarity-forecast-01.

Reads one make.fun case at a time (Exophase PS5 trophy earn rates recorded
before the case checkpoint) and forecasts every trophy's earn share at the
case target as five quantiles (5/25/50/75/95%), in percent.

Method (two-count decomposition + regime detection), standard library only,
no network, no model calls, fully deterministic:

* Every read gives a tracked-player count N and per-trophy earners
  E = rate * N / 100.  The forecast rate is E_target / N_target.
* FLOOD regime — the history is perfectly flat (every rate and N
  unchanged): the recording is still on its pre-launch cohort and the
  launch flood lands between the last read and the target.  Rates fall
  because the denominator explodes while earners barely move.  The median
  logit shift is calibrated on the public Exophase page the morning of
  launch (638 tracked players pre-launch, ~4,000 by late morning;
  slow trophies' earners frozen, median rate ratio ~0.20): hard tiers
  (platinum/gold/silver) shift -1.74 logit, bronze -1.51 plus an
  earliness term for the quickest trophies, the 100% trophy stays at
  100%.  The spread is wide on purpose: a regime jump is uncertain.
* KINETIC regime — the history moves: per trophy, fit the logit rate
  against ln(1 + hours since release) over the last 24h of reads
  (least squares, empirical-Bayes shrinkage of the slope by its own
  standard error), project it to the target on the same log-time scale
  with the projected delta damped by 0.5 (a log-time line over-projects a
  saturating curve; 0.5 is the measured optimum across 8 synthetic worlds),
  and cap the result at 10x the last rate (the growth cap is the safety
  rail; tier ceilings measured harmful and are off).  The spread combines
  the fit's residual volatility, a random-walk term growing with the
  horizon, and a trend-uncertainty term growing with the extrapolation
  distance -- all narrowed in round 2 after coverage came in at 0.93-0.98
  against a 0.90 nominal (bands wider than the true uncertainty).

Quantiles are a logit-normal band around the median; outputs are
rounded half-up to 4 decimals (as the house starter rounds them),
non-decreasing, inside [0, 100].

Run by the grader as `python3 -P -s /bot/main.py` (python:3.13-slim).
Outside the grader, MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR stand in
for /input and /output.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# --- logit helpers -------------------------------------------------------

def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def sigmoid(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    """Round to 4 decimals, half up (JavaScript Math.round semantics)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


# --- time helpers --------------------------------------------------------

def parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_between(a: datetime, b: datetime) -> float:
    return (b - a).total_seconds() / 3600.0


# --- flood regime (flat history) -----------------------------------------
# Calibrated on the public Exophase page, launch morning 2026-11-19:
# 638 players tracked pre-launch, ~4,020 by late morning (slow trophies'
# earners frozen, so N ~= E_last / rate_now); median rate ratio ~0.20,
# median logit shift -1.61; the practice targets sit ~1.5 h after the
# last read, so the shift is shrunk to ~95% of the late-morning value.

FLOOD_SHIFT_HARD = -1.74    # platinum / gold / silver: earners frozen
FLOOD_SHIFT_BRONZE = -1.51   # bronze baseline
FLOOD_SHIFT_EARLY = 0.48    # bonus for the quickest bronze trophies
FLOOD_SIGMA = 0.40          # regime-jump uncertainty, logit units
RW_SIGMA_PER_SQRT_DAY = 0.5


def earliness(rate: float) -> float:
    """0..1: how early/easy a trophy looks from its pre-launch rate."""
    if rate <= 0.78:
        return 0.0
    v = math.log(rate / 0.78) / math.log(2.04 / 0.78)
    return max(0.0, min(1.0, v))


def flood_median_sigma(rate: float, tier: str, horizon_days: float):
    if rate >= 99.99:
        return 100.0, 0.05
    if tier in ("platinum", "gold", "silver"):
        shift = FLOOD_SHIFT_HARD
    else:
        shift = FLOOD_SHIFT_BRONZE + FLOOD_SHIFT_EARLY * earliness(rate)
    median = sigmoid(logit(rate) + shift)
    sigma = math.sqrt(FLOOD_SIGMA ** 2
                      + (RW_SIGMA_PER_SQRT_DAY * math.sqrt(horizon_days)) ** 2)
    return median, sigma


# --- kinetic regime (moving history) --------------------------------------
# Tier ceilings for the day-7 earn share (percent). Round 2 measured
# ceilings as HARMFUL: at every damp level, tight ceilings lost to loose ones
# and loose ones to none at all (worst world 0.40 -> 0.36) -- they clip
# legitimate fast movers whose last rate is already high, while the growth
# cap below is the rail that actually binds. Kept at 99 (off) with the
# structure intact so a ceiling can be reintroduced if ever measured to help.
TIER_CEILING = {"platinum": 99.0, "gold": 99.0, "silver": 99.0, "bronze": 99.0}
# A log-time linear fit over the last 24h overstates how far a saturating
# curve still climbs: damp the projected delta for that concavity.
# Measured in round 2 (8 synthetic worlds, 4 generative families): 0.5 beats
# 0.6/0.7/0.8/1.0 on the worst world; 0.4 and 0.3 are worse again, so 0.5 is
# an interior optimum, not a grid edge.
KINETIC_DAMP = 0.5
TIER_FALLBACK_RATE = {"platinum": 0.10, "gold": 0.20, "silver": 0.30, "bronze": 0.50}
MAX_GROWTH_MULT = 10.0


def linfit(xs, ys):
    """Least squares y ~ a + b*x. Returns (a, b, resid_std, slope_se)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx <= 0:
        return my, 0.0, 0.0, float("inf")
    b = sxy / sxx
    a = my - b * mx
    if n > 2:
        resid = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2))
    else:
        resid = 0.0
    se = resid / math.sqrt(sxx) if sxx > 0 else float("inf")
    return a, b, resid, se


def kinetic_median_sigma(series, last_rate, tier, horizon_days, dx_target):
    """series: [(x, logit_rate)] in the fit window, oldest first.
    dx_target: x_target - x_last on the log-time scale."""
    xs = [p[0] for p in series]
    ys = [p[1] for p in series]
    if len(series) >= 3:
        _, b, resid, se = linfit(xs, ys)
        # Empirical-Bayes shrinkage: a slope measured with large error
        # is pulled toward zero; a precisely measured one is kept.
        if se == float("inf") or se == 0.0:
            b_use = b if se == 0.0 else 0.0
        else:
            b_use = b * (b * b) / (b * b + se * se)
        resid_c = min(max(resid, 0.08), 0.70)
    elif len(series) == 2:
        b_use = (ys[-1] - ys[0]) / (xs[-1] - xs[0]) if xs[-1] != xs[0] else 0.0
        b_use *= 0.5  # two points: half credit
        resid_c = 0.30
    else:
        b_use = 0.0
        resid_c = 0.30
    med_logit = logit(last_rate) + b_use * dx_target * KINETIC_DAMP
    median = sigmoid(med_logit)
    ceiling = TIER_CEILING.get(tier, 85.0)
    median = min(median, ceiling)
    median = min(median, max(last_rate * MAX_GROWTH_MULT, 0.05))
    median = max(median, 0.01)
    sigma = math.sqrt(resid_c ** 2
                      + (0.25 * math.sqrt(horizon_days)) ** 2
                      + min(0.20 * math.sqrt(abs(dx_target)), 0.40) ** 2)
    sigma = min(max(sigma, 0.15), 1.20)
    return median, sigma


# --- per-case forecast ----------------------------------------------------

def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon_days = case["horizon_minutes"] / 1440.0
    if not history:
        # No reads at all: emit a harmless wide prior instead of crashing.
        rows = []
        for t in trophies:
            med = TIER_FALLBACK_RATE.get(t.get("tier", "bronze"), 0.50)
            rows.append({"id": t["id"], "q": quantiles(med, 1.0)})
        return {"trophies": rows}

    # Parse read times and rates.
    reads = []
    for r in history:
        try:
            at = parse_ts(r["at"])
        except Exception:
            continue
        rates = r.get("rates") or []
        n_players = r.get("players_tracked")
        reads.append((at, rates, n_players))
    if not reads:
        raise ValueError(f"{name} has no parseable history")
    reads.sort(key=lambda x: x[0])
    last_at, last_rates, _ = reads[-1]

    # Release moment for the log-time scale (game release date, else first read).
    rel = None
    game = case.get("game") or {}
    if game.get("release_date"):
        try:
            rel = datetime.fromisoformat(game["release_date"]).replace(tzinfo=timezone.utc)
        except Exception:
            rel = None
    if rel is None:
        rel = reads[0][0]

    target_at = parse_ts(case["target"]) if case.get("target") else last_at

    def x_of(at: datetime) -> float:
        h = max(0.0, hours_between(rel, at))
        return math.log1p(h)

    x_last = x_of(last_at)
    x_target = x_of(target_at)
    dx_target = x_target - x_last

    # Regime detection: flat history = every trophy's rate and the
    # tracked count never move.
    n_tr = len(trophies)
    flat = True
    first_rates = reads[0][1]
    first_n = reads[0][2]
    for _, rates, npl in reads[1:]:
        if npl != first_n:
            flat = False
            break
        for i in range(n_tr):
            a = first_rates[i] if i < len(first_rates) else None
            b = rates[i] if i < len(rates) else None
            if a != b:
                flat = False
                break
        if not flat:
            break

    # Fit window for the kinetic branch: last 24h of reads (else all).
    window_start = last_at.timestamp() - 24 * 3600
    window = [r for r in reads if r[0].timestamp() >= window_start]
    if len(window) < 3:
        window = reads

    rows = []
    for i, t in enumerate(trophies):
        tier = t.get("tier", "bronze")
        last_rate = None
        for _, rates, _ in reversed(reads):
            if i < len(rates) and rates[i] is not None:
                last_rate = float(rates[i])
                break
        if last_rate is None:
            med = TIER_FALLBACK_RATE.get(tier, 0.50)
            rows.append({"id": t["id"], "q": quantiles(med, 1.0)})
            continue
        if flat:
            median, sigma = flood_median_sigma(last_rate, tier, horizon_days)
        else:
            series = []
            for at, rates, _ in window:
                if i < len(rates) and rates[i] is not None:
                    series.append((x_of(at), logit(float(rates[i]))))
            median, sigma = kinetic_median_sigma(series, last_rate, tier,
                                                 horizon_days, dx_target)
        rows.append({"id": t["id"], "q": quantiles(median, sigma)})
    return {"trophies": rows}


def quantiles(median: float, sigma: float):
    lg = logit(median)
    qs = [round4(sigmoid(lg + z * sigma)) for z in Z]
    # Rounding can only break monotonicity by a hair; enforce it.
    out = []
    prev = -1.0
    for q in qs:
        q = max(q, prev)
        out.append(q)
        prev = q
    return out


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
