"""Trophy-earn-rate forecaster for gta6-trophy-rarity-forecast-01.

Forecasts, for every GTA VI PlayStation 5 trophy, its Exophase earn rate at the
case's target, as five quantiles per trophy, scored by pinball loss on the
logit scale.

Model, per trophy, from the case's own read history:
  * A logit-linear trend: logit(rate) regressed on ln(1 + hours since the
    game's release), fitted over the last 36 hours of reads. Slopes are
    shrunk toward their tier's median slope (platinum/gold/silver/bronze),
    because a trophy's own short series is noisy and trophies of a tier move
    together. Platinum's slope is damped hard: earning it needs every other
    trophy first, so it is the slowest mover of all.
  * An earned-counts view: earned(t) = rate(t) * players_tracked(t) / 100.
    Log earned-counts and log players-tracked are each fitted against the
    same log-time axis, and the ratio at the target gives a second median.
    This separates the two forces that move a rate: players earning the
    trophy (pushing it up) and new players being tracked (diluting it down).
    The two medians are blended in logit space.
  * Structure the data alone cannot see: trophies for "Complete Chapter N"
    must be non-increasing in N (you cannot finish chapter 2 without chapter
    1), enforced by isotonic regression on their medians; the platinum
    median is capped below the rarest other trophy's median.
  * Spread: a random-walk term growing with the square root of the horizon,
    plus the trend fit's residual scatter, plus an extrapolation term for
    how far past the last read the target sits, plus a wide flat-history
    allowance when the series has never moved at all (a flat launch-night
    history hides Exophase refreshes that slash every rate at once, as the
    04:30 UTC refresh did). The spread is floored and capped so no trophy
    is forecast with false precision.

Standard library only, no network, deterministic: the same case always
writes the same bytes. Run it as the grader does:

    MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist, median

# The grader's -P leaves /bot off sys.path: put it back first, so an `import
# model` finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [NormalDist().inv_cdf(tau) for tau in LEVELS]

# How far back the trend fit looks, in hours before the checkpoint.
WINDOW_HOURS = 36.0
# Shrinkage strength of a trophy's own slope toward its tier's median slope:
# weight on the own slope is n / (n + SLOPE_SHRINK).
SLOPE_SHRINK = 5.0
# Slope damping by tier (the rarer the tier, the slower it moves).
TIER_DAMP = {"platinum": 0.3, "gold": 0.7, "silver": 0.9, "bronze": 1.0}
# Random-walk spread, logit units per square-root day (the house baseline's).
SIGMA_RW = 0.5
# Spread floor / cap, logit units.
SPREAD_MIN = 0.5
SPREAD_MAX = 2.5
# Extra spread when a series has never moved: the history is hiding refreshes.
FLAT_SPREAD = 1.0
# A trophy at (essentially) 100% gets a narrow spread: it is automatic.
FULL_SPREAD = 0.35
CHAPTER_RE = re.compile(r"chapter\s+(\d+)", re.IGNORECASE)


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """x rounded to 4 decimals, half up, as JavaScript's Math.round does it.

    A whole number comes back as an int, so json.dumps writes 100 where
    JavaScript writes 100 (not 100.0).
    """
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def ols(xs, ys):
    """Least squares y ~ a + b*x. Returns (a, b, resid_std) or None."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    if n > 2:
        resid = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2))
    else:
        resid = 0.0
    return (a, b, resid)


def isotonic_decreasing(values):
    """Pool adjacent violators: returns the closest non-increasing sequence."""
    blocks = [[v, 1] for v in values]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] < blocks[i + 1][0]:
            total = blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1]
            count = blocks[i][1] + blocks[i + 1][1]
            blocks[i:i + 2] = [[total / count, count]]
            if i > 0:
                i -= 1
        else:
            i += 1
    out = []
    for value, count in blocks:
        out.extend([value] * count)
    return out


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon_min = case.get("horizon_minutes") or 0
    d_days = horizon_min / 1440.0

    release = parse_time(case["game"]["release_date"] + "T00:00:00Z"
                         if len(case["game"]["release_date"]) == 10
                         else case["game"]["release_date"])
    checkpoint = parse_time(case["at"])

    def hours_since_release(moment):
        return (moment - release).total_seconds() / 3600.0

    def log_time(moment):
        return math.log(max(0.01, 1.0 + hours_since_release(moment)))

    x_target = log_time(parse_time(case["target"]))
    window_start = checkpoint.timestamp() - WINDOW_HOURS * 3600.0
    recent = [r for r in history if parse_time(r["at"]).timestamp() >= window_start]
    if len(recent) < 4:
        recent = history
    x_last = log_time(parse_time(history[-1]["at"])) if history else log_time(checkpoint)

    # Per-trophy series over the window: (x, rate, players).
    series = []
    for i, trophy in enumerate(trophies):
        pts = []
        for read in recent:
            rates = read.get("rates") or []
            if i >= len(rates) or rates[i] is None:
                continue
            pts.append((log_time(parse_time(read["at"])),
                        float(rates[i]),
                        float(read.get("players_tracked") or 0)))
        series.append(pts)

    # Last non-null rate per trophy, from the whole history.
    last_rate = []
    for i in range(len(trophies)):
        rate = None
        for read in reversed(history):
            rates = read.get("rates") or []
            if i < len(rates) and rates[i] is not None:
                rate = float(rates[i])
                break
        last_rate.append(rate)

    # Tier median slopes from the trophies whose own trend fit is usable.
    own_fit = []
    for i, trophy in enumerate(trophies):
        pts = series[i]
        fit = None
        if len(pts) >= 3 and len({round(p[0], 6) for p in pts}) >= 3:
            fit = ols([p[0] for p in pts], [logit(p[1]) for p in pts])
        own_fit.append(fit)
    tier_slopes = {}
    for tier in ("platinum", "gold", "silver", "bronze"):
        slopes = [own_fit[i][1] for i, t in enumerate(trophies)
                  if t.get("tier") == tier and own_fit[i] is not None
                  and own_fit[i][2] < 1.0]
        if slopes:
            tier_slopes[tier] = median(slopes)

    medians = []
    spreads = []
    for i, trophy in enumerate(trophies):
        tier = trophy.get("tier") or "bronze"
        damp = TIER_DAMP.get(tier, 1.0)
        pts = series[i]
        r_last = last_rate[i]
        if r_last is None:
            r_last = 1.0

        # A trophy everyone already holds stays (essentially) held.
        if r_last >= 99.99:
            medians.append(logit(99.99))
            spreads.append(FULL_SPREAD)
            continue

        # --- Component 1: logit-linear trend, slope shrunk to the tier. ---
        y_trend = None
        resid = 0.3
        fit = own_fit[i]
        tier_slope = tier_slopes.get(tier, 0.0)
        if fit is not None:
            a, b, resid = fit
            n = len(pts)
            w = n / (n + SLOPE_SHRINK)
            b_shrunk = (w * b + (1.0 - w) * tier_slope) * damp
            # Re-anchored at the window's mean, so only the shrunk slope
            # carries the forecast forward from the data's own centre.
            mean_x = sum(p[0] for p in pts) / n
            mean_y = sum(logit(p[1]) for p in pts) / n
            y_trend = mean_y + b_shrunk * (x_target - mean_x)
        else:
            # No usable fit: drift from the last rate at the tier's slope.
            y_trend = logit(r_last) + tier_slope * damp * (x_target - x_last)

        # --- Component 2: earned counts over players tracked. ---
        y_counts = None
        w_counts = 0.0
        e_pts = [(p[0], max(p[1] / 100.0 * p[2], 0.5)) for p in pts if p[2] > 0]
        n_pts = [(p[0], p[2]) for p in pts if p[2] > 0]
        if len(e_pts) >= 4 and len(n_pts) >= 4:
            fit_e = ols([p[0] for p in e_pts], [math.log(p[1]) for p in e_pts])
            fit_n = ols([p[0] for p in n_pts], [math.log(p[1]) for p in n_pts])
            if fit_e is not None and fit_n is not None:
                log_e = fit_e[0] + fit_e[1] * x_target
                log_n = fit_n[0] + fit_n[1] * x_target
                r_counts = 100.0 * math.exp(log_e - log_n)
                if 0.0 < r_counts < 100.0:
                    y_counts = logit(r_counts)
                    w_counts = 0.5

        if y_counts is not None:
            y_med = (1.0 - w_counts) * y_trend + w_counts * y_counts
        else:
            y_med = y_trend
        y_med = max(logit(0.02), min(logit(99.99), y_med))
        medians.append(y_med)

        # --- Spread. ---
        flat = len(pts) >= 2 and len({p[1] for p in pts}) == 1
        s = math.sqrt((SIGMA_RW * math.sqrt(d_days)) ** 2
                      + resid ** 2
                      + (0.4 * max(0.0, x_target - x_last)) ** 2
                      + (FLAT_SPREAD if flat else 0.0) ** 2)
        if tier == "platinum":
            s = max(s, 1.2)
        spreads.append(min(SPREAD_MAX, max(SPREAD_MIN, s)))

    # --- Structure: chapter trophies are non-increasing in chapter order. ---
    chapters = []
    for i, trophy in enumerate(trophies):
        m = CHAPTER_RE.search(trophy.get("description") or "")
        if m:
            chapters.append((int(m.group(1)), i))
    chapters.sort()
    if len(chapters) >= 2:
        idxs = [i for _, i in chapters]
        fixed = isotonic_decreasing([medians[i] for i in idxs])
        for pos, i in enumerate(idxs):
            medians[i] = fixed[pos]

    # --- Structure: the platinum cannot exceed the rarest other trophy. ---
    plat_idx = [i for i, t in enumerate(trophies) if t.get("tier") == "platinum"]
    if plat_idx and len(trophies) > 1:
        others = [sigmoid(medians[i]) for i in range(len(trophies)) if i not in plat_idx]
        cap = logit(0.95 * min(others)) if others else None
        if cap is not None:
            for i in plat_idx:
                medians[i] = min(medians[i], cap)

    rows = []
    for i, trophy in enumerate(trophies):
        qs = [round4(min(100.0, max(0.01, sigmoid(medians[i] + z * spreads[i]))))
              for z in Z]
        # Rounding can only keep order; enforce it anyway.
        for k in range(1, len(qs)):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
