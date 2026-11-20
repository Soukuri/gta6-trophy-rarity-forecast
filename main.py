"""Trophy-earn-rate forecaster for gta6-trophy-rarity-forecast-01.

Forecasts, for every GTA VI PlayStation 5 trophy, its Exophase earn rate at
the case's target, as five quantiles per trophy, scored by pinball loss on
the logit scale.

The page the house records is a step function: Exophase refreshes its
counts every few hours, so hundreds of recorded reads are a few dozen
states. Everything here is fitted on those refresh segments - maximal runs
of reads with the same players-tracked and rates - never on raw reads,
whose duplicates would crush the residual spread and let long flat
stretches dominate the slopes.

Model, per case:
  * Players-tracked and each trophy's earners are forecast separately with
    recency-weighted log-log fits over the last 18 hours of segments
    (weights halving every 24 hours), anchored at the last segment with the
    growth shrunk; the earners' slopes are also shrunk toward the
    cross-trophy median slope. A rate is a ratio of two moving quantities,
    and its recent dynamics live in both. A trophy whose latest segments
    lack a rate keeps the segment where its rate was last seen as its
    anchor - that segment's player count and log-time - never the latest
    segment's.
  * A segment-level logit trend of the rate itself, its slope shrunk
    toward its tier's median slope and damped for the rare tiers, is
    blended with the counts view (0.62 / 0.38).
  * When the page in view has never refreshed - one or two segments, the
    launch-night shape - no fit is possible, so the forecast prices the
    refresh that likely lands before the target: players multiply faster
    than earners, and deeper trophies' earners slowest of all, so rates
    fall by a trophy-specific factor under a wide spread. The calibration
    is the one refresh the house documented: 638 -> 3,868 players in under
    five hours, earners up only x4.5 in total and x1.3 outside the
    automatic trophy.
  * Structure: "Complete Chapter N" medians are non-increasing in N
    (isotonic), the final-mission trophy cannot exceed the last chapter's,
    the platinum is capped below the rarest other trophy and below the
    100%-completion trophy it requires.
  * Spread: a random-walk term over the horizon, the weighted fits'
    residuals, an extrapolation term, a Poisson term for trophies with
    few earners, and a wide allowance for series that have never moved.
    Floored and capped so no trophy is forecast with false precision.

Standard library only, no network, deterministic. Run as the grader does:

    MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from statistics import NormalDist, median

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [NormalDist().inv_cdf(tau) for tau in LEVELS]



SLOPE_SHRINK = 5.0
TIER_DAMP = {"platinum": 0.3, "gold": 0.7, "silver": 0.9, "bronze": 1.0}
PLAYERS_DAMP = 0.8
EARN_DAMP = 1.0
W_COUNTS = 0.62
SIGMA_RW = 0.5
SPREAD_MIN = 0.5
SPREAD_MAX = 2.5
FLAT_SPREAD = 1.0
FULL_SPREAD = 0.35
POISSON_CAP = 0.6
# Refresh prior for a page that has not refreshed in view: log players
# growth ~ REF_A * ln(1 + span/REF_B hours); earners grow at kappa of that.
REF_A = 2.0
REF_B = 3.0
REF_SPREAD = 0.55
CHAPTER_RE = re.compile(r"chapter\s+(\d+)", re.IGNORECASE)
TIER_PRIOR = {"platinum": 0.02, "gold": 0.05, "silver": 0.2, "bronze": 0.5}


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
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def ols_w(xs, ys, ws):
    """Weighted least squares y ~ a + b*x. Returns (a, b, resid_std) or None."""
    n = len(xs)
    if n < 2:
        return None
    sw = sum(ws)
    if sw <= 0:
        return None
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return None
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    a = my - b * mx
    if n > 2:
        resid = math.sqrt(sum(w * (y - (a + b * x)) ** 2 for x, y, w in zip(xs, ys, ws)) / sw)
    else:
        resid = 0.0
    return (a, b, resid)


def isotonic_decreasing(values):
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


def segments_of(history):
    """Compress reads into refresh segments: (time, players, rates tuple)."""
    segs = []
    for read in history:
        key = (read.get("players_tracked"), tuple(read.get("rates") or []))
        if segs and segs[-1][0] == key:
            continue
        segs.append((key, parse_time(read["at"])))
    return [(t, key[0], key[1]) for key, t in segs]


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon_min = case.get("horizon_minutes") or 0
    d_days = horizon_min / 1440.0

    release = parse_time(case["game"]["release_date"] + "T00:00:00Z"
                         if len(case["game"]["release_date"]) == 10
                         else case["game"]["release_date"])
    checkpoint = parse_time(case["at"])
    target = parse_time(case["target"])
    if horizon_min <= 0:
        # No stated horizon: fall back to the checkpoint-to-target span
        # so the random-walk spread term never silently vanishes and
        # leaves the forecast overconfident.
        horizon_min = max(0.0, (target - checkpoint).total_seconds() / 60.0)
        d_days = horizon_min / 1440.0

    def log_time(moment):
        hours = (moment - release).total_seconds() / 3600.0
        return math.log(max(0.01, 1.0 + hours))

    x_target = log_time(target)
    WIN_H = 18.0
    HL_H = 24.0
    segs = segments_of(history)
    n_seg = len(segs)
    seg_x = [log_time(t) for (t, _, _) in segs]
    seg_p = [float(p or 0) for (_, p, _) in segs]
    x_last = seg_x[-1] if segs else log_time(checkpoint)
    p_last = seg_p[-1] if segs else 0.0
    # Hours from the last segment's first read to the target: the span over
    # which unseen refreshes act when the page has not refreshed in view.
    span_hours = max(0.0, (target - (segs[-1][0] if segs else checkpoint)).total_seconds() / 3600.0)

    # Last non-null rate and earners per trophy (over segments). Earners
    # are counted with the players of the segment where the rate was
    # actually observed, and that segment's log-time is kept as the
    # anchor: a rate missing from the latest segments is stale, and
    # pairing it with the latest (larger) player count would overstate
    # the earners, while extrapolating from the last segment would
    # understate how long ago the observation was made.
    last_rate = []
    last_earn = []
    last_x = []
    for i in range(len(trophies)):
        rate = None
        earn = None
        x_obs = x_last
        for k in range(n_seg - 1, -1, -1):
            _, p, rates = segs[k]
            if i < len(rates) and rates[i] is not None:
                rate = float(rates[i])
                earn = rate / 100.0 * float(p or 0)
                x_obs = seg_x[k]
                break
        last_rate.append(rate)
        last_earn.append(earn)
        last_x.append(x_obs)

    # --- Global players forecast: recent-window weighted log-log fit on
    # segments, anchored at the last segment, growth shrunk. ---
    seg_h = [(t - release).total_seconds() / 3600.0 for (t, _, _) in segs]
    h_last = seg_h[-1] if segs else 0.0
    ckpt_h = (checkpoint - release).total_seconds() / 3600.0

    def windowed(pts):
        """pts: list of (x, h, y). Returns (xs, ys, ws) over the recent
        window (all points if the window holds fewer than 4)."""
        win = [q for q in pts if q[1] >= ckpt_h - WIN_H]
        if len(win) < 4:
            win = pts
        xs = [q[0] for q in win]
        ys = [q[2] for q in win]
        ws = [0.5 ** (max(0.0, h_last - q[1]) / HL_H) for q in win]
        return xs, ys, ws

    p_target = p_last
    if n_seg >= 3 and p_last > 0:
        pts = [(x, h, math.log(pp)) for x, h, pp in zip(seg_x, seg_h, seg_p) if pp > 0]
        if len(pts) >= 3:
            xs, ys, ws = windowed(pts)
            fit = ols_w(xs, ys, ws)
            if fit is not None:
                growth = math.exp(fit[1] * (x_target - x_last))
                p_target = p_last * growth ** PLAYERS_DAMP
    p_target = max(p_target, p_last)

    # --- Per-trophy segment series. ---
    # earn points: (x, E) ; rate points: (x, logit(rate)) for 0.02<=r<=99.9
    earn_pts = []
    rate_pts = []
    for i in range(len(trophies)):
        ep, rp = [], []
        for x, h, (_, p, rates) in zip(seg_x, seg_h, segs):
            if i >= len(rates) or rates[i] is None or not p:
                continue
            r = float(rates[i])
            ep.append((x, h, r / 100.0 * float(p)))
            if 0.02 <= r <= 99.9:
                rp.append((x, h, logit(r)))
        earn_pts.append(ep)
        rate_pts.append(rp)

    # Earners log-slopes per trophy and their cross-trophy median (shrinkage).
    earn_fit = []
    for i in range(len(trophies)):
        pts = earn_pts[i]
        fit = None
        if len(pts) >= 3 and len({round(q[0], 6) for q in pts}) >= 3:
            xs, ys, ws = windowed([(x, h, math.log(e + 0.5)) for x, h, e in pts])
            if len(xs) >= 3:
                fit = ols_w(xs, ys, ws)
        earn_fit.append(fit)
    slopes_e = [f[1] for f in earn_fit if f is not None and f[2] < 1.5]
    med_slope_e = median(slopes_e) if slopes_e else 0.0

    # Tier median rate-slopes (segment fits).
    rate_fit = []
    for i in range(len(trophies)):
        pts = rate_pts[i]
        fit = None
        if len(pts) >= 3 and len({round(q[0], 6) for q in pts}) >= 3:
            xs, ys, ws = windowed(pts)
            if len(xs) >= 3:
                fit = ols_w(xs, ys, ws)
        rate_fit.append(fit)
    tier_slopes = {}
    for tier in ("platinum", "gold", "silver", "bronze"):
        slopes = [rate_fit[i][1] for i, t in enumerate(trophies)
                  if t.get("tier") == tier and rate_fit[i] is not None
                  and rate_fit[i][2] < 1.0]
        if slopes:
            tier_slopes[tier] = median(slopes)

    medians = []
    spreads = []
    for i, trophy in enumerate(trophies):
        tier = trophy.get("tier") or "bronze"
        damp = TIER_DAMP.get(tier, 1.0)
        r_last = last_rate[i]
        e_last = last_earn[i]

        if r_last is not None and r_last >= 99.99:
            medians.append(logit(99.99))
            spreads.append(FULL_SPREAD)
            continue
        if r_last is None:
            medians.append(logit(TIER_PRIOR.get(tier, 0.5)))
            spreads.append(1.6)
            continue

        # --- Counts view. ---
        y_counts = None
        fit_e = earn_fit[i]
        if n_seg >= 3 and e_last is not None and e_last >= 0.5 and fit_e is not None and p_target > 0:
            n = len(earn_pts[i])
            w = n / (n + SLOPE_SHRINK)
            b = (w * fit_e[1] + (1.0 - w) * med_slope_e) * (0.5 if tier == "platinum" else 1.0)
            e_target = e_last * math.exp(b * (x_target - last_x[i])) ** EARN_DAMP
            e_target = max(e_target, e_last)
            e_target = min(e_target, p_target)
            r_counts = 100.0 * e_target / p_target
            if 0.0 < r_counts < 100.0:
                y_counts = logit(r_counts)
        elif n_seg <= 2 and p_last > 0:
            # The page has not refreshed in view: price the refresh that
            # likely lands before the target. Players multiply faster than
            # earners; deeper trophies' earners slowest.
            mu_p = REF_A * math.log(1.0 + span_hours / REF_B)
            if r_last >= 1.0:
                kappa = 0.72
            elif r_last >= 0.1:
                kappa = 0.60
            else:
                kappa = 0.47
            if tier == "platinum":
                kappa = 0.30
            r_med = r_last * math.exp((kappa - 1.0) * mu_p)
            y_counts = logit(max(0.01, r_med))

        # --- Trend view (segments). ---
        y_trend = None
        resid = 0.35
        fit_r = rate_fit[i]
        tier_slope = tier_slopes.get(tier, 0.0)
        if fit_r is not None:
            n = len(rate_pts[i])
            w = n / (n + SLOPE_SHRINK)
            b = (w * fit_r[1] + (1.0 - w) * tier_slope) * damp
            y_trend = logit(r_last) + b * (x_target - last_x[i])
            resid = fit_r[2]
        elif n_seg >= 3:
            y_trend = logit(r_last) + tier_slope * damp * (x_target - last_x[i])

        if y_counts is not None and y_trend is not None:
            y_med = W_COUNTS * y_counts + (1.0 - W_COUNTS) * y_trend
        elif y_counts is not None:
            y_med = y_counts
        elif y_trend is not None:
            y_med = y_trend
        else:
            y_med = logit(r_last)
        y_med = max(logit(0.02), min(logit(99.99), y_med))
        medians.append(y_med)

        # --- Spread. ---
        flat = len({q[2] for q in earn_pts[i]}) <= 1 and n_seg >= 2
        pois = min(POISSON_CAP, 1.0 / math.sqrt(max(e_last or 0.0, 1.0)))
        s = math.sqrt((SIGMA_RW * math.sqrt(d_days)) ** 2
                      + resid ** 2
                      + (0.35 * max(0.0, x_target - x_last)) ** 2
                      + (FLAT_SPREAD if flat else 0.0) ** 2
                      + pois ** 2
                      + (REF_SPREAD if n_seg <= 2 else 0.0) ** 2)
        if tier == "platinum":
            s = max(s, 1.2)
        spreads.append(min(SPREAD_MAX, max(SPREAD_MIN, s)))

    # --- Structure: chapter trophies non-increasing in chapter order. ---
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
        # The final mission needs the last chapter first.
        ch_last = idxs[-1]
        for i, trophy in enumerate(trophies):
            if "final mission." in (trophy.get("description") or ""):
                medians[i] = min(medians[i], medians[ch_last])

    # --- Structure: platinum below the rarest other trophy and below the
    # 100%-completion trophy (which the platinum itself requires). ---
    plat_idx = [i for i, t in enumerate(trophies) if t.get("tier") == "platinum"]
    if plat_idx and len(trophies) > 1:
        others = [sigmoid(medians[i]) for i in range(len(trophies)) if i not in plat_idx]
        cap = logit(0.95 * min(others)) if others else None
        for i, trophy in enumerate(trophies):
            if "100% Completion" in (trophy.get("description") or ""):
                cap = min(cap, medians[i]) if cap is not None else medians[i]
        if cap is not None:
            for i in plat_idx:
                medians[i] = min(medians[i], cap)

    rows = []
    for i, trophy in enumerate(trophies):
        qs = [round4(min(100.0, max(0.01, sigmoid(medians[i] + z * spreads[i]))))
              for z in Z]
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
