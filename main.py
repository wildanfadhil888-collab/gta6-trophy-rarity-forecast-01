"""Refresh-aware earners/players forecaster for Exophase trophy rates.

The model works in counts: each page read is a snapshot of players tracked and
of the players who earned each trophy.  A refresh changes that snapshot in one
step, so projections evolve through new player cohorts rather than rates.
"""

import json
import math
import os
import random
from pathlib import Path


INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
LEVELS = (0.05, 0.25, 0.50, 0.75, 0.95)
Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)
SIMULATIONS = 2000


def clamp_rate(value):
    return min(99.99, max(0.01, float(value)))


def logit_rate(value):
    value = clamp_rate(value) / 100.0
    return math.log(value / (1.0 - value))


def inv_logit(value):
    if value >= 0:
        z = math.exp(-min(value, 700.0))
        return 100.0 / (1.0 + z)
    z = math.exp(max(value, -700.0))
    return 100.0 * z / (1.0 + z)


def rounded(value):
    value = min(100.0, max(0.0, value))
    answer = math.floor(value * 10000.0 + 0.5) / 10000.0
    return int(answer) if answer.is_integer() else answer


def quantile(sorted_values, probability):
    if not sorted_values:
        return 0.0
    place = (len(sorted_values) - 1) * probability
    low = int(place)
    high = min(low + 1, len(sorted_values) - 1)
    fraction = place - low
    return sorted_values[low] * (1.0 - fraction) + sorted_values[high] * fraction


def elapsed_minutes(earlier, later):
    """Parse the UTC timestamps used in cases without third-party date tools."""
    def minute_index(timestamp):
        date, clock = timestamp.split("T", 1)
        year, month, day = (int(part) for part in date.split("-"))
        hh, mm = (int(part) for part in clock[:5].split(":"))
        # Gregorian day number, sufficient for differences over this horizon.
        adjusted_year = year - (1 if month <= 2 else 0)
        era = adjusted_year // 400
        year_of_era = adjusted_year - era * 400
        adjusted_month = month + ( -3 if month > 2 else 9)
        day_of_year = (153 * adjusted_month + 2) // 5 + day - 1
        day_of_era = year_of_era * 365 + year_of_era // 4 - year_of_era // 100 + day_of_year
        return ((era * 146097 + day_of_era) * 24 + hh) * 60 + mm
    return minute_index(later) - minute_index(earlier)


def is_refresh(previous, current):
    if previous["players_tracked"] != current["players_tracked"]:
        return True
    before, after = previous["rates"], current["rates"]
    for a, b in zip(before, after):
        if a is None or b is None:
            if a is not b:
                return True
        elif abs(float(a) - float(b)) > 1e-10:
            return True
    return False


def make_history(case):
    reads = case["history"]
    if not reads:
        raise ValueError("case has no history")
    events = []
    for index in range(1, len(reads)):
        if is_refresh(reads[index - 1], reads[index]):
            events.append((index - 1, index))
    return reads, events


def practice_scenarios(case, last):
    """Launch-night cohort prior as a family of count allocations."""
    rates = [max(0.01, float(value or 0.01)) for value in last["rates"]]
    old_players = float(last["players_tracked"])
    if old_players <= 0:
        old_players = 638.0
    new_players = 3868.0 - old_players
    old_earners = [rate * old_players / 100.0 for rate in rates]
    special = next((i for i, rate in enumerate(rates) if rate >= 99.995), None)
    residual = max(0.0, 4276.0 - float(last.get("trophies_earned", 943)) - new_players)
    residual = min(103.0, residual)

    eligible = [i for i in range(len(rates)) if i != special]
    ranked = sorted(eligible, key=lambda i: (-rates[i], i))
    scenarios = []
    for width in (10, 12, 15, 18, 20):
        selected = ranked[:min(width, len(ranked))]
        weights = [rates[i] for i in selected]
        total = sum(weights) or 1.0
        additions = [0.0] * len(rates)
        for i, weight in zip(selected, weights):
            additions[i] = residual * weight / total
        scenarios.append(_rates_after_cohort(old_earners, additions, new_players,
                                              special, len(rates)))

    # A second public-shape alternative allocates residual earners over the
    # first ten earnable trophies in list/story order with geometric decay.
    story_order = [i for i in eligible
                   if str(case["trophies"][i].get("tier", "")).lower() != "platinum"]
    ordered = story_order[:min(10, len(story_order))]
    weights = [0.9 ** rank for rank in range(len(ordered))]
    total = sum(weights) or 1.0
    additions = [0.0] * len(rates)
    for i, weight in zip(ordered, weights):
        additions[i] = residual * weight / total
    scenarios.append(_rates_after_cohort(old_earners, additions, new_players,
                                          special, len(rates)))
    return scenarios, special


def _rates_after_cohort(old_earners, additions, new_players, special, count):
    out = []
    for i in range(count):
        if i == special:
            out.append(100.0)
        else:
            out.append(max(0.01, 100.0 * (old_earners[i] + additions[i]) / 3868.0))
    return out


def poisson(rng, mean):
    """Poisson sampler, using a product method at small means and normal tail."""
    if mean <= 0.0:
        return 0
    if mean < 30.0:
        limit = math.exp(-mean)
        product = 1.0
        count = 0
        while product > limit:
            product *= rng.random()
            count += 1
        return count - 1
    return max(0, int(math.floor(rng.gauss(mean, math.sqrt(mean)) + 0.5)))


def normal_binomial(rng, n, probability):
    if n <= 0 or probability <= 0.0:
        return 0.0
    if probability >= 1.0:
        return float(n)
    mean = n * probability
    variance = mean * (1.0 - probability)
    if variance < 8.0 and n < 300:
        # Exact Bernoulli loops are bounded to small cohorts.
        return float(sum(1 for _ in range(n) if rng.random() < probability))
    if mean < 8.0 and probability < 0.02:
        return float(poisson(rng, mean))
    return max(0.0, min(float(n), rng.gauss(mean, math.sqrt(variance))))


def refresh_model(case, reads, events):
    """Fit cohort size, cadence and smoothed trophy-level earn probabilities."""
    count = len(case["trophies"])
    last = reads[-1]
    base_n = max(1.0, float(last["players_tracked"]))
    base_e = [max(0.0, min(base_n, (float(v or 0.0) * base_n / 100.0)))
              for v in last["rates"]]
    base_rates = [100.0 * e / base_n for e in base_e]
    tier = [str(t.get("tier", "bronze")).lower() for t in case["trophies"]]

    multipliers = []
    gaps = []
    observations = [[] for _ in range(count)]
    previous_refresh_at = None
    for before_i, after_i in events:
        before, after = reads[before_i], reads[after_i]
        old_n = max(1.0, float(before["players_tracked"]))
        new_n = max(old_n, float(after["players_tracked"]))
        new_players = max(1.0, new_n - old_n)
        multipliers.append(max(1.0001, new_n / old_n))
        if previous_refresh_at is not None:
            gap = elapsed_minutes(previous_refresh_at, after["at"])
            if gap > 0:
                gaps.append(float(gap))
        previous_refresh_at = after["at"]
        for i, (old_rate, new_rate) in enumerate(zip(before["rates"], after["rates"])):
            if old_rate is None or new_rate is None:
                continue
            old_e = float(old_rate) * old_n / 100.0
            new_e = float(new_rate) * new_n / 100.0
            added = max(0.0, new_e - old_e)
            p = min(1.0, added / new_players)
            current_rate = max(0.01, min(99.99, float(old_rate))) / 100.0
            observations[i].append((p, current_rate, tier[i], new_players))

    if multipliers:
        multiplier = sorted(multipliers)[len(multipliers) // 2]
    else:
        multiplier = 1.10
    # A decaying excess growth factor avoids compounding a launch spike forever.
    decay = 0.82
    if len(multipliers) >= 2:
        ratios = []
        for a, b in zip(multipliers, multipliers[1:]):
            if a > 1.0:
                ratios.append((b - 1.0) / (a - 1.0))
        if ratios:
            decay = min(0.98, max(0.35, sorted(ratios)[len(ratios) // 2]))
    if gaps:
        cadence = sorted(gaps)[len(gaps) // 2]
    else:
        # With one observed refresh its exact cadence is unidentified. Use the
        # read cadence as a conservative lower-bound proxy, capped at 12 hours.
        if len(reads) > 1:
            read_gaps = [elapsed_minutes(a["at"], b["at"])
                         for a, b in zip(reads, reads[1:])]
            read_gaps = [float(g) for g in read_gaps if g > 0]
            cadence = min(720.0, max(60.0, sorted(read_gaps)[len(read_gaps) // 2])) if read_gaps else 240.0
        else:
            cadence = 240.0

    # Pool all refresh observations to fit a log-linear cohort-acquisition
    # curve. Tier offsets are shrunk toward the common curve; trophy estimates
    # retain their own evidence with a modest prior equivalent to 80 players.
    xs, ys, ws = [], [], []
    for i in range(count):
        for p, rate, _, n in observations[i]:
            if p <= 0.0 or p >= 1.0:
                continue
            xs.append(math.log(rate / (1.0 - rate)))
            ys.append(math.log(p / (1.0 - p)))
            ws.append(min(n, 50000.0))
    if len(xs) >= 3:
        sw = sum(ws)
        mx = sum(x * w for x, w in zip(xs, ws)) / sw
        my = sum(y * w for y, w in zip(ys, ws)) / sw
        variance = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
        slope = (sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws)) /
                 variance) if variance > 1e-9 else 1.0
        slope = min(1.5, max(0.25, slope))
        intercept = my - slope * mx
        residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
        resid_sd = math.sqrt(sum(r * r for r in residuals) / max(1, len(residuals) - 2))
    else:
        slope, intercept, resid_sd = 0.9, -0.3, 0.7
        residuals = [-0.7, 0.0, 0.7]

    tier_residual = {}
    for name in ("platinum", "gold", "silver", "bronze"):
        members = [(p, rate, n) for group in observations for p, rate, t, n in group
                   if t == name and 0.0 < p < 1.0]
        if len(members) >= 3:
            residual_values = [math.log(p / (1.0 - p)) -
                               (intercept + slope * math.log(rate / (1.0 - rate)))
                               for p, rate, _ in members]
            tier_residual[name] = sum(residual_values) / (len(residual_values) + 4.0)
        else:
            tier_residual[name] = 0.0

    probabilities = []
    for i in range(count):
        if base_rates[i] >= 99.995:
            probabilities.append(1.0)
            continue
        x = logit_rate(base_rates[i])
        fitted = min(0.98, max(0.0001, 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0,
                         intercept + slope * x + tier_residual.get(tier[i], 0.0)))))))
        own = observations[i]
        # Beta-style shrinkage: data from this trophy is blended with the
        # pooled rate curve, weighted by observed cohort sizes.
        own_weight = sum(min(n, 50000.0) for _, _, _, n in own)
        if own:
            own_rate = sum(p * min(n, 50000.0) for p, _, _, n in own) / max(1.0, own_weight)
            shrink = own_weight / (own_weight + 80.0)
            fitted = fitted * (1.0 - shrink) + own_rate * shrink
        probabilities.append(fitted)

    # Bootstrap the observed cohort multipliers for parameter uncertainty.
    multiplier_log_sd = 0.14 if len(multipliers) < 3 else 0.08
    if len(multipliers) >= 2:
        logs = [math.log(m) for m in multipliers]
        avg = sum(logs) / len(logs)
        observed_sd = math.sqrt(sum((x - avg) ** 2 for x in logs) / (len(logs) - 1))
        multiplier_log_sd = max(0.035, min(0.35, observed_sd / math.sqrt(len(logs))))
    return {
        "n": base_n, "earners": base_e, "rates": base_rates,
        "tiers": tier, "probabilities": probabilities,
        "multipliers": multipliers or [multiplier], "multiplier": multiplier,
        "decay": decay, "cadence": cadence,
        "residual_sd": max(0.25, min(1.5, resid_sd)),
        "bootstrap_residuals": residuals,
        "observations_per_trophy": [len(group) for group in observations],
        "multiplier_log_sd": multiplier_log_sd,
        "special": next((i for i, r in enumerate(base_rates) if r >= 99.995), None),
    }


def simulate_refreshes(case, reads, events):
    model = refresh_model(case, reads, events)
    horizon = max(0.0, float(case.get("horizon_minutes", 0.0)))
    expected_count = horizon / max(30.0, model["cadence"])
    # A refresh may land just outside the target: Poisson timing models that
    # uncertainty, with a minimum modest within-refresh drift distribution.
    rng = random.Random(719231)
    count = len(model["rates"])
    paths = [[] for _ in range(count)]
    for _ in range(SIMULATIONS):
        refresh_count = poisson(rng, expected_count)
        players = model["n"]
        earners = list(model["earners"])
        total_new_players = 0
        # A cohort-acquisition estimate is uncertain for the whole forecast,
        # not a fresh independent error for each player refresh. Drawing it
        # once per path preserves that parameter uncertainty over long horizons.
        path_probabilities = []
        for i, probability in enumerate(model["probabilities"]):
            if i == model["special"]:
                path_probabilities.append(1.0)
            else:
                observations = max(1, model["observations_per_trophy"][i])
                residuals = model["bootstrap_residuals"]
                # Resample fitted residuals with replacement to carry observed
                # refresh-to-refresh profile uncertainty into each path.
                bootstrap_shift = sum(residuals[rng.randrange(len(residuals))]
                                      for _ in range(observations)) / observations
                noisy_logit = math.log(probability / (1.0 - probability)) + bootstrap_shift
                path_probabilities.append(1.0 / (1.0 + math.exp(-max(-25.0, min(25.0, noisy_logit)))))
        bootstrapped = model["multipliers"][rng.randrange(len(model["multipliers"]))]
        base_mult = bootstrapped
        for step in range(refresh_count):
            decay_factor = model["decay"] ** step
            mean_mult = 1.0 + (base_mult - 1.0) * decay_factor
            multiplier = math.exp(math.log(max(1.0001, mean_mult)) +
                                  rng.gauss(0.0, model["multiplier_log_sd"]))
            multiplier = max(1.0001, min(3.0, multiplier))
            added_players = max(1, int(players * (multiplier - 1.0) + 0.5))
            total_new_players += added_players
            players += added_players
        for i, probability in enumerate(path_probabilities):
            if i == model["special"]:
                earners[i] = players
            elif total_new_players:
                # Conditional on a path's fitted cohort probability, the
                # independent Bernoulli additions from refreshes sum to one
                # binomial draw over all added players.
                earners[i] += normal_binomial(rng, total_new_players, probability)
        for i in range(count):
            rate = 100.0 if i == model["special"] else max(0.01, min(100.0, 100.0 * earners[i] / players))
            # Small within-refresh logit drift, reduced by remaining distance
            # from the binary floor and ceiling by the bounded inverse link.
            if i != model["special"]:
                rate = inv_logit(logit_rate(rate) + rng.gauss(0.0, 0.12 * math.sqrt(horizon / 1440.0)))
            paths[i].append(rate)
    result = []
    for i, values in enumerate(paths):
        values.sort()
        if i == model["special"]:
            result.append([100.0] * 5)
        else:
            result.append([quantile(values, level) for level in LEVELS])
    return result, model


def forecast_case(case):
    if tuple(case.get("levels", LEVELS)) != LEVELS:
        raise ValueError("unexpected quantile levels")
    reads, events = make_history(case)
    if events:
        quantiles, _ = simulate_refreshes(case, reads, events)
    else:
        scenarios, special = practice_scenarios(case, reads[-1])
        count = len(case["trophies"])
        # Each cohort profile is one plausible launch refresh. Shared profile
        # choice retains the across-trophy structure; fixed-seed logit noise
        # supplies modest within-profile uncertainty.
        rng = random.Random(381901)
        samples = [[] for _ in range(count)]
        horizon_scale = math.sqrt(max(0.0, float(case.get("horizon_minutes", 0))) / 90.0)
        base_spread = 0.05 * max(0.75, min(1.5, horizon_scale))
        for _ in range(SIMULATIONS):
            scenario = scenarios[rng.randrange(len(scenarios))]
            for i, median in enumerate(scenario):
                if i == special:
                    samples[i].append(100.0)
                else:
                    samples[i].append(inv_logit(logit_rate(median) + rng.gauss(0.0, base_spread)))
        quantiles = []
        for i, values in enumerate(samples):
            values.sort()
            quantiles.append([quantile(values, level) for level in LEVELS])

    rows = []
    for trophy, values in zip(case["trophies"], quantiles):
        q = []
        previous = 0.0
        for value in values:
            clean = rounded(value)
            clean = max(previous, clean)
            q.append(clean)
            previous = clean
        rows.append({"id": trophy["id"], "q": q})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        output = forecast_case(case)
        target = OUTPUT_DIR / (path.stem + ".out")
        target.write_text(json.dumps(output, separators=(",", ":"), allow_nan=False) + "\n",
                          encoding="utf-8")


if __name__ == "__main__":
    main()
