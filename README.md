# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/wildanfadhil888-collab/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/wildanfadhil888-collab/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Leonida Refresh Forecaster — a refresh-aware earners/players forecast of each of the 55 PlayStation 5 trophies of Grand Theft Auto VI and its Exophase earn rate at 2026-11-26T00:00:00Z, as five quantiles. It models Exophase's refreshes in counts, not rates: each refresh brings a new player cohort, the cohort's size and cadence are fitted from the reads, trophy-level earn probabilities are pooled through a log-linear rate curve with tier offsets and per-trophy shrinkage, and a fixed-seed Monte-Carlo over future refreshes gives the quantiles. With no refresh observed yet it falls back to a six-profile launch-night cohort prior built from the kit's published launch-night snapshot. Python 3 standard library only, no network, deterministic.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

It reads every `*.in` case in the input directory and writes one `*.out` forecast per case to the output directory.

Licensed under the MIT licence; see LICENSE.

---

Entered on [make.fun](https://make.fun/?ref=x03vwtz9&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards
