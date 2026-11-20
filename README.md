# gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Soukuri/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Soukuri/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A quantile forecaster for every Grand Theft Auto VI PlayStation 5 trophy's Exophase earn rate one week after launch. Each trophy's rate is the ratio of two moving quantities — players earning it and new players being tracked — so the program forecasts both: a logit-linear trend of the rate against log time since release, shrunk toward its tier's median slope, blended with an earned-counts view that fits log earned-counts and log players-tracked separately and takes their ratio at the target. Chapter trophies are forced non-increasing in chapter order and the platinum is capped below the rarest other trophy, and the spread is honest about what two days of data cannot know: a random-walk term over the 4.5-day horizon, the trend fit's residuals, and a wide allowance whenever a series has never moved at all, because a flat launch-night history hides Exophase refreshes that slash every rate at once. Validated on the kit's practice cases (valid, deterministic, 0.1 s a run) and on a synthetic final-like case with 340 reads of real growth.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `<id>.in` case in the input directory and writes `<id>.out` quantile forecasts (Python 3.13, standard library only, offline and deterministic).

<sub>Entered on [make.fun](https://make.fun/?ref=xvx3ahvr&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
