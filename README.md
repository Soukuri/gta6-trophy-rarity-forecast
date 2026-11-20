# gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Soukuri/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Soukuri/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A quantile forecaster for every Grand Theft Auto VI PlayStation 5 trophy's Exophase earn rate one week after launch. The recorded page is a step function - Exophase refreshes its counts every few hours - so everything is fitted on refresh segments (maximal runs of identical reads), never on raw reads. Players-tracked and each trophy's earners are forecast separately with recency-weighted log-log fits over the last 18 hours of segments, anchored at the last segment with shrunk growth, and the rate is their ratio, blended with a segment-level logit trend of the rate shrunk toward its tier's median slope. When the page in view has never refreshed (the launch-night shape), the forecast instead prices the refresh that likely lands before the target: players multiply faster than earners - deepest trophies slowest - calibrated on the one refresh the house documented (638 -> 3,868 players in under five hours, non-automatic earners up only x1.3). Chapter trophies are forced non-increasing in chapter order, the final-mission trophy is capped at the last chapter's median, and the platinum below the rarest other trophy; the spread combines a random-walk term, fit residuals, a Poisson term for few-earner trophies and a wide allowance for series that have never moved. Validated on the kit's practice cases (valid, deterministic, ~0.1 s a run) and on simulated launch histories with refresh steps, where it cuts the previous version's mean pinball loss by roughly a third on final-shaped cases and over 40% on practice-shaped ones.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `<id>.in` case in the input directory and writes `<id>.out` quantile forecasts (Python 3.13, standard library only, offline and deterministic).

<sub>Entered on [make.fun](https://make.fun/?ref=xvx3ahvr&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
