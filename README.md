# Football Player Market Value Prediction

A machine learning project for predicting football player market values based on their on-pitch performance statistics, player characteristics, and competition data.

The project combines football performance statistics with historical Transfermarkt market valuations to build a dataset suitable for training regression models.

## Project Overview

Football player market value depends on many factors, including age, position, playing time, attacking and defensive performance, and the competition in which the player plays.

The goal of this project is to explore how well these factors can be used to estimate a player's market value using machine learning.

The project currently focuses on outfield players, while goalkeepers will be handled separately because their performance metrics differ significantly.

## Data

The project combines two main data sources:

- Player performance statistics for the 2024/25 season
- Historical player market valuations from Transfermarkt data

The performance dataset contains metrics related to:

- Playing time
- Goals and assists
- Expected goals (xG) and expected assisted goals (xAG)
- Shooting
- Passing and progressive passing
- Possession and progressive carries
- Take-ons
- Defensive actions
- Ball recoveries
- Aerial duels

## Data Processing

Several preprocessing steps are performed before modeling.

### Player Aggregation

Players may appear multiple times in the statistics dataset after changing clubs during the season.

Their statistics are aggregated into a single player record while preserving relevant metadata such as clubs, primary position, and main competition.

### Percentage Statistics

Percentage-based statistics such as:

- Pass completion percentage
- Take-on success percentage
- Aerial duel win percentage

are recalculated from their underlying totals instead of averaging percentages across clubs.

### Market Value Matching

Player statistics and market valuation data originate from different datasets and do not share a common player identifier.

A multi-stage entity matching pipeline is therefore used:

1. Exact name matching
2. Normalized name matching
3. Fuzzy name matching
4. Club similarity validation
5. Age validation
6. Disambiguation of players sharing the same name

Only high-confidence matches are retained.

From 2,494 unique outfield players in the processed statistics dataset, 2,139 players were successfully matched with market valuation data.

| Matching method | Players |
|---|---:|
| Exact name | 1,939 |
| Normalized name | 133 |
| High-confidence fuzzy | 44 |
| Ambiguous exact name | 23 |
| **Total** | **2,139** |

This corresponds to approximately **85.8%** of the processed outfield player dataset.

## Target Variable

The prediction target is the player's market value in euros.

Initial exploratory analysis shows that market values are strongly right-skewed, with a relatively small number of highly valuable players.

For the matched dataset:

- Median market value: €6.0M
- Mean market value: €12.8M
- Maximum market value: €200M

A logarithmic transformation of the target is being explored to reduce skewness and improve regression performance.

```python
log_market_value = np.log1p(market_value_in_eur)
