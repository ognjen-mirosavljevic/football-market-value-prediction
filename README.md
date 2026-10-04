# Football Player Market Value Prediction

A machine learning project for predicting football players' market values based on on-field performance, player characteristics, competition, and club context.

The pipeline combines 2024/25 season performance statistics with historical Transfermarkt valuations. The current version focuses on outfield players; goalkeepers are handled separately because they require a different set of performance metrics.

## Project Structure

```text
football-market-value-prediction/
├── data/
│   ├── raw/                         # source CSV files
│   └── processed/                   # generated intermediate outputs (gitignored)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_data_matching.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_modeling.ipynb
├── src/
│   ├── data_processing.py
│   ├── matching.py
│   └── features.py
├── models/
├── README.md
├── requirements.txt
└── .gitignore
```

## Notebook Workflow

The notebooks should be run in order:

1. **`01_eda.ipynb`** — data quality analysis, duplicates, missing values, goalkeeper/outfield split, playing time, initial feature selection, and target distribution.
2. **`02_data_matching.ipynb`** — player aggregation, `identity_key`, `MainSquad`, seasonal valuations, and exact → normalized → fuzzy → ambiguous entity matching. The output is `data/processed/final_players.csv`, together with audit tables for accepted and unresolved identities.
3. **`03_feature_engineering.ipynb`** — log-transformed target, availability indicators, a single train/test split, raw and per-90 features, and reliability shrinkage. The outputs are `train_features.csv` and `test_features.csv`.
4. **`04_modeling.ipynb`** — Models A–I, Random Forest and XGBoost feature importance, Model H with club context, Model I with previous market value, ablation analysis, error analysis, and visualizations.

Shared logic is stored in `src/`, keeping the notebooks concise and focused on explanation and results.

## Key Methodological Decisions

### Player Aggregation and Identity

A player may have multiple rows after changing clubs during the season.

Counting statistics are summed, while `MainSquad`, primary position, and competition are taken from the row in which the player recorded the most minutes. Percentage-based statistics are recalculated from their aggregated numerators and denominators.

Known same-name/same-age identity collisions are resolved explicitly and in an auditable manner in `src/data_processing.py`.

### Entity Matching

The performance and Transfermarkt datasets do not share a common identifier.

Entity matching therefore follows four stages:

1. unique exact-name matching;
2. unique normalized-name matching;
3. fuzzy name matching with club and age validation;
4. disambiguation of exact-name collisions using club and age.

Conservation checks ensure that player identities are neither duplicated nor silently lost during join operations.

### Reliability-Adjusted Per-90 Features

Naive per-90 statistics can become extreme for players with very limited playing time.

To reduce this instability, each player's rate is shrunk toward the positional average:

```text
w = Min / (Min + 450)

adjusted = w * player_per90 + (1 - w) * position_baseline
```

The positional baseline is calculated using only the training set to prevent data leakage.

### Target

Market values are strongly right-skewed, so the models predict a log-transformed target:

```python
log_market_value = np.log1p(market_value_in_eur)
```

Evaluation metrics are reported both in log space and in euros.

## Modeling

The project evaluates a sequence of increasingly capable models:

- Raw Ridge Regression
- Naive per-90 Ridge Regression
- Reliability-adjusted per-90 Ridge Regression
- Random Forest
- Tuned Random Forest
- XGBoost
- Tuned XGBoost
- XGBoost with club context
- XGBoost with previous market value

This progression makes it possible to evaluate the effect of feature engineering, nonlinear models, club context, and historical valuation information independently.

## Results

The best performance-based model uses XGBoost with player statistics, age, playing time, competition, position, and club context.

It achieved:

| Metric | Model H |
|---|---:|
| MAE (log) | 0.5203 |
| RMSE (log) | 0.6805 |
| R² | 0.7829 |
| MAE (€) | €5.29M |

Adding the player's most recent market valuation from before the start of the 2024/25 season creates a separate forecasting model.

Model I achieved:

| Metric | Model I |
|---|---:|
| MAE (log) | 0.3294 |
| RMSE (log) | 0.4711 |
| R² | 0.8959 |
| MAE (€) | €3.43M |

These models represent two related but distinct tasks:

- **Model H — Performance-based valuation:** estimates market value from current-season football performance and context.
- **Model I — Market-value forecasting:** predicts a new valuation using both historical valuation and current-season information.

## Ablation Study

To measure how much information comes from historical valuation versus current-season performance, the models were compared on the same subset of 415 test players with an available previous valuation.

| Feature Set | MAE (log) | RMSE (log) | R² | MAE (€) |
|---|---:|---:|---:|---:|
| Previous value only | 0.5161 | 0.7753 | 0.6767 | €5.22M |
| Performance + context | 0.5072 | 0.6631 | 0.7636 | €5.43M |
| Previous value + performance + context | **0.3116** | **0.4384** | **0.8967** | **€3.48M** |

The results indicate that historical valuation and current-season performance provide complementary information.

The combined model substantially outperforms both the historical-value baseline and the performance-only model.

## Error Analysis

Error analysis showed that the forecasting model performs best when player valuations remain relatively stable.

Large market-value changes are more difficult to predict, particularly for breakout players and major market repricing events.

The correlation between absolute market-value change and absolute prediction error was approximately:

```text
0.56
```

This suggests that prediction difficulty increases as the magnitude of a player's market-value change increases.

The model also tends to underestimate some extreme high-value players, illustrating the difficulty of predicting superstar valuations using structured performance data alone.

## Running the Project

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it and install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

Launch Jupyter:

```bash
jupyter notebook
```

Then run the notebooks sequentially:

```text
01_eda.ipynb
        ↓
02_data_matching.ipynb
        ↓
03_feature_engineering.ipynb
        ↓
04_modeling.ipynb
```

`RUN_TUNING = False` in the modeling notebook uses fixed hyperparameters for fast and reproducible execution.

Set:

```python
RUN_TUNING = True
```

to run `RandomizedSearchCV` for the tuned Random Forest and XGBoost models.

## Data and Generated Files

Input datasets are stored in `data/raw/`.

Files in `data/processed/` and trained models are generated artifacts and are not version-controlled, except for `.gitkeep` placeholder files.

The processing pipeline can regenerate these artifacts by running the notebooks in order.

## Limitations

The current version focuses on outfield players. Goalkeepers require position-specific performance metrics and are therefore excluded from the current modeling pipeline.

Market value is influenced by factors that are not fully captured by match statistics, including reputation, contracts, transfer demand, injuries, international performances, and broader market conditions.

The current evaluation also uses a random train/test split. A future version could use temporal validation across multiple seasons to evaluate performance on genuinely unseen future data.

## Future Work

Potential extensions include:

- a dedicated goalkeeper model;
- multi-season performance data;
- temporal train/test validation;
- contract information;
- international appearances;
- improved modeling of breakout players;
- model explainability using SHAP;
- automated prediction pipelines for future seasons.

## Tech Stack

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- RapidFuzz
- Matplotlib
- Jupyter
