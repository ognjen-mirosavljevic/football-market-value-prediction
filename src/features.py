"""Leakage-safe feature engineering for market-value models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


PERCENTAGE_FEATURES = ["Cmp%", "Succ%", "Won%"]
PER90_FEATURES = [
    "Gls",
    "Ast",
    "xG",
    "xAG",
    "Sh",
    "SoT",
    "xA",
    "KP",
    "PrgP",
    "PPA",
    "Touches",
    "Succ",
    "PrgC",
    "CPA",
    "Mis",
    "Dis",
    "Tkl",
    "TklW",
    "Int",
    "Blocks_stats_defense",
    "Clr",
    "Err",
    "Recov",
    "Won",
    "Lost_stats_misc",
]
RAW_NUMERIC_FEATURES = ["Age", "Min", *PER90_FEATURES, *PERCENTAGE_FEATURES]
AVAILABILITY_FEATURES = [f"{feature}_available" for feature in PERCENTAGE_FEATURES]
CATEGORICAL_FEATURES = ["Pos", "Comp"]


def prepare_modeling_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Add the log target and missingness indicators without imputing values."""
    result = df.copy()
    result["log_market_value"] = np.log1p(result["market_value_in_eur"])
    for feature in PERCENTAGE_FEATURES:
        result[f"{feature}_available"] = result[feature].notna().astype("int8")
    return result


def add_per90_features(df: pd.DataFrame, features: list[str] = PER90_FEATURES) -> pd.DataFrame:
    """Add naive per-90 rates; zero-minute rows remain undefined."""
    result = df.copy()
    minutes = result["Min"].replace(0, np.nan)
    for feature in features:
        result[f"{feature}_per90"] = result[feature] / minutes * 90
    return result


def add_shrunk_per90_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str] = PER90_FEATURES,
    k: float = 450,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shrink per-90 rates toward position baselines learned on training data only."""
    train_result = add_per90_features(train, features)
    test_result = add_per90_features(test, features)
    train_weight = train_result["Min"] / (train_result["Min"] + k)
    test_weight = test_result["Min"] / (test_result["Min"] + k)

    for feature in features:
        per90 = f"{feature}_per90"
        adjusted = f"{feature}_per90_adjusted"
        totals = train_result.groupby("Pos")[[feature, "Min"]].sum(min_count=1)
        position_baseline = totals[feature] / totals["Min"].replace(0, np.nan) * 90
        global_baseline = (
            train_result[feature].sum(min_count=1)
            / train_result["Min"].sum(min_count=1)
            * 90
        )
        train_baseline = train_result["Pos"].map(position_baseline).fillna(global_baseline)
        test_baseline = test_result["Pos"].map(position_baseline).fillna(global_baseline)
        train_result[adjusted] = train_weight * train_result[per90] + (1 - train_weight) * train_baseline
        test_result[adjusted] = test_weight * test_result[per90] + (1 - test_weight) * test_baseline
    return train_result, test_result


def build_feature_sets(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    shrinkage_k: float = 450,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create one reproducible split and all raw/per-90/shrunk feature variants."""
    modeling = prepare_modeling_frame(df)
    train, test = train_test_split(
        modeling,
        test_size=test_size,
        random_state=random_state,
    )
    train, test = add_shrunk_per90_features(train, test, k=shrinkage_k)
    return train, test


def add_previous_market_value(
    df: pd.DataFrame,
    valuations: pd.DataFrame,
    season_start: str = "2024-07-01",
) -> pd.DataFrame:
    """Attach the last strictly pre-season valuation for Model I."""
    history = valuations.copy()
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    previous = (
        history.loc[history["date"] < pd.Timestamp(season_start)]
        .sort_values("date")
        .groupby("player_id", as_index=False)
        .tail(1)[["player_id", "date", "market_value_in_eur"]]
        .rename(
            columns={
                "date": "previous_value_date",
                "market_value_in_eur": "previous_market_value",
            }
        )
    )
    result = df.merge(previous, on="player_id", how="left", validate="one_to_one")
    result["log_previous_market_value"] = np.log1p(result["previous_market_value"])
    result["previous_value_available"] = result["previous_market_value"].notna().astype("int8")
    return result
