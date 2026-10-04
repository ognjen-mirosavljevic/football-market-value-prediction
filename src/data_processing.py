"""Loading and aggregation utilities for FBref-style player statistics."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


BASIC_FEATURES = ["Player", "Nation", "Pos", "Squad", "Comp", "Age", "Min"]
ATTACKING_FEATURES = ["Gls", "Ast", "xG", "xAG", "Sh", "SoT"]
PASSING_FEATURES = ["Cmp%", "xA", "KP", "PrgP", "PPA"]
POSSESSION_FEATURES = ["Touches", "Succ", "Succ%", "PrgC", "CPA", "Mis", "Dis"]
DEFENSIVE_FEATURES = ["Tkl", "TklW", "Int", "Blocks_stats_defense", "Clr", "Err"]
MISC_FEATURES = ["Recov", "Won", "Lost_stats_misc", "Won%"]
AGGREGATION_HELPERS = ["Cmp", "Att", "Att_stats_possession"]

SELECTED_FEATURES = list(
    dict.fromkeys(
        BASIC_FEATURES
        + ATTACKING_FEATURES
        + PASSING_FEATURES
        + POSSESSION_FEATURES
        + DEFENSIVE_FEATURES
        + MISC_FEATURES
        + AGGREGATION_HELPERS
    )
)

SUM_FEATURES = [
    "Min",
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

# Same-name, same-age players cannot be separated safely from FBref metadata alone.
# These verified collisions are deliberately explicit and auditable.
DEFAULT_COLLISION_KEYS = {("Vitinha", 24)}


def load_player_stats(path: str) -> pd.DataFrame:
    """Load the raw performance table."""
    return pd.read_csv(path)


def select_outfield_players(df: pd.DataFrame) -> pd.DataFrame:
    """Keep outfield rows and only columns used by the project pipeline."""
    missing = sorted(set(SELECTED_FEATURES) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required performance columns: {missing}")
    return df.loc[df["Pos"] != "GK", SELECTED_FEATURES].copy()


def add_identity_key(
    df: pd.DataFrame,
    collision_keys: Iterable[tuple[str, int]] = DEFAULT_COLLISION_KEYS,
) -> pd.DataFrame:
    """Create a stable identity key and split explicitly verified collisions."""
    result = df.copy()
    age_key = result["Age"].map(lambda value: str(int(value)) if pd.notna(value) else "NA")
    result["identity_key"] = result["Player"].astype(str) + "_" + age_key

    for player, age in collision_keys:
        mask = (result["Player"] == player) & (result["Age"] == age)
        result.loc[mask, "identity_key"] = (
            result.loc[mask, "identity_key"]
            + "_"
            + result.loc[mask, "Squad"].astype(str)
        )
    return result


def safe_percentage(numerator: float, denominator: float) -> float:
    """Return a percentage, preserving undefined zero-denominator cases as NaN."""
    if pd.isna(denominator) or denominator == 0:
        return np.nan
    return numerator / denominator * 100


def aggregate_player(group: pd.DataFrame) -> dict[str, object]:
    """Aggregate club rows for one identity into a single season record."""
    main_row = group.loc[group["Min"].fillna(-1).idxmax()]
    result: dict[str, object] = {
        "identity_key": group["identity_key"].iloc[0],
        "Player": group["Player"].iloc[0],
        "Nation": group["Nation"].dropna().iloc[0] if group["Nation"].notna().any() else np.nan,
        "Age": group["Age"].dropna().iloc[0] if group["Age"].notna().any() else np.nan,
        "Pos": str(main_row["Pos"]).split(",")[0],
        "Squad": " / ".join(group["Squad"].dropna().astype(str).unique()),
        "MainSquad": main_row["Squad"],
        "Comp": main_row["Comp"],
    }

    for feature in SUM_FEATURES:
        result[feature] = group[feature].sum(min_count=1)

    total_cmp = group["Cmp"].sum(min_count=1)
    total_passes = group["Att"].sum(min_count=1)
    total_takeons = group["Att_stats_possession"].sum(min_count=1)
    total_won = group["Won"].sum(min_count=1)
    total_lost = group["Lost_stats_misc"].sum(min_count=1)
    result["Cmp%"] = safe_percentage(total_cmp, total_passes)
    result["Succ%"] = safe_percentage(result["Succ"], total_takeons)
    result["Won%"] = safe_percentage(total_won, total_won + total_lost)
    return result


def aggregate_players(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a selected outfield table and validate identity conservation."""
    keyed = add_identity_key(df) if "identity_key" not in df.columns else df.copy()
    records = [
        aggregate_player(group)
        for _, group in keyed.groupby("identity_key", dropna=False, sort=True)
    ]
    aggregated = pd.DataFrame(records)
    if len(aggregated) != keyed["identity_key"].nunique(dropna=False):
        raise AssertionError("Identity conservation failed during aggregation")
    if aggregated["identity_key"].duplicated().any():
        raise AssertionError("Aggregated identity_key must be unique")
    return aggregated


def build_aggregated_dataset(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load, select and aggregate raw performance data."""
    selected = add_identity_key(select_outfield_players(load_player_stats(path)))
    return aggregate_players(selected), selected
