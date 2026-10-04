"""Entity matching between performance data and Transfermarkt valuations."""

from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process


MATCH_COLUMNS = [
    "identity_key",
    "Player",
    "Age",
    "player_id",
    "market_value_in_eur",
    "match_method",
]


def normalize_text(value: object) -> str:
    """Lowercase, remove accents/punctuation and collapse whitespace."""
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value).lower().strip())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text)).strip()


def club_similarity(fbref_clubs: object, tm_club: object) -> float:
    """Compare a Transfermarkt club with every slash-separated FBref club."""
    if pd.isna(tm_club):
        return 0.0
    target = normalize_text(tm_club)
    clubs = str(fbref_clubs).split("/")
    return max((fuzz.token_set_ratio(normalize_text(club), target) for club in clubs), default=0.0)


def age_score(difference: float) -> int:
    """Convert an age difference into the matching score used by the project."""
    if pd.isna(difference):
        return 0
    return {0: 100, 1: 80, 2: 50}.get(int(difference), 0)


def prepare_market_values(
    players_path: str,
    valuations_path: str,
    season_start: str = "2024-07-01",
    season_end: str = "2025-06-30",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the last in-season valuation per player and the full valuation history."""
    players = pd.read_csv(players_path)
    valuations = pd.read_csv(valuations_path)
    valuations["date"] = pd.to_datetime(valuations["date"], errors="coerce")
    in_season = valuations.loc[
        valuations["date"].between(pd.Timestamp(season_start), pd.Timestamp(season_end))
    ]
    latest = (
        in_season.sort_values("date")
        .groupby("player_id", as_index=False)
        .tail(1)[["player_id", "date", "market_value_in_eur", "current_club_name"]]
    )
    market = latest.merge(
        players[["player_id", "name", "date_of_birth", "position"]],
        on="player_id",
        how="left",
        validate="one_to_one",
    )
    market["normalized_name"] = market["name"].map(normalize_text)
    return market, valuations


def _tm_age(frame: pd.DataFrame, reference_date: str) -> pd.Series:
    birth = pd.to_datetime(frame["date_of_birth"], errors="coerce")
    return np.floor((pd.Timestamp(reference_date) - birth).dt.days / 365.25)


def match_players(
    aggregated: pd.DataFrame,
    market: pd.DataFrame,
    reference_date: str = "2024-08-01",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run exact, normalized, fuzzy and ambiguous-exact matching stages."""
    fbref_name_counts = aggregated["Player"].value_counts()
    tm_name_counts = market["name"].value_counts()
    unique_fbref = set(fbref_name_counts[fbref_name_counts == 1].index)
    unique_tm = set(tm_name_counts[tm_name_counts == 1].index)

    safe_fbref = aggregated.loc[aggregated["Player"].isin(unique_fbref)].copy()
    unique_market = market.loc[market["name"].isin(unique_tm)].copy()
    exact_candidates = safe_fbref.merge(
        unique_market,
        left_on="Player",
        right_on="name",
        how="left",
        validate="one_to_one",
    )
    exact = exact_candidates.loc[exact_candidates["player_id"].notna()].copy()
    exact["match_method"] = "exact"

    unmatched = exact_candidates.loc[
        exact_candidates["player_id"].isna(), aggregated.columns
    ].copy()
    unmatched["normalized_name"] = unmatched["Player"].map(normalize_text)
    normalized_counts = market["normalized_name"].value_counts()
    unique_normalized = market.loc[
        market["normalized_name"].isin(normalized_counts[normalized_counts == 1].index)
    ]
    normalized_candidates = unmatched.merge(
        unique_normalized,
        on="normalized_name",
        how="left",
        suffixes=("", "_tm"),
        validate="many_to_one",
    )
    normalized = normalized_candidates.loc[normalized_candidates["player_id"].notna()].copy()
    normalized["match_method"] = "normalized"

    still_unmatched = normalized_candidates.loc[
        normalized_candidates["player_id"].isna(), aggregated.columns
    ].copy()
    still_unmatched["normalized_name"] = still_unmatched["Player"].map(normalize_text)
    tm_names = market["normalized_name"].dropna().unique().tolist()

    def best_name(name: str) -> tuple[str | None, float]:
        match = process.extractOne(name, tm_names, scorer=fuzz.ratio)
        return (None, 0.0) if match is None else (match[0], float(match[1]))

    best = still_unmatched["normalized_name"].map(best_name)
    still_unmatched[["best_tm_name", "name_score"]] = pd.DataFrame(
        best.tolist(), index=still_unmatched.index
    )
    fuzzy_candidates = still_unmatched.merge(
        market,
        left_on="best_tm_name",
        right_on="normalized_name",
        how="left",
        suffixes=("_fbref", "_tm"),
    )
    fuzzy_candidates["club_score"] = fuzzy_candidates.apply(
        lambda row: club_similarity(row["Squad"], row["current_club_name"]), axis=1
    )
    fuzzy_candidates["tm_age"] = _tm_age(fuzzy_candidates, reference_date)
    fuzzy_candidates["age_diff"] = (fuzzy_candidates["Age"] - fuzzy_candidates["tm_age"]).abs()
    fuzzy_candidates["age_score"] = fuzzy_candidates["age_diff"].map(age_score)
    fuzzy_candidates["candidate_score"] = (
        0.50 * fuzzy_candidates["name_score"]
        + 0.30 * fuzzy_candidates["club_score"]
        + 0.20 * fuzzy_candidates["age_score"]
    )
    best_fuzzy = (
        fuzzy_candidates.sort_values("candidate_score", ascending=False)
        .groupby("identity_key", as_index=False, dropna=False)
        .first()
    )
    fuzzy = best_fuzzy.loc[
        (best_fuzzy["name_score"] >= 90)
        & (best_fuzzy["club_score"] >= 70)
        & (best_fuzzy["age_diff"] <= 1)
    ].copy()
    fuzzy["match_method"] = "fuzzy"

    ambiguous_fbref = aggregated.loc[~aggregated["Player"].isin(unique_fbref)].copy()
    ambiguous_candidates = ambiguous_fbref.merge(
        market,
        left_on="Player",
        right_on="name",
        how="inner",
        suffixes=("", "_tm"),
    )
    ambiguous_candidates["club_score"] = ambiguous_candidates.apply(
        lambda row: club_similarity(row["Squad"], row["current_club_name"]), axis=1
    )
    ambiguous_candidates["tm_age"] = _tm_age(ambiguous_candidates, reference_date)
    ambiguous_candidates["age_diff"] = (
        ambiguous_candidates["Age"] - ambiguous_candidates["tm_age"]
    ).abs()
    ambiguous_candidates["age_score"] = ambiguous_candidates["age_diff"].map(age_score)
    ambiguous_candidates["candidate_score"] = (
        0.60 * ambiguous_candidates["club_score"]
        + 0.40 * ambiguous_candidates["age_score"]
    )
    best_ambiguous = (
        ambiguous_candidates.sort_values("candidate_score", ascending=False)
        .groupby("identity_key", as_index=False, dropna=False)
        .first()
    )
    ambiguous = best_ambiguous.loc[
        (best_ambiguous["club_score"] >= 70) & (best_ambiguous["age_diff"] <= 1)
    ].copy()
    ambiguous["match_method"] = "ambiguous_exact"

    matched = pd.concat(
        [stage[MATCH_COLUMNS] for stage in (exact, normalized, fuzzy, ambiguous)],
        ignore_index=True,
    )
    matched = matched.drop_duplicates("identity_key", keep="first")
    if matched["player_id"].duplicated().any():
        duplicates = matched.loc[matched["player_id"].duplicated(False), "player_id"].tolist()
        raise AssertionError(f"Transfermarkt IDs matched more than once: {duplicates[:10]}")

    unresolved = aggregated.loc[~aggregated["identity_key"].isin(matched["identity_key"])].copy()
    if len(matched) + len(unresolved) != len(aggregated):
        raise AssertionError("Matching conservation check failed")
    return matched, unresolved


def build_final_dataset(
    aggregated: pd.DataFrame,
    market: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Match identities and return final, match-audit and unresolved tables."""
    matches, unresolved = match_players(aggregated, market)
    final = aggregated.merge(
        matches,
        on="identity_key",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_match"),
    )
    if len(final) != len(matches):
        raise AssertionError("Final merge lost accepted matches")
    return final, matches, unresolved
