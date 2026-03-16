"""
Issue Detection Pipeline for feedback analysis.

Purpose:
- Read sentiment-enriched review files (typically *_senti.csv)
- Detect issue categories using a multilingual keyword taxonomy (de/en)
- Add issue confidence and severity signals
- Produce summary and trend tables for product insights

Main entry point:
- run_issue_detection_pipeline(...)

Typical output:
- *_issues.csv with columns such as:
    issue_primary, issue_labels, issue_confidence,
    issue_severity_score, issue_severity_level
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd


FUNCTIONS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FUNCTIONS_DIR.parent
DEFAULT_SEARCH_DIR = PROJECT_ROOT / "data"
TEXT_COLUMN_CANDIDATES = ("content_clean", "clean_text", "content")

ISSUE_TAXONOMY: dict[str, list[str]] = {
    "login_auth": [
        "login",
        "sign in",
        "signin",
        "authentication",
        "auth",
        "password",
        "passwort",
        "anmelden",
        "anmeldung",
    ],
    "connection_pairing": [
        "connection",
        "connect",
        "pairing",
        "pair",
        "bluetooth",
        "wifi",
        "verbindung",
        "verbinden",
        "koppeln",
    ],
    "charging_range": [
        "charging",
        "charge",
        "battery",
        "range",
        "lade",
        "laden",
        "akku",
        "reichweite",
        "wallbox",
    ],
    "remote_services": [
        "remote",
        "remote start",
        "unlock",
        "lock",
        "precondition",
        "vorklimatisierung",
        "vorheizen",
        "entriegeln",
        "verriegeln",
    ],
    "navigation_maps": [
        "navigation",
        "map",
        "maps",
        "route",
        "navi",
        "karten",
        "verkehr",
    ],
    "performance_stability": [
        "crash",
        "freez",
        "lag",
        "slow",
        "bug",
        "error",
        "stürzt",
        "absturz",
        "hängt",
        "fehler",
    ],
    "update_version": [
        "update",
        "updated",
        "version",
        "release",
        "patch",
        "aktualisierung",
        "neue version",
    ],
    "ui_usability": [
        "ui",
        "ux",
        "menu",
        "menü",
        "usability",
        "bedienung",
        "kompliziert",
        "unübersichtlich",
    ],
}


def load_csv(path: str | Path) -> pd.DataFrame:
    """Loads a CSV file with UTF-8 BOM support."""
    return pd.read_csv(path, encoding="utf-8-sig")


def save_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """Saves a DataFrame as CSV with UTF-8 BOM encoding."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def find_senti_csv_files(search_dir: str | Path = DEFAULT_SEARCH_DIR) -> list[Path]:
    """Finds all *_senti.csv files below the provided directory."""
    search_dir = Path(search_dir)
    if not search_dir.exists():
        return []
    return sorted(path for path in search_dir.rglob("*_senti.csv") if path.is_file())


def get_senti_csv_options(search_dir: str | Path = DEFAULT_SEARCH_DIR) -> list[dict]:
    """Returns senti CSV options in a UI-friendly structure."""
    options = []
    for path in find_senti_csv_files(search_dir):
        try:
            label = str(path.relative_to(PROJECT_ROOT))
        except ValueError:
            label = str(path)

        options.append(
            {
                "label": label,
                "path": path,
                "path_str": str(path),
                "name": path.name,
                "stem": path.stem,
            }
        )

    return options


def build_issue_output_path(senti_path: str | Path) -> Path:
    """Builds the output path with *_issues.csv suffix."""
    senti_path = Path(senti_path)
    base_name = senti_path.stem[:-6] if senti_path.stem.endswith("_senti") else senti_path.stem
    return senti_path.with_name(f"{base_name}_issues.csv")


def resolve_senti_csv_path(
    selection: str | int | Path,
    senti_files: list[Path] | None = None,
    base_dir: str | Path = PROJECT_ROOT,
) -> Path:
    """Resolves senti CSV selection by index or path."""
    if isinstance(selection, int):
        if not senti_files:
            raise ValueError("No senti CSV files available for index selection.")
        index = selection - 1
        if 0 <= index < len(senti_files):
            return senti_files[index]
        raise ValueError("Selected file index is out of range.")

    candidate = Path(str(selection).strip().strip('"'))
    if not candidate.is_absolute():
        candidate = Path(base_dir) / candidate

    if candidate.exists() and candidate.is_file() and candidate.suffix.lower() == ".csv":
        return candidate

    raise ValueError("Invalid senti CSV selection.")


def choose_senti_csv_cli(senti_files: list[Path]) -> Path:
    """Prompts the user to choose a senti CSV file or enter a path."""
    if senti_files:
        print("Available senti CSV files:")
        for index, path in enumerate(senti_files, start=1):
            print(f"{index}. {path}")
        print()

    while True:
        selection = input("Select a file number or enter a file path: ").strip().strip('"')
        try:
            resolved_selection: str | int = int(selection) if selection.isdigit() else selection
            return resolve_senti_csv_path(resolved_selection, senti_files=senti_files)
        except ValueError:
            print("Invalid selection. Please try again.")


def resolve_text_column(
    df: pd.DataFrame,
    text_column: str | None = None,
    candidates: tuple[str, ...] = TEXT_COLUMN_CANDIDATES,
) -> str:
    """Returns the text column used for issue detection."""
    if text_column and text_column in df.columns:
        return text_column

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    raise ValueError("No usable text column found. Provide one of: " + ", ".join(candidates))


def _normalize_text(text: Any) -> str:
    raw = "" if pd.isna(text) else str(text)
    text_lower = raw.lower()
    text_clean = re.sub(r"\s+", " ", text_lower)
    return text_clean.strip()


def _keyword_in_text(text: str, keyword: str) -> bool:
    keyword_norm = keyword.lower().strip()
    if not keyword_norm:
        return False

    if " " in keyword_norm:
        return keyword_norm in text

    pattern = r"\b" + re.escape(keyword_norm) + r"\b"
    return re.search(pattern, text) is not None


def detect_issue_matches(
    text: str,
    taxonomy: dict[str, list[str]] = ISSUE_TAXONOMY,
) -> dict[str, list[str]]:
    """Returns matched issue keywords per issue category for a single text."""
    matches: dict[str, list[str]] = {}
    normalized_text = _normalize_text(text)

    if not normalized_text:
        return matches

    for category, keywords in taxonomy.items():
        found_keywords = [keyword for keyword in keywords if _keyword_in_text(normalized_text, keyword)]
        if found_keywords:
            matches[category] = sorted(set(found_keywords))

    return matches


def build_issue_result_from_matches(
    matches: dict[str, list[str]],
    include_unknown: bool = True,
    max_labels: int = 3,
) -> dict:
    """Builds labels, primary label and confidence from keyword matches."""
    if not matches:
        labels = ["unknown"] if include_unknown else []
        return {
            "issue_primary": labels[0] if labels else None,
            "issue_labels": labels,
            "issue_keyword_hits": 0,
            "issue_confidence": 0.0,
            "issue_matches": {},
        }

    ranked = sorted(matches.items(), key=lambda item: (-len(item[1]), item[0]))
    selected = ranked[:max(1, max_labels)]

    labels = [item[0] for item in selected]
    selected_matches = {item[0]: item[1] for item in selected}
    total_hits = sum(len(values) for values in selected_matches.values())
    top_hits = len(selected_matches[labels[0]]) if labels else 0

    confidence = min(1.0, top_hits / 3.0)

    return {
        "issue_primary": labels[0] if labels else ("unknown" if include_unknown else None),
        "issue_labels": labels,
        "issue_keyword_hits": int(total_hits),
        "issue_confidence": float(confidence),
        "issue_matches": selected_matches,
    }


def detect_issues_in_dataframe(
    df: pd.DataFrame,
    text_column: str,
    taxonomy: dict[str, list[str]] = ISSUE_TAXONOMY,
    include_unknown: bool = True,
    max_labels: int = 3,
) -> pd.DataFrame:
    """Detects issue labels per row and appends issue-related columns."""
    result_df = df.copy()
    text_series = result_df[text_column].fillna("").astype(str)

    issue_records = []
    for text in text_series:
        matches = detect_issue_matches(text, taxonomy=taxonomy)
        issue_info = build_issue_result_from_matches(
            matches,
            include_unknown=include_unknown,
            max_labels=max_labels,
        )
        issue_records.append(issue_info)

    issue_df = pd.DataFrame(issue_records, index=result_df.index)
    result_df["issue_primary"] = issue_df["issue_primary"].fillna("unknown")
    result_df["issue_labels"] = issue_df["issue_labels"].apply(lambda labels: "|".join(labels) if labels else "")
    result_df["issue_keyword_hits"] = pd.to_numeric(issue_df["issue_keyword_hits"], errors="coerce").fillna(0).astype(int)
    result_df["issue_confidence"] = pd.to_numeric(issue_df["issue_confidence"], errors="coerce").fillna(0.0)
    result_df["issue_matches_json"] = issue_df["issue_matches"].apply(json.dumps)
    return result_df


def _normalize_thumbs(thumbs: pd.Series) -> pd.Series:
    thumbs_numeric = pd.to_numeric(thumbs, errors="coerce").fillna(0)
    max_value = float(thumbs_numeric.max())
    if max_value <= 0:
        return pd.Series(0.0, index=thumbs_numeric.index)
    return thumbs_numeric.apply(lambda value: math.log1p(value) / math.log1p(max_value))


def _compute_recency_score(date_series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(date_series, errors="coerce", utc=True)
    if parsed.notna().sum() == 0:
        return pd.Series(0.5, index=date_series.index)

    timestamps = parsed.astype("int64", copy=False)
    min_ts = timestamps[parsed.notna()].min()
    max_ts = timestamps[parsed.notna()].max()
    if max_ts == min_ts:
        return pd.Series(0.5, index=date_series.index)

    recency = (timestamps - min_ts) / (max_ts - min_ts)
    recency = recency.where(parsed.notna(), 0.5)
    return recency.astype(float)


def add_issue_severity_scores(
    df: pd.DataFrame,
    sentiment_column: str = "sentiment",
    thumbs_column: str = "thumbsUpCount",
    date_column: str = "at",
    confidence_column: str = "issue_confidence",
) -> pd.DataFrame:
    """Adds severity score and level columns based on sentiment, impact and recency."""
    result_df = df.copy()

    sentiment_series = result_df.get(sentiment_column, pd.Series(index=result_df.index, dtype="object"))
    sentiment_weight = sentiment_series.fillna("").astype(str).str.lower().map(
        {
            "negative": 1.0,
            "positive": 0.25,
        }
    ).fillna(0.5)

    thumbs_norm = _normalize_thumbs(result_df.get(thumbs_column, pd.Series(index=result_df.index, dtype="float")))
    recency_score = _compute_recency_score(result_df.get(date_column, pd.Series(index=result_df.index, dtype="object")))
    issue_confidence = pd.to_numeric(
        result_df.get(confidence_column, pd.Series(index=result_df.index, dtype="float")),
        errors="coerce",
    ).fillna(0.0)

    severity = (
        0.5 * sentiment_weight
        + 0.2 * thumbs_norm
        + 0.2 * recency_score
        + 0.1 * issue_confidence
    ).clip(0.0, 1.0)

    def to_level(score: float) -> str:
        if score >= 0.75:
            return "critical"
        if score >= 0.55:
            return "high"
        if score >= 0.35:
            return "medium"
        return "low"

    result_df["issue_severity_score"] = severity
    result_df["issue_severity_level"] = severity.apply(to_level)
    return result_df


def summarize_issues(
    df: pd.DataFrame,
    issue_column: str = "issue_primary",
    sentiment_column: str = "sentiment",
) -> pd.DataFrame:
    """Builds an issue summary table with volume and negative share."""
    if issue_column not in df.columns:
        return pd.DataFrame()

    summary = df.copy()
    summary["is_negative"] = summary.get(sentiment_column, "").astype(str).str.lower().eq("negative")

    agg = summary.groupby(issue_column, dropna=False).agg(
        review_count=(issue_column, "size"),
        negative_share=("is_negative", "mean"),
    )

    if "issue_severity_score" in summary.columns:
        agg["avg_severity"] = summary.groupby(issue_column, dropna=False)["issue_severity_score"].mean()

    agg = agg.reset_index().sort_values(["review_count", issue_column], ascending=[False, True])
    return agg


def summarize_issue_trends(
    df: pd.DataFrame,
    date_column: str = "at",
    issue_column: str = "issue_primary",
    freq: str = "M",
) -> pd.DataFrame:
    """Builds issue trend counts by time period."""
    if issue_column not in df.columns:
        return pd.DataFrame()

    trend_df = df.copy()
    trend_df[date_column] = pd.to_datetime(trend_df.get(date_column), errors="coerce")
    trend_df = trend_df.dropna(subset=[date_column])
    if trend_df.empty:
        return pd.DataFrame()

    trend_df["period"] = trend_df[date_column].dt.to_period(freq).astype(str)
    trend = trend_df.groupby(["period", issue_column], dropna=False).size().reset_index(name="review_count")
    return trend.sort_values(["period", "review_count"], ascending=[True, False])


def run_issue_detection_pipeline(
    input_path: str | Path,
    output_path: str | Path | None = None,
    text_column: str | None = None,
    include_unknown: bool = True,
    max_labels: int = 3,
    add_severity: bool = True,
    save_if_changed: bool = True,
) -> dict:
    """Runs issue detection on a *_senti.csv and optionally saves *_issues.csv."""
    input_path = Path(input_path)
    df_original = load_csv(input_path)
    selected_text_column = resolve_text_column(df_original, text_column=text_column)

    df_result = detect_issues_in_dataframe(
        df_original,
        text_column=selected_text_column,
        include_unknown=include_unknown,
        max_labels=max_labels,
    )

    if add_severity:
        df_result = add_issue_severity_scores(df_result)

    has_changes = not df_result.equals(df_original)
    saved_path: Path | None = None

    if has_changes and save_if_changed:
        final_output_path = Path(output_path) if output_path else build_issue_output_path(input_path)
        saved_path = save_csv(df_result, final_output_path)

    issue_summary = summarize_issues(df_result)
    trend_summary = summarize_issue_trends(df_result)

    return {
        "input_path": input_path,
        "output_path": saved_path,
        "changed": has_changes,
        "saved": saved_path is not None,
        "text_column": selected_text_column,
        "row_count": int(len(df_result)),
        "issue_summary": issue_summary,
        "trend_summary": trend_summary,
        "dataframe": df_result,
    }


def main() -> None:
    """CLI entry point for issue detection."""
    senti_files = find_senti_csv_files()
    input_path = choose_senti_csv_cli(senti_files)
    result = run_issue_detection_pipeline(input_path)

    if result["saved"]:
        print(f"Issue file saved to: {result['output_path']}")
    else:
        print("No changes detected. No issue file saved.")

    summary = result["issue_summary"]
    if not summary.empty:
        print("Top detected issues:")
        print(summary.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
