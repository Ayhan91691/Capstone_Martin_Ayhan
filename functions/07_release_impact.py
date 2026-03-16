"""
Release Impact Analyzer for app feedback.

Purpose:
- Compare quality before/after app versions (updates)
- Quantify regression/improvement based on negative rate, score and severity
- Return a structured release-impact table for reporting and dashboards

Main entry point:
- run_release_impact_pipeline(...)

Typical artifact:
- *_release_impact.csv
- *_release_issue_trends.csv
- *_top10_rising_issues_per_version.csv
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pandas as pd


FUNCTIONS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FUNCTIONS_DIR.parent
DEFAULT_SEARCH_DIR = PROJECT_ROOT / "data"
TEXT_COLUMN_CANDIDATES = ("content_clean", "clean_text", "content")


def load_csv(path: str | Path) -> pd.DataFrame:
    """Loads a CSV file with UTF-8 BOM support."""
    return pd.read_csv(path, encoding="utf-8-sig")


def save_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """Saves a DataFrame as CSV with UTF-8 BOM encoding."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def find_feedback_files(search_dir: str | Path = DEFAULT_SEARCH_DIR) -> list[Path]:
    """Finds *_issues.csv and *_senti.csv files for release impact analysis."""
    search_dir = Path(search_dir)
    if not search_dir.exists():
        return []

    files = []
    for pattern in ("*_issues.csv", "*_senti.csv"):
        files.extend(path for path in search_dir.rglob(pattern) if path.is_file())

    return sorted(set(files))


def get_feedback_options(search_dir: str | Path = DEFAULT_SEARCH_DIR) -> list[dict]:
    """Returns feedback file options in a UI-friendly structure."""
    options = []
    for path in find_feedback_files(search_dir):
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


def _strip_known_suffix(stem: str) -> str:
    for suffix in ("_issues", "_senti", "_clean"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def build_release_output_path(input_path: str | Path) -> Path:
    """Builds output path with *_release_impact.csv suffix."""
    input_path = Path(input_path)
    base_name = _strip_known_suffix(input_path.stem)
    return input_path.with_name(f"{base_name}_release_impact.csv")


def build_release_issue_trend_output_path(input_path: str | Path) -> Path:
    """Builds output path with *_release_issue_trends.csv suffix."""
    input_path = Path(input_path)
    base_name = _strip_known_suffix(input_path.stem)
    return input_path.with_name(f"{base_name}_release_issue_trends.csv")


def build_top_rising_output_path(input_path: str | Path) -> Path:
    """Builds output path with *_top10_rising_issues_per_version.csv suffix."""
    input_path = Path(input_path)
    base_name = _strip_known_suffix(input_path.stem)
    return input_path.with_name(f"{base_name}_top10_rising_issues_per_version.csv")


def resolve_feedback_path(
    selection: str | int | Path,
    feedback_files: list[Path] | None = None,
    base_dir: str | Path = PROJECT_ROOT,
) -> Path:
    """Resolves feedback CSV selection by index or path."""
    if isinstance(selection, int):
        if not feedback_files:
            raise ValueError("No feedback files available for index selection.")
        index = selection - 1
        if 0 <= index < len(feedback_files):
            return feedback_files[index]
        raise ValueError("Selected file index is out of range.")

    candidate = Path(str(selection).strip().strip('"'))
    if not candidate.is_absolute():
        candidate = Path(base_dir) / candidate

    if candidate.exists() and candidate.is_file() and candidate.suffix.lower() == ".csv":
        return candidate

    raise ValueError("Invalid feedback CSV selection.")


def choose_feedback_file_cli(feedback_files: list[Path]) -> Path:
    """Prompts the user to choose a feedback CSV file or enter a path."""
    if feedback_files:
        print("Available feedback files:")
        for index, path in enumerate(feedback_files, start=1):
            print(f"{index}. {path}")
        print()

    while True:
        selection = input("Select a file number or enter a file path: ").strip().strip('"')
        try:
            resolved_selection: str | int = int(selection) if selection.isdigit() else selection
            return resolve_feedback_path(resolved_selection, feedback_files=feedback_files)
        except ValueError:
            print("Invalid selection. Please try again.")


def _load_issue_detection_module() -> Any:
    """Loads 05_issue_detection.py as module to reuse issue/severity logic."""
    module_path = FUNCTIONS_DIR / "05_issue_detection.py"
    if not module_path.exists():
        raise FileNotFoundError("05_issue_detection.py not found in functions directory.")

    spec = importlib.util.spec_from_file_location("issue_detection_05", module_path)
    if spec is None or spec.loader is None:
        raise ImportError("Could not load 05_issue_detection.py")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ensure_text_column(df: pd.DataFrame, text_column: str | None = None) -> str:
    if text_column and text_column in df.columns:
        return text_column

    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in df.columns:
            return candidate

    raise ValueError("No usable text column found. Provide one of: " + ", ".join(TEXT_COLUMN_CANDIDATES))


def ensure_issue_columns(
    df: pd.DataFrame,
    text_column: str | None = None,
) -> tuple[pd.DataFrame, str]:
    """Ensures issue and severity columns exist, using 05_issue_detection when needed."""
    selected_text_column = _ensure_text_column(df, text_column=text_column)

    has_issue = "issue_primary" in df.columns
    has_severity = "issue_severity_score" in df.columns and "issue_severity_level" in df.columns

    if has_issue and has_severity:
        return df.copy(), selected_text_column

    issue_module = _load_issue_detection_module()
    enriched = issue_module.detect_issues_in_dataframe(df, text_column=selected_text_column)
    enriched = issue_module.add_issue_severity_scores(enriched)
    return enriched, selected_text_column


def compute_release_impact(
    df: pd.DataFrame,
    version_column: str = "appVersion",
    date_column: str = "at",
    score_column: str = "score",
    sentiment_column: str = "sentiment",
    severity_column: str = "issue_severity_score",
) -> pd.DataFrame:
    """Computes release-level quality metrics and deltas versus previous release."""
    release_df = df.copy()

    if version_column not in release_df.columns:
        release_df[version_column] = "unknown"

    release_df[version_column] = release_df[version_column].fillna("unknown").astype(str).str.strip()
    release_df.loc[release_df[version_column] == "", version_column] = "unknown"

    release_df[date_column] = pd.to_datetime(release_df.get(date_column), errors="coerce")

    score_values = pd.to_numeric(release_df.get(score_column), errors="coerce")
    sentiment_values = release_df.get(sentiment_column, pd.Series(index=release_df.index, dtype="object"))
    negative_flag = sentiment_values.fillna("").astype(str).str.lower().eq("negative")

    severity_values = pd.to_numeric(release_df.get(severity_column), errors="coerce")
    if severity_values.notna().sum() == 0:
        severity_values = pd.Series(0.5, index=release_df.index)

    release_df["_negative_flag"] = negative_flag.astype(float)
    release_df["_score_numeric"] = score_values
    release_df["_severity_numeric"] = severity_values.fillna(0.5)

    agg = release_df.groupby(version_column, dropna=False).agg(
        review_count=(version_column, "size"),
        negative_rate=("_negative_flag", "mean"),
        average_score=("_score_numeric", "mean"),
        average_severity=("_severity_numeric", "mean"),
        first_seen=(date_column, "min"),
        last_seen=(date_column, "max"),
    )

    agg["average_score"] = agg["average_score"].fillna(0.0)

    # Sort by first appearance date, then version string for stable ordering.
    agg = agg.reset_index().rename(columns={version_column: "appVersion"})
    agg = agg.sort_values(["first_seen", "appVersion"], na_position="last").reset_index(drop=True)

    score_norm = (agg["average_score"] / 5.0).clip(0.0, 1.0)
    severity_norm = (1.0 - agg["average_severity"].fillna(0.5)).clip(0.0, 1.0)

    agg["quality_index"] = (
        0.5 * (1.0 - agg["negative_rate"]) + 0.3 * score_norm + 0.2 * severity_norm
    ).clip(0.0, 1.0)

    agg["delta_negative_rate"] = agg["negative_rate"] - agg["negative_rate"].shift(1)
    agg["delta_quality_index"] = agg["quality_index"] - agg["quality_index"].shift(1)
    agg["delta_review_count"] = agg["review_count"] - agg["review_count"].shift(1)

    def classify_impact(row: pd.Series) -> str:
        delta_neg = row.get("delta_negative_rate")
        delta_q = row.get("delta_quality_index")
        if pd.isna(delta_neg) or pd.isna(delta_q):
            return "baseline"
        if delta_neg >= 0.03 or delta_q <= -0.03:
            return "regression"
        if delta_neg <= -0.03 or delta_q >= 0.03:
            return "improvement"
        return "stable"

    agg["release_impact"] = agg.apply(classify_impact, axis=1)

    return agg


def compute_issue_release_trends(
    df: pd.DataFrame,
    version_column: str = "appVersion",
    issue_column: str = "issue_primary",
    date_column: str = "at",
    sentiment_column: str = "sentiment",
    min_issue_count_per_version: int = 3,
    min_delta_negative_rate: float = 0.02,
) -> pd.DataFrame:
    """Computes issue trend deltas across app versions to find rising release issues."""
    trend_df = df.copy()

    if version_column not in trend_df.columns:
        trend_df[version_column] = "unknown"
    trend_df[version_column] = trend_df[version_column].fillna("unknown").astype(str).str.strip()
    trend_df.loc[trend_df[version_column] == "", version_column] = "unknown"

    if issue_column not in trend_df.columns:
        trend_df[issue_column] = "unknown"
    trend_df[issue_column] = trend_df[issue_column].fillna("unknown").astype(str).str.strip()

    trend_df[date_column] = pd.to_datetime(trend_df.get(date_column), errors="coerce")

    sentiment_values = trend_df.get(sentiment_column, pd.Series(index=trend_df.index, dtype="object"))
    trend_df["_is_negative"] = sentiment_values.fillna("").astype(str).str.lower().eq("negative").astype(int)

    trend_df = trend_df[~trend_df[issue_column].str.lower().isin({"none", "unknown", ""})].copy()
    if trend_df.empty:
        return pd.DataFrame(
            columns=[
                "appVersion",
                "issue_primary",
                "version_review_count",
                "issue_review_count",
                "issue_negative_count",
                "issue_rate",
                "issue_negative_rate",
                "delta_issue_rate",
                "delta_issue_negative_rate",
                "issue_trend",
            ]
        )

    version_totals = trend_df.groupby(version_column, dropna=False).agg(
        version_review_count=(version_column, "size"),
        version_negative_count=("_is_negative", "sum"),
        first_seen=(date_column, "min"),
    )
    version_totals = version_totals.reset_index()

    version_order = (
        version_totals.sort_values(["first_seen", version_column], na_position="last")
        .reset_index(drop=True)
        .reset_index()
        .rename(columns={"index": "_version_order"})[[version_column, "_version_order"]]
    )

    issue_agg = trend_df.groupby([version_column, issue_column], dropna=False).agg(
        issue_review_count=(issue_column, "size"),
        issue_negative_count=("_is_negative", "sum"),
    )
    issue_agg = issue_agg.reset_index()

    merged = issue_agg.merge(
        version_totals[[version_column, "version_review_count"]],
        on=version_column,
        how="left",
    )
    merged = merged.merge(version_order, on=version_column, how="left")

    merged["issue_rate"] = merged["issue_review_count"] / merged["version_review_count"].clip(lower=1)
    merged["issue_negative_rate"] = (
        merged["issue_negative_count"] / merged["version_review_count"].clip(lower=1)
    )

    merged = merged.sort_values([issue_column, "_version_order", version_column]).reset_index(drop=True)
    merged["delta_issue_rate"] = merged.groupby(issue_column)["issue_rate"].diff()
    merged["delta_issue_negative_rate"] = merged.groupby(issue_column)["issue_negative_rate"].diff()

    def classify_issue_trend(row: pd.Series) -> str:
        delta_value = row.get("delta_issue_negative_rate")
        if pd.isna(delta_value):
            return "baseline"
        if row.get("issue_negative_count", 0) < min_issue_count_per_version:
            return "low_volume"
        if delta_value >= min_delta_negative_rate:
            return "rising"
        if delta_value <= -min_delta_negative_rate:
            return "falling"
        return "stable"

    merged["issue_trend"] = merged.apply(classify_issue_trend, axis=1)

    return merged.rename(columns={version_column: "appVersion", issue_column: "issue_primary"}).drop(
        columns=["_version_order"], errors="ignore"
    )


def build_top_rising_issues_per_version(
    issue_trends_df: pd.DataFrame,
    top_k_per_version: int = 10,
) -> pd.DataFrame:
    """Builds a compact top-k table of rising issues for each app version."""
    if issue_trends_df.empty:
        return pd.DataFrame(
            columns=[
                "appVersion",
                "issue_primary",
                "issue_negative_count",
                "issue_negative_rate",
                "delta_issue_negative_rate",
                "issue_rate",
                "delta_issue_rate",
                "issue_trend",
            ]
        )

    rising = issue_trends_df[issue_trends_df["issue_trend"] == "rising"].copy()
    if rising.empty:
        return pd.DataFrame(
            columns=[
                "appVersion",
                "issue_primary",
                "issue_negative_count",
                "issue_negative_rate",
                "delta_issue_negative_rate",
                "issue_rate",
                "delta_issue_rate",
                "issue_trend",
            ]
        )

    rising = rising.sort_values(
        ["appVersion", "delta_issue_negative_rate", "issue_negative_count"],
        ascending=[True, False, False],
    )
    top = rising.groupby("appVersion", dropna=False).head(top_k_per_version).reset_index(drop=True)

    keep_columns = [
        "appVersion",
        "issue_primary",
        "issue_negative_count",
        "issue_negative_rate",
        "delta_issue_negative_rate",
        "issue_rate",
        "delta_issue_rate",
        "issue_trend",
    ]
    return top[keep_columns]


def run_release_impact_pipeline(
    input_path: str | Path,
    output_path: str | Path | None = None,
    issue_trend_output_path: str | Path | None = None,
    top_rising_output_path: str | Path | None = None,
    text_column: str | None = None,
    min_issue_count_per_version: int = 3,
    min_delta_negative_rate: float = 0.02,
    top_k_per_version: int = 10,
    ensure_issues: bool = True,
    save_output: bool = True,
) -> dict:
    """Runs release impact analysis and optionally saves *_release_impact.csv."""
    input_path = Path(input_path)
    df = load_csv(input_path)

    selected_text_column = _ensure_text_column(df, text_column=text_column)

    if ensure_issues:
        df_enriched, selected_text_column = ensure_issue_columns(df, text_column=selected_text_column)
    else:
        df_enriched = df.copy()

    impact_df = compute_release_impact(df_enriched)
    issue_trends_df = compute_issue_release_trends(
        df_enriched,
        min_issue_count_per_version=min_issue_count_per_version,
        min_delta_negative_rate=min_delta_negative_rate,
    )
    top_rising_df = build_top_rising_issues_per_version(
        issue_trends_df,
        top_k_per_version=top_k_per_version,
    )

    saved_path: Path | None = None
    saved_issue_trend_path: Path | None = None
    saved_top_rising_path: Path | None = None
    if save_output:
        final_output_path = Path(output_path) if output_path else build_release_output_path(input_path)
        saved_path = save_csv(impact_df, final_output_path)
        final_issue_trend_output_path = (
            Path(issue_trend_output_path)
            if issue_trend_output_path
            else build_release_issue_trend_output_path(input_path)
        )
        saved_issue_trend_path = save_csv(issue_trends_df, final_issue_trend_output_path)
        final_top_rising_output_path = (
            Path(top_rising_output_path) if top_rising_output_path else build_top_rising_output_path(input_path)
        )
        saved_top_rising_path = save_csv(top_rising_df, final_top_rising_output_path)

    regressions = impact_df[impact_df["release_impact"] == "regression"].copy()
    improvements = impact_df[impact_df["release_impact"] == "improvement"].copy()
    rising_issue_trends = top_rising_df.copy()

    return {
        "input_path": input_path,
        "output_path": saved_path,
        "issue_trend_output_path": saved_issue_trend_path,
        "top_rising_output_path": saved_top_rising_path,
        "row_count": int(len(df_enriched)),
        "release_count": int(len(impact_df)),
        "text_column": selected_text_column,
        "release_impact": impact_df,
        "issue_trends": issue_trends_df,
        "top_rising_issues_per_version": top_rising_df,
        "rising_issue_trends": rising_issue_trends,
        "regressions": regressions,
        "improvements": improvements,
    }


def main() -> None:
    """CLI entry point for release impact analysis."""
    feedback_files = find_feedback_files()
    input_path = choose_feedback_file_cli(feedback_files)
    result = run_release_impact_pipeline(input_path)

    if result["output_path"]:
        print(f"Release impact file saved to: {result['output_path']}")
    if result["issue_trend_output_path"]:
        print(f"Release issue trend file saved to: {result['issue_trend_output_path']}")
    if result["top_rising_output_path"]:
        print(f"Top rising issues file saved to: {result['top_rising_output_path']}")

    table = result["release_impact"]
    if not table.empty:
        print("Release impact overview:")
        print(table.tail(10).to_string(index=False))

    issue_trends = result["rising_issue_trends"]
    if not issue_trends.empty:
        print("Rising issues after releases:")
        print(
            issue_trends[
                [
                    "appVersion",
                    "issue_primary",
                    "issue_negative_count",
                    "issue_negative_rate",
                    "delta_issue_negative_rate",
                    "issue_trend",
                ]
            ]
            .sort_values(["appVersion", "delta_issue_negative_rate"], ascending=[True, False])
            .head(20)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
