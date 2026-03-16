"""
Early-Warning Alerts for app feedback.

Purpose:
- Detect unusual spikes in negative issue feedback over time
- Build alert rows per period and issue category
- Support dashboarding and automated monitoring workflows

Main entry point:
- run_early_alert_pipeline(...)

Typical artifact:
- *_alerts.csv
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
    """Saves DataFrame as CSV with UTF-8 BOM encoding."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def find_feedback_files(search_dir: str | Path = DEFAULT_SEARCH_DIR) -> list[Path]:
    """Finds *_issues.csv and *_senti.csv files suitable for alerting."""
    search_dir = Path(search_dir)
    if not search_dir.exists():
        return []

    files = []
    for pattern in ("*_issues.csv", "*_senti.csv"):
        files.extend(path for path in search_dir.rglob(pattern) if path.is_file())

    return sorted(set(files))


def _strip_known_suffix(stem: str) -> str:
    for suffix in ("_issues", "_senti", "_clean"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def build_alert_output_path(input_path: str | Path) -> Path:
    """Builds output path with *_alerts.csv suffix."""
    input_path = Path(input_path)
    base_name = _strip_known_suffix(input_path.stem)
    return input_path.with_name(f"{base_name}_alerts.csv")


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
    """CLI prompt to select a feedback CSV file."""
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
    """Loads 05_issue_detection.py dynamically to enrich issue columns when needed."""
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
    has_severity = "issue_severity_score" in df.columns

    if has_issue and has_severity:
        return df.copy(), selected_text_column

    issue_module = _load_issue_detection_module()
    enriched = issue_module.detect_issues_in_dataframe(df, text_column=selected_text_column)
    enriched = issue_module.add_issue_severity_scores(enriched)
    return enriched, selected_text_column


def build_issue_period_timeseries(
    df: pd.DataFrame,
    date_column: str = "at",
    issue_column: str = "issue_primary",
    sentiment_column: str = "sentiment",
    period: str = "W",
) -> pd.DataFrame:
    """Aggregates total/negative counts and rates by period and issue."""
    ts_df = df.copy()
    ts_df[date_column] = pd.to_datetime(ts_df.get(date_column), errors="coerce")
    ts_df = ts_df[ts_df[date_column].notna()].copy()

    if issue_column not in ts_df.columns:
        ts_df[issue_column] = "unknown"

    ts_df[issue_column] = ts_df[issue_column].fillna("unknown").astype(str)
    ts_df["period"] = ts_df[date_column].dt.to_period(period).dt.to_timestamp()

    negative_mask = ts_df.get(sentiment_column, pd.Series(index=ts_df.index, dtype="object"))
    ts_df["is_negative"] = negative_mask.fillna("").astype(str).str.lower().eq("negative").astype(int)

    grouped = ts_df.groupby(["period", issue_column], dropna=False).agg(
        total_count=(issue_column, "size"),
        negative_count=("is_negative", "sum"),
    )
    grouped = grouped.reset_index().rename(columns={issue_column: "issue_primary"})
    grouped["negative_rate"] = grouped["negative_count"] / grouped["total_count"].clip(lower=1)

    return grouped.sort_values(["issue_primary", "period"]).reset_index(drop=True)


def detect_spike_alerts(
    timeseries_df: pd.DataFrame,
    baseline_periods: int = 4,
    z_threshold: float = 2.0,
    min_negative_count: int = 5,
    min_rate_jump: float = 0.10,
) -> pd.DataFrame:
    """Flags periods where negative signal spikes relative to rolling baseline."""
    if timeseries_df.empty:
        return timeseries_df.copy()

    df = timeseries_df.copy()

    def add_baseline(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values("period").copy()

        group["baseline_neg_mean"] = (
            group["negative_count"].shift(1).rolling(baseline_periods, min_periods=2).mean()
        )
        group["baseline_neg_std"] = (
            group["negative_count"].shift(1).rolling(baseline_periods, min_periods=2).std()
        )
        group["baseline_rate_mean"] = (
            group["negative_rate"].shift(1).rolling(baseline_periods, min_periods=2).mean()
        )

        group["baseline_neg_mean"] = group["baseline_neg_mean"].fillna(0.0)
        group["baseline_neg_std"] = group["baseline_neg_std"].fillna(0.0)
        group["baseline_rate_mean"] = group["baseline_rate_mean"].fillna(0.0)

        std_safe = group["baseline_neg_std"].replace(0.0, 1.0)
        group["z_score_negative_count"] = (
            group["negative_count"] - group["baseline_neg_mean"]
        ) / std_safe

        group["rate_jump"] = group["negative_rate"] - group["baseline_rate_mean"]

        count_spike = (
            (group["negative_count"] >= min_negative_count)
            & (group["z_score_negative_count"] >= z_threshold)
        )
        rate_spike = (
            (group["negative_count"] >= min_negative_count)
            & (group["rate_jump"] >= min_rate_jump)
        )

        group["is_alert"] = (count_spike | rate_spike).astype(int)

        def label_alert(row: pd.Series) -> str:
            if row["is_alert"] == 0:
                return "none"
            if row["z_score_negative_count"] >= (z_threshold + 1.0):
                return "high"
            if row["z_score_negative_count"] >= z_threshold:
                return "medium"
            return "low"

        group["alert_level"] = group.apply(label_alert, axis=1)
        return group

    df = df.groupby("issue_primary", group_keys=False).apply(add_baseline)

    alerts = df[df["is_alert"] == 1].copy()
    alerts = alerts.sort_values(["period", "z_score_negative_count"], ascending=[True, False])
    return alerts.reset_index(drop=True)


def run_early_alert_pipeline(
    input_path: str | Path,
    output_path: str | Path | None = None,
    text_column: str | None = None,
    period: str = "W",
    baseline_periods: int = 4,
    z_threshold: float = 2.0,
    min_negative_count: int = 5,
    min_rate_jump: float = 0.10,
    ensure_issues: bool = True,
    save_output: bool = True,
) -> dict:
    """Runs early-warning alert detection and optionally saves *_alerts.csv."""
    input_path = Path(input_path)
    df = load_csv(input_path)

    selected_text_column = _ensure_text_column(df, text_column=text_column)

    if ensure_issues:
        df_enriched, selected_text_column = ensure_issue_columns(df, text_column=selected_text_column)
    else:
        df_enriched = df.copy()

    timeseries_df = build_issue_period_timeseries(
        df_enriched,
        date_column="at",
        issue_column="issue_primary",
        sentiment_column="sentiment",
        period=period,
    )

    alerts_df = detect_spike_alerts(
        timeseries_df,
        baseline_periods=baseline_periods,
        z_threshold=z_threshold,
        min_negative_count=min_negative_count,
        min_rate_jump=min_rate_jump,
    )

    saved_path: Path | None = None
    if save_output:
        final_output_path = Path(output_path) if output_path else build_alert_output_path(input_path)
        saved_path = save_csv(alerts_df, final_output_path)

    return {
        "input_path": input_path,
        "output_path": saved_path,
        "row_count": int(len(df_enriched)),
        "timeseries_rows": int(len(timeseries_df)),
        "alert_count": int(len(alerts_df)),
        "text_column": selected_text_column,
        "timeseries": timeseries_df,
        "alerts": alerts_df,
    }


def main() -> None:
    """CLI entry point for early-warning alert detection."""
    feedback_files = find_feedback_files()
    input_path = choose_feedback_file_cli(feedback_files)
    result = run_early_alert_pipeline(input_path)

    if result["output_path"]:
        print(f"Alert file saved to: {result['output_path']}")

    alerts = result["alerts"]
    if alerts.empty:
        print("No alerts detected.")
    else:
        print("Detected alerts:")
        print(alerts.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
