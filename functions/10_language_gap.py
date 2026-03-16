"""
Language Gap Analysis for multilingual app feedback.

Purpose:
- Compare issue prevalence between German and English feedback
- Compare sentiment rates between languages
- Quantify language-specific product quality gaps

Main entry point:
- run_language_gap_pipeline(...)

Typical artifacts:
- *_language_gap.csv
- *_language_sentiment_gap.csv
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
    """Finds *_issues.csv and *_senti.csv files suitable for language gap analysis."""
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


def build_language_gap_output_path(input_path: str | Path) -> Path:
    """Builds output path with *_language_gap.csv suffix."""
    input_path = Path(input_path)
    base_name = _strip_known_suffix(input_path.stem)
    return input_path.with_name(f"{base_name}_language_gap.csv")


def build_language_sentiment_output_path(input_path: str | Path) -> Path:
    """Builds output path with *_language_sentiment_gap.csv suffix."""
    input_path = Path(input_path)
    base_name = _strip_known_suffix(input_path.stem)
    return input_path.with_name(f"{base_name}_language_sentiment_gap.csv")


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
    if has_issue:
        return df.copy(), selected_text_column

    issue_module = _load_issue_detection_module()
    enriched = issue_module.detect_issues_in_dataframe(df, text_column=selected_text_column)
    enriched = issue_module.add_issue_severity_scores(enriched)
    return enriched, selected_text_column


def build_issue_language_gap_table(
    df: pd.DataFrame,
    language_column: str = "lang",
    issue_column: str = "issue_primary",
    focus_languages: tuple[str, str] = ("de", "en"),
) -> pd.DataFrame:
    """Builds per-issue rate differences between two focus languages."""
    gap_df = df.copy()

    if language_column not in gap_df.columns:
        raise ValueError("Language column not found. Expected column: 'lang'.")

    if issue_column not in gap_df.columns:
        gap_df[issue_column] = "unknown"

    gap_df[language_column] = gap_df[language_column].fillna("unknown").astype(str).str.lower()
    gap_df[issue_column] = gap_df[issue_column].fillna("unknown").astype(str)

    lang_totals = gap_df.groupby(language_column).size().rename("lang_total")

    issue_lang = gap_df.groupby([issue_column, language_column]).size().rename("issue_count").reset_index()
    issue_lang = issue_lang.merge(lang_totals.reset_index(), on=language_column, how="left")
    issue_lang["issue_rate"] = issue_lang["issue_count"] / issue_lang["lang_total"].clip(lower=1)

    pivot = issue_lang.pivot(index=issue_column, columns=language_column, values="issue_rate").fillna(0.0)

    lang_a, lang_b = focus_languages
    if lang_a not in pivot.columns:
        pivot[lang_a] = 0.0
    if lang_b not in pivot.columns:
        pivot[lang_b] = 0.0

    result = pivot[[lang_a, lang_b]].reset_index().rename(
        columns={issue_column: "issue_primary", lang_a: f"rate_{lang_a}", lang_b: f"rate_{lang_b}"}
    )

    result["gap_signed"] = result[f"rate_{lang_b}"] - result[f"rate_{lang_a}"]
    result["gap_abs"] = result["gap_signed"].abs()
    result["higher_language"] = result["gap_signed"].apply(
        lambda value: lang_b if value > 0 else (lang_a if value < 0 else "equal")
    )

    return result.sort_values("gap_abs", ascending=False).reset_index(drop=True)


def build_sentiment_language_gap_table(
    df: pd.DataFrame,
    language_column: str = "lang",
    sentiment_column: str = "sentiment",
    focus_languages: tuple[str, str] = ("de", "en"),
) -> pd.DataFrame:
    """Builds sentiment-rate comparison between focus languages."""
    sent_df = df.copy()

    if language_column not in sent_df.columns:
        raise ValueError("Language column not found. Expected column: 'lang'.")
    if sentiment_column not in sent_df.columns:
        raise ValueError("Sentiment column not found. Expected column: 'sentiment'.")

    sent_df[language_column] = sent_df[language_column].fillna("unknown").astype(str).str.lower()
    sent_df[sentiment_column] = sent_df[sentiment_column].fillna("unknown").astype(str).str.lower()

    summary = (
        sent_df.groupby(language_column)[sentiment_column]
        .value_counts(normalize=True)
        .rename("rate")
        .reset_index()
    )

    pivot = summary.pivot(index=sentiment_column, columns=language_column, values="rate").fillna(0.0)

    lang_a, lang_b = focus_languages
    if lang_a not in pivot.columns:
        pivot[lang_a] = 0.0
    if lang_b not in pivot.columns:
        pivot[lang_b] = 0.0

    result = pivot[[lang_a, lang_b]].reset_index().rename(
        columns={
            sentiment_column: "sentiment",
            lang_a: f"rate_{lang_a}",
            lang_b: f"rate_{lang_b}",
        }
    )
    result["gap_signed"] = result[f"rate_{lang_b}"] - result[f"rate_{lang_a}"]
    result["gap_abs"] = result["gap_signed"].abs()

    return result.sort_values("gap_abs", ascending=False).reset_index(drop=True)


def run_language_gap_pipeline(
    input_path: str | Path,
    issue_output_path: str | Path | None = None,
    sentiment_output_path: str | Path | None = None,
    text_column: str | None = None,
    focus_languages: tuple[str, str] = ("de", "en"),
    ensure_issues: bool = True,
    save_output: bool = True,
) -> dict:
    """Runs language gap analysis and optionally saves issue + sentiment gap CSVs."""
    input_path = Path(input_path)
    df = load_csv(input_path)

    selected_text_column = _ensure_text_column(df, text_column=text_column)

    if ensure_issues:
        df_enriched, selected_text_column = ensure_issue_columns(df, text_column=selected_text_column)
    else:
        df_enriched = df.copy()

    issue_gap_df = build_issue_language_gap_table(
        df_enriched,
        language_column="lang",
        issue_column="issue_primary",
        focus_languages=focus_languages,
    )
    sentiment_gap_df = build_sentiment_language_gap_table(
        df_enriched,
        language_column="lang",
        sentiment_column="sentiment",
        focus_languages=focus_languages,
    )

    saved_issue_path: Path | None = None
    saved_sentiment_path: Path | None = None

    if save_output:
        final_issue_path = (
            Path(issue_output_path) if issue_output_path else build_language_gap_output_path(input_path)
        )
        final_sentiment_path = (
            Path(sentiment_output_path)
            if sentiment_output_path
            else build_language_sentiment_output_path(input_path)
        )

        saved_issue_path = save_csv(issue_gap_df, final_issue_path)
        saved_sentiment_path = save_csv(sentiment_gap_df, final_sentiment_path)

    return {
        "input_path": input_path,
        "issue_output_path": saved_issue_path,
        "sentiment_output_path": saved_sentiment_path,
        "row_count": int(len(df_enriched)),
        "text_column": selected_text_column,
        "issue_gap": issue_gap_df,
        "sentiment_gap": sentiment_gap_df,
        "focus_languages": focus_languages,
    }


def main() -> None:
    """CLI entry point for language gap analysis."""
    feedback_files = find_feedback_files()
    input_path = choose_feedback_file_cli(feedback_files)
    result = run_language_gap_pipeline(input_path)

    if result["issue_output_path"]:
        print(f"Issue language gap file saved to: {result['issue_output_path']}")
    if result["sentiment_output_path"]:
        print(f"Sentiment language gap file saved to: {result['sentiment_output_path']}")

    issue_gap = result["issue_gap"]
    sentiment_gap = result["sentiment_gap"]

    if issue_gap.empty:
        print("No issue language gaps available.")
    else:
        print("Top issue gaps:")
        print(issue_gap.head(20).to_string(index=False))

    if sentiment_gap.empty:
        print("No sentiment language gaps available.")
    else:
        print("Sentiment gaps:")
        print(sentiment_gap.to_string(index=False))


if __name__ == "__main__":
    main()
