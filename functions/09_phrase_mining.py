"""
Root-Cause Phrase Mining for issue-focused feedback analysis.

Purpose:
- Extract top n-gram phrases from issue-specific negative reviews
- Surface likely root causes in language users actually use
- Provide compact phrase tables for dashboards and reporting

Main entry point:
- run_phrase_mining_pipeline(...)

Typical artifact:
- *_phrases.csv
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


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
    """Finds *_issues.csv and *_senti.csv files suitable for phrase mining."""
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


def build_phrase_output_path(input_path: str | Path) -> Path:
    """Builds output path with *_phrases.csv suffix."""
    input_path = Path(input_path)
    base_name = _strip_known_suffix(input_path.stem)
    return input_path.with_name(f"{base_name}_phrases.csv")


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


def _normalize_text_series(text_series: pd.Series) -> pd.Series:
    normalized = text_series.fillna("").astype(str).str.strip()
    normalized = normalized[normalized.str.len() > 0]
    return normalized


def extract_top_phrases(
    texts: pd.Series,
    top_k: int = 20,
    min_df: int = 2,
    ngram_range: tuple[int, int] = (2, 3),
    stop_words: list[str] | None = None,
) -> pd.DataFrame:
    """Extracts top n-gram phrases from a text series by frequency."""
    clean_texts = _normalize_text_series(texts)
    if clean_texts.empty:
        return pd.DataFrame(columns=["phrase", "count", "share"])

    vectorizer = CountVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        stop_words=stop_words,
        token_pattern=r"(?u)\b\w+\b",
    )

    matrix = vectorizer.fit_transform(clean_texts)
    if matrix.shape[1] == 0:
        return pd.DataFrame(columns=["phrase", "count", "share"])

    counts = matrix.sum(axis=0).A1
    phrases = vectorizer.get_feature_names_out()

    phrase_df = pd.DataFrame({"phrase": phrases, "count": counts})
    phrase_df = phrase_df.sort_values("count", ascending=False).head(top_k).reset_index(drop=True)

    total_count = phrase_df["count"].sum()
    phrase_df["share"] = phrase_df["count"] / total_count if total_count > 0 else 0.0
    return phrase_df


def build_issue_phrase_table(
    df: pd.DataFrame,
    text_column: str = "content_clean",
    issue_column: str = "issue_primary",
    sentiment_column: str = "sentiment",
    top_k_per_issue: int = 20,
    min_reviews_per_issue: int = 10,
    min_df: int = 2,
    ngram_range: tuple[int, int] = (2, 3),
) -> pd.DataFrame:
    """Builds top phrase table per issue category from negative feedback."""
    work_df = df.copy()

    if issue_column not in work_df.columns:
        work_df[issue_column] = "unknown"

    work_df[issue_column] = work_df[issue_column].fillna("unknown").astype(str)

    if sentiment_column in work_df.columns:
        negative_mask = work_df[sentiment_column].fillna("").astype(str).str.lower().eq("negative")
        work_df = work_df[negative_mask].copy()

    issue_tables: list[pd.DataFrame] = []

    for issue, group in work_df.groupby(issue_column, dropna=False):
        if str(issue).lower() in {"none", "unknown"}:
            continue
        if len(group) < min_reviews_per_issue:
            continue

        phrase_df = extract_top_phrases(
            group[text_column],
            top_k=top_k_per_issue,
            min_df=min_df,
            ngram_range=ngram_range,
        )
        if phrase_df.empty:
            continue

        phrase_df.insert(0, "issue_primary", issue)
        phrase_df.insert(1, "review_count", len(group))
        issue_tables.append(phrase_df)

    if not issue_tables:
        return pd.DataFrame(columns=["issue_primary", "review_count", "phrase", "count", "share"])

    result = pd.concat(issue_tables, ignore_index=True)
    return result.sort_values(["issue_primary", "count"], ascending=[True, False]).reset_index(drop=True)


def run_phrase_mining_pipeline(
    input_path: str | Path,
    output_path: str | Path | None = None,
    text_column: str | None = None,
    top_k_per_issue: int = 20,
    min_reviews_per_issue: int = 10,
    min_df: int = 2,
    ngram_range: tuple[int, int] = (2, 3),
    ensure_issues: bool = True,
    save_output: bool = True,
) -> dict:
    """Runs phrase mining and optionally saves *_phrases.csv."""
    input_path = Path(input_path)
    df = load_csv(input_path)

    selected_text_column = _ensure_text_column(df, text_column=text_column)

    if ensure_issues:
        df_enriched, selected_text_column = ensure_issue_columns(df, text_column=selected_text_column)
    else:
        df_enriched = df.copy()

    phrase_df = build_issue_phrase_table(
        df_enriched,
        text_column=selected_text_column,
        issue_column="issue_primary",
        sentiment_column="sentiment",
        top_k_per_issue=top_k_per_issue,
        min_reviews_per_issue=min_reviews_per_issue,
        min_df=min_df,
        ngram_range=ngram_range,
    )

    saved_path: Path | None = None
    if save_output:
        final_output_path = Path(output_path) if output_path else build_phrase_output_path(input_path)
        saved_path = save_csv(phrase_df, final_output_path)

    return {
        "input_path": input_path,
        "output_path": saved_path,
        "row_count": int(len(df_enriched)),
        "phrase_count": int(len(phrase_df)),
        "text_column": selected_text_column,
        "phrases": phrase_df,
    }


def main() -> None:
    """CLI entry point for root-cause phrase mining."""
    feedback_files = find_feedback_files()
    input_path = choose_feedback_file_cli(feedback_files)
    result = run_phrase_mining_pipeline(input_path)

    if result["output_path"]:
        print(f"Phrase file saved to: {result['output_path']}")

    phrase_df = result["phrases"]
    if phrase_df.empty:
        print("No phrases extracted.")
    else:
        print("Top phrases by issue:")
        print(phrase_df.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
