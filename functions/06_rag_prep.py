"""
RAG Preparation Pipeline for review feedback.

Purpose:
- Convert feedback rows into retrieval-ready documents
- Attach structured metadata (sentiment, issue labels, version, language, etc.)
- Build and persist a lightweight TF-IDF retriever bundle

Main entry point:
- run_rag_preparation_pipeline(...)

Typical artifacts:
- *_rag_docs.jsonl
- *_rag_retriever.pkl
"""

from __future__ import annotations

import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


FUNCTIONS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FUNCTIONS_DIR.parent
DEFAULT_SEARCH_DIR = PROJECT_ROOT / "data"
TEXT_COLUMN_CANDIDATES = ("content_clean", "clean_text", "content")
DEFAULT_METADATA_COLUMNS = (
    "reviewId",
    "appId",
    "appTitle",
    "lang",
    "country",
    "score",
    "sentiment",
    "issue_primary",
    "issue_labels",
    "issue_severity_level",
    "issue_severity_score",
    "appVersion",
    "at",
)


def load_csv(path: str | Path) -> pd.DataFrame:
    """Loads a CSV file with UTF-8 BOM support."""
    return pd.read_csv(path, encoding="utf-8-sig")


def save_jsonl(records: list[dict], path: str | Path) -> Path:
    """Saves a list of dictionaries as JSON Lines."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    return path


def save_pickle(obj: Any, path: str | Path) -> Path:
    """Serializes an object to a pickle file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(obj, file)
    return path


def load_pickle(path: str | Path) -> Any:
    """Loads a pickle object from disk."""
    with Path(path).open("rb") as file:
        return pickle.load(file)


def find_feedback_csv_files(search_dir: str | Path = DEFAULT_SEARCH_DIR) -> list[Path]:
    """Finds candidate feedback files for RAG preparation."""
    search_dir = Path(search_dir)
    if not search_dir.exists():
        return []

    files = []
    for pattern in ("*_issues.csv", "*_senti.csv", "*_clean.csv"):
        files.extend(path for path in search_dir.rglob(pattern) if path.is_file())

    return sorted(set(files))


def get_feedback_csv_options(search_dir: str | Path = DEFAULT_SEARCH_DIR) -> list[dict]:
    """Returns candidate feedback files in a UI-friendly structure."""
    options = []
    for path in find_feedback_csv_files(search_dir):
        try:
            label = str(path.relative_to(PROJECT_ROOT))
        except ValueError:
            label = str(path)

        stage = "unknown"
        if path.stem.endswith("_issues"):
            stage = "issues"
        elif path.stem.endswith("_senti"):
            stage = "sentiment"
        elif path.stem.endswith("_clean"):
            stage = "clean"

        options.append(
            {
                "label": label,
                "path": path,
                "path_str": str(path),
                "name": path.name,
                "stem": path.stem,
                "stage": stage,
            }
        )

    return options


def _strip_known_suffix(stem: str) -> str:
    for suffix in ("_issues", "_senti", "_clean"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def build_rag_output_paths(
    input_path: str | Path,
    artifact_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Builds default output paths for RAG docs and retriever artifacts."""
    input_path = Path(input_path)
    output_dir = Path(artifact_dir) if artifact_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = _strip_known_suffix(input_path.stem)
    return {
        "documents_jsonl": output_dir / f"{base_name}_rag_docs.jsonl",
        "retriever_pkl": output_dir / f"{base_name}_rag_retriever.pkl",
    }


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


def choose_feedback_csv_cli(feedback_files: list[Path]) -> Path:
    """Prompts the user to choose a feedback CSV file or enter a path."""
    if feedback_files:
        print("Available feedback CSV files:")
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


def resolve_text_column(
    df: pd.DataFrame,
    text_column: str | None = None,
    candidates: tuple[str, ...] = TEXT_COLUMN_CANDIDATES,
) -> str:
    """Returns the text column used for RAG document creation."""
    if text_column and text_column in df.columns:
        return text_column

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    raise ValueError("No usable text column found. Provide one of: " + ", ".join(candidates))


def _build_doc_id(row: pd.Series, index: int, id_column: str = "reviewId") -> str:
    value = row.get(id_column)
    if pd.notna(value) and str(value).strip():
        return f"{str(value).strip()}_{index}"
    return f"row_{index}"


def _build_retrieval_text(row: pd.Series, text_column: str) -> str:
    parts = [str(row.get(text_column, "") or "").strip()]

    issue_primary = str(row.get("issue_primary", "") or "").strip()
    sentiment = str(row.get("sentiment", "") or "").strip()

    if issue_primary and issue_primary.lower() != "unknown":
        parts.append(f"issue {issue_primary}")
    if sentiment:
        parts.append(f"sentiment {sentiment}")

    return " | ".join(part for part in parts if part)


def build_rag_documents(
    df: pd.DataFrame,
    text_column: str,
    metadata_columns: tuple[str, ...] = DEFAULT_METADATA_COLUMNS,
    source_file: str | Path | None = None,
) -> pd.DataFrame:
    """Builds RAG-ready review documents with metadata."""
    source = str(source_file) if source_file else ""
    working_df = df.copy()
    working_df[text_column] = working_df[text_column].fillna("").astype(str).str.strip()
    working_df = working_df[working_df[text_column] != ""].copy()

    records = []
    for index, (_, row) in enumerate(working_df.iterrows()):
        metadata = {}
        for column in metadata_columns:
            if column in row.index:
                value = row[column]
                if pd.notna(value):
                    metadata[column] = value if not isinstance(value, pd.Timestamp) else value.isoformat()

        record = {
            "doc_id": _build_doc_id(row, index=index),
            "text": str(row[text_column]),
            "retrieval_text": _build_retrieval_text(row, text_column=text_column),
            "source_file": source,
            "metadata_json": json.dumps(metadata, ensure_ascii=False),
        }

        for key, value in metadata.items():
            record[key] = value

        records.append(record)

    return pd.DataFrame(records)


def docs_dataframe_to_jsonl_records(docs_df: pd.DataFrame) -> list[dict]:
    """Converts docs DataFrame rows into JSONL document records."""
    records: list[dict] = []
    for _, row in docs_df.iterrows():
        metadata = {}
        metadata_json = row.get("metadata_json")
        if isinstance(metadata_json, str) and metadata_json:
            try:
                metadata = json.loads(metadata_json)
            except json.JSONDecodeError:
                metadata = {}

        records.append(
            {
                "doc_id": row.get("doc_id"),
                "text": row.get("text", ""),
                "retrieval_text": row.get("retrieval_text", ""),
                "source_file": row.get("source_file", ""),
                "metadata": metadata,
            }
        )

    return records


def train_tfidf_retriever(
    docs_df: pd.DataFrame,
    text_column: str = "retrieval_text",
    max_features: int = 20000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 1,
) -> dict:
    """Trains a TF-IDF retriever bundle from RAG documents."""
    if docs_df.empty:
        raise ValueError("Cannot train retriever on empty documents DataFrame.")

    text_series = docs_df[text_column].fillna("").astype(str)
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
    )
    matrix = vectorizer.fit_transform(text_series)

    metadata_columns = [
        column
        for column in docs_df.columns
        if column not in {"retrieval_text", "metadata_json"}
    ]

    docs_metadata = docs_df[metadata_columns].to_dict(orient="records")

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "vectorizer": vectorizer,
        "matrix": matrix,
        "text_column": text_column,
        "docs": docs_metadata,
    }


def query_tfidf_retriever(
    query: str,
    retriever_bundle: dict,
    top_k: int = 5,
    min_score: float = 0.0,
) -> list[dict]:
    """Retrieves the most similar documents for a query string."""
    if query is None or not str(query).strip():
        return []

    vectorizer = retriever_bundle["vectorizer"]
    matrix = retriever_bundle["matrix"]
    docs = retriever_bundle.get("docs", [])

    query_vector = vectorizer.transform([str(query).strip()])
    similarity = cosine_similarity(query_vector, matrix).flatten()

    if similarity.size == 0:
        return []

    top_indices = similarity.argsort()[::-1][: max(1, top_k)]

    results = []
    for index in top_indices:
        score = float(similarity[index])
        if score < min_score:
            continue

        doc = dict(docs[index]) if index < len(docs) else {"doc_id": f"doc_{index}"}
        doc["score"] = score
        results.append(doc)

    return results


def run_rag_preparation_pipeline(
    input_path: str | Path,
    output_docs_path: str | Path | None = None,
    output_retriever_path: str | Path | None = None,
    text_column: str | None = None,
    artifact_dir: str | Path | None = None,
    save_artifacts: bool = True,
) -> dict:
    """Builds RAG documents and a retriever bundle from a feedback CSV file."""
    input_path = Path(input_path)
    df = load_csv(input_path)
    selected_text_column = resolve_text_column(df, text_column=text_column)

    docs_df = build_rag_documents(
        df,
        text_column=selected_text_column,
        source_file=input_path,
    )

    retriever_bundle = train_tfidf_retriever(docs_df)

    paths = build_rag_output_paths(input_path, artifact_dir=artifact_dir)
    final_docs_path = Path(output_docs_path) if output_docs_path else paths["documents_jsonl"]
    final_retriever_path = Path(output_retriever_path) if output_retriever_path else paths["retriever_pkl"]

    saved_docs_path: Path | None = None
    saved_retriever_path: Path | None = None

    if save_artifacts:
        jsonl_records = docs_dataframe_to_jsonl_records(docs_df)
        saved_docs_path = save_jsonl(jsonl_records, final_docs_path)
        saved_retriever_path = save_pickle(retriever_bundle, final_retriever_path)

    return {
        "input_path": input_path,
        "text_column": selected_text_column,
        "row_count": int(len(df)),
        "document_count": int(len(docs_df)),
        "documents_path": saved_docs_path,
        "retriever_path": saved_retriever_path,
        "documents": docs_df,
        "retriever_bundle": retriever_bundle,
    }


def main() -> None:
    """CLI entry point for RAG document preparation."""
    feedback_files = find_feedback_csv_files()
    input_path = choose_feedback_csv_cli(feedback_files)
    result = run_rag_preparation_pipeline(input_path)

    print(f"RAG documents built: {result['document_count']}")
    if result["documents_path"]:
        print(f"Documents JSONL: {result['documents_path']}")
    if result["retriever_path"]:
        print(f"Retriever bundle: {result['retriever_path']}")


if __name__ == "__main__":
    main()
