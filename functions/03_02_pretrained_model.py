from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score


FUNCTIONS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FUNCTIONS_DIR.parent
TEXT_COLUMN_CANDIDATES = ("lemmatized_text", "content_clean", "clean_text", "content")
DEFAULT_LABEL_MODE = "three_class"
DEFAULT_MODEL_KEYS = (
    "xlm_roberta_twitter_sentiment",
    "mbert_star_sentiment",
    "german_sentiment_bert",
    "english_sst2_distilbert",
)

PRETRAINED_MODEL_SPECS: dict[str, dict[str, Any]] = {
    "xlm_roberta_twitter_sentiment": {
        "model_id": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
        "task": "text-classification",
        "label_space": "three_class",
        "supported_langs": {"de", "en"},
    },
    "mbert_star_sentiment": {
        "model_id": "nlptown/bert-base-multilingual-uncased-sentiment",
        "task": "text-classification",
        "label_space": "five_star",
        "supported_langs": {"de", "en"},
    },
    "german_sentiment_bert": {
        "model_id": "oliverguhr/german-sentiment-bert",
        "task": "text-classification",
        "label_space": "three_class",
        "supported_langs": {"de"},
    },
    "english_sst2_distilbert": {
        "model_id": "distilbert-base-uncased-finetuned-sst-2-english",
        "task": "text-classification",
        "label_space": "binary",
        "supported_langs": {"en"},
    },
}


def _to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_serializable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def load_csv(path: str | Path) -> pd.DataFrame:
    """Loads a CSV file with UTF-8 BOM support."""
    return pd.read_csv(path, encoding="utf-8-sig")


def save_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """Saves a DataFrame as CSV with UTF-8 BOM encoding."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def save_json(data: Any, path: str | Path) -> Path:
    """Saves JSON with UTF-8 encoding."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(_to_serializable(data), file, indent=2, ensure_ascii=False)
    return path


def find_clean_csv_files(search_dir: str | Path = PROJECT_ROOT / "data") -> list[Path]:
    """Finds all *_clean.csv files below the provided directory."""
    search_dir = Path(search_dir)
    if not search_dir.exists():
        return []
    return sorted(path for path in search_dir.rglob("*_clean.csv") if path.is_file())


def get_clean_csv_options(search_dir: str | Path = PROJECT_ROOT / "data") -> list[dict]:
    """Returns clean CSV file options in a UI-friendly format."""
    options = []
    for path in find_clean_csv_files(search_dir):
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


def resolve_clean_csv_path(
    selection: str | int | Path,
    clean_files: list[Path] | None = None,
    base_dir: str | Path = PROJECT_ROOT,
) -> Path:
    """Resolves a selected clean CSV either from an index or file path."""
    if isinstance(selection, int):
        if not clean_files:
            raise ValueError("No clean CSV files are available for index-based selection.")
        index = selection - 1
        if 0 <= index < len(clean_files):
            return clean_files[index]
        raise ValueError("Selected file index is out of range.")

    candidate = Path(str(selection).strip().strip('"'))
    if not candidate.is_absolute():
        candidate = Path(base_dir) / candidate

    if candidate.exists() and candidate.is_file() and candidate.suffix.lower() == ".csv":
        return candidate

    raise ValueError("Invalid clean CSV selection.")


def choose_clean_csv_cli(clean_files: list[Path]) -> Path:
    """Prompts the user to choose a clean CSV file or enter a file path."""
    if clean_files:
        print("Available clean CSV files:")
        for index, path in enumerate(clean_files, start=1):
            print(f"{index}. {path}")
        print()

    while True:
        selection = input("Select a file number or enter a file path: ").strip().strip('"')
        try:
            resolved_selection: str | int = int(selection) if selection.isdigit() else selection
            return resolve_clean_csv_path(resolved_selection, clean_files=clean_files)
        except ValueError:
            print("Invalid selection. Please try again.")


def resolve_text_column(
    df: pd.DataFrame,
    text_column: str | None = None,
    candidates: tuple[str, ...] = TEXT_COLUMN_CANDIDATES,
) -> str:
    """Returns the text column to use for sentiment modeling."""
    if text_column and text_column in df.columns:
        return text_column

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    raise ValueError("No usable text column found. Provide one of: " + ", ".join(candidates))


def _strip_clean_suffix(stem: str) -> str:
    return stem[:-6] if stem.endswith("_clean") else stem


def build_pretrained_summary_output_path(clean_path: str | Path) -> Path:
    """Builds output path for benchmark summary CSV."""
    clean_path = Path(clean_path)
    base_name = _strip_clean_suffix(clean_path.stem)
    return clean_path.with_name(f"{base_name}_pretrained_benchmark.csv")


def build_pretrained_report_output_path(clean_path: str | Path) -> Path:
    """Builds output path for benchmark JSON report."""
    clean_path = Path(clean_path)
    base_name = _strip_clean_suffix(clean_path.stem)
    return clean_path.with_name(f"{base_name}_pretrained_model_results.json")


def build_pretrained_senti_output_path(clean_path: str | Path) -> Path:
    """Builds output path for best-model sentiment predictions CSV."""
    clean_path = Path(clean_path)
    base_name = _strip_clean_suffix(clean_path.stem)
    return clean_path.with_name(f"{base_name}_pretrained_senti.csv")


def score_to_three_class_label(score: float | int | None) -> str | None:
    """Maps star score to three-class sentiment label."""
    if pd.isna(score):
        return None
    try:
        value = float(score)
    except (TypeError, ValueError):
        return None

    if value >= 4:
        return "positive"
    if value <= 2:
        return "negative"
    return "neutral"


def score_to_binary_label(score: float | int | None) -> str | None:
    """Maps star score to binary sentiment label."""
    if pd.isna(score):
        return None
    try:
        value = float(score)
    except (TypeError, ValueError):
        return None

    if value >= 4:
        return "positive"
    if value <= 2:
        return "negative"
    return None


def get_target_labels_for_mode(label_mode: str) -> list[str]:
    """Returns ordered target labels for a configured label mode."""
    if label_mode == "binary":
        return ["negative", "positive"]
    if label_mode == "three_class":
        return ["negative", "neutral", "positive"]
    raise ValueError("Invalid label_mode. Use 'binary' or 'three_class'.")


def build_labeled_feedback_data(
    df: pd.DataFrame,
    text_column: str,
    score_column: str = "score",
    label_mode: str = DEFAULT_LABEL_MODE,
) -> pd.DataFrame:
    """Builds labeled evaluation data from review scores."""
    if score_column not in df.columns:
        raise ValueError(f"Missing required score column: {score_column}")

    base_columns = [text_column, score_column]
    if "lang" in df.columns:
        base_columns.append("lang")

    work_df = df[base_columns].copy()
    work_df[text_column] = work_df[text_column].fillna("").astype(str).str.strip()
    work_df = work_df[work_df[text_column] != ""].copy()

    if label_mode == "binary":
        work_df["target_label"] = work_df[score_column].apply(score_to_binary_label)
    elif label_mode == "three_class":
        work_df["target_label"] = work_df[score_column].apply(score_to_three_class_label)
    else:
        raise ValueError("Invalid label_mode. Use 'binary' or 'three_class'.")

    work_df = work_df.dropna(subset=["target_label"]).copy()
    work_df["target_label"] = work_df["target_label"].astype(str).str.lower()

    if "lang" in work_df.columns:
        work_df["lang"] = work_df["lang"].fillna("unknown").astype(str).str.lower().str.strip()

    return work_df


def maybe_sample_eval_data(
    eval_df: pd.DataFrame,
    max_eval_rows: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Optionally downsamples evaluation data for faster benchmarking."""
    if max_eval_rows is None or max_eval_rows <= 0 or len(eval_df) <= max_eval_rows:
        return eval_df
    return eval_df.sample(n=max_eval_rows, random_state=random_state).reset_index(drop=True)


def _require_transformers_pipeline() -> Any:
    """Loads transformers.pipeline lazily and raises a clear dependency error."""
    try:
        from transformers import pipeline as hf_pipeline
    except ImportError as exc:
        raise ImportError(
            "Missing dependency 'transformers'. Install: pip install transformers torch sentencepiece"
        ) from exc

    return hf_pipeline


def load_model_pipeline(
    model_id: str,
    task: str = "text-classification",
    device: int = -1,
) -> Any:
    """Creates a Hugging Face inference pipeline for a model id."""
    hf_pipeline = _require_transformers_pipeline()
    try:
        return hf_pipeline(
            task=task,
            model=model_id,
            tokenizer=model_id,
            device=device,
            model_kwargs={"use_safetensors": True},
        )
    except (OSError, RuntimeError):
        # Fallback: try without use_safetensors (some older models have no safetensors file)
        return hf_pipeline(task=task, model=model_id, tokenizer=model_id, device=device)


def _extract_best_output(item: Any) -> dict:
    if isinstance(item, dict):
        return item

    if isinstance(item, list):
        best: dict | None = None
        best_score = float("-inf")
        for entry in item:
            if not isinstance(entry, dict):
                continue
            score = float(entry.get("score", 0.0))
            if score > best_score:
                best = entry
                best_score = score
        return best or {}

    return {}


def _map_label_from_five_star(raw_label: str) -> str | None:
    match = re.search(r"([1-5])", raw_label)
    if not match:
        return None

    stars = int(match.group(1))
    if stars <= 2:
        return "negative"
    if stars == 3:
        return "neutral"
    return "positive"


def map_model_label_to_sentiment(raw_label: Any, label_space: str) -> str | None:
    """Maps model-specific labels to canonical sentiment labels."""
    label = str(raw_label).strip().lower()
    label = label.replace("__label__", "").strip()

    if label_space == "five_star":
        return _map_label_from_five_star(label)

    if label_space == "binary":
        if label in {"label_0", "0", "negative", "neg", "1 star", "2 stars"}:
            return "negative"
        if label in {"label_1", "1", "positive", "pos", "4 stars", "5 stars"}:
            return "positive"
        if "neg" in label:
            return "negative"
        if "pos" in label:
            return "positive"
        return None

    if label_space == "three_class":
        if label in {"label_0", "0", "negative", "neg", "negativ"}:
            return "negative"
        if label in {"label_1", "1", "neutral", "neu", "3 stars", "3 star"}:
            return "neutral"
        if label in {"label_2", "2", "positive", "pos", "positiv"}:
            return "positive"

        if "neutral" in label or "neu" in label:
            return "neutral"
        if "neg" in label:
            return "negative"
        if "pos" in label:
            return "positive"

    return None


def predict_with_pretrained_model(
    text_values: pd.Series,
    model_key: str,
    batch_size: int = 16,
    max_length: int = 256,
    device: int = -1,
) -> pd.DataFrame:
    """Runs a pretrained model on text values and returns mapped predictions."""
    if model_key not in PRETRAINED_MODEL_SPECS:
        raise ValueError(f"Unknown model_key: {model_key}")

    spec = PRETRAINED_MODEL_SPECS[model_key]
    model_pipe = load_model_pipeline(spec["model_id"], task=spec["task"], device=device)

    text_series = text_values.fillna("").astype(str)
    raw_outputs = model_pipe(
        text_series.tolist(),
        truncation=True,
        max_length=max_length,
        batch_size=batch_size,
    )

    if isinstance(raw_outputs, dict):
        outputs = [raw_outputs]
    else:
        outputs = list(raw_outputs)

    mapped_labels: list[str | None] = []
    raw_labels: list[str | None] = []
    confidences: list[float | None] = []

    for output in outputs:
        best = _extract_best_output(output)
        raw_label = best.get("label") if isinstance(best, dict) else None
        score_value = best.get("score") if isinstance(best, dict) else None

        mapped = map_model_label_to_sentiment(raw_label, label_space=spec["label_space"])

        raw_labels.append(None if raw_label is None else str(raw_label))
        try:
            confidences.append(None if score_value is None else float(score_value))
        except (TypeError, ValueError):
            confidences.append(None)
        mapped_labels.append(mapped)

    return pd.DataFrame(
        {
            "pred_label": mapped_labels,
            "pred_raw_label": raw_labels,
            "pred_confidence": confidences,
        },
        index=text_series.index,
    )


def filter_eval_data_for_model(
    eval_df: pd.DataFrame,
    supported_langs: set[str] | None = None,
    language_column: str = "lang",
) -> pd.DataFrame:
    """Filters evaluation rows by model language support when possible."""
    if not supported_langs:
        return eval_df.copy()

    if language_column not in eval_df.columns:
        return eval_df.copy()

    allowed = {str(item).lower() for item in supported_langs}
    return eval_df[eval_df[language_column].astype(str).str.lower().isin(allowed)].copy()


def compute_classification_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    labels: list[str],
) -> dict[str, Any]:
    """Computes core metrics and a classification report."""
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "classification_report": _to_serializable(report),
    }

    for label in labels:
        label_report = report.get(label, {}) if isinstance(report, dict) else {}
        metrics[f"f1_{label}"] = float(label_report.get("f1-score", 0.0))

    return metrics


def evaluate_pretrained_model(
    eval_df: pd.DataFrame,
    text_column: str,
    model_key: str,
    label_mode: str = DEFAULT_LABEL_MODE,
    batch_size: int = 16,
    max_length: int = 256,
    device: int = -1,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluates one pretrained model on labeled evaluation data."""
    if model_key not in PRETRAINED_MODEL_SPECS:
        raise ValueError(f"Unknown model_key: {model_key}")

    spec = PRETRAINED_MODEL_SPECS[model_key]
    filtered = filter_eval_data_for_model(eval_df, supported_langs=spec.get("supported_langs"))
    labels = get_target_labels_for_mode(label_mode)

    if filtered.empty:
        return {
            "model_key": model_key,
            "model_id": spec["model_id"],
            "label_mode": label_mode,
            "status": "skipped_no_rows",
            "eval_rows": 0,
            "pred_rows": 0,
            "coverage": 0.0,
        }, pd.DataFrame()

    start_time = time.perf_counter()
    predictions = predict_with_pretrained_model(
        filtered[text_column],
        model_key=model_key,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )
    runtime_seconds = float(time.perf_counter() - start_time)

    eval_result = filtered.copy()
    eval_result["pred_label"] = predictions["pred_label"]
    eval_result["pred_raw_label"] = predictions["pred_raw_label"]
    eval_result["pred_confidence"] = predictions["pred_confidence"]

    valid_mask = eval_result["pred_label"].isin(labels)
    coverage = float(valid_mask.mean()) if len(valid_mask) > 0 else 0.0

    metrics: dict[str, Any] = {}
    if valid_mask.any():
        y_true = eval_result.loc[valid_mask, "target_label"].astype(str)
        y_pred = eval_result.loc[valid_mask, "pred_label"].astype(str)
        metrics = compute_classification_metrics(y_true, y_pred, labels=labels)

    row = {
        "model_key": model_key,
        "model_id": spec["model_id"],
        "label_space": spec.get("label_space"),
        "label_mode": label_mode,
        "supported_langs": ",".join(sorted(spec.get("supported_langs", set()))),
        "status": "ok",
        "eval_rows": int(len(eval_result)),
        "pred_rows": int(valid_mask.sum()),
        "coverage": coverage,
        "runtime_seconds": runtime_seconds,
        "accuracy": float(metrics.get("accuracy", 0.0)),
        "f1_macro": float(metrics.get("f1_macro", 0.0)),
        "f1_weighted": float(metrics.get("f1_weighted", 0.0)),
        "f1_negative": float(metrics.get("f1_negative", 0.0)),
        "f1_neutral": float(metrics.get("f1_neutral", 0.0)),
        "f1_positive": float(metrics.get("f1_positive", 0.0)),
        "metrics": metrics,
    }

    return row, eval_result


def benchmark_pretrained_models(
    input_path: str | Path,
    text_column: str | None = None,
    score_column: str = "score",
    label_mode: str = DEFAULT_LABEL_MODE,
    candidate_model_keys: tuple[str, ...] | list[str] | None = None,
    max_eval_rows: int | None = None,
    batch_size: int = 16,
    max_length: int = 256,
    device: int = -1,
) -> dict[str, Any]:
    """Benchmarks multiple pretrained models on a clean CSV."""
    input_path = Path(input_path)
    df = load_csv(input_path)
    selected_text_column = resolve_text_column(df, text_column=text_column)

    eval_df = build_labeled_feedback_data(
        df,
        text_column=selected_text_column,
        score_column=score_column,
        label_mode=label_mode,
    )
    eval_df = maybe_sample_eval_data(eval_df, max_eval_rows=max_eval_rows)

    model_keys = tuple(candidate_model_keys) if candidate_model_keys else DEFAULT_MODEL_KEYS

    result_rows: list[dict[str, Any]] = []
    detailed_evaluations: dict[str, pd.DataFrame] = {}

    for model_key in model_keys:
        if model_key not in PRETRAINED_MODEL_SPECS:
            result_rows.append(
                {
                    "model_key": model_key,
                    "status": "error_unknown_model_key",
                    "eval_rows": 0,
                    "pred_rows": 0,
                    "coverage": 0.0,
                    "accuracy": 0.0,
                    "f1_macro": 0.0,
                    "f1_weighted": 0.0,
                    "f1_negative": 0.0,
                    "f1_neutral": 0.0,
                    "f1_positive": 0.0,
                    "runtime_seconds": 0.0,
                    "error": "Unknown model key.",
                }
            )
            continue

        try:
            row, eval_result = evaluate_pretrained_model(
                eval_df,
                text_column=selected_text_column,
                model_key=model_key,
                label_mode=label_mode,
                batch_size=batch_size,
                max_length=max_length,
                device=device,
            )
            result_rows.append(row)
            if not eval_result.empty:
                detailed_evaluations[model_key] = eval_result
        except Exception as exc:
            spec = PRETRAINED_MODEL_SPECS[model_key]
            result_rows.append(
                {
                    "model_key": model_key,
                    "model_id": spec["model_id"],
                    "label_mode": label_mode,
                    "status": "error",
                    "eval_rows": 0,
                    "pred_rows": 0,
                    "coverage": 0.0,
                    "accuracy": 0.0,
                    "f1_macro": 0.0,
                    "f1_weighted": 0.0,
                    "f1_negative": 0.0,
                    "f1_neutral": 0.0,
                    "f1_positive": 0.0,
                    "runtime_seconds": 0.0,
                    "error": str(exc),
                }
            )

    benchmark_df = pd.DataFrame(result_rows)
    if not benchmark_df.empty:
        benchmark_df = benchmark_df.sort_values(
            ["status", "f1_macro", "accuracy", "coverage"],
            ascending=[True, False, False, False],
        ).reset_index(drop=True)

    ok_rows = benchmark_df[benchmark_df.get("status", "") == "ok"].copy()
    best_model_key: str | None = None

    if not ok_rows.empty:
        best_row = ok_rows.sort_values(
            ["f1_macro", "accuracy", "coverage"],
            ascending=[False, False, False],
        ).iloc[0]
        best_model_key = str(best_row["model_key"])

    class_counts = eval_df["target_label"].value_counts().to_dict() if not eval_df.empty else {}

    return {
        "input_path": input_path,
        "text_column": selected_text_column,
        "score_column": score_column,
        "label_mode": label_mode,
        "eval_rows": int(len(eval_df)),
        "class_counts": class_counts,
        "benchmark": benchmark_df,
        "best_model_key": best_model_key,
        "detailed_evaluations": detailed_evaluations,
    }


def apply_pretrained_model_to_clean_csv(
    input_path: str | Path,
    model_key: str,
    output_path: str | Path | None = None,
    text_column: str | None = None,
    target_column: str = "sentiment_pretrained",
    batch_size: int = 16,
    max_length: int = 256,
    device: int = -1,
    save_if_changed: bool = True,
) -> dict[str, Any]:
    """Applies one pretrained model to all rows in a clean CSV."""
    if model_key not in PRETRAINED_MODEL_SPECS:
        raise ValueError(f"Unknown model_key: {model_key}")

    input_path = Path(input_path)
    df_original = load_csv(input_path)
    selected_text_column = resolve_text_column(df_original, text_column=text_column)

    predictions = predict_with_pretrained_model(
        df_original[selected_text_column],
        model_key=model_key,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )

    df_result = df_original.copy()
    df_result[target_column] = predictions["pred_label"].fillna("unknown")
    df_result[f"{target_column}_confidence"] = predictions["pred_confidence"]
    df_result[f"{target_column}_raw_label"] = predictions["pred_raw_label"]
    df_result["pretrained_model_key"] = model_key
    df_result["pretrained_model_id"] = PRETRAINED_MODEL_SPECS[model_key]["model_id"]

    changed = not df_result.equals(df_original)
    saved_path: Path | None = None

    if changed and save_if_changed:
        final_output_path = Path(output_path) if output_path else build_pretrained_senti_output_path(input_path)
        saved_path = save_csv(df_result, final_output_path)

    counts = df_result[target_column].value_counts(dropna=False).to_dict()

    return {
        "input_path": input_path,
        "output_path": saved_path,
        "changed": changed,
        "saved": saved_path is not None,
        "model_key": model_key,
        "text_column": selected_text_column,
        "target_column": target_column,
        "row_count": int(len(df_result)),
        "sentiment_counts": counts,
        "dataframe": df_result,
    }


def run_pretrained_model_pipeline(
    input_path: str | Path,
    summary_output_path: str | Path | None = None,
    report_output_path: str | Path | None = None,
    prediction_output_path: str | Path | None = None,
    text_column: str | None = None,
    score_column: str = "score",
    label_mode: str = DEFAULT_LABEL_MODE,
    candidate_model_keys: tuple[str, ...] | list[str] | None = None,
    max_eval_rows: int | None = None,
    batch_size: int = 16,
    max_length: int = 256,
    device: int = -1,
    save_outputs: bool = True,
    save_best_predictions: bool = True,
) -> dict[str, Any]:
    """Runs pretrained model benchmark, picks best model and optionally saves artifacts."""
    benchmark_result = benchmark_pretrained_models(
        input_path=input_path,
        text_column=text_column,
        score_column=score_column,
        label_mode=label_mode,
        candidate_model_keys=candidate_model_keys,
        max_eval_rows=max_eval_rows,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )

    input_path = Path(input_path)
    benchmark_df = benchmark_result["benchmark"]
    best_model_key = benchmark_result["best_model_key"]

    saved_summary_path: Path | None = None
    saved_report_path: Path | None = None
    saved_prediction_path: Path | None = None
    prediction_result: dict[str, Any] | None = None

    if save_outputs:
        final_summary_path = (
            Path(summary_output_path) if summary_output_path else build_pretrained_summary_output_path(input_path)
        )
        saved_summary_path = save_csv(benchmark_df, final_summary_path)

        report_payload = {
            "input_path": benchmark_result["input_path"],
            "text_column": benchmark_result["text_column"],
            "score_column": benchmark_result["score_column"],
            "label_mode": benchmark_result["label_mode"],
            "eval_rows": benchmark_result["eval_rows"],
            "class_counts": benchmark_result["class_counts"],
            "best_model_key": best_model_key,
            "benchmark": benchmark_df.to_dict(orient="records"),
        }

        final_report_path = (
            Path(report_output_path) if report_output_path else build_pretrained_report_output_path(input_path)
        )
        saved_report_path = save_json(report_payload, final_report_path)

    if save_best_predictions and best_model_key:
        prediction_result = apply_pretrained_model_to_clean_csv(
            input_path=input_path,
            model_key=best_model_key,
            output_path=prediction_output_path,
            text_column=text_column,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
            save_if_changed=save_outputs,
        )
        saved_prediction_path = prediction_result.get("output_path")

    return {
        "input_path": input_path,
        "text_column": benchmark_result["text_column"],
        "score_column": benchmark_result["score_column"],
        "label_mode": benchmark_result["label_mode"],
        "eval_rows": benchmark_result["eval_rows"],
        "class_counts": benchmark_result["class_counts"],
        "benchmark": benchmark_df,
        "best_model_key": best_model_key,
        "summary_output_path": saved_summary_path,
        "report_output_path": saved_report_path,
        "prediction_output_path": saved_prediction_path,
        "prediction_result": prediction_result,
    }


def choose_label_mode_cli(default_mode: str = DEFAULT_LABEL_MODE) -> str:
    """CLI helper to choose binary or three-class evaluation mode."""
    print("Label mode options:")
    print("1. three_class (negative, neutral, positive)")
    print("2. binary (negative, positive)")
    print(f"Press Enter to use default: {default_mode}")

    while True:
        value = input("Choose label mode [1/2]: ").strip()
        if value == "":
            return default_mode
        if value == "1":
            return "three_class"
        if value == "2":
            return "binary"
        if value in {"three_class", "binary"}:
            return value
        print("Invalid label mode. Please try again.")


def main() -> None:
    """CLI entry point for pretrained model benchmarking and prediction."""
    clean_files = find_clean_csv_files()
    input_path = choose_clean_csv_cli(clean_files)
    label_mode = choose_label_mode_cli()

    result = run_pretrained_model_pipeline(
        input_path=input_path,
        label_mode=label_mode,
        save_outputs=True,
        save_best_predictions=True,
    )

    print("Pretrained benchmark completed.")
    print(f"Best model key: {result['best_model_key']}")
    if result["summary_output_path"]:
        print(f"Summary CSV saved to: {result['summary_output_path']}")
    if result["report_output_path"]:
        print(f"Report JSON saved to: {result['report_output_path']}")
    if result["prediction_output_path"]:
        print(f"Prediction CSV saved to: {result['prediction_output_path']}")

    benchmark_df = result["benchmark"]
    if not benchmark_df.empty:
        columns = [
            "model_key",
            "status",
            "eval_rows",
            "coverage",
            "accuracy",
            "f1_macro",
            "f1_weighted",
            "runtime_seconds",
        ]
        available_columns = [column for column in columns if column in benchmark_df.columns]
        print("Benchmark overview:")
        print(benchmark_df[available_columns].to_string(index=False))


if __name__ == "__main__":
    main()