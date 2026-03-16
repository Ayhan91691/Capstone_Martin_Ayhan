from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any

import pandas as pd


FUNCTIONS_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATTERN = "*_best_model.pkl"


def find_best_model_files(search_dir: str | Path = FUNCTIONS_DIR) -> list[Path]:
    """Finds all saved sentiment model files in the given directory."""
    search_dir = Path(search_dir)
    if not search_dir.exists():
        return []
    return sorted(path for path in search_dir.glob(DEFAULT_MODEL_PATTERN) if path.is_file())


def select_default_model_path(search_dir: str | Path = FUNCTIONS_DIR) -> Path:
    """Selects the default model path (prefers de+en, otherwise newest model)."""
    model_files = find_best_model_files(search_dir)
    if not model_files:
        raise FileNotFoundError("No *_best_model.pkl file found in functions directory.")

    de_en_models = [path for path in model_files if "_de+en_" in path.stem]
    if de_en_models:
        return max(de_en_models, key=lambda path: path.stat().st_mtime)

    return max(model_files, key=lambda path: path.stat().st_mtime)


def resolve_model_path(model_path: str | Path | None = None) -> Path:
    """Resolves the model path from input or auto-selects a default model."""
    if model_path is None:
        return select_default_model_path(FUNCTIONS_DIR)

    candidate = Path(model_path)
    if candidate.is_absolute() and candidate.exists() and candidate.is_file():
        return candidate

    relative_candidates = [
        Path.cwd() / candidate,
        FUNCTIONS_DIR / candidate,
    ]

    for path in relative_candidates:
        if path.exists() and path.is_file():
            return path

    raise FileNotFoundError(f"Model file not found: {model_path}")


def load_model_bundle(model_path: str | Path | None = None) -> dict:
    """Loads a serialized model bundle from a .pkl file."""
    resolved_path = resolve_model_path(model_path)
    with resolved_path.open("rb") as file:
        model_bundle = pickle.load(file)

    if "pipeline" not in model_bundle:
        raise ValueError("Invalid model bundle: missing 'pipeline'.")

    model_bundle = dict(model_bundle)
    model_bundle["model_path"] = resolved_path
    return model_bundle


def _normalize_target_id(raw_prediction: Any, is_regression: bool, threshold: float = 0.5) -> int:
    if is_regression:
        value = float(raw_prediction)
        return 1 if value >= threshold else 0

    try:
        value = float(raw_prediction)
        return int(round(value)) if value >= 0.5 else 0
    except (TypeError, ValueError):
        text = str(raw_prediction).strip().lower()
        if text in {"positive", "pos", "1", "true"}:
            return 1
        return 0


def _target_id_to_label(target_id: int, target_labels: dict | None = None) -> str:
    labels = target_labels or {0: "negative", 1: "positive"}
    return (
        labels.get(target_id)
        or labels.get(str(target_id))
        or ("positive" if target_id == 1 else "negative")
    )


def predict_sentiment_text(
    text: str,
    model_path: str | Path | None = None,
    model_bundle: dict | None = None,
) -> dict:
    """
    Predicts sentiment (positive/negative) for one input sentence.

    This function is UI-ready and can be used directly in Streamlit.
    """
    if text is None or not str(text).strip():
        raise ValueError("Input text must not be empty.")

    bundle = model_bundle if model_bundle is not None else load_model_bundle(model_path)
    pipeline = bundle["pipeline"]
    is_regression = bool(bundle.get("is_regression", False))
    threshold = float(bundle.get("threshold", 0.5))

    raw_prediction = pipeline.predict(pd.Series([str(text)]))[0]
    target_id = _normalize_target_id(raw_prediction, is_regression=is_regression, threshold=threshold)
    sentiment = _target_id_to_label(target_id, target_labels=bundle.get("target_labels"))

    return {
        "text": str(text),
        "sentiment": sentiment,
        "target_id": int(target_id),
        "model_name": bundle.get("model_name", "unknown_model"),
        "model_path": str(bundle.get("model_path", model_path or "")),
    }


def predict_sentiment_batch(
    texts: list[str],
    model_path: str | Path | None = None,
    model_bundle: dict | None = None,
) -> list[dict]:
    """Predicts sentiment for a list of input sentences."""
    bundle = model_bundle if model_bundle is not None else load_model_bundle(model_path)
    pipeline = bundle["pipeline"]
    is_regression = bool(bundle.get("is_regression", False))
    threshold = float(bundle.get("threshold", 0.5))

    series = pd.Series(["" if text is None else str(text) for text in texts])
    raw_predictions = pipeline.predict(series)

    results = []
    for text, raw_prediction in zip(series.tolist(), raw_predictions):
        target_id = _normalize_target_id(raw_prediction, is_regression=is_regression, threshold=threshold)
        sentiment = _target_id_to_label(target_id, target_labels=bundle.get("target_labels"))
        results.append(
            {
                "text": text,
                "sentiment": sentiment,
                "target_id": int(target_id),
                "model_name": bundle.get("model_name", "unknown_model"),
                "model_path": str(bundle.get("model_path", model_path or "")),
            }
        )

    return results


def main() -> None:
    """Simple CLI for interactive sentiment prediction."""
    model_arg = " ".join(sys.argv[1:]).strip()
    model_path = model_arg if model_arg else None

    try:
        bundle = load_model_bundle(model_path)
    except Exception as error:
        print(f"Could not load model: {error}")
        return

    print(f"Using model: {bundle.get('model_name', 'unknown_model')}")
    print(f"Model path: {bundle.get('model_path')}")
    print("Type a sentence in German or English. Type 'exit' to stop.")

    while True:
        text = input("Text: ").strip()
        if text.lower() in {"exit", "quit", "q"}:
            break
        if not text:
            print("Please enter a sentence.")
            continue

        result = predict_sentiment_text(text, model_bundle=bundle)
        print(f"Sentiment: {result['sentiment']}")


if __name__ == "__main__":
    main()
