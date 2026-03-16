from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    make_scorer,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


FUNCTIONS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FUNCTIONS_DIR.parent
DEFAULT_ARTIFACT_DIR = FUNCTIONS_DIR
TEXT_COLUMN_CANDIDATES = ("lemmatized_text", "content_clean", "clean_text", "content")
TARGET_LABEL_MAP = {0: "negative", 1: "positive"}
SCORING_KEYS = ("f1_macro", "accuracy", "f1_positive")
DEFAULT_USE_GRID_SEARCH = True
DEFAULT_USE_NEURAL_MODELS = True


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


def build_senti_output_path(clean_path: str | Path) -> Path:
    """Builds the output path with *_senti.csv suffix."""
    clean_path = Path(clean_path)
    base_name = clean_path.stem[:-6] if clean_path.stem.endswith("_clean") else clean_path.stem
    return clean_path.with_name(f"{base_name}_senti.csv")


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

    raise ValueError(
        "No usable text column found. Provide one of: " + ", ".join(candidates)
    )


def score_to_binary_target(score: float | int | None) -> int | None:
    """Maps score to binary target labels: positive=1, negative=0, neutral=None."""
    if pd.isna(score):
        return None

    try:
        value = float(score)
    except (TypeError, ValueError):
        return None

    if value >= 4:
        return 1
    if value <= 2:
        return 0
    return None


def build_binary_training_data(
    df: pd.DataFrame,
    text_column: str,
    score_column: str = "score",
) -> pd.DataFrame:
    """Builds binary training data from score labels and text."""
    if score_column not in df.columns:
        raise ValueError(f"Missing required score column: {score_column}")

    train_df = df[[text_column, score_column]].copy()
    train_df[text_column] = train_df[text_column].fillna("").astype(str).str.strip()
    train_df = train_df[train_df[text_column] != ""]
    train_df["target_id"] = train_df[score_column].apply(score_to_binary_target)
    train_df = train_df.dropna(subset=["target_id"]).copy()
    train_df["target_id"] = train_df["target_id"].astype(int)
    return train_df


def get_model_candidates(
    random_state: int = 42,
    use_neural_models: bool = False,
) -> dict[str, dict[str, Any]]:
    """Returns candidate models for benchmarking."""
    candidates: dict[str, dict[str, Any]] = {
        "linear_regression": {
            "estimator": LinearRegression(),
            "is_regression": True,
        },
        "logistic_regression": {
            "estimator": LogisticRegression(max_iter=2000, class_weight="balanced"),
            "is_regression": False,
        },
        "linear_svc": {
            "estimator": LinearSVC(max_iter=5000, class_weight="balanced"),
            "is_regression": False,
        },
        "sgd_classifier": {
            "estimator": SGDClassifier(
                loss="hinge",
                class_weight="balanced",
                random_state=random_state,
                max_iter=2000,
                tol=1e-3,
            ),
            "is_regression": False,
        },
        "complement_nb": {
            "estimator": ComplementNB(alpha=1.0),
            "is_regression": False,
        },
    }

    if use_neural_models:
        candidates["mlp_classifier"] = {
            "estimator": MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                learning_rate_init=1e-3,
                max_iter=300,
                early_stopping=True,
                n_iter_no_change=10,
                random_state=random_state,
            ),
            "is_regression": False,
            "vectorizer_params": {
                "max_features": 5000,
                "ngram_range": (1, 1),
            },
        }

    return candidates


def get_grid_search_candidates(
    random_state: int = 42,
    use_neural_models: bool = False,
) -> dict[str, dict[str, Any]]:
    """Returns model candidates and parameter grids for hyperparameter search."""
    candidates: dict[str, dict[str, Any]] = {
        "linear_regression": {
            "estimator": LinearRegression(),
            "is_regression": True,
            "param_grid": {
                "tfidf__max_features": [5000, 10000],
                "tfidf__ngram_range": [(1, 1), (1, 2)],
                "tfidf__min_df": [1, 2],
            },
        },
        "logistic_regression": {
            "estimator": LogisticRegression(max_iter=4000, class_weight="balanced"),
            "is_regression": False,
            "param_grid": {
                "tfidf__max_features": [5000, 10000],
                "tfidf__ngram_range": [(1, 1), (1, 2)],
                "model__C": [0.5, 1.0, 2.0],
                "model__solver": ["liblinear", "lbfgs"],
            },
        },
        "linear_svc": {
            "estimator": LinearSVC(max_iter=8000, class_weight="balanced"),
            "is_regression": False,
            "param_grid": {
                "tfidf__max_features": [5000, 10000],
                "tfidf__ngram_range": [(1, 1), (1, 2)],
                "model__C": [0.5, 1.0, 2.0],
            },
        },
        "sgd_classifier": {
            "estimator": SGDClassifier(
                random_state=random_state,
                class_weight="balanced",
                max_iter=5000,
                tol=1e-3,
            ),
            "is_regression": False,
            "param_grid": {
                "tfidf__max_features": [5000, 10000],
                "tfidf__ngram_range": [(1, 1), (1, 2)],
                "model__loss": ["hinge", "log_loss"],
                "model__alpha": [1e-4, 1e-3],
            },
        },
        "complement_nb": {
            "estimator": ComplementNB(),
            "is_regression": False,
            "param_grid": {
                "tfidf__max_features": [5000, 10000],
                "tfidf__ngram_range": [(1, 1), (1, 2)],
                "model__alpha": [0.5, 1.0, 2.0],
            },
        },
    }

    if use_neural_models:
        candidates["mlp_classifier"] = {
            "estimator": MLPClassifier(
                random_state=random_state,
                max_iter=350,
                early_stopping=True,
                n_iter_no_change=10,
            ),
            "is_regression": False,
            "param_grid": {
                "tfidf__max_features": [3000, 5000],
                "tfidf__ngram_range": [(1, 1)],
                "tfidf__min_df": [1, 2],
                "model__hidden_layer_sizes": [(64,), (128,), (128, 64)],
                "model__alpha": [1e-4, 1e-3],
                "model__learning_rate_init": [1e-3, 5e-4],
            },
        }

    return candidates


def predict_target_ids(
    pipeline: Pipeline,
    text_values: pd.Series,
    is_regression: bool,
    threshold: float = 0.5,
) -> pd.Series:
    """Predicts binary target ids (0/1) from text values."""
    if text_values.empty:
        return pd.Series(index=text_values.index, dtype="int64")

    raw_prediction = pipeline.predict(text_values)
    prediction = pd.Series(raw_prediction, index=text_values.index)

    if is_regression:
        return (prediction >= threshold).astype(int)

    numeric_prediction = pd.to_numeric(prediction, errors="coerce").fillna(0)
    return numeric_prediction.round().astype(int).clip(0, 1)


def target_ids_to_labels(target_ids: pd.Series) -> pd.Series:
    """Converts binary target ids (0/1) into sentiment labels."""
    return target_ids.map(TARGET_LABEL_MAP)


def normalize_target_ids(y_pred: Any, threshold: float = 0.5) -> pd.Series:
    """Normalizes predictions from regressors/classifiers to binary ids (0/1)."""
    series = pd.Series(y_pred)
    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.notna().all():
        if numeric.isin([0, 1]).all():
            return numeric.astype(int)
        return (numeric >= threshold).astype(int)

    mapped = series.astype(str).str.lower().map({"negative": 0, "positive": 1})
    return mapped.fillna(0).astype(int)


def f1_macro_from_predictions(y_true: Any, y_pred: Any) -> float:
    """Computes macro F1 from raw model predictions."""
    y_pred_ids = normalize_target_ids(y_pred)
    return float(f1_score(y_true, y_pred_ids, average="macro", zero_division=0))


def accuracy_from_predictions(y_true: Any, y_pred: Any) -> float:
    """Computes accuracy from raw model predictions."""
    y_pred_ids = normalize_target_ids(y_pred)
    return float(accuracy_score(y_true, y_pred_ids))


def f1_positive_from_predictions(y_true: Any, y_pred: Any) -> float:
    """Computes positive-class F1 from raw model predictions."""
    y_pred_ids = normalize_target_ids(y_pred)
    return float(f1_score(y_true, y_pred_ids, average="binary", zero_division=0))


def get_grid_search_scoring() -> dict[str, Any]:
    """Returns multi-metric scoring dict for GridSearchCV."""
    return {
        "f1_macro": make_scorer(f1_macro_from_predictions),
        "accuracy": make_scorer(accuracy_from_predictions),
        "f1_positive": make_scorer(f1_positive_from_predictions),
    }


def compute_binary_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Computes metrics for binary sentiment predictions."""
    precision_pos, recall_pos, f1_pos, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        pos_label=1,
        zero_division=0,
    )

    y_true_labels = target_ids_to_labels(y_true)
    y_pred_labels = target_ids_to_labels(y_pred)

    report = classification_report(
        y_true_labels,
        y_pred_labels,
        labels=["negative", "positive"],
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_positive": float(f1_pos),
        "precision_positive": float(precision_pos),
        "recall_positive": float(recall_pos),
        "classification_report": _to_serializable(report),
    }


def evaluate_candidate_models(
    train_df: pd.DataFrame,
    text_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
    max_features: int = 10000,
    ngram_range: tuple[int, int] = (1, 2),
    use_neural_models: bool = False,
) -> dict:
    """Trains and evaluates multiple models on a train/test split."""
    x_values = train_df[text_column]
    y_values = train_df["target_id"]

    x_train, x_test, y_train, y_test = train_test_split(
        x_values,
        y_values,
        test_size=test_size,
        random_state=random_state,
        stratify=y_values,
    )

    candidates = get_model_candidates(
        random_state=random_state,
        use_neural_models=use_neural_models,
    )
    results: dict[str, dict] = {}
    trained_pipelines: dict[str, Pipeline] = {}

    for model_name, spec in candidates.items():
        estimator = spec["estimator"]
        is_regression = bool(spec["is_regression"])
        base_vectorizer_kwargs = {
            "max_features": max_features,
            "ngram_range": ngram_range,
        }
        vectorizer_kwargs = {
            **base_vectorizer_kwargs,
            **spec.get("vectorizer_params", {}),
        }

        try:
            pipeline = Pipeline(
                [
                    (
                        "tfidf",
                        TfidfVectorizer(**vectorizer_kwargs),
                    ),
                    ("model", estimator),
                ]
            )
            pipeline.fit(x_train, y_train)
            y_pred = predict_target_ids(
                pipeline,
                x_test,
                is_regression=is_regression,
                threshold=0.5,
            )
            metrics = compute_binary_metrics(y_test, y_pred)

            results[model_name] = {
                "status": "ok",
                "model_name": model_name,
                "is_regression": is_regression,
                "metrics": metrics,
                "n_train": int(len(x_train)),
                "n_test": int(len(x_test)),
            }
            trained_pipelines[model_name] = pipeline
        except Exception as error:
            results[model_name] = {
                "status": "error",
                "model_name": model_name,
                "is_regression": is_regression,
                "error": str(error),
                "n_train": int(len(x_train)),
                "n_test": int(len(x_test)),
            }

    return {
        "results": results,
        "pipelines": trained_pipelines,
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
    }


def select_best_model(
    model_results: dict[str, dict],
    primary_metric: str = "f1_macro",
) -> str:
    """Selects the best model by metric priority."""
    eligible = {
        model_name: result
        for model_name, result in model_results.items()
        if result.get("status") == "ok"
    }
    if not eligible:
        raise ValueError("No model could be trained successfully.")

    return max(
        eligible,
        key=lambda model_name: (
            eligible[model_name]["metrics"].get(primary_metric, 0.0),
            eligible[model_name]["metrics"].get("accuracy", 0.0),
            eligible[model_name]["metrics"].get("f1_positive", 0.0),
        ),
    )


def build_artifact_paths(
    input_path: str | Path,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    name_suffix: str | None = None,
) -> dict[str, Path]:
    """Builds output paths for model report JSON and best model pickle."""
    input_path = Path(input_path)
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    base_name = input_path.stem[:-6] if input_path.stem.endswith("_clean") else input_path.stem
    suffix = f"_{name_suffix}" if name_suffix else ""

    return {
        "report_json": artifact_dir / f"{base_name}{suffix}_model_results.json",
        "best_model": artifact_dir / f"{base_name}{suffix}_best_model.pkl",
    }


def run_grid_search_model_selection_pipeline(
    input_path: str | Path,
    text_column: str | None = None,
    score_column: str = "score",
    test_size: float = 0.2,
    random_state: int = 42,
    primary_metric: str = "f1_macro",
    use_neural_models: bool = False,
    cv: int = 3,
    n_jobs: int = -1,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    save_artifacts: bool = True,
) -> dict:
    """
    Runs GridSearchCV for multiple models, selects the best, and optionally saves artifacts.
    """
    input_path = Path(input_path)
    df = load_csv(input_path)
    selected_text_column = resolve_text_column(df, text_column=text_column)

    train_df = build_binary_training_data(df, text_column=selected_text_column, score_column=score_column)
    if train_df.empty:
        raise ValueError("No trainable rows found after filtering to binary sentiment labels.")

    class_count = train_df["target_id"].nunique()
    if class_count < 2:
        raise ValueError("Need at least two classes (positive and negative) to train models.")

    x_values = train_df[selected_text_column]
    y_values = train_df["target_id"]

    x_train, x_test, y_train, y_test = train_test_split(
        x_values,
        y_values,
        test_size=test_size,
        random_state=random_state,
        stratify=y_values,
    )

    candidates = get_grid_search_candidates(
        random_state=random_state,
        use_neural_models=use_neural_models,
    )
    scoring = get_grid_search_scoring()

    results: dict[str, dict] = {}
    tuned_pipelines: dict[str, Pipeline] = {}

    for model_name, spec in candidates.items():
        estimator = spec["estimator"]
        is_regression = bool(spec["is_regression"])
        param_grid = spec["param_grid"]

        try:
            pipeline = Pipeline(
                [
                    ("tfidf", TfidfVectorizer()),
                    ("model", estimator),
                ]
            )

            grid = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                scoring=scoring,
                refit=primary_metric,
                cv=cv,
                n_jobs=n_jobs,
                error_score="raise",
            )
            grid.fit(x_train, y_train)

            best_pipeline = grid.best_estimator_
            y_pred_test = predict_target_ids(
                best_pipeline,
                x_test,
                is_regression=is_regression,
                threshold=0.5,
            )
            test_metrics = compute_binary_metrics(y_test, y_pred_test)

            best_cv_scores = {}
            for scoring_key in SCORING_KEYS:
                cv_key = f"mean_test_{scoring_key}"
                if cv_key in grid.cv_results_:
                    best_cv_scores[scoring_key] = float(grid.cv_results_[cv_key][grid.best_index_])

            results[model_name] = {
                "status": "ok",
                "model_name": model_name,
                "is_regression": is_regression,
                "best_params": _to_serializable(grid.best_params_),
                "best_cv_scores": best_cv_scores,
                "metrics": test_metrics,
                "cv": int(cv),
                "n_train": int(len(x_train)),
                "n_test": int(len(x_test)),
            }
            tuned_pipelines[model_name] = best_pipeline
        except Exception as error:
            results[model_name] = {
                "status": "error",
                "model_name": model_name,
                "is_regression": is_regression,
                "error": str(error),
                "cv": int(cv),
                "n_train": int(len(x_train)),
                "n_test": int(len(x_test)),
            }

    best_model_name = select_best_model(results, primary_metric=primary_metric)
    best_model_result = results[best_model_name]
    best_pipeline = tuned_pipelines[best_model_name]

    best_model_bundle = {
        "model_name": best_model_name,
        "pipeline": best_pipeline,
        "is_regression": bool(best_model_result["is_regression"]),
        "threshold": 0.5,
        "text_column": selected_text_column,
        "score_column": score_column,
        "target_labels": TARGET_LABEL_MAP,
        "selection_mode": "grid_search",
        "cv": int(cv),
        "n_jobs": int(n_jobs),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    class_counts = {
        TARGET_LABEL_MAP[int(target_id)]: int(count)
        for target_id, count in train_df["target_id"].value_counts().to_dict().items()
    }

    report_payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_mode": "grid_search",
        "input_path": str(input_path),
        "text_column": selected_text_column,
        "score_column": score_column,
        "use_neural_models": bool(use_neural_models),
        "train_rows": int(len(train_df)),
        "class_counts": class_counts,
        "test_size": float(test_size),
        "random_state": int(random_state),
        "primary_metric": primary_metric,
        "cv": int(cv),
        "n_jobs": int(n_jobs),
        "best_model": {
            "model_name": best_model_name,
            "metrics": best_model_result["metrics"],
            "best_params": best_model_result.get("best_params", {}),
            "best_cv_scores": best_model_result.get("best_cv_scores", {}),
        },
        "models": results,
    }

    artifact_paths = build_artifact_paths(input_path, artifact_dir=artifact_dir, name_suffix="grid")
    report_json_path: Path | None = None
    best_model_path: Path | None = None

    if save_artifacts:
        report_json_path = save_json_report(report_payload, artifact_paths["report_json"])
        best_model_path = save_model_bundle(best_model_bundle, artifact_paths["best_model"])

    return {
        "input_path": input_path,
        "text_column": selected_text_column,
        "score_column": score_column,
        "train_rows": int(len(train_df)),
        "class_counts": class_counts,
        "primary_metric": primary_metric,
        "model_results": results,
        "best_model_name": best_model_name,
        "best_model_metrics": best_model_result["metrics"],
        "report_json_path": report_json_path,
        "best_model_path": best_model_path,
        "model_bundle": best_model_bundle,
    }


def save_json_report(payload: dict, path: str | Path) -> Path:
    """Saves a JSON report to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_to_serializable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def save_model_bundle(bundle: dict, path: str | Path) -> Path:
    """Saves a model bundle with pipeline and metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(bundle, file)
    return path


def load_model_bundle(path: str | Path) -> dict:
    """Loads a serialized model bundle."""
    with Path(path).open("rb") as file:
        return pickle.load(file)


def run_model_selection_pipeline(
    input_path: str | Path,
    text_column: str | None = None,
    score_column: str = "score",
    test_size: float = 0.2,
    random_state: int = 42,
    primary_metric: str = "f1_macro",
    use_neural_models: bool = False,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    save_artifacts: bool = True,
) -> dict:
    """
    Trains and benchmarks multiple sentiment models, then selects and stores the best model.

    Returns benchmark results, selected model metadata, and optional artifact paths.
    """
    input_path = Path(input_path)
    df = load_csv(input_path)
    selected_text_column = resolve_text_column(df, text_column=text_column)

    train_df = build_binary_training_data(df, text_column=selected_text_column, score_column=score_column)
    if train_df.empty:
        raise ValueError("No trainable rows found after filtering to binary sentiment labels.")

    class_count = train_df["target_id"].nunique()
    if class_count < 2:
        raise ValueError("Need at least two classes (positive and negative) to train models.")

    evaluation = evaluate_candidate_models(
        train_df,
        text_column=selected_text_column,
        test_size=test_size,
        random_state=random_state,
        use_neural_models=use_neural_models,
    )

    best_model_name = select_best_model(evaluation["results"], primary_metric=primary_metric)
    best_model_result = evaluation["results"][best_model_name]
    best_pipeline = evaluation["pipelines"][best_model_name]

    best_model_bundle = {
        "model_name": best_model_name,
        "pipeline": best_pipeline,
        "is_regression": bool(best_model_result["is_regression"]),
        "threshold": 0.5,
        "text_column": selected_text_column,
        "score_column": score_column,
        "target_labels": TARGET_LABEL_MAP,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    class_counts = {
        TARGET_LABEL_MAP[int(target_id)]: int(count)
        for target_id, count in train_df["target_id"].value_counts().to_dict().items()
    }

    report_payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "text_column": selected_text_column,
        "score_column": score_column,
        "use_neural_models": bool(use_neural_models),
        "train_rows": int(len(train_df)),
        "class_counts": class_counts,
        "test_size": float(test_size),
        "random_state": int(random_state),
        "primary_metric": primary_metric,
        "best_model": {
            "model_name": best_model_name,
            "metrics": best_model_result["metrics"],
        },
        "models": evaluation["results"],
    }

    artifact_paths = build_artifact_paths(input_path, artifact_dir=artifact_dir)
    report_json_path: Path | None = None
    best_model_path: Path | None = None

    if save_artifacts:
        report_json_path = save_json_report(report_payload, artifact_paths["report_json"])
        best_model_path = save_model_bundle(best_model_bundle, artifact_paths["best_model"])

    return {
        "input_path": input_path,
        "text_column": selected_text_column,
        "score_column": score_column,
        "train_rows": int(len(train_df)),
        "class_counts": class_counts,
        "primary_metric": primary_metric,
        "model_results": evaluation["results"],
        "best_model_name": best_model_name,
        "best_model_metrics": best_model_result["metrics"],
        "report_json_path": report_json_path,
        "best_model_path": best_model_path,
        "model_bundle": best_model_bundle,
    }


def apply_model_bundle_to_clean_csv(
    input_path: str | Path,
    model_bundle: dict,
    output_path: str | Path | None = None,
    text_column: str | None = None,
    target_column: str = "sentiment",
    save_if_changed: bool = True,
) -> dict:
    """Applies a model bundle to a clean CSV and optionally saves *_senti.csv."""
    input_path = Path(input_path)
    df_original = load_csv(input_path)

    preferred_text_column = text_column or model_bundle.get("text_column")
    selected_text_column = resolve_text_column(df_original, text_column=preferred_text_column)

    text_values = df_original[selected_text_column].fillna("").astype(str).str.strip()
    predicted_ids = predict_target_ids(
        model_bundle["pipeline"],
        text_values,
        is_regression=bool(model_bundle.get("is_regression", False)),
        threshold=float(model_bundle.get("threshold", 0.5)),
    )

    df_result = df_original.copy()
    df_result[target_column] = target_ids_to_labels(predicted_ids).fillna("negative")

    has_changes = not df_result.equals(df_original)
    saved_path: Path | None = None

    if has_changes and save_if_changed:
        final_output_path = Path(output_path) if output_path else build_senti_output_path(input_path)
        saved_path = save_csv(df_result, final_output_path)

    sentiment_counts = df_result[target_column].value_counts(dropna=False).to_dict()

    return {
        "input_path": input_path,
        "output_path": saved_path,
        "changed": has_changes,
        "saved": saved_path is not None,
        "text_column": selected_text_column,
        "target_column": target_column,
        "row_count": int(len(df_result)),
        "positive_count": int(sentiment_counts.get("positive", 0)),
        "negative_count": int(sentiment_counts.get("negative", 0)),
        "dataframe": df_result,
    }


def apply_saved_sentiment_model(
    input_path: str | Path,
    model_path: str | Path,
    output_path: str | Path | None = None,
    text_column: str | None = None,
    target_column: str = "sentiment",
    save_if_changed: bool = True,
) -> dict:
    """Loads a saved model and applies it to a clean CSV file."""
    model_bundle = load_model_bundle(model_path)
    return apply_model_bundle_to_clean_csv(
        input_path=input_path,
        model_bundle=model_bundle,
        output_path=output_path,
        text_column=text_column,
        target_column=target_column,
        save_if_changed=save_if_changed,
    )


def train_select_and_apply_sentiment(
    input_path: str | Path,
    output_path: str | Path | None = None,
    text_column: str | None = None,
    score_column: str = "score",
    target_column: str = "sentiment",
    test_size: float = 0.2,
    random_state: int = 42,
    primary_metric: str = "f1_macro",
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    save_if_changed: bool = True,
    use_grid_search: bool = False,
    use_neural_models: bool = False,
    grid_cv: int = 3,
    grid_n_jobs: int = -1,
) -> dict:
    """
    Full pipeline: benchmark models, pick best model, save artifacts, apply sentiment prediction.
    """
    if use_grid_search:
        selection_result = run_grid_search_model_selection_pipeline(
            input_path=input_path,
            text_column=text_column,
            score_column=score_column,
            test_size=test_size,
            random_state=random_state,
            primary_metric=primary_metric,
            use_neural_models=use_neural_models,
            cv=grid_cv,
            n_jobs=grid_n_jobs,
            artifact_dir=artifact_dir,
            save_artifacts=True,
        )
    else:
        selection_result = run_model_selection_pipeline(
            input_path=input_path,
            text_column=text_column,
            score_column=score_column,
            test_size=test_size,
            random_state=random_state,
            primary_metric=primary_metric,
            use_neural_models=use_neural_models,
            artifact_dir=artifact_dir,
            save_artifacts=True,
        )

    prediction_result = apply_model_bundle_to_clean_csv(
        input_path=input_path,
        model_bundle=selection_result["model_bundle"],
        output_path=output_path,
        text_column=text_column,
        target_column=target_column,
        save_if_changed=save_if_changed,
    )

    return {
        "input_path": Path(input_path),
        "selection_mode": "grid_search" if use_grid_search else "baseline_benchmark",
        "use_neural_models": bool(use_neural_models),
        "best_model_name": selection_result["best_model_name"],
        "best_model_metrics": selection_result["best_model_metrics"],
        "report_json_path": selection_result["report_json_path"],
        "best_model_path": selection_result["best_model_path"],
        "model_results": selection_result["model_results"],
        "sentiment_output_path": prediction_result["output_path"],
        "sentiment_saved": prediction_result["saved"],
        "text_column": prediction_result["text_column"],
        "target_column": prediction_result["target_column"],
        "positive_count": prediction_result["positive_count"],
        "negative_count": prediction_result["negative_count"],
        "row_count": prediction_result["row_count"],
        "dataframe": prediction_result["dataframe"],
    }


def analyze_clean_csv_sentiment(
    input_path: str | Path,
    output_path: str | Path | None = None,
    text_column: str | None = None,
    score_column: str = "score",
    target_column: str = "sentiment",
    save_if_changed: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
    primary_metric: str = "f1_macro",
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    use_grid_search: bool = False,
    use_neural_models: bool = False,
    grid_cv: int = 3,
    grid_n_jobs: int = -1,
) -> dict:
    """Backward-compatible wrapper around the full train/select/apply pipeline."""
    return train_select_and_apply_sentiment(
        input_path=input_path,
        output_path=output_path,
        text_column=text_column,
        score_column=score_column,
        target_column=target_column,
        test_size=test_size,
        random_state=random_state,
        primary_metric=primary_metric,
        artifact_dir=artifact_dir,
        save_if_changed=save_if_changed,
        use_grid_search=use_grid_search,
        use_neural_models=use_neural_models,
        grid_cv=grid_cv,
        grid_n_jobs=grid_n_jobs,
    )


def run_sentiment_pipeline(
    input_path: str | Path,
    output_path: str | Path | None = None,
    text_column: str | None = None,
    use_grid_search: bool = False,
    use_neural_models: bool = False,
    grid_cv: int = 3,
    grid_n_jobs: int = -1,
) -> Path | None:
    """Compatibility wrapper returning only the saved *_senti.csv path."""
    result = train_select_and_apply_sentiment(
        input_path=input_path,
        output_path=output_path,
        text_column=text_column,
        use_grid_search=use_grid_search,
        use_neural_models=use_neural_models,
        grid_cv=grid_cv,
        grid_n_jobs=grid_n_jobs,
    )
    return result["sentiment_output_path"]


def main() -> None:
    """CLI entry point to select a clean CSV and run full sentiment model pipeline."""
    clean_files = find_clean_csv_files()
    input_path = choose_clean_csv_cli(clean_files)
    result = train_select_and_apply_sentiment(
        input_path,
        use_grid_search=DEFAULT_USE_GRID_SEARCH,
        use_neural_models=DEFAULT_USE_NEURAL_MODELS,
    )

    print(f"Best model: {result['best_model_name']}")
    print(f"Benchmark report: {result['report_json_path']}")
    print(f"Saved model: {result['best_model_path']}")

    if result["sentiment_saved"]:
        print(f"Sentiment file saved to: {result['sentiment_output_path']}")
    else:
        print("No changes detected. No sentiment output file saved.")


if __name__ == "__main__":
    main()
