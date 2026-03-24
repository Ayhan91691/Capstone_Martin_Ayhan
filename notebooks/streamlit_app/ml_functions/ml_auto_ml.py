import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)


HF_MODELS = {
    "english_sst2_distilbert": "distilbert-base-uncased-finetuned-sst-2-english",
    "twitter_roberta_english_sentiment": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "mbert_star_sentiment": "nlptown/bert-base-multilingual-uncased-sentiment",
}


def _make_target(score, mode: str):
    if mode == "3class":
        if score >= 4:
            return "positive"
        if score <= 2:
            return "negative"
        return "neutral"

    if mode == "2class":
        if score >= 4:
            return "positive"
        if score <= 2:
            return "negative"
        return None

    raise ValueError("mode must be '3class' or '2class'")


def _normalize_label(model_key: str, raw_label: str, mode: str):
    label = str(raw_label).strip().lower()

    if model_key == "mbert_star_sentiment":
        if "1" in label or "2" in label:
            return "negative"
        if "4" in label or "5" in label:
            return "positive"
        return "neutral" if mode == "3class" else None

    if model_key == "twitter_roberta_english_sentiment":
        if label in {"label_0", "negative"}:
            return "negative"
        if label in {"label_1", "neutral"}:
            return "neutral" if mode == "3class" else None
        if label in {"label_2", "positive"}:
            return "positive"

    if model_key == "english_sst2_distilbert":
        if "positive" in label:
            return "positive"
        if "negative" in label:
            return "negative"

    return None


def create_confusion_matrix(
    y_true,
    y_pred,
    labels: list[str],
    save_dir: Path,
    title: str,
):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    cm_df.to_csv(save_dir / "confusion_matrix.csv", encoding="utf-8-sig")

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    tick_marks = range(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(save_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Confusion Matrix gespeichert: {save_dir / 'confusion_matrix.csv'}", flush=True)
    print(f"Confusion Matrix Plot gespeichert: {save_dir / 'confusion_matrix.png'}", flush=True)

    return cm_df


def compare_and_save_best_hf_model(
    file_path: str,
    text_col: str = "content",
    mode: str = "3class",
    sample_size: int | None = 100,
):
    print(f"\n=== Starte Modellvergleich | mode={mode} ===", flush=True)
    print(f"Datei: {file_path}", flush=True)

    df = pd.read_csv(file_path, encoding="utf-8-sig")

    required_cols = [text_col, "score"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Fehlende Spalten in CSV: {missing_cols}")

    df = df[[text_col, "score"]].dropna().copy()
    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[df[text_col] != ""]
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["score"])

    print(f"Datensätze nach Cleaning: {len(df)}", flush=True)

    df["target"] = df["score"].apply(lambda x: _make_target(x, mode))
    df = df.dropna(subset=["target"])

    if df.empty:
        raise ValueError("Nach dem Preprocessing sind keine Daten mehr übrig.")

    print("Klassenverteilung:", flush=True)
    print(df["target"].value_counts(), flush=True)

    class_counts = df["target"].value_counts()
    if class_counts.min() < 2:
        raise ValueError(
            f"Zu wenige Samples pro Klasse für stratified split: {class_counts.to_dict()}"
        )

    X_train, X_test, y_train, y_test = train_test_split(
        df[text_col],
        df["target"],
        test_size=0.2,
        random_state=42,
        stratify=df["target"],
    )

    print(f"Train-Größe: {len(X_train)} | Test-Größe: {len(X_test)}", flush=True)

    if sample_size is not None:
        X_test = X_test.iloc[:sample_size]
        y_test = y_test.iloc[:sample_size]
        print(f"Testmenge für Evaluation begrenzt auf: {len(X_test)}", flush=True)

    if mode == "3class":
        model_keys = [
            "twitter_roberta_english_sentiment",
            "mbert_star_sentiment",
        ]
        labels = ["negative", "neutral", "positive"]
    else:
        model_keys = [
            "english_sst2_distilbert",
            "twitter_roberta_english_sentiment",
            "mbert_star_sentiment",
        ]
        labels = ["negative", "positive"]

    results = []
    best_name = ""
    best_score = -1.0
    best_accuracy = -1.0
    best_model_key = ""
    best_model_id = ""

    for model_key in model_keys:
        model_id = HF_MODELS[model_key]
        print(f"\nTeste Modell: {model_key} ({model_id})", flush=True)

        try:
            print("-> Lade Pipeline ...", flush=True)
            clf = pipeline(
                "text-classification",
                model=model_id,
                tokenizer=model_id,
            )
            print("-> Pipeline geladen", flush=True)

            test_texts = X_test.tolist()
            print(f"-> Anzahl Testtexte: {len(test_texts)}", flush=True)

            print("-> Starte Vorhersage ...", flush=True)
            raw_preds = clf(
                test_texts,
                truncation=True,
                max_length=256,
                batch_size=8,
            )
            print("-> Vorhersage fertig", flush=True)

            pred = [_normalize_label(model_key, x["label"], mode) for x in raw_preds]

            eval_df = pd.DataFrame(
                {
                    "y_true": y_test.tolist(),
                    "y_pred": pred,
                }
            ).dropna()

            if eval_df.empty:
                print(f"-> Übersprungen: {model_key} | keine validen Vorhersagen", flush=True)
                continue

            acc = accuracy_score(eval_df["y_true"], eval_df["y_pred"])
            f1 = f1_score(eval_df["y_true"], eval_df["y_pred"], average="macro")

            results.append(
                {
                    "model": model_key,
                    "model_id": model_id,
                    "accuracy": acc,
                    "f1_macro": f1,
                }
            )

            print(f"-> Accuracy: {acc:.4f} | F1 macro: {f1:.4f}", flush=True)

            if f1 > best_score:
                best_score = f1
                best_accuracy = acc
                best_name = model_key
                best_model_key = model_key
                best_model_id = model_id

        except Exception as e:
            print(f"-> Fehler bei Modell {model_key}: {e}", flush=True)
            continue

    if not results:
        raise ValueError("Kein passendes HF-Modell konnte ausgewertet werden.")

    results_df = pd.DataFrame(results).sort_values("f1_macro", ascending=False)

    save_dir = Path(file_path).parent / f"best_model_{mode}_auto"
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSpeichere bestes Modell nach: {save_dir}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(best_model_id)
    model = AutoModelForSequenceClassification.from_pretrained(best_model_id)
    tokenizer.save_pretrained(save_dir)
    model.save_pretrained(save_dir)

    print("\nBerechne Confusion Matrix für bestes Modell ...", flush=True)

    clf_best = pipeline(
        "text-classification",
        model=best_model_id,
        tokenizer=best_model_id,
    )

    best_raw_preds = clf_best(
        X_test.tolist(),
        truncation=True,
        max_length=256,
        batch_size=8,
    )
    best_pred = [_normalize_label(best_model_key, x["label"], mode) for x in best_raw_preds]

    eval_best_df = pd.DataFrame(
        {
            "y_true": y_test.tolist(),
            "y_pred": best_pred,
        }
    ).dropna()

    cm_df = create_confusion_matrix(
        y_true=eval_best_df["y_true"],
        y_pred=eval_best_df["y_pred"],
        labels=labels,
        save_dir=save_dir,
        title=f"Confusion Matrix ({mode})",
    )

    meta = {
        "best_model_key": best_model_key,
        "best_model_id": best_model_id,
        "mode": mode,
        "accuracy": best_accuracy,
        "f1_macro": best_score,
        "type": "huggingface_auto",
    }

    with open(save_dir / "model_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\nErgebnisse:", flush=True)
    print(results_df, flush=True)
    print(f"\nConfusion Matrix ({mode}):", flush=True)
    print(cm_df, flush=True)
    print(
        f"\nBest model: {best_name} | Accuracy: {best_accuracy:.3f} | F1: {best_score:.3f}",
        flush=True,
    )
    print(f"Saved: {save_dir}", flush=True)

    return results_df, str(save_dir), cm_df, best_model_key


if __name__ == "__main__":
    file_path = "data/My_BMW_en_raw_clean.csv"

    results_3, model_dir_3, cm_3, best_model_key_3 = compare_and_save_best_hf_model(
        file_path=file_path,
        mode="3class",
        sample_size=100,
    )

    results_2, model_dir_2, cm_2, best_model_key_2 = compare_and_save_best_hf_model(
        file_path=file_path,
        mode="2class",
        sample_size=100,
    )

    print("\nTeste gespeicherte Modelle ...", flush=True)

    clf_3 = pipeline(
        "text-classification",
        model=model_dir_3,
        tokenizer=model_dir_3,
    )
    raw_pred_3 = clf_3("this app is great")[0]
    normalized_pred_3 = _normalize_label(best_model_key_3, raw_pred_3["label"], "3class")
    print(
        "3class:",
        {
            "raw_label": raw_pred_3["label"],
            "normalized_label": normalized_pred_3,
            "score": raw_pred_3["score"],
        },
        flush=True,
    )

    clf_2 = pipeline(
        "text-classification",
        model=model_dir_2,
        tokenizer=model_dir_2,
    )
    raw_pred_2 = clf_2("this app is great")[0]
    normalized_pred_2 = _normalize_label(best_model_key_2, raw_pred_2["label"], "2class")
    print(
        "2class:",
        {
            "raw_label": raw_pred_2["label"],
            "normalized_label": normalized_pred_2,
            "score": raw_pred_2["score"],
        },
        flush=True,
    )