import pandas as pd
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.metrics import accuracy_score, f1_score


def compare_and_save_best_model(file_path: str, text_col: str = "content", mode: str = "3class"):
    df = pd.read_csv(file_path, encoding="utf-8-sig")
    df = df[[text_col, "score"]].dropna().copy()
    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[df[text_col] != ""]
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["score"])

    if mode == "3class":
        def make_target(value):
            if value >= 4:
                return "positive"
            if value <= 2:
                return "negative"
            return "neutral"
    elif mode == "2class":
        def make_target(value):
            if value >= 4:
                return "positive"
            if value <= 2:
                return "negative"
            return None
    else:
        raise ValueError("mode must be '3class' or '2class'")

    df["target"] = df["score"].apply(make_target)
    df = df.dropna(subset=["target"])

    classes = sorted(df["target"].unique().tolist())
    n_classes = len(classes)

    if n_classes < 2:
        raise ValueError(f"Zu wenige Klassen für {mode}: {classes}")

    if n_classes == 2:
        models = {
            "logreg": LogisticRegression(max_iter=2000, class_weight="balanced"),
            "linearsvc": LinearSVC(class_weight="balanced"),
            "sgd": SGDClassifier(loss="hinge", class_weight="balanced", random_state=42),
            "mnb": MultinomialNB(),
            "cnb": ComplementNB()
        }
    else:
        models = {
            "logreg": LogisticRegression(max_iter=2000, class_weight="balanced"),
            "linearsvc": LinearSVC(class_weight="balanced"),
            "sgd": SGDClassifier(loss="hinge", class_weight="balanced", random_state=42),
            "cnb": ComplementNB()
        }

    X_train, X_test, y_train, y_test = train_test_split(
        df[text_col], df["target"], test_size=0.2, random_state=42, stratify=df["target"]
    )

    results = []
    best_model = None
    best_score = -1
    best_name = ""

    for name, model in models.items():
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
            ("model", model)
        ])

        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)

        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average="macro")

        results.append({
            "model": name,
            "accuracy": acc,
            "f1_macro": f1
        })

        if f1 > best_score:
            best_score = f1
            best_model = pipe
            best_name = name

    results_df = pd.DataFrame(results).sort_values("f1_macro", ascending=False)

    out_path = Path(file_path).with_name(f"best_model_{mode}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(best_model, f)

    print(results_df)
    print(f"\nBest model: {best_name} | F1: {best_score:.3f}")
    print(f"Saved: {out_path}")

    return results_df, best_model

if __name__ == "__main__":
    # results_3, best_model_3 = compare_and_save_best_model(
    # "data/My_BMW_en_raw_clean.csv",
    # mode="3class"
    # )

    # results_2, best_model_2 = compare_and_save_best_model(
    #     "data/My_BMW_en_raw_clean.csv",
    #     mode="2class"
    # )
    
    # import pickle
    # with open("data/best_model_3class.pkl", "rb") as f:
    #     model_3 = pickle.load(f)
    # print(model_3.predict(["this app is great"]))

    with open("data/best_model_2class.pkl", "rb") as f:
        model_2 = pickle.load(f)
    print(model_2.predict(["this is not a good app"]))