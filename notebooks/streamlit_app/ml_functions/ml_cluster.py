import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


def cluster_negative_issues(file_path: str, text_col: str = "content", k: int = 5, model_name: str = "all-MiniLM-L6-v2"):
    df = pd.read_csv(file_path, encoding="utf-8-sig")
    df = df[[text_col, "score"]].dropna().copy()
    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[df[text_col] != ""]
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["score"])

    df = df[df["score"] <= 2].copy()
    if df.empty:
        raise ValueError("Keine negativen Bewertungen vorhanden.")

    embeddings = SentenceTransformer(model_name).encode(df[text_col].tolist(), show_progress_bar=False)

    k = min(k, len(df))
    if k < 1:
        raise ValueError("k muss mindestens 1 sein.")

    if k == 1:
        df["issue_cluster"] = "issue_0"
    else:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings)
        df["issue_cluster"] = [f"issue_{label}" for label in labels]

    def extract_top_keywords(texts, top_n=8):
        if len(texts) == 0:
            return []
        vec = TfidfVectorizer(stop_words="english", max_features=2000, ngram_range=(1, 2))
        X = vec.fit_transform(texts)
        scores = X.mean(axis=0).A1
        terms = vec.get_feature_names_out()
        top_idx = scores.argsort()[::-1][:top_n]
        return [terms[i] for i in top_idx]

    def make_issue_label(keywords):
        kw = " ".join(keywords).lower()

        if any(x in kw for x in ["login", "sign in", "signin", "password", "account", "authentication"]):
            return "login_issue"
        if any(x in kw for x in ["charge", "charging", "charger", "cable", "battery", "connection", "connected"]):
            return "charging_issue"
        if any(x in kw for x in ["crash", "freez", "stuck", "loading", "open app", "close app", "update"]):
            return "app_stability_issue"
        if any(x in kw for x in ["payment", "pay", "billing", "invoice", "subscription", "purchase"]):
            return "payment_issue"
        if any(x in kw for x in ["navigation", "map", "route", "location", "gps"]):
            return "navigation_issue"
        if any(x in kw for x in ["support", "service", "contact", "help"]):
            return "support_issue"
        if any(x in kw for x in ["octopus", "tariff", "intelligent go", "energy", "electricity"]):
            return "tariff_integration_issue"
        if any(x in kw for x in ["pairing", "bluetooth", "connectivity", "device", "phone"]):
            return "connectivity_issue"

        return None

    summary_rows = []
    cluster_to_label = {}

    for cluster_name, cluster_df in df.groupby("issue_cluster"):
        keywords = extract_top_keywords(cluster_df[text_col].tolist())
        issue_label = make_issue_label(keywords)
        examples = cluster_df[text_col].head(3).tolist()

        cluster_to_label[cluster_name] = issue_label

        summary_rows.append({
            "issue_cluster": cluster_name,
            "issue_label": issue_label,
            "review_count": len(cluster_df),
            "top_keywords": ", ".join(keywords),
            "example_1": examples[0] if len(examples) > 0 else "",
            "example_2": examples[1] if len(examples) > 1 else "",
            "example_3": examples[2] if len(examples) > 2 else "",
        })

    df["issue_label"] = df["issue_cluster"].map(cluster_to_label)
    df["issue_label"] = df["issue_label"].fillna("")

    summary_df = pd.DataFrame(summary_rows).sort_values("review_count", ascending=False)

    out_csv = Path(file_path).with_name(Path(file_path).stem + "_negative_issues.csv")
    summary_csv = Path(file_path).with_name(Path(file_path).stem + "_top_issues.csv")

    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    print(summary_df)
    print(f"Saved clustered reviews: {out_csv}")
    print(f"Saved top issues: {summary_csv}")

    return df, summary_df


if __name__ == "__main__":
    clustered_df, issues_df = cluster_negative_issues(
        "data/My_BMW_en_raw_clean.csv",
        text_col="content_clean",
        k=5
    )
    print(issues_df)