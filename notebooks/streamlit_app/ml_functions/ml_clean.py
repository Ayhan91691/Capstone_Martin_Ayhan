import re
import pandas as pd
from pathlib import Path

def clean_csv(file_path: str):
    df = pd.read_csv(file_path, encoding="utf-8-sig")
    df_clean = df.copy()

    if "content" not in df_clean.columns:
        raise ValueError("The input file must contain a 'content' column.")

    if "reviewId" in df_clean.columns:
        subset = ["appId", "reviewId"] if "appId" in df_clean.columns else ["reviewId"]
        df_clean = df_clean.drop_duplicates(subset=subset)

    df_clean = df_clean.dropna(subset=["content"])
    df_clean = df_clean[df_clean["content"].astype(str).str.split().str.len() >= 2]

    keep_columns = [
        "reviewId", "content", "score", "thumbsUpCount",
        "reviewCreatedVersion", "at", "appVersion",
        "appTitle", "appId", "country", "lang"
    ]
    df_clean = df_clean[[col for col in keep_columns if col in df_clean.columns]]

    if "score" in df_clean.columns:
        df_clean["score"] = pd.to_numeric(df_clean["score"], errors="coerce").astype("Int64")
    if "at" in df_clean.columns:
        df_clean["at"] = pd.to_datetime(df_clean["at"], errors="coerce")

    def basic_text_clean(text):
        text = re.sub(r"<[^>]+>", "", str(text)).strip()
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9äöüÄÖÜß\s]", "", text)
        return text.strip()

    df_clean["content_clean"] = df_clean["content"].astype(str).apply(basic_text_clean)

    df_clean = df_clean.reset_index(drop=True)

    out = Path(file_path).with_name(Path(file_path).stem + "_clean.csv")
    df_clean.to_csv(out, index=False, encoding="utf-8-sig")
    return df_clean

if __name__ == "__main__":
    clean_csv("data/My_BMW_en_raw.csv")

