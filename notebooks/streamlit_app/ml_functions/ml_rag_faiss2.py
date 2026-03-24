import json
import faiss
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


def search_rag(rag_dir: str | Path, query: str, k: int = 5, negative_only: bool = False):
    rag_dir = Path(rag_dir)
    if not rag_dir.is_absolute():
        rag_dir = PROJECT_ROOT / rag_dir

    config_path = rag_dir / "config.json"
    data_path = rag_dir / "data.csv"
    index_path = rag_dir / "index.faiss"

    if not config_path.exists():
        raise FileNotFoundError(f"config.json nicht gefunden: {config_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"data.csv nicht gefunden: {data_path}")
    if not index_path.exists():
        raise FileNotFoundError(f"index.faiss nicht gefunden: {index_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    df = pd.read_csv(data_path, encoding="utf-8-sig")
    index = faiss.read_index(str(index_path))
    embedder = SentenceTransformer(config["model_name"])

    query_embedding = embedder.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(query_embedding, min(max(k * 5, k), len(df)))

    result = df.iloc[indices[0]].copy()
    result["distance"] = distances[0]

    if negative_only and "score" in result.columns:
        result["score"] = pd.to_numeric(result["score"], errors="coerce")
        result = result[result["score"] <= 2]

    return result.head(k).reset_index(drop=True)


def _shorten_text(text: str, max_chars: int = 220) -> str:
    text = str(text).strip().replace("\n", " ")
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."


def ask_rag(
    rag_dir: str | Path,
    query: str,
    k: int = 4,
    negative_only: bool = True,
):
    print("\n🔍 Retrieval gestartet...", flush=True)

    docs = search_rag(rag_dir, query, k=k, negative_only=negative_only)

    if docs.empty:
        return "No matching reviews found."

    print(f"Gefundene Reviews: {len(docs)}", flush=True)

    context_parts = []
    for i, (_, row) in enumerate(docs.iterrows(), start=1):
        text = _shorten_text(row.get("content", ""), max_chars=220)
        score = row.get("score", "")
        context_parts.append(f"Review {i} (score {score}): {text}")

    context = "\n".join(context_parts)

    prompt = f"""
Analyze these negative app reviews.

Task:
1. Write a short summary of the main problems.
2. Group the feedback into exactly 3 categories.
3. For each category, suggest solutions.
4. Suggest 3 realistic new app features.

Output format:

Problems:
- ...
- ...
- ...

Categories:
1. <label>
- complaints: ...
- solutions: ...

2. <label>
- complaints: ...
- solutions: ...

3. <label>
- complaints: ...
- solutions: ...

Feature ideas:
- ...
- ...
- ...

Reviews:
{context}

Question: {query}
""".strip()

    print("📏 Prompt Länge:", len(prompt), flush=True)
    print("🤖 Lade Modell...", flush=True)

    generator = pipeline("text2text-generation", model="google/flan-t5-base")

    # Für spätere Verwendung:
    # generator = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.2")

    print("🚀 Generierung gestartet...", flush=True)

    response = generator(
        prompt,
        max_new_tokens=180,
        truncation=True,
    )[0]["generated_text"]

    print("✅ Generierung fertig", flush=True)

    return response


if __name__ == "__main__":
    rag_dir = DATA_DIR / "My_BMW_en_raw_clean_rag"

    print("\n=== SEARCH RESULTS ===\n")
    print("Verwendeter rag_dir:", rag_dir)
    print("config exists:", (rag_dir / "config.json").exists())

    results = search_rag(
        rag_dir,
        "What are the main issues with the BMW app?",
        k=5,
        negative_only=True
    )

    print(results[["content", "score", "distance"]])

    print("\n=== AI ANALYSIS ===\n")

    answer = ask_rag(
        rag_dir,
        "Analyze complaints and suggest improvements",
        k=4,
        negative_only=True
    )

    print("\n=== RESULT ===\n")
    print(answer)