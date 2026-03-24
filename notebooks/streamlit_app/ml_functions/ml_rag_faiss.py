import json
import faiss
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


def build_rag_csv(
    file_path: str,
    text_col: str = "content",
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
):
    file_path = Path(file_path)
    if not file_path.is_absolute():
        file_path = PROJECT_ROOT / file_path

    df = pd.read_csv(file_path, encoding="utf-8-sig").copy()

    if text_col not in df.columns:
        raise ValueError(f"The input file must contain a '{text_col}' column.")

    df = df.dropna(subset=[text_col])
    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[df[text_col] != ""].reset_index(drop=True)

    embedder = SentenceTransformer(model_name)
    embeddings = embedder.encode(
        df[text_col].tolist(),
        convert_to_numpy=True,
        show_progress_bar=True
    ).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    base = file_path.with_suffix("")
    rag_dir = base.parent / f"{base.stem}_rag"
    rag_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(rag_dir / "index.faiss"))
    np.save(rag_dir / "embeddings.npy", embeddings)
    df.to_csv(rag_dir / "data.csv", index=False, encoding="utf-8-sig")

    with open(rag_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(
            {"text_col": text_col, "model_name": model_name},
            f,
            ensure_ascii=False,
            indent=2
        )

    return rag_dir


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


def ask_rag(
    rag_dir: str | Path,
    query: str,
    k: int = 10,
    negative_only: bool = True,
    llm_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
):
    docs = search_rag(rag_dir, query, k=k, negative_only=negative_only)

    if docs.empty:
        return "No matching reviews found."

    context_parts = []
    for _, row in docs.iterrows():
        text = row.get("content", "")
        score = row.get("score", "")
        context_parts.append(f"- Score: {score} | Review: {text}")

    context = "\n".join(context_parts)

    prompt = f"""[INST]
You are a product analyst for a mobile app.

Use ONLY the reviews below.

Your task:

1. Summarize the main problems (max 5 bullet points)

2. Group the negative feedback into 3–5 categories
   - Give each category a short label
   - Assign typical problems

3. For each category:
   - Suggest concrete improvements / solutions

4. Suggest NEW FEATURES for the app
   - Be creative but realistic

Output format:

## Problems
- ...

## Categories
### Category 1: ...
- ...

## Solutions
### Category 1:
- ...

## Feature Ideas
- ...

Rules:
- Be concise
- Do not invent information
- Do not mention positive feedback

Reviews:
{context}

Question:
{query}
[/INST]
"""

    generator = pipeline("text-generation", model=llm_name)

    response = generator(
        prompt,
        max_new_tokens=350,
        truncation=True,
        do_sample=False,
        pad_token_id=generator.tokenizer.eos_token_id
    )[0]["generated_text"]

    if "[/INST]" in response:
        response = response.split("[/INST]", 1)[-1].strip()

    return response


if __name__ == "__main__":
    rag_dir = DATA_DIR / "My_BMW_en_raw_clean_rag"

    print("\n=== SEARCH RESULTS ===\n")
    print("Verwendeter rag_dir:", rag_dir)
    print("config exists:", (rag_dir / "config.json").exists())

    results = search_rag(
        rag_dir,
        "What are the main issues with the BMW app?",
        k=10,
        negative_only=True
    )

    print(results[["content", "score", "distance"]])

    print("\n=== AI ANALYSIS ===\n")

    answer = ask_rag(
        rag_dir,
        "Analyze user complaints and suggest improvements",
        k=10,
        negative_only=True
    )

    print(answer)