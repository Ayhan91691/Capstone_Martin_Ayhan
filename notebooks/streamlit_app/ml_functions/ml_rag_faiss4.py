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


def _shorten_text(text: str, max_chars: int = 300) -> str:
    text = str(text).strip().replace("\n", " ")
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."


def _build_context(docs: pd.DataFrame, max_chars: int = 300) -> str:
    context_parts = []
    for i, (_, row) in enumerate(docs.iterrows(), start=1):
        text = _shorten_text(row.get("content", ""), max_chars=max_chars)
        score = row.get("score", "")
        context_parts.append(f"Review {i} | Score: {score} | Text: {text}")
    return "\n".join(context_parts)


def _load_generator():
    return pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.2")


def _generate_mistral(generator, prompt: str, max_new_tokens: int = 220) -> str:
    print("📏 Prompt Länge:", len(prompt), flush=True)
    print("🚀 Generierung gestartet...", flush=True)

    response = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_full_text=False,
        pad_token_id=generator.tokenizer.eos_token_id
    )[0]["generated_text"]

    print("✅ Generierung fertig", flush=True)
    return response.strip()


def ask_rag(
    rag_dir: str | Path,
    query: str,
    k: int = 6,
    negative_only: bool = True,
):
    print("\n🔍 Retrieval gestartet...", flush=True)

    docs = search_rag(rag_dir, query, k=k, negative_only=negative_only)

    if docs.empty:
        return "No matching reviews found."

    print(f"Gefundene Reviews: {len(docs)}", flush=True)

    context = _build_context(docs, max_chars=260)

    print("🤖 Lade Mistral...", flush=True)
    generator = _load_generator()
    print("✅ Modell geladen", flush=True)

    prompt_1 = f"""[INST]
You are a product analyst for a mobile app.

Use ONLY the reviews below.

Task:
1. Summarize the main problems in max 5 bullet points.
2. Group the complaints into exactly 3 categories.
3. Give each category a short label.
4. For each category, describe the typical complaints briefly.

Output format:

## Problems
- ...
- ...
- ...

## Categories
### Category 1: <label>
- ...
- ...

### Category 2: <label>
- ...
- ...

### Category 3: <label>
- ...
- ...

Rules:
- Use only the reviews below.
- Do not invent information.
- Do not mention positive feedback.
- Keep it concise.
- Return exactly 3 categories.

Reviews:
{context}

Question:
{query}
[/INST]"""

    part_1 = _generate_mistral(generator, prompt_1, max_new_tokens=220)

    prompt_2 = f"""[INST]
You are a product analyst for a mobile app.

Use ONLY the analysis below.

Task:
1. For each of the 3 categories, suggest 2 to 3 concrete improvements / solutions.
2. Suggest 3 to 5 realistic new app features inspired by the complaints.

Output format:

## Solutions
### Category 1: <label>
- ...
- ...

### Category 2: <label>
- ...
- ...

### Category 3: <label>
- ...
- ...

## Feature Ideas
- ...
- ...
- ...

Rules:
- Base your answer only on the analysis below.
- Be practical and concise.
- Do not invent unrelated problems.

Analysis:
{part_1}
[/INST]"""

    part_2 = _generate_mistral(generator, prompt_2, max_new_tokens=260)

    final_result = f"""## Problems and Categories

{part_1}

## Solutions and Feature Ideas

{part_2}"""

    return final_result


if __name__ == "__main__":
    rag_dir = DATA_DIR / "My_BMW_en_raw_clean_rag"

    print("\n=== SEARCH RESULTS ===\n")
    print("Verwendeter rag_dir:", rag_dir)
    print("config exists:", (rag_dir / "config.json").exists())

    results = search_rag(
        rag_dir,
        "What are the main issues with the BMW app?",
        k=6,
        negative_only=True
    )

    print(results[["content", "score", "distance"]])

    print("\n=== AI ANALYSIS ===\n")

    answer = ask_rag(
        rag_dir,
        "Analyze complaints and suggest improvements",
        k=6,
        negative_only=True
    )

    print("\n=== RESULT ===\n")
    print(answer)