import json
import os
from functools import lru_cache
from pathlib import Path

import faiss
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

ENV_PATH = PROJECT_ROOT / ".env"
CURRENT_DIR = Path(__file__).resolve().parent
PROMPT_CONFIG_PATH = CURRENT_DIR / "prompt_configs.json"

load_dotenv(dotenv_path=ENV_PATH)


# =========================
# ENV / CLIENT
# =========================
@lru_cache(maxsize=1)
def get_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(f"GROQ_API_KEY not found in .env or environment. Looked at: {ENV_PATH}")
    return Groq(api_key=api_key)


# =========================
# CACHED LOADERS
# =========================
@lru_cache(maxsize=4)
def load_embedder(model_name: str):
    return SentenceTransformer(model_name)


@lru_cache(maxsize=8)
def load_rag_assets(rag_dir_str: str):
    rag_dir = Path(rag_dir_str)

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

    return config, df, index


@lru_cache(maxsize=1)
def load_prompt_configs() -> dict:
    if not PROMPT_CONFIG_PATH.exists():
        raise FileNotFoundError(f"prompt_configs.json nicht gefunden: {PROMPT_CONFIG_PATH}")

    with open(PROMPT_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================
# HELPERS
# =========================
def _resolve_rag_dir(rag_dir: str | Path) -> Path:
    rag_dir = Path(rag_dir)
    if not rag_dir.is_absolute():
        rag_dir = PROJECT_ROOT / rag_dir
    return rag_dir


def _shorten_text(text: str, max_chars: int = 260) -> str:
    text = str(text).strip().replace("\n", " ")
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."


def _build_system_prompt(
    prompt_key: str,
    prompt_configs: dict,
    question: str,
    context: str,
) -> str:
    if prompt_key not in prompt_configs:
        available = ", ".join(prompt_configs.keys())
        raise ValueError(f"Unknown prompt_key '{prompt_key}'. Available prompt keys: {available}")

    system_prompt_template = prompt_configs[prompt_key]["system_prompt"]

    if prompt_key == "report":
        return system_prompt_template.format(
            issue_name=question,
            context=context,
        )

    return system_prompt_template


# =========================
# RETRIEVAL
# =========================
def search_rag(
    rag_dir: str | Path,
    query: str,
    k: int = 5,
    negative_only: bool = False,
) -> pd.DataFrame:
    rag_dir = _resolve_rag_dir(rag_dir)

    config, df, index = load_rag_assets(str(rag_dir))
    embedder = load_embedder(config["model_name"])

    query_embedding = embedder.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(query_embedding, min(max(k * 5, k), len(df)))

    result = df.iloc[indices[0]].copy()
    result["distance"] = distances[0]

    if negative_only and "score" in result.columns:
        result["score"] = pd.to_numeric(result["score"], errors="coerce")
        result = result[result["score"] <= 2]

    return result.head(k).reset_index(drop=True)


# =========================
# GROQ QA
# =========================
def ask_rag_question_groq(
    rag_dir: str | Path,
    question: str,
    k: int = 5,
    negative_only: bool = False,
    max_context_chars: int = 260,
    model_name: str = "llama-3.1-8b-instant",
    temperature: float = 0.0,
    prompt_key: str = "strict",
) -> dict:
    """
    Nutzt einen vorhandenen RAG-Ordner und beantwortet eine Frage per Groq.
    Es wird nichts gespeichert.

    Rückgabe:
    {
        "question": ...,
        "answer": ...,
        "sources": DataFrame,
        "used_reviews": list[str],
        "model_name": ...,
        "temperature": ...,
        "prompt_key": ...,
        "prompt_label": ...
    }
    """
    try:
        temperature = float(temperature)
    except Exception:
        temperature = 0.0

    temperature = max(0.0, min(1.0, temperature))

    prompt_configs = load_prompt_configs()

    docs = search_rag(
        rag_dir=rag_dir,
        query=question,
        k=k,
        negative_only=negative_only,
    )

    if docs.empty:
        prompt_label = prompt_configs.get(prompt_key, {}).get("label", prompt_key)
        return {
            "question": question,
            "answer": "No matching reviews found.",
            "sources": docs,
            "used_reviews": [],
            "model_name": model_name,
            "temperature": temperature,
            "prompt_key": prompt_key,
            "prompt_label": prompt_label,
        }

    context_parts = []
    used_reviews = []

    for i, (_, row) in enumerate(docs.iterrows(), start=1):
        text = _shorten_text(row.get("content", ""), max_chars=max_context_chars)
        score = row.get("score", "")
        context_parts.append(f"Review {i} | Score: {score} | Text: {text}")
        used_reviews.append(text)

    context = "\n".join(context_parts)

    system_prompt = _build_system_prompt(
        prompt_key=prompt_key,
        prompt_configs=prompt_configs,
        question=question,
        context=context,
    )

    user_prompt = f"""
Reviews:
{context}

Question:
{question}
""".strip()

    client = get_groq_client()

    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )

    answer = completion.choices[0].message.content.strip()
    prompt_label = prompt_configs[prompt_key]["label"]

    return {
        "question": question,
        "answer": answer,
        "sources": docs,
        "used_reviews": used_reviews,
        "model_name": model_name,
        "temperature": temperature,
        "prompt_key": prompt_key,
        "prompt_label": prompt_label,
    }


# =========================
# OPTIONAL CONVENIENCE FUNCTION
# =========================
def ask_bmw_rag_question_groq(
    question: str,
    k: int = 5,
    negative_only: bool = False,
    model_name: str = "llama-3.1-8b-instant",
    temperature: float = 0.0,
    prompt_key: str = "strict",
) -> dict:
    rag_dir = DATA_DIR / "My_BMW_en_raw_clean_rag"
    return ask_rag_question_groq(
        rag_dir=rag_dir,
        question=question,
        k=k,
        negative_only=negative_only,
        model_name=model_name,
        temperature=temperature,
        prompt_key=prompt_key,
    )


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("ENV_PATH:", ENV_PATH)
    print("PROMPT_CONFIG_PATH:", PROMPT_CONFIG_PATH)
    print("GROQ_API_KEY loaded:", bool(os.getenv("GROQ_API_KEY")))

    result = ask_bmw_rag_question_groq(
        question="What are the main login and connectivity problems mentioned by users?",
        k=5,
        negative_only=True,
        model_name="llama-3.1-8b-instant",
        temperature=0.0,
        prompt_key="strict",
    )

    print("\n=== QUESTION ===\n")
    print(result["question"])

    print("\n=== MODEL ===\n")
    print(result["model_name"])

    print("\n=== TEMPERATURE ===\n")
    print(result["temperature"])

    print("\n=== PROMPT ===\n")
    print(result["prompt_label"])

    print("\n=== ANSWER ===\n")
    print(result["answer"])

    print("\n=== SOURCES ===\n")
    print(result["sources"][["content", "score", "distance"]])