import json
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from transformers import pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


# =========================
# BUILD RAG DATABASE
# =========================
def build_rag_csv(
    file_path: str | Path,
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


# =========================
# CACHED LOADERS
# =========================
@lru_cache(maxsize=4)
def load_embedder(model_name: str):
    return SentenceTransformer(model_name)


@lru_cache(maxsize=2)
def load_generator():
    return pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.2")


@lru_cache(maxsize=8)
def load_rag_assets(rag_dir_str: str):
    rag_dir = Path(rag_dir_str)

    config_path = rag_dir / "config.json"
    data_path = rag_dir / "data.csv"
    index_path = rag_dir / "index.faiss"
    embeddings_path = rag_dir / "embeddings.npy"

    if not config_path.exists():
        raise FileNotFoundError(f"config.json nicht gefunden: {config_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"data.csv nicht gefunden: {data_path}")
    if not index_path.exists():
        raise FileNotFoundError(f"index.faiss nicht gefunden: {index_path}")
    if not embeddings_path.exists():
        raise FileNotFoundError(f"embeddings.npy nicht gefunden: {embeddings_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    df = pd.read_csv(data_path, encoding="utf-8-sig")
    index = faiss.read_index(str(index_path))
    embeddings = np.load(embeddings_path).astype("float32")

    return config, df, index, embeddings


# =========================
# SEARCH (klassisches RAG)
# =========================
def search_rag(
    rag_dir: str | Path,
    query: str,
    k: int = 5,
    negative_only: bool = False
):
    rag_dir = Path(rag_dir)
    if not rag_dir.is_absolute():
        rag_dir = PROJECT_ROOT / rag_dir

    config, df, index, _ = load_rag_assets(str(rag_dir))
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
# CACHE HELPERS
# =========================
def _cache_is_fresh(cache_file: Path, source_files: list[Path]) -> bool:
    if not cache_file.exists():
        return False

    cache_mtime = cache_file.stat().st_mtime
    return all(src.exists() and src.stat().st_mtime <= cache_mtime for src in source_files)


def _shorten_text(text: str, max_chars: int = 280) -> str:
    text = str(text).strip().replace("\n", " ")
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."


# =========================
# CLUSTER + SAMPLE
# =========================
def cluster_sample_reviews(
    rag_dir: str | Path,
    n_clusters: int = 5,
    samples_per_cluster: int = 2,
    negative_only: bool = True,
    force_recompute: bool = False,
):
    rag_dir = Path(rag_dir)
    if not rag_dir.is_absolute():
        rag_dir = PROJECT_ROOT / rag_dir

    cache_dir = rag_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_file = cache_dir / (
        f"cluster_sample_c{n_clusters}_s{samples_per_cluster}_neg{int(negative_only)}.csv"
    )

    source_files = [
        rag_dir / "data.csv",
        rag_dir / "embeddings.npy",
        rag_dir / "config.json",
    ]

    if not force_recompute and _cache_is_fresh(cache_file, source_files):
        print(f"📦 Lade Cluster-Cache: {cache_file}", flush=True)
        return pd.read_csv(cache_file, encoding="utf-8-sig")

    print("🧠 Baue Cluster-Sampling neu auf...", flush=True)

    _, df, _, embeddings = load_rag_assets(str(rag_dir))

    work_df = df.copy()
    work_df["source_index"] = work_df.index

    if negative_only and "score" in work_df.columns:
        work_df["score"] = pd.to_numeric(work_df["score"], errors="coerce")
        work_df = work_df[work_df["score"] <= 2].copy()

    work_df = work_df.reset_index(drop=True)

    if work_df.empty:
        raise ValueError("Keine Reviews für das Clustering gefunden.")

    source_indices = work_df["source_index"].to_numpy()
    work_embeddings = embeddings[source_indices]

    actual_clusters = min(n_clusters, len(work_df))
    if actual_clusters < 1:
        raise ValueError("Zu wenige Datenpunkte für Clustering.")

    kmeans = KMeans(
        n_clusters=actual_clusters,
        random_state=42,
        n_init=10,
    )
    cluster_labels = kmeans.fit_predict(work_embeddings)

    work_df["cluster_id"] = cluster_labels

    sampled_parts = []

    for cluster_id in range(actual_clusters):
        cluster_mask = work_df["cluster_id"] == cluster_id
        cluster_df = work_df.loc[cluster_mask].copy()

        if cluster_df.empty:
            continue

        cluster_source_indices = cluster_df["source_index"].to_numpy()
        cluster_vectors = embeddings[cluster_source_indices]
        centroid = kmeans.cluster_centers_[cluster_id]

        distances = np.linalg.norm(cluster_vectors - centroid, axis=1)
        cluster_df["cluster_distance"] = distances

        cluster_df = cluster_df.sort_values("cluster_distance", ascending=True)
        cluster_df = cluster_df.head(samples_per_cluster).copy()

        sampled_parts.append(cluster_df)

    if not sampled_parts:
        raise ValueError("Es konnten keine repräsentativen Reviews aus den Clustern gewählt werden.")

    sampled_df = pd.concat(sampled_parts, ignore_index=True)
    sampled_df = sampled_df.sort_values(["cluster_id", "cluster_distance"]).reset_index(drop=True)

    sampled_df.to_csv(cache_file, index=False, encoding="utf-8-sig")
    print(f"💾 Cluster-Cache gespeichert: {cache_file}", flush=True)

    return sampled_df


# =========================
# BUILD CONTEXT FROM CLUSTERS
# =========================
def build_cluster_context(
    sampled_df: pd.DataFrame,
    max_chars_per_review: int = 260
) -> str:
    parts = []

    for cluster_id in sorted(sampled_df["cluster_id"].unique()):
        cluster_df = sampled_df[sampled_df["cluster_id"] == cluster_id].copy()

        parts.append(f"Algorithmic Cluster {cluster_id + 1}:")

        for _, row in cluster_df.iterrows():
            text = _shorten_text(row.get("content", ""), max_chars=max_chars_per_review)
            score = row.get("score", "")
            parts.append(f"- Score: {score} | Review: {text}")

        parts.append("")

    return "\n".join(parts).strip()


# =========================
# GENERATION
# =========================
def _generate_mistral(generator, prompt: str, max_new_tokens: int = 220) -> str:
    print("📏 Prompt Länge:", len(prompt), flush=True)
    print("🚀 Generierung gestartet...", flush=True)

    response = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_full_text=False,
        pad_token_id=generator.tokenizer.eos_token_id,
    )[0]["generated_text"]

    print("✅ Generierung fertig", flush=True)
    return response.strip()


# =========================
# ASK RAG WITH CLUSTERING
# =========================
def ask_rag_clustered(
    rag_dir: str | Path,
    query: str,
    n_clusters: int = 5,
    samples_per_cluster: int = 2,
    negative_only: bool = True,
    force_recompute_cache: bool = False,
):
    print("\n🔍 Starte Cluster-basiertes Retrieval...", flush=True)

    sampled_df = cluster_sample_reviews(
        rag_dir=rag_dir,
        n_clusters=n_clusters,
        samples_per_cluster=samples_per_cluster,
        negative_only=negative_only,
        force_recompute=force_recompute_cache,
    )

    if sampled_df.empty:
        return "No matching reviews found."

    print(f"Gefundene repräsentative Reviews: {len(sampled_df)}", flush=True)

    context = build_cluster_context(sampled_df, max_chars_per_review=240)

    print("🤖 Lade Mistral...", flush=True)
    generator = load_generator()
    print("✅ Modell geladen", flush=True)

    prompt_1 = f"""[INST]
You are a product analyst for a mobile app.

Below you see representative negative user reviews sampled from 5 algorithmic clusters.
Different algorithmic clusters may belong to the same business problem category.

Use ONLY the reviews below.

Task:
1. Summarize the main problems in max 5 bullet points.
2. Consolidate the complaints into 3 to 5 business categories.
3. Give each category a short label.
4. For each category, briefly describe the typical complaints.

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

### Category 4: <label>
- ...
- ...

### Category 5: <label>
- ...
- ...

Rules:
- Use only the reviews below.
- Do not invent information.
- Do not mention positive feedback.
- Keep it concise.
- Return between 3 and 5 categories.
- You may merge algorithmic clusters into fewer business categories if appropriate.

Reviews:
{context}

Question:
{query}
[/INST]"""

    part_1 = _generate_mistral(generator, prompt_1, max_new_tokens=260)

    prompt_2 = f"""[INST]
You are a product analyst for a mobile app.

Use ONLY the analysis below.

Task:
1. For each category, suggest 2 to 3 concrete improvements / solutions.
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

### Category 4: <label>
- ...
- ...

### Category 5: <label>
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
- If the analysis contains only 3 or 4 categories, only return solutions for those categories.

Analysis:
{part_1}
[/INST]"""

    part_2 = _generate_mistral(generator, prompt_2, max_new_tokens=280)

    final_result = f"""## Problems and Categories

{part_1}

## Solutions and Feature Ideas

{part_2}"""

    return final_result, sampled_df


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    rag_dir = DATA_DIR / "My_BMW_en_raw_clean_rag"

    print("\n=== CLUSTER SAMPLE ===\n")
    print("Verwendeter rag_dir:", rag_dir)
    print("config exists:", (rag_dir / "config.json").exists())

    sampled_reviews = cluster_sample_reviews(
        rag_dir=rag_dir,
        n_clusters=5,
        samples_per_cluster=2,
        negative_only=True,
        force_recompute=False,
    )

    print(sampled_reviews[["cluster_id", "content", "score"]])

    print("\n=== AI ANALYSIS ===\n")

    answer, sampled_reviews = ask_rag_clustered(
        rag_dir=rag_dir,
        query="Analyze complaints and suggest improvements",
        n_clusters=5,
        samples_per_cluster=2,
        negative_only=True,
        force_recompute_cache=False,
    )

    print("\n=== RESULT ===\n")
    print(answer)