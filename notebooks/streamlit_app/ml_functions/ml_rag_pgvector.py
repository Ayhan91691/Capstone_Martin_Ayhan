import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import pipeline


def load_settings():
    load_dotenv()

    settings = {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", "5432")),
        "dbname": os.getenv("POSTGRES_DB"),
        "user": os.getenv("POSTGRES_USER"),
        "password": os.getenv("POSTGRES_PASSWORD"),
        "embedding_model": os.getenv(
            "EMBEDDING_MODEL",
            "paraphrase-multilingual-MiniLM-L12-v2"
        ),
        "llm_model": os.getenv(
            "LLM_MODEL",
            "mistralai/Mistral-7B-Instruct-v0.2"
        ),
    }

    missing = [k for k in ["dbname", "user", "password"] if not settings[k]]
    if missing:
        raise ValueError(
            f"Fehlende ENV-Variablen: {', '.join(missing)}. "
            "Bitte in der .env-Datei setzen."
        )

    return settings


def get_connection():
    cfg = load_settings()
    return psycopg.connect(
        host=cfg["host"],
        port=cfg["port"],
        dbname=cfg["dbname"],
        user=cfg["user"],
        password=cfg["password"],
        autocommit=True
    )


def ensure_vector_extension(conn):
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")


def ensure_table(conn, table_name: str, vector_dim: int):
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id BIGSERIAL PRIMARY KEY,
                source_row_id INTEGER,
                content TEXT NOT NULL,
                score DOUBLE PRECISION,
                metadata JSONB,
                embedding vector({vector_dim})
            );
        """)

        # optional: Index für schnellere Ähnlichkeitssuche
        # Für Cosine Distance:
        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS {table_name}_embedding_cosine_idx
            ON {table_name}
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """)


def reset_table(conn, table_name: str):
    with conn.cursor() as cur:
        cur.execute(f"TRUNCATE TABLE {table_name} RESTART IDENTITY;")


def build_rag_postgres(
    file_path: str,
    table_name: str = "bmw_reviews_rag",
    text_col: str = "content",
    score_col: str = "score",
    model_name: str | None = None,
    reset: bool = False
):
    cfg = load_settings()
    model_name = model_name or cfg["embedding_model"]

    df = pd.read_csv(file_path, encoding="utf-8-sig").copy()

    if text_col not in df.columns:
        raise ValueError(f"Die Eingabedatei muss eine Spalte '{text_col}' enthalten.")

    df = df.dropna(subset=[text_col])
    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[df[text_col] != ""].reset_index(drop=True)

    embedder = SentenceTransformer(model_name)
    embeddings = embedder.encode(
        df[text_col].tolist(),
        convert_to_numpy=True,
        show_progress_bar=True
    ).astype("float32")

    vector_dim = embeddings.shape[1]

    conn = get_connection()
    ensure_vector_extension(conn)
    ensure_table(conn, table_name, vector_dim)

    if reset:
        reset_table(conn, table_name)

    rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        content = row[text_col]
        score = None
        if score_col in row.index:
            try:
                score = float(row[score_col]) if pd.notna(row[score_col]) else None
            except Exception:
                score = None

        metadata = row.to_dict()
        embedding = embeddings[i].tolist()

        rows.append((i, content, score, json.dumps(metadata, ensure_ascii=False), embedding))

    with conn.cursor() as cur:
        cur.executemany(
            f"""
            INSERT INTO {table_name} (
                source_row_id,
                content,
                score,
                metadata,
                embedding
            )
            VALUES (%s, %s, %s, %s::jsonb, %s);
            """,
            rows
        )

    conn.close()

    return {
        "table_name": table_name,
        "rows_inserted": len(rows),
        "vector_dim": vector_dim,
        "model_name": model_name
    }


def search_rag(
    query: str,
    table_name: str = "bmw_reviews_rag",
    k: int = 5,
    negative_only: bool = False,
    model_name: str | None = None
) -> pd.DataFrame:
    cfg = load_settings()
    model_name = model_name or cfg["embedding_model"]

    embedder = SentenceTransformer(model_name)
    query_embedding = embedder.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")[0].tolist()

    conn = get_connection()

    sql = f"""
        SELECT
            id,
            source_row_id,
            content,
            score,
            metadata,
            embedding <=> %s::vector AS distance
        FROM {table_name}
    """

    params = [query_embedding]

    if negative_only:
        sql += " WHERE score IS NOT NULL AND score <= 2"

    sql += " ORDER BY embedding <=> %s::vector LIMIT %s"
    params.append(query_embedding)
    params.append(k)

    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    conn.close()

    if not rows:
        return pd.DataFrame(columns=["id", "source_row_id", "content", "score", "metadata", "distance"])

    result = pd.DataFrame(
        rows,
        columns=["id", "source_row_id", "content", "score", "metadata", "distance"]
    )
    return result


def ask_rag(
    query: str,
    table_name: str = "bmw_reviews_rag",
    k: int = 5,
    negative_only: bool = True,
    model_name: str | None = None,
    llm_name: str | None = None
):
    cfg = load_settings()
    model_name = model_name or cfg["embedding_model"]
    llm_name = llm_name or cfg["llm_model"]

    docs = search_rag(
        query=query,
        table_name=table_name,
        k=k,
        negative_only=negative_only,
        model_name=model_name
    )

    if docs.empty:
        return "No matching reviews found."

    context_parts = []
    for _, row in docs.iterrows():
        text = row.get("content", "")
        score = row.get("score", "")
        distance = row.get("distance", "")
        context_parts.append(
            f"- Score: {score} | Distance: {distance:.4f} | Review: {text}"
        )

    context = "\n".join(context_parts)

    prompt = f"""[INST]
You are analyzing customer feedback for the BMW app.

Use only the reviews below.
Answer briefly and precisely.
List the main issues as short bullet points.
Do not invent information.
Do not mention positive feedback unless the question explicitly asks for it.

Reviews:
{context}

Question: {query}
[/INST]
"""

    generator = pipeline("text-generation", model=llm_name)

    response = generator(
        prompt,
        max_new_tokens=180,
        truncation=True,
        do_sample=False,
        pad_token_id=generator.tokenizer.eos_token_id
    )[0]["generated_text"]

    if "[/INST]" in response:
        response = response.split("[/INST]", 1)[-1].strip()

    return response


if __name__ == "__main__":
    FILE_PATH = "data/My_BMW_en_clean.csv"
    TABLE_NAME = "bmw_reviews_rag"

    # Nur beim ersten Mal oder nach Datenänderung:
    # info = build_rag_postgres(
    #     file_path=FILE_PATH,
    #     table_name=TABLE_NAME,
    #     text_col="content",
    #     score_col="score",
    #     reset=True
    # )
    # print(info)

    results = search_rag(
        query="What are the main issues with the BMW app?",
        table_name=TABLE_NAME,
        k=5,
        negative_only=True
    )
    print(results[["content", "score", "distance"]])

    answer = ask_rag(
        query="What are the main issues with the BMW app?",
        table_name=TABLE_NAME,
        k=5,
        negative_only=True
    )
    print("\nAnswer:\n")
    print(answer)

    # pip install pandas numpy sentence-transformers transformers torch psycopg python-dotenv pgvector