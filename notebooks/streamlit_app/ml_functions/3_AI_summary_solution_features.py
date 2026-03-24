import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Analysis Result",
    page_icon="📊",
    layout="wide",
)

# =========================
# STYLING
# =========================
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    h1, h2, h3, h4, h5, h6, p, div, span, label {
        color: #f8fafc !important;
    }

    .hero-box {
        padding: 1.8rem 2rem;
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(59,130,246,0.22), rgba(168,85,247,0.18));
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 12px 40px rgba(0,0,0,0.25);
        margin-bottom: 1.5rem;
    }

    .glass-card {
        padding: 1.2rem 1.3rem;
        border-radius: 22px;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 8px 24px rgba(0,0,0,0.18);
        backdrop-filter: blur(8px);
    }

    .kpi-card {
        padding: 1rem 1.2rem;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(30,41,59,0.9), rgba(17,24,39,0.95));
        border: 1px solid rgba(255,255,255,0.06);
        box-shadow: 0 10px 28px rgba(0,0,0,0.22);
        min-height: 120px;
    }

    .kpi-label {
        font-size: 0.9rem;
        color: #cbd5e1 !important;
        margin-bottom: 0.35rem;
    }

    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        line-height: 1.1;
    }

    .subtle {
        color: #cbd5e1 !important;
        font-size: 0.95rem;
    }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.06);
        padding: 1rem;
        border-radius: 18px;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 18px;
        overflow: hidden;
    }

    .section-title {
        margin-top: 0.4rem;
        margin-bottom: 0.8rem;
        font-size: 1.2rem;
        font-weight: 700;
    }

    .footer-note {
        color: #94a3b8 !important;
        font-size: 0.85rem;
        margin-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# HELPERS
# =========================
RESULTS_ROOT = Path("data")


def find_result_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []

    matches = []
    for p in root.rglob("analysis_outputs/*"):
        if p.is_dir():
            json_file = p / "analysis_result.json"
            md_file = p / "analysis_result.md"
            csv_file = p / "sampled_reviews.csv"
            if json_file.exists() and md_file.exists() and csv_file.exists():
                matches.append(p)

    return sorted(matches, reverse=True)


def load_result_bundle(result_dir: Path):
    json_path = result_dir / "analysis_result.json"
    md_path = result_dir / "analysis_result.md"
    csv_path = result_dir / "sampled_reviews.csv"

    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    markdown_text = md_path.read_text(encoding="utf-8")
    reviews_df = pd.read_csv(csv_path, encoding="utf-8-sig")

    return payload, markdown_text, reviews_df


def extract_preview_lists(markdown_text: str):
    lines = [line.strip() for line in markdown_text.splitlines()]

    problems = []
    general_ideas = []
    tech_ideas = []

    current_section = None

    for line in lines:
        if not line:
            continue

        lower = line.lower()

        if "## problems and categories" in lower:
            current_section = "top"
            continue
        if "## general app ideas" in lower:
            current_section = "general"
            continue
        if "## technical / programming suggestions" in lower:
            current_section = "tech"
            continue

        if line.startswith("## "):
            current_section = None

        if current_section == "top" and line.startswith("- ") and len(problems) < 5:
            problems.append(line[2:].strip())

        if current_section == "general" and line.startswith("- ") and len(general_ideas) < 5:
            general_ideas.append(line[2:].strip())

        if current_section == "tech" and line.startswith("- ") and len(tech_ideas) < 5:
            tech_ideas.append(line[2:].strip())

    return problems, general_ideas, tech_ideas


def safe_option_label(path: Path) -> str:
    parent_name = path.parent.name
    return f"{parent_name} / {path.name}"


# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("Analysis Explorer")

    result_dirs = find_result_dirs(RESULTS_ROOT)

    if not result_dirs:
        st.error("No analysis result folders found.")
        st.stop()

    selected_dir = st.selectbox(
        "Select analysis run",
        options=result_dirs,
        format_func=safe_option_label,
    )

    st.caption(f"Folder: {selected_dir}")

    show_json = st.toggle("Show raw JSON", value=False)
    show_md = st.toggle("Show raw Markdown", value=False)
    show_csv = st.toggle("Show sampled reviews table", value=True)


# =========================
# LOAD DATA
# =========================
payload, markdown_text, reviews_df = load_result_bundle(selected_dir)

created_at = payload.get("created_at", "-")
query = payload.get("query", "-")
n_clusters = payload.get("n_clusters", "-")
samples_per_cluster = payload.get("samples_per_cluster", "-")
negative_only = payload.get("negative_only", "-")
rag_dir = payload.get("rag_dir", "-")

problems_preview, general_preview, tech_preview = extract_preview_lists(markdown_text)

cluster_count = int(reviews_df["cluster_id"].nunique()) if "cluster_id" in reviews_df.columns else 0
review_count = len(reviews_df)


# =========================
# HERO
# =========================
st.markdown(
    """
    <div class="hero-box">
        <div style="font-size: 0.95rem; letter-spacing: 0.08em; text-transform: uppercase; color: #cbd5e1;">
            Product Intelligence
        </div>
        <div style="font-size: 2.6rem; font-weight: 800; margin-top: 0.35rem;">
            Analysis Result
        </div>
        <div class="subtle" style="margin-top: 0.5rem; max-width: 1000px;">
            A polished insight board for clustered user feedback, category-level solutions,
            feature opportunities, and technical improvement ideas.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================
# TOP KPIS
# =========================
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Analysis Run</div>
            <div class="kpi-value">{created_at}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k2:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Clusters Used</div>
            <div class="kpi-value">{cluster_count}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k3:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Sampled Reviews</div>
            <div class="kpi-value">{review_count}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k4:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Per Cluster</div>
            <div class="kpi-value">{samples_per_cluster}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================
# META + QUERY
# =========================
left, right = st.columns([2, 1])

with left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Analysis Query</div>', unsafe_allow_html=True)
    st.write(query)
    st.markdown(
        f"""
        <div class="footer-note">
            Source RAG directory: {rag_dir}<br>
            Negative only: {negative_only}<br>
            Requested clusters: {n_clusters}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Quick Snapshot</div>', unsafe_allow_html=True)
    st.metric("Problem bullets found", len(problems_preview))
    st.metric("General app ideas", len(general_preview))
    st.metric("Technical suggestions", len(tech_preview))
    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# HIGHLIGHTS
# =========================
h1, h2, h3 = st.columns(3)

with h1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### Top Problems")
    if problems_preview:
        for item in problems_preview:
            st.markdown(f"- {item}")
    else:
        st.caption("No preview extracted.")
    st.markdown("</div>", unsafe_allow_html=True)

with h2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### General App Ideas")
    if general_preview:
        for item in general_preview:
            st.markdown(f"- {item}")
    else:
        st.caption("No preview extracted.")
    st.markdown("</div>", unsafe_allow_html=True)

with h3:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### Technical Suggestions")
    if tech_preview:
        for item in tech_preview:
            st.markdown(f"- {item}")
    else:
        st.caption("No preview extracted.")
    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# MAIN TABS
# =========================
tab1, tab2, tab3, tab4 = st.tabs(
    ["✨ Executive View", "🧠 Markdown Report", "🗂 Structured Data", "🧾 Sampled Reviews"]
)

with tab1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("## Executive View")
    st.markdown(
        """
        This section is designed for quick stakeholder review.
        Use it when you want to understand the core story of the feedback without diving into raw files.
        """
    )
    st.divider()
    st.markdown(markdown_text)
    st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("## Markdown Report")
    st.code(markdown_text, language="markdown")
    if show_md:
        st.download_button(
            "Download Markdown",
            data=markdown_text,
            file_name="analysis_result.md",
            mime="text/markdown",
            use_container_width=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("## Structured JSON Data")
    st.json(payload, expanded=False)
    if show_json:
        st.download_button(
            "Download JSON",
            data=json.dumps(payload, ensure_ascii=False, indent=2),
            file_name="analysis_result.json",
            mime="application/json",
            use_container_width=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

with tab4:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("## Sampled Reviews")
    st.caption(
        "These are the representative reviews selected from the algorithmic clusters and used for the analysis."
    )

    df_view = reviews_df.copy()

    preferred_cols = [
        "cluster_id",
        "score",
        "content",
        "cluster_distance",
        "source_index",
    ]
    available_cols = [c for c in preferred_cols if c in df_view.columns]
    if available_cols:
        df_view = df_view[available_cols + [c for c in df_view.columns if c not in available_cols]]

    if "cluster_id" in df_view.columns:
        selected_clusters = st.multiselect(
            "Filter by cluster",
            options=sorted(df_view["cluster_id"].dropna().unique().tolist()),
            default=sorted(df_view["cluster_id"].dropna().unique().tolist()),
        )
        df_view = df_view[df_view["cluster_id"].isin(selected_clusters)]

    if show_csv:
        st.dataframe(df_view, use_container_width=True, height=500)

    csv_bytes = reviews_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        "Download sampled reviews CSV",
        data=csv_bytes,
        file_name="sampled_reviews.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)