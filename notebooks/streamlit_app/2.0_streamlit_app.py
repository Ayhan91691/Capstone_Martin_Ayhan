# ==============================
# IMPORTS
# ==============================
import streamlit as st
import pandas as pd
from google_play_scraper import reviews, app, search

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="AI Feedback Analyzer", layout="wide")

# ==============================
# FUNCTIONS
# ==============================

# 🔹 Reviews laden
@st.cache_data
def fetch_reviews(app_id):
    result, _ = reviews(
        app_id,
        lang="en",
        country="us",
        count=200
    )

    if not result:
        return pd.DataFrame()

    df = pd.DataFrame(result)

    if "content" not in df.columns:
        return pd.DataFrame()

    df["text"] = df["content"]
    return df[["text"]]


# 🔹 Apps suchen (Google Play)
@st.cache_data
def search_apps(query):
    results = search(
        query,
        lang="en",
        country="us",
        n_hits=20
    )

    apps = []
    for r in results:
        apps.append((r["title"], r["appId"]))

    return apps


# 🔹 Issue Detection
def detect_issues(text):
    text = text.lower()

    issues = []

    if any(w in text for w in ["slow", "lag", "crash", "freeze"]):
        issues.append("Performance")

    if any(w in text for w in ["login", "password", "account"]):
        issues.append("Account")

    if any(w in text for w in ["price", "expensive", "cost"]):
        issues.append("Pricing")

    if any(w in text for w in ["ads", "advertisement"]):
        issues.append("Ads")

    if any(w in text for w in ["ui", "design", "interface"]):
        issues.append("UX/UI")

    if not issues:
        issues.append("Other")

    return issues


# 🔹 Kategorien → Suchbegriffe
CATEGORY_QUERIES = {
    "Automotive": "car app driving vehicle",
    "Gaming": "mobile games",
    "Food": "food delivery restaurant",
    "Finance": "banking finance money",
}

# ==============================
# SIDEBAR
# ==============================
st.sidebar.header("⚙️ Einstellungen")

language = st.sidebar.selectbox("Sprache", ["English", "German"])

category_filter = st.sidebar.selectbox(
    "Kategorie",
    ["All", "Automotive", "Gaming", "Food", "Finance"]
)

# ==============================
# DYNAMISCHE APP SUCHE
# ==============================
app_id = None

if category_filter != "All":

    query = CATEGORY_QUERIES.get(category_filter, category_filter)

    with st.spinner("🔍 Suche Apps..."):
        apps = search_apps(query)

    if len(apps) > 0:
        app_names = [name for name, _ in apps]

        selected_app = st.sidebar.selectbox(
            "📱 App auswählen",
            app_names
        )

        app_id = dict(apps)[selected_app]

    else:
        st.sidebar.warning("Keine Apps gefunden")

else:
    app_id = st.sidebar.text_input("Google Play App ID")

# ==============================
# BUTTON
# ==============================
if st.sidebar.button("📥 Reviews herunterladen"):

    if not app_id:
        st.error("Bitte App auswählen!")
    else:
        df = fetch_reviews(app_id)

        if df.empty:
            st.error("Keine Reviews gefunden")
        else:
            st.session_state.df = df
            st.success("✅ Reviews geladen!")


# ==============================
# MAIN UI
# ==============================
st.title("🤖 AI Customer Feedback Analyzer")

if "df" not in st.session_state:
    st.warning("Bitte lade zuerst Reviews.")
    st.stop()

df = st.session_state.df

st.write(f"📊 Anzahl Reviews: {len(df)}")

# ==============================
# ISSUE ANALYSIS
# ==============================
df["issues"] = df["text"].apply(detect_issues)

all_issues = [issue for sublist in df["issues"] for issue in sublist]

issue_counts = pd.Series(all_issues).value_counts()

st.subheader("🔥 Top Issues")

for issue, count in issue_counts.items():
    if st.button(f"{issue} ({count})"):
        filtered = df[df["issues"].apply(lambda x: issue in x)]
        st.dataframe(filtered)




# ==============================
# Dashboard
# ==============================
import matplotlib.pyplot as plt

# ==============================
# SENTIMENT (EINFACH)
# ==============================
def detect_sentiment(text):
    text = text.lower()

    positive_words = ["good", "great", "love", "excellent", "amazing", "perfect"]
    negative_words = ["bad", "slow", "crash", "hate", "bug", "problem", "error"]

    pos = sum(word in text for word in positive_words)
    neg = sum(word in text for word in negative_words)

    if pos > neg:
        return "Positive"
    elif neg > pos:
        return "Negative"
    else:
        return "Neutral"


df["sentiment"] = df["text"].apply(detect_sentiment)

# ==============================
# DASHBOARD LAYOUT
# ==============================
st.markdown("---")
st.header("📊 Dashboard")

col1, col2 = st.columns(2)

# ==============================
# SENTIMENT PIE CHART
# ==============================
with col1:
    st.subheader("🧠 Sentiment")

    sentiment_counts = df["sentiment"].value_counts()

    fig1, ax1 = plt.subplots()
    ax1.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%')
    ax1.set_title("Sentiment Distribution")

    st.pyplot(fig1)


# ==============================
# ISSUE PIE CHART
# ==============================
with col2:
    st.subheader("🔥 Issue Distribution")

    fig2, ax2 = plt.subplots()
    ax2.pie(issue_counts, labels=issue_counts.index, autopct='%1.1f%%')
    ax2.set_title("Issue Breakdown")

    st.pyplot(fig2)


# ==============================
# BAR CHART (TOP ISSUES)
# ==============================
st.subheader("📈 Top Issues")

fig3, ax3 = plt.subplots()
issue_counts.plot(kind="bar", ax=ax3)
ax3.set_title("Top Issues")
ax3.set_xlabel("Issue")
ax3.set_ylabel("Count")

st.pyplot(fig3)


# ==============================
# EXECUTIVE SUMMARY
# ==============================
st.subheader("🧠 Executive Summary")

top_issue = issue_counts.idxmax()
top_sentiment = df["sentiment"].value_counts().idxmax()

st.markdown(f"""
**📌 Key Insights:**

- Most common issue: **{top_issue}**
- Overall sentiment: **{top_sentiment}**
- Total reviews analyzed: **{len(df)}**

👉 Recommendation:
Focus on improving **{top_issue}** to significantly increase user satisfaction.
""")



























import streamlit as st
import pandas as pd
from google_play_scraper import reviews
import plotly.express as px

st.set_page_config(layout="wide")

# ===== HEADER =====
st.markdown("## 🤖 AI Customer Feedback Dashboard")
st.markdown("---")

# =========================
# 🚗 AUTOMOTIVE APPS + LOGOS
# =========================
AUTOMOTIVE_APPS = {
    "BMW": {
        "id": "de.bmw.connected",
        "logo": "https://upload.wikimedia.org/wikipedia/commons/4/44/BMW.svg"
    },
    "Mercedes": {
        "id": "com.daimler.ris.mercedesme",
        "logo": "https://upload.wikimedia.org/wikipedia/commons/9/90/Mercedes-Logo.svg"
    },
    "Tesla": {
        "id": "com.teslamotors.tesla",
        "logo": "https://upload.wikimedia.org/wikipedia/commons/b/bd/Tesla_Motors.svg"
    },
    "Audi": {
        "id": "de.myaudi.mobile.assistant",
        "logo": "https://upload.wikimedia.org/wikipedia/commons/6/6f/Audi_logo.svg"
    },
    "Toyota": {
        "id": "com.toyota.oneapp.eu",
        "logo": "https://upload.wikimedia.org/wikipedia/commons/9/9d/Toyota_logo.svg"
    },
    "Volkswagen": {
        "id": "de.volkswagen.vwconnect",
        "logo": "https://upload.wikimedia.org/wikipedia/commons/6/6d/Volkswagen_logo.svg"
    }
}

# =========================
# 🔎 SIDEBAR
# =========================
st.sidebar.header("🔎 Search Data")

selected_apps = st.sidebar.multiselect(
    "Select Automotive Apps",
    list(AUTOMOTIVE_APPS.keys()),
    default=["BMW"]
)

# =========================
# 📥 FETCH REVIEWS
# =========================
def fetch_reviews(app_id):
    result, _ = reviews(app_id, count=100)

    df = pd.DataFrame(result)

    if "content" not in df.columns:
        return pd.DataFrame()

    df["text"] = df["content"]
    df["app"] = app_id
    return df

# =========================
# 🧠 SENTIMENT
# =========================
def simple_sentiment(text):
    text = text.lower()

    if any(w in text for w in ["good", "great", "love"]):
        return "Positive"
    elif any(w in text for w in ["bad", "hate", "worst", "crash"]):
        return "Negative"
    else:
        return "Neutral"

# =========================
# 🔥 ISSUE DETECTION
# =========================
def detect_issues(text):
    text = text.lower()

    if "crash" in text:
        return "Crash"
    elif "login" in text:
        return "Login"
    elif "slow" in text:
        return "Performance"
    else:
        return "Other"

# =========================
# 🚀 LOAD DATA BUTTON
# =========================
if st.sidebar.button("🚀 Load Data"):

    all_data = []

    for app in selected_apps:
        app_id = AUTOMOTIVE_APPS[app]["id"]

        df = fetch_reviews(app_id)

        if not df.empty:
            df["app"] = app
            df["sentiment"] = df["text"].apply(simple_sentiment)
            df["issue"] = df["text"].apply(detect_issues)

            # Fake geo data (für Map)
            df["lat"] = 50 + (hash(app) % 20)
            df["lon"] = 10 + (hash(app) % 20)

            all_data.append(df)

    if all_data:
        st.session_state.df = pd.concat(all_data)
        st.success("✅ Data loaded!")

# =========================
# 📊 MAIN
# =========================
st.title("🤖 AI Customer Feedback Analyzer")

if "df" not in st.session_state:
    st.info("👉 Select apps and load data")
    st.stop()

df = st.session_state.df

# =========================
# 🖼️ LOGOS
# =========================
st.subheader("Selected Brands")

cols = st.columns(len(selected_apps))

for i, app in enumerate(selected_apps):
    with cols[i]:
        st.image(AUTOMOTIVE_APPS[app]["logo"], width=80)
        st.caption(app)

# ===== KPIs =====
col1, col2, col3, col4 = st.columns(4)

total = len(df)
positive = (df["sentiment"] == "Positive").sum()
negative = (df["sentiment"] == "Negative").sum()
neutral = (df["sentiment"] == "Neutral").sum()

col1.metric("📊 Total Reviews", total)
col2.metric("😊 Positive", positive)
col3.metric("😐 Neutral", neutral)
col4.metric("😡 Negative", negative)

st.markdown("---")

# ===== MAP + INSIGHT =====
col_map, col_info = st.columns([2, 1])

with col_map:
    st.markdown("### 🌍 Global App Distribution")

    fig_map = px.scatter_mapbox(
        df,
        lat="lat",
        lon="lon",
        hover_name="app",
        zoom=1,
        height=350
    )
    fig_map.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig_map, use_container_width=True)

with col_info:
    st.markdown("### 🧠 Key Insights")

    top_issue = df["issue"].explode().value_counts().idxmax()
    st.info(f"🔥 Top Issue: {top_issue}")

    pos_rate = round(positive / total * 100, 1)
    st.success(f"😊 Positive Rate: {pos_rate}%")

    st.warning("⚠️ Focus Area: Improve UX / Performance")

st.markdown("---")

# ===== CHARTS =====
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 💬 Sentiment Distribution")

    sentiment_counts = df["sentiment"].value_counts().reset_index()
    fig1 = px.pie(sentiment_counts, names="sentiment", values="count")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.markdown("### 🔥 Issue Breakdown")

    issue_counts = df["issue"].explode().value_counts().reset_index()
    fig2 = px.bar(issue_counts, x="issue", y="count")
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ===== EXECUTIVE SUMMARY =====
st.markdown("### 🧾 Executive Summary")

st.markdown(f"""
- 📊 Total Reviews: **{total}**
- 🔥 Top Issue: **{top_issue}**
- 😊 Positive Rate: **{pos_rate}%**
- 🚀 Recommendation: Improve **{top_issue}**
""")