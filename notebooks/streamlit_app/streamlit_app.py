import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(layout="wide")

# ---------------------------
# CSS (CARD DESIGN)
# ---------------------------
st.markdown("""
<style>
[data-testid="metric-container"] {
    background-color: #1e1e1e;
    padding: 16px;
    border-radius: 12px;
    border: 1px solid #333;
}

.block-container {
    padding-top: 2.5rem;  /* mehr Abstand nach oben */
}
h3 {
    margin-top: 0 !important;
    padding-top: 0 !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# SIDEBAR (PRO FILTER)
# ---------------------------
with st.sidebar:
    st.title("🔍 Filters")

    timeframe = st.radio("Timeframe", ["7d", "30d", "90d"], horizontal=True)

    st.markdown("### Sentiment")
    col1, col2, col3 = st.columns(3)
    col1.button("🟢 Positive")
    col2.button("🟡 Neutral")
    col3.button("🔴 Negative")

    st.markdown("### Issues")
    st.multiselect("", ["Delivery", "Pricing", "UX", "Support"])

    st.text_input("Search reviews...")

    st.divider()
    st.button("Reset")

# ---------------------------
# HEADER
# ---------------------------
col1, col2 = st.columns([6,2])

with col1:
    c1, c2 = st.columns([1,6])

    with c1:
        st.image("logo.jpeg", width=50)

    with c2:
        st.markdown("### AI Insight Dashboard")

with col2:
    st.markdown("""
    <div style="text-align: right;">
        📍 Munich <br>
        🕒 14:32 <br>
        🌤 18°C
    </div>
    """, unsafe_allow_html=True)


# ---------------------------
# KPI CARDS
# ---------------------------
k1, k2, k3, k4, k5 = st.columns(5)

k1.metric("Reviews", "12.4K")
k2.metric("Positive", "62%")
k3.metric("Negative", "23%", "+5%")
k4.metric("Rating", "4.1 ⭐")
k5.metric("Top Issue", "Delivery")

# ---------------------------
# DATA
# ---------------------------
df = pd.DataFrame({
    "day": list(range(20)),
    "Positive": np.random.randint(20, 50, 20),
    "Neutral": np.random.randint(10, 30, 20),
    "Negative": np.random.randint(5, 20, 20)
})

issues = pd.DataFrame({
    "Issue": ["Delivery", "Pricing", "UX", "Support"],
    "Count": [120, 80, 60, 40]
}).sort_values(by="Count")

strengths = pd.DataFrame({
    "Strength": ["Design", "Speed", "Quality", "Usability"],
    "Count": [150, 130, 100, 90]
}).sort_values(by="Count")

# ---------------------------
# HERO: SENTIMENT TREND
# ---------------------------
st.subheader("Sentiment Trend")

fig = px.line(df, x="day", y=["Positive","Neutral","Negative"])
fig.update_layout(height=350, legend=dict(orientation="h"))
st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# ISSUES + STRENGTHS
# ---------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("🚨 Top Issues")

    fig_issues = px.bar(
        issues,
        x="Count",
        y="Issue",
        orientation="h",
        text="Count"
    )
    fig_issues.update_traces(marker_color="#ff4d4d", textposition="outside")
    fig_issues.update_layout(height=300)
    st.plotly_chart(fig_issues, use_container_width=True)

with col2:
    st.subheader("💚 Top Strengths")

    fig_strengths = px.bar(
        strengths,
        x="Count",
        y="Strength",
        orientation="h",
        text="Count"
    )
    fig_strengths.update_traces(marker_color="#00cc88", textposition="outside")
    fig_strengths.update_layout(height=300)
    st.plotly_chart(fig_strengths, use_container_width=True)

# ---------------------------
# AI INSIGHT (WICHTIG!)
# ---------------------------
st.subheader("🤖 AI Insight")

st.info("""
Negative sentiment is increasing due to delivery delays.

**Key Insight:** Delivery is the main issue impacting user satisfaction.  
**Recommendation:** Improve logistics tracking and communication.
""")

# ---------------------------
# REVIEWS + MESSAGES
# ---------------------------
col3, col4 = st.columns(2)

with col3:
    st.subheader("💬 Example Reviews")

    st.write("❌ 'Delivery took 10 days...'")
    st.write("❌ 'App crashes on checkout...'")
    st.write("✅ 'Great design and usability'")

with col4:
    st.subheader("📬 Messages")

    st.write("📩 Delivery issue (Unread)")
    st.write("📩 UI feedback (Read)")












    import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(layout="wide")

# ---------------------------
# CSS (CARD DESIGN)
# ---------------------------
st.markdown("""
<style>
[data-testid="metric-container"] {
    background-color: #1e1e1e;
    padding: 16px;
    border-radius: 12px;
    border: 1px solid #333;
}

.block-container {
    padding-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# SIDEBAR (PRO FILTER)
# ---------------------------
with st.sidebar:
    st.title("🔍 Filters")

    timeframe = st.radio("Timeframe", ["7d", "30d", "90d"], horizontal=True)

    st.markdown("### Sentiment")
    col1, col2, col3 = st.columns(3)
    col1.button("🟢 Positive")
    col2.button("🟡 Neutral")
    col3.button("🔴 Negative")

    st.markdown("### Issues")
    st.multiselect("", ["Delivery", "Pricing", "UX", "Support"])

    st.text_input("Search reviews...")

    st.divider()
    st.button("Reset")

# ---------------------------
# HEADER
# ---------------------------
col1, col2 = st.columns([6,2])

with col1:
    c1, c2 = st.columns([1,6])

    with c1:
        st.image("logo.jpeg", width=50)

    with c2:
        st.markdown("### AI Insight Dashboard")

with col2:
    st.markdown("""
    <div style="text-align: right;">
        📍 Munich <br>
        🕒 14:32 <br>
        🌤 18°C
    </div>
    """, unsafe_allow_html=True)


# ---------------------------
# KPI CARDS
# ---------------------------
k1, k2, k3, k4, k5 = st.columns(5)

k1.metric("Reviews", "12.4K")
k2.metric("Positive", "62%")
k3.metric("Negative", "23%", "+5%")
k4.metric("Rating", "4.1 ⭐")
k5.metric("Top Issue", "Delivery")

# ---------------------------
# DATA
# ---------------------------
df = pd.DataFrame({
    "day": list(range(20)),
    "Positive": np.random.randint(20, 50, 20),
    "Neutral": np.random.randint(10, 30, 20),
    "Negative": np.random.randint(5, 20, 20)
})

issues = pd.DataFrame({
    "Issue": ["Delivery", "Pricing", "UX", "Support"],
    "Count": [120, 80, 60, 40]
}).sort_values(by="Count")

strengths = pd.DataFrame({
    "Strength": ["Design", "Speed", "Quality", "Usability"],
    "Count": [150, 130, 100, 90]
}).sort_values(by="Count")

# ---------------------------
# HERO: SENTIMENT TREND
# ---------------------------
st.subheader("Sentiment Trend")

fig = px.line(df, x="day", y=["Positive","Neutral","Negative"])
fig.update_layout(height=350, legend=dict(orientation="h"))
st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# ISSUES + STRENGTHS
# ---------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("🚨 Top Issues")

    fig_issues = px.bar(
        issues,
        x="Count",
        y="Issue",
        orientation="h",
        text="Count"
    )
    fig_issues.update_traces(marker_color="#ff4d4d", textposition="outside")
    fig_issues.update_layout(height=300)
    st.plotly_chart(fig_issues, use_container_width=True)

with col2:
    st.subheader("💚 Top Strengths")

    fig_strengths = px.bar(
        strengths,
        x="Count",
        y="Strength",
        orientation="h",
        text="Count"
    )
    fig_strengths.update_traces(marker_color="#00cc88", textposition="outside")
    fig_strengths.update_layout(height=300)
    st.plotly_chart(fig_strengths, use_container_width=True)

# ---------------------------
# AI INSIGHT (WICHTIG!)
# ---------------------------
st.subheader("🤖 AI Insight")

st.info("""
Negative sentiment is increasing due to delivery delays.

**Key Insight:** Delivery is the main issue impacting user satisfaction.  
**Recommendation:** Improve logistics tracking and communication.
""")

# ---------------------------
# REVIEWS + MESSAGES
# ---------------------------
col3, col4 = st.columns(2)

with col3:
    st.subheader("💬 Example Reviews")

    st.write("❌ 'Delivery took 10 days...'")
    st.write("❌ 'App crashes on checkout...'")
    st.write("✅ 'Great design and usability'")

with col4:
    st.subheader("📬 Messages")

    st.write("📩 Delivery issue (Unread)")
    st.write("📩 UI feedback (Read)")