import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Global Pizza Intelligence 2025", layout="wide")

# --------------------------------------------------
# COLOR THEME
# --------------------------------------------------
PRIMARY = "#D93662"
YELLOW = "#F2E749"
ORANGE = "#F29829"
DEEP_ORANGE = "#F25922"
RED = "#D91A1A"
DARK_BG = "#0E1117"

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown(f"""
<style>
body {{
    background-color: {DARK_BG};
}}
.kpi-card {{
    background: linear-gradient(135deg, #111111, #1a1a1a);
    padding: 25px;
    border-radius: 18px;
    border-left: 6px solid {YELLOW};
    box-shadow: 0px 4px 20px rgba(242,231,73,0.2);
    text-align: center;
}}
.kpi-title {{
    color: #BBBBBB;
    font-size: 14px;
}}
.kpi-value {{
    color: {YELLOW};
    font-size: 34px;
    font-weight: bold;
}}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# TITLE
# --------------------------------------------------
st.title("🍕 Global Pizza Consumption Intelligence 2025")

uploaded_file = st.file_uploader("Upload Pizza Dataset (CSV)", type=["csv"])

if uploaded_file is None:
    st.info("Upload your CSV to begin analysis.")
    st.stop()

df = pd.read_csv(uploaded_file)

# --------------------------------------------------
# AUTO DETECT COLUMNS
# --------------------------------------------------
country_col = [c for c in df.columns if "country" in c.lower()][0]
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

# Sidebar
st.sidebar.header("📊 Analysis Controls")

metric_col = st.sidebar.selectbox("Select Consumption Metric", numeric_cols)

view_mode = st.sidebar.radio(
    "Market View",
    ["Total Market Size", "Per Capita Performance"]
)

animate_toggle = st.sidebar.toggle("Enable Animated View")

# --------------------------------------------------
# CLEAN NUMERIC
# --------------------------------------------------
df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")
df = df.dropna(subset=[metric_col])

# --------------------------------------------------
# KPIs
# --------------------------------------------------
total_countries = df[country_col].nunique()
total_consumption = df[metric_col].sum()
avg_consumption = df[metric_col].mean()
top_country = df.loc[df[metric_col].idxmax(), country_col]

st.subheader("📈 Executive Snapshot")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Countries Covered</div>
        <div class="kpi-value">{total_countries}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Total Consumption</div>
        <div class="kpi-value">{round(total_consumption,2)}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Average Per Country</div>
        <div class="kpi-value">{round(avg_consumption,2)}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Top Market</div>
        <div class="kpi-value">{top_country}</div>
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# TOP 10 BAR CHART
# --------------------------------------------------
st.subheader("🏆 Top 10 Markets")

top10 = df.sort_values(metric_col, ascending=False).head(10)

fig_bar = px.bar(
    top10,
    x=country_col,
    y=metric_col,
    color=metric_col,
    color_continuous_scale=[PRIMARY, YELLOW, ORANGE, RED],
)

fig_bar.update_layout(
    xaxis_title="Country",
    yaxis_title=metric_col,
    plot_bgcolor=DARK_BG,
    paper_bgcolor=DARK_BG,
)

st.plotly_chart(fig_bar, use_container_width=True)

# --------------------------------------------------
# AREA CHART
# --------------------------------------------------
st.subheader("📊 Market Share Distribution (Top 5)")

top5 = df.sort_values(metric_col, ascending=False).head(5)

fig_area = px.area(
    top5,
    x=country_col,
    y=metric_col,
    color=country_col,
    color_discrete_sequence=[PRIMARY, YELLOW, ORANGE, DEEP_ORANGE, RED],
)

fig_area.update_layout(
    xaxis_title="Country",
    yaxis_title=metric_col,
    plot_bgcolor=DARK_BG,
    paper_bgcolor=DARK_BG,
)

st.plotly_chart(fig_area, use_container_width=True)

# --------------------------------------------------
# SCATTER CHART
# --------------------------------------------------
if len(numeric_cols) >= 2:
    st.subheader("🔎 Relationship Analysis")

    fig_scatter = px.scatter(
        df,
        x=numeric_cols[0],
        y=numeric_cols[1],
        color=metric_col,
        hover_name=country_col,
        color_continuous_scale=[PRIMARY, YELLOW, RED],
    )

    fig_scatter.update_layout(
        xaxis_title=numeric_cols[0],
        yaxis_title=numeric_cols[1],
        plot_bgcolor=DARK_BG,
        paper_bgcolor=DARK_BG,
    )

    st.plotly_chart(fig_scatter, use_container_width=True)

# --------------------------------------------------
# LINE CHART (Projection)
# --------------------------------------------------
st.subheader("📈 5-Year Market Projection")

df_sorted = df.sort_values(metric_col)

X = np.arange(len(df_sorted)).reshape(-1, 1)
y = df_sorted[metric_col].values

model = LinearRegression()
model.fit(X, y)

future_X = np.arange(len(df_sorted) + 5).reshape(-1, 1)
predictions = model.predict(future_X)

projection_df = pd.DataFrame({
    "Index": future_X.flatten(),
    "Projected Consumption": predictions
})

fig_line = px.line(
    projection_df,
    x="Index",
    y="Projected Consumption",
    color_discrete_sequence=[YELLOW]
)

fig_line.update_layout(
    xaxis_title="Time Index",
    yaxis_title="Projected Consumption",
    plot_bgcolor=DARK_BG,
    paper_bgcolor=DARK_BG,
)

st.plotly_chart(fig_line, use_container_width=True)

# --------------------------------------------------
# WORLD MAP
# --------------------------------------------------
st.subheader("🌍 Global Pizza Heat Map")

fig_map = px.choropleth(
    df,
    locations=country_col,
    locationmode="country names",
    color=metric_col,
    color_continuous_scale=[PRIMARY, YELLOW, ORANGE, RED],
)

fig_map.update_layout(
    plot_bgcolor=DARK_BG,
    paper_bgcolor=DARK_BG,
)

st.plotly_chart(fig_map, use_container_width=True)

# --------------------------------------------------
# AI SUMMARY
# --------------------------------------------------
st.subheader("🧠 AI Executive Intelligence Summary")

summary = f"""
The global pizza market spans {total_countries} countries with a total 
consumption of {round(total_consumption,2)} units.

The dominant market is {top_country}, significantly outperforming peers.

Average consumption per country stands at {round(avg_consumption,2)}.

Fun Fact 🍕:
The top 3 markets account for nearly 
{round((top10[metric_col].head(3).sum()/total_consumption)*100,2)}% 
of global pizza activity!
"""

st.markdown(summary)