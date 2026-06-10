import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="European Bank Churn Dashboard",
    page_icon="🏦",
    layout="wide"
)

# =====================================
# LOAD DATA
# =====================================
df = pd.read_csv("European_Bank_Cleaned.csv")

model = joblib.load("churn_model.pkl")
features = joblib.load("feature_columns.pkl")

# =====================================
# TITLE
# =====================================
st.title("🏦 European Bank Customer Churn Dashboard")
st.markdown("Interactive Customer Churn Analysis Dashboard")

st.markdown("""
This dashboard analyzes customer churn patterns for a European bank,
identifying high-risk customer segments, revenue risk, and key churn drivers
through interactive visualizations and machine learning insights.
""")

# =====================================
# SIDEBAR FILTERS
# =====================================
st.sidebar.header("Filters")

geo_filter = st.sidebar.multiselect(
    "Select Geography",
    options=df["Geography"].unique(),
    default=df["Geography"].unique()
)

gender_filter = st.sidebar.multiselect(
    "Select Gender",
    options=df["Gender"].unique(),
    default=df["Gender"].unique()
)

filtered_df = df[
    (df["Geography"].isin(geo_filter))
    &
    (df["Gender"].isin(gender_filter))
]

# =====================================
# KPI SECTION
# =====================================
total_customers = len(filtered_df)

churned_customers = filtered_df["Exited"].sum()

churn_rate = (
    churned_customers / total_customers * 100
)

avg_balance = filtered_df["Balance"].mean()

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Customers",
    f"{total_customers:,}"
)

col2.metric(
    "Churned Customers",
    f"{churned_customers:,}"
)

col3.metric(
    "Churn Rate",
    f"{churn_rate:.2f}%"
)

col4.metric(
    "Average Balance",
    f"€{avg_balance:,.0f}"
)

st.markdown("---")

# =====================================
# CHART 1 + CHART 2
# =====================================
col1, col2 = st.columns(2)

with col1:

    fig = px.pie(
        filtered_df,
        names="Exited",
        title="Overall Churn Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    geo_churn = (
        filtered_df.groupby("Geography")["Exited"]
        .mean()
        .reset_index()
    )

    geo_churn["Exited"] *= 100

    fig = px.bar(
        geo_churn,
        x="Geography",
        y="Exited",
        title="Churn Rate by Geography"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =====================================
# CHART 3 + CHART 4
# =====================================
col1, col2 = st.columns(2)

with col1:

    gender_churn = (
        filtered_df.groupby("Gender")["Exited"]
        .mean()
        .reset_index()
    )

    gender_churn["Exited"] *= 100

    fig = px.bar(
        gender_churn,
        x="Gender",
        y="Exited",
        title="Churn Rate by Gender"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    age_churn = (
        filtered_df.groupby("AgeGroup")["Exited"]
        .mean()
        .reset_index()
    )

    age_churn["Exited"] *= 100

    fig = px.bar(
        age_churn,
        x="AgeGroup",
        y="Exited",
        title="Churn Rate by Age Group"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =====================================
# CHART 5 + CHART 6
# =====================================
col1, col2 = st.columns(2)

with col1:

    credit_churn = (
        filtered_df.groupby("CreditScoreBand")["Exited"]
        .mean()
        .reset_index()
    )

    credit_churn["Exited"] *= 100

    fig = px.bar(
        credit_churn,
        x="CreditScoreBand",
        y="Exited",
        title="Credit Score Band vs Churn"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    fig = px.box(
        filtered_df,
        x="Exited",
        y="Balance",
        title="Balance Distribution by Churn Status"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =====================================
# SALARY VS BALANCE
# =====================================
fig = px.scatter(
    filtered_df,
    x="EstimatedSalary",
    y="Balance",
    color="Exited",
    title="Salary vs Balance Analysis"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================
# REVENUE RISK
# =====================================
st.subheader("Revenue Risk Analysis")

revenue_risk = filtered_df[
    filtered_df["Exited"] == 1
]["Balance"].sum()

st.metric(
    "Total Revenue at Risk",
    f"€{revenue_risk:,.0f}"
)

# =====================================
# HIGH VALUE CUSTOMERS
# =====================================
st.subheader("High Value Customer Churn Analysis")

high_value = filtered_df[
    filtered_df["Balance"]
    >
    filtered_df["Balance"].quantile(0.75)
]

fig = px.histogram(
    high_value,
    x="Balance",
    color="Exited",
    title="High Value Customer Churn"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================
# DATA PREVIEW
# =====================================
with st.expander("View Dataset"):

    st.dataframe(filtered_df)
    st.subheader("Key Business Insights")

st.subheader("📈 Top Features Driving Churn")

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

fig = px.bar(
    importance_df.head(10),
    x="Importance",
    y="Feature",
    orientation="h",
    title="Top 10 Most Important Features"
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption(
    "Built with Python, Streamlit, Plotly, Pandas, Scikit-Learn and Machine Learning."
)