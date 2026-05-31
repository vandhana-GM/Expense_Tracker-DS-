import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

from utils import clean_data, feature_engineering, get_category_summary, get_monthly_summary


DEFAULT_DATA_PATH = "data/budget_data.csv"


st.set_page_config(page_title="Expense Buddy", layout="wide")

st.title("Expense Buddy - Spending Analysis")
st.caption("Analyze your expenses by category, month, and unusually high transactions.")

uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

try:
    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        data_source = uploaded_file.name
    else:
        raw_df = pd.read_csv(DEFAULT_DATA_PATH)
        data_source = DEFAULT_DATA_PATH

    df = clean_data(raw_df)
    df = feature_engineering(df)
except FileNotFoundError:
    st.error(f"Could not find `{DEFAULT_DATA_PATH}`. Upload a CSV file from the sidebar.")
    st.stop()
except ValueError as error:
    st.error(str(error))
    st.stop()

st.sidebar.success(f"Using: {data_source}")

total_spend = df["amount"].sum()
avg_transaction = df["amount"].mean()
highest_transaction = df["amount"].max()
transaction_count = len(df)

metric_cols = st.columns(4)
metric_cols[0].metric("Total Spend", f"{total_spend:,.2f}")
metric_cols[1].metric("Transactions", f"{transaction_count:,}")
metric_cols[2].metric("Average Transaction", f"{avg_transaction:,.2f}")
metric_cols[3].metric("Highest Transaction", f"{highest_transaction:,.2f}")

tab_overview, tab_data = st.tabs(["Dashboard", "Data Preview"])

with tab_overview:
    cat = get_category_summary(df)
    monthly = get_monthly_summary(df)

    chart_cols = st.columns(2)

    with chart_cols[0]:
        st.subheader("Category-wise Spending")
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        cat.plot(kind="bar", ax=ax1, color="#2f7d7e")
        ax1.set_xlabel("Category")
        ax1.set_ylabel("Amount")
        ax1.tick_params(axis="x", rotation=35)
        st.pyplot(fig1, use_container_width=True)

    with chart_cols[1]:
        st.subheader("Spending Distribution")
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        cat.plot(kind="pie", autopct="%1.1f%%", ax=ax2)
        ax2.set_ylabel("")
        st.pyplot(fig2, use_container_width=True)

    st.subheader("Monthly Spending")
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    monthly.plot(kind="line", marker="o", ax=ax3, color="#b84a62")
    ax3.set_xlabel("Month")
    ax3.set_ylabel("Amount")
    ax3.tick_params(axis="x", rotation=35)
    st.pyplot(fig3, use_container_width=True)

    st.subheader("Overspending Transactions")
    over = df[df["amount"] > avg_transaction].sort_values("amount", ascending=False)
    st.dataframe(over[["date", "category", "amount"]], use_container_width=True)

    st.subheader("Next Month Prediction")
    if len(monthly) >= 2:
        monthly_df = monthly.reset_index()
        monthly_df["month_number"] = range(1, len(monthly_df) + 1)

        X = monthly_df[["month_number"]]
        y = monthly_df["amount"]

        model = LinearRegression()
        model.fit(X, y)

        next_month = pd.DataFrame({"month_number": [monthly_df["month_number"].max() + 1]})
        pred = model.predict(next_month)[0]

        st.success(f"Predicted next month expense: {pred:,.2f}")
    else:
        st.info("At least two months of data are needed for prediction.")

with tab_data:
    st.subheader("Raw Data")
    st.dataframe(raw_df.head(100), use_container_width=True)

    st.subheader("Cleaned Data")
    st.dataframe(df.head(100), use_container_width=True)
