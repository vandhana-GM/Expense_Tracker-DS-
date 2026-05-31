import pandas as pd


CATEGORY_FIXES = {
    "Coffe": "Coffee",
    "Restuarant": "Restaurant",
}


def clean_data(df):
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()

    required_columns = {"date", "category", "amount"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required column(s): {missing}")

    df = df.dropna(subset=["date", "category", "amount"])
    df = df.drop_duplicates()

    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["date", "amount"])

    df["category"] = df["category"].astype(str).str.strip().str.title()
    df["category"] = df["category"].replace(CATEGORY_FIXES)

    return df


def feature_engineering(df):
    df = df.copy()
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["month_name"] = df["date"].dt.strftime("%b")
    df["year_month"] = df["date"].dt.tz_convert(None).dt.to_period("M").astype(str)
    return df


def get_category_summary(df):
    return df.groupby("category")["amount"].sum().sort_values(ascending=False)


def get_monthly_summary(df):
    return df.groupby("year_month")["amount"].sum()
