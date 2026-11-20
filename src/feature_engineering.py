"""Day 2-3: structural features and joins with restaurant metadata."""
import re

import numpy as np
import pandas as pd


def load_raw(data_dir):
    reviews = pd.read_csv(f"{data_dir}/restaurant_reviews.csv")
    metadata = pd.read_csv(f"{data_dir}/restaurant_metadata.csv")
    return reviews, metadata


def add_review_id(reviews: pd.DataFrame) -> pd.DataFrame:
    """The raw file has no review_id. Build a stable one from the raw row position."""
    out = reviews.copy()
    out.insert(0, "review_id", [f"R{i:05d}" for i in out.index])
    return out


def parse_reviewer_metadata(s: pd.Series) -> pd.DataFrame:
    """'104 Reviews , 5129 Followers' -> reviewer_total_reviews, reviewer_followers."""
    reviews = s.str.extract(r"(\d+)\s*Review", flags=re.I)[0].astype(float)
    followers = s.str.extract(r"(\d+)\s*Follower", flags=re.I)[0].astype(float).fillna(0)
    return pd.DataFrame({"reviewer_total_reviews": reviews, "reviewer_followers": followers})


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["review_datetime"] = pd.to_datetime(df["Time"], format="%m/%d/%Y %H:%M", errors="coerce")
    dt = df["review_datetime"].dt
    df["review_year"] = dt.year
    df["review_month"] = dt.to_period("M").astype(str)
    df["review_hour"] = dt.hour
    df["review_weekday"] = dt.day_name()
    df["is_weekend"] = dt.dayofweek >= 5
    return df


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["review_chars"] = df["Review"].fillna("").str.len()
    df["review_words"] = df["Review_clean"].str.split().str.len()
    df["has_picture"] = df["Pictures"] > 0
    df["exclamation_count"] = df["Review"].fillna("").str.count("!")
    letters = df["Review"].fillna("").str.count(r"[A-Za-z]")
    upper = df["Review"].fillna("").str.count(r"[A-Z]")
    df["caps_ratio"] = (upper / letters.replace(0, np.nan)).fillna(0).round(3)
    # Reviews mentioning ordering / delivery are an (approximate) delivery-context flag.
    df["is_delivery_context"] = df["Review_clean"].str.contains(
        r"\b(?:deliver\w*|zomato|swiggy|order(?:ed|ing)? (?:online|through|via|from)|parcel|takeaway|take away)\b",
        regex=True,
    )
    return df


def prepare_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    m = metadata.copy()
    m["cost_for_two"] = pd.to_numeric(m["Cost"].astype(str).str.replace(",", ""), errors="coerce")
    m["n_cuisines"] = m["Cuisines"].fillna("").str.split(",").str.len()
    m["primary_cuisine"] = m["Cuisines"].fillna("").str.split(",").str[0].str.strip()
    m["cost_band"] = pd.cut(
        m["cost_for_two"], bins=[0, 400, 700, 1000, 5000],
        labels=["Budget (<=400)", "Mid (401-700)", "Upper-mid (701-1000)", "Premium (>1000)"],
    )
    keep = ["Name", "cost_for_two", "cost_band", "primary_cuisine", "n_cuisines", "Cuisines"]
    return m[keep].rename(columns={"Name": "Restaurant"})
