"""Business-facing aggregations built on top of the enriched review table."""
import numpy as np
import pandas as pd

from .complaints import CATEGORY_COLUMNS, CATEGORY_PRIORITY

COMPLAINT_COLS = [CATEGORY_COLUMNS[c] for c in CATEGORY_PRIORITY]


def complaint_summary(df: pd.DataFrame) -> pd.DataFrame:
    """For each complaint category: volume, rating impact and text sentiment."""
    rows = []
    rated = df.dropna(subset=["Rating_numeric"])
    for cat in CATEGORY_PRIORITY:
        col = CATEGORY_COLUMNS[cat]
        has = rated[rated[col] == 1]
        none = rated[rated[col] == 0]
        rows.append({
            "category": cat,
            "reviews": int(df[col].sum()),
            "share_of_reviews_%": round(df[col].mean() * 100, 1),
            "avg_rating_when_present": round(has.Rating_numeric.mean(), 2),
            "avg_rating_when_absent": round(none.Rating_numeric.mean(), 2),
            "rating_drop": round(has.Rating_numeric.mean() - none.Rating_numeric.mean(), 2),
            "pct_1_2_star_%": round((has.Rating_numeric <= 2).mean() * 100, 1),
            "avg_vader": round(df.loc[df[col] == 1, "vader_compound"].mean(), 3),
        })
    out = pd.DataFrame(rows)
    out["priority_rank"] = (out["reviews"] * -out["rating_drop"]).rank(ascending=False).astype(int)  # volume x damage
    return out.sort_values("priority_rank").reset_index(drop=True)


def complaint_cooccurrence(df: pd.DataFrame) -> pd.DataFrame:
    m = df[COMPLAINT_COLS].to_numpy().T @ df[COMPLAINT_COLS].to_numpy()
    return pd.DataFrame(m, index=CATEGORY_PRIORITY, columns=CATEGORY_PRIORITY)


def delivery_time_buckets(df: pd.DataFrame) -> pd.DataFrame:
    d = df.dropna(subset=["Delivery_Time_Minutes"]).copy()
    d["delivery_bucket"] = pd.cut(d.Delivery_Time_Minutes, bins=[0, 30, 45, 60, 90, 400],
                                  labels=["<=30 min", "31-45", "46-60", "61-90", ">90 min"])
    return d.groupby("delivery_bucket", observed=True).agg(
        reviews=("review_id", "count"), avg_rating=("Rating_numeric", "mean"),
        pct_negative_text=("vader_label", lambda s: (s == "Negative").mean() * 100)).round(2).reset_index()


def monthly_trend(df: pd.DataFrame, min_reviews=40) -> pd.DataFrame:
    g = df.groupby("review_month").agg(
        reviews=("review_id", "count"), avg_rating=("Rating_numeric", "mean"),
        complaint_rate=("has_complaint", "mean"),
        negative_text_rate=("vader_label", lambda s: (s == "Negative").mean())).reset_index()
    g = g[g.reviews >= min_reviews].copy()
    g["complaint_rate"] = (g.complaint_rate * 100).round(1)
    g["negative_text_rate"] = (g.negative_text_rate * 100).round(1)
    g["avg_rating"] = g.avg_rating.round(2)
    return g


def context_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Delivery-context reviews vs the rest (a proxy for online-order vs dine-in experience)."""
    g = df.groupby("is_delivery_context")
    out = g.agg(reviews=("review_id", "count"), avg_rating=("Rating_numeric", "mean"),
                complaint_rate=("has_complaint", "mean"),
                negative_text_rate=("vader_label", lambda s: (s == "Negative").mean()),
                **{c.replace("complaint_", "rate_"): (c, "mean") for c in COMPLAINT_COLS}).reset_index()
    out["is_delivery_context"] = out["is_delivery_context"].map({True: "Delivery / online order", False: "Dine-in / other"})
    for c in out.columns[3:]:
        out[c] = (out[c] * 100).round(1)
    out["avg_rating"] = out["avg_rating"].round(2)
    return out


def restaurant_scorecard(df: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("Restaurant")
    sc = g.agg(
        reviews=("review_id", "count"),
        avg_rating=("Rating_numeric", "mean"),
        pct_1_2_star=("Rating_numeric", lambda s: (s <= 2).mean() * 100),
        avg_vader=("vader_compound", "mean"),
        pct_negative_text=("vader_label", lambda s: (s == "Negative").mean() * 100),
        complaint_rate=("has_complaint", lambda s: s.mean() * 100),
        **{c.replace("complaint_", "rate_"): (c, lambda s: s.mean() * 100) for c in COMPLAINT_COLS},
    ).reset_index()
    rate_cols = [c.replace("complaint_", "rate_") for c in COMPLAINT_COLS]
    sc["top_complaint"] = sc[rate_cols].idxmax(axis=1).map(
        {c.replace("complaint_", "rate_"): n for n, c in zip(CATEGORY_PRIORITY, COMPLAINT_COLS)})
    sc.loc[sc[rate_cols].max(axis=1) == 0, "top_complaint"] = "None"
    # Attention score (0-100): higher = needs attention. 50% rating gap, 30% complaint rate, 20% negative text.
    sc["attention_score"] = (0.5 * (5 - sc.avg_rating) / 4 * 100 + 0.3 * sc.complaint_rate + 0.2 * sc.pct_negative_text).round(1)
    sc = sc.merge(metadata, on="Restaurant", how="left")
    sc["value_index"] = (sc.avg_rating / (sc.cost_for_two / 500)).round(2)  # stars per 500 INR for two
    num = sc.select_dtypes("number").columns
    sc[num] = sc[num].round(2)
    return sc.sort_values("attention_score", ascending=False).reset_index(drop=True)


def segment_summary(scorecard: pd.DataFrame, by: str, min_restaurants=3) -> pd.DataFrame:
    g = scorecard.groupby(by, observed=True).agg(
        restaurants=("Restaurant", "count"), avg_rating=("avg_rating", "mean"),
        complaint_rate=("complaint_rate", "mean"), avg_cost=("cost_for_two", "mean"))
    return g[g.restaurants >= min_restaurants].round(2).sort_values("avg_rating", ascending=False).reset_index()


def audit_sample(df: pd.DataFrame, n=15, seed=7) -> pd.DataFrame:
    """Random reviews per complaint category for manual precision checks."""
    frames = []
    for cat in CATEGORY_PRIORITY:
        d = df[df[CATEGORY_COLUMNS[cat]] == 1].sample(n, random_state=seed)
        d = d.assign(audit_category=cat)[["audit_category", "review_id", "Rating", "complaint_evidence", "Review"]]
        d["is_correct (fill Y/N)"] = ""
        frames.append(d)
    return pd.concat(frames, ignore_index=True)
