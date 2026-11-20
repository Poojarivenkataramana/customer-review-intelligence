"""All project charts. Every function saves a PNG and returns the matplotlib figure."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .complaints import CATEGORY_PRIORITY
from .sentiment_nlp import SENT_ORDER

FIG_DIR = Path(__file__).resolve().parents[1] / "outputs" / "figures"
PALETTE = {"Positive": "#2a9d8f", "Neutral": "#e9c46a", "Negative": "#e76f51"}
CAT_COLORS = dict(zip(CATEGORY_PRIORITY, ["#264653", "#e76f51", "#8ab17d", "#e9c46a", "#2a9d8f", "#9b5de5"]))
sns.set_theme(style="whitegrid", context="notebook", font_scale=0.95)


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")
    return fig


def rating_distribution(df):
    fig, ax = plt.subplots(figsize=(7, 4))
    vc = df["Rating_numeric"].dropna().round().astype(int).value_counts().sort_index()
    bars = ax.bar(vc.index.astype(str), vc.values, color=["#e76f51", "#f4a261", "#e9c46a", "#8ab17d", "#2a9d8f"])
    ax.bar_label(bars, padding=2)
    ax.set(title="Rating distribution (half-stars rounded)", xlabel="Stars", ylabel="Reviews")
    return _save(fig, "01_rating_distribution.png")


def complaint_categories(summary, baseline):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    s = summary.sort_values("reviews")
    axes[0].barh(s.category, s.reviews, color=[CAT_COLORS[c] for c in s.category])
    for i, v in enumerate(s.reviews):
        axes[0].text(v + 10, i, f"{v:,} ({s['share_of_reviews_%'].iloc[i]}%)", va="center", fontsize=9)
    axes[0].set(title="Reviews mentioning each complaint", xlabel="Reviews (multi-label)")
    axes[0].set_xlim(0, s.reviews.max() * 1.3)
    s2 = summary.sort_values("rating_drop", ascending=False)
    axes[1].barh(s2.category, s2.avg_rating_when_present, color=[CAT_COLORS[c] for c in s2.category])
    base = baseline
    axes[1].axvline(base, color="k", ls="--", lw=1)
    axes[1].text(base + 0.03, -0.45, f"reviews with no complaint: {base:.2f}", fontsize=8)
    for i, v in enumerate(s2.avg_rating_when_present):
        axes[1].text(v + 0.03, i, f"{v:.2f}", va="center", fontsize=9)
    axes[1].invert_yaxis(); axes[1].set(title="Average rating when complaint is present", xlabel="Avg stars", xlim=(0, 5))
    return _save(fig, "02_complaint_categories.png")


def complaint_cooccurrence(co):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    norm = co.div(np.diag(co), axis=0) * 100
    sns.heatmap(norm, annot=co, fmt="d", cmap="Blues", cbar_kws={"label": "% of row category also in column"}, ax=ax)
    ax.set_title("Complaint co-occurrence (counts; colour = row-normalised %)")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    return _save(fig, "03_complaint_cooccurrence.png")


def sentiment_vs_rating(df, crosstab):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), gridspec_kw={"width_ratios": [1, 1.2, 1.2]})
    share = df["vader_label"].value_counts(normalize=True).reindex(SENT_ORDER) * 100
    axes[0].bar(share.index, share.values, color=[PALETTE[i] for i in share.index])
    for i, v in enumerate(share.values):
        axes[0].text(i, v + 1, f"{v:.1f}%", ha="center")
    axes[0].set(title="VADER sentiment distribution", ylabel="% of reviews", ylim=(0, 90))
    d = df.dropna(subset=["Rating_numeric"]).copy()
    d["stars"] = d.Rating_numeric.round().astype(int)
    sns.boxplot(data=d, x="stars", y="vader_compound", color="#8ecae6", fliersize=1, ax=axes[1])
    axes[1].axhline(0, color="k", lw=.8)
    axes[1].set(title="VADER score by star rating", xlabel="Stars", ylabel="Compound score")
    norm = crosstab.div(crosstab.sum(axis=1), axis=0) * 100
    sns.heatmap(norm, annot=crosstab.map(lambda x: f"{x:,}") + "\n(" + norm.round(0).astype(int).astype(str) + "%)",
                fmt="", cmap="YlGnBu", cbar=False, ax=axes[2])
    axes[2].set(title="Stars-based label (rows) vs VADER (columns)", xlabel="VADER", ylabel="Rating label")
    return _save(fig, "04_sentiment_vs_rating.png")


def ngram_panels(neg, pos, title, name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, d, col, lab in ((axes[0], neg, PALETTE["Negative"], "Negative reviews"), (axes[1], pos, PALETTE["Positive"], "Positive reviews")):
        d = d.sort_values("count")
        ax.barh(d.ngram, d["count"], color=col)
        ax.set_title(lab)
    fig.suptitle(title, y=1.02, fontsize=13)
    return _save(fig, name)


def word_importance(neg, pos, metrics):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    n = neg.sort_values("coef")
    axes[0].barh(n.term, n.coef, color=PALETTE["Negative"]); axes[0].set_title("Words pushing toward a 1-2 star review")
    p = pos.sort_values("coef", ascending=False)
    axes[1].barh(p.term, -p.coef, color=PALETTE["Positive"]); axes[1].set_title("Words pushing toward a 4-5 star review")
    fig.suptitle(f"TF-IDF + logistic regression (test AUC {metrics['auc']}, accuracy {metrics['accuracy']})", y=1.02)
    for a in axes: a.set_xlabel("Weight")
    return _save(fig, "06_word_importance.png")


def tfidf_terms(tf):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, g in zip(axes, ["Negative", "Positive"]):
        d = tf[tf.rating_sentiment == g].sort_values("mean_tfidf")
        ax.barh(d.term, d.mean_tfidf, color=PALETTE[g]); ax.set_title(f"Top TF-IDF terms: {g} (by stars)")
    return _save(fig, "05_tfidf_terms.png")


def delivery_time(df, buckets):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    d = df.dropna(subset=["Delivery_Time_Minutes"])
    axes[0].hist(d.Delivery_Time_Minutes, bins=24, color="#264653")
    axes[0].axvline(d.Delivery_Time_Minutes.median(), color="#e76f51", ls="--", label=f"median {d.Delivery_Time_Minutes.median():.0f} min")
    axes[0].legend(); axes[0].set(title=f"Delivery times mentioned in reviews (n={len(d)})", xlabel="Minutes", ylabel="Reviews")
    b = axes[1].bar(buckets.delivery_bucket.astype(str), buckets.avg_rating, color="#2a9d8f")
    axes[1].bar_label(b, labels=[f"{r:.2f}\n(n={n})" for r, n in zip(buckets.avg_rating, buckets.reviews)], padding=2, fontsize=9)
    axes[1].set(title="Average rating by delivery time", ylabel="Avg stars", ylim=(0, 5))
    return _save(fig, "07_delivery_time.png")


def restaurant_priority(sc, top=15):
    d = sc.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    left = np.zeros(len(d))
    for cat in CATEGORY_PRIORITY:
        col = "rate_" + {"Staff & Service": "staff_service", "Food Quality": "food_quality"}.get(cat, cat.lower())
        ax.barh(d.Restaurant, d[col], left=left, color=CAT_COLORS[cat], label=cat)
        left += d[col].to_numpy()
    for i, (a, r) in enumerate(zip(d.attention_score, d.avg_rating)):
        ax.text(left[i] + 1, i, f"score {a:.0f} | {r:.2f}\u2605", va="center", fontsize=8.5)
    ax.set(title=f"{top} restaurants needing most attention (stacked complaint rates, % of reviews)", xlabel="Complaint mentions per 100 reviews")
    ax.legend(loc="lower right", fontsize=8); ax.set_xlim(0, left.max() * 1.25)
    return _save(fig, "08_restaurant_priority.png")


def monthly_trend(mt):
    fig, ax = plt.subplots(figsize=(11, 4))
    x = pd.to_datetime(mt.review_month)
    ax.plot(x, mt.avg_rating, marker="o", color="#264653", label="Avg rating")
    ax.set_ylabel("Avg stars"); ax.set_ylim(2.5, 5)
    ax2 = ax.twinx(); ax2.plot(x, mt.complaint_rate, marker="s", color="#e76f51", label="Complaint rate %"); ax2.set_ylabel("Complaint rate %")
    ax2.grid(False)
    ax.set_title("Monthly rating and complaint rate (months with 40+ reviews)")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, loc="lower left")
    return _save(fig, "09_monthly_trend.png")


def cost_vs_quality(sc):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    sc_ = ax.scatter(sc.cost_for_two, sc.avg_rating, c=sc.complaint_rate, cmap="RdYlGn_r", s=60, edgecolor="k", lw=.4)
    z = np.polyfit(sc.cost_for_two, sc.avg_rating, 1); xs = np.linspace(sc.cost_for_two.min(), sc.cost_for_two.max(), 50)
    ax.plot(xs, np.polyval(z, xs), "k--", lw=1)
    r = sc.cost_for_two.corr(sc.avg_rating)
    ax.set(title=f"Price vs rating (r = {r:.2f}); colour = complaint rate", xlabel="Cost for two (INR)", ylabel="Avg rating")
    plt.colorbar(sc_, label="Complaint rate %")
    for _, row in pd.concat([sc.nlargest(3, "avg_rating"), sc.nsmallest(3, "avg_rating")]).iterrows():
        ax.annotate(row.Restaurant[:22], (row.cost_for_two, row.avg_rating), fontsize=7.5, xytext=(4, 4), textcoords="offset points")
    return _save(fig, "10_cost_vs_rating.png")


def mismatch_chart(df):
    d = df.dropna(subset=["Rating_numeric"])
    t = d.mismatch_type.value_counts().drop("Consistent", errors="ignore")
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(t.index, t.values, color=["#e76f51", "#e9c46a"][: len(t)]); ax.set_title("Where stars and text disagree strongly")
    for i, v in enumerate(t.values): ax.text(v + 3, i, str(v), va="center")
    return _save(fig, "11_mismatches.png")
