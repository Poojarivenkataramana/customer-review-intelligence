"""End-to-end pipeline:  python run_pipeline.py

raw CSVs -> cleaning -> features -> delivery-time -> complaints -> sentiment/NLP
         -> business tables -> figures -> data/processed + outputs/
"""
import json
from pathlib import Path

import pandas as pd

from src import analysis as an
from src import visualization as viz
from src.complaints import CATEGORY_COLUMNS, CATEGORY_PRIORITY, add_complaint_columns
from src.feature_engineering import (add_review_id, add_text_features, add_time_features, load_raw,
                                     parse_reviewer_metadata, prepare_metadata)
from src.sentiment_nlp import (add_sentiment, agreement_metrics, agreement_table, flag_mismatches, ngrams_by_group,
                               restaurant_signature_terms, tfidf_top_terms, top_ngrams, train_word_importance)
from src.text_cleaning import clean_review, extract_delivery_time

ROOT = Path(__file__).resolve().parent
DATA, PROC, TAB = ROOT / "data", ROOT / "data" / "processed", ROOT / "outputs" / "tables"
for p in (PROC, TAB):
    p.mkdir(parents=True, exist_ok=True)


def build_dataset():
    raw, meta = load_raw(DATA)
    log = {"raw_rows": len(raw)}
    raw = add_review_id(raw)                                   # raw file has no review_id
    df = raw.dropna(subset=["Review", "Rating"], how="all").copy()   # 38 fully-empty rows
    log["after_drop_empty_rows"] = len(df)
    df["Rating_numeric"] = pd.to_numeric(df["Rating"], errors="coerce")  # 1 row has Rating == 'Like'
    log["non_numeric_ratings"] = int(df.Rating_numeric.isna().sum())
    df["Review_clean"] = df["Review"].apply(clean_review)
    df = df[df["Review_clean"] != ""].copy()                    # emoji-only / '.' / missing text
    log["after_drop_empty_text"] = len(df)
    log["repeat_reviewer_text_kept"] = int(df.duplicated(["Restaurant", "Reviewer", "Review"]).sum())  # different timestamps -> kept
    df = pd.concat([df, parse_reviewer_metadata(df["Metadata"])], axis=1)
    df = add_time_features(df)
    df = add_text_features(df)
    df[["Delivery_Time_Match", "Delivery_Time_Minutes"]] = df["Review_clean"].apply(lambda t: pd.Series(extract_delivery_time(t)))
    df = df.merge(prepare_metadata(meta), on="Restaurant", how="left")
    df = add_sentiment(df)
    df = add_complaint_columns(df)
    df = flag_mismatches(df)
    return df.reset_index(drop=True), meta, log


def main():
    df, meta, log = build_dataset()
    print("dataset:", df.shape, log)

    # ---- business tables ---------------------------------------------------
    summary = an.complaint_summary(df)
    co = an.complaint_cooccurrence(df)
    buckets = an.delivery_time_buckets(df)
    trend = an.monthly_trend(df)
    ctx = an.context_comparison(df)
    sc = an.restaurant_scorecard(df, prepare_metadata(meta))
    cost_seg = an.segment_summary(sc, "cost_band")
    cuisine_seg = an.segment_summary(sc, "primary_cuisine", min_restaurants=4)

    # ---- NLP tables --------------------------------------------------------
    ct = agreement_table(df)
    am = agreement_metrics(df)
    bi_neg = top_ngrams(df.loc[df.rating_sentiment == "Negative", "Review_clean"], (2, 2), 15)
    bi_pos = top_ngrams(df.loc[df.rating_sentiment == "Positive", "Review_clean"], (2, 2), 15)
    tri = ngrams_by_group(df, "rating_sentiment", ["Negative", "Positive"], (3, 3), 12)
    tf = tfidf_top_terms(df, "rating_sentiment", ["Negative", "Positive"], 12)
    sig = restaurant_signature_terms(df)
    metrics, neg_words, pos_words = train_word_importance(df)
    sc = sc.merge(sig, on="Restaurant", how="left")

    # ---- save tables & data ------------------------------------------------
    keep = ["review_id", "Restaurant", "Reviewer", "Review", "Review_clean", "Rating", "Rating_numeric", "review_datetime",
            "review_year", "review_month", "review_hour", "review_weekday", "is_weekend", "Pictures", "has_picture",
            "reviewer_total_reviews", "reviewer_followers", "review_chars", "review_words", "exclamation_count", "caps_ratio",
            "is_delivery_context", "Delivery_Time_Match", "Delivery_Time_Minutes",
            "vader_compound", "vader_label", "textblob_polarity", "textblob_subjectivity", "textblob_label",
            "rating_sentiment", "mismatch_type", *CATEGORY_COLUMNS.values(), "complaint_count", "has_complaint",
            "complaint_primary", "complaint_categories", "complaint_evidence", "cost_for_two", "cost_band", "primary_cuisine"]
    df[keep].to_csv(PROC / "reviews_enriched.csv", index=False)
    tables = {"complaint_summary": summary, "complaint_cooccurrence": co.reset_index(names="category"),
              "delivery_time_vs_rating": buckets, "monthly_trend": trend, "delivery_vs_dinein": ctx,
              "restaurant_scorecard": sc, "cost_band_summary": cost_seg, "cuisine_summary": cuisine_seg,
              "sentiment_vs_rating_crosstab": ct.reset_index(), "sentiment_engine_agreement": am,
              "bigrams_negative": bi_neg, "bigrams_positive": bi_pos, "trigrams": tri, "tfidf_top_terms": tf,
              "word_importance_negative": neg_words, "word_importance_positive": pos_words,
              "restaurant_signature_terms": sig,
              "mismatch_examples": df[df.mismatch_type != "Consistent"][["review_id", "Restaurant", "Rating", "vader_compound", "mismatch_type", "Review"]].head(200),
              "complaint_audit_sample": an.audit_sample(df)}
    for name, t in tables.items():
        t.to_csv(TAB / f"{name}.csv", index=False)

    # ---- figures -----------------------------------------------------------
    viz.rating_distribution(df)
    viz.complaint_categories(summary, df.loc[~df.has_complaint, "Rating_numeric"].mean())
    viz.complaint_cooccurrence(co)
    viz.sentiment_vs_rating(df, ct)
    viz.ngram_panels(bi_neg, bi_pos, "Most frequent bigrams by star-rating group", "05a_bigrams.png")
    viz.tfidf_terms(tf)
    viz.word_importance(neg_words, pos_words, metrics)
    viz.delivery_time(df, buckets)
    viz.restaurant_priority(sc)
    viz.monthly_trend(trend)
    viz.cost_vs_quality(sc)
    viz.mismatch_chart(df)

    # ---- headline numbers for the report ----------------------------------
    rated = df.dropna(subset=["Rating_numeric"])
    headline = {
        "pipeline_log": log, "final_rows": len(df), "restaurants": int(df.Restaurant.nunique()),
        "date_range": [str(df.review_datetime.min()), str(df.review_datetime.max())],
        "avg_rating": round(rated.Rating_numeric.mean(), 2),
        "complaint_rate_%": round(df.has_complaint.mean() * 100, 1),
        "multi_complaint_%": round((df.complaint_count > 1).mean() * 100, 1),
        "vader_distribution_%": (df.vader_label.value_counts(normalize=True) * 100).round(1).to_dict(),
        "rating_sentiment_distribution_%": (df.rating_sentiment.value_counts(normalize=True) * 100).round(1).to_dict(),
        "delivery_times_found": int(df.Delivery_Time_Minutes.notna().sum()),
        "delivery_median_min": float(df.Delivery_Time_Minutes.median()),
        "word_model": metrics,
        "mismatch_counts": df.mismatch_type.value_counts().to_dict(),
        "avg_rating_complaint_vs_not": [round(rated[rated.has_complaint].Rating_numeric.mean(), 2),
                                        round(rated[~rated.has_complaint].Rating_numeric.mean(), 2)],
        "cost_rating_corr": round(sc.cost_for_two.corr(sc.avg_rating), 3),
        "chars_negative_vs_positive": [round(df[df.rating_sentiment == s].review_chars.median()) for s in ("Negative", "Positive")],
    }
    (ROOT / "outputs" / "headline_metrics.json").write_text(json.dumps(headline, indent=2, default=str))
    print(json.dumps(headline, indent=2, default=str))
    return df, sc


if __name__ == "__main__":
    main()
