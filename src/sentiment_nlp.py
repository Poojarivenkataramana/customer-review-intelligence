"""Day 4: sentiment analysis (VADER + TextBlob), n-grams, TF-IDF and a word-importance model.

Why VADER runs on the RAW review and TF-IDF on the CLEANED review
-----------------------------------------------------------------
VADER is rule-based and uses punctuation ("!!!"), capitalisation ("AMAZING"), emojis and
negation handling. Our regex cleaning would remove exactly those signals, so VADER gets
the original text. TF-IDF / n-grams want normalised tokens, so they get Review_clean.
"""
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

POS_T, NEG_T = 0.05, -0.05  # standard VADER thresholds

# Keep negations: "not good" and "good" are opposites.
NEGATIONS = {"not", "no", "never", "nothing", "nobody", "none", "neither", "nor", "cannot", "without", "n't"}
DOMAIN_STOP = {"restaurant", "place", "zomato", "ordered", "order", "went", "came", "got", "just", "really",
               "also", "like", "one", "us", "would", "will", "go", "get", "restro", "hyderabad"}
STOPWORDS = (set(ENGLISH_STOP_WORDS) - NEGATIONS) | DOMAIN_STOP
SENT_ORDER = ["Negative", "Neutral", "Positive"]


# ----------------------------------------------------------------------------
# Sentiment
# ----------------------------------------------------------------------------
def label_from_compound(c: float) -> str:
    return "Positive" if c >= POS_T else "Negative" if c <= NEG_T else "Neutral"


def rating_to_sentiment(r):
    """Star rating -> weak label. 1-2 negative, 3 neutral, 4-5 positive."""
    if pd.isna(r):
        return np.nan
    return "Negative" if r <= 2 else "Neutral" if r < 4 else "Positive"


def add_sentiment(df: pd.DataFrame, raw_col="Review") -> pd.DataFrame:
    df = df.copy()
    sia = SentimentIntensityAnalyzer()
    raw = df[raw_col].fillna("").str.replace(r"\s+", " ", regex=True)
    df["vader_compound"] = [sia.polarity_scores(t)["compound"] for t in raw]
    df["vader_label"] = df["vader_compound"].apply(label_from_compound)
    tb = [TextBlob(t).sentiment for t in raw]
    df["textblob_polarity"] = [s.polarity for s in tb]
    df["textblob_subjectivity"] = [s.subjectivity for s in tb]
    df["textblob_label"] = df["textblob_polarity"].apply(label_from_compound)
    df["rating_sentiment"] = df["Rating_numeric"].apply(rating_to_sentiment)
    return df


def agreement_table(df: pd.DataFrame, pred="vader_label", truth="rating_sentiment") -> pd.DataFrame:
    d = df.dropna(subset=[truth])
    return pd.crosstab(d[truth], d[pred]).reindex(index=SENT_ORDER, columns=SENT_ORDER, fill_value=0)


def agreement_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy / rank-correlation of each sentiment engine vs the star rating."""
    d = df.dropna(subset=["Rating_numeric"])
    rows = []
    for name, comp, lab in (("VADER", "vader_compound", "vader_label"), ("TextBlob", "textblob_polarity", "textblob_label")):
        rows.append({
            "engine": name,
            "exact_agreement_%": round((d[lab] == d["rating_sentiment"]).mean() * 100, 1),
            "polarity_agreement_%": round(  # ignoring the vague 3-star / neutral zone
                (d[(d.rating_sentiment != "Neutral") & (d[lab] != "Neutral")][lab]
                 == d[(d.rating_sentiment != "Neutral") & (d[lab] != "Neutral")]["rating_sentiment"]).mean() * 100, 1),
            "spearman_vs_rating": round(d[comp].corr(d["Rating_numeric"], method="spearman"), 3),
            "pearson_vs_rating": round(d[comp].corr(d["Rating_numeric"]), 3),
        })
    return pd.DataFrame(rows)


def flag_mismatches(df: pd.DataFrame) -> pd.DataFrame:
    """Reviews where text sentiment and stars strongly disagree (sarcasm, wrong stars, mixed reviews)."""
    d = df.copy()
    d["mismatch_type"] = "Consistent"
    d.loc[(d.Rating_numeric >= 4) & (d.vader_compound <= -0.3), "mismatch_type"] = "High stars, negative text"
    d.loc[(d.Rating_numeric <= 2) & (d.vader_compound >= 0.5), "mismatch_type"] = "Low stars, positive text"
    return d


# ----------------------------------------------------------------------------
# n-grams
# ----------------------------------------------------------------------------
def top_ngrams(texts, ngram=(2, 2), top_k=15, min_df=3) -> pd.DataFrame:
    vec = CountVectorizer(ngram_range=ngram, stop_words=list(STOPWORDS), min_df=min_df)
    X = vec.fit_transform(texts)
    counts = np.asarray(X.sum(axis=0)).ravel()
    out = pd.DataFrame({"ngram": vec.get_feature_names_out(), "count": counts})
    return out.sort_values("count", ascending=False).head(top_k).reset_index(drop=True)


def ngrams_by_group(df, group_col, groups, ngram=(2, 2), top_k=15, text_col="Review_clean") -> pd.DataFrame:
    frames = []
    for g in groups:
        t = top_ngrams(df.loc[df[group_col] == g, text_col], ngram, top_k)
        t.insert(0, group_col, g)
        frames.append(t)
    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------------------
# TF-IDF
# ----------------------------------------------------------------------------
def fit_tfidf(texts, ngram=(1, 2), min_df=5, max_df=0.7, max_features=20000):
    vec = TfidfVectorizer(ngram_range=ngram, stop_words=list(STOPWORDS), min_df=min_df, max_df=max_df,
                          sublinear_tf=True, max_features=max_features)
    X = vec.fit_transform(texts)
    return vec, X


def tfidf_top_terms(df, group_col, groups, top_k=15, text_col="Review_clean") -> pd.DataFrame:
    """Mean TF-IDF weight of each term inside each group (fit once on the full corpus)."""
    vec, X = fit_tfidf(df[text_col])
    terms = vec.get_feature_names_out()
    frames = []
    for g in groups:
        mask = (df[group_col] == g).to_numpy()
        mean_w = np.asarray(X[mask].mean(axis=0)).ravel()
        idx = mean_w.argsort()[::-1][:top_k]
        frames.append(pd.DataFrame({group_col: g, "term": terms[idx], "mean_tfidf": mean_w[idx].round(4)}))
    return pd.concat(frames, ignore_index=True)


def restaurant_signature_terms(df, top_k=5, text_col="Review_clean") -> pd.DataFrame:
    """Terms that are distinctive for one restaurant: one 'document' per restaurant."""
    docs = df.groupby("Restaurant")[text_col].apply(" ".join)
    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words=list(STOPWORDS), min_df=1, max_df=0.15, sublinear_tf=True)
    X = vec.fit_transform(docs)
    terms = vec.get_feature_names_out()
    rows = []
    for i, name in enumerate(docs.index):
        row = X[i].toarray().ravel()
        top = row.argsort()[::-1][:top_k]
        rows.append({"Restaurant": name, "signature_terms": ", ".join(terms[top])})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Which words predict a bad rating?  (TF-IDF + logistic regression)
# ----------------------------------------------------------------------------
def train_word_importance(df, text_col="Review_clean", top_k=20, seed=42):
    d = df.dropna(subset=["Rating_numeric"])
    d = d[(d.Rating_numeric <= 2) | (d.Rating_numeric >= 4)]          # drop ambiguous 3-star
    y = (d.Rating_numeric <= 2).astype(int)                           # 1 = negative
    X_tr, X_te, y_tr, y_te = train_test_split(d[text_col], y, test_size=0.2, random_state=seed, stratify=y)
    vec, Xtr = fit_tfidf(X_tr, min_df=3)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.5)
    clf.fit(Xtr, y_tr)
    Xte = vec.transform(X_te)
    pred = clf.predict(Xte)
    proba = clf.predict_proba(Xte)[:, 1]
    report = classification_report(y_te, pred, target_names=["Positive (4-5)", "Negative (1-2)"], output_dict=True)
    metrics = {"auc": round(roc_auc_score(y_te, proba), 3), "accuracy": round(report["accuracy"], 3),
               "negative_f1": round(report["Negative (1-2)"]["f1-score"], 3),
               "negative_recall": round(report["Negative (1-2)"]["recall"], 3),
               "negative_precision": round(report["Negative (1-2)"]["precision"], 3),
               "n_train": len(y_tr), "n_test": len(y_te)}
    terms = vec.get_feature_names_out()
    order = clf.coef_[0].argsort()
    neg = pd.DataFrame({"term": terms[order[::-1][:top_k]], "coef": clf.coef_[0][order[::-1][:top_k]].round(3)})
    pos = pd.DataFrame({"term": terms[order[:top_k]], "coef": clf.coef_[0][order[:top_k]].round(3)})
    return metrics, neg, pos
