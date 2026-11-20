# Customer Review Intelligence - Project Report
*Hyderabad restaurant reviews (Zomato), 100 restaurants, May 2016 - May 2019*

## 1. Objective
Build an end-to-end customer-feedback analytics workflow that turns raw review text into
(a) structured complaint categories, (b) sentiment and key phrases, and
(c) restaurant-level business insights - showing practical Python, regex, NLP, feature engineering and
business problem-solving.

## 2. Data and preparation
| Step | Rows | Note |
|---|---|---|
| Raw reviews | 10,000 | 7 columns, **no `review_id`** -> created (`R00000`...) from the raw row position |
| Drop rows with neither review nor rating | 9,962 | 38 fully empty rows (all "American Wild Wings") |
| Drop reviews empty after cleaning | **9,944** | emoji-only, "." or "___" reviews (18) |

* Ratings: 1 non-numeric value (`Like`) kept as missing and excluded from rating statistics.
* Six same-reviewer/same-text pairs were **kept**: the timestamps differ, so they are separate reviews.
* Features added: parsed reviewer reviews/followers, date/time parts, review length, caps ratio, exclamation count,
  delivery-context flag, cost for two (numeric), cost band, primary cuisine, delivery time in minutes.
* Metadata is joined on restaurant name (100/100 match).

**Corrections to the exploratory cleaning code** (all covered by `tests/test_rules.py`):
1. `1.5 hours` became `1 5 hours` after cleaning and was read as 5 hours = 300 min -> decimals are now preserved.
2. `didn't` became `didn t`, breaking negation -> contractions expanded before punctuation removal.
3. The word "order" counted as delivery context, so dine-in waits ("ordered, waited 25 min at the table")
   were reported as delivery times (168 matches, about half false) -> now only deliver/arrive/zomato/swiggy/packed... count (74 genuine).

## 3. Complaint categorisation (rule-based, multi-label)
Regex lexicons per category with a negation guard ("not rude"), exclusions ("late night", "spill proof"),
and delivery-context logic (dine-in "service was late" -> Staff & Service). Every flag stores its **evidence phrase**.

| Category | Reviews | % of reviews | Avg stars when present | Star drop vs. no complaint in that category | % rated 1-2★ |
|---|---|---|---|---|---|
| Food Quality | 1,648 | 16.6 | 2.06 | -1.84 | 64.8 |
| Staff & Service | 967 | 9.7 | 2.18 | -1.58 | 62.8 |
| Pricing (incl. portion/value) | 674 | 6.8 | 2.82 | -0.84 | 41.2 |
| Delivery | 243 | 2.4 | 1.72 | -1.93 | 78.2 |
| Hygiene | 200 | 2.0 | 1.66 | -1.98 | 81.0 |
| Packaging | 40 | 0.4 | 2.25 | -1.35 | 60.0 |

* **29.6%** of reviews contain a classifiable complaint; 7.1% contain two or more categories.
* **Validation:** reviews with a complaint average **2.32★ vs 4.14★** without; a complaint is detected in about
  69% of 1-2★ reviews but only about 5% of 5★ reviews.
* **Manual spot-check:** about 12-14 reviews per category read by hand; roughly 80-90% were genuine complaints
  (informal, not a formal evaluation). `outputs/tables/complaint_audit_sample.csv` is a ready sheet for a proper precision measurement.

## 4. Sentiment and NLP
* **VADER** (on raw text) labels 74.3% positive / 7.1% neutral / 18.6% negative; star-based labels are 62.9 / 12.6 / 24.4.
* **VADER vs stars:** 91.9% polarity agreement when both sides take a position (76.2% exact over three classes; Spearman 0.58).
  TextBlob: 91.7% / 74.1% / Spearman 0.68 - similar, so VADER was kept as primary and TextBlob stored as a second opinion.
* **Where they disagree:** 72% of 3★ reviews read as positive text, and 295 reviews have 1-2★ with strongly positive text.
  Reading them shows long mixed reviews - praise for ambience or food, but one dealbreaker (slow staff, delay, wrong order) decided the rating.
  Only 34 reviews are 4-5★ with clearly negative text.
* **Sentiment hides some complaints:** Pricing (avg VADER +0.25) and Staff & Service (+0.02) complaints are written politely, yet cost 0.8-1.6 stars; Hygiene (-0.22) and Delivery (-0.13) read the angriest.
* **n-grams:** negative bigrams are `not good`, `bad experience`, `worst experience`, `not worth`, `waste money`, `not recommend`; positive ones are `good food`, `good service`, `ambience good`, `great food`, `value money`.
* **TF-IDF + logistic regression** (1-2★ vs 4-5★, 80/20 split): AUC 0.989, accuracy 95.9%, negative-class F1 0.93. Top negative words: `worst, bad, no, pathetic, disappointed, poor, waste, horrible, stale, cold, tasteless`.
  (A vocabulary sanity check - not a production model, since stars and text are written together.)
* **Restaurant signature terms** (TF-IDF, one document per restaurant) surface each place's known dishes and staff names.

## 5. Business insights
1. **Priority restaurants** (attention score = 50% rating gap + 30% complaint rate + 20% negative-text share):
   Hotel Zara Hi-Fi (2.40★, 60% complaint rate), Asian Meal Box (2.58★), Pakwaan Grand (2.71★), Mathura Vilas (2.82★, mostly service),
   The Chocolate Room, Behrouz Biryani, Shree Santosh Dhaba, KFC. For **74 of 100 restaurants Food Quality is the top complaint**; Staff & Service leads for 19.
2. **Biggest lever = Food Quality:** most frequent and about -1.8★ per mention. Hygiene and Delivery are rare but the most damaging per mention (about -2★, ~80% are 1-2★).
3. **Delivery speed:** among reviews quoting a time (n=74), average rating falls from 2.5★ (<=30 min) to 1.5★ (46-60) and 1.0★ (>90 min). Median stated delivery time: 40 min.
   (People mostly quote a time when annoyed, so read as a direction, not a population estimate.)
4. **Delivery-context reviews** (12.6% of reviews) average 3.14★ vs 3.67★ for dine-in; complaint rate 40.5% vs 28.0%.
5. **Price vs satisfaction:** premium restaurants (>INR 1,000 for two) average 3.95★ vs 3.31★ for budget (r = 0.42).
   Budget places draw more food-quality (19.6 vs 13.5 per 100 reviews) and hygiene complaints (3.2 vs 1.2); premium and upper-mid places draw more *service* complaints (12.6-13.1 vs 4.8).
6. **Time trend:** flat at about 3.4-3.5★ from Aug 2018 to Mar 2019, better in Apr-May 2019 - but the file holds each restaurant's latest ~100 reviews, so months are unevenly covered; treat with caution.

### Recommendations
* Review the top-15 attention list monthly; route by colour mix in `08_restaurant_priority.png` (kitchen / floor staff / logistics).
* Target freshness, temperature and under-cooking first: stale, cold, half-cooked, tasteless recur.
* Service training and wait-time alerts (table-ready, order-status updates), especially for mid/premium venues.
* Alert on delivery promises beyond 60 minutes; escalate hygiene mentions immediately.
* Use complaint flags *alongside* sentiment - sentiment alone under-weights polite price and service complaints.

## 6. Limitations and next steps
* Rules, not a trained model: about 69% of 1-2★ reviews get a category. Missing: wrong/missing items, ambience, parking/reservation issues, very short reviews. Label ~500 reviews, measure precision/recall, then train a classifier.
* VADER misses sarcasm and averages mixed reviews; aspect-based sentiment (sentiment per topic) is the natural upgrade.
* Equal ~100 reviews per restaurant means volumes cannot be compared across restaurants - only rates.
* Portion-size complaints are counted under Pricing (value for money).
* Delivery context is a keyword proxy; the data has no order-type field.
* The attention-score weights are a judgement call, documented in `src/analysis.py`.

## 7. Reproduce
```
pip install -r requirements.txt
python run_pipeline.py          # ~15 s: data/processed/reviews_enriched.csv, outputs/tables/*, outputs/figures/*
python tests/test_rules.py      # regression tests
```
Notebooks `01`-`05` walk through the same steps with explanations; `03`-`05` import the code in `src/`.
