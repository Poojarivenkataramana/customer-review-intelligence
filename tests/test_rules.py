"""Regression tests for cleaning, delivery-time extraction and complaint rules.
Run:  python tests/test_rules.py     (or: pytest tests)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.complaints import classify_review
from src.text_cleaning import clean_review, extract_delivery_time


def cats(text):
    return {k for k, v in classify_review(clean_review(text)).items() if v}


def test_cleaning():
    assert clean_review("Delivered in 1.5 hours!!!") == "delivered in 1.5 hours"
    assert clean_review("They didn't deliver") == "they did not deliver"
    assert clean_review("Visit https://x.com @ramana GREAT") == "visit great"
    assert clean_review(None) == ""
    assert clean_review("😍😍") == "" and clean_review("___") == ""


def test_delivery_time():
    f = lambda t: extract_delivery_time(clean_review(t))[1]
    assert f("food was delivered after 1.5 hours") == 90
    assert f("delivery took 45 mins") == 45
    assert f("we ordered and waited 25 minutes at the table") is None   # dine-in wait
    assert f("open 24 hours and the delivery was quick") is None        # > 6 h sanity limit


def test_complaints():
    assert cats("Delivery was very late and the food was cold") == {"Delivery", "Food Quality"}
    assert cats("We had a late night dinner here") == set()
    assert cats("The staff was not rude at all") == set()
    assert cats("Staff was rude and ignored us") == {"Staff & Service"}
    assert cats("Packaging was bad and the gravy spilled") == {"Packaging"}
    assert cats("Loved the spill proof packaging") == set()
    assert cats("Way too expensive for the quantity") == {"Pricing"}
    assert cats("Prices are not high, very reasonable") == set()
    assert cats("Found a hair in my biryani") == {"Food Quality", "Hygiene"}
    assert cats("The room smells amazing") == set()
    assert cats("Service was a bit late") == {"Staff & Service"}
    assert cats("Ordered via Zomato and it was delayed by an hour") == {"Delivery"}
    assert cats("We waited almost an hour for a table") == {"Staff & Service"}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
