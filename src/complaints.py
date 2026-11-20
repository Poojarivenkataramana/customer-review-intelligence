"""Day 3: rule-based complaint categorisation (multi-label).

Design
------
* Works on the *cleaned* text (contractions already expanded: "didn't" -> "did not").
* Two kinds of patterns per category:
    - `cues`    : single negative words ("rude", "stale", "expensive"). These are
                  suppressed when a negator ("not", "no", "never", "without") appears in
                  the 4 words before them ("not expensive", "never late").
    - `phrases` : full complaint phrases that already contain their own negation
                  ("not fresh", "took too long"). These are never suppressed.
* `exclusions` remove known false friends ("late night", "late lunch").
* A review can belong to several categories. `Complaint_Primary` is the category with
  the most pattern hits (ties resolved by CATEGORY_PRIORITY).
"""
import re
from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd

NEGATORS = {"not", "no", "never", "without", "nothing", "hardly", "neither", "nor", "cannot"}
NEG_WINDOW = 4

_ADJ = r"(?:very |so |too |really |extremely |quite |a bit |bit |just |also |was |were |is |are |been |being |be )*"
_NEG_QUALITY = (r"bad|worst|terrible|awful|poor|horrible|pathetic|disgusting|bland|cold|stale|oily|"
                r"dry|burnt|burned|soggy|raw|tasteless|overcooked|undercooked|salty|sour|smelly|"
                r"stinking|rubbery|chewy|unfresh|mediocre|substandard|inedible|disappointing|average")


@dataclass
class Category:
    name: str
    cues: List[str] = field(default_factory=list)
    phrases: List[str] = field(default_factory=list)
    exclusions: List[str] = field(default_factory=list)
    ctx_cues: List[str] = field(default_factory=list)  # ambiguous cues: need delivery context


CATEGORIES: Dict[str, Category] = {
    "Delivery": Category(
        "Delivery",
        cues=[r"undelivered"],
        ctx_cues=[r"late", r"delays?", r"delayed", r"cancel(?:led|ed|ling|lation)?"],
        phrases=[
            r"deliver\w*\W+(?:\w+\W+){0,3}(?:late|slow|delayed|took|never)",
            r"delivery\W+(?:\w+\W+){0,2}(?:bad|poor|worst|pathetic|horrible|terrible|disappointing)",
            r"(?:late|slow|worst|pathetic|poor|bad|horrible|terrible) (?:\w+ )?delivery",
            r"(?:did not|never|yet to|not) (?:deliver|arrive|receive|get (?:my|the) (?:order|food|delivery))\w*",
            r"order (?:was |got |is )?(?:not (?:delivered|received|accepted)|never (?:came|arrived))",
            r"no delivery",
            r"(?:wrong|missing|incomplete) (?:order|items?|food|dish)",
            r"items? (?:was |were |is |are )?missing",
            r"still waiting (?:for|since)",
            r"refund",
            r"deliver\w* (?:boy|guy|executive|person|partner)\W+(?:\w+\W+){0,3}(?:rude|late|lost|called|misbehav\w*)",
            r"took (?:too |so |very )?(?:long|much time|forever|ages|hours?)\W+(?:\w+\W+){0,4}(?:deliver|arrive|reach)",
            r"(?:deliver|arrive|reach)\w*\W+(?:\w+\W+){0,6}after (?:\d|an? (?:hour|long)|more than)",
        ],
        exclusions=[r"late (?:night|evening|lunch|dinner|breakfast|afternoon|morning|hours?|opening|closing|at night)",
                    r"too late to", r"lately", r"later"],
    ),
    "Food Quality": Category(
        "Food Quality",
        cues=[r"stale", r"tasteless", r"bland", r"undercooked", r"overcooked", r"uncooked", r"burnt", r"soggy",
              r"rubbery", r"inedible", r"unfresh", r"watery", r"greasy"],
        phrases=[
            rf"(?:food|taste|tasted|quality|biryani|chicken|mutton|pizza|burger|rice|paneer|curry|starters?|dessert|cake|"
            rf"naan|roti|gravy|dish(?:es)?|fish|prawns?|noodles|manchurian|sandwich|pasta|dal|dosa|idli|soup)\W+{_ADJ}\W*"
            rf"(?:\w+\W+){{0,2}}(?:{_NEG_QUALITY})",
            r"not (?:very |so |that |at all )?(?:fresh|tasty|good|up to (?:the )?mark|worth|hot|cooked|edible|delicious|satisfying)"
            r"\W+(?:\w+\W+){0,3}(?:food|taste|quality|biryani|chicken|dish|item|item)",
            r"(?:food|taste|quality|biryani|chicken|dish)\W+(?:\w+\W+){0,2}not (?:very |so |that |at all )?"
            r"(?:fresh|tasty|good|up to (?:the )?mark|worth|hot|cooked|edible|delicious|satisfying|great|nice|authentic)",
            r"(?:bad|poor|worst|horrible|terrible|pathetic|awful|average|low|substandard|disappointing) "
            r"(?:\w+ )?(?:food|taste|quality)",
            r"tasted? (?:\w+ )?(?:bad|awful|weird|funny|sour|stale|bland|off|average|horrible|terrible)",
            r"half (?:cooked|baked|done)|under cooked|over cooked|not cooked|not properly cooked|not fresh",
            r"food (?:was )?(?:cold|stale|raw|dry|oily|salty|spoilt|spoiled|rotten)",
            r"(?:cold|stale|raw|dry|oily|salty|spoilt|spoiled|rotten|burnt) (?:\w+ )?(?:food|biryani|chicken|rice|pizza|burger|curry)",
            r"(?:no|lack of|lacks|lacking) (?:\w+ )?(?:taste|flavou?r)",
            r"food poisoning|stomach (?:ache|upset|infection|pain|issue)|loose motion|vomit\w*|diarrh\w*",
            r"(?:waste|worst) (?:of )?(?:food|taste)",
            r"not (?:at all )?(?:satisfied|impressed) (?:with|by) (?:the )?(?:food|taste)",
            r"(?:bad|worst|horrible|terrible|pathetic|awful|disgusting|poor|pathetically|tasteless) (?:\w+ ){0,2}"
            r"(?:biryani|biriyani|briyani|dish(?:es)?|curry|meal|starters?|mains?|desserts?|items?|cuisine|meat|chicken|mutton|"
            r"fish|prawns?|paneer|burger|pizza|pasta|noodles|rice|sandwich|momos?|shawarma|kebabs?|rolls?|cake|coffee|tea)",
            r"taste ?less|not (?:very |so |that |at all )?tasty|taste (?:was |is )?(?:not|nothing|average|ok ok)|not made well|not prepared|preparation (?:was )?not",
            r"food poison\w*|too (?:much )?(?:oil|oily|salt|salty|sweet|spicy)|(?:too|so|very) much (?:oil|salt)|full of oil|oil (?:was )?(?:floating|everywhere)",
            r"(?:plastic|insect|stone|metal|spider|fly|bug|worm|cockroach|hair)s? (?:piece |strand )?(?:in|inside|found in) (?:my |the |our )?(?:food|dish|biryani|drink|shake|cake|burger|rice|curry)",
        ],
        exclusions=[r"above average", r"not bad", r"not too bad", r"nothing (?:was )?bad", r"no complaints?"],
    ),
    "Packaging": Category(
        "Packaging",
        cues=[r"spill(?:ed|ing|age)?", r"leak(?:ed|ing|s)?", r"tampered"],
        exclusions=[r"spill ?proof", r"leak ?proof", r"water pipe", r"drainage"],
        phrases=[
            r"packag\w*\W+(?:\w+\W+){0,3}(?:bad|poor|worst|horrible|terrible|pathetic|awful|messy|not (?:good|proper|properly|great|up)|"
            r"leak\w*|spill\w*|open|broken|torn|damaged|sloppy|flimsy|useless|disappointing|cheap)",
            r"(?:bad|poor|worst|horrible|terrible|pathetic|awful|messy|improper|cheap|flimsy|sloppy) (?:\w+ )?packag\w*",
            r"(?:not|poorly|badly|improperly|carelessly|loosely) (?:\w+ )?(?:packed|packaged|sealed|wrapped)",
            r"(?:packed|packaged|sealed|wrapped) (?:\w+ )?(?:badly|poorly|improperly|carelessly|loosely)",
            r"(?:container|box|parcel|packet|bag|seal|lid|packing)s?\W+(?:was |were |came |arrived |found |got |had )?(?:\w+ )?(?:open|opened|broke|broken|leak\w*|spill\w*|torn|damaged|tampered|crushed|loose|cracked)\b",
            r"(?:broken|tampered|crushed|torn|damaged|leaking|leaked) (?:\w+ )?(?:seal|container|box|parcel|packet|packing|packaging)",
            r"(?:gravy|curry|raita|soup|sauce|salan|drink|juice)\W+(?:\w+\W+){0,3}(?:spill\w*|leak\w*|all over)",
        ],
    ),
    "Pricing": Category(
        "Pricing",
        cues=[r"overpriced", r"expensive", r"costly", r"pricey", r"pricy", r"exorbitant", r"hefty", r"ripoff",
              r"overcharg\w*", r"overcharge"],
        phrases=[
            r"over priced",
            r"rip ?offs?",
            r"(?:very |too |so |bit |little |slightly )?(?:high|steep|more|excessive|unreasonable) (?:\w+ )?(?:price|prices|pricing|cost|charges?|bill|rates?)",
            r"(?:price|prices|pricing|cost|charges?|bill|rates?)\W+(?:\w+\W+){0,2}(?:high|steep|excessive|unreasonable|too much|much higher|on the higher)",
            r"not (?:worth|value for|worthy)(?: (?:the )?(?:money|price|cost|it|every))?",
            r"(?:no|not|zero|poor|less) (?:\w+ )?value for money",
            r"(?:waste of|wasted|waste) (?:\w+ )?(?:money|rupees|bucks)",
            r"money (?:waste|wasted)",
            r"(?:hidden|extra|additional|unnecessary|surprise) (?:\w+ )?(?:charges?|fees?|tax(?:es)?)",
            r"charged (?:\w+ )?(?:extra|more|too much|separately|for (?:water|packaging|everything))",
            r"too much for (?:the )?(?:price|money|what)",
            r"(?:quantity|portion)s? (?:is |was |are |were )?(?:\w+ )?(?:very |too |so )?(?:less|small|tiny|low|little|insufficient|meagre|meager)",
            r"(?:less|small|tiny|low|little|poor|meagre|meager) (?:\w+ )?(?:quantity|portion)s?",
        ],
    ),
    "Staff & Service": Category(
        "Staff & Service",
        cues=[r"rude", r"rudely", r"impolite", r"arrogant", r"unprofessional", r"ignored", r"ignoring", r"inattentive",
              r"misbehav\w*", r"unresponsive", r"unhelpful", r"careless", r"disrespect\w*", r"inefficient", r"clueless", r"negligen\w*"],
        phrases=[
            r"(?:bad|poor|worst|horrible|terrible|pathetic|awful|slow|pathetic|pathetic|average|disappointing|lousy|lacking|ordinary|pathetically) "
            r"(?:\w+ )?(?:service|staff|behaviou?r|attitude|hospitality|management|waiters?|servers?|customer (?:care|service|support))",
            r"(?:service|staff|waiters?|servers?|management|manager|behaviou?r|attitude|hospitality|customer (?:care|service|support)|team|crew)"
            r"\W+(?:\w+\W+){0,2}(?:rude|slow|poor|bad|worst|terrible|horrible|pathetic|awful|lacking|disappointing|average|not "
            r"(?:good|great|helpful|courteous|polite|friendly|attentive|responsive|professional|up to|satisfactory|cooperative)|"
            r"could be better|needs? (?:to )?improv\w*|useless|casual|careless|inattentive|unprofessional)",
            r"(?:no one|nobody|none) (?:\w+ )?(?:came|attended|bothered|responded|served|cared|there to|available|was there)",
            r"(?:long|slow|too long|so long) (?:\w+ )?(?:wait|waiting)",
            r"(?:wait\w*|kept waiting|takes?|took)(?: for)? (?:\w+ ){0,3}(?:\d+ ?(?:mins?|minutes?|hours?|hrs?)|an hour|half an hour|ages|forever)",
            r"(?:zero|no|little|lack of|lacking|poor|bad) (?:\w+ )?(?:attention|etiquette|manners|courtesy|hospitality|professionalism|response|coordination)",
            r"(?:needs?|need) (?:a lot of |lot of |some )?improvement",
            r"(?:kept|made) (?:us |me )?(?:\w+ )?waiting",
            r"(?:denied|refused) to (?:\w+ )?(?:accept|serve|take|give|allow)",
            r"(?:wait|waiting) (?:time )?(?:was |is )?(?:very |too |so )?(?:long|more|high|ridiculous|bad|horrible)",
            r"waited (?:for )?(?:\w+ )?(?:more than|over|almost|about|around|nearly|atleast|at least|for)? ?\d+ ?(?:mins?|minutes?|hours?|hrs?)",
            r"waited (?:so |too )?(?:long|forever|ages)",
            r"took (?:so |too |very )?(?:long|forever|ages|much time) (?:\w+ )?(?:to|for) (?:serve|take|bring|get|give|attend|come|clear|prepare|order|seat)",
            r"(?:forgot|forget|forgotten) (?:\w+ )?(?:order|our|my|to)",
            r"had to (?:ask|remind|call|request|follow up|wait|run after) (?:\w+ )?(?:multiple|several|repeatedly|again|many|twice|thrice|them)",
            r"(?:shouted|yelled|argued|scolded|insulted|fought|quarrel\w*|harass\w*|misbehav\w*)",
            r"(?:did not|never|not) (?:\w+ )?(?:listen|respond|attend|bother|care|help|serve|apologi[sz]e|take (?:the )?order)",
            r"customer (?:care|service|support) (?:\w+ )?(?:did not|never|not|useless|pathetic|bad|poor)",
            r"(?:slow|poor|bad|worst) service",
        ],
    ),
    "Hygiene": Category(
        "Hygiene",
        cues=[r"unhygienic", r"unhygenic", r"dirty", r"filthy", r"unclean", r"cockroach(?:es)?", r"insects?", r"worms?",
              r"flies", r"rat", r"rats", r"fungus", r"mouldy", r"moldy"],
        phrases=[
            r"hair (?:in|on|inside|strand)|found (?:a |some )?hair|(?:a|the|long) hair (?:was|in|on|inside)|hair(?:s)? (?:found|in)",
            r"not (?:very |so |at all |that )?(?:hygienic|clean|neat|maintained|well maintained)",
            r"smell(?:s|ed|ing|y)? (?:\w+ )?(?:bad|foul|awful|terrible|weird|strange|funny|unpleasant|nasty|rotten|fishy|stale|off|horrible|worst|disgusting|pungent|strong)",
            r"(?:bad|foul|awful|terrible|weird|strange|funny|unpleasant|nasty|rotten|fishy|stale|horrible|disgusting|pungent) (?:smell|odou?r)",
            r"stink\w*|stench|foul odou?r",
            r"(?:plastic|insect|stone|metal|spider|fly|bug|worm|cockroach)s? (?:piece |strand )?(?:in|inside|found in) (?:my |the |our )?(?:food|dish|biryani|drink|shake|cake|burger|rice|curry)",
            r"(?:poor|bad|worst|lack of|no) (?:\w+ )?(?:hygiene|cleanliness|sanitation)",
            r"(?:washroom|restroom|toilet|table|tables|plates?|glasses?|cutlery|kitchen|floor)s?\W+(?:\w+\W+){0,3}(?:dirty|filthy|unclean|stained|smelly|greasy|sticky|unhygienic)",
        ],
    ),
}

CATEGORY_PRIORITY = ["Delivery", "Food Quality", "Packaging", "Pricing", "Staff & Service", "Hygiene"]
CATEGORY_COLUMNS = {c: "complaint_" + re.sub(r"\W+", "_", c.lower()).strip("_") for c in CATEGORIES}

_COMPILED = {
    name: {
        "cues": [re.compile(rf"\b{p}\b") for p in cat.cues],
        "ctx_cues": [re.compile(rf"\b{p}\b") for p in cat.ctx_cues],
        "phrases": [re.compile(rf"\b(?:{p})") for p in cat.phrases],
        "exclusions": [re.compile(rf"\b{p}") for p in cat.exclusions],
    }
    for name, cat in CATEGORIES.items()
}


# Negative adjectives: "not rude", "never late", "not bad" are POSITIVE statements.
_NEG_WORDS = (r"rude|slow|poor|bad|worst|terrible|horrible|pathetic|awful|late|delayed|expensive|costly|pricey|"
              r"stale|cold|dry|bland|dirty|unhygienic|filthy|unclean|oily|salty|burnt|raw|tasteless|disappointing|"
              r"overpriced|careless|inattentive|unprofessional|arrogant|impolite|unhelpful|smelly|average|high|steep|excessive|unreasonable|long")
_DOUBLE_NEG = re.compile(rf"\b(?:not|never|no|without|nothing)\s+(?:\w+\s+)?(?:very |so |that |at all |too |really )?(?:{_NEG_WORDS})\b")


def _is_negated(text: str, start: int) -> bool:
    prev_words = re.findall(r"[a-z']+", text[max(0, start - 40):start])[-NEG_WINDOW:]
    return any(w in NEGATORS for w in prev_words)


DELIVERY_CTX = re.compile(
    r"\b(?:deliver\w*|zomato|swiggy|parcel|take ?away|packag\w*|packed|packing|online|app|rider|courier|dispatch\w*|"
    r"order(?:ed|ing)? (?:via|through|from|on|for home)|home)\b"
)


def category_hits(text: str, name: str, delivery_ctx: bool = True) -> List[str]:
    """Return the list of matched strings (evidence) for one category.

    Ambiguous cues ("late", "delay", "cancelled") only count as Delivery when the review
    shows a delivery context; otherwise they are dine-in service delays (see classify_review).
    """
    if not isinstance(text, str) or not text:
        return []
    comp = _COMPILED[name]
    masked = text
    for ex in comp["exclusions"]:
        masked = ex.sub(lambda m: " " * len(m.group()), masked)  # keep offsets stable
    hits = []
    if delivery_ctx:
        for rx in comp["ctx_cues"]:
            for m in rx.finditer(masked):
                if not _is_negated(masked, m.start()):
                    hits.append(m.group().strip())
    for rx in comp["cues"]:
        for m in rx.finditer(masked):
            if not _is_negated(masked, m.start()):
                hits.append(m.group().strip())
    for rx in comp["phrases"]:
        for m in rx.finditer(masked):
            if _DOUBLE_NEG.search(m.group()):
                continue
            hits.append(m.group().strip())
    return hits


def classify_review(text: str, delivery_minutes=None) -> dict:
    """Return per-category evidence for a single review."""
    ctx = bool(DELIVERY_CTX.search(text)) if isinstance(text, str) else False
    result = {name: category_hits(text, name, ctx) for name in CATEGORIES}
    if not ctx:  # "service was late" in a dine-in review is a service complaint
        result["Staff & Service"] += [f"{h} (service)" for h in category_hits(text, "Delivery", True)
                                      if h.split()[0] in {"late", "delay", "delays", "delayed"}]
    # Extracted numeric delivery time >= 60 min counts as a delivery complaint too.
    if delivery_minutes is not None and not pd.isna(delivery_minutes) and delivery_minutes >= 60:
        result["Delivery"].append(f"delivery time {int(delivery_minutes)} min")
    return result


def add_complaint_columns(df: pd.DataFrame, text_col="Review_clean", delivery_col="Delivery_Time_Minutes") -> pd.DataFrame:
    df = df.copy()
    evidence = [
        classify_review(t, d) for t, d in zip(df[text_col], df[delivery_col] if delivery_col in df else [None] * len(df))
    ]
    for name, col in CATEGORY_COLUMNS.items():
        df[col] = [int(len(e[name]) > 0) for e in evidence]
    df["complaint_count"] = df[list(CATEGORY_COLUMNS.values())].sum(axis=1)
    df["has_complaint"] = df["complaint_count"] > 0

    def primary(e):
        counts = {n: len(h) for n, h in e.items() if h}
        if not counts:
            return "No Complaint Detected"
        return sorted(counts, key=lambda n: (-counts[n], CATEGORY_PRIORITY.index(n)))[0]

    df["complaint_primary"] = [primary(e) for e in evidence]
    df["complaint_categories"] = [
        ", ".join(n for n in CATEGORY_PRIORITY if e[n]) or "None" for e in evidence
    ]
    df["complaint_evidence"] = [
        " | ".join(f"{n}: {e[n][0]}" for n in CATEGORY_PRIORITY if e[n]) for e in evidence
    ]
    return df
