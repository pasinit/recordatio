"""Shared text-processing pipeline for keyword retrieval (RETRIEVAL_DESIGN.md §3.1).

ONE pure code path used verbatim at index time and query time (gate R6):
number/unit normalization → unicode61-compatible tokenization → simplemma
lemmatization (EN first, IT fallback). The FTS index stores this module's
output; FTS queries are built from this module's output — phrase matching
breaks silently if the two ever diverge, so nothing here may be duplicated
elsewhere.

Number normalization runs on the raw string BEFORE tokenization: thousands
separators and currency symbols are unicode61 token separators, so they must
be rewritten while still attached to their digits. The enumerated formats of
§3.1 are the full scope — resist growing this (the corpus measurement showed
only 5% of queries carry digits; this is index hygiene, not a feature).

Deterministic and offline: simplemma ships pinned lemma data; no model calls.
"""

import re

import simplemma

# Languages for lemmatization: English first, Italian fallback (§3.1).
_LANGS = ("en", "it")

# unicode61-compatible tokens: runs of letters/digits (no underscore — SQLite's
# unicode61 treats "_" as a separator, Python's \w does not).
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)

# Currency symbol → ISO code, emitted AFTER the value (§3.1: "£2,000" → "2000 gbp").
_CURRENCY_CODES = {"£": "gbp", "$": "usd", "€": "eur"}

# One currency amount: symbol, digits with optional thousands separators,
# optional decimals, optional k/K multiplier. Ranges ("£150k–£200k") need no
# special casing: each side normalizes independently and the hyphen/en-dash is
# a tokenizer separator.
_CURRENCY_RE = re.compile(r"[£$€]\s?(\d{1,3}(?:,\d{3})*|\d+)(\.\d+)?([kK])?")

# Plain thousands-separated number ("150,000" → "150000"), only when not part
# of a currency match (currency is rewritten first).
_THOUSANDS_RE = re.compile(r"\b\d{1,3}(?:,\d{3})+\b")

# Stopword list frozen for the value model AND the keyword source's loose-term
# extraction (§3.3 step 3 reuses the budgeter's list; it lives here so the
# budgeter and the keyword source share one committed copy).
STOPWORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "if", "of", "to", "in", "on",
        "at", "by", "for", "with", "about", "as", "into", "is", "are", "was",
        "were", "be", "been", "being", "do", "does", "did", "have", "has",
        "had", "i", "me", "my", "mine", "myself", "you", "your", "we", "us",
        "our", "it", "its", "this", "that", "these", "those", "what", "which",
        "who", "whom", "whose", "how", "when", "where", "why", "can", "could",
        "would", "should", "will", "shall", "may", "might", "must", "not",
        "no", "so", "than", "then", "them", "they", "he", "she", "his", "her",
        "from", "up", "out", "over", "know", "tell",
    }
)


def _normalize_currency(match: re.Match) -> str:
    value = match.group(1).replace(",", "")
    decimals = match.group(2) or ""
    if match.group(3):  # k/K multiplier on a currency amount: £150k → 150000
        value = str(int(float(value + decimals) * 1000))
        decimals = ""
    code = _CURRENCY_CODES[match.group(0).lstrip()[0]]
    return f"{value}{decimals} {code}"


def normalize_numbers(text: str) -> str:
    """Apply §3.1's enumerated number/unit rewrites to a raw string.

    - currency amounts → value then ISO code ("£2,000" → "2000 gbp");
    - k/K-suffixed currency amounts expanded ("£150k" → "150000 gbp");
    - plain thousands separators dropped ("150,000" → "150000");
    - everything else (percentages, dates, times) passes through untouched.
    Ranges split for free: each side normalizes independently and the
    hyphen/en-dash between them is a tokenizer separator.
    """
    text = _CURRENCY_RE.sub(_normalize_currency, text)
    return _THOUSANDS_RE.sub(lambda m: m.group(0).replace(",", ""), text)


def tokenize(text: str) -> list[str]:
    """unicode61-compatible raw tokens (no normalization, no lemmas)."""
    return _TOKEN_RE.findall(text)


def lemmatize(token: str) -> str:
    """simplemma lemma, EN→IT chain, lowercased (lowercasing is the pipeline's
    job so index- and query-side casing can never diverge)."""
    return simplemma.lemmatize(token, lang=_LANGS).lower()


def pipeline_tokens(text: str) -> list[str]:
    """The full §3.1 pipeline: normalize numbers → tokenize → lemmatize."""
    return [lemmatize(t) for t in tokenize(normalize_numbers(text))]


def normalize_for_index(text: str) -> str:
    """Pipeline output as the space-joined string stored in the FTS index.

    The same function renders gazetteer phrases at query time (gate R7): a
    phrase built here matches indexed text built here, by construction.
    """
    return " ".join(pipeline_tokens(text))


def content_words(text: str) -> set[str]:
    """Pipeline tokens minus stopwords — the lexical-coverage vocabulary.

    Used by the budgeter's coverage-relevance fallback (decision D, amended)
    and the keyword source's loose OR terms (§3.3 step 3), so both speak the
    same token language as the index.
    """
    return {t for t in pipeline_tokens(text) if t not in STOPWORDS}
