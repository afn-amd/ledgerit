"""Parsing primitives shared by every bank profile.

The two hard problems in these statements are (1) Camelot's `stream` flavor
splits columns inconsistently across pages, and (2) every bank encodes the
transaction *type* differently. We solve both with a row-centric parser:

  * date    = first cell in the row that parses as a date
  * balance = last "money" cell in the row (signed: Cr = +, Dr = -)
  * amount/type are derived from the *running balance delta* (balance falls
    -> debit, rises -> credit). This is bank-agnostic and self-validating.
  * explicit Debit/Credit columns (when present) are used as a cross-check
    and as a fallback when the balance delta is unavailable.
"""

from __future__ import annotations

import re
from datetime import datetime

CANONICAL_COLUMNS = [
    "Date",
    "Description",
    "Reference",
    "Debit",
    "Credit",
    "Balance",
    "Type",
]

# --------------------------------------------------------------------------
# Dates
# --------------------------------------------------------------------------

_DATE_FORMATS = [
    "%d/%m/%Y",
    "%d/%m/%y",
    "%d-%m-%Y",
    "%d-%m-%y",
    "%d-%b-%Y",
    "%d-%b-%y",
    "%d %b %Y",
    "%d %b %y",
    "%d-%B-%Y",
    "%d %B %Y",
    "%d.%m.%Y",
    "%d.%m.%y",
]

# A token that *might* be a date (cheap pre-filter before strptime).
_DATE_LIKE = re.compile(
    r"^\d{1,2}[\s/.\-][A-Za-z0-9]{1,9}[\s/.\-]\d{2,4}$"
)


def parse_date(cell):
    """Return ISO date string (YYYY-MM-DD) if *cell* is a recognised date, else None."""
    if cell is None:
        return None
    s = str(cell).strip()
    if not s or not _DATE_LIKE.match(s):
        return None
    for fmt in _DATE_FORMATS:
        try:
            dt = datetime.strptime(s, fmt)
        except ValueError:
            continue
        # Two-digit years: treat 70-99 as 19xx, else 20xx (statements are recent).
        if dt.year < 1970:
            dt = dt.replace(year=dt.year + 100)
        return dt.strftime("%Y-%m-%d")
    return None


def parse_date_leading(cell):
    """Date for a whole cell, or for its first whitespace-delimited token.

    Some formats (e.g. IDBI ledger) pack 'TxnDate ValueDate TxnId' into one
    single-space-separated field; we still want the leading transaction date.
    """
    iso = parse_date(cell)
    if iso:
        return iso
    s = str(cell).strip()
    if " " in s:
        return parse_date(s.split(" ", 1)[0])
    return None


def first_date(cells):
    """Index and ISO value of the first date-like cell, or (None, None)."""
    for i, c in enumerate(cells):
        iso = parse_date_leading(c)
        if iso:
            return i, iso
    return None, None


# --------------------------------------------------------------------------
# Money
# --------------------------------------------------------------------------

# Whole cell is a number, optionally with a Dr/Cr marker. Handles:
#   "30,700.00"  "667996.16 Dr."  "50,000.00(Cr)"  "35117.37cr"
#   "443359.49Cr"  "650640.45 (Cr)"  "256.00"  "1499646"  "2022.1"
_MONEY_RE = re.compile(
    r"""^\s*
        (?P<neg>-)?
        (?P<num>\d{1,3}(?:,\d{2,3})*(?:\.\d+)?|\d+(?:\.\d+)?)
        \s*
        (?P<mark>\(?\s*(?:DR|CR)\s*\)?\.?)?
        \s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)


def parse_money(cell):
    """Parse a money cell.

    Returns (value, sign) where value is a non-negative float (magnitude) and
    sign is 'DR', 'CR' or None. Returns (None, None) if the cell is not money.
    """
    if cell is None:
        return None, None
    s = str(cell).strip()
    if not s:
        return None, None
    m = _MONEY_RE.match(s)
    if not m:
        return None, None
    raw = m.group("num")
    has_sep = "." in raw or "," in raw
    digits = raw.replace(",", "")
    # Bare integers that are really references / account numbers, not money:
    # leading-zero strings (e.g. 0000452954753350) or very long runs (>=12).
    if not has_sep and m.group("mark") is None:
        if len(digits) >= 12:
            return None, None
        if len(digits) > 1 and digits[0] == "0":
            return None, None
    try:
        val = float(digits)
    except ValueError:
        return None, None
    sign = None
    mark = m.group("mark")
    if mark:
        sign = "DR" if "d" in mark.lower() else "CR"
    elif m.group("neg"):
        sign = "DR"
    return val, sign


def is_money(cell):
    val, _ = parse_money(cell)
    return val is not None


def signed_balance(cell):
    """Signed balance value: Dr -> negative, Cr/none -> positive. None if not money."""
    val, sign = parse_money(cell)
    if val is None:
        return None
    return -val if sign == "DR" else val


def last_balance(cells):
    """Index and signed value of the running balance.

    Prefers the right-most money cell that carries a Dr/Cr marker (statement
    balances are signed, whereas stray numbers like cheque numbers are not);
    falls back to the right-most plain money cell when nothing is marked.
    """
    fallback = None
    for i in range(len(cells) - 1, -1, -1):
        val, sign = parse_money(cells[i])
        if val is None:
            continue
        if sign is not None:
            return i, (-val if sign == "DR" else val)
        if fallback is None:
            fallback = (i, val)
    return fallback if fallback is not None else (None, None)


# --------------------------------------------------------------------------
# Cell cleaning / classification
# --------------------------------------------------------------------------

# Page header / footer boilerplate. When a transaction row gets merged with a
# page footer (common on the last row of a page), everything from the first
# marker onwards is dropped so stray numbers (pincodes, GSTINs) aren't parsed
# as money.
_BOILERPLATE = [
    "hdfc bank limited",
    "contents of this statement",
    "registered office",
    "this is a computer generated",
    "constituent notifies",
    "date stamp",
    "jammu and kashmir bank",
    "idbi bank ltd",
    "state account branch gstn",
    "hdfc bank gstin",
    "printed by",
    # Union Bank (Vyom) page-footer legend / disclaimer
    "neft : national electronic fund transfer",
    "this is system generated",
    "unionbankofindia",
    "request to our customers",
    "bharat bill payment service",
]


def strip_boilerplate(text):
    """Truncate *text* at the earliest boilerplate marker."""
    low = text.lower()
    cut = len(text)
    for marker in _BOILERPLATE:
        i = low.find(marker)
        if i != -1 and i < cut:
            cut = i
    return text[:cut].strip()


def clean(cell):
    """Normalise a raw cell: collapse internal whitespace, drop boilerplate."""
    if cell is None:
        return ""
    s = str(cell)
    if s.strip().lower() == "nan":
        return ""
    s = re.sub(r"\s+", " ", s).strip()
    return strip_boilerplate(s)


def has_alpha(cell):
    return bool(re.search(r"[A-Za-z]", str(cell)))


def split_money_cells(cells):
    """Split a cell made of several space-separated money values into separate
    cells, e.g. '750.00 39,401.43' -> ['750.00', '39,401.43']. Camelot stream
    sometimes merges the amount and balance columns; this restores them so the
    right-most value is read as the balance."""
    out = []
    for c in cells:
        parts = str(c).split()
        if len(parts) >= 2 and all(is_money(p) for p in parts):
            out.extend(parts)
        else:
            out.append(c)
    return out


_REF_RE = re.compile(
    r"(?:[A-Z]{2,6}[-/:]\d{4,})"          # IMPS-..., NEFT:..., KPG-...
    r"|(?:\b[A-Z]\d{6,}\b)"               # S30885416
    r"|(?:\b\d{10,}\b)",                  # long numeric ref
)


def guess_reference(text):
    """Best-effort reference token pulled from description text."""
    if not text:
        return ""
    m = _REF_RE.search(text)
    return m.group(0) if m else ""


# --------------------------------------------------------------------------
# Noise detection
# --------------------------------------------------------------------------

_NOISE_SUBSTRINGS = [
    "opening balance",
    "closing balance",
    "brought forward",
    "carried forward",
    "b/f",
    "c/f",
    "statement of account",
    "statement summary",
    "page no",
    "grand total",
    "transaction total",
    "computer generated",
    "this is a system",
    "continued",
    "page total",
    "b/f balance",
    "debit amount",
    "credit amount",
    "tran id",
    "ledger print",
    "ledger report",
    "(cid:",
    "service outlet",
    "order by gl",
    "peg review",
    "gl sub head",
    # Union Bank (Vyom) page-footer legend / disclaimer rows
    "national electronic fund transfer",
    "real time gross settlement",
    "this is system generated",
    "unionbankofindia",
    "registered office",
]


def is_noise(joined_lower, header_keywords=None):
    """True if a row is a non-transaction line (totals, repeated headers, footers)."""
    for s in _NOISE_SUBSTRINGS:
        if s in joined_lower:
            return True
    if header_keywords and all(k in joined_lower for k in header_keywords):
        return True
    return False


# Phrases that mark the START of a page footer / disclaimer block. Once one is
# seen, every following continuation line is footer (legend, URL, address,
# scattered across columns) until the next dated transaction — so we suppress
# the whole block rather than trying to match each scattered fragment.
_FOOTER_MARKERS = [
    "national electronic fund transfer",
    "this is system generated",
    "this is a computer generated",
    "contents of this statement",
    "constituent notifies",
    "registered office",
    "request to our customers",
    # page-break header block (HDFC etc.): "Page No .: 2" is the first line,
    # followed by the repeated account-info block (Account Branch, Cust ID,
    # Account No, A/C Open Date, IFSC, Nomination, Statement From ...).
    "page no",
    "account branch",
    "statement of account",
]


def is_footer_start(joined_lower):
    return any(m in joined_lower for m in _FOOTER_MARKERS)


def is_opening(joined_lower):
    return (
        "opening balance" in joined_lower
        or "brought forward" in joined_lower
        or re.search(r"\bb/?f\b", joined_lower) is not None
    )


_TIME_RE = re.compile(r"\d{1,2}:\d{2}:\d{2}")


def is_report_timestamp(cells):
    """True if the leading date cell also carries a clock time (HH:MM:SS).

    These are report-generation banners (e.g. IDBI 'DD-MM-YYYY HH:MM:SS ... Page N'),
    not transactions — a real transaction's date cell never contains a time.
    """
    idx, iso = first_date(cells)
    return idx is not None and bool(_TIME_RE.search(cells[idx]))
