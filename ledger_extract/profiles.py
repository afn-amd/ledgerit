"""Per-bank configuration.

Most of the heavy lifting is done generically by ``engine.py`` (row-centric
parsing + running-balance type derivation). A profile only needs to say:

  engine          : "columnar" (cells already split into columns) or
                    "linear"   (Camelot collapsed the page to one column;
                                split each line on runs of >=2 spaces).
  header_keywords : lower-case tokens that all appear in the table's header
                    row. Used to locate where transactions begin and to drop
                    repeated headers. ``None`` => rely on the (date + balance)
                    guard alone (statements with no detectable header row).
  debit_col / credit_col : explicit column indices, only for headerless
                    formats where columns can't be mapped from a header.
"""

from __future__ import annotations

PROFILES = {
    "hdfc": {
        "engine": "columnar",
        "header_keywords": ["narration", "balance"],
    },
    "kotak": {
        "engine": "columnar",
        "header_keywords": ["narration", "balance"],
    },
    "canara": {
        "engine": "columnar",
        "header_keywords": ["particulars", "balance"],
    },
    "indusind": {
        "engine": "columnar",
        "header_keywords": ["particulars", "balance"],
    },
    "indian": {
        "engine": "columnar",
        "header_keywords": ["details", "balance"],
    },
    "union": {
        "engine": "columnar",
        "header_keywords": ["remarks", "balance"],
    },
    "uco": {
        # Like PNB, UCO wraps the particulars in rows *around* the amount line
        # (line 1 above the date row, continuation below).
        "engine": "columnar",
        "header_keywords": ["particulars", "balance"],
        "debit_col": 2,
        "credit_col": 3,
        "narration_around": True,
    },
    "pnb": {
        # PNB's camelot output has no header row; transactions start straight
        # after the account block. Columns are stable: date, withdrawal,
        # deposit, balance(Dr./Cr.), ..., narration. The narration wraps in
        # rows *around* the amount line (above and below it), so it needs the
        # proximity-based narration assignment.
        "engine": "columnar",
        "header_keywords": None,
        "debit_col": 1,
        "credit_col": 2,
        "narration_around": True,
    },
    "jk": {
        "engine": "linear",
        "header_keywords": None,
    },
    "idbi": {
        "engine": "linear",
        "header_keywords": None,
    },
    "generic": {
        "engine": "columnar",
        "header_keywords": None,
    },
}

# Header-cell keywords used to map columns to roles (columnar fallback path).
ROLE_KEYWORDS = {
    "debit": ["withdrawal", "withdrawl", "debit", "dr"],
    "credit": ["deposit", "credit", "cr"],
    "balance": ["balance"],
}


def get_profile(key):
    return PROFILES.get(key, PROFILES["generic"])
