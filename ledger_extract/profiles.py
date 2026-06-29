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
        # On some HDFC statements Camelot collapses the 7 columns into a few
        # cells separated by '\n' (Date\nNarration, Ref\nValueDt\nWithdrawal...);
        # explode_newlines splits them back into per-column cells so the generic
        # parser sees a clean row. find_ref_tokens captures the alphanumeric
        # Chq./Ref.No. column (e.g. CBINR.../UTIBR...) as the Reference. The
        # narration is kept verbatim (nothing stripped from the description).
        "engine": "columnar",
        "header_keywords": ["narration", "balance"],
        "explode_newlines": True,
        "find_ref_tokens": True,
    },
    "kotak": {
        "engine": "columnar",
        "header_keywords": ["narration", "balance"],
    },
    "canara": {
        # No Chq./Ref. column (Date | Particulars | Deposits | Withdrawals |
        # Balance) -> don't fabricate a reference from the narration.
        "engine": "columnar",
        "header_keywords": ["particulars", "balance"],
        "no_reference": True,
    },
    "indusind": {
        # IndusInd has a dedicated Chq No/Ref No column (e.g. S30885416);
        # find_ref_tokens picks that standalone cell as the Reference instead
        # of a number embedded in the narration, and keeps it out of Description.
        "engine": "columnar",
        "header_keywords": ["particulars", "balance"],
        "find_ref_tokens": True,
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
        # No Chq./Ref. column (Date | Particulars | Withdrawals | Deposits |
        # Balance) -> don't fabricate a reference from the narration.
        "engine": "columnar",
        "header_keywords": ["particulars", "balance"],
        "debit_col": 2,
        "credit_col": 3,
        "narration_around": True,
        "no_reference": True,
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
        # No dedicated ref column; the ref lives inside the narration only.
        "no_reference": True,
    },
    "jk": {
        # No dedicated ref column (linear layout); don't fabricate one.
        "engine": "linear",
        "header_keywords": None,
        "no_reference": True,
    },
    "idbi": {
        # IDBI's ledger packs the instrument-number + transaction-id into a
        # cell right after the date (e.g. "M453009 25180"); drop_code_cells
        # keeps that out of the Description (the reference is still captured).
        "engine": "linear",
        "header_keywords": None,
        "drop_code_cells": True,
    },
    "cbi": {
        # Central Bank of India: Post Date | Value Date | Branch Code | Cheque
        # Number | Transaction Description | Debit | Credit | Balance. The
        # header spans two physical rows ("Value/Date", "Cheque/Number"); the
        # first line carries all the role words, so it's enough to key on. The
        # balance cell carries an explicit CR/DR marker ("76.14 CR").
        # drop_code_cells keeps the all-numeric Branch Code (e.g. "01353") and
        # Cheque Number columns out of the Description.
        "engine": "columnar",
        "header_keywords": ["transaction description", "balance"],
        "drop_code_cells": True,
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
