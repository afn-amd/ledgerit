"""Identify the bank/format of a statement from its extracted text.

Detection uses bank names and (more reliably) IFSC prefixes. Returns a
profile key understood by ``profiles.get_profile``.
"""

from __future__ import annotations

import re

# (profile_key, [substrings that identify it]). Order matters: first match wins.
_SIGNATURES = [
    ("jk",       ["jammu and kashmir bank", "jaka0"]),
    ("idbi",     ["idbi bank", "customer account ledger"]),
    ("kotak",    ["kotak", "kkbk0"]),
    ("canara",   ["canara", "cnrb0"]),
    ("indusind", ["indusind", "indb0"]),
    ("indian",   ["indian bank", "idib0"]),
    ("pnb",      ["punjab national", "punb0"]),
    ("union",    ["union bank", "ubin0", "vyom"]),
    ("uco",      ["uco bank", "ucobank", "ucba0"]),
    ("hdfc",     ["hdfc bank", "hdfc0"]),
    ("cbi",      ["central bank of india", "cbin0"]),
]


# An IFSC code: 4-letter bank prefix, a literal '0', 6 alphanumerics.
_IFSC_CODE_RE = re.compile(r"\b([a-z]{4})0[a-z0-9]{6}\b")
# The label that precedes the account's OWN IFSC in the statement header
# ("RTGS/NEFT IFSC : HDFC0009357", "IFSC Code KKBK0006605", "IFS Code : ...").
_IFSC_LABEL_RE = re.compile(r"\bifsc?\b|\bifs\s*code\b")


def detect_bank_by_labeled_ifsc(text):
    """Profile key from an IFSC code that sits next to an "IFSC" label, or None.

    detect_bank() below scans whatever text the table extraction produced, and
    relies on the owner bank appearing before any counterparty. That breaks on
    layouts where the account-info header never reaches that text (Kotak
    current account: the branding is a logo image and the header block sits
    outside the detected table area), letting a beneficiary's IFSC deep in a
    narration win. The owner's IFSC, however, is always printed with an
    explicit label — "IFSC Code KKBK0006605", "RTGS/NEFT IFSC : HDFC0009357" —
    while narration IFSCs are bare ("NEFT CR-KKBK0000958-..."). So a labeled
    IFSC is an owner-bank signal independent of layout AND of the text order
    quirks of PDF extraction (PyMuPDF emits content order, not visual order).

    Returns None when no labeled IFSC is found, or when the labeled IFSC
    belongs to a bank with no profile — callers then fall back to detect_bank.
    """
    low = text.lower()
    for m in _IFSC_LABEL_RE.finditer(low):
        # The code follows the label, possibly across a ':' and/or a newline.
        code = _IFSC_CODE_RE.search(low, m.end(), m.end() + 40)
        if not code:
            continue
        prefix = code.group(1) + "0"
        for key, needles in _SIGNATURES:
            if prefix in needles:
                return key
        return None  # labeled IFSC of a bank we have no profile for
    return None


def detect_bank(full_text):
    """Return a profile key, or 'generic' if nothing matches.

    Only the top of the statement is inspected: the account-holder's own bank
    name and IFSC live in the header, whereas transaction narrations further
    down mention *other* banks' names and IFSC codes (UPI/NEFT counterparties)
    and would cause misdetection.
    """
    low = full_text[:4000].lower()
    best_key, best_pos = "generic", len(low) + 1
    for key, needles in _SIGNATURES:
        for n in needles:
            pos = low.find(n)
            if pos != -1 and pos < best_pos:
                best_key, best_pos = key, pos
    return best_key
