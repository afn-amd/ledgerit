"""Identify the bank/format of a statement from its extracted text.

Detection uses bank names and (more reliably) IFSC prefixes. Returns a
profile key understood by ``profiles.get_profile``.
"""

from __future__ import annotations

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
