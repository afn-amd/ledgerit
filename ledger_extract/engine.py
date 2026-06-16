"""The bank-agnostic parsing engine.

Two passes:

1. ``parse_rows``  walks the raw table rows, identifies transaction rows
   (first cell that is a date + a right-most money cell that is the balance),
   merges continuation lines into the description, and records the raw signals
   needed to decide the type.

2. ``finalize``   assigns Debit / Credit / Type per transaction using this
   priority:
       a. an amount cell carrying an explicit (Dr)/(Cr) marker  (drift-proof);
       b. the running-balance delta, direction-aware              (drift-proof);
       c. explicit Debit/Credit columns                          (fallback).
"""

from __future__ import annotations

import re

from . import common as C
from .profiles import ROLE_KEYWORDS

_SPLIT_2SPACES = re.compile(r"\s{2,}")
_LONG_DIGITS = re.compile(r"^\d{10,}$")


def _linearize(row):
    """For 'linear' statements Camelot yields a single fat column; split it."""
    if len(row) == 1:
        return [c for c in _SPLIT_2SPACES.split(row[0].strip()) if c != ""]
    return row




def _build_colmap(header_cells):
    """Map role -> column index by scanning header cells for keywords."""
    colmap = {}
    for idx, cell in enumerate(header_cells):
        low = cell.lower()
        for role, kws in ROLE_KEYWORDS.items():
            if role in colmap:
                continue
            if any(k in low for k in kws):
                colmap[role] = idx
    return colmap


def _reference(cells, balance_idx):
    """Best-effort reference: a long digit cell or a ref-like token."""
    for i, c in enumerate(cells):
        if i == balance_idx:
            continue
        if _LONG_DIGITS.match(c):
            return c
    # fall back to a ref pattern anywhere in the row text
    return C.guess_reference(" ".join(cells))


def _description(cells, date_idx, balance_idx, ref):
    parts = []
    for i, c in enumerate(cells):
        if not c or i == date_idx or i == balance_idx:
            continue
        if C.parse_date(c):           # drop value-date columns
            continue
        if C.is_money(c):             # drop amount/balance numbers
            continue
        if c == ref:                  # drop the reference token
            continue
        parts.append(c)
    return " ".join(parts).strip()


def parse_rows(tables_rows, profile):
    """Return (transactions, opening_balance)."""
    if profile.get("narration_around"):
        return _parse_around(tables_rows, profile)

    engine = profile.get("engine", "columnar")
    header_keywords = profile.get("header_keywords")

    started = header_keywords is None
    colmap = {}
    txns = []
    current = None
    opening = None
    in_footer = False

    for table in tables_rows:
        for raw in table:
            cells = _linearize(raw) if engine == "linear" else list(raw)
            cells = C.split_money_cells([C.clean(c) for c in cells])
            joined = " ".join(cells).strip()
            if not joined:
                continue
            # skip separator / rule lines (dashes, underscores, etc.)
            if not any(ch.isalnum() for ch in joined):
                continue
            low = joined.lower()

            # Locate the start of the transaction table.
            if not started:
                if header_keywords and all(k in low for k in header_keywords):
                    started = True
                    colmap = _build_colmap(cells)
                continue

            # A page footer/disclaimer block, or a report-generation banner
            # (date cell carrying a clock time), starts a block we suppress
            # until the next dated transaction.
            if C.is_footer_start(low) or C.is_report_timestamp(cells):
                in_footer = True
                continue

            # Capture an opening / brought-forward balance to seed the deltas.
            if C.is_opening(low):
                _, bal = C.last_balance(cells)
                if bal is not None and opening is None:
                    opening = bal
                continue

            # Drop totals, footers and repeated headers.
            if C.is_noise(low, header_keywords):
                continue

            date_idx, iso = C.first_date(cells)
            bal_idx, bal = C.last_balance(cells)
            leading_date = (
                date_idx is not None
                and all(not cells[i] for i in range(date_idx))
            )

            if engine == "linear":
                # date + amount/balance may live on separate lines, so a new
                # transaction begins whenever a line *starts* with a date and
                # carries either money or a description (skips bare banners).
                rest = [c for k, c in enumerate(cells) if k != date_idx and c]
                has_content = any(C.is_money(c) for c in rest) or bal is not None \
                    or any(C.has_alpha(c) for c in rest)
                is_txn = bool(iso) and leading_date and has_content
            else:
                is_txn = (
                    iso is not None
                    and bal is not None
                    and date_idx is not None
                    and bal_idx is not None
                    and date_idx < bal_idx
                )

            if is_txn:
                in_footer = False  # a real transaction ends any footer block
                # amount cell with an explicit Dr/Cr marker (not the balance)
                amt_val = amt_sign = None
                for i, c in enumerate(cells):
                    if i == bal_idx:
                        continue
                    v, s = C.parse_money(c)
                    if v is not None and s is not None:
                        amt_val, amt_sign = v, s
                        break
                # explicit debit/credit columns (header- or profile-mapped),
                # tolerant of stream column drift: classify the amount cell by
                # the deposit-column boundary rather than an exact index.
                d_idx = profile.get("debit_col", colmap.get("debit"))
                c_idx = profile.get("credit_col", colmap.get("credit"))
                col_dr, col_cr = _classify_amount(cells, bal_idx, d_idx, c_idx)

                ref = _reference(cells, bal_idx)
                current = {
                    "Date": iso,
                    "Description": _description(cells, date_idx, bal_idx, ref),
                    "Reference": ref,
                    "Balance": bal,
                    "_amt_val": amt_val,
                    "_amt_sign": amt_sign,
                    "_col_dr": col_dr,
                    "_col_cr": col_cr,
                }
                txns.append(current)
            elif in_footer:
                # inside a footer block -> drop the line entirely
                continue
            elif current is not None and not leading_date:
                # continuation line -> extend description; pick up the balance,
                # the explicit Dr/Cr marker amount and Debit/Credit columns when
                # they were carried over to a wrapped line (e.g. JK Bank).
                extra = _description(cells, None, bal_idx, current["Reference"])
                if extra:
                    current["Description"] = (
                        current["Description"] + " " + extra
                    ).strip()
                if current["Balance"] is None and bal is not None:
                    current["Balance"] = bal
                if current["_amt_sign"] is None:
                    for i, c in enumerate(cells):
                        if i == bal_idx:
                            continue
                        v, s = C.parse_money(c)
                        if v is not None and s is not None:
                            current["_amt_val"], current["_amt_sign"] = v, s
                            break

    return txns, opening


def _parse_around(tables_rows, profile):
    """Parser for statements (PNB) whose narration wraps in rows *around* the
    amount line: line 1 sits in the row above the amount, continuation lines in
    the row(s) below it; short narrations sit on the amount row itself.

    Assignment rule for a narration-only row R between amount A (above) and
    amount B (below): R belongs to B (as B's first line) only when R sits
    immediately above B *and* B has no narration of its own; otherwise R is a
    continuation of the preceding amount A. The leading account-info block
    (before the first amount) is skipped.
    """
    import bisect

    d_idx = profile.get("debit_col")
    c_idx = profile.get("credit_col")
    header_keywords = profile.get("header_keywords")
    items = []          # ordered list of {"k": "amt"/"narr", ...}
    opening = None
    in_footer = False
    started = header_keywords is None
    seen_amount = False

    for table in tables_rows:
        for raw in table:
            cells = C.split_money_cells([C.clean(c) for c in raw])
            joined = " ".join(cells).strip()
            if not joined or not any(ch.isalnum() for ch in joined):
                continue
            low = joined.lower()
            if C.is_footer_start(low) or C.is_report_timestamp(cells):
                in_footer = True
                continue
            if C.is_opening(low):
                _, b = C.last_balance(cells)
                if b is not None and opening is None:
                    opening = b
                continue
            if C.is_noise(low, None):
                continue

            # When the format has a header row, use it to find the table start
            # (this keeps the first transaction's line-1, which sits above the
            # first amount row). Otherwise rely on the leading-block skip below.
            if not started:
                if header_keywords and all(k in low for k in header_keywords):
                    started = True
                continue

            date_idx, iso = C.first_date(cells)
            bal_idx, bal = C.last_balance(cells)
            leading = date_idx is not None and all(
                not cells[i] for i in range(date_idx)
            )
            is_amt = (
                iso is not None and bal is not None
                and leading and date_idx < bal_idx
            )
            if is_amt:
                in_footer = False
                seen_amount = True
                ref = _reference(cells, bal_idx)
                col_dr, col_cr = _classify_amount(cells, bal_idx, d_idx, c_idx)
                items.append({
                    "k": "amt", "Date": iso, "Balance": bal, "Reference": ref,
                    "text": _description(cells, date_idx, bal_idx, ref),
                    "_col_dr": col_dr, "_col_cr": col_cr, "extra": [],
                })
            elif in_footer:
                continue
            elif header_keywords is None and not seen_amount:
                continue  # PNB: skip the leading account-info block
            else:
                txt = _description(cells, None, bal_idx, "")
                if txt:
                    items.append({"k": "narr", "text": txt})

    amt_pos = [i for i, it in enumerate(items) if it["k"] == "amt"]
    for i, it in enumerate(items):
        if it["k"] != "narr":
            continue
        j = bisect.bisect_left(amt_pos, i)
        nxt = amt_pos[j] if j < len(amt_pos) else None
        prv = amt_pos[j - 1] if j > 0 else None
        if nxt is not None and nxt == i + 1 and not items[nxt]["text"]:
            target = nxt          # first line of the following amount
        elif prv is not None:
            target = prv          # continuation of the preceding amount
        else:
            target = nxt
        if target is not None:
            items[target]["extra"].append((i, it["text"]))

    txns = []
    for pos, it in enumerate(items):
        if it["k"] != "amt":
            continue
        parts = list(it["extra"])
        if it["text"]:
            parts.append((pos, it["text"]))
        parts.sort(key=lambda x: x[0])
        desc = " ".join(t for _, t in parts).strip()
        ref = it["Reference"] or C.guess_reference(desc)
        txns.append({
            "Date": it["Date"], "Description": desc, "Reference": ref,
            "Balance": it["Balance"], "_amt_val": None, "_amt_sign": None,
            "_col_dr": it["_col_dr"], "_col_cr": it["_col_cr"],
        })
    return txns, opening


def _col_val(cells, idx, balance_idx):
    if idx is None or idx == balance_idx or idx >= len(cells):
        return None
    v, _ = C.parse_money(cells[idx])
    return v


def _classify_amount(cells, bal_idx, d_idx, c_idx):
    """Drift-tolerant Debit/Credit from separate columns.

    Finds the transaction amount (the right-most money cell that isn't the
    balance) and decides debit vs credit by the deposit column's position:
    anything left of the deposit column is a withdrawal, at/right of it is a
    deposit. This survives Camelot placing the value one cell off from the
    header. Used only as a fallback (first row / zero delta).
    """
    if d_idx is None and c_idx is None:
        return None, None
    for i in range(len(cells) - 1, -1, -1):
        if i == bal_idx:
            continue
        v, _ = C.parse_money(cells[i])
        if not v:           # skip blanks and 0.00 placeholders in the empty column
            continue
        if c_idx is not None and i >= c_idx:
            return None, v          # credit
        if d_idx is not None and (c_idx is None or i < c_idx):
            return v, None          # debit
        return None, None
    return None, None


def _direction_ascending(txns):
    asc = desc = 0
    prev = None
    for t in txns:
        d = t["Date"]
        if prev is not None and d != prev:
            if d > prev:
                asc += 1
            else:
                desc += 1
        prev = d
    return asc >= desc


def finalize(txns, opening):
    """Fill Debit / Credit / Type and clean up helper fields."""
    if not txns:
        return txns

    ascending = _direction_ascending(txns)
    n = len(txns)

    for i, t in enumerate(txns):
        debit = credit = None
        ttype = None

        # (a) explicit marker on the amount cell
        if t["_amt_sign"] is not None and t["_amt_val"] is not None:
            ttype = t["_amt_sign"]
            if ttype == "DR":
                debit = t["_amt_val"]
            else:
                credit = t["_amt_val"]
        else:
            # (b) running-balance delta (direction-aware)
            if ascending:
                prev = txns[i - 1]["Balance"] if i > 0 else opening
            else:
                prev = txns[i + 1]["Balance"] if i < n - 1 else opening
            bal = t["Balance"]
            delta = None
            if bal is not None and prev is not None:
                delta = round(bal - prev, 2)

            if delta is not None and abs(delta) >= 0.005:
                if delta > 0:
                    credit, ttype = abs(delta), "CR"
                else:
                    debit, ttype = abs(delta), "DR"
            # (c) explicit debit/credit columns
            elif t["_col_dr"]:
                debit, ttype = t["_col_dr"], "DR"
            elif t["_col_cr"]:
                credit, ttype = t["_col_cr"], "CR"

        t["Debit"] = round(debit, 2) if debit is not None else None
        t["Credit"] = round(credit, 2) if credit is not None else None
        t["Type"] = ttype
        t["Balance"] = round(t["Balance"], 2) if t["Balance"] is not None else None

    # strip helper fields
    for t in txns:
        for k in ("_amt_val", "_amt_sign", "_col_dr", "_col_cr"):
            t.pop(k, None)

    # safety net: drop rows that carry no financial information at all (stray
    # page banners / headers that slipped past the noise filter).
    txns = [
        t
        for t in txns
        if t["Balance"] is not None
        or t["Debit"] is not None
        or t["Credit"] is not None
    ]
    return txns
