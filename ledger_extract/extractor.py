"""Camelot `stream` extraction layer.

Reads a PDF (decrypting first if needed) and returns the list of raw page
tables as cleaned row-lists, plus a flat text dump used for bank detection.
"""

from __future__ import annotations

import os

import re

import camelot
from PyPDF2 import PdfReader

from . import common as C

_REFISH = re.compile(r"^[A-Za-z0-9/\-]{8,}$")


def _row_tokens(cells):
    """Identifying tokens of a row: dates, money values and reference-like
    strings. Position-independent, so the same transaction split into different
    columns by camelot still yields the same token set."""
    # Explode any newline-merged cells so a date/money/ref packed behind a
    # '\n' (HDFC) is still tokenised for dedup coverage.
    flat = []
    for c in cells:
        flat.extend(str(c).split("\n"))
    toks = set()
    for c in C.split_money_cells(flat):
        c = c.strip()
        if not c:
            continue
        iso = C.parse_date_leading(c)
        if iso:
            toks.add("D:" + iso)
            continue
        v, _ = C.parse_money(c)
        if v is not None:
            toks.add("M:%.2f" % v)
            continue
        if _REFISH.match(c) and any(ch.isdigit() for ch in c):
            toks.add("R:" + c)
    return toks


def _dedup_tables(cam_tables):
    """Drop redundant tables. Camelot's stream often emits, per page, a main
    transaction table plus smaller fragment/duplicate tables whose content is
    already covered by the main one (HDFC fragments, UCO's duplicated tables).
    Per page, largest first, a table is dropped when >60% of its content rows
    are token-covered by tables already kept for that page."""
    by_page = {}
    order = []
    for t in cam_tables:
        if t.page not in by_page:
            by_page[t.page] = []
            order.append(t.page)
        by_page[t.page].append(t)

    keep_ids = set()
    for page in order:
        tabs = by_page[page]
        ranked = sorted(tabs, key=lambda t: -(t.df.shape[0] * t.df.shape[1]))
        seen = set()
        for t in ranked:
            rows = [[_raw_clean(c) for c in r] for r in t.df.values.tolist()]
            content = [tk for tk in (_row_tokens(r) for r in rows) if tk]
            if not content:
                continue
            covered = sum(1 for tk in content if tk <= seen)
            if covered / len(content) > 0.6:
                continue  # redundant fragment / duplicate
            keep_ids.add(id(t))
            for tk in content:
                seen |= tk
    # preserve camelot's original (page, position) order among kept tables
    return [t for t in cam_tables if id(t) in keep_ids]


def _raw_clean(cell):
    """Light clean: trim ends, PRESERVE internal runs of spaces (the 'linear'
    engine relies on >=2-space gaps as column boundaries) AND preserve newlines.
    Some layouts -- notably HDFC -- pack several logical columns into one cell
    separated by '\\n'; the engine explodes those for such profiles, while every
    other path collapses them again via common.clean()."""
    if cell is None:
        return ""
    s = str(cell)
    if s.strip().lower() == "nan":
        return ""
    s = s.replace("\r", "\n")
    return s.strip()


def _is_encrypted(pdf_path):
    try:
        return PdfReader(pdf_path).is_encrypted
    except Exception:
        return False


def read_tables(pdf_path, password=None, pages="all"):
    """Run Camelot stream and return (tables_as_rows, full_text).

    tables_as_rows : list of tables; each table is a list of rows; each row is
                     a list of cleaned cell strings.
    full_text      : every cell joined with spaces/newlines (for detection).
    """
    kwargs = dict(pages=pages, flavor="stream")
    if password:
        kwargs["password"] = password

    tables = _dedup_tables(camelot.read_pdf(pdf_path, **kwargs))

    tables_as_rows = []
    text_parts = []
    for t in tables:
        df = t.df
        if df.empty:
            continue
        rows = []
        for raw_row in df.values.tolist():
            cells = [_raw_clean(c) for c in raw_row]
            rows.append(cells)
            text_parts.append(re.sub(r"\s+", " ", " ".join(cells)))
        if rows:
            tables_as_rows.append(rows)

    return tables_as_rows, "\n".join(text_parts)


def resolve_password(pdf_path, password_map=None):
    """Return the password for *pdf_path* if it is encrypted, else None."""
    if not _is_encrypted(pdf_path):
        return None
    name = os.path.basename(pdf_path)
    if password_map and name in password_map:
        return password_map[name]
    # Many of these files use the digits embedded in the filename as the
    # password (e.g. HDFC_54212352.pdf -> 54212352).
    import re

    digits = re.findall(r"\d{6,}", name)
    return digits[0] if digits else None
