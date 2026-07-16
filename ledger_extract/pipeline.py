"""Orchestration: PDF -> normalized transactions DataFrame + validation."""

from __future__ import annotations

import os

import fitz  # PyMuPDF — independent text-layer read used to spot under-capture
import pandas as pd

from .common import CANONICAL_COLUMNS, parse_date
from .detect import detect_bank, detect_bank_by_labeled_ifsc
from .engine import _direction_ascending, finalize, parse_rows
from .extractor import read_tables, resolve_password
from .profiles import get_profile


def _text_layer_txn_estimate(pdf_path, password):
    """Independent lower-bound on the number of transaction rows, read straight
    from the PDF text layer with PyMuPDF (bypassing Camelot entirely).

    Every transaction row starts with a leading date, so we count text lines
    whose first one-to-three whitespace tokens parse as a date. This is used
    only to detect when Camelot's default stream pass truncated a tall-row
    layout and silently dropped transactions -- see extract_statement. Returns 0
    on any error, which simply disables the under-capture retry (never wrong,
    just no help).
    """
    try:
        doc = fitz.open(pdf_path)
        try:
            if doc.needs_pass:
                doc.authenticate(password or "")
            n = 0
            for page in doc:
                for line in page.get_text("text").splitlines():
                    w = line.strip().split()
                    if w and any(
                        len(w) >= k and parse_date(" ".join(w[:k]))
                        for k in (3, 2, 1)
                    ):
                        n += 1
            return n
        finally:
            doc.close()
    except Exception:
        return 0


def _page1_text(pdf_path, password):
    """Page-1 text layer, read directly with PyMuPDF.

    Feeds detect_bank_by_labeled_ifsc: read_tables' full_text only contains the
    cells of the tables Camelot found — on layouts where the account-info
    header sits outside the table area (e.g. Kotak current account, whose
    branding is a logo image and whose own IFSC lives in that header block),
    the owner bank is invisible to detect_bank and the first counterparty IFSC
    in a narration wins (that Kotak file detected as "hdfc" off a beneficiary's
    HDFC0000001). The page-1 text layer always carries the header block, and
    the labeled-IFSC scan is order-independent, so it doesn't matter that
    PyMuPDF emits content order rather than visual order. Returns "" on any
    error so a probe failure never changes the old behaviour.
    """
    try:
        doc = fitz.open(pdf_path)
        try:
            if doc.needs_pass:
                doc.authenticate(password or "")
            return doc[0].get_text("text") if doc.page_count else ""
        finally:
            doc.close()
    except Exception:
        return ""


def extract_statement(pdf_path, password_map=None):
    """Process one PDF. Returns (df, meta) where meta carries diagnostics."""
    password = resolve_password(pdf_path, password_map)
    # Owner bank from the labeled IFSC in the page-1 header, when present.
    # This outranks detect_bank's earliest-substring scan, which a counterparty
    # IFSC can hijack when the account-info header is missing from full_text.
    header_bank = detect_bank_by_labeled_ifsc(_page1_text(pdf_path, password))

    def _read_and_parse(edge_tol):
        tables_rows, full_text = read_tables(
            pdf_path, password=password, edge_tol=edge_tol
        )
        bank = header_bank or detect_bank(full_text)
        txns, opening = parse_rows(tables_rows, get_profile(bank))
        # Profile-mismatch guard: a bank profile is tuned to one layout, and
        # the same bank ships others (Kotak current-account says "Description"
        # where the profile's header keyword expects "Narration"), so a correct
        # detection can still parse nothing. Zero rows today means a guaranteed
        # "couldn't convert", so falling back to the layout-agnostic generic
        # profile can only rescue files, never change a working one.
        if not txns and bank != "generic":
            g_txns, g_opening = parse_rows(tables_rows, get_profile("generic"))
            if g_txns:
                txns, opening = g_txns, g_opening
        return txns, opening, bank

    txns, opening, bank = _read_and_parse(None)

    # Some layouts (e.g. Central Bank of India) confine the transaction grid to
    # a narrow horizontal band that Camelot's default stream column-grouping
    # drops entirely, yielding zero transactions. Retry once with a wider
    # edge tolerance before giving up. Done only on an empty first pass so files
    # that already extract are untouched (the wider grouping can over-merge
    # columns on other layouts).
    if not txns:
        txns, opening, bank = _read_and_parse(500)
    else:
        # Under-capture guard. On tall-row layouts with wide vertical gaps
        # between transactions (e.g. Indian Bank, whose rows carry a multi-line
        # narration and INR-prefixed amounts), Camelot's default stream table
        # detection bounds each page's table to just its first transaction and
        # drops the rest. The running-balance parser still reconciles perfectly
        # -- each surviving row's amount is derived from the balance delta, so
        # the skipped transactions are silently absorbed into the next captured
        # row -- which hides the loss. Compare the parsed count against an
        # independent text-layer transaction count; if the default pass came up
        # materially short, retry with the wide edge tolerance and adopt it only
        # when it genuinely captures more rows. A file that already extracts
        # fully sits at ~100% of the estimate, so this adds no work and cannot
        # change its result.
        expected = _text_layer_txn_estimate(pdf_path, password)
        if expected and len(txns) < 0.9 * expected:
            wide = _read_and_parse(500)
            if len(wide[0]) > len(txns):
                txns, opening, bank = wide

    txns = finalize(txns, opening)

    df = pd.DataFrame(txns, columns=CANONICAL_COLUMNS) if txns else pd.DataFrame(
        columns=CANONICAL_COLUMNS
    )

    meta = {
        "file": os.path.basename(pdf_path),
        "bank": bank,
        "encrypted": password is not None,
        "opening_balance": opening,
        **_validate(df, opening),
    }
    return df, meta


def _validate(df, opening):
    """Sanity-check the parse: counts, totals and running-balance continuity."""
    n = len(df)
    n_dr = int((df["Type"] == "DR").sum())
    n_cr = int((df["Type"] == "CR").sum())
    n_untyped = int(df["Type"].isna().sum()) if n else 0
    sum_dr = float(df["Debit"].fillna(0).sum()) if n else 0.0
    sum_cr = float(df["Credit"].fillna(0).sum()) if n else 0.0

    # Continuity: for each row, prev_balance +/- amount == balance.
    breaks = 0
    closing = None
    if n:
        ascending = _direction_ascending(df.to_dict("records"))
        bals = df["Balance"].tolist()
        debs = df["Debit"].fillna(0).tolist()
        creds = df["Credit"].fillna(0).tolist()
        seq = list(range(n)) if ascending else list(range(n - 1, -1, -1))
        prev = opening
        for idx in seq:
            if bals[idx] is None:
                continue
            expected_delta = creds[idx] - debs[idx]
            if prev is not None:
                actual = round(bals[idx] - prev, 2)
                if abs(actual - round(expected_delta, 2)) > 1.0:
                    breaks += 1
            prev = bals[idx]
        closing = df["Balance"].iloc[-1] if ascending else df["Balance"].iloc[0]

    return {
        "n_txns": n,
        "n_dr": n_dr,
        "n_cr": n_cr,
        "n_untyped": n_untyped,
        "sum_debit": round(sum_dr, 2),
        "sum_credit": round(sum_cr, 2),
        "continuity_breaks": breaks,
        "closing_balance": closing,
    }
