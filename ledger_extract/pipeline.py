"""Orchestration: PDF -> normalized transactions DataFrame + validation."""

from __future__ import annotations

import os

import pandas as pd

from .common import CANONICAL_COLUMNS
from .detect import detect_bank
from .engine import _direction_ascending, finalize, parse_rows
from .extractor import read_tables, resolve_password
from .profiles import get_profile


def extract_statement(pdf_path, password_map=None):
    """Process one PDF. Returns (df, meta) where meta carries diagnostics."""
    password = resolve_password(pdf_path, password_map)
    tables_rows, full_text = read_tables(pdf_path, password=password)

    bank = detect_bank(full_text)
    profile = get_profile(bank)

    txns, opening = parse_rows(tables_rows, profile)
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
