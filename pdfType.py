"""Decide whether a bank-statement PDF should be extracted with Camelot's
``lattice`` flavor or its ``stream`` flavor.

The distinction is *not* "does the page have any drawn lines" -- many
``stream``-style statements (HDFC, Indian, Union) carry plenty of rectangles
used for shading, header boxes or a single framed summary. The real question is
whether the **transaction rows themselves sit inside a ruled grid** that cuts
the content into columns.

The discriminator used here runs Camelot ``lattice`` (which detects ruling
lines via image morphology, exactly as the real extractor would) and looks at
the largest table it finds:

* ``lattice`` PDFs -> a tall table whose text is genuinely spread across
  several columns (date | narration | debit | credit | balance ...).
* ``stream`` PDFs -> either no grid table at all, or a degenerate one. The
  instructive case is *Union Bank*: lattice "finds" a 22-row, 6-column table at
  100% accuracy, but every row's text collapses into a single cell because the
  page only has horizontal rules -- the vertical lines don't segment anything.
  That shows up as ``populated_columns == 1``.

So a PDF is classified ``lattice`` only when the best grid table has both
enough rows AND enough genuinely-populated columns. Validated on the bundled
``input/`` (stream) and ``input2/`` (lattice) corpora: 47/47 correct.
"""

from __future__ import annotations

import os
import re
import sys
import warnings
from dataclasses import dataclass

warnings.filterwarnings("ignore")

import camelot
from PyPDF2 import PdfReader

# --- decision thresholds -------------------------------------------------
# The largest lattice table must have at least this many rows ...
MIN_ROWS = 3
# ... and at least this many columns that are actually populated (non-empty in
# >= COLUMN_FILL_RATIO of rows). Together these reject framed summary boxes
# (too few rows) and single-column "horizontal rules only" grids (too few
# populated columns, e.g. Union Bank).
MIN_POPULATED_COLUMNS = 3
COLUMN_FILL_RATIO = 0.30

# Passwords that the filename-digit heuristic cannot recover. Extend as needed.
KNOWN_PASSWORDS: dict[str, str] = {}

# How many leading pages to sample. Page 1 is usually representative; sampling a
# few guards against a statement whose first page is mostly a header.
SAMPLE_PAGES = 3


@dataclass
class PdfTypeResult:
    """Outcome of classifying one PDF."""

    path: str
    category: str          # "lattice" or "stream"
    flavor: str            # camelot flavor to use: same as category
    rows: int              # rows in the best lattice table
    columns: int           # columns in the best lattice table
    populated_columns: int # columns populated in >= COLUMN_FILL_RATIO of rows
    reason: str

    @property
    def has_table_structure(self) -> bool:
        return self.category == "lattice"


def _resolve_password(path: str, password: str | None) -> str | None:
    """Return a password for *path* if it is encrypted, else None.

    Precedence: explicit argument > known-password map > digits embedded in the
    filename (e.g. ``HDFC_54212352.pdf`` -> ``54212352``).
    """
    try:
        if not PdfReader(path).is_encrypted:
            return None
    except Exception:
        # Unreadable header; let camelot surface the real error later.
        return password
    if password:
        return password
    name = os.path.basename(path)
    if name in KNOWN_PASSWORDS:
        return KNOWN_PASSWORDS[name]
    digits = re.findall(r"\d{6,}", name)
    return digits[0] if digits else ""


def _page_count(path: str, password: str | None) -> int:
    try:
        reader = PdfReader(path)
        if reader.is_encrypted:
            reader.decrypt(password or "")
        return max(1, len(reader.pages))
    except Exception:
        return 1


def _best_grid_table(path: str, password: str | None, pages: str):
    """Run Camelot lattice and return (rows, columns, populated_columns) for the
    table that maximises rows * populated_columns. (0, 0, 0) if none found."""
    kwargs = dict(pages=pages, flavor="lattice")
    if password is not None:
        kwargs["password"] = password

    tables = camelot.read_pdf(path, **kwargs)

    best = (0, 0, 0)
    best_score = -1
    for tbl in tables:
        df = tbl.df
        n_rows, n_cols = df.shape
        if n_rows == 0 or n_cols == 0:
            continue
        threshold = max(1, COLUMN_FILL_RATIO * n_rows)
        populated = 0
        for ci in range(n_cols):
            filled = sum(1 for v in df.iloc[:, ci] if str(v).strip())
            if filled >= threshold:
                populated += 1
        score = n_rows * populated
        if score > best_score:
            best_score = score
            best = (n_rows, n_cols, populated)
    return best


def classify_pdf(path: str, password: str | None = None) -> PdfTypeResult:
    """Classify *path* as ``"lattice"`` (has a ruled tabular structure) or
    ``"stream"`` (no usable grid).

    Raises FileNotFoundError if the file is missing. Decryption / parsing
    failures are raised as RuntimeError with a clear message.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    pw = _resolve_password(path, password)
    n_pages = _page_count(path, pw)
    pages = "1-%d" % min(SAMPLE_PAGES, n_pages)

    try:
        rows, columns, populated = _best_grid_table(path, pw, pages)
    except Exception as exc:
        msg = str(exc)
        if "not been decrypted" in msg or "decrypt" in msg.lower():
            raise RuntimeError(
                f"{os.path.basename(path)}: encrypted and no valid password "
                f"supplied (pass password=...)."
            ) from exc
        raise RuntimeError(f"{os.path.basename(path)}: lattice probe failed: {msg}") from exc

    is_lattice = rows >= MIN_ROWS and populated >= MIN_POPULATED_COLUMNS
    if is_lattice:
        reason = (f"grid table with {rows} rows across {populated} populated "
                  f"columns -> ruled tabular structure")
    elif rows == 0:
        reason = "no ruled grid table detected -> stream"
    elif populated < MIN_POPULATED_COLUMNS:
        reason = (f"grid found but text collapses into {populated} column(s) "
                  f"(rules don't segment columns) -> stream")
    else:
        reason = f"grid table too small ({rows} rows) -> stream"

    category = "lattice" if is_lattice else "stream"
    return PdfTypeResult(
        path=path,
        category=category,
        flavor=category,
        rows=rows,
        columns=columns,
        populated_columns=populated,
        reason=reason,
    )


def _main(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Classify a bank-statement PDF as lattice (ruled table) or "
                    "stream (no usable grid).")
    parser.add_argument("paths", nargs="+",
                        help="PDF file(s) or folder(s) to classify")
    parser.add_argument("--password", default=None,
                        help="password for encrypted PDFs")
    args = parser.parse_args(argv)

    targets: list[str] = []
    for p in args.paths:
        if os.path.isdir(p):
            for name in sorted(os.listdir(p)):
                if name.lower().endswith(".pdf"):
                    targets.append(os.path.join(p, name))
        else:
            targets.append(p)

    exit_code = 0
    for path in targets:
        try:
            res = classify_pdf(path, password=args.password)
            print(f"{os.path.basename(path):45} {res.category.upper():8} "
                  f"(rows={res.rows}, pop_cols={res.populated_columns}) "
                  f"-- {res.reason}")
        except Exception as exc:
            exit_code = 1
            print(f"{os.path.basename(path):45} {'ERROR':8} -- {exc}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
