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
import unicodedata
import warnings
from dataclasses import dataclass

warnings.filterwarnings("ignore")

import camelot
import fitz  # PyMuPDF — cheap vector/text geometry used by the fast pre-check
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


# ===========================================================================
# Fast pre-check (PyMuPDF) — skip the expensive Camelot lattice probe entirely
# for the common case.
# ===========================================================================
#
# classify_pdf() runs a full Camelot *lattice* extraction (Ghostscript renders
# every sampled page to an image, then OpenCV morphology finds the rules) just
# to answer "grid or not". That render is the single most expensive step in the
# whole upload, and its result is thrown away before the real extraction runs.
#
# Most uploads are structure-less "stream" statements with no ruled transaction
# grid at all. PyMuPDF can see that in milliseconds by reading the page's vector
# drawing operators and text positions directly — no rendering. So we do a cheap
# probe first:
#
#   * If the page has NO band that even *looks* like a ruled grid (a tall run of
#     horizontal rules that also carries >= MIN_PROBE_COLS interior vertical
#     column dividers), and the page has real extractable text, we are confident
#     it is "stream" and skip Camelot outright.
#   * Anything that *might* be a grid — or a low-text page that could be a
#     scanned image PyMuPDF can't see into — is deferred to the unchanged
#     Camelot classify_pdf() for an identical, authoritative verdict.
#
# PyMuPDF (vector) and Camelot (raster morphology) genuinely disagree on a few
# borderline layouts (e.g. Union Bank draws column lines Camelot's raster pass
# ignores), so the probe never *overrides* Camelot on ambiguous files — it only
# short-circuits the clearly-stream majority. Validated to reproduce
# classify_pdf's decision on the bundled input/ + input2/ corpora exactly.

_PROBE_LINE_TOL = 2.0        # max thickness (pt) for a segment to count as a rule
_PROBE_MIN_LEN = 20.0        # min length (pt) for a segment to count at all
_PROBE_BAND_GAP = 40.0       # h-rule gap (pt) that splits one table band from the next
_PROBE_DEDUP_TOL = 3.0       # merge coordinates within this many pt
_PROBE_BIG_ROWS = 10         # a band this tall is a candidate transaction grid
_PROBE_COL_COV = 0.10        # a v-cluster covering this fraction of a band = a column sep
_PROBE_MIN_COLS = 3          # >= this many interior column seps => defer to Camelot
_PROBE_MIN_WORDS = 40        # fewer words across the sample => maybe scanned => defer


def _probe_segments(page):
    """Collect axis-aligned rule segments from *page*'s vector drawings.

    Returns (h_lines, v_lines) where h_lines = [(y, x0, x1)] and
    v_lines = [(x, y0, y1)]. Rectangles are decomposed into their four edges so
    a table drawn as boxes still yields its rules.
    """
    h_lines, v_lines = [], []

    def add(x0, y0, x1, y1):
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        if dy <= _PROBE_LINE_TOL and dx >= _PROBE_MIN_LEN:
            h_lines.append(((y0 + y1) / 2.0, min(x0, x1), max(x0, x1)))
        elif dx <= _PROBE_LINE_TOL and dy >= _PROBE_MIN_LEN:
            v_lines.append(((x0 + x1) / 2.0, min(y0, y1), max(y0, y1)))

    for d in page.get_drawings():
        for it in d["items"]:
            if it[0] == "l":
                add(it[1].x, it[1].y, it[2].x, it[2].y)
            elif it[0] == "re":
                r = it[1]
                add(r.x0, r.y0, r.x1, r.y0)   # top
                add(r.x0, r.y1, r.x1, r.y1)   # bottom
                add(r.x0, r.y0, r.x0, r.y1)   # left
                add(r.x1, r.y0, r.x1, r.y1)   # right
    return h_lines, v_lines


def _probe_covered(intervals):
    """Total length covered by a set of (lo, hi) intervals (their union)."""
    if not intervals:
        return 0.0
    intervals = sorted(intervals)
    total = 0.0
    lo, hi = intervals[0]
    for a, b in intervals[1:]:
        if a <= hi:
            hi = max(hi, b)
        else:
            total += hi - lo
            lo, hi = a, b
    return total + (hi - lo)


def _probe_page_has_grid(page):
    """True if *page* shows a band that *might* be a ruled transaction grid and so
    must be confirmed by Camelot. A band qualifies on EITHER of two independent
    signals (logical OR, deliberately conservative):

      (i)  it is tall — >= _PROBE_BIG_ROWS rows of horizontal rules; or
      (ii) it is multi-column — >= _PROBE_MIN_COLS interior vertical dividers.

    Both signals are needed because the two failure modes are geometrically
    distinct. A real grid may present a tall band whose vertical rules are too
    short/light to detect (mode i catches it on row count alone), or a short band
    that is nonetheless cleanly segmented into columns (mode ii catches it on
    column count alone). Requiring BOTH — as an earlier revision did — let a
    tall-but-column-less grid (e.g. Axis Bank) escape as 'stream'. Since the probe
    only ever *skips* Camelot on a confident 'stream', a false 'no-grid' here is a
    routing error, so we err toward deferring.

    A column divider is an x-cluster of vertical segments (grid tables often draw
    each cell's border separately) whose combined y-coverage spans at least
    _PROBE_COL_COV of the band height. The two outermost clusters (the frame) are
    excluded so a plain boxed block isn't mistaken for a multi-column grid.
    """
    h_lines, v_lines = _probe_segments(page)
    if not h_lines or not v_lines:
        return False

    ys = sorted({round(y, 1) for y, _, _ in h_lines})
    bands = [[ys[0]]]
    for y in ys[1:]:
        if y - bands[-1][-1] > _PROBE_BAND_GAP:
            bands.append([y])
        else:
            bands[-1].append(y)

    for b in bands:
        rows = len(b) - 1
        # Signal (i): a tall ruled band on its own is enough to defer.
        if rows >= _PROBE_BIG_ROWS:
            return True
        top, bot = b[0], b[-1]
        rh = bot - top
        if rh <= 0:
            continue
        # Signal (ii): count interior column dividers within this (shorter) band.
        vs = sorted(v_lines, key=lambda s: s[0])
        clusters = []                       # [x_first, [(y0, y1), ...]]
        for x, y0, y1 in vs:
            yy0, yy1 = max(y0, top), min(y1, bot)   # clip to this band
            if yy1 - yy0 <= 0:
                continue
            if clusters and abs(x - clusters[-1][0]) <= _PROBE_DEDUP_TOL:
                clusters[-1][1].append((yy0, yy1))
            else:
                clusters.append([x, [(yy0, yy1)]])
        if len(clusters) < 2:
            continue
        xs = [c[0] for c in clusters]
        frame = {round(min(xs)), round(max(xs))}
        interior = sum(
            1 for c in clusters
            if _probe_covered(c[1]) / rh >= _PROBE_COL_COV
            and round(c[0]) not in frame
        )
        if interior >= _PROBE_MIN_COLS:
            return True
    return False


def _probe_is_confident_stream(path, password):
    """Cheap PyMuPDF verdict: return True only when we are confident the PDF is
    a structure-less 'stream' statement and Camelot can be skipped. On any doubt
    (possible grid, low-text/maybe-scanned page, or any error) return False so
    the caller defers to Camelot.
    """
    try:
        pw = _resolve_password(path, password)
        doc = fitz.open(path)
        try:
            if doc.needs_pass:
                doc.authenticate(pw or "")
            sample = min(SAMPLE_PAGES, doc.page_count)
            words = 0
            for i in range(sample):
                page = doc[i]
                words += len(page.get_text("words"))
                if _probe_page_has_grid(page):
                    return False        # might be a grid -> let Camelot decide
            if words < _PROBE_MIN_WORDS:
                return False            # little text -> maybe scanned -> Camelot
            return True                 # confident: no grid, real text -> stream
        finally:
            doc.close()
    except Exception:
        return False                    # never let the probe break classification


# ===========================================================================
# Scanned-PDF detection (PyMuPDF) — reject image-only statements early.
# ===========================================================================
#
# A "scanned" statement is a photograph/scan of paper wrapped in a PDF: each
# page is one big raster image with no real text layer, so every text-based
# extractor (Camelot lattice, ledger_extract stream) returns nothing usable.
# We want to catch these BEFORE the expensive extraction and tell the user
# plainly, rather than failing later with a generic "no transactions" message.
#
# The signal is deliberately simple and cheap (no rendering, no OCR):
#
#   * A page with a real text layer (>= _SCAN_MIN_CHARS characters of actual
#     selectable text) is NOT a scan — even an OCR'd/searchable scan is fine
#     because its text can be extracted like any digital PDF.
#   * A page with almost no text that is instead covered by a large raster
#     image (an image block spanning >= _SCAN_IMAGE_AREA_RATIO of the page)
#     is an image page.
#
# Only the first page is inspected: a scanned statement is scanned throughout,
# so page 1 is representative, and looking at a single page keeps the probe
# essentially free. A first page that carries a real text layer is left for the
# normal extractors to attempt.

_SCAN_MIN_CHARS = 100          # >= this many real chars on a page => has a text layer
_SCAN_IMAGE_AREA_RATIO = 0.5   # an image covering >= this fraction of the page => image page


def _page_is_image_scan(page) -> bool:
    """True if *page* has no usable text layer but is dominated by a raster
    image — i.e. it looks like a scanned/photographed page."""
    if len(page.get_text("text").strip()) >= _SCAN_MIN_CHARS:
        return False  # real (or OCR'd) text present -> extractable, not a scan

    page_area = abs(page.rect.width * page.rect.height)
    if page_area <= 0:
        return False

    # Image blocks carry a bbox in the "dict" extraction (type == 1). Take the
    # largest single image's coverage — a scan is one big page-sized image.
    largest = 0.0
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") == 1:  # image block
            x0, y0, x1, y1 = block.get("bbox", (0, 0, 0, 0))
            largest = max(largest, abs((x1 - x0) * (y1 - y0)))
    return (largest / page_area) >= _SCAN_IMAGE_AREA_RATIO


def is_scanned_pdf(path: str, password: str | None = None) -> bool:
    """Return True when *path* is a scanned (image-only) PDF from which no text
    can be extracted. Inspects only the first page via PyMuPDF (the page-1 layout
    is representative for whole-document scans, and this keeps the probe cheap);
    returns False on any error so a probe failure never blocks a legitimate
    upload.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    try:
        pw = _resolve_password(path, password)
        doc = fitz.open(path)
        try:
            if doc.needs_pass:
                doc.authenticate(pw or "")
            if doc.page_count == 0:
                return False
            page = doc[0]
            if len(page.get_text("text").strip()) >= _SCAN_MIN_CHARS:
                return False               # real text layer -> not scanned
            # No usable text; scanned only if the page is a big raster image
            # (guards against a genuinely blank/near-empty first page).
            return _page_is_image_scan(page)
        finally:
            doc.close()
    except Exception:
        return False


# ===========================================================================
# Undecodable-text detection (PyMuPDF) — reject PDFs whose text layer is glyph
# codes rather than characters.
# ===========================================================================
#
# Some statements embed their fonts as Type3 (or a subset with a custom
# encoding) and ship NO ToUnicode CMap. The page then has a perfectly real text
# layer, but nothing maps those glyph ids back to characters: pdfminer/Camelot
# emit "(cid:17)(cid:16)" and PyMuPDF hands back raw control bytes. Extraction
# "succeeds" and produces a full table of junk like
#
#     {"Value Date": "//,*(cid:17),(cid:16)*(cid:16)- N+12(cid:18)//(cid:18)"}
#
# is_scanned_pdf does NOT catch these: the page reports plenty of characters, so
# it looks like a normal digital PDF. Only their *content* gives it away, hence
# this separate probe.
#
# The measure is the share of non-whitespace characters that are control codes
# (< U+0020) or sit in a private-use / unassigned Unicode category — exactly what
# an unmapped glyph decodes to, and essentially zero on a healthy statement.

_UNDECODABLE_RATIO = 0.10   # >= this share of junk characters => unreadable
_UNDECODABLE_MIN_CHARS = 200  # need this much text before the ratio means anything


def _undecodable_ratio(text: str) -> float:
    """Share of non-whitespace characters that decode to nothing meaningful."""
    total = bad = 0
    for ch in text:
        if ch.isspace():
            continue
        total += 1
        if ord(ch) < 32 or unicodedata.category(ch) in ("Co", "Cn"):
            bad += 1
    return (bad / total) if total else 0.0


def has_undecodable_text(path: str, password: str | None = None) -> bool:
    """True when the PDF's text layer can't be decoded into real characters.

    Samples the same leading pages as the other probes and returns False on any
    error, or when there simply isn't enough text to judge — a probe failure
    must never block an upload that would otherwise have worked.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    try:
        pw = _resolve_password(path, password)
        doc = fitz.open(path)
        try:
            if doc.needs_pass:
                doc.authenticate(pw or "")
            parts = []
            for i in range(min(SAMPLE_PAGES, doc.page_count)):
                parts.append(doc[i].get_text("text"))
            text = "".join(parts)
            if len(text.strip()) < _UNDECODABLE_MIN_CHARS:
                return False        # too little text to call — let it through
            return _undecodable_ratio(text) >= _UNDECODABLE_RATIO
        finally:
            doc.close()
    except Exception:
        return False


def classify_pdf_fast(path: str, password: str | None = None) -> PdfTypeResult:
    """Production classifier: a fast PyMuPDF pre-check that skips Camelot for the
    confidently-stream majority, deferring every possibly-tabular PDF to the
    Camelot-based classify_pdf(). Returns the same PdfTypeResult shape either way
    so callers are unchanged.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    if _probe_is_confident_stream(path, password):
        return PdfTypeResult(
            path=path,
            category="stream",
            flavor="stream",
            rows=0,
            columns=0,
            populated_columns=0,
            reason="pymupdf probe: no ruled grid + real text -> stream "
                   "(Camelot lattice probe skipped)",
        )
    return classify_pdf(path, password=password)


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
