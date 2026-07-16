import os
import re
import io
import json
import gc
import secrets
import tempfile
import warnings
from functools import wraps
from datetime import datetime, timedelta, timezone

warnings.filterwarnings("ignore", category=DeprecationWarning)

from flask import (
    Flask, request, jsonify, send_from_directory, send_file, session
)
from werkzeug.security import generate_password_hash, check_password_hash
from pymongo import MongoClient
from bson import ObjectId
from bson.errors import InvalidId
import camelot
import pandas as pd
import torch
from PyPDF2 import PdfReader, PdfWriter
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# Lattice-vs-stream classifier and the stream (no-ruled-grid) extractor.
# classify_pdf_fast adds a PyMuPDF pre-check that skips the expensive Camelot
# lattice probe for the confidently-stream majority (deferring only
# possibly-tabular PDFs to Camelot for an identical verdict).
from pdfType import classify_pdf_fast, is_scanned_pdf
from ledger_extract.pipeline import extract_statement
from ledger_extract.common import parse_date_leading

app = Flask(__name__, static_folder=".")

# Secret key signs the session cookie that keeps a user logged in. In
# production set the SECRET_KEY env var. With no env var (local dev) we
# persist a randomly generated key to a gitignored file rather than ship a
# guessable hardcoded default — this keeps sessions stable across reloads
# while never baking a predictable secret into source.
def _load_secret_key():
    env_key = os.environ.get("SECRET_KEY")
    if env_key:
        return env_key

    key_file = os.path.join(os.path.dirname(__file__), ".flask_secret")
    try:
        if os.path.exists(key_file):
            with open(key_file, "r") as fh:
                saved = fh.read().strip()
                if saved:
                    return saved
        generated = secrets.token_hex(32)
        with open(key_file, "w") as fh:
            fh.write(generated)
        return generated
    except OSError:
        # Filesystem not writable — fall back to an in-memory key. Sessions
        # won't survive a restart, but the secret is still unguessable.
        return secrets.token_hex(32)


app.secret_key = _load_secret_key()

# ---------------------------------------------------------------------------
# MongoDB connection
# ---------------------------------------------------------------------------
# Defaults to a local mongod on the standard port. Point MONGO_URI elsewhere
# (e.g. an Atlas connection string) without touching the code.
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
mongo_client = MongoClient(MONGO_URI)
db = mongo_client[os.environ.get("MONGO_DB", "ledgerit")]

users = db["users"]              # registered accounts
statements = db["statements"]    # saved extraction results, per user
messages = db["messages"]        # contact-form submissions

# Mobile number is the login identifier, so it must be unique. Guard the call
# so the app still starts (with a clear warning) when mongod isn't running yet;
# the index gets created on the first successful DB operation either way.
try:
    users.create_index("mobile", unique=True)
except Exception as exc:  # pragma: no cover - depends on local mongod
    print(f"[ledgerit] WARNING: could not reach MongoDB at {MONGO_URI}: {exc}")


def current_user():
    # Return the logged-in user's id from the signed session cookie, or None.
    return session.get("user_id")


def _to_oid(value):
    # Parse a string into an ObjectId, returning None on any malformed input
    # so route handlers can answer 404 instead of raising.
    try:
        return ObjectId(value)
    except (InvalidId, TypeError):
        return None


def current_user_doc():
    # Fetch the logged-in user's document fresh from the DB (or None). Reading
    # live — rather than trusting the session — means a demotion or deactivation
    # takes effect on the user's very next request.
    uid = current_user()
    if not uid:
        return None
    oid = _to_oid(uid)
    if not oid:
        return None
    return users.find_one({"_id": oid})


def login_required(view):
    # Guard JSON API routes: respond 401 instead of redirecting so the
    # front-end can react (the pages are static and handle auth themselves).
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not current_user():
            return jsonify({"error": "AUTH_REQUIRED"}), 401
        return view(*args, **kwargs)
    return wrapped


VALID_ROLES = ("super_admin", "admin", "salesperson", "user")
# Roles that can reach the admin console at all.
STAFF_ROLES = ("super_admin", "admin", "salesperson")


def user_role(user):
    # Single source of truth for a user's role. Falls back to the legacy
    # is_super_admin / is_admin flags for documents created before the `role`
    # field existed.
    if not user:
        return "user"
    r = user.get("role")
    if r in VALID_ROLES:
        return r
    if user.get("is_super_admin"):
        return "super_admin"
    if user.get("is_admin"):
        return "admin"
    return "user"


def require_roles(*roles):
    # Guard a route so only the listed roles may call it. Role is read live from
    # the DB every request, so a role change takes effect immediately. The
    # server is the real enforcement — the UI only hides what a role can't use.
    def decorator(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            user = current_user_doc()
            if not user:
                return jsonify({"error": "AUTH_REQUIRED"}), 401
            if user_role(user) not in roles:
                return jsonify({"error": "FORBIDDEN"}), 403
            return view(*args, **kwargs)
        return wrapped
    return decorator


# Convenience guards.
super_admin_required = require_roles("super_admin")
admin_required = require_roles("super_admin", "admin")          # secondary admin + main
staff_required = require_roles(*STAFF_ROLES)                    # any console user

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "row_classifier")

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

id2label = {
    0: "header",
    1: "entry",
    2: "c-entry",
    3: "info",
    4: "remove"
}


def preprocess_raw_df(df):
    cleaned_rows = []

    for _, row in df.iterrows():
        new_row = []

        for cell in row:
            cell = str(cell).replace("\n", "")
            cell = " ".join(cell.split())
            new_row.append(cell)

        cleaned_rows.append(new_row)

    return pd.DataFrame(cleaned_rows)


def split_header_row(row):
    row = [str(cell).strip() for cell in row]

    new_row = row.copy()

    for i, cell in enumerate(row):

        if "\n" in cell:

            parts = [p.strip() for p in cell.split("\n") if p.strip()]

            new_row[i] = parts[0]

            next_index = i + 1

            for part in parts[1:]:

                placed = False

                while next_index < len(new_row):

                    if str(new_row[next_index]).strip() == "":
                        new_row[next_index] = part
                        next_index += 1
                        placed = True
                        break

                    next_index += 1

                if not placed:
                    # No empty column to the right — keep the wrapped word in
                    # its own cell instead of dropping it ("Running\nBalance"
                    # -> "Running Balance", Yes Bank). The balance summary and
                    # the Tally export locate the balance column by the word
                    # "balance", so losing it broke both.
                    new_row[i] = (new_row[i] + " " + part).strip()

    return new_row


# Pairs of opposite amount-column words that some layouts print inside ONE
# header cell. Yes Bank draws "Withdrawals" and "Deposits" as two ruled columns
# but Camelot reads both words into the Withdrawals header cell (they sit on a
# single text line spanning the boundary), leaving the Deposits header cell
# empty -> "UNKNOWN". The data cells underneath are split correctly, so only
# the header names need re-homing.
_OPPOSITE_HEADER_PAIRS = [
    {"withdrawal", "deposit"},
    {"debit", "credit"},
    {"dr", "cr"},
]


def _norm_header_word(word):
    # lowercase, letters only, singular: "Withdrawals" -> "withdrawal".
    w = re.sub(r"[^a-z]", "", str(word).lower())
    return w[:-1] if w.endswith("s") else w


def split_merged_opposite_headers(headers):
    """Split a two-word opposite-pair header cell across itself and the next
    empty ("UNKNOWN") column: ["Withdrawals Deposits", "UNKNOWN"] ->
    ["Withdrawals", "Deposits"]. Fires only when the cell holds exactly two
    words forming a known debit/credit-style pair AND the following header is
    blank, so ordinary headers are never touched."""
    headers = list(headers)
    for i in range(len(headers) - 1):
        if str(headers[i + 1]).strip().upper() != "UNKNOWN":
            continue
        words = str(headers[i]).split()
        if len(words) != 2:
            continue
        pair = {_norm_header_word(words[0]), _norm_header_word(words[1])}
        if pair in _OPPOSITE_HEADER_PAIRS:
            headers[i], headers[i + 1] = words[0], words[1]
    return headers


def clean_entry_row(row):
    cleaned = []

    for cell in row:
        cell = str(cell).replace("\n", " ")
        cell = " ".join(cell.split())
        cleaned.append(cell)

    return cleaned


# Keyword groups that almost every bank-statement header contains.
# A row is considered a header if it matches at least two distinct groups
# (e.g. a "date" word AND a "balance"/"amount" word). This deterministic
# check is a safety net for when the ML row-classifier fails to tag the
# header row (seen with unfamiliar layouts such as ICICI's
# "Withdrawal (Dr) / Deposit (Cr)" format).
HEADER_KEYWORD_GROUPS = [
    ("date",),
    ("balance",),
    ("withdrawal", "deposit", "debit", "credit", "dr", "cr"),
    ("narration", "remarks", "particulars", "details", "description"),
    ("cheque", "ref no", "reference"),
    ("amount",),
]

# Phrases that mark a SUMMARY / TOTALS row, not a column header. Bank
# statements often end with a "Statement Summary" block whose cells
# ("Brought Forward", "Total Debits", "Closing Balance", "Dr Count" ...)
# would otherwise match the header keywords above and get mis-promoted.
SUMMARY_ROW_MARKERS = [
    "brought forward", "total debit", "total credit", "closing balance",
    "opening balance", "dr count", "cr count", "count", "statement summary",
    "page total", "carried forward",
]


def looks_like_header(row_values):
    # Join the row's cells into one lowercase string.
    joined = " ".join(str(v) for v in row_values).lower()

    if not joined.strip():
        return False

    # Reject summary / totals rows outright (e.g. the end-of-statement
    # "Brought Forward / Total Debits / Closing Balance" block in SBI).
    if any(marker in joined for marker in SUMMARY_ROW_MARKERS):
        return False

    # A header should be mostly words, not transaction data. Reject rows
    # that are dominated by digits (real entries have amounts/dates/refs).
    digits = sum(c.isdigit() for c in joined)
    letters = sum(c.isalpha() for c in joined)
    if letters == 0 or digits > letters:
        return False

    groups_matched = 0
    for group in HEADER_KEYWORD_GROUPS:
        if any(kw in joined for kw in group):
            groups_matched += 1

    # Require a date-type column AND at least one more group. A genuine
    # transaction header always has a date column; this extra requirement
    # further guards against promoting stray non-header rows.
    has_date = "date" in joined
    return has_date and groups_matched >= 2


# Matches a date token at the very START of a string, covering the formats
# seen across Indian bank statements:
#   04-04-25, 04/04/2025, 04.04.2025   (DD-MM-YY[YY])
#   2025-04-04                          (YYYY-MM-DD)
#   01-MAR-2024, 03 May 2026            (textual month)
# Group 1 captures the date so it can be lifted out of the cell.
LEADING_DATE_RE = re.compile(
    r"^\s*("
    r"\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}"            # 04-04-25 / 04/04/2025
    r"|\d{4}[-/.]\d{1,2}[-/.]\d{1,2}"             # 2025-04-04
    r"|\d{1,2}[-/\s][A-Za-z]{3,9}[-/\s]\d{2,4}"   # 01-MAR-2024 / 03 May 2026
    r")\b"
)

# Header keywords used to locate the date column and the description column
# when re-homing a misplaced date.
_DATE_HDR_KW = "date"
_DESC_HDR_KW = ("particular", "narration", "description", "details", "remarks")


def relocate_leading_date(records, headers):
    # On tall, multi-line rows (e.g. Ujjivan Small Finance Bank) Camelot
    # misaligns the boundary between the date column and the description
    # column, in BOTH directions:
    #
    #   Case A — the date leaks INTO the description and the date cell is
    #            left blank:  Date="" | Particular="04-04-25 MB/IMPS/..."
    #   Case B — the first line of the description leaks INTO the date cell:
    #            Date="12-07-25 MB/NEFT DR/LUCKU" | Particular="PRODUCTS AU..."
    #
    # Re-home the date so it sits alone in the date column and the full
    # description stays intact, so the row displays and exports correctly.
    desc_col = next(
        (h for h in headers
         if any(kw in str(h).lower() for kw in _DESC_HDR_KW)),
        None
    )

    if not desc_col:
        return records

    # The misaligned boundary is always between the description and the date
    # column immediately to its LEFT. On two-date layouts (SBI:
    # "Txn Date | Value Date | Description") that is "Value Date", NOT the
    # first "Txn Date" — so search leftward from the description for the
    # nearest date column, and only fall back to the first date column overall
    # (single-date layouts like Ujjivan, where the date leads the row).
    desc_idx = headers.index(desc_col)
    date_col = next(
        (h for h in reversed(headers[:desc_idx])
         if _DATE_HDR_KW in str(h).lower()),
        None
    )
    if not date_col:
        date_col = next(
            (h for h in headers if _DATE_HDR_KW in str(h).lower()),
            None
        )

    if not date_col or date_col == desc_col:
        return records

    for rec in records:
        date_val = str(rec.get(date_col, "")).strip()
        desc_val = str(rec.get(desc_col, "")).strip()

        if not date_val:
            # Case A: date sitting at the front of the description.
            match = LEADING_DATE_RE.match(desc_val)
            if match:
                rec[date_col] = match.group(1)
                rec[desc_col] = desc_val[match.end():].lstrip()
        else:
            # Case B: description text trailing the date in the date cell.
            match = LEADING_DATE_RE.match(date_val)
            if match:
                trailing = date_val[match.end():].lstrip()
                # Only move genuine description text (has letters) — never a
                # stray second date/number, guarding two-date layouts.
                if trailing and re.search(r"[A-Za-z]", trailing):
                    rec[date_col] = match.group(1)
                    rec[desc_col] = (trailing + " " + desc_val).strip()

    return records


def _leading_date_rows(df):
    # Count rows whose FIRST cell starts with a date. A genuine transaction
    # table leads every row with the transaction date; auxiliary tables on the
    # same page (a loan statement's "List Of PDC's Cleared" / "EMI's Unpaid"
    # blocks, or the account-info header) do not.
    return sum(
        1 for v in df.iloc[:, 0]
        if LEADING_DATE_RE.match(str(v))
    )


def _unlock_pdf(reader, password):
    """Resolve the effective open password for a (possibly) encrypted *reader*.

    Returns ("ok", effective_password), ("need", None) or ("wrong", None).

    Many bank statements are encrypted only to set owner-level permission
    restrictions (no printing/copying) while leaving the *user* (open) password
    blank. Normal PDF viewers open those without prompting, yet
    ``reader.is_encrypted`` is still True — so we must try the empty password
    before demanding one, otherwise we'd ask for a password the file doesn't have.
    """
    if not reader.is_encrypted:
        return "ok", password
    # Owner-only encryption: the empty user password opens it.
    if reader.decrypt("") != 0:
        return "ok", ""
    if not password:
        return "need", None
    if reader.decrypt(password) == 0:
        return "wrong", None
    return "ok", password


def process_pdf(pdf_path, password=None):

    reader = PdfReader(pdf_path)

    status, password = _unlock_pdf(reader, password)

    if status == "need":
        return None, None, "PASSWORD_REQUIRED"
    if status == "wrong":
        return None, None, "WRONG_PASSWORD"

    if reader.is_encrypted:

        tables = camelot.read_pdf(
            pdf_path,
            pages="all",
            flavor="lattice",
            password=password,
            suppress_stdout=True
        )

    else:

        tables = camelot.read_pdf(
            pdf_path,
            pages="all",
            flavor="lattice",
            suppress_stdout=True
        )

    if len(tables) == 0:
        return None, None, "TABLE-LESS"

    non_empty = [t.df for t in tables if not t.df.empty]

    if not non_empty:
        return None, None, "TABLE-LESS"

    # Keep only the transaction table(s): those whose first column is
    # predominantly dates. This drops auxiliary tables Camelot finds on the
    # same pages (e.g. a loan statement's "List Of PDC's Cleared" / "EMI's
    # Unpaid" lists and the account-info block) which otherwise pollute the
    # output with junk rows and, having a different column count, introduce a
    # spurious all-empty "nan" column on concat. Fall back to all tables if
    # the heuristic matches none (unusual layout) so nothing is lost.
    txn_dfs = [df for df in non_empty if _leading_date_rows(df) >= 2]
    all_dfs = txn_dfs if txn_dfs else non_empty

    final_df = pd.concat(all_dfs, ignore_index=True)

    if final_df.empty:
        return None, None, "TABLE-LESS"

    original_df = final_df.copy()

    df = preprocess_raw_df(final_df)

    texts = []

    for _, row in df.iterrows():

        text = " ".join(
            str(cell).strip()
            for cell in row
            if str(cell).strip()
        )

        texts.append(" ".join(text.split()))

    predicted_labels = []

    for text in texts:

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = model(**inputs)

        prediction = torch.argmax(outputs.logits, dim=1).item()

        predicted_labels.append(id2label[prediction])

    original_df["predicted_label"] = predicted_labels

    # Fallback header rescue — runs BEFORE the "remove" filter.
    # The ML classifier sometimes fails to tag the header row on unfamiliar
    # layouts (e.g. ICICI's "Withdrawal (Dr) / Deposit (Cr)" format), labeling
    # it "entry", "info", or even "remove". If no row is tagged "header",
    # scan every row (including ones about to be removed) for a header-like
    # one and re-tag the first match so it survives downstream.
    #
    # Such rescued headers are tagged "header-kw" (keyword-detected) rather
    # than "header": camelot has already placed each header word in its own
    # column for these layouts, so they must be cleaned, NOT run through
    # split_header_row (which redistributes multi-line cells and would
    # truncate already-separated headers to their first word).
    rescued_header = False
    if not (original_df["predicted_label"] == "header").any():
        for idx in original_df.index:
            row_values = original_df.loc[idx].drop("predicted_label").tolist()
            if looks_like_header(row_values):
                original_df.at[idx, "predicted_label"] = "header-kw"
                rescued_header = True
                break

    df = original_df[
        original_df["predicted_label"] != "remove"
    ]

    fixed_rows = []

    for idx, row in df.iterrows():

        label = row["predicted_label"]

        row_values = row[:-1].tolist()

        if label == "header":
            row_values = split_header_row(row_values)

        elif label == "header-kw":
            # keyword-rescued header: columns already separated by camelot,
            # so just normalise whitespace.
            row_values = clean_entry_row(row_values)

        elif label in ["entry", "c-entry"]:
            row_values = clean_entry_row(row_values)

        row_values.append(label)

        fixed_rows.append(row_values)

    df = pd.DataFrame(fixed_rows)

    df = df.rename(
        columns={df.columns[-1]: "predicted_label"}
    )

    # Treat a keyword-rescued header the same as a classifier-detected one.
    header_indices = df[
        df["predicted_label"].isin(["header", "header-kw"])
    ].index.tolist()

    if header_indices:

        first_header_index = header_indices[0]

        header_row = df.loc[first_header_index]

        df = df.drop(
            df[
                (df["predicted_label"] == "header")
                & (df.index != first_header_index)
            ].index
        )

        headers = [
            str(v).strip() or "UNKNOWN"
            for v in header_row[:-1]
        ]

        # "Withdrawals Deposits" read into one cell with the next header left
        # "UNKNOWN" (Yes Bank) -> put each word over its own data column.
        headers = split_merged_opposite_headers(headers)

    else:

        default_headers = [
            "Value Date",
            "Post Date",
            "Details",
            "Ref No/Cheque No",
            "Debit",
            "Credit",
            "Balance"
        ]

        num_cols = len(df.columns) - 1

        headers = default_headers[:num_cols]

        while len(headers) < num_cols:
            headers.append(f"UNKNOWN_{len(headers)+1}")

    records = []

    current_record = None

    for idx, row in df.iterrows():

        label = row["predicted_label"]

        row_values = row[:-1].tolist()

        if label == "entry":

            current_record = {
                h: str(v).strip()
                for h, v in zip(headers, row_values)
            }

            records.append(current_record)

        elif label == "c-entry" and current_record is not None:

            for h, v in zip(headers, row_values):

                v = str(v).strip()

                if v:
                    current_record[h] += " " + v

    # Re-home any date that leaked into the description column.
    relocate_leading_date(records, headers)

    gc.collect()

    return records, headers, None


def _stream_cell_to_str(value):
    # The stream DataFrame holds floats (amounts/balance), None/NaN for missing
    # cells, and strings (date/description/type). Normalise every cell to the
    # string form the front-end expects: blanks for missing, 2dp for money.
    if value is None:
        return ""
    if isinstance(value, float):
        if value != value:        # NaN
            return ""
        return f"{value:.2f}"
    return str(value)


def process_pdf_stream(pdf_path, password=None):
    # Stream path (no ruled grid). Delegates to ledger_extract, then reshapes
    # its canonical DataFrame (Date, Description, Reference, Debit, Credit,
    # Balance, Type) into the same {headers, data} contract the lattice path
    # returns, so results.html / the Tally-XML / CSV exporters work unchanged.
    pw_map = (
        {os.path.basename(pdf_path): password} if password else None
    )

    df, _meta = extract_statement(pdf_path, pw_map)

    # The derived CR/DR "Type" column isn't wanted in the output.
    if "Type" in df.columns:
        df = df.drop(columns=["Type"])

    headers = list(df.columns)
    records = [
        {k: _stream_cell_to_str(v) for k, v in rec.items()}
        for rec in df.to_dict("records")
    ]

    gc.collect()

    if not records:
        return None, None, "TABLE-LESS"

    return records, headers, None


_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Month-name-first dates as printed by some banks (e.g. Bandhan's "June30, 2025"
# / "June 30, 2025"), which parse_date_leading -- tuned for day-first formats --
# doesn't cover.
_MONTH_FIRST_FORMATS = ("%B%d, %Y", "%B %d, %Y", "%b%d, %Y", "%b %d, %Y")


def _row_date(value):
    """ISO date string (YYYY-MM-DD, lexically sortable) for *value*, or None.

    Handles every date shape the extractors emit: the stream path already emits
    ISO dates; the lattice path keeps the statement's printed format -- usually
    day-first (30/09/2025, 30-Sep-2025), which parse_date_leading normalises,
    but sometimes month-first (June30, 2025), handled here.
    """
    s = str(value).strip()
    if _ISO_DATE_RE.match(s):
        return s
    iso = parse_date_leading(s)
    if iso:
        return iso
    for fmt in _MONTH_FIRST_FORMATS:
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _to_chronological(records, headers):
    """Return *records* oldest-first.

    Some banks (e.g. Bandhan) print transactions newest-first. We want the
    table, the opening/closing-balance summary (results.html computes opening
    from the first row) and the CSV/XML exports to read oldest-first, so when a
    statement is clearly in reverse-chronological order we flip it.

    We only *reverse* the rows, never sort by date: that keeps multiple
    transactions on the same day in their original relative sequence (the
    running balance depends on it) while still turning newest-first into
    oldest-first. Applies to both the lattice and stream paths.
    """
    if not records or not headers:
        return records

    # Locate the date column: the header whose cells most often parse as dates.
    date_col, best_hits = None, 0
    for h in headers:
        hits = sum(1 for r in records if _row_date(r.get(h, "")))
        if hits > best_hits:
            date_col, best_hits = h, hits
    if date_col is None or best_hits < 2:
        return records

    # Compare the run of ascending vs descending steps between dated rows.
    asc = desc = 0
    prev = None
    for r in records:
        iso = _row_date(r.get(date_col, ""))
        if iso is None:
            continue
        if prev is not None and iso != prev:
            if iso > prev:
                asc += 1
            else:
                desc += 1
        prev = iso

    return list(reversed(records)) if desc > asc else records


def _drop_empty_date_columns(records, headers):
    """Remove any date column that is blank in every row.

    Some statements (e.g. SBI) print two date columns — "Txn Date" and
    "Value Date" — where one is populated and the other is entirely empty.
    An empty date column is dead weight in the table/CSV and, worse, shadows
    the real one during Tally XML voucher generation: the exporter picks the
    first date-like header it finds, so every voucher ends up dated blank and
    gets skipped ("No vouchers could be generated"). Drop the empty date
    columns so only the populated one survives, keeping the display, CSV and
    XML export consistent.

    Scoped to date columns on purpose — dropping *any* all-blank column would
    remove e.g. an empty "Credit" column from a payments-only statement and
    break the dual debit/credit detection downstream.
    """
    if not records or not headers:
        return records, headers

    def is_date_header(h):
        return "date" in str(h).lower()

    def col_all_blank(h):
        return all(str(r.get(h, "")).strip() == "" for r in records)

    kept = [
        h for h in headers
        if not (is_date_header(h) and col_all_blank(h))
    ]

    if len(kept) == len(headers):
        return records, headers   # nothing dropped

    new_records = [{h: r.get(h, "") for h in kept} for r in records]
    return new_records, kept


def _drop_empty_optional_columns(records, headers):
    """Remove an all-blank Reference / cheque / code column from the output.

    Many statements have no dedicated Chq./Ref.No. column, so the extractor's
    "Reference" column comes back empty for every row (e.g. Indian Bank, whose
    narration carries the transfer ref inline). An empty identifier column is
    dead weight in the table, CSV and Tally XML, so drop it.

    Scoped to *identifier* columns (reference / ref / cheque / chq / code) on
    purpose — like _drop_empty_date_columns, dropping *any* all-blank column
    would remove an empty "Credit" column from a payments-only statement and
    break the dual debit/credit view. Date/Description/Debit/Credit/Balance are
    never dropped even when blank.
    """
    if not records or not headers:
        return records, headers

    def is_optional_header(h):
        hl = str(h).lower()
        return any(
            k in hl for k in ("reference", "ref no", "ref.", "cheque", "chq", "code")
        )

    def col_all_blank(h):
        return all(str(r.get(h, "")).strip() == "" for r in records)

    kept = [
        h for h in headers
        if not (is_optional_header(h) and col_all_blank(h))
    ]

    if len(kept) == len(headers):
        return records, headers   # nothing dropped

    new_records = [{h: r.get(h, "") for h in kept} for r in records]
    return new_records, kept


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/results.html")
def results_page():
    return send_from_directory(".", "results.html")


@app.route("/upload.html")
def upload_page():
    return send_from_directory(".", "upload.html")


@app.route("/statements.html")
def statements_page():
    return send_from_directory(".", "statements.html")


@app.route("/admin.html")
def admin_page():
    # The page itself is public to serve, but it calls /api/me on load and
    # bounces non-admins; every byte of data behind it is admin_required.
    return send_from_directory(".", "admin.html")


@app.route("/plans.html")
def plans_page():
    return send_from_directory(".", "plans.html")


@app.route("/register.html")
def register_page():
    return send_from_directory(".", "register.html")


@app.route("/login.html")
def login_page():
    return send_from_directory(".", "login.html")


@app.route("/contact.html")
def contact_page():
    return send_from_directory(".", "contact.html")


@app.route("/contactus.html")
def contactus_page():
    return send_from_directory(".", "contactus.html")


# Serve front-end assets (logos, favicon, etc.) from the assets/ folder only.
# Scoping to "assets" means this route cannot reach app.py or the model
# weights. send_from_directory rejects path-traversal attempts on its own.
@app.route("/assets/<path:filename>")
def assets(filename):
    return send_from_directory("assets", filename)


# Free-plan page allowance. Every registered (non-admin) user can extract up
# to this many PDF pages before they must upgrade. Stored per-user as
# `page_limit` so the cap can be raised for an individual account straight from
# the database (or admin console); this constant is only the default applied to
# new accounts and to older ones that predate the field.
FREE_PAGE_LIMIT = 100


# ---------------------------------------------------------------------------
# Brute-force lockout
# ---------------------------------------------------------------------------
# After MAX_FAILED_LOGINS wrong passwords in a row, the account is locked for
# LOCKOUT_DURATION. The counter lives on the user document (not the session), so
# an attacker can't reset it by clearing cookies or switching browsers.
MAX_FAILED_LOGINS = 3
LOCKOUT_DURATION = timedelta(minutes=3)


def _lock_seconds_left(user):
    # Seconds until the account unlocks; 0 when it isn't locked. Mongo hands back
    # naive datetimes, so pin them to UTC before comparing.
    until = user.get("lock_until")
    if not until:
        return 0
    if until.tzinfo is None:
        until = until.replace(tzinfo=timezone.utc)
    left = (until - datetime.now(timezone.utc)).total_seconds()
    return max(0, int(left))


def _humanize(seconds):
    # "2 hours 45 minutes" / "12 minutes" — for the message shown on the login
    # page. Rounds *up* to the next whole minute so we never tell someone to come
    # back sooner than the lock actually lifts (a 3-minute lock with 2m01s left
    # reads "3 minutes", not "2").
    hours, minutes = divmod(-(-seconds // 60), 60)
    parts = []
    if hours:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    return " ".join(parts) if parts else "under a minute"


def _locked_response(seconds_left):
    return jsonify({
        "error": (
            "Too many failed attempts. This account is locked for security — "
            f"try again in {_humanize(seconds_left)}."
        ),
        "locked": True,
        "retry_after": seconds_left,
    }), 423  # 423 Locked


# ---------------------------------------------------------------------------
# Auth API
# ---------------------------------------------------------------------------
@app.route("/api/register", methods=["POST"])
def register():
    data = request.get_json(silent=True) or {}

    name = (data.get("name") or "").strip()
    mobile = (data.get("mobile") or "").strip()
    password = data.get("password") or ""

    if not name:
        return jsonify({"error": "Name is required"}), 400
    if not re.fullmatch(r"\d{10}", mobile):
        return jsonify({"error": "Enter a valid 10-digit mobile number"}), 400
    if len(password) < 8:
        return jsonify({"error": "Password must be at least 8 characters"}), 400

    if users.find_one({"mobile": mobile}):
        return jsonify({"error": "An account with this mobile already exists"}), 409

    result = users.insert_one({
        "name": name,
        "mobile": mobile,
        "password_hash": generate_password_hash(password),
        "role": "user",
        # Free-plan page credit: how many PDF pages they've extracted and the
        # cap. page_limit is per-user so it can be bumped from the DB to grant
        # someone more without changing code.
        "pages_used": 0,
        "page_limit": FREE_PAGE_LIMIT,
        "created_at": datetime.now(timezone.utc),
    })

    # Log the new user straight in.
    session["user_id"] = str(result.inserted_id)
    session["name"] = name

    return jsonify({"ok": True, "name": name})


@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}

    mobile = (data.get("mobile") or "").strip()
    password = data.get("password") or ""

    user = users.find_one({"mobile": mobile})

    # Unknown mobile: there's no account to lock, so point them at registration
    # instead of a generic failure. The `unregistered` flag lets the login page
    # turn "Register" into a link.
    if not user:
        return jsonify({"error": "New User? Register.", "unregistered": True}), 401

    # A locked account is refused before the password is even checked — so the
    # lock holds even if the attacker eventually guesses right.
    seconds_left = _lock_seconds_left(user)
    if seconds_left:
        return _locked_response(seconds_left)

    if not check_password_hash(user["password_hash"], password):
        # The lock window has passed if we got here, so a stale lock_until from a
        # previous lockout is cleared as we count this attempt.
        failed = int(user.get("failed_logins", 0)) + 1

        if failed >= MAX_FAILED_LOGINS:
            users.update_one(
                {"_id": user["_id"]},
                {"$set": {
                    "failed_logins": 0,
                    "lock_until": datetime.now(timezone.utc) + LOCKOUT_DURATION,
                }},
            )
            return _locked_response(int(LOCKOUT_DURATION.total_seconds()))

        users.update_one(
            {"_id": user["_id"]},
            {"$set": {"failed_logins": failed}, "$unset": {"lock_until": ""}},
        )
        left = MAX_FAILED_LOGINS - failed
        return jsonify({
            "error": (
                "Invalid mobile number or password. "
                f"{left} attempt{'s' if left != 1 else ''} left before this "
                f"account is locked for {_humanize(int(LOCKOUT_DURATION.total_seconds()))}."
            ),
            "attempts_left": left,
        }), 401

    # Correct password — wipe the failure streak so the count only ever tracks
    # *consecutive* failures.
    if user.get("failed_logins") or user.get("lock_until"):
        users.update_one(
            {"_id": user["_id"]},
            {"$unset": {"failed_logins": "", "lock_until": ""}},
        )

    session["user_id"] = str(user["_id"])
    session["name"] = user.get("name", "")

    # Surface the role so the login page can route staff straight to the admin
    # console instead of the upload page.
    role = user_role(user)
    return jsonify({
        "ok": True,
        "name": user.get("name", ""),
        "role": role,
        "is_admin": role in STAFF_ROLES,
    })


@app.route("/api/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"ok": True})


@app.route("/api/me")
def me():
    user = current_user_doc()
    if not user:
        return jsonify({"authenticated": False})

    role = user_role(user)
    is_staff = role in STAFF_ROLES
    limit = int(user.get("page_limit", FREE_PAGE_LIMIT))
    used = int(user.get("pages_used", 0))

    return jsonify({
        "authenticated": True,
        "name": user.get("name", "") or session.get("name", ""),
        "role": role,
        # is_admin = "has console access" (any staff role) — used by the other
        # pages to reveal the Admin link and route logged-in staff.
        "is_admin": is_staff,
        "is_super_admin": role == "super_admin",
        # Page-credit status for the upload page meter. Staff are unlimited.
        "unlimited": is_staff,
        "pages_used": used,
        "page_limit": limit,
        "pages_remaining": None if is_staff else max(0, limit - used),
    })


# ---------------------------------------------------------------------------
# Saved statements API
# ---------------------------------------------------------------------------
@app.route("/api/statements")
@login_required
def list_statements():
    # Newest first; exclude the bulky row data, PDF bytes and stored password
    # from the list view.
    docs = statements.find(
        {"user_id": current_user()},
        {"data": 0, "pdf": 0, "pdf_password": 0}
    ).sort("created_at", -1)

    out = []
    for d in docs:
        out.append({
            "id": str(d["_id"]),
            "filename": d.get("filename", ""),
            "row_count": d.get("row_count", 0),
            "page_count": d.get("page_count", 0),
            # Older docs predate the status field — treat them as successes.
            "status": d.get("status", "success"),
            "error": d.get("error"),
            "created_at": d.get("created_at").isoformat()
            if d.get("created_at") else None,
        })

    return jsonify({"statements": out})


@app.route("/api/statements/<sid>")
@login_required
def get_statement(sid):
    from bson import ObjectId
    from bson.errors import InvalidId

    try:
        oid = ObjectId(sid)
    except (InvalidId, TypeError):
        return jsonify({"error": "Not found"}), 404

    # Exclude the PDF bytes and stored password — the JSON view only needs the
    # parsed data.
    doc = statements.find_one(
        {"_id": oid, "user_id": current_user()},
        {"pdf": 0, "pdf_password": 0}
    )
    if not doc:
        return jsonify({"error": "Not found"}), 404

    return jsonify({
        "headers": doc.get("headers", []),
        "data": doc.get("data", []),
        "filename": doc.get("filename", ""),
        "has_pdf": bool(doc.get("content_type")),
        "page_count": doc.get("page_count", 0),
        "status": doc.get("status", "success"),
        "error": doc.get("error"),
    })


def _decrypt_pdf_bytes(data, password):
    # Return a copy of the PDF with its encryption stripped, so it opens in the
    # browser / PDF viewer without prompting for a password. If the file isn't
    # actually encrypted, or the stored password no longer unlocks it, return
    # the original bytes unchanged (the viewer can still prompt as a fallback).
    try:
        reader = PdfReader(io.BytesIO(data))
        if not reader.is_encrypted:
            return data
        if not password or reader.decrypt(password) == 0:
            return data

        writer = PdfWriter()
        for page in reader.pages:
            writer.add_page(page)

        buf = io.BytesIO()
        writer.write(buf)
        return buf.getvalue()
    except Exception:
        # Never let a decrypt hiccup break the download — fall back to raw bytes.
        return data


@app.route("/api/statements/<sid>/pdf")
@login_required
def download_statement_pdf(sid):
    # Stream the stored source PDF back to its owner. Scoped to the current
    # user so one account can't fetch another's file.
    from bson import ObjectId
    from bson.errors import InvalidId

    try:
        oid = ObjectId(sid)
    except (InvalidId, TypeError):
        return jsonify({"error": "Not found"}), 404

    doc = statements.find_one(
        {"_id": oid, "user_id": current_user()},
        {"pdf": 1, "filename": 1, "content_type": 1, "pdf_password": 1}
    )
    if not doc or not doc.get("pdf"):
        return jsonify({"error": "Not found"}), 404

    # Decrypt with the stored password (if any) before serving, so the user is
    # never asked for the password again when viewing or downloading.
    pdf_bytes = bytes(doc["pdf"])
    if doc.get("pdf_password"):
        pdf_bytes = _decrypt_pdf_bytes(pdf_bytes, doc["pdf_password"])

    # ?inline=1 renders the PDF in the browser (results.html viewer); the
    # default still forces a download (statements.html re-download button).
    inline = request.args.get("inline") in ("1", "true", "yes")

    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype=doc.get("content_type", "application/pdf"),
        as_attachment=not inline,
        download_name=doc.get("filename") or "statement.pdf",
    )


@app.route("/api/statements/<sid>/issue", methods=["POST"])
@login_required
def raise_statement_issue(sid):
    # An owner flags one of their statements as problematic (e.g. a bad
    # extraction). The report is stored on the statement itself as an `issue`
    # subdoc so it stays tied to its source; admins triage these from the
    # Issues tab. Scoped to the current user so nobody can flag another's file.
    oid = _to_oid(sid)
    if not oid:
        return jsonify({"error": "Not found"}), 404

    data = request.get_json(silent=True) or {}
    note = (data.get("note") or "").strip()[:2000]

    issue = {
        "note": note,
        "status": "open",
        "created_at": datetime.now(timezone.utc),
        "resolved_at": None,
        "resolved_by": None,
    }

    result = statements.update_one(
        {"_id": oid, "user_id": current_user()},
        {"$set": {"issue": issue}},
    )
    if result.matched_count == 0:
        return jsonify({"error": "Not found"}), 404

    return jsonify({"ok": True})


@app.route("/api/issues")
@login_required
def list_my_issues():
    # The current user's raised statement issues, for the "My Issues" panel on
    # statements.html. Mirrors the admin view but scoped to the owner and with
    # only the fields that page renders. Open issues first, then newest.
    docs = list(statements.find(
        {"user_id": current_user(), "issue": {"$exists": True}},
        {"data": 0, "pdf": 0, "pdf_password": 0},
    ))

    out = []
    for d in docs:
        issue = d.get("issue") or {}
        out.append({
            "statement_id": str(d["_id"]),
            "filename": d.get("filename", ""),
            "title": issue.get("title", ""),
            "note": issue.get("note", ""),
            "status": issue.get("status", "open"),
            # Screenshots aren't stored on issues yet — always false so the
            # panel doesn't request a screenshot endpoint that doesn't exist.
            "has_screenshot": False,
            "created_at": _iso(issue.get("created_at")),
            "resolved_at": _iso(issue.get("resolved_at")),
        })

    # Number by the order raised (oldest = #1) so each issue keeps a stable
    # label, then present open first and newest within each group.
    out.sort(key=lambda r: r["created_at"] or "")
    for i, r in enumerate(out, start=1):
        r["number"] = "#" + str(i)

    open_first = [r for r in out if r["status"] != "resolved"]
    resolved = [r for r in out if r["status"] == "resolved"]
    open_first.sort(key=lambda r: r["created_at"] or "", reverse=True)
    resolved.sort(key=lambda r: r["created_at"] or "", reverse=True)
    return jsonify({"issues": open_first + resolved})


# ---------------------------------------------------------------------------
# Contact form API
# ---------------------------------------------------------------------------
@app.route("/api/contact", methods=["POST"])
def contact():
    # Two different forms post here: the sales form on contact.html
    # (orgName/orgType, phone) and the general contact form on contactus.html
    # (subject/message, mobile). Accept the superset so both work, recording a
    # "source" tag and only the fields that were actually sent.
    data = request.get_json(silent=True) or {}

    def field(*names):
        for n in names:
            v = data.get(n)
            if v:
                return str(v).strip()
        return ""

    doc = {
        "source": field("source") or "contact",
        "first_name": field("firstName"),
        "last_name": field("lastName"),
        "email": field("email"),
        # contact.html uses "phone"; contactus.html uses "mobile".
        "phone": field("phone", "mobile"),
        "org_name": field("orgName"),
        "org_type": field("orgType"),
        "subject": field("subject"),
        "message": field("message"),
        "status": "open",
        "created_at": datetime.now(timezone.utc),
    }

    # A message needs at least an email or phone to be actionable.
    if not doc["email"] and not doc["phone"]:
        return jsonify({"error": "Email or phone is required"}), 400

    messages.insert_one(doc)

    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Admin API
# ---------------------------------------------------------------------------
# Every route here is admin_required: it exposes data across ALL users, so the
# server-side guard is the real enforcement (the hidden nav link is cosmetic).
def _iso(dt):
    # Mongo hands back naive datetimes that are really UTC (we always store
    # datetime.now(timezone.utc)). Emit them WITH the +00:00 offset — a bare
    # "2026-07-16T05:30:00" is parsed by browsers as *local* time, which
    # shifted every admin-page timestamp by the viewer's UTC offset.
    if not dt:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _user_statement_counts():
    # One aggregation → {user_id: count}, so the users list can show each
    # account's statement count without a query per row.
    counts = {}
    for row in statements.aggregate([
        {"$group": {"_id": "$user_id", "n": {"$sum": 1}}}
    ]):
        counts[row["_id"]] = row["n"]
    return counts


@app.route("/api/admin/stats")
@admin_required
def admin_stats():
    total_users = users.count_documents({})
    total_admins = users.count_documents({"role": {"$in": ["super_admin", "admin"]}})
    total_statements = statements.count_documents({})
    # "Open" = not yet handled. General messages get resolved; sales enquiries
    # are closed as purchased / not_purchased. Missing status (older docs) is
    # counted as open. $nin matches missing fields too.
    open_messages = messages.count_documents(
        {"status": {"$nin": ["resolved", "purchased", "not_purchased"]}}
    )
    total_messages = messages.count_documents({})
    open_issues = statements.count_documents({"issue.status": "open"})

    # Total rows extracted across every saved statement.
    rows_agg = list(statements.aggregate([
        {"$group": {"_id": None, "n": {"$sum": "$row_count"}}}
    ]))
    total_rows = rows_agg[0]["n"] if rows_agg else 0

    # Free-plan users closest to exhausting their 100-page credit. A user is on
    # the free plan when their cap is still the default FREE_PAGE_LIMIT (paid
    # accounts have a bumped page_limit). Older docs may lack the field — treat
    # a missing page_limit as the free default. Ranked by pages used, top 5.
    free_q = {
        "role": {"$nin": list(STAFF_ROLES)},
        "pages_used": {"$gt": 0},
        "$or": [
            {"page_limit": FREE_PAGE_LIMIT},
            {"page_limit": {"$exists": False}},
        ],
    }
    free_usage = []
    for u in users.find(
        free_q,
        {"name": 1, "mobile": 1, "pages_used": 1, "page_limit": 1},
    ).sort("pages_used", -1).limit(5):
        free_usage.append({
            "name": u.get("name", ""),
            "mobile": u.get("mobile", ""),
            "pages_used": int(u.get("pages_used", 0)),
            "page_limit": int(u.get("page_limit", FREE_PAGE_LIMIT)),
        })

    return jsonify({
        "totals": {
            "users": total_users,
            "admins": total_admins,
            "statements": total_statements,
            "rows": total_rows,
            "messages": total_messages,
            "open_messages": open_messages,
            "open_issues": open_issues,
        },
        "free_usage": free_usage,
    })


@app.route("/api/admin/users")
@super_admin_required
def admin_users():
    q = (request.args.get("q") or "").strip()
    # The main admin is never listed here — it's managed out-of-band, not from
    # the admin UI. Only the super admin can manage users.
    query = {"role": {"$ne": "super_admin"}, "is_super_admin": {"$ne": True}}
    if q:
        # Match name (case-insensitive) or mobile (substring). re.escape keeps
        # a user-supplied string from being interpreted as a regex.
        safe = re.escape(q)
        query["$or"] = [
            {"name": {"$regex": safe, "$options": "i"}},
            {"mobile": {"$regex": safe}},
        ]

    counts = _user_statement_counts()

    # Open issues raised per user (keyed by user_id).
    issue_counts = {}
    for row in statements.aggregate([
        {"$match": {"issue.status": "open"}},
        {"$group": {"_id": "$user_id", "n": {"$sum": 1}}},
    ]):
        issue_counts[row["_id"]] = row["n"]

    # Open messages per user. Contact-form messages aren't tied to a user id,
    # so they're linked by phone number (which matches the user's mobile).
    msg_counts = {}
    for row in messages.aggregate([
        {"$match": {"status": {"$ne": "resolved"}}},
        {"$group": {"_id": "$phone", "n": {"$sum": 1}}},
    ]):
        if row["_id"]:
            msg_counts[row["_id"]] = row["n"]

    docs = users.find(query, {"password_hash": 0}).sort("created_at", -1)

    out = []
    for d in docs:
        uid = str(d["_id"])
        role = user_role(d)
        limit = int(d.get("page_limit", FREE_PAGE_LIMIT))
        # User type: a staff role (admin/salesperson), a user who bought a plan
        # (page_limit bumped above the free cap), or a default free user.
        if role in ("admin", "salesperson"):
            user_type = role
        elif limit > FREE_PAGE_LIMIT:
            user_type = "paid"
        else:
            user_type = "free"
        out.append({
            "id": uid,
            "name": d.get("name", ""),
            "mobile": d.get("mobile", ""),
            "role": role,
            "user_type": user_type,
            "statement_count": counts.get(uid, 0),
            "issue_count": issue_counts.get(uid, 0),
            "message_count": msg_counts.get(d.get("mobile"), 0),
            "created_at": _iso(d.get("created_at")),
        })

    return jsonify({"users": out})


@app.route("/api/admin/users/<uid>/role", methods=["POST"])
@super_admin_required
def admin_set_role(uid):
    # Assign a user's role (user / admin / salesperson). Only the super admin
    # can do this. The super_admin role can't be granted or removed here.
    oid = _to_oid(uid)
    if not oid:
        return jsonify({"error": "Not found"}), 404

    data = request.get_json(silent=True) or {}
    new_role = (data.get("role") or "").strip()
    if new_role not in ("user", "admin", "salesperson"):
        return jsonify({"error": "Invalid role"}), 400

    target = users.find_one({"_id": oid})
    if not target:
        return jsonify({"error": "Not found"}), 404

    # The main admin's role is protected.
    if user_role(target) == "super_admin" or target.get("is_super_admin"):
        return jsonify({"error": "The main admin's role can't be changed."}), 403

    # Keep the legacy is_admin flag in sync for any code still reading it.
    users.update_one(
        {"_id": oid},
        {"$set": {"role": new_role, "is_admin": new_role == "admin"}},
    )
    return jsonify({"ok": True, "role": new_role})


@app.route("/api/admin/users/<uid>", methods=["DELETE"])
@super_admin_required
def admin_delete_user(uid):
    # Delete a user account. Their statements are PRESERVED (the admin keeps
    # them) — each one is stamped with the former owner's details and marked
    # owner_deleted so the admin still sees who it belonged to. You can't delete
    # yourself or the last admin.
    oid = _to_oid(uid)
    if not oid:
        return jsonify({"error": "Not found"}), 404

    if str(oid) == current_user():
        return jsonify({"error": "You can't delete yourself."}), 400

    target = users.find_one({"_id": oid})
    if not target:
        return jsonify({"error": "Not found"}), 404

    # The main (first) admin is protected: nobody can delete this account.
    if user_role(target) == "super_admin" or target.get("is_super_admin"):
        return jsonify({"error": "The main admin can't be deleted."}), 403

    # Stamp the owner's details onto their statements before removing the
    # account, so the admin's statement list can still attribute them.
    statements.update_many(
        {"user_id": str(oid)},
        {"$set": {
            "owner_name": target.get("name", ""),
            "owner_mobile": target.get("mobile", ""),
            "owner_deleted": True,
        }},
    )

    users.delete_one({"_id": oid})
    return jsonify({"ok": True})


@app.route("/api/admin/statements")
@admin_required
def admin_statements():
    # Every statement across all users, newest first. Owner details are joined
    # live from the users collection; for statements whose owner was deleted we
    # fall back to the stamped owner_name/owner_mobile.
    q = (request.args.get("q") or "").strip()

    docs = list(statements.find(
        {}, {"data": 0, "pdf": 0, "pdf_password": 0}
    ).sort("created_at", -1))

    # Resolve current owners in one pass.
    owner_ids = {
        _to_oid(d.get("user_id")) for d in docs if _to_oid(d.get("user_id"))
    }
    owners = {}
    if owner_ids:
        for u in users.find(
            {"_id": {"$in": list(owner_ids)}},
            {"name": 1, "mobile": 1, "role": 1, "is_admin": 1, "is_super_admin": 1},
        ):
            owners[str(u["_id"])] = u

    out = []
    for d in docs:
        live = owners.get(d.get("user_id"))
        # This view is about end users — skip statements uploaded by staff.
        if live and user_role(live) in STAFF_ROLES:
            continue
        if live:
            owner_name = live.get("name", "")
            owner_mobile = live.get("mobile", "")
            owner_deleted = False
        else:
            owner_name = d.get("owner_name", "")
            owner_mobile = d.get("owner_mobile", "")
            owner_deleted = bool(d.get("owner_deleted")) or not d.get("user_id")

        issue = d.get("issue") or {}
        row = {
            "id": str(d["_id"]),
            "filename": d.get("filename", ""),
            "bank": d.get("bank") or "",
            "row_count": d.get("row_count", 0),
            "page_count": d.get("page_count", 0),
            "owner_name": owner_name,
            "owner_mobile": owner_mobile,
            "owner_deleted": owner_deleted,
            # pdf bytes are excluded from this query; content_type is the
            # reliable "a source PDF was stored" signal.
            "has_pdf": bool(d.get("content_type")),
            # Conversion outcome — older docs predate the field, treat as success.
            "status": d.get("status", "success"),
            "issue_status": issue.get("status"),
            "created_at": _iso(d.get("created_at")),
        }

        if q:
            hay = " ".join([
                row["filename"], row["bank"], owner_name, owner_mobile
            ]).lower()
            if q.lower() not in hay:
                continue
        out.append(row)

    return jsonify({"statements": out})


@app.route("/api/admin/statements/<sid>")
@admin_required
def admin_get_statement(sid):
    # Full extracted data for one statement (admins are not owner-scoped), so
    # the admin can open any user's result in results.html.
    oid = _to_oid(sid)
    if not oid:
        return jsonify({"error": "Not found"}), 404

    doc = statements.find_one(
        {"_id": oid},
        {"pdf": 0, "pdf_password": 0},
    )
    if not doc:
        return jsonify({"error": "Not found"}), 404

    return jsonify({
        "headers": doc.get("headers", []),
        "data": doc.get("data", []),
        "filename": doc.get("filename", ""),
        "page_count": doc.get("page_count", 0),
    })


@app.route("/api/admin/statements/<sid>/pdf")
@admin_required
def admin_statement_pdf(sid):
    # Download any user's source PDF (admins are not owner-scoped here).
    oid = _to_oid(sid)
    if not oid:
        return jsonify({"error": "Not found"}), 404

    doc = statements.find_one(
        {"_id": oid},
        {"pdf": 1, "filename": 1, "content_type": 1, "pdf_password": 1},
    )
    if not doc or not doc.get("pdf"):
        return jsonify({"error": "Not found"}), 404

    pdf_bytes = bytes(doc["pdf"])
    if doc.get("pdf_password"):
        pdf_bytes = _decrypt_pdf_bytes(pdf_bytes, doc["pdf_password"])

    inline = request.args.get("inline") in ("1", "true", "yes")
    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype=doc.get("content_type", "application/pdf"),
        as_attachment=not inline,
        download_name=doc.get("filename") or "statement.pdf",
    )


@app.route("/api/admin/messages")
@staff_required
def admin_messages():
    # Contact-form inbox. Scoped by role: admins see general (contactus)
    # enquiries, salespeople see sales (contact) enquiries, the super admin
    # sees both. Sales = source != "contactus"; General = source == "contactus".
    me = current_user_doc()
    role = user_role(me)
    query = {}
    if role == "admin":
        query = {"source": "contactus"}
    elif role == "salesperson":
        # A salesperson only sees sales enquiries assigned to them.
        query = {"source": {"$ne": "contactus"}, "salesperson_id": str(me["_id"])}

    docs = messages.find(query).sort("created_at", -1)
    out = []
    for d in docs:
        out.append({
            "id": str(d["_id"]),
            "source": d.get("source", ""),
            "first_name": d.get("first_name", ""),
            "last_name": d.get("last_name", ""),
            "email": d.get("email", ""),
            "phone": d.get("phone", ""),
            "org_name": d.get("org_name", ""),
            "org_type": d.get("org_type", ""),
            "subject": d.get("subject", ""),
            "message": d.get("message", ""),
            "salesperson_id": d.get("salesperson_id", ""),
            "salesperson_name": d.get("salesperson_name", ""),
            "status": d.get("status", "open"),
            "created_at": _iso(d.get("created_at")),
            "decided_at": _iso(d.get("decided_at")),
        })

    # List of salespeople for the super admin's assignment dropdown.
    salespeople = []
    if role == "super_admin":
        for sp in users.find({"role": "salesperson"}, {"name": 1, "mobile": 1}).sort("name", 1):
            salespeople.append({
                "id": str(sp["_id"]),
                "name": sp.get("name", "") or sp.get("mobile", ""),
            })

    return jsonify({"messages": out, "salespeople": salespeople})


@app.route("/api/admin/messages/<mid>/status", methods=["POST"])
@require_roles("super_admin", "salesperson")
def admin_set_message_status(mid):
    # Set a sales enquiry's outcome: open / purchased / not_purchased.
    oid = _to_oid(mid)
    if not oid:
        return jsonify({"error": "Not found"}), 404

    data = request.get_json(silent=True) or {}
    status = (data.get("status") or "").strip()
    if status not in ("open", "purchased", "not_purchased"):
        return jsonify({"error": "Invalid status"}), 400

    doc = messages.find_one({"_id": oid})
    if not doc:
        return jsonify({"error": "Not found"}), 404

    # A salesperson can only update enquiries assigned to them.
    me = current_user_doc()
    if user_role(me) == "salesperson" and doc.get("salesperson_id") != str(me["_id"]):
        return jsonify({"error": "FORBIDDEN"}), 403

    # Stamp when the enquiry was closed (purchased / not_purchased); clear it
    # again if it's reopened.
    update = {"status": status}
    update["decided_at"] = (
        datetime.now(timezone.utc)
        if status in ("purchased", "not_purchased") else None
    )

    messages.update_one({"_id": oid}, {"$set": update})
    return jsonify({"ok": True, "status": status})


@app.route("/api/admin/messages/<mid>/salesperson", methods=["POST"])
@super_admin_required
def admin_set_salesperson(mid):
    # Assign (or clear) the salesperson handling a sales enquiry. Only the super
    # admin assigns. The id must belong to an actual salesperson account.
    oid = _to_oid(mid)
    if not oid:
        return jsonify({"error": "Not found"}), 404

    data = request.get_json(silent=True) or {}
    sp_id = (data.get("salesperson_id") or "").strip()

    if not sp_id:
        update = {"salesperson_id": "", "salesperson_name": ""}
    else:
        sp_oid = _to_oid(sp_id)
        sp = users.find_one({"_id": sp_oid}) if sp_oid else None
        if not sp or user_role(sp) != "salesperson":
            return jsonify({"error": "Not a salesperson"}), 400
        update = {
            "salesperson_id": str(sp["_id"]),
            "salesperson_name": sp.get("name", "") or sp.get("mobile", ""),
        }

    result = messages.update_one({"_id": oid}, {"$set": update})
    if result.matched_count == 0:
        return jsonify({"error": "Not found"}), 404
    return jsonify({"ok": True, **update})


@app.route("/api/admin/messages/<mid>/resolve", methods=["POST"])
@admin_required
def admin_resolve_message(mid):
    # Toggle a contact message between open and resolved.
    oid = _to_oid(mid)
    if not oid:
        return jsonify({"error": "Not found"}), 404

    data = request.get_json(silent=True) or {}
    new_status = "resolved" if data.get("resolved", True) else "open"

    update = {"status": new_status}
    if new_status == "resolved":
        update["resolved_at"] = datetime.now(timezone.utc)
        update["resolved_by"] = current_user()
    else:
        update["resolved_at"] = None
        update["resolved_by"] = None

    result = messages.update_one({"_id": oid}, {"$set": update})
    if result.matched_count == 0:
        return jsonify({"error": "Not found"}), 404
    return jsonify({"ok": True, "status": new_status})


@app.route("/api/admin/issues")
@admin_required
def admin_issues():
    # Statement issues raised by users, open first then newest. Owner details
    # are joined live, falling back to the stamped name for deleted owners.
    docs = list(statements.find(
        {"issue": {"$exists": True}},
        {"data": 0, "pdf": 0, "pdf_password": 0},
    ))

    owner_ids = {
        _to_oid(d.get("user_id")) for d in docs if _to_oid(d.get("user_id"))
    }
    owners = {}
    if owner_ids:
        for u in users.find(
            {"_id": {"$in": list(owner_ids)}},
            {"name": 1, "mobile": 1},
        ):
            owners[str(u["_id"])] = u

    out = []
    for d in docs:
        issue = d.get("issue") or {}
        live = owners.get(d.get("user_id"))
        owner_name = (live or {}).get("name") or d.get("owner_name", "")
        owner_mobile = (live or {}).get("mobile") or d.get("owner_mobile", "")
        out.append({
            "statement_id": str(d["_id"]),
            "filename": d.get("filename", ""),
            "owner_name": owner_name,
            "owner_mobile": owner_mobile,
            "owner_deleted": not live and bool(d.get("owner_deleted")),
            "note": issue.get("note", ""),
            "status": issue.get("status", "open"),
            "created_at": _iso(issue.get("created_at")),
            "resolved_at": _iso(issue.get("resolved_at")),
        })

    # Open issues first, then by recency within each group.
    open_first = [r for r in out if r["status"] != "resolved"]
    resolved = [r for r in out if r["status"] == "resolved"]
    open_first.sort(key=lambda r: r["created_at"] or "", reverse=True)
    resolved.sort(key=lambda r: r["created_at"] or "", reverse=True)
    return jsonify({"issues": open_first + resolved})


@app.route("/api/admin/statements/<sid>/issue/resolve", methods=["POST"])
@admin_required
def admin_resolve_issue(sid):
    # Mark a statement's issue resolved (or reopen it).
    oid = _to_oid(sid)
    if not oid:
        return jsonify({"error": "Not found"}), 404

    data = request.get_json(silent=True) or {}
    resolve = data.get("resolved", True)

    if resolve:
        update = {
            "issue.status": "resolved",
            "issue.resolved_at": datetime.now(timezone.utc),
            "issue.resolved_by": current_user(),
        }
    else:
        update = {
            "issue.status": "open",
            "issue.resolved_at": None,
            "issue.resolved_by": None,
        }

    result = statements.update_one(
        {"_id": oid, "issue": {"$exists": True}},
        {"$set": update},
    )
    if result.matched_count == 0:
        return jsonify({"error": "Not found"}), 404
    return jsonify({"ok": True})


def _save_failed_statement(pdf_file, tmp_path, page_count, reason, password=None, bank=None):
    # Record a failed conversion so it still shows up in the owner's statements
    # list (flagged as failed) and the source PDF can be re-downloaded. Mirrors
    # the success-path document but with status="failed" and no extracted rows.
    # Best-effort and guarded: a logging failure must never mask the real error,
    # and failed attempts are NOT charged against the user's page credit.
    if not current_user():
        return None

    try:
        with open(tmp_path, "rb") as fh:
            pdf_bytes = fh.read()
    except OSError:
        pdf_bytes = None

    doc = {
        "user_id": current_user(),
        "filename": getattr(pdf_file, "filename", "") or "",
        "bank": bank,
        "content_type": "application/pdf",
        "status": "failed",
        "error": reason,
        "headers": [],
        "data": [],
        "row_count": 0,
        "page_count": page_count or 0,
        "created_at": datetime.now(timezone.utc),
    }
    if pdf_bytes is not None:
        doc["pdf"] = pdf_bytes
    if password:
        doc["pdf_password"] = password

    try:
        result = statements.insert_one(doc)
        return str(result.inserted_id)
    except Exception as exc:
        app.logger.warning("Could not save failed statement: %s", exc)
        return None


@app.route("/process", methods=["POST"])
def process():

    if "pdf" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    pdf_file = request.files["pdf"]

    password = request.form.get("password", None) or None

    # Bank the user selected on the upload page (free-typed when "Other").
    bank = request.form.get("bank", "").strip() or None

    with tempfile.NamedTemporaryFile(
        suffix=".pdf",
        delete=False
    ) as tmp:

        pdf_file.save(tmp.name)

        tmp_path = tmp.name

    # Defined up front so the except handler can reference them even if the
    # failure happens before extraction. saved_ok guards against recording a
    # failed attempt when a success record was already written.
    page_count = 0
    saved_ok = False

    try:

        # Shared password / encryption gate (covers BOTH the lattice and
        # stream paths) so the front-end's password pill behaves the same
        # regardless of which extractor ends up running.
        reader = PdfReader(tmp_path)
        unlock_status, password = _unlock_pdf(reader, password)
        if unlock_status == "need":
            return jsonify({"error": "PASSWORD_REQUIRED"}), 200
        if unlock_status == "wrong":
            return jsonify({"error": "WRONG_PASSWORD"}), 200

        # Total pages in the source PDF — stored alongside the extraction so the
        # statements list can show it. Reading after the decrypt gate ensures
        # encrypted files report their real page count. NB: decrypting AES
        # page content requires PyCryptodome (see requirements.txt); without it
        # this raises DependencyError and pages would silently read as 0, so we
        # log any failure rather than swallowing it.
        try:
            page_count = len(reader.pages)
        except Exception as exc:
            app.logger.warning("Could not count PDF pages: %s", exc)
            page_count = 0

        # Scanned/image-only statements have no text layer, so every extractor
        # would return nothing usable. Detect them up front (cheap PyMuPDF probe,
        # no OCR) and tell the user plainly instead of running — and being
        # charged for — an extraction that is doomed to fail.
        if is_scanned_pdf(tmp_path, password=password):
            _save_failed_statement(
                pdf_file, tmp_path, page_count,
                "This looks like a scanned (image-based) PDF.",
                password, bank,
            )
            saved_ok = True
            return jsonify({
                "error": "SCANNED_PDF",
                "message": "This looks like a scanned (image-based) PDF. We can "
                           "only read statements with selectable text — please "
                           "upload a digital PDF exported from your bank."
            }), 200

        # Free-plan page-credit gate. Registered (non-admin) users may extract
        # up to their page_limit; admins are unlimited. Read live from the DB
        # so a limit/usage changed there (or in the admin console) takes effect
        # on the next upload. Checked BEFORE the expensive extraction so a user
        # who is out of credit isn't made to wait for work we'll discard.
        gate_user = current_user_doc()
        if gate_user and user_role(gate_user) not in STAFF_ROLES:
            limit = int(gate_user.get("page_limit", FREE_PAGE_LIMIT))
            used = int(gate_user.get("pages_used", 0))
            remaining = max(0, limit - used)

            if remaining <= 0:
                # Save the statement (flagged failed, no rows) so it still
                # appears in the owner's list and the source PDF can be
                # re-downloaded — but don't extract or charge any pages.
                stmt_id = _save_failed_statement(
                    pdf_file, tmp_path, page_count,
                    (f"You've used all {limit} free pages. "
                     f"Upgrade your plan to continue."),
                    password, bank,
                )
                saved_ok = True
                return jsonify({
                    "error": "PAGE_LIMIT_REACHED",
                    "message": (f"You've used all {limit} free pages. "
                                f"Upgrade your plan to continue."),
                    "pages_used": used,
                    "page_limit": limit,
                    "pages_remaining": 0,
                    "statement_id": stmt_id,
                }), 200

            if page_count > remaining:
                # Statement is too long for the remaining credit: we won't
                # extract or charge, but still save it (flagged failed) so it
                # shows in the owner's list and the PDF stays retrievable.
                stmt_id = _save_failed_statement(
                    pdf_file, tmp_path, page_count,
                    (f"This statement has {page_count} pages but you have only "
                     f"{remaining} of {limit} free pages left. Upgrade your "
                     f"plan to continue."),
                    password, bank,
                )
                saved_ok = True
                return jsonify({
                    "error": "INSUFFICIENT_PAGES",
                    "message": (f"This statement has {page_count} pages but you "
                                f"have only {remaining} of {limit} free pages "
                                f"left. Upgrade your plan to continue."),
                    "pages_used": used,
                    "page_limit": limit,
                    "pages_remaining": remaining,
                    "statement_id": stmt_id,
                }), 200

        # Decide how to extract: a ruled tabular grid (lattice) goes through
        # the DistilBERT row-classifier pipeline as before; a structure-less
        # statement (stream) goes through ledger_extract.
        classification = classify_pdf_fast(tmp_path, password=password)

        if classification.category == "stream":
            records, headers, error = process_pdf_stream(tmp_path, password)
        else:
            records, headers, error = process_pdf(tmp_path, password)

        if error == "PASSWORD_REQUIRED":
            return jsonify({
                "error": "PASSWORD_REQUIRED"
            }), 200

        if error == "WRONG_PASSWORD":
            return jsonify({
                "error": "WRONG_PASSWORD"
            }), 200

        if error == "TABLE-LESS":
            _save_failed_statement(
                pdf_file, tmp_path, page_count,
                "We couldn't read any transactions from this PDF",
                password, bank,
            )
            return jsonify({
                "error": "We couldn't read any transactions from this PDF"
            }), 200

        # Reverse-chronological statements (newest-first, e.g. Bandhan) are
        # flipped to oldest-first before persisting/returning so the table,
        # balance summary and exports all read chronologically.
        records = _to_chronological(records, headers)

        # Drop any date column that is empty in every row (e.g. SBI's blank
        # "Txn Date" alongside a populated "Value Date"). Done before persist +
        # return so the table, CSV and XML export never see the dead column.
        records, headers = _drop_empty_date_columns(records, headers)

        # Likewise drop an all-blank identifier column (e.g. an empty
        # "Reference" column on statements with no dedicated Chq./Ref. column,
        # such as Indian Bank) so it never reaches the table, CSV or XML.
        records, headers = _drop_empty_optional_columns(records, headers)

        # Persist the extraction for logged-in users so they can revisit it.
        # We also store the original PDF inline as BSON binary so the user can
        # re-download the source file. pymongo stores Python `bytes` as a BSON
        # binary field directly. Read from the temp file (still present here —
        # the finally block deletes it after we return).
        statement_id = None
        if current_user():
            with open(tmp_path, "rb") as fh:
                pdf_bytes = fh.read()

            stmt_doc = {
                "user_id": current_user(),
                "filename": pdf_file.filename,
                "bank": bank,
                "pdf": pdf_bytes,
                "content_type": "application/pdf",
                "status": "success",
                "headers": headers,
                "data": records,
                "row_count": len(records),
                "page_count": page_count,
                "created_at": datetime.now(timezone.utc),
            }

            # If the PDF was password-protected, keep the password the user
            # entered so we can decrypt it on later view/download and never
            # prompt them again. Stored as-is (not hashed) because we must be
            # able to reuse it to unlock the file — see the security note this
            # implies: anyone with DB access can open the statement.
            if password:
                stmt_doc["pdf_password"] = password

            result = statements.insert_one(stmt_doc)
            statement_id = str(result.inserted_id)
            saved_ok = True

            # Charge the extracted pages against the user's free-plan credit.
            # $inc creates pages_used for older accounts that predate the field.
            # Admins are tracked too but never blocked by the gate above.
            if page_count:
                uid_oid = _to_oid(current_user())
                if uid_oid:
                    users.update_one(
                        {"_id": uid_oid},
                        {"$inc": {"pages_used": page_count}}
                    )

        return jsonify({
            "headers": headers,
            "data": records,
            "page_count": page_count,
            "statement_id": statement_id
        })

    except Exception as e:

        # Log the real error server-side for debugging, but return a generic
        # message so internal details (paths, library internals, DB errors)
        # aren't leaked to — or reflected back into — the client.
        app.logger.exception("Error processing PDF: %s", e)

        # Record the failed attempt too (unless we already saved a success
        # before the error), so it surfaces in the owner's statements list.
        if not saved_ok:
            _save_failed_statement(
                pdf_file, tmp_path, page_count,
                "Something went wrong while processing the PDF.",
                password, bank,
            )

        return jsonify({
            "error": "Something went wrong while processing the PDF."
        }), 500

    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    app.run(debug=True, port=5000)