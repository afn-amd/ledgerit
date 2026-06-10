import os
import json
import gc
import tempfile
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from flask import Flask, request, jsonify, send_from_directory
import camelot
import pandas as pd
import torch
from PyPDF2 import PdfReader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

app = Flask(__name__, static_folder=".")

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

                while next_index < len(new_row):

                    if str(new_row[next_index]).strip() == "":
                        new_row[next_index] = part
                        next_index += 1
                        break

                    next_index += 1

    return new_row


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


def looks_like_header(row_values):
    # Join the row's cells into one lowercase string and count how many
    # distinct keyword groups appear. Two or more -> it's a header.
    joined = " ".join(str(v) for v in row_values).lower()

    if not joined.strip():
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

    return groups_matched >= 2


def process_pdf(pdf_path, password=None):

    reader = PdfReader(pdf_path)

    if reader.is_encrypted:

        if not password:
            return None, None, "PASSWORD_REQUIRED"

        status = reader.decrypt(password)

        if status == 0:
            return None, None, "WRONG_PASSWORD"

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

    all_dfs = [t.df for t in tables if not t.df.empty]

    if not all_dfs:
        return None, None, "TABLE-LESS"

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

    gc.collect()

    return records, headers, None


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/results.html")
def results_page():
    return send_from_directory(".", "results.html")


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


@app.route("/process", methods=["POST"])
def process():

    if "pdf" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    pdf_file = request.files["pdf"]

    password = request.form.get("password", None) or None

    with tempfile.NamedTemporaryFile(
        suffix=".pdf",
        delete=False
    ) as tmp:

        pdf_file.save(tmp.name)

        tmp_path = tmp.name

    try:

        records, headers, error = process_pdf(
            tmp_path,
            password
        )

        if error == "PASSWORD_REQUIRED":
            return jsonify({
                "error": "PASSWORD_REQUIRED"
            }), 200

        if error == "WRONG_PASSWORD":
            return jsonify({
                "error": "WRONG_PASSWORD"
            }), 200

        if error == "TABLE-LESS":
            return jsonify({
                "error": "No tables found in PDF"
            }), 200

        return jsonify({
            "headers": headers,
            "data": records
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500

    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    app.run(debug=True, port=5000)