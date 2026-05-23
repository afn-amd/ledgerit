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

    df = original_df[
        original_df["predicted_label"] != "remove"
    ]

    fixed_rows = []

    for idx, row in df.iterrows():

        label = row["predicted_label"]

        row_values = row[:-1].tolist()

        if label == "header":
            row_values = split_header_row(row_values)

        elif label in ["entry", "c-entry"]:
            row_values = clean_entry_row(row_values)

        row_values.append(label)

        fixed_rows.append(row_values)

    df = pd.DataFrame(fixed_rows)

    df = df.rename(
        columns={df.columns[-1]: "predicted_label"}
    )

    header_indices = df[
        df["predicted_label"] == "header"
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