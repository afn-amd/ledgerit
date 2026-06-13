import os
import json
import camelot
import pandas as pd
import torch
import warnings
import gc

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning
)

from PyPDF2 import PdfReader

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification
)

# ----------------------------------------
# PATHS
# ----------------------------------------

_BASE = os.path.dirname(__file__)

PDF_PATH = os.path.join(_BASE, "latticePDF", "Bandhan Bank.pdf")

MODEL_PATH = os.path.join(_BASE, "models", "row_classifier")

# ----------------------------------------
# OUTPUT FOLDERS
# ----------------------------------------

JSON_OUTPUT_FOLDER = os.path.join(_BASE, "cellJSON")

INFO_OUTPUT_FOLDER = os.path.join(_BASE, "infoJSON")

# Create folders if they don't exist
os.makedirs(JSON_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(INFO_OUTPUT_FOLDER, exist_ok=True)

# ----------------------------------------
# PDF NAME
# ----------------------------------------

pdf_name = os.path.splitext(
    os.path.basename(PDF_PATH)
)[0]

# ----------------------------------------
# OUTPUT FILES
# ----------------------------------------

OUTPUT_JSON = os.path.join(
    JSON_OUTPUT_FOLDER,
    f"{pdf_name}.json"
)

INFO_OUTPUT = os.path.join(
    INFO_OUTPUT_FOLDER,
    f"{pdf_name}_info.json"
)

# ----------------------------------------
# LOAD MODEL
# ----------------------------------------

tokenizer = DistilBertTokenizerFast.from_pretrained(
    MODEL_PATH
)

model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.eval()

# ----------------------------------------
# LABEL MAP
# ----------------------------------------

id2label = {
    0: "header",
    1: "entry",
    2: "c-entry",
    3: "info",
    4: "remove"
}

# ----------------------------------------
# PRE-CLEAN RAW DATAFRAME
# ----------------------------------------

def preprocess_raw_df(df):

    cleaned_rows = []

    for _, row in df.iterrows():

        new_row = []

        for cell in row:

            cell = str(cell)

            # Replace newline with space
            cell = cell.replace("\n", "")

            # Remove extra spaces
            cell = " ".join(cell.split())

            new_row.append(cell)

        cleaned_rows.append(new_row)

    return pd.DataFrame(cleaned_rows)

# ----------------------------------------
# FIX HEADER ROWS
# ----------------------------------------

def split_header_row(row):

    row = [str(cell).strip() for cell in row]

    new_row = row.copy()

    for i, cell in enumerate(row):

        if "\n" in cell:

            parts = [
                p.strip()
                for p in cell.split("\n")
                if p.strip() != ""
            ]

            # Keep first part
            new_row[i] = parts[0]

            # Fill adjacent empty cells
            next_index = i + 1

            for part in parts[1:]:

                while next_index < len(new_row):

                    if str(new_row[next_index]).strip() == "":

                        new_row[next_index] = part
                        next_index += 1
                        break

                    next_index += 1

    return new_row

# ----------------------------------------
# FIX ENTRY ROWS
# ----------------------------------------

def clean_entry_row(row):

    cleaned = []

    for cell in row:

        cell = str(cell)

        # Replace newline with space
        cell = cell.replace("\n", " ")

        # Remove extra spaces
        cell = " ".join(cell.split())

        cleaned.append(cell)

    return cleaned

# ----------------------------------------
# EXTRACT TABLES FROM PDF
# ----------------------------------------

try:

    reader = PdfReader(PDF_PATH)

    # ----------------------------------------
    # PASSWORD CHECK
    # ----------------------------------------

    if reader.is_encrypted:

        password = input(
            "PDF is password protected. Enter password: "
        )

        decrypt_status = reader.decrypt(password)

        if decrypt_status == 0:

            print("WRONG PASSWORD")
            exit()

        print("Password accepted!")

        tables = camelot.read_pdf(
            PDF_PATH,
            pages="all",
            flavor="lattice",
            password=password,
            suppress_stdout=True
        )

    else:

        print("PDF is not password protected.")

        tables = camelot.read_pdf(
            PDF_PATH,
            pages="all",
            flavor="lattice",
            suppress_stdout=True
        )

    # ----------------------------------------
    # CHECK TABLES
    # ----------------------------------------

    if len(tables) == 0:

        print("TABLE-LESS")
        exit()

    all_dfs = [
        table.df
        for table in tables
        if not table.df.empty
    ]

    if len(all_dfs) == 0:

        print("TABLE-LESS")
        exit()

    # ----------------------------------------
    # COMBINE TABLES
    # ----------------------------------------

    final_df = pd.concat(
        all_dfs,
        ignore_index=True
    )

    if final_df.empty:

        print("TABLE-LESS")
        exit()

    print("\nTABLE EXTRACTION DONE")

except Exception as e:

    print("PDF Extraction Error:", str(e))
    exit()

# ----------------------------------------
# PRESERVE ORIGINAL
# ----------------------------------------

original_df = final_df.copy()

# ----------------------------------------
# CLEAN FOR PREDICTION
# ----------------------------------------

df = preprocess_raw_df(final_df)

# ----------------------------------------
# CREATE TEXT
# ----------------------------------------

texts = []

for _, row in df.iterrows():

    text = " ".join(
        str(cell).strip()
        for cell in row
        if str(cell).strip() != ""
    )

    text = " ".join(text.split())

    texts.append(text)

# ----------------------------------------
# PREDICT LABELS
# ----------------------------------------

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

    prediction = torch.argmax(
        outputs.logits,
        dim=1
    ).item()

    label = id2label[prediction]

    predicted_labels.append(label)

# ----------------------------------------
# ADD LABELS
# ----------------------------------------

original_df["predicted_label"] = predicted_labels

# ----------------------------------------
# STORE INFO ROWS
# ----------------------------------------

info_rows = original_df[
    original_df["predicted_label"] == "info"
]

info_rows.to_json(
    INFO_OUTPUT,
    orient="records",
    indent=4
)

# ----------------------------------------
# REMOVE REMOVE ROWS
# ----------------------------------------

df = original_df[
    original_df["predicted_label"] != "remove"
]

# ----------------------------------------
# APPLY ROW FIXES
# ----------------------------------------

fixed_rows = []

for idx, row in df.iterrows():

    label = row["predicted_label"]

    row_values = row[:-1].tolist()

    # ----------------------------------------
    # HEADER
    # ----------------------------------------

    if label == "header":

        row_values = split_header_row(
            row_values
        )

    # ----------------------------------------
    # ENTRY / C-ENTRY
    # ----------------------------------------

    elif label in ["entry", "c-entry"]:

        row_values = clean_entry_row(
            row_values
        )

    # Add label back
    row_values.append(label)

    fixed_rows.append(row_values)

# ----------------------------------------
# REBUILD DATAFRAME
# ----------------------------------------

df = pd.DataFrame(fixed_rows)

df = df.rename(
    columns={
        df.columns[-1]: "predicted_label"
    }
)

# ----------------------------------------
# KEEP FIRST HEADER ONLY
# ----------------------------------------

header_indices = df[
    df["predicted_label"] == "header"
].index.tolist()

# ----------------------------------------
# CASE 1 -> HEADER EXISTS
# ----------------------------------------

if len(header_indices) > 0:

    first_header_index = header_indices[0]

    header_row = df.loc[first_header_index]

    # Remove duplicate headers
    df = df.drop(
        df[
            (df["predicted_label"] == "header")
            &
            (df.index != first_header_index)
        ].index
    )

    headers = []

    for val in header_row[:-1]:

        val = str(val).strip()

        if val == "":

            val = "UNKNOWN"

        headers.append(val)

# ----------------------------------------
# CASE 2 -> NO HEADER FOUND
# ----------------------------------------

else:

    print("\nNO HEADER FOUND -> USING DEFAULT HEADERS")

    default_headers = [
        "Value Date",
        "Post Date",
        "Details",
        "Ref No/Cheque No",
        "Debit",
        "Credit",
        "Balance"
    ]

    # Get maximum columns excluding label column
    num_cols = len(df.columns) - 1

    # Trim or expand default headers
    headers = default_headers[:num_cols]

    while len(headers) < num_cols:

        headers.append(
            f"UNKNOWN_{len(headers)+1}"
        )

# ----------------------------------------
# PROCESS ENTRIES
# ----------------------------------------

records = []

current_record = None

for idx, row in df.iterrows():

    label = row["predicted_label"]

    row_values = row[:-1].tolist()

    # ----------------------------------------
    # ENTRY
    # ----------------------------------------

    if label == "entry":

        current_record = {}

        for h, v in zip(headers, row_values):

            current_record[h] = str(v).strip()

        records.append(current_record)

    # ----------------------------------------
    # CONTINUATION ENTRY
    # ----------------------------------------

    elif label == "c-entry":

        if current_record is not None:

            for h, v in zip(headers, row_values):

                v = str(v).strip()

                if v != "":

                    current_record[h] += " " + v

# ----------------------------------------
# SAVE JSON
# ----------------------------------------

with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        records,
        f,
        indent=4,
        ensure_ascii=False
    )

print("\nDONE")

print(f"\nJSON saved at:\n{OUTPUT_JSON}")

print(f"\nInfo rows saved at:\n{INFO_OUTPUT}")

gc.collect()