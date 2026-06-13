import os
import json
import pandas as pd
import torch

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification
)

# ----------------------------------------
# PATHS
# ----------------------------------------

_BASE = os.path.dirname(__file__)

MODEL_PATH = os.path.join(_BASE, "models", "row_classifier")

INPUT_CSV = os.path.join(_BASE, "output", "new.csv")

OUTPUT_JSON = os.path.join(_BASE, "final_output.json")

INFO_OUTPUT = os.path.join(_BASE, "info_rows.json")

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
# PRE-CLEAN RAW CSV
# ----------------------------------------

def preprocess_raw_csv(df):

    cleaned_rows = []

    for _, row in df.iterrows():

        new_row = []

        for cell in row:

            cell = str(cell)

            # Replace newline with space
            cell = cell.replace("\n", " ")

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

        # Check multiline header cell
        if "\n" in cell:

            parts = [
                p.strip()
                for p in cell.split("\n")
                if p.strip() != ""
            ]

            # Keep first part in same cell
            new_row[i] = parts[0]

            # Fill next empty adjacent cells
            next_index = i + 1

            for part in parts[1:]:

                while next_index < len(new_row):

                    # Fill only empty cells
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
# LOAD CSV
# ----------------------------------------

raw_df = pd.read_csv(INPUT_CSV, dtype=str).fillna("")

# Preserve original raw dataframe
original_df = raw_df.copy()

# Clean dataframe for prediction
df = preprocess_raw_csv(raw_df)

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

# Add labels to ORIGINAL dataframe
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
    # HEADER ROW
    # ----------------------------------------

    if label == "header":

        row_values = split_header_row(row_values)

    # ----------------------------------------
    # ENTRY / C-ENTRY ROW
    # ----------------------------------------

    elif label in ["entry", "c-entry"]:

        row_values = clean_entry_row(row_values)

    # Add label back
    row_values.append(label)

    fixed_rows.append(row_values)

# ----------------------------------------
# REBUILD DATAFRAME
# ----------------------------------------

df = pd.DataFrame(fixed_rows)

# Rename last column
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

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:

    json.dump(
        records,
        f,
        indent=4,
        ensure_ascii=False
    )

print("\nDONE")
print(f"\nJSON saved at:\n{OUTPUT_JSON}")
print(f"\nInfo rows saved at:\n{INFO_OUTPUT}")