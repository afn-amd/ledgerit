import os
import pandas as pd

# ----------------------------------------
# INPUT CSV
# ----------------------------------------

input_csv = r"C:\Users\ahmed\Desktop\ledgerit\output\110005500732_2025-10-01_2025-12-31_20072026151340.csv"

# ----------------------------------------
# OUTPUT CSV
# ----------------------------------------

output_csv = r"C:\Users\ahmed\Desktop\ledgerit\output\110005500732_2025-10-01_2025-12-31_20072026151340.csv"

# ----------------------------------------
# LOAD CSV
# ----------------------------------------

df = pd.read_csv(
    input_csv,
    dtype=str
).fillna("")

# ----------------------------------------
# LAST COLUMN = LABEL
# ----------------------------------------

label_col = df.columns[-1]

# ----------------------------------------
# CREATE TEXT COLUMN
# ----------------------------------------

texts = []

for _, row in df.iterrows():

    text = " ".join(
        str(value).strip()
        for value in row[:-1]
        if str(value).strip() != ""
    )

    text = " ".join(text.split())

    texts.append(text)

# ----------------------------------------
# CREATE NEW DATAFRAME
# ----------------------------------------

new_df = pd.DataFrame({
    "text": texts,
    "label": df[label_col]
})

# ----------------------------------------
# SAVE CSV
# ----------------------------------------

new_df.to_csv(
    output_csv,
    index=False
)

print(
    f"Saved successfully:\n{output_csv}"
)