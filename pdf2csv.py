import os
import camelot
import pandas as pd
from PyPDF2 import PdfReader

# ----------------------------------------
# PDF FILE PATH
# ----------------------------------------

pdf_path = r"D:\ledgerit\latticePDF\ICICI Bank2.pdf"

# ----------------------------------------
# OUTPUT FOLDER
# ----------------------------------------

output_folder = r"C:\Users\ahmed\Desktop\ledgerit\output"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# ----------------------------------------
# OUTPUT CSV PATH
# ----------------------------------------

pdf_name = os.path.splitext(
    os.path.basename(pdf_path)
)[0]

csv_output_path = os.path.join(
    output_folder,
    f"{pdf_name}.csv"
)

# ----------------------------------------
# PDF PROCESSING
# ----------------------------------------

try:

    # Check if PDF is encrypted
    reader = PdfReader(pdf_path)

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
            pdf_path,
            pages="all",
            flavor="lattice",
            password=password
        )

    else:

        print("PDF is not password protected.")

        tables = camelot.read_pdf(
            pdf_path,
            pages="all",
            flavor="lattice"
        )

    # ----------------------------------------
    # CHECK IF TABLES EXIST
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

    # ----------------------------------------
    # SAVE CSV
    # ----------------------------------------

    final_df.to_csv(
        csv_output_path,
        index=False
    )

    print(
        f"CSV saved successfully at:\n{csv_output_path}"
    )

except Exception as e:

    print(
        "Error:",
        str(e)
    )