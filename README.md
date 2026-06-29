---
title: Ledgerit
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# Ledgerit

Ledgerit is a Flask web app for extracting transaction tables from bank statement PDFs and converting them into clean, structured records. The main application lives in `app.py`.

The app accepts a PDF upload, extracts tables with Camelot, classifies each extracted row with a local DistilBERT model, removes irrelevant rows, joins continuation rows, and returns normalized JSON that can be rendered by the frontend.

## What `app.py` Does

`app.py` provides:

- A Flask server for the Ledgerit web interface.
- PDF table extraction using Camelot's lattice mode.
- Password handling for encrypted PDFs.
- Row classification using a local Hugging Face DistilBERT model.
- Cleanup logic for headers, transaction rows, and continuation rows.
- JSON output containing detected table headers and extracted transaction records.

## Project Structure

```text
.
|-- app.py
|-- index.html
|-- results.html
|-- models/
|   `-- row_classifier/
|       |-- config.json
|       |-- model.safetensors
|       |-- tokenizer.json
|       `-- tokenizer_config.json
|-- requirements.txt
|-- requirements-cpu.txt
`-- Dockerfile
```

## Model

`app.py` loads the classifier from:

```text
models/row_classifier
```

The classifier predicts one of these labels:

| Label | Meaning |
| --- | --- |
| `header` | Table header row |
| `entry` | Main transaction row |
| `c-entry` | Continuation row for the previous transaction |
| `info` | Informational row |
| `remove` | Row to exclude from final output |

Rows labeled `remove` are discarded. Rows labeled `c-entry` are appended to the most recent `entry` row.

## Requirements

Python 3.11 is recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

For CPU-only deployment, use:

```bash
pip install -r requirements-cpu.txt
```

Camelot lattice extraction also requires Ghostscript to be installed on the system.

## Running Locally

Start the Flask app:

```bash
python app.py
```

The local development server runs on:

```text
http://127.0.0.1:5000
```

Open the URL in a browser and upload a bank statement PDF.

## Routes

### `GET /`

Serves `index.html`, the main upload page.

### `GET /results.html`

Serves `results.html`, the results page.

### `POST /process`

Processes an uploaded PDF and returns extracted records as JSON.

Expected form fields:

| Field | Required | Description |
| --- | --- | --- |
| `pdf` | Yes | PDF file to process |
| `password` | No | Password for encrypted PDFs |

Example request:

```bash
curl -X POST http://127.0.0.1:5000/process \
  -F "pdf=@statement.pdf"
```

Example request for a password-protected PDF:

```bash
curl -X POST http://127.0.0.1:5000/process \
  -F "pdf=@statement.pdf" \
  -F "password=your-password"
```

Successful response:

```json
{
  "headers": ["Value Date", "Post Date", "Details", "Ref No/Cheque No", "Debit", "Credit", "Balance"],
  "data": [
    {
      "Value Date": "01/01/2026",
      "Post Date": "01/01/2026",
      "Details": "Sample transaction",
      "Ref No/Cheque No": "ABC123",
      "Debit": "100.00",
      "Credit": "",
      "Balance": "900.00"
    }
  ]
}
```

Error responses:

| Error | Meaning |
| --- | --- |
| `No file uploaded` | Request did not include a `pdf` file |
| `PASSWORD_REQUIRED` | PDF is encrypted and no password was supplied |
| `WRONG_PASSWORD` | Supplied PDF password was incorrect |
| `We couldn't read any transactions from this PDF` | Camelot could not detect tables in the PDF |

## Processing Flow

1. The uploaded PDF is saved to a temporary file.
2. `PyPDF2` checks whether the file is encrypted.
3. Camelot extracts tables from all pages using `flavor="lattice"`.
4. Extracted rows are normalized by removing extra whitespace and line breaks.
5. Each row is classified by the DistilBERT model.
6. Duplicate headers and removable rows are filtered out.
7. Continuation rows are merged into the previous transaction.
8. The temporary PDF file is deleted.
9. The app returns JSON containing `headers` and `data`.

## Docker Deployment

Build the Docker image:

```bash
docker build -t ledgerit .
```

Run the container:

```bash
docker run -p 7860:7860 ledgerit
```

The container runs the app with Gunicorn on:

```text
http://localhost:7860
```

Before building, make sure the real model weights exist at:

```text
models/row_classifier/model.safetensors
```

If the project uses Git LFS, run:

```bash
git lfs pull
```

The Dockerfile checks that `model.safetensors` is not just a Git LFS pointer file.

## Notes

- The app is optimized for PDFs that contain ruled tables because Camelot uses lattice extraction.
- Scanned image-only PDFs may not work unless OCR is added before table extraction.
- The model is loaded once at startup, so startup may take a few seconds.
- Each Gunicorn worker loads its own copy of the model; the Dockerfile uses one worker to keep memory usage lower.
