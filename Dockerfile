FROM python:3.11-slim

# System dependencies:
#   ghostscript     -> required by Camelot's lattice table extraction
#   libglib2.0-0    -> shared lib some PDF/CV deps load at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        ghostscript \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Install CPU-only PyTorch from the official CPU wheel index.
#    (Your repo pinned +cu121 GPU wheels, which the app never uses and
#     which fail to install on a CPU host.)
RUN pip install --no-cache-dir torch==2.5.1 \
        --index-url https://download.pytorch.org/whl/cpu

# 2) Install everything else from PyPI (torch lines already stripped,
#    gunicorn added).
COPY requirements-cpu.txt .
RUN pip install --no-cache-dir -r requirements-cpu.txt

# 3) Copy the app. IMPORTANT: the build context must contain the REAL
#    model weights (~268 MB), not the 134-byte Git LFS pointer.
#    Run `git lfs pull` locally before building, or the model won't load.
COPY . .

# Fail fast at build time if the model is still an LFS pointer stub.
RUN python -c "import os; p='models/row_classifier/model.safetensors'; s=os.path.getsize(p); assert s > 1_000_000, f'model.safetensors is only {s} bytes - this is a Git LFS pointer, not the real weights. Run: git lfs pull'"

EXPOSE 7860

# Single worker keeps memory low (each worker loads its own ~268 MB model).
# Long timeout because PDF table extraction can take a while.
# Binds to $PORT when the platform sets it (Render, etc.), else 7860 (HF Spaces).
CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:${PORT:-7860} --workers 1 --threads 4 --timeout 300"]
