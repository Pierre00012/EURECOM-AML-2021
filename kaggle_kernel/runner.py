
import os
import json
import subprocess
import tempfile
import shutil
from pathlib import Path

# ==================================================
# CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parents[1]
KAGGLE_DIR = ROOT / "kaggle_kernel"

NOTEBOOK_NAME = "classification-glaucome.ipynb"
METADATA_NAME = "kernel-metadata.json"

NOTEBOOK = KAGGLE_DIR / NOTEBOOK_NAME
METADATA = KAGGLE_DIR / METADATA_NAME

SHA = os.environ.get("GIT_SHA")

if not SHA:
    raise ValueError("GIT_SHA is missing.")

# ==================================================
# VERIFICATION
# ==================================================

if not NOTEBOOK.exists():
    raise FileNotFoundError(f"Notebook not found: {NOTEBOOK}")

if not METADATA.exists():
    raise FileNotFoundError(f"Metadata not found: {METADATA}")

print(f"Git commit: {SHA}")

# ==================================================
# PREPARATION DU NOTEBOOK
# ==================================================

with tempfile.TemporaryDirectory() as tmp:

    workdir = Path(tmp) / "kernel"
    workdir.mkdir()

    # Copier le notebook
    shutil.copy2(NOTEBOOK, workdir / NOTEBOOK_NAME)

    # Copier les métadonnées
    shutil.copy2(METADATA, workdir / METADATA_NAME)

    # Lire le notebook
    notebook_path = workdir / NOTEBOOK_NAME

    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    # Injecter le SHA dans les cellules
    for cell in notebook["cells"]:

        if cell["cell_type"] != "code":
            continue

        source = cell["source"]

        if isinstance(source, list):
            source = "".join(source)

        source = source.replace("__GIT_SHA__", SHA)

        cell["source"] = source.splitlines(keepends=True)

    # Sauvegarder le notebook
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

    # ==================================================
    # ENVOI VERS KAGGLE
    # ==================================================

    print("Uploading notebook to Kaggle...")

    subprocess.run(
        [
            "kaggle",
            "kernels",
            "push",
            "-p",
            str(workdir)
        ],
        check=True
    )

print("Notebook successfully submitted to Kaggle.")
