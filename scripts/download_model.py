"""
Pull both parts of Qwen2.5‑7B‑Instruct‑Q4_K_M.gguf and cache them in /models.
"""

from huggingface_hub import hf_hub_download
from pathlib import Path

# TODO: there was a problem with importing DEST from constants.py, so we import it here directly - Change this later
REPO = "Qwen/Qwen2.5-7B-Instruct-GGUF"
FILES = [
    "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
    "qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf",
]
# The primary model file (first part) that llama-cpp will load
PRIMARY_FILE = FILES[0]
DEST = Path("/models") / PRIMARY_FILE


import logging

logging.basicConfig(level=logging.INFO)


def main():
    DEST.parent.mkdir(parents=True, exist_ok=True)

    # Check if primary model file already exists
    if DEST.exists():
        logging.info("✔ primary model file already present")
        # Check if all parts are present
        missing_files = []
        for file in FILES:
            file_path = DEST.parent / file
            if not file_path.exists():
                missing_files.append(file)

        if not missing_files:
            logging.info("✔ all model parts already present")
            return
        else:
            logging.info(f"⚠ missing model parts: {missing_files}")

    logging.info("⬇ downloading model parts …")

    # Download all parts of the model
    for i, file in enumerate(FILES, 1):
        file_path = DEST.parent / file
        if file_path.exists():
            logging.info(f"✔ part {i}/2 already exists: {file}")
            continue

        logging.info(f"⬇ downloading part {i}/2: {file}")
        hf_hub_download(
            repo_id=REPO,
            filename=file,
            local_dir=DEST.parent,
            local_dir_use_symlinks=False,
        )
        logging.info(f"✅ downloaded part {i}/2: {file}")

    logging.info(f"✅ all model parts saved to {DEST.parent}")
    logging.info(f"Primary model file: {DEST}")


if __name__ == "__main__":
    main()
