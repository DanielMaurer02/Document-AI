from pathlib import Path


REPO = "Qwen/Qwen2.5-7B-Instruct-GGUF"
FILES = [
    "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
    "qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf",
]
# The primary model file (first part) that llama-cpp will load
PRIMARY_FILE = FILES[0]
DEST = Path("/models") / PRIMARY_FILE
