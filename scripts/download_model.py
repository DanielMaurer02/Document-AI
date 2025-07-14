"""
Pull Qwen2.5‑7B‑Instruct‑Q4_K_M.gguf once and cache it in /models.
"""
from huggingface_hub import hf_hub_download
from ai_service.utils.constants import REPO, FILE, DEST


def main():
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists():
        print("✔ model already present")
        return
    print("⬇ downloading model …")
    hf_hub_download(
        repo_id=REPO,
        filename=FILE,
        local_dir=DEST.parent,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print("✅ saved to", DEST)

if __name__ == "__main__":
    main()
