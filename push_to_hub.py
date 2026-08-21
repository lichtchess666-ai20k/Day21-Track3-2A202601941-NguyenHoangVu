"""One-off: publish adapters/correct to the Hugging Face Hub (lab21 bonus B5).

    python push_to_hub.py <hf-username>

Needs a WRITE token: https://huggingface.co/settings/tokens -> New token -> Write.
Run `hf auth login` once first, or set $HF_TOKEN.
"""
import pathlib
import sys

from huggingface_hub import HfApi

NAME = "qwen3.5-4b-lora-vi-ticket-triage"
FOLDER = pathlib.Path(__file__).parent / "adapters" / "correct"


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    repo_id = f"{sys.argv[1]}/{NAME}"

    weights = FOLDER / "adapter_model.safetensors"
    if not weights.exists():
        print(f"missing {weights} -- nothing to publish")
        return 1

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(
        folder_path=str(FOLDER),
        repo_id=repo_id,
        repo_type="model",
        commit_message="LoRA adapter for Vietnamese ticket triage (Lab 21, Day 21 Track 3)",
    )
    print(f"\nhttps://huggingface.co/{repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
