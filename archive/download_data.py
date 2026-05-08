"""
download_data.py
────────────────
One-time script to download Twin-2K-500 from HuggingFace.

Data hierarchy in Twin-2K-500:
  Level 1 (raw)       wave1_3_persona_json  ← each person's answer to every question
                                               stored in survey_json_with_human_response
  Level 2 (processed) wave1_3_persona_text  ← same data rendered as natural language
  Level 3 (outcomes)  wave4_Q_wave4_A       ← behavioral experiment results

We download all three so you can decide what to use.

Usage:
    pip install datasets pyarrow
    python download_data.py

    # if HuggingFace requires login:
    pip install huggingface_hub
    huggingface-cli login          ← paste your HF token
    python download_data.py

After running, set in config.py:
    USE_REAL_DATA  = True
    LOCAL_DATA_DIR = RAW_DIR

Saved files (all in .gitignore — participant data never committed to git):
    data/raw/persona_json.parquet   ← individual question-level responses (JSON)
    data/raw/persona_text.parquet   ← same data as readable text (for LLM prompts)
    data/raw/wave4.parquet          ← behavioral outcomes
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from datasets import load_dataset  # type: ignore
except ImportError:
    print("ERROR: run  pip install datasets pyarrow  first.")
    sys.exit(1)

from src import config

RAW_DIR = config.RAW_DIR

# All configurations to download
DOWNLOADS = [
    # (hf_config_name,          local_filename,          description)
    ("wave1_3_persona_json",  "persona_json.parquet",
     "individual question responses in JSON — the raw per-question answers"),
    ("wave1_3_persona_text",  "persona_text.parquet",
     "same data rendered as natural language text — used for LLM prompts"),
    ("wave4_Q_wave4_A",       "wave4.parquet",
     "Wave 4 behavioral experiment outcomes"),
]


def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    for hf_config, filename, description in DOWNLOADS:
        out_path = os.path.join(RAW_DIR, filename)

        if os.path.exists(out_path):
            print(f"[skip] {filename} already exists.")
            continue

        print(f"\nDownloading: {filename}")
        print(f"  ({description})")
        try:
            ds = load_dataset(config.HF_DATASET_NAME, hf_config, split="train")
            df = ds.to_pandas()
            df.to_parquet(out_path, index=False)
            print(f"  → {len(df)} rows saved to {out_path}")
            print(f"  → columns: {list(df.columns)}")
        except Exception as e:
            print(f"  ERROR: {e}")
            print("  If this is a permissions error, run: huggingface-cli login")

    print("\n" + "=" * 50)
    print("Done. Files in data/raw/:")
    for f in os.listdir(RAW_DIR):
        path = os.path.join(RAW_DIR, f)
        size_mb = os.path.getsize(path) / 1e6
        print(f"  {f:35s}  {size_mb:.1f} MB")

    print("\nNow update config.py:")
    print("    USE_REAL_DATA  = True")
    print("    LOCAL_DATA_DIR = 'data/raw'")


if __name__ == "__main__":
    main()
