"""Build a small digital-twin validation layer from Twin-2K-500.

This script keeps the raw Arrow files local and writes only compact,
reviewable artifacts:
  data/digital_twin_personas_sample.csv
  data/digital_twin_story_ground_truth.csv
  data/digital_twin_story_predictions.csv
  outputs/digital_twin_validation.png

The validation task asks GPT to predict a real participant's story-belief
answers from that participant's rich persona summary plus the same story
chapter the participant read.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
from dotenv import load_dotenv
from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]
FULL_PERSONA_DIR = ROOT / "raw/full_persona/0.0.0/f883165a3026fde855dfd448e0cd16443ab257b6"
STORY_BELIEFS_ARROW = ROOT / "twin_2k_500_mega_study/story_beliefs/data-00000-of-00001.arrow"
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
MODEL = "gpt-4.1-nano"


def iter_arrow_rows(path: Path) -> dict:
    with pa.memory_map(str(path), "r") as source:
        reader = ipc.open_stream(source)
        for batch in reader:
            rows = batch.to_pylist()
            for row in rows:
                yield row


def load_personas() -> dict[str, dict]:
    personas: dict[str, dict] = {}
    for path in sorted(FULL_PERSONA_DIR.glob("*.arrow")):
        for row in iter_arrow_rows(path):
            personas[str(row["pid"])] = {
                "pid": str(row["pid"]),
                "persona_summary": row["persona_summary"],
                "persona_text": row["persona_text"],
            }
    return personas


def selected_answer(question: dict):
    answers = question.get("Answers") or {}
    if "SelectedByPosition" in answers:
        return answers.get("SelectedByPosition")
    if "Text" in answers:
        return answers.get("Text")
    if "Values" in answers:
        return answers.get("Values")
    return None


def walk_questions(survey_json: str):
    payload = json.loads(survey_json)
    for block in payload.get("Elements", []):
        for question in block.get("Questions", []):
            yield question


def load_story_ground_truth() -> pd.DataFrame:
    rows = []
    for row in iter_arrow_rows(STORY_BELIEFS_ARROW):
        pid = str(row["PID"]).replace("pid_", "")
        chapter1 = None
        out = {
            "pid": pid,
            "human_valence_1to7": np.nan,
            "human_arousal_1to7": np.nan,
            "human_interest_1to5": np.nan,
        }
        for question in walk_questions(row["survey_json_with_human_response"]):
            qname = question.get("QuestionName")
            if qname == "Q290":
                chapter1 = question.get("QuestionText")
            elif qname == "Achap1_text_val":
                out["human_valence_1to7"] = selected_answer(question)
            elif qname == "Achap1_text_aro":
                out["human_arousal_1to7"] = selected_answer(question)
            elif qname == "Achap1_interest":
                out["human_interest_1to5"] = selected_answer(question)
        out["chapter_text"] = chapter1
        rows.append(out)
    df = pd.DataFrame(rows)
    for col in ["human_valence_1to7", "human_arousal_1to7", "human_interest_1to5"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["chapter_text", "human_valence_1to7", "human_arousal_1to7", "human_interest_1to5"])


def prompt_for_prediction(persona_summary: str, chapter_text: str) -> str:
    return f"""You are simulating a specific survey participant as faithfully as possible.

Participant persona:
{persona_summary[:3500]}

Story chapter shown to the participant:
{chapter_text}

Predict how this same participant would answer the survey questions after reading the chapter.
Return strict JSON only with:
  valence_1to7: integer 1-7, where 1 means very negative, 4 neutral, 7 very positive
  arousal_1to7: integer 1-7, where 1 means very low energy, 4 medium energy, 7 very high energy
  interest_1to5: integer 1-5, where 1 means not at all interested, 5 very interested
  rationale: one short sentence explaining the prediction
"""


def parse_json(text: str) -> dict:
    text = (text or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}


def clip_int(value, lo: int, hi: int) -> int:
    try:
        return int(np.clip(round(float(value)), lo, hi))
    except Exception:
        return int((lo + hi) // 2)


def run_openai_predictions(df: pd.DataFrame, personas: dict[str, dict], limit: int, refresh: bool) -> pd.DataFrame:
    pred_path = DATA_DIR / "digital_twin_story_predictions.csv"
    if pred_path.exists() and not refresh:
        existing = pd.read_csv(pred_path)
    else:
        existing = pd.DataFrame()

    done = set(existing["pid"].astype(str)) if not existing.empty else set()
    todo = df[~df["pid"].astype(str).isin(done)].head(max(0, limit - len(done))).copy()
    if todo.empty:
        return existing.head(limit)

    load_dotenv(ROOT / ".env")
    client = OpenAI()
    new_rows = []
    for _, row in todo.iterrows():
        persona = personas.get(str(row["pid"]))
        if not persona:
            continue
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You output strict JSON only. No prose, no markdown fences."},
                {"role": "user", "content": prompt_for_prediction(persona["persona_summary"], row["chapter_text"])},
            ],
            temperature=0,
            max_tokens=160,
        )
        data = parse_json(response.choices[0].message.content or "{}")
        new_rows.append({
            "pid": row["pid"],
            "model": MODEL,
            "pred_valence_1to7": clip_int(data.get("valence_1to7"), 1, 7),
            "pred_arousal_1to7": clip_int(data.get("arousal_1to7"), 1, 7),
            "pred_interest_1to5": clip_int(data.get("interest_1to5"), 1, 5),
            "rationale": str(data.get("rationale", ""))[:300],
        })
        print(f"predicted pid={row['pid']}")

    combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    combined = combined.drop_duplicates("pid", keep="first").head(limit)
    combined.to_csv(pred_path, index=False)
    return combined


def summarize_demographics(text: str) -> dict:
    fields = {}
    for label in ["Gender", "Age", "Education level", "Race", "Political affiliation", "Income", "Political views"]:
        match = re.search(rf"{re.escape(label)}: ([^\\n]+?)(?= [A-Z][A-Za-z ]+:|$)", text)
        fields[label.lower().replace(" ", "_")] = match.group(1).strip() if match else ""
    return fields


def write_persona_sample(personas: dict[str, dict], pids: list[str]) -> None:
    rows = []
    for pid in pids:
        persona = personas.get(pid)
        if not persona:
            continue
        rows.append({
            "pid": pid,
            **summarize_demographics(persona["persona_summary"]),
            "persona_summary_excerpt": persona["persona_summary"][:900].replace("\n", " "),
        })
    pd.DataFrame(rows).to_csv(DATA_DIR / "digital_twin_personas_sample.csv", index=False)


def make_validation_chart(merged: pd.DataFrame) -> pd.DataFrame:
    metrics = []
    specs = [
        ("Valence", "human_valence_1to7", "pred_valence_1to7"),
        ("Arousal", "human_arousal_1to7", "pred_arousal_1to7"),
        ("Interest", "human_interest_1to5", "pred_interest_1to5"),
    ]
    for label, human, pred in specs:
        pop = float(merged[human].mean())
        twin_mae = float((merged[pred] - merged[human]).abs().mean())
        pop_mae = float((pop - merged[human]).abs().mean())
        within_one = float(((merged[pred] - merged[human]).abs() <= 1).mean())
        metrics.append({"task": label, "model": "LLM digital twin", "mae": twin_mae, "within_one": within_one})
        metrics.append({"task": label, "model": "Population mean", "mae": pop_mae, "within_one": np.nan})
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(DATA_DIR / "digital_twin_validation_metrics.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    tasks = [s[0] for s in specs]
    x = np.arange(len(tasks))
    width = 0.34
    twin = metrics_df[metrics_df["model"] == "LLM digital twin"]["mae"].to_numpy()
    pop = metrics_df[metrics_df["model"] == "Population mean"]["mae"].to_numpy()
    ax.bar(x - width / 2, twin, width, label="LLM digital twin", color="#2ca02c")
    ax.bar(x + width / 2, pop, width, label="Population mean baseline", color="#7f7f7f")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_ylabel("Mean absolute error")
    ax.set_title("Digital-twin personas predict held-out story-belief responses")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    for i, value in enumerate(twin):
        ax.text(i - width / 2, value + 0.03, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    for i, value in enumerate(pop):
        ax.text(i + width / 2, value + 0.03, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "digital_twin_validation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return metrics_df


def main(limit: int, refresh: bool) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    personas = load_personas()
    truth = load_story_ground_truth()
    truth = truth[truth["pid"].isin(personas.keys())].sort_values("pid").reset_index(drop=True)
    truth.to_csv(DATA_DIR / "digital_twin_story_ground_truth.csv", index=False)
    write_persona_sample(personas, truth["pid"].head(limit).astype(str).tolist())
    preds = run_openai_predictions(truth, personas, limit=limit, refresh=refresh)
    merged = truth.merge(preds, on="pid", how="inner")
    merged.to_csv(DATA_DIR / "digital_twin_story_validation_merged.csv", index=False)
    metrics = make_validation_chart(merged)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    main(limit=args.limit, refresh=args.refresh)
