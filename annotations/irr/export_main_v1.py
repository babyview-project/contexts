"""Export main-v1 annotations, practice attempts, gold labels and attention checks to CSVs for irr.qmd.

Run from anywhere: python annotations/irr/export_main_v1.py. Reads MONGODB_URI from annotations/.env.
"""

import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient

BATCH = "main-v1"
ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "data"


def norm(s):
    return s.fillna("").astype(str).str.strip().str.lower()


def main():
    load_dotenv(ROOT / ".env")
    client = MongoClient(os.environ["MONGODB_URI"], serverSelectionTimeoutMS=5000)
    db = client.get_default_database(default="activity_annotations")
    ann = pd.DataFrame(list(db.annotations.find({"batch": BATCH}, {"_id": 0})))
    att = pd.DataFrame(list(db.trainingattempts.find({"batch": BATCH}, {"_id": 0, "results": 0})))
    client.close()
    assert not ann.empty, f"no annotations in batch {BATCH}"

    # Annotations from before taskType existed count as 'do' (as in lib/store.js)
    ann["taskType"] = ann["taskType"].fillna("do")
    for c in ["isTraining", "isAttentionCheck"]:
        ann[c] = ann[c].eq(True)
    for c in ["primaryActivity", "anyoneInteracting"]:
        ann[c] = norm(ann[c])
    ann = ann[
        ["videoFilename", "annotatorName", "taskType", "primaryActivity", "anyoneInteracting",
         "isTraining", "isAttentionCheck", "attemptNumber", "updatedAt"]
    ]
    att = att[["annotatorName", "taskType", "attemptNumber", "createdAt", "nCorrect", "nTotal", "score", "passed"]]

    # Current gold labels, one row per (modality, clip)
    gold = pd.read_csv(ROOT / "gold_standards.csv", dtype=str).fillna("")
    for c in ["type", "modality", "primary_activity", "anyone_interacting", "disabled"]:
        gold[c] = gold[c].str.strip().str.lower()
    gold = (
        gold[gold["type"] == "gold"]
        .drop_duplicates(["modality", "video_filename"], keep="last")
        .rename(columns={"modality": "taskType", "video_filename": "videoFilename",
                         "primary_activity": "gold_activity", "anyone_interacting": "gold_interacting"})
    )
    gold["disabled"] = gold["disabled"] == "yes"
    gold = gold[["taskType", "videoFilename", "old_video_filename", "gold_activity", "gold_interacting",
                 "disabled", "pa_reasoning", "ai_reasoning"]]

    config = json.loads((ROOT / "config.json").read_text())
    checks = pd.DataFrame(
        [{"taskType": t, "videoFilename": c["videoFilename"], "expected_activity": c["primaryActivity"].strip().lower()}
         for t, cs in config["attentionChecks"].items() for c in cs]
    )

    OUT.mkdir(exist_ok=True)
    for name, df in [("annotations", ann), ("training_attempts", att), ("gold_standards", gold), ("attention_checks", checks)]:
        df.to_csv(OUT / f"{name}.csv", index=False)
        print(f"{name}.csv: {len(df)} rows")


if __name__ == "__main__":
    main()
