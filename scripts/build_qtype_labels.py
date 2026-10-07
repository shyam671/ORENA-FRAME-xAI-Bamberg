#!/usr/bin/env python
"""Build a category sidecar map for curriculum / balanced sampling.

For each record in a LLaVA-style training JSON (``id`` / ``image``|``video`` /
``conversations``), assign a category and write a sidecar ``{id: category}``
JSON next to the source file.

``--label_mode`` picks what the category *is*. ``CurriculumSampler`` treats
categories as opaque strings, so the same sampler balances whatever axis is
labelled here:

    qtype               question type only (default, back-compatible)
    procedure           source procedure only  -> "heico" / "lapchole"
    procedure_x_qtype   both axes crossed      -> "heico|Recognition", ...

The procedure axis matters because frame/train.json is 7,210 heico vs 5,163
lapchole, and the skew is far worse per source clip: 20 heico videos (~360 QA
each) against 72 lapchole videos (~72 QA each). Balancing on qtype alone also
lets the procedure mix drift, because heico is 62.5% Recognition while lapchole
is 56.5%.

Primary label source is GROUND TRUTH: the source FOCUS parquet files carry a
``primary_capability`` taxonomy code (``1a``, ``2b``, ``3a`` ...). The training
JSON ``id`` is ``f"{dataset}_{parquet_id}"``, which joins back to the parquet
100% (verified on frame/train.json: 12,373/12,373). The leading digit of the
capability code maps directly onto the five categories:

    1 -> Recognition   (object recognition / identity / spatial localization)
    2 -> Temporal      (temporal grounding)
    3 -> Aggregation   (object / event aggregation)
    4 -> Event/proc    (event and procedural understanding)
    5 -> Complex       (complex reasoning)

If a record's id is missing from the parquet (should not happen for the FOCUS
data), we fall back to a coarse keyword classifier over the question text.

Note on the frame/ track: it contains ONLY Recognition and Aggregation
questions (Temporal/Event/Complex live in the segment/procedure tracks), so a
curriculum on frame/ is necessarily a 2-way Recognition->Aggregation schedule.
In ``procedure_x_qtype`` mode the handful of stray non-frame labels (exactly one
Temporal record in frame/train.json) are folded into Aggregation so the cell
count stays at 4 -- see FOLD_TO_FRAME_TRACK and the size warning below.

Cell count vs global batch: ``CurriculumSampler._largest_remainder`` splits each
GLOBAL batch across categories exactly, so a category whose share rounds to zero
is NEVER sampled. With equal weights and more categories than the global batch
size, only the first ``global_batch`` categories are ever drawn. Keep the number
of categories <= GLOBAL_BATCH_SIZE (16 in the v15/v16 scripts); this script
refuses to emit more than --max_categories cells.

Usage:
    # default, unchanged behaviour
    python scripts/build_qtype_labels.py \
        --data_path /home/staff/srai/orena/data/focus_vqa_frame/frame/train.json \
        --parquet_root /data/local/orena/dataset

    # procedure-balanced sampling (see scripts/curriculum_schedule_proc.json)
    python scripts/build_qtype_labels.py \
        --data_path /home/staff/srai/orena/data/focus_vqa_frame/frame/train.json \
        --parquet_root /data/local/orena/dataset \
        --label_mode procedure_x_qtype
"""
import argparse
import glob
import json
import os
import re
from collections import Counter

CATEGORY_BY_DIGIT = {
    "1": "Recognition",
    "2": "Temporal",
    "3": "Aggregation",
    "4": "Event/proc",
    "5": "Complex",
}

LABEL_MODES = ("qtype", "procedure", "procedure_x_qtype")

# frame/ is a Recognition+Aggregation track; anything else is a stray handful
# (1 record in train.json). Folded in procedure_x_qtype mode to keep 4 cells.
FOLD_TO_FRAME_TRACK = {
    "Temporal": "Aggregation",
    "Event/proc": "Aggregation",
    "Complex": "Aggregation",
}

DEFAULT_OUT_NAME = {
    "qtype": "qtype_map.json",
    "procedure": "proc_map.json",
    "procedure_x_qtype": "proc_qtype_map.json",
}

# Fallback keyword rules (ordered, first match wins). Only used when a record's
# id cannot be joined to the parquet. These are deliberately coarse.
_KEYWORD_RULES = [
    ("Temporal", re.compile(r"\bat what time\b|last visible|first visible|duration", re.I)),
    ("Event/proc", re.compile(r"currently (grasp|being|manipulat)|grasped by|being grasped|used to", re.I)),
    ("Complex", re.compile(r"\bwhy\b|because|would happen|consequence|multi[- ]step", re.I)),
    ("Aggregation", re.compile(r"list all|how many different|which combination|co-occur|are all visible", re.I)),
]


def keyword_category(question: str) -> str:
    for cat, rx in _KEYWORD_RULES:
        if rx.search(question):
            return cat
    return "Recognition"


def load_capability_map(parquet_root: str) -> dict:
    """Return {(dataset, int_id): capability_code} from all FOCUS parquet files."""
    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover
        raise SystemExit(f"pandas required to read parquet labels: {e}")

    cap = {}
    files = sorted(glob.glob(os.path.join(parquet_root, "*", "qa", "*", "*.parquet")))
    for f in files:
        # .../dataset/<ds>/qa/<split>/<file>.parquet
        ds = f.split(os.sep)[-4]
        df = pd.read_parquet(f, columns=["id", "primary_capability"])
        for rid, pc in zip(df["id"].tolist(), df["primary_capability"].tolist()):
            cap[(ds, int(rid))] = pc
    return cap


def parse_id(json_id: str):
    """'heico_1591576' -> ('heico', 1591576). Returns (None, None) if unparseable."""
    if "_" not in json_id:
        return None, None
    ds, num = json_id.rsplit("_", 1)
    if not num.isdigit():
        return None, None
    return ds, int(num)


def record_procedure(rec: dict) -> str:
    """Source procedure for a record: 'heico', 'lapchole', ...

    Primary source is the id prefix (``heico_1591576``); if that does not parse,
    fall back to the leading path component of the media field, which carries the
    same dataset name (``heico/frames_overlay/...``). Verified: both agree on all
    12,373 records of frame/train.json.
    """
    ds, _ = parse_id(rec["id"])
    if ds:
        return ds
    media = rec.get("image") or rec.get("video") or ""
    if isinstance(media, list):
        media = media[0] if media else ""
    return media.split("/")[0] if media else "__unknown__"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True, help="LLaVA-style training JSON")
    ap.add_argument("--parquet_root", default="/data/local/orena/dataset",
                    help="Root holding <ds>/qa/<split>/*.parquet")
    ap.add_argument("--label_mode", default="qtype", choices=LABEL_MODES,
                    help="What the emitted category is (default: qtype)")
    ap.add_argument("--max_categories", type=int, default=16,
                    help="Refuse to emit more cells than this; must stay <= the "
                         "training GLOBAL_BATCH_SIZE or small cells never sample")
    ap.add_argument("--schedule", default=None,
                    help="Schedule JSON this sidecar will feed. If given, abort when the "
                         "emitted labels do not match its `categories` exactly.")
    ap.add_argument("--out", default=None,
                    help="Output sidecar path (default: <data_dir>/<mode-specific name>)")
    args = ap.parse_args()

    out = args.out or os.path.join(os.path.dirname(args.data_path),
                                   DEFAULT_OUT_NAME[args.label_mode])

    with open(args.data_path) as fh:
        data = json.load(fh)

    # The parquet join is only needed when the qtype axis is actually used.
    need_qtype = args.label_mode in ("qtype", "procedure_x_qtype")
    cap = {}
    if need_qtype:
        cap = load_capability_map(args.parquet_root)
        print(f"Loaded {len(cap)} capability labels from parquet under {args.parquet_root}")

    label_map = {}
    counts = Counter()
    qtype_counts = Counter()
    proc_counts = Counter()
    source = Counter()
    folded = Counter()
    for rec in data:
        rid = rec["id"]
        ds, num = parse_id(rid)

        cat = None
        if need_qtype:
            code = cap.get((ds, num)) if ds is not None else None
            if code is not None:
                cat = CATEGORY_BY_DIGIT.get(str(code)[0], "Recognition")
                source["parquet"] += 1
            else:
                question = rec["conversations"][0]["value"]
                cat = keyword_category(question)
                source["keyword_fallback"] += 1
            qtype_counts[cat] += 1

        proc = record_procedure(rec)
        proc_counts[proc] += 1

        if args.label_mode == "qtype":
            label = cat
        elif args.label_mode == "procedure":
            label = proc
        else:
            if cat in FOLD_TO_FRAME_TRACK:
                folded[cat] += 1
                cat = FOLD_TO_FRAME_TRACK[cat]
            label = f"{proc}|{cat}"

        label_map[rid] = label
        counts[label] += 1

    # CurriculumSampler folds any label not listed in the schedule into fallback_category, and it
    # does so SILENTLY. A total mismatch is caught downstream (every cell but one ends up empty and
    # the constructor raises), but a partial one is not: a single unlisted label just adds its
    # records to the fallback cell, quietly re-weighting the very axis the schedule is balancing.
    # Checking here turns that into an abort that names the strays.
    if args.schedule:
        with open(args.schedule) as fh:
            scheduled = set(json.load(fh)["categories"])
        emitted = set(counts)
        strays, unused = emitted - scheduled, scheduled - emitted
        if strays or unused:
            raise SystemExit(
                f"\nSidecar does not match {args.schedule}:\n"
                + (f"  would be folded into fallback_category: {sorted(strays)}\n" if strays else "")
                + (f"  scheduled but absent from the data:     {sorted(unused)}\n" if unused else "")
                + "Fix --label_mode or the schedule's `categories` so the two agree."
            )
        print(f"Schedule check: labels match {os.path.basename(args.schedule)} exactly "
              f"({len(emitted)} cells); fallback_category is unreachable.")

    if len(counts) > args.max_categories:
        raise SystemExit(
            f"{len(counts)} categories exceeds --max_categories={args.max_categories}. "
            "CurriculumSampler splits each GLOBAL batch across categories exactly, so "
            "cells beyond the global batch size would never be sampled. Coarsen the "
            "labels or raise GLOBAL_BATCH_SIZE (and this flag) together."
        )

    with open(out, "w") as fh:
        json.dump(label_map, fh)

    total = sum(counts.values())
    print(f"\nLabeled {total} records -> {out}  (label_mode={args.label_mode})")
    if source:
        print(f"Label source: {dict(source)}")
    if folded:
        print(f"Folded into the frame/ track: {dict(folded)} -> Aggregation")

    print(f"\nCategory distribution ({len(counts)} cells):")
    for cat, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {cat:24s} {n:6d}  {100 * n / total:5.1f}%")

    # Always show both marginals -- the point of the exercise is the interaction
    # between them, and it is cheap to print.
    if args.label_mode == "procedure_x_qtype":
        print("\nMarginals:")
        for name, c in (("procedure", proc_counts), ("qtype", qtype_counts)):
            pretty = ", ".join(f"{k}={v} ({100 * v / total:.1f}%)"
                               for k, v in sorted(c.items(), key=lambda kv: -kv[1]))
            print(f"  {name:10s} {pretty}")


if __name__ == "__main__":
    main()
