"""Pre-flight check on the v13 data splits, shared by run_v13.sh and smoke_test.sh.

A stale qtype sidecar would degrade the curriculum without erroring, so verify it covers every
training id. The splits must also stay id-disjoint, or eval_loss quietly stops being held out.

    python scripts/check_splits.py TRAIN_JSON VAL_JSON QTYPE_MAP
"""
import json
import sys

train_path, val_path, qmap_path = sys.argv[1:4]
with open(train_path) as fh:
    train = json.load(fh)
with open(val_path) as fh:
    val = json.load(fh)
with open(qmap_path) as fh:
    qmap = json.load(fh)

missing = [r["id"] for r in train if r["id"] not in qmap]
if missing:
    sys.exit(f"[ERROR] {len(missing)}/{len(train)} ids missing from the qtype sidecar "
             f"(e.g. {missing[:3]}). Re-run with REBUILD_DATA=1.")

leaked = {r["id"] for r in train} & {r["id"] for r in val}
if leaked:
    sys.exit(f"[ERROR] {len(leaked)} ids appear in both train and val "
             f"(e.g. {sorted(leaked)[:3]}); the eval split is not held out.")

print(f"[DATA] qtype sidecar covers all {len(train)} training records; "
      f"{len(val)} eval records, no id overlap")
