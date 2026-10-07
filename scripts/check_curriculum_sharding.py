#!/usr/bin/env python
"""Assert that multi-GPU data sharding preserves the curriculum's per-step composition.

CurriculumSampler emits a *flat, global* index stream: indices [t*G, (t+1)*G) are the exact
category mix that optimizer step t is supposed to see, where G is the global batch size. The
sampler knows nothing about ranks -- train_sft.py builds it with
``G = per_device_batch * accum * world_size`` and every rank constructs an identical order (the
order comes from a single ``np.random.RandomState(seed)`` at construction; no torch RNG, no epoch
re-seeding).

Sharding is therefore delegated entirely to Accelerate: transformers' Trainer hands the sampler to
``accelerator.prepare(DataLoader(...))``, which wraps the batch sampler in ``BatchSamplerShard``.
That shard hands rank r the micro-batches r, r+W, r+2W, ... So over one optimizer step -- A
accumulation micro-steps on each of W ranks -- the ranks jointly consume micro-batches
[k*A*W, (k+1)*A*W), i.e. exactly the sampler's block of G indices. The curriculum survives intact,
just split across devices.

That is a property of Accelerate's sharding, not something this repo controls, so this script pins
it down as a test: an accelerate upgrade that changes the sharding pattern (e.g. to contiguous
per-rank chunks) would silently reshape every optimizer step's category mix without erroring.
Failing here costs seconds; failing silently costs a 7-epoch run.

Usage:
    python scripts/check_curriculum_sharding.py                       # real data, default geometry
    python scripts/check_curriculum_sharding.py --world 2 --batch 8   # a specific geometry
"""

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from accelerate.data_loader import BatchSamplerShard
from torch.utils.data import BatchSampler

from dataset.curriculum_sampler import CurriculumSampler

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA = "/home/staff/srai/orena/data/focus_vqa_frame/frame"


def load_qtypes(train_path, qtype_map_path):
    """Mirror SupervisedDataset.qtypes: dataset order, unmapped ids -> "__unknown__"."""
    with open(train_path) as fh:
        records = json.load(fh)
    with open(qtype_map_path) as fh:
        qtype_map = json.load(fh)
    return [qtype_map.get(rec.get("id"), "__unknown__") for rec in records]


def check(qtypes, schedule, global_batch, world, batch, epochs):
    """Return (ok, detail) for one geometry."""
    accum, rem = divmod(global_batch, batch * world)
    if accum < 1 or rem:
        return None, f"batch={batch} x world={world} does not divide global_batch={global_batch}"

    def order():
        return list(iter(CurriculumSampler(
            qtypes=qtypes, schedule=schedule,
            global_batch_size=global_batch, num_epochs=epochs, seed=42,
        )))

    reference = order()
    # Each rank builds its own sampler (same seed -> same order) and takes its shard, exactly as
    # each training process does.
    per_rank = [
        list(BatchSamplerShard(
            BatchSampler(order(), batch_size=batch, drop_last=False),
            num_processes=world, process_index=r,
            split_batches=False, even_batches=True,
        ))
        for r in range(world)
    ]

    steps = min(len(shard) for shard in per_rank) // accum
    if steps == 0:
        return None, "fewer than one full optimizer step in an epoch chunk"

    for k in range(steps):
        got = sorted(
            idx
            for r in range(world)
            for j in range(accum)
            for idx in per_rank[r][k * accum + j]
        )
        want = sorted(reference[k * global_batch:(k + 1) * global_batch])
        if got != want:
            mix_got = Counter(qtypes[i] for i in got)
            mix_want = Counter(qtypes[i] for i in want)
            return False, (f"step {k}: sharded batch != global batch\n"
                           f"      got  {dict(mix_got)}\n"
                           f"      want {dict(mix_want)}")

    return True, f"accum={accum} steps={steps}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("DATA", DEFAULT_DATA),
                    help="Split directory; only used to resolve --train/--qtype_map defaults.")
    ap.add_argument("--train", default=None,
                    help="Training JSON (default: <data>/train_all_new.json).")
    ap.add_argument("--qtype_map", default=None,
                    help="qtype sidecar JSON (default: <data>/qtype_map_all.json).")
    ap.add_argument("--schedule", default=os.path.join(REPO, "scripts", "curriculum_schedule.json"))
    ap.add_argument("--global_batch", type=int, default=int(os.environ.get("GLOBAL_BATCH", 32)))
    ap.add_argument("--world", type=int, default=int(os.environ.get("NGPU", 2)))
    ap.add_argument("--batch", type=int, default=None,
                    help="Check only this per-device batch instead of every divisor.")
    args = ap.parse_args()

    if args.world < 2:
        print(f"[shard-check] world={args.world}; sharding is a no-op, nothing to verify")
        return 0

    train_path = args.train or os.path.join(args.data, "train_all_new.json")
    qtype_map_path = args.qtype_map or os.path.join(args.data, "qtype_map_all.json")
    qtypes = load_qtypes(train_path, qtype_map_path)
    with open(args.schedule) as fh:
        schedule = json.load(fh)
    print(f"[shard-check] {len(qtypes)} records, global_batch={args.global_batch}, "
          f"world={args.world}, mix={dict(Counter(qtypes))}")

    # Every per-device batch the stage table could plausibly use, so raising `batch` during the
    # PROBE sweep never outruns what has been verified.
    candidates = ([args.batch] if args.batch
                  else [b for b in (1, 2, 4, 8, 16, 32, 64)
                        if b * args.world <= args.global_batch
                        and args.global_batch % (b * args.world) == 0])

    failed = False
    for batch in candidates:
        for epochs in (1, 2, 4):
            ok, detail = check(qtypes, schedule, args.global_batch, args.world, batch, epochs)
            if ok is None:
                print(f"  batch={batch:3d} epochs={epochs}  SKIP  ({detail})")
                continue
            print(f"  batch={batch:3d} epochs={epochs}  {'exact-match=True' if ok else 'FAIL'}  {detail}")
            failed |= not ok

    if failed:
        print("\n[shard-check] FAILED -- Accelerate no longer reconstitutes the global curriculum "
              "batch across ranks. Do NOT train multi-GPU until this is resolved: make "
              "CurriculumSampler rank-aware, or run with NGPU=1.", file=sys.stderr)
        return 1

    print("[shard-check] OK -- every optimizer step sees exactly the intended global batch")
    return 0


if __name__ == "__main__":
    sys.exit(main())
