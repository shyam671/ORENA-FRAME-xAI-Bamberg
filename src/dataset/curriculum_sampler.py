"""Curriculum sampler: shifts the mix of question types across training.

The mix is defined by a phase schedule (see ``scripts/curriculum_schedule.json``)
whose boundaries are fractions of the TOTAL number of optimizer steps over the
whole run (all epochs) -- one easy->hard sweep, not per-epoch.

Design (deterministic + resume-safe):
  * We know total optimizer steps ``T = ceil(len / global_batch) * num_epochs``.
  * For each step ``t`` we look up its phase from ``t / T`` and get category
    weights. We convert ``weights * global_batch`` to integer per-category counts
    with largest-remainder rounding (they sum to exactly ``global_batch``), then
    draw that many indices from each category's pool WITH REPLACEMENT (by cycling
    a reshuffled per-category list, which spreads repeats evenly rather than
    clustering them). The block is shuffled so grad-accum micro-batches each see a
    representative mix.
  * Concatenating all ``T`` blocks gives a flat index order of length
    ``T * global_batch``. It is split into ``num_epochs`` contiguous chunks; the
    sampler yields one chunk per epoch (``__len__`` returns the chunk length so
    HF Trainer computes the right number of steps per epoch).

Because both the mix and the order are precomputed from a fixed seed, the run is
fully reproducible and safe to resume.
"""
import math
from collections import Counter

import numpy as np
from torch.utils.data import Sampler


def _largest_remainder(weights, total):
    """Split ``total`` integer slots across keys proportional to ``weights``.

    Returns a dict {key: count} summing to exactly ``total``.
    """
    keys = list(weights.keys())
    w = np.array([max(0.0, float(weights[k])) for k in keys], dtype=float)
    s = w.sum()
    if s <= 0:
        # Degenerate: spread evenly.
        base = total // len(keys)
        counts = {k: base for k in keys}
        for k in keys[: total - base * len(keys)]:
            counts[k] += 1
        return counts
    exact = w / s * total
    floor = np.floor(exact).astype(int)
    remainder = total - int(floor.sum())
    # Hand out the remaining slots to the largest fractional parts.
    order = np.argsort(-(exact - floor))
    counts = {keys[i]: int(floor[i]) for i in range(len(keys))}
    for i in range(remainder):
        counts[keys[order[i]]] += 1
    return counts


class _CyclicPool:
    """Draws indices from a fixed pool with replacement, reshuffling each cycle.

    Spreads repeats: every element is used once before any is used twice.
    """

    def __init__(self, indices, rng):
        self._indices = np.asarray(indices, dtype=np.int64)
        self._rng = rng
        self._order = self._reshuffle()
        self._pos = 0

    def _reshuffle(self):
        order = self._indices.copy()
        self._rng.shuffle(order)
        return order

    def draw(self, n):
        out = []
        while n > 0:
            if self._pos >= len(self._order):
                self._order = self._reshuffle()
                self._pos = 0
            take = min(n, len(self._order) - self._pos)
            out.append(self._order[self._pos: self._pos + take])
            self._pos += take
            n -= take
        return np.concatenate(out) if out else np.array([], dtype=np.int64)


class CurriculumSampler(Sampler):
    """Yields dataset indices whose question-type mix follows a phase schedule.

    Args:
        qtypes: list/array of category labels aligned to dataset order (index i ->
            category of ``dataset[i]``).
        schedule: dict parsed from the schedule JSON (``categories``,
            ``fallback_category``, ``phases``).
        global_batch_size: samples per optimizer step (per_device_batch *
            grad_accum * world_size). Category composition is exact per this block.
        num_epochs: number of epochs the Trainer will run.
        seed: RNG seed for reproducibility.
    """

    def __init__(self, qtypes, schedule, global_batch_size, num_epochs, seed=42):
        self.num_samples = len(qtypes)
        self.global_batch_size = int(global_batch_size)
        self.num_epochs = max(1, int(num_epochs))
        self.seed = int(seed)

        self.categories = list(schedule["categories"])
        self.fallback = schedule.get("fallback_category", self.categories[0])
        self.phases = sorted(schedule["phases"], key=lambda p: p["upto"])

        # Map every sample to a scheduled category (fold unlisted labels).
        cat_set = set(self.categories)
        folded = [c if c in cat_set else self.fallback for c in qtypes]
        self._pool_counts = Counter(folded)
        missing = [c for c in self.categories if self._pool_counts.get(c, 0) == 0]
        if missing:
            raise ValueError(
                f"CurriculumSampler: no examples for scheduled categories {missing}. "
                f"Available: {dict(self._pool_counts)}. Fix the schedule or the data."
            )

        self._cat_indices = {
            c: np.where(np.array(folded) == c)[0] for c in self.categories
        }

        self._full_order = self._build_order()
        self._chunk_len = len(self._full_order) // self.num_epochs
        self._epoch = 0

    # -- schedule helpers -------------------------------------------------
    def _weights_for(self, frac):
        for p in self.phases:
            if frac < p["upto"] or math.isclose(frac, p["upto"]):
                return p["weights"]
        return self.phases[-1]["weights"]

    def _build_order(self):
        steps_per_epoch = math.ceil(self.num_samples / self.global_batch_size)
        total_steps = steps_per_epoch * self.num_epochs

        rng = np.random.RandomState(self.seed)
        pools = {c: _CyclicPool(self._cat_indices[c], rng) for c in self.categories}

        order = np.empty(total_steps * self.global_batch_size, dtype=np.int64)
        for t in range(total_steps):
            frac = t / total_steps
            weights = {c: self._weights_for(frac).get(c, 0) for c in self.categories}
            counts = _largest_remainder(weights, self.global_batch_size)
            block = np.concatenate(
                [pools[c].draw(counts[c]) for c in self.categories]
            )
            rng.shuffle(block)
            order[t * self.global_batch_size: (t + 1) * self.global_batch_size] = block
        return order

    # -- torch Sampler API ------------------------------------------------
    def set_epoch(self, epoch):
        self._epoch = int(epoch) % self.num_epochs

    def __iter__(self):
        e = self._epoch
        start = e * self._chunk_len
        end = start + self._chunk_len
        chunk = self._full_order[start:end]
        # Advance for the next epoch even if Trainer never calls set_epoch.
        self._epoch = (self._epoch + 1) % self.num_epochs
        return iter(int(i) for i in chunk)

    def __len__(self):
        return self._chunk_len

    # -- introspection (for verification / logging) -----------------------
    def phase_composition(self):
        """Return {phase_label: {category: fraction}} realized over the full order."""
        steps_per_epoch = math.ceil(self.num_samples / self.global_batch_size)
        total_steps = steps_per_epoch * self.num_epochs
        gb = self.global_batch_size
        idx_to_cat = {}
        for c in self.categories:
            for i in self._cat_indices[c]:
                idx_to_cat[int(i)] = c
        report = {}
        bounds = [0.0] + [p["upto"] for p in self.phases]
        for pi, p in enumerate(self.phases):
            lo, hi = bounds[pi], bounds[pi + 1]
            s0, s1 = int(lo * total_steps), int(hi * total_steps)
            seg = self._full_order[s0 * gb: s1 * gb]
            cnt = Counter(idx_to_cat.get(int(i), self.fallback) for i in seg)
            tot = sum(cnt.values()) or 1
            report[f"{lo:.0%}-{hi:.0%}"] = {c: cnt.get(c, 0) / tot for c in self.categories}
        return report
