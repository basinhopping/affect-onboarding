"""Night 12's test analysis (container), as RULE_night12_tests.txt (analyses/night12_prep_20261012) fixed it on
2026-10-12 at 06:27 UTC and amended it at 06:44 UTC, before any file of night 12 had left the node.

Reads only the test files of the 8 rows' selection records (select_night12.py). The paired bootstrap, UA, t interval
and Holm functions are those of families N9B to N9G (pairing.py of analyses/night9_20261009), with the rule's
generator seed 20261012 and 2,000 draws (pairing.N_BOOT as in every earlier family; the rule's 06:44 UTC amendment), the families computed in
the order N12A, N12B. All rows: session out, four classes, seeds 6 to 10.

  N12A  the tuned 3-block CNN (Holm over 3): pool_3block_mean, pool_3block_max and pool_3block_attention, each minus
        pool_3block_sd (its recorded SD pooling)
  N12B  cnn_final_3 (Holm over 3): pool_final_3_sd, pool_final_3_max and pool_final_3_attention, each minus
        pool_final_3_mean (its recorded mean pooling)
  A row listed in unfinished_rows.json (select_night12.py) has no record: its comparison leaves its family and the
  Holm count shrinks, as the rule says.
  Apart, without a test of a difference: each row's test UA with its 95% t interval over seeds (fold means per seed),
  and within each finalist the order of the four poolings.

Usage (AFFECT_REPO only for tests): python3 night12_test_analysis.py     writes night12_test.json beside itself
"""
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("AFFECT_REPO", "/home/claude/affect"))
spec = importlib.util.spec_from_file_location("pairing", REPO / "artifacts/analyses/night9_20261009/pairing.py")
pairing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pairing)

SEED = 20261012
pairing.ROOT = REPO / "artifacts"
pairing.N_BOOT = 2000
pairing.RNG = np.random.default_rng(SEED)
FAMILIES = [
    ("N12A", "the tuned 3-block CNN, pooling, session out, seeds 6 to 10", "pool_3block_sd", "SD pooling",
     [("mean pooling", "pool_3block_mean"), ("max pooling", "pool_3block_max"),
      ("attention pooling", "pool_3block_attention")]),
    ("N12B", "cnn_final_3, pooling, session out, seeds 6 to 10", "pool_final_3_mean", "mean pooling",
     [("SD pooling", "pool_final_3_sd"), ("max pooling", "pool_final_3_max"),
      ("attention pooling", "pool_final_3_attention")]),
]


def main():
    unfinished_file = HERE / "unfinished_rows.json"
    unfinished = {u["record"] for u in json.loads(unfinished_file.read_text())}
    cache = {}

    def runs(name):
        if name not in cache:
            cache[name] = pairing.runs_of(name)
            assert sorted({s for s, _ in cache[name]}) == [6, 7, 8, 9, 10], name
            assert len(cache[name]) == 25, (name, len(cache[name]))
        return cache[name]

    out = {"seed": SEED, "n_boot": pairing.N_BOOT, "unfinished": sorted(unfinished), "families": {}, "rows": {},
           "order": {}}
    for code, title, base, base_words, others in FAMILIES:
        if base in unfinished:
            print(f"\n{code}: every comparison left out (its base row {base} is unfinished)")
            out["families"][code] = {"left_out": [o for _, o in others]}
            continue
        pairs = {f"{words} minus {base_words}": (name, base) for words, name in others if name not in unfinished}
        left = [name for _, name in others if name in unfinished]
        if not pairs:
            out["families"][code] = {"left_out": left}
            continue
        res = pairing.family(f"{code}, {title} (Holm over {len(pairs)})",
                             {k: (runs(a), runs(b)) for k, (a, b) in pairs.items()})
        out["families"][code] = dict({k: dict(v, a=pairs[k][0], b=pairs[k][1]) for k, v in res.items()},
                                     left_out=left)
        if left:
            print(f"  left out (unfinished rows): {left}")

    print("\nEach row's test UA, 95% t interval over seeds 6 to 10 (fold means per seed), and the order (highest first)")
    for code, title, base, base_words, others in FAMILIES:
        members = {base_words: base, **{w: n for w, n in others}}
        present = {k: v for k, v in members.items() if v not in unfinished}
        for k, name in present.items():
            fm = pairing.fold_means(runs(name))
            out["rows"][name] = {"per_seed": fm, "ua": pairing.t_interval(fm.values())}
        order = sorted(present, key=lambda k: -out["rows"][present[k]]["ua"]["mean"])
        out["order"][code] = [(k, present[k], out["rows"][present[k]]["ua"]["mean"]) for k in order]
        print(f"  {title}: " + " > ".join(f"{k} {out['rows'][present[k]]['ua']['mean']:.4f}" for k in order))
    for name, r in sorted(out["rows"].items()):
        u = r["ua"]
        print(f"  {name:28s} {u['mean']:.4f} [{u['low']:.4f}, {u['high']:.4f}]")
    (HERE / "night12_test.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(f"\nwrote {HERE / 'night12_test.json'}")


if __name__ == "__main__":
    main()
