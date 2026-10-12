"""Night 12 (2026-10-11, Sunday night Pacific): the pooling test on the two finalist CNNs.

Dr. Luitel asked what the grid searches found about pooling, specifically SD pooling. The screens and grids
answer it on validation only, around other configurations. This run swaps the pooling of the two finalists and
nothing else, on the confirmation seeds 6 to 10 and the five session-out folds:
  cnn_3block_pooling_1    threeblock_final (e0c211fcced3, SD pooling) as recorded, and the same with mean, max and
                          attention pooling
  cnn_final_3_pooling_1   cnn_final_3 (7977141b6ba3, mean pooling) as recorded, and the same with SD, max and
                          attention pooling
Each base is the recorded selection's settings unchanged, so its 25 runs repeat the confirmation's runs of the same
configuration, fold and seed (a reproducibility check). 2 x 4 configurations x 5 seeds x 5 folds = 200 jobs.
Writes sweeps/cnn_3block_pooling_1.json and sweeps/cnn_final_3_pooling_1.json, after checking with the repository's
own sweep expansion that each holds 100 jobs, that each base hashes to its selection's configuration, and that
every variant differs from its base in model.pooling only. Run from the repository root:
    PYTHONPATH=src python3 -B artifacts/analyses/night12_prep_20261012/make_night12.py
"""
import copy
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))

from affect import catalog  # noqa: E402
from affect.sweep import design  # noqa: E402

catalog.load_all()
SEEDS = [6, 7, 8, 9, 10]
FOLDS = ["test_S1", "test_S2", "test_S3", "test_S4", "test_S5"]
NAMES = {"stats": "SD pooling", "mean": "mean pooling", "max": "max pooling", "attention": "attention pooling"}
PLAN = [
    ("cnn_3block_pooling_1", "threeblock_final", "stats", ["mean", "max", "attention"], 110,
     "the tuned 3-block CNN"),
    ("cnn_final_3_pooling_1", "cnn_final_3", "mean", ["stats", "max", "attention"], 109, "cnn_final_3"),
]


def flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


for sweep_id, selection, base_pooling, others, priority, who in PLAN:
    sel = json.loads((REPO / "artifacts/selections" / selection / "selection.json").read_text())
    base = copy.deepcopy(sel["settings"])
    assert base["model"]["pooling"] == base_pooling, (selection, base["model"]["pooling"])
    assert base["data"]["split"] == "leave_one_session_out" and base["data"]["label_set"] == "four_merged"
    sweep = {
        "sweep_id": sweep_id,
        "description": (
            f"Pooling test on {who} ({selection}, {sel['config_hash'][:12]}), for Dr. Luitel's question about pooling "
            f"and SD pooling (meeting of 2026-10-12): the recorded configuration ({NAMES[base_pooling]}) and the same with "
            + ", ".join(NAMES[o] for o in others[:-1]) + f" and {NAMES[others[-1]]} instead, nothing else changed; "
            "session-out folds, the four classes, seeds 6 to 10 (the confirmation seeds), so the base's 25 runs "
            f"repeat {sel['sweep_id']}'s runs of the same configuration. Which rows are read on test, and how, is "
            "fixed in RULE_night12_tests.txt (analyses/night12_prep_20261012) before any run. Priority "
            f"{priority}. Validation only until the selection records exist."),
        "folds": FOLDS,
        "base": base,
        "stages": [
            {"name": "base", "kind": "base", "priority": priority, "seeds": SEEDS},
            {"name": "pooling", "kind": "variants", "priority": priority, "seeds": SEEDS,
             "variants": {NAMES[o]: {"model.pooling": o} for o in others}},
        ],
    }
    jobs, summary = design.expand(sweep)
    assert len(jobs) == 100, len(jobs)
    assert summary["base_config_hash"] == sel["config_hash"], (summary["base_config_hash"], sel["config_hash"])
    by_config = {}
    for job in jobs:
        by_config.setdefault(job["config_hash"] if isinstance(job, dict) else job.config_hash, job)
    print(f"{sweep_id}: {len(jobs)} jobs, {len(by_config)} configurations, base {summary['base_config_hash'][:12]} "
          f"= {selection}")
    base_flat = flat(base)
    for label, override in sweep["stages"][1]["variants"].items():
        changed = flat(design.apply(base, override))
        diff = {k: (base_flat.get(k), v) for k, v in changed.items() if base_flat.get(k) != v}
        assert diff == {"model.pooling": (base_pooling, override["model.pooling"])}, diff
        print(f"  {label}: differs from the base in {diff}")
    out = REPO / "sweeps" / f"{sweep_id}.json"
    if out.exists():
        raise SystemExit(f"{out} exists; not overwritten")
    out.write_text(json.dumps(sweep, indent=1) + "\n")
    print(f"  written {out.relative_to(REPO)}")
