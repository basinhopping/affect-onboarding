"""Selection records of night 12's 8 rows (container), as RULE_night12_tests.txt (analyses/night12_prep_20261012)
fixed them on 2026-10-12 at 06:27 UTC, before any file of night 12 had left the node. Every configuration of the two
sweeps is a row (each finalist's recorded settings, and the same with only model.pooling changed), so these records
choose nothing; they exist so that each row's test files are read only after its record is written.

  cnn_3block_pooling_1    pool_3block_sd (base), pool_3block_mean, pool_3block_max, pool_3block_attention
  cnn_final_3_pooling_1   pool_final_3_mean (base), pool_final_3_sd, pool_final_3_max, pool_final_3_attention

A row whose configuration has not all 25 runs (folds test_S1 to test_S5, seeds 6 to 10) complete in its sweep's newest
analyze report is not recorded: it is listed as unfinished, as the rule says, and its comparison leaves its family.

Checks before anything is written (each failure stops the script):
  - each sweep's newest finished analyze report exists and covers the sweep file's folds, with 1 to 4 configurations
    (a configuration with no complete run is absent from the report), each of them one of the sweep's 4 rows;
  - each row's configuration, read from its runs by the repository's own loader, has the finalist record's settings
    with model.pooling set to the row's pooling (unchanged for the base), and hashes to its configuration;
  - no test file (predictions_test.csv, metrics_test.json) of these sweeps is in the container;
  - no record of these names exists yet.
After writing, each record is read back and checked against its report entry.

Usage (AFFECT_REPO only for tests):
  python3 select_night12.py            writes the records with `affect.cli select`
  python3 select_night12.py --print    prints what it would write and writes nothing
"""
import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(os.environ.get("AFFECT_REPO", "/home/claude/affect"))
ART = REPO / "artifacts"
SEEDS = [6, 7, 8, 9, 10]
VALIDATION_KEYS = ("ua", "interval", "seeds", "wa", "macro_f1", "recall")
TEST_FILES = ("predictions_test.csv", "metrics_test.json")
RULE = "RULE_night12_tests.txt of analyses/night12_prep_20261012"
WORDS = {"stats": "SD pooling", "mean": "mean pooling", "max": "max pooling", "attention": "attention pooling"}
FINALIST_WORDS = {"threeblock_final": "the tuned 3-block CNN (threeblock_final)",
                  "cnn_final_3": "cnn_final_3"}
ROWS = [  # (sweep, record name, finalist record, pooling)
    ("cnn_3block_pooling_1", "pool_3block_sd", "threeblock_final", "stats"),
    ("cnn_3block_pooling_1", "pool_3block_mean", "threeblock_final", "mean"),
    ("cnn_3block_pooling_1", "pool_3block_max", "threeblock_final", "max"),
    ("cnn_3block_pooling_1", "pool_3block_attention", "threeblock_final", "attention"),
    ("cnn_final_3_pooling_1", "pool_final_3_mean", "cnn_final_3", "mean"),
    ("cnn_final_3_pooling_1", "pool_final_3_sd", "cnn_final_3", "stats"),
    ("cnn_final_3_pooling_1", "pool_final_3_max", "cnn_final_3", "max"),
    ("cnn_final_3_pooling_1", "pool_final_3_attention", "cnn_final_3", "attention"),
]


def record(name):
    return json.loads((ART / "selections" / name / "selection.json").read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", action="store_true", dest="dry")
    args = ap.parse_args()
    sys.path.insert(0, str(REPO / "src"))
    from affect import catalog
    from affect.analysis.selection import latest_analysis
    from affect.analysis.validation import _configs, load
    from affect.runs.spec import config_hash, validate_job
    catalog.load_all()

    assert len({n for _, n, _, _ in ROWS}) == 8
    existing = [n for _, n, _, _ in ROWS if (ART / "selections" / n).exists()]
    if existing and not args.dry:
        raise SystemExit(f"selection records exist already, nothing written: {existing}")
    sweeps = sorted({s for s, _, _, _ in ROWS})
    tests_here = sorted(str(p.relative_to(ART)) for sweep in sweeps for name in TEST_FILES
                        for p in (ART / "runs" / sweep).glob(f"*/{name}"))
    print(f"test files of these sweeps in the container: {len(tests_here)}")
    if tests_here:
        raise SystemExit(f"nothing written: the records come before any test file of these rows reaches the "
                         f"container, and it holds {len(tests_here)}, for example {tests_here[0]}")

    reports, configs = {}, {}
    for sweep in sweeps:
        sweep_file = json.loads((REPO / "sweeps" / f"{sweep}.json").read_text())
        if not (ART / "runs" / sweep).is_dir():
            reports[sweep] = None
            continue
        folder, rep = latest_analysis(sweep)
        assert rep is not None, f"{sweep}: no finished analyze report"
        assert rep["sweep_id"] == sweep and rep["folds"] == list(sweep_file["folds"]), sweep
        assert 1 <= rep["n_configs"] <= 4, (sweep, rep["n_configs"])   # a configuration without runs is absent
        loaded_file, runs = load(sweep)
        configs[sweep] = _configs(loaded_file, runs)
        reports[sweep] = (folder, rep)

    plan, unfinished = [], []   # plan: (sweep, name, entry, report folder, reason)
    row_hashes = {s: set() for s in sweeps}
    for sweep, name, fin, pooling in ROWS:
        if reports[sweep] is None:
            unfinished.append((name, sweep, "no run folder in the container"))
            continue
        folder, rep = reports[sweep]
        expected = copy.deepcopy(record(fin)["settings"])
        assert expected["model"]["pooling"] in WORDS, fin
        is_base = expected["model"]["pooling"] == pooling
        expected["model"]["pooling"] = pooling
        h = config_hash(validate_job(expected))
        row_hashes[sweep].add(h)
        entry = next((x for x in rep["ranked"] if x["config_hash"] == h), None)
        if entry is None:
            unfinished.append((name, sweep, "configuration not in the report (no complete seed)"))
            continue
        assert (h == rep["baseline"]) == is_base, (name, "base row and report baseline disagree")
        settings = configs[sweep][h]["settings"]
        assert config_hash(validate_job(settings)) == h, (name, "settings rebuild the hash")
        if entry["seeds"] != SEEDS or entry["n_runs"] != 25:
            unfinished.append((name, sweep, f"{entry['n_runs']} complete runs, seeds {entry['seeds']}"))
            continue
        kind = "the recorded configuration" if is_base else f"the recorded configuration with {WORDS[pooling]}"
        reason = (f"{FINALISTS_WORDS(fin)}: {kind}, nothing else changed; session out, four classes, seeds 6 to 10. "
                  f"Every configuration of the sweep is a row of {RULE}, fixed before any run of night 12, so nothing "
                  f"is chosen here; recorded before any test file of the sweep reached the container.")
        plan.append((sweep, name, entry, folder, reason))

    for sweep in sweeps:
        if reports[sweep] is not None:
            stray = {x["config_hash"] for x in reports[sweep][1]["ranked"]} - row_hashes[sweep]
            assert not stray, (sweep, "configurations in the report that are no row", sorted(stray))
    for name, sweep, why in unfinished:
        print(f"UNFINISHED, not recorded: {name} ({sweep}): {why}")
    for sweep, name, x, folder, reason in plan:
        print(f"{name}: {x['config_hash'][:12]} of {sweep}, validation UA {x['ua']:.4f} over seeds 6 to 10, "
              f"{x['n_runs']} runs, report {folder.name}")
        if not args.dry:
            subprocess.run([sys.executable, "-m", "affect.cli", "select", "--sweep", sweep, "--config", x["config_hash"],
                            "--name", name, "--reason", reason],
                           cwd=REPO, env=dict(os.environ, PYTHONPATH=str(REPO / "src")), check=True,
                           stdout=subprocess.DEVNULL)
    if args.dry:
        print(f"SELECTIONS PRINTED ({len(plan)} rows, {len(unfinished)} unfinished), NOTHING WRITTEN")
        return
    for sweep, name, x, folder, reason in plan:
        w = record(name)
        assert w["sweep_id"] == sweep and w["config_hash"] == x["config_hash"], name
        assert w["analysis_report"]["folder"] == folder.name, (name, w["analysis_report"]["folder"])
        assert all(w["validation"][k] == x[k] for k in VALIDATION_KEYS), name
        assert w["reason"] == reason, name
    (Path(__file__).resolve().parent / "unfinished_rows.json").write_text(
        json.dumps([{"record": n, "sweep": s, "why": w} for n, s, w in unfinished], indent=1) + "\n")
    print(f"records read back: each of the {len(plan)} equals its report entry; unfinished rows: {len(unfinished)} "
          f"(unfinished_rows.json)")
    print("SELECTIONS DONE")


def FINALISTS_WORDS(fin):
    return FINALIST_WORDS[fin]


if __name__ == "__main__":
    main()
