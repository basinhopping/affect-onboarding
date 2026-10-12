"""Repeat check of night 12 (container, 2026-10-12), before any night-12 test file reaches the container.

Each finalist's recorded configuration is a row of its night-12 sweep on seeds 6 to 10, the seeds its confirmation
already ran: e0c211fcced3 in cnn_3block_confirm_1, 7977141b6ba3 in cnn_confirm_3. The same configuration, fold and seed
must give the same run (a copy in another sweep too, such as cnn_regression_3's seed-6 runs, must match as well). As night 8's repeat_check_val.py: for every complete run of the two sweeps, every folder with
the same run ID in another sweep is compared: predictions_val.csv by SHA-256, metrics_val.json without each epoch's
wall-clock times (affect.analysis.compare), and the two test files by SHA-256, the night-12 ones from the pull's list
(never opened) and the earlier ones from the pull lists, or from their files when no list holds them (those were read
on night 10 already). affect.cli compare-runs cannot run here before the test files come: it hashes them.

Usage: python3 repeat_check_night12.py      (writes repeat_check_night12.json beside itself)
"""
import json
import sys
from pathlib import Path

REPO = Path("/home/claude/affect")
sys.path.insert(0, str(REPO / "src"))
from affect.analysis.compare import _val_metrics, differences  # noqa: E402
from affect.hashing import sha256_file  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS = REPO / "artifacts" / "runs"
SWEEPS = {"cnn_3block_pooling_1": ("e0c211fcced3", "cnn_3block_confirm_1"),
          "cnn_final_3_pooling_1": ("7977141b6ba3", "cnn_confirm_3")}
TEST_FILES = ("predictions_test.csv", "metrics_test.json")


def listed_hashes() -> dict:
    table = {}
    for p in sorted((REPO / "artifacts" / "_pulled").glob("*.files.sha256")):
        for line in open(p, encoding="utf-8"):
            h, path = line.rstrip("\n").split("  ", 1)
            if table.get(path, h) != h:
                raise SystemExit(f"{path} has two hashes in the pull lists")
            table[path] = h
    return table


def complete(folder: Path) -> bool:
    return folder.is_dir() and not folder.name.endswith(".tmp") and ".failed-" not in folder.name \
        and (folder / "metrics_val.json").is_file()


def main() -> None:
    listed = listed_hashes()
    out, problems = {"sweeps": {}}, []
    for sweep, (base, confirm) in SWEEPS.items():
        others = sorted(p.name for p in RUNS.iterdir() if p.is_dir() and p.name != sweep and not p.name.startswith("smoke_"))
        runs = sorted(f for f in (RUNS / sweep).iterdir() if complete(f))
        n_copies = n_identical = 0
        for mine in runs:
            here = []
            if any((mine / t).exists() for t in TEST_FILES):
                here.append("a test file is in the container")
            copies = [o for o in others if complete(RUNS / o / mine.name)]
            if mine.name.startswith(base) and confirm not in copies:
                here.append(f"expected a copy in {confirm}; found {copies}")
            for other in copies:
                folder = RUNS / other / mine.name
                if sha256_file(folder / "predictions_val.csv")[1] != sha256_file(mine / "predictions_val.csv")[1]:
                    here.append(f"predictions_val.csv differs from {other}'s")
                here += [f"{other}: {d}" for d in differences(_val_metrics(mine), _val_metrics(folder), "metrics_val")[:3]]
                for name in TEST_FILES:
                    a = listed.get(f"runs/{sweep}/{mine.name}/{name}")
                    b = listed.get(f"runs/{other}/{mine.name}/{name}") or (
                        sha256_file(folder / name)[1] if (folder / name).is_file() else None)
                    if a is None or b is None:
                        here.append(f"{name}: no hash ({'this run' if a is None else other})")
                    elif a != b:
                        here.append(f"{name} differs from {other}'s by SHA-256 (not read)")
            n_copies += bool(copies)
            n_identical += bool(copies) and not here
            problems += [f"{mine.name}: {p}" for p in here]
        out["sweeps"][sweep] = {"complete_runs": len(runs), "with_copies": n_copies, "identical": n_identical}
        print(f"{sweep}: {len(runs)} complete runs, {n_copies} with a copy elsewhere, {n_identical} identical "
              f"(validation files, and test files by SHA-256)")
    out["problems"], out["passed"] = problems, not problems
    (HERE / "repeat_check_night12.json").write_text(json.dumps(out, indent=1) + "\n")
    for p in problems[:15]:
        print("  " + p)
    print("REPEAT CHECK PASSED" if not problems else f"REPEAT CHECK FAILED: {len(problems)} problems")


if __name__ == "__main__":
    main()
