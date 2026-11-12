#!/usr/bin/env python3
"""Print arm means and the primary contrast from the recorded experiment."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics", type=Path,
        default=Path(__file__).resolve().parents[1] / "reports/final/metrics.json",
    )
    args = parser.parse_args()
    metrics = json.loads(args.metrics.read_text())
    arms = defaultdict(list)
    for run in metrics["per_run"]:
        arms[run["arm"]].append(run)

    print("Arm  Runs    SRCC  Raw PLCC  Valid / total predictions")
    for arm, runs in sorted(arms.items()):
        srcc = mean(run["SRCC"] for run in runs)
        plcc = mean(run["raw_PLCC"] for run in runs)
        valid = sum(run["valid"] for run in runs)
        total = sum(run["total"] for run in runs)
        print(f"{arm:<4} {len(runs):>4}  {srcc:.4f}    {plcc:.4f}  {valid:>5} / {total}")

    primary = next(c for c in metrics["contrasts"]
                   if c["contrast"] == "T3-T1" and c["metric"] == "SRCC")
    lo, hi = primary["interval95"]
    print(f"\nT3−T1 SRCC: {primary['mean_difference']:+.4f}; "
          f"95% interval [{lo:+.4f}, {hi:+.4f}]")
    print(f"Bootstrap: {primary['bootstrap']}")


if __name__ == "__main__":
    main()
