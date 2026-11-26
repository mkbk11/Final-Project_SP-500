"""Run the S&P 500 Python stages in dependency order.

Usage: python python/run_pipeline.py [--include-validation] [--include-fusion] [--include-literature]
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-validation", action="store_true",
                        help="Run the optional leakage and split experiments (stage 07)")
    parser.add_argument("--include-fusion", action="store_true",
                        help="Run the optional MLP, trading, and market-state stages (08 and 09)")
    parser.add_argument("--include-literature", action="store_true",
                        help="Create paper comparison tables (requires fusion)")
    args = parser.parse_args()
    if args.include_literature and not args.include_fusion:
        parser.error("--include-literature requires --include-fusion")

    directory = Path(__file__).resolve().parent
    stages = ["01", "02", "03", "04", "05", "06"]
    if args.include_validation:
        stages.append("07")
    if args.include_fusion:
        stages.extend(["08", "09"])
    scripts = {**dict.fromkeys(("01", "02", "03"), "prepare.py"),
               **dict.fromkeys(("04", "05", "06", "07"), "models.py"),
               **dict.fromkeys(("08", "09"), "analysis.py")}
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8",
               SP500_INCLUDE_FUSION="1" if args.include_fusion else "0",
               SP500_INCLUDE_LITERATURE="1" if args.include_literature else "0")
    for stage in stages:
        print(f"\nRunning stage {stage}", flush=True)
        subprocess.run([sys.executable, "-X", "utf8", str(directory / scripts[stage]), stage],
                       cwd=directory.parent, env=env, check=True)
    print("\nFinished. Results are in output/ (latest runs are saved by timestamp).")


if __name__ == "__main__":
    main()
