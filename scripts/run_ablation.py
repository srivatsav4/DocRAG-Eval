"""Full ablation grid: 5 strategies x 3 chunking strategies x k in {3,5,10}, with plots and bootstrap comparison."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from src.config import CHUNKING_CONFIGS, Settings
from src.evaluation.experiment_runner import run_grid


def main() -> None:
    s = Settings.from_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--allow-uncalibrated", action="store_true")
    a = ap.parse_args()
    summary = run_grid(s, a.split, "full", None, list(CHUNKING_CONFIGS), allow_uncalibrated=a.allow_uncalibrated)
    print("ablation results ->", summary["out_dir"])


if __name__ == "__main__":
    main()
