from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.lsst_pipeline import run_mock_lsst_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a mock Rubin/LSST-style packaging pipeline into DeepLense-ready folders."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/lens-finding-test"))
    parser.add_argument("--output-root", type=Path, default=Path("tmp/lsst_mock_pipeline"))
    parser.add_argument("--max-per-folder", type=int, default=16)
    parser.add_argument("--cutout-size", type=int, default=64)
    parser.add_argument("--report-path", type=Path, default=Path("reports/lsst_mock_pipeline_run.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)

    report = run_mock_lsst_pipeline(
        data_root=args.data_root,
        output_root=args.output_root,
        max_per_folder=args.max_per_folder,
        cutout_size=args.cutout_size,
    )
    args.report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
