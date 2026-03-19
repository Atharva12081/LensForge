from __future__ import annotations

from collections import Counter
from pathlib import Path

from .cutout import center_crop
from .fetch import fetch_array
from .package import package_processed_sample, write_manifest
from .preprocess import percentile_normalize
from .query import query_mock_survey


def run_mock_lsst_pipeline(
    data_root: Path,
    output_root: Path,
    max_per_folder: int | None = None,
    cutout_size: int = 64,
) -> dict[str, object]:
    records = query_mock_survey(data_root=data_root, max_per_folder=max_per_folder)
    packaged_rows: list[dict[str, object]] = []

    for record in records:
        array = fetch_array(record.source_path)
        cutout = center_crop(array, cutout_size)
        processed = percentile_normalize(cutout)
        packaged_rows.append(package_processed_sample(output_root=output_root, record=record, array=processed))

    split_counts = Counter(row["split"] for row in packaged_rows)
    class_counts = Counter(f"{row['split']}_{row['class_name']}" for row in packaged_rows)
    provenance = {
        "pipeline_name": "mock_lsst_to_deeplense",
        "data_root": str(data_root),
        "output_root": str(output_root),
        "max_per_folder": max_per_folder,
        "cutout_size": cutout_size,
        "num_records": len(packaged_rows),
        "split_counts": dict(split_counts),
        "class_counts": dict(class_counts),
        "stages": ["query", "fetch", "cutout", "preprocess", "package"],
    }
    write_manifest(output_root=output_root, rows=packaged_rows, provenance=provenance)
    return provenance
