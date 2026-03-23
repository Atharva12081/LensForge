from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .query import SurveyObject


def package_processed_sample(
    output_root: Path,
    record: SurveyObject,
    array: np.ndarray,
) -> dict[str, object]:
    dataset_root = output_root / "deeplense_dataset"
    manifests_root = output_root / "manifests"
    dataset_root.mkdir(parents=True, exist_ok=True)
    manifests_root.mkdir(parents=True, exist_ok=True)

    folder_name = f"{record.split}_{record.class_name}"
    destination = dataset_root / folder_name / f"{record.object_id}.npy"
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(destination, array.astype(np.float32, copy=False))

    return {
        "object_id": record.object_id,
        "split": record.split,
        "class_name": record.class_name,
        "label": record.label,
        "source_kind": record.source_kind,
        "dataset_type": record.dataset_type,
        "collection": record.collection,
        "bands": list(record.bands),
        "source_path": str(record.source_path),
        "packaged_path": str(destination),
        "shape": list(array.shape),
    }


def write_manifest(output_root: Path, rows: list[dict[str, object]], provenance: dict[str, object]) -> None:
    manifests_root = output_root / "manifests"
    manifests_root.mkdir(parents=True, exist_ok=True)

    manifest_csv = manifests_root / "manifest.csv"
    with manifest_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "object_id",
                "split",
                "class_name",
                "label",
                "source_kind",
                "dataset_type",
                "collection",
                "bands",
                "source_path",
                "packaged_path",
                "shape",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    (manifests_root / "manifest.json").write_text(json.dumps(rows, indent=2))
    (manifests_root / "provenance.json").write_text(json.dumps(provenance, indent=2))
