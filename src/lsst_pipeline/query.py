from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


FOLDER_TO_LABEL = {
    "train_lenses": 1,
    "test_lenses": 1,
    "train_nonlenses": 0,
    "test_nonlenses": 0,
}


@dataclass(frozen=True)
class SurveyObject:
    object_id: str
    split: str
    class_name: str
    label: int
    source_path: Path


def query_mock_survey(data_root: Path, max_per_folder: int | None = None) -> list[SurveyObject]:
    records: list[SurveyObject] = []
    for folder_name, label in FOLDER_TO_LABEL.items():
        folder = data_root / folder_name
        files = sorted(folder.glob("*.npy"))
        if max_per_folder is not None:
            files = files[:max_per_folder]
        split, class_name = folder_name.split("_", 1)
        for path in files:
            records.append(
                SurveyObject(
                    object_id=f"{folder_name}_{path.stem}",
                    split=split,
                    class_name=class_name,
                    label=label,
                    source_path=path,
                )
            )
    return records
