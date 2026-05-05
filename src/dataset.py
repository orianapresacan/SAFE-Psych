import json
from pathlib import Path
from typing import List

from .schemas import Sample, Section


def load_dataset(path: str) -> List[Sample]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    samples = []
    for item in data:
        sections = [
            Section(**sec)
            for sec in sorted(item.get("sections", []), key=lambda x: x["order"])
        ]

        samples.append(
            Sample(
                sample_id=str(item["sample_id"]),
                sections=sections,
            )
        )

    return samples