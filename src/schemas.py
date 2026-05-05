from dataclasses import dataclass
from typing import List


@dataclass
class Section:
    order: int
    content: str


@dataclass
class Sample:
    sample_id: str
    sections: List[Section]


@dataclass
class ModelOutput:
    sample_id: str
    prompt: str
    raw_output: str
    parsed_output: str