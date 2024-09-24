from typing import Iterable


def fixed_with_overlap(doc: str, width: int, overlap: int = 150) -> Iterable[str]:
    yield from (doc[i : i + width] for i in range(0, len(doc), width - overlap))
