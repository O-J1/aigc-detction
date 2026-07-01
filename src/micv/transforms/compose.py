from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from PIL import Image


class RecordAwareCompose:
    needs_record = True

    def __init__(self, transform_steps: Sequence[object]) -> None:
        self.transforms: list[Any] = list(transform_steps)

    def __call__(self, image: Image.Image, record: Any) -> Any:
        out: Any = image
        for transform in self.transforms:
            if getattr(transform, "needs_record", False):
                out = transform(out, record)
            else:
                out = transform(out)
        return out