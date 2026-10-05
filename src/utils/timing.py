from __future__ import annotations

import time
from contextlib import contextmanager


class StageTimer:
    """Accumulates wall-clock milliseconds per named stage."""

    def __init__(self) -> None:
        self.ms: dict[str, float] = {}
        self._t0 = time.perf_counter()

    @contextmanager
    def stage(self, name: str):
        t = time.perf_counter()
        try:
            yield
        finally:
            self.ms[name] = self.ms.get(name, 0.0) + (time.perf_counter() - t) * 1000.0

    def total_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000.0

    def as_dict(self) -> dict[str, float]:
        d = {f"{k}_ms": round(v, 2) for k, v in self.ms.items()}
        d["total_ms"] = round(self.total_ms(), 2)
        return d
