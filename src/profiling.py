"""Замер времени и пиковой памяти процесса, включая нативные аллокации (Arrow, SQLite, LightGBM).

tracemalloc видит только память Python и замедляет код на Pandas, поэтому здесь его нет:
память снимается отдельным потоком как RSS процесса.
"""
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import psutil


@dataclass
class Measurement:
    seconds: float = 0.0
    peak_mb: float = 0.0


@contextmanager
def measure(interval: float = 0.02) -> Iterator[Measurement]:
    """Прирост пиковой памяти процесса (МБ) и время выполнения блока."""
    process = psutil.Process()
    baseline = peak = process.memory_info().rss
    stop = threading.Event()

    def sample() -> None:
        nonlocal peak
        while not stop.wait(interval):
            peak = max(peak, process.memory_info().rss)

    sampler = threading.Thread(target=sample, daemon=True)
    result = Measurement()
    sampler.start()
    start = time.perf_counter()
    try:
        yield result
    finally:
        result.seconds = time.perf_counter() - start
        stop.set()
        sampler.join()
        peak = max(peak, process.memory_info().rss)
        result.peak_mb = max(peak - baseline, 0) / 2**20
