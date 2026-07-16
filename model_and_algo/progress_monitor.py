"""求解过程心跳监控：默认每 60s 输出一次当前状态。"""

from __future__ import annotations

import sys
import threading
import time
from contextlib import contextmanager
from typing import Callable


class ProgressMonitor:
    """按固定间隔打印状态；verbose 求解时启用。"""

    def __init__(
        self,
        label: str,
        *,
        interval_sec: float = 60.0,
        enabled: bool = True,
    ) -> None:
        self.label = label
        self.interval_sec = max(1.0, interval_sec)
        self.enabled = enabled
        self._start = time.perf_counter()
        self._last_report = self._start
        self._phase = "init"
        self._extra = ""

    def elapsed(self) -> float:
        return time.perf_counter() - self._start

    def set_phase(self, phase: str, **kwargs: object) -> None:
        self._phase = phase
        parts = [f"{k}={v}" for k, v in kwargs.items()]
        self._extra = "  ".join(parts)

    def report(self, message: str | None = None, *, force: bool = False) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        if not force and now - self._last_report < self.interval_sec:
            return
        self._last_report = now
        elapsed = now - self._start
        msg = message or f"phase={self._phase}"
        if self._extra:
            msg = f"{msg}  {self._extra}"
        print(f"[{self.label}] +{elapsed:6.0f}s  {msg}", flush=True, file=sys.stderr)

    @contextmanager
    def phase(self, name: str, **kwargs: object):
        prev_extra = self._extra
        self.set_phase(name, **kwargs)
        self.report(force=True)
        try:
            yield self
        finally:
            self._extra = prev_extra


@contextmanager
def heartbeat_while(
    label: str,
    status: Callable[[], str],
    *,
    interval_sec: float = 60.0,
    enabled: bool = True,
):
    """长耗时阻塞段（定价 / MIP）期间后台每 interval_sec 打印状态。"""
    if not enabled:
        yield
        return

    stop = threading.Event()
    start = time.perf_counter()

    def _loop() -> None:
        while not stop.wait(interval_sec):
            try:
                detail = status()
            except Exception as exc:  # noqa: BLE001
                detail = f"status_error={exc}"
            elapsed = time.perf_counter() - start
            print(f"[{label}] +{elapsed:6.0f}s  {detail}", flush=True, file=sys.stderr)

    thread = threading.Thread(target=_loop, daemon=True, name=f"heartbeat-{label}")
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1.0)
