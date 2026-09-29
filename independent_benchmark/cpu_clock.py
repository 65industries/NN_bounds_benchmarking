"""Opt-in process-CPU ledger for numerical work, not diagnostic bookkeeping.

The default source, ``time.process_time``, measures user + system CPU across
process threads (not wall time). Exclusions change accounting only: every
check still runs and every exception still propagates. Without an active
``numerical_clock`` they are no-ops, preserving the legacy timing policy.

This is a synchronous ledger for one numerical run. Context-local activation
avoids global opt-in state; it does not attribute CPU to concurrent workers.
Small clock/context/decorator overhead outside excluded spans remains charged.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import time
from typing import NamedTuple, ParamSpec, TypeVar

__all__ = ["CPUClock", "numerical_clock", "active_clock", "excluded", "diagnostic"]


class _Mark(NamedTuple):
    cpu_seconds: float
    excluded_seconds: float


class CPUClock:
    """Measure CPU between marks minus the union of diagnostic spans.

    ``clock`` may be any monotonic seconds source, including a deterministic
    fake. Marks belong to the clock that created them. Both marks and elapsed
    reads work inside exclusions: a still-open span is accounted up to the
    same source timestamp, so history snapshots cannot charge diagnostics.
    """

    def __init__(self, clock: Callable[[], float] = time.process_time):
        self._clock = clock
        self._excluded_seconds = 0.0
        self._exclusion_start: float | None = None
        self._exclusion_depth = 0

    def _excluded_at(self, now: float) -> float:
        return self._excluded_seconds + (
            0.0 if self._exclusion_start is None else now - self._exclusion_start
        )

    @property
    def excluded_seconds(self) -> float:
        """Cumulative excluded CPU, including any currently open span."""
        if self._exclusion_start is None:
            return self._excluded_seconds
        return self._excluded_at(self._clock())

    def mark(self) -> _Mark:
        now = self._clock()
        return _Mark(now, self._excluded_at(now))

    def elapsed(self, mark: _Mark) -> float:
        now = self._clock()
        return (now - mark.cpu_seconds) - (self._excluded_at(now) - mark.excluded_seconds)

    def _begin_exclusion(self) -> None:
        if self._exclusion_depth == 0:
            self._exclusion_start = self._clock()
        self._exclusion_depth += 1

    def _end_exclusion(self) -> None:
        self._exclusion_depth -= 1
        if self._exclusion_depth == 0:
            assert self._exclusion_start is not None
            self._excluded_seconds += self._clock() - self._exclusion_start
            self._exclusion_start = None


_active: ContextVar[CPUClock | None] = ContextVar("numerical_cpu_clock", default=None)


def active_clock() -> CPUClock | None:
    """Return the current opt-in ledger, or None for legacy accounting."""
    return _active.get()


@contextmanager
def numerical_clock(clock: Callable[[], float] = time.process_time) -> Iterator[CPUClock]:
    """Activate a fresh ledger; restore the prior context even on failure.

    The optional source is for deterministic tests. Normal callers use
    ``with numerical_clock() as ledger`` and ``ledger.mark()/elapsed(mark)``.
    """
    ledger = CPUClock(clock=clock)
    token = _active.set(ledger)
    try:
        yield ledger
    finally:
        _active.reset(token)


@contextmanager
def excluded() -> Iterator[None]:
    """Exclude a diagnostic span exactly once, including nested/failed calls."""
    ledger = active_clock()
    if ledger is None:
        yield
        return
    ledger._begin_exclusion()
    try:
        yield
    finally:
        ledger._end_exclusion()


_P = ParamSpec("_P")
_R = TypeVar("_R")


def diagnostic(fn: Callable[_P, _R]) -> Callable[_P, _R]:
    """Exclude a synchronous diagnostic, retaining results and fail-closed checks."""
    @wraps(fn)
    def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        if active_clock() is None:
            return fn(*args, **kwargs)
        with excluded():
            return fn(*args, **kwargs)
    return wrapped
