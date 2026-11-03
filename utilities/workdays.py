"""Business-day helpers (weekends only; no holiday calendar).

The two range helpers differ on purpose: per-day runs snap weekend endpoints
back to Friday, stacked (time-series) runs simply skip weekend days.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Iterator, List


def business_day_offset(anchor: date, offset: int) -> date:
    """Shift ``anchor`` by ``offset`` weekdays (negative = backwards)."""
    step = 1 if offset > 0 else -1
    current = anchor
    for _ in range(abs(offset)):
        current += timedelta(days=step)
        while current.weekday() >= 5:
            current += timedelta(days=step)
    return current


def ensure_business_day(value: date) -> date:
    """Return ``value`` if it is Mon-Fri, else the preceding Friday."""
    return value if value.weekday() < 5 else value - timedelta(days=value.weekday() - 4)


def iter_business_days(start: date, end: date) -> Iterator[date]:
    """Weekdays in ``[start, end]``; weekend endpoints are skipped, not snapped."""
    current = start
    while current <= end:
        if current.weekday() < 5:
            yield current
        current += timedelta(days=1)


def business_date_range(start: date, end: date) -> List[date]:
    """Weekdays in ``[start, end]`` after snapping both endpoints back to a weekday."""
    start, end = ensure_business_day(start), ensure_business_day(end)
    if start > end:
        raise ValueError("start_date must be on or before end_date")
    return list(iter_business_days(start, end))
