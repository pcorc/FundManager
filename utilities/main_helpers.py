"""Helper functions extracted from main.py for cleaner organization."""
from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping as MappingABC, Sequence as SequenceABC
from datetime import date, datetime, timedelta
from typing import Mapping, Sequence

from config.fund_registry import FundRegistry


def filter_registry(registry: FundRegistry, funds: Sequence[str]) -> FundRegistry:
    """Return a registry containing only the requested funds (if any)."""
    if not funds:
        return registry

    missing = [fund for fund in funds if fund not in registry.funds]
    if missing:
        raise ValueError(f"Requested funds not in registry: {', '.join(sorted(missing))}")

    filtered = FundRegistry()
    filtered.funds = {fund: registry.funds[fund] for fund in funds}
    return filtered


def log_processing_summary(logger: logging.Logger, label: str, summary: Mapping[str, object]) -> None:
    """Log processing summary with key-value pairs."""
    pretty = ", ".join(f"{key}={value}" for key, value in summary.items())
    logger.info("Processing summary (%s): %s", label, pretty)


def log_generated_paths(logger: logging.Logger, paths: Mapping[str, str]) -> None:
    """Log all generated artifact paths."""
    if not paths:
        logger.warning("No report artefacts were produced")
        return
    for label, path in sorted(paths.items()):
        logger.info("Generated %s -> %s", label, path)


def extract_override_dates(payload: dict[str, object]) -> list[date]:
    """Extract multiple dates from override payload (as_of_dates or date_range)."""
    if not payload:
        return []

    dates: list[date] = []

    if "as_of_dates" in payload:
        dates.extend(coerce_date_sequence(payload.pop("as_of_dates"), "as_of_dates"))

    if "date_range" in payload:
        dates.extend(expand_date_range(payload.pop("date_range")))

    ordered: list[date] = []
    seen: set[date] = set()
    for run_date in dates:
        if run_date not in seen:
            seen.add(run_date)
            ordered.append(run_date)
    return ordered


def coerce_date_sequence(values: object, field_name: str) -> list[date]:
    """Convert various input formats to a list of dates."""
    if isinstance(values, MappingABC):
        raise TypeError(f"{field_name} override must be a sequence of dates, not a mapping")
    if isinstance(values, (str, bytes)):
        text = values.decode() if isinstance(values, bytes) else values
        tokens = [token.strip() for token in text.replace(";", ",").split(",") if token.strip()]
        if len(tokens) > 1:
            return [coerce_date(token, field_name) for token in tokens]
        return [coerce_date(text.strip(), field_name)]
    if isinstance(values, (date, datetime)):
        return [coerce_date(values, field_name)]
    if isinstance(values, Iterable):
        return [coerce_date(item, field_name) for item in values]
    raise TypeError(f"Unsupported {field_name} override payload: {type(values)!r}")


def expand_date_range(raw_range: object) -> list[date]:
    """Expand a date range specification into a list of business days."""
    if isinstance(raw_range, MappingABC):
        start_value = (
            raw_range.get("start")
            or raw_range.get("start_date")
            or raw_range.get("from")
        )
        end_value = raw_range.get("end") or raw_range.get("end_date") or raw_range.get("to")
    elif isinstance(raw_range, SequenceABC) and len(raw_range) == 2:
        start_value, end_value = raw_range
    else:
        raise TypeError("date_range override must be a mapping with 'start'/'end' or a 2-tuple")

    if start_value is None or end_value is None:
        raise ValueError("date_range override requires both start and end values")

    start_date = coerce_date(start_value, "date_range.start")
    end_date = coerce_date(end_value, "date_range.end")

    if start_date > end_date:
        raise ValueError("date_range start must be on or before end")

    return list(iter_business_days_inclusive(start_date, end_date))


def iter_business_days_inclusive(start: date, end: date) -> Iterable[date]:
    """Generate all business days (Mon-Fri) between start and end, inclusive."""
    current = start
    while current <= end:
        if current.weekday() < 5:
            yield current
        current += timedelta(days=1)


def coerce_date(value: object, field_name: str) -> date:
    """Convert various types to a date object."""
    if isinstance(value, date):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value).date()
        except ValueError as exc:
            raise ValueError(f"Invalid ISO date for {field_name}: {value!r}") from exc
    raise TypeError(f"Unsupported type for {field_name}: {type(value)!r}")