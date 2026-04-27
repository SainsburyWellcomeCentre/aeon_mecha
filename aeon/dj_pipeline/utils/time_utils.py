"""Time/datetime utilities for the DJ pipeline."""

import datetime


def parse_epoch_timestamp(name: str) -> datetime.datetime:
    """Parse an epoch directory name into a naive datetime.

    Handles both formats:

    - Old (hyphenated): ``2026-04-15T09-03-01``
    - New (compact ISO 8601): ``2026-04-15T090301Z``
    """
    date_str, time_str = name.split("T")
    return datetime.datetime.fromisoformat(
        date_str + "T" + time_str.replace("-", ":")
    ).replace(tzinfo=None)
