"""
fishstick DateTime Module
=========================
A comprehensive datetime system providing flexible datetime manipulation,
parsing, formatting, and timezone handling capabilities.

Features:
- DateTime: Full datetime with timestamp conversion and manipulation
- Date: Date-only operations with calendar utilities
- Time: Time-only operations
- Timezone: Timezone handling and conversion
- Duration: Duration/interval operations
- Parsing: Flexible datetime/date/time parsing
- Calendar: Calendar utilities (leap years, days in month, etc.)
- Utilities: timestamp_now, sleep, Timer context manager
"""

from __future__ import annotations

import time as time_module
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, date, time as dt_time, timedelta, timezone
from typing import Optional, Union


# ============================================================================
# DATETIME
# ============================================================================


class DateTime:
    """Comprehensive datetime wrapper with extended functionality."""

    def __init__(self, dt: Optional[datetime] = None):
        self._dt = dt if dt is not None else datetime.now()

    @property
    def datetime(self) -> datetime:
        return self._dt

    @classmethod
    def now(cls, tz: Optional[timezone] = None) -> DateTime:
        if tz is not None:
            return cls(datetime.now(tz))
        return cls(datetime.now())

    @classmethod
    def from_timestamp(cls, ts: float, tz: Optional[timezone] = None) -> DateTime:
        if tz is not None:
            return cls(datetime.fromtimestamp(ts, tz))
        return cls(datetime.fromtimestamp(ts))

    def to_timestamp(self) -> float:
        return self._dt.timestamp()

    def add_days(self, days: int) -> DateTime:
        return DateTime(self._dt + timedelta(days=days))

    def add_hours(self, hours: int) -> DateTime:
        return DateTime(self._dt + timedelta(hours=hours))

    def add_minutes(self, minutes: int) -> DateTime:
        return DateTime(self._dt + timedelta(minutes=minutes))

    def add_seconds(self, seconds: int) -> DateTime:
        return DateTime(self._dt + timedelta(seconds=seconds))

    def subtract(self, other: DateTime) -> Duration:
        return Duration(self._dt - other._dt)

    def year(self) -> int:
        return self._dt.year

    def month(self) -> int:
        return self._dt.month

    def day(self) -> int:
        return self._dt.day

    def hour(self) -> int:
        return self._dt.hour

    def minute(self) -> int:
        return self._dt.minute

    def second(self) -> int:
        return self._dt.second

    def microsecond(self) -> int:
        return self._dt.microsecond

    def weekday(self) -> int:
        return self._dt.weekday()

    def isoweekday(self) -> int:
        return self._dt.isoweekday()

    def date(self) -> Date:
        return Date(self._dt.date())

    def time(self) -> Time:
        return Time(
            dt_time(
                self._dt.hour, self._dt.minute, self._dt.second, self._dt.microsecond
            )
        )

    def replace(
        self,
        year: Optional[int] = None,
        month: Optional[int] = None,
        day: Optional[int] = None,
        hour: Optional[int] = None,
        minute: Optional[int] = None,
        second: Optional[int] = None,
        microsecond: Optional[int] = None,
    ) -> DateTime:
        return DateTime(
            self._dt.replace(
                year=year if year is not None else self._dt.year,
                month=month if month is not None else self._dt.month,
                day=day if day is not None else self._dt.day,
                hour=hour if hour is not None else self._dt.hour,
                minute=minute if minute is not None else self._dt.minute,
                second=second if second is not None else self._dt.second,
                microsecond=microsecond
                if microsecond is not None
                else self._dt.microsecond,
            )
        )

    def __str__(self) -> str:
        return self._dt.isoformat()

    def __repr__(self) -> str:
        return f"DateTime({self._dt!r})"

    def __add__(self, other: Duration) -> DateTime:
        return DateTime(self._dt + other._delta)

    def __sub__(self, other: Union[DateTime, Duration]) -> Union[DateTime, Duration]:
        if isinstance(other, DateTime):
            return Duration(self._dt - other._dt)
        return DateTime(self._dt - other._delta)


# ============================================================================
# DATE
# ============================================================================


class Date:
    """Comprehensive date wrapper with calendar utilities."""

    def __init__(self, d: Optional[date] = None):
        self._date = d if d is not None else date.today()

    @property
    def date(self) -> date:
        return self._date

    @classmethod
    def today(cls) -> Date:
        return cls(date.today())

    @classmethod
    def from_ymd(cls, year: int, month: int, day: int) -> Date:
        return cls(date(year, month, day))

    @classmethod
    def from_timestamp(cls, ts: float) -> Date:
        return cls(date.fromtimestamp(ts))

    def to_datetime(self, hour: int = 0, minute: int = 0, second: int = 0) -> DateTime:
        return DateTime(
            datetime(
                self._date.year, self._date.month, self._date.day, hour, minute, second
            )
        )

    def day_of_week(self) -> int:
        return self._date.weekday()

    def iso_weekday(self) -> int:
        return self._date.isoweekday()

    def year(self) -> int:
        return self._date.year

    def month(self) -> int:
        return self._date.month

    def day(self) -> int:
        return self._date.day

    def add_days(self, days: int) -> Date:
        return Date(self._date + timedelta(days=days))

    def add_months(self, months: int) -> Date:
        month = self._date.month - 1 + months
        year = self._date.year + month // 12
        month = month % 12 + 1
        day = min(self._date.day, Calendar.days_in_month(year, month))
        return Date(date(year, month, day))

    def add_years(self, years: int) -> Date:
        try:
            return Date(date(self._date.year + years, self._date.month, self._date.day))
        except ValueError:
            day = Calendar.days_in_month(self._date.year + years, self._date.month)
            return Date(date(self._date.year + years, self._date.month, day))

    def weekday_name(self) -> str:
        days = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        return days[self._date.weekday()]

    def month_name(self) -> str:
        months = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]
        return months[self._date.month - 1]

    def __str__(self) -> str:
        return self._date.isoformat()

    def __repr__(self) -> str:
        return f"Date({self._date!r})"

    def __add__(self, other: Duration) -> Date:
        return Date(self._date + other._delta)

    def __sub__(self, other: Union[Date, Duration]) -> Union[Date, Duration]:
        if isinstance(other, Date):
            return Duration(self._date - other._date)
        return Date(self._date - other._delta)


# ============================================================================
# TIME
# ============================================================================


class Time:
    """Comprehensive time wrapper."""

    def __init__(self, t: Optional[dt_time] = None):
        self._time = t if t is not None else dt_time.now()

    @property
    def time(self) -> dt_time:
        return self._time

    @classmethod
    def now(cls) -> Time:
        return cls(dt_time.now())

    @classmethod
    def from_hms(
        cls, hour: int, minute: int, second: int = 0, microsecond: int = 0
    ) -> Time:
        return cls(dt_time(hour, minute, second, microsecond))

    @classmethod
    def from_seconds(cls, total_seconds: int) -> Time:
        return cls(dt_time.fromtimestamp(total_seconds).time())

    def hour(self) -> int:
        return self._time.hour

    def minute(self) -> int:
        return self._time.minute

    def second(self) -> int:
        return self._time.second

    def microsecond(self) -> int:
        return self._time.microsecond

    def total_seconds(self) -> int:
        return self._time.hour * 3600 + self._time.minute * 60 + self._time.second

    def add_hours(self, hours: int) -> Time:
        new_dt = datetime.combine(date.today(), self._time) + timedelta(hours=hours)
        return Time(new_dt.time())

    def add_minutes(self, minutes: int) -> Time:
        new_dt = datetime.combine(date.today(), self._time) + timedelta(minutes=minutes)
        return Time(new_dt.time())

    def add_seconds(self, seconds: int) -> Time:
        new_dt = datetime.combine(date.today(), self._time) + timedelta(seconds=seconds)
        return Time(new_dt.time())

    def __str__(self) -> str:
        return self._time.isoformat()

    def __repr__(self) -> str:
        return f"Time({self._time!r})"


# ============================================================================
# TIMEZONE
# ============================================================================


class Timezone:
    """Timezone handling and conversion utilities."""

    def __init__(self, tz: timezone):
        self._tz = tz

    @property
    def timezone(self) -> timezone:
        return self._tz

    @classmethod
    def UTC(cls) -> Timezone:
        return cls(timezone.utc)

    @classmethod
    def local_tz(cls) -> Timezone:
        return cls(timezone.utc)

    @classmethod
    def offset(cls, hours: int, minutes: int = 0) -> Timezone:
        return cls(timezone(timedelta(hours=hours, minutes=minutes)))

    @classmethod
    def convert_tz(cls, dt: DateTime, from_tz: Timezone, to_tz: Timezone) -> DateTime:
        utc_dt = dt._dt.replace(tzinfo=from_tz._tz).astimezone(timezone.utc)
        converted = utc_dt.astimezone(to_tz._tz)
        return DateTime(converted.replace(tzinfo=None))

    @classmethod
    def to_utc(cls, dt: DateTime, from_tz: Timezone) -> DateTime:
        localized = dt._dt.replace(tzinfo=from_tz._tz)
        utc_dt = localized.astimezone(timezone.utc)
        return DateTime(utc_dt.replace(tzinfo=None))

    @classmethod
    def from_utc(cls, dt: DateTime, to_tz: Timezone) -> DateTime:
        utc_dt = dt._dt.replace(tzinfo=timezone.utc)
        localized = utc_dt.astimezone(to_tz._tz)
        return DateTime(localized.replace(tzinfo=None))

    def __str__(self) -> str:
        offset = self._tz.utcoffset(datetime.now())
        hours, remainder = divmod(abs(offset.seconds), 3600)
        minutes = remainder // 60
        sign = "+" if offset >= timedelta(0) else "-"
        return f"UTC{sign}{hours:02d}:{minutes:02d}"

    def __repr__(self) -> str:
        return f"Timezone({self._tz!r})"


# ============================================================================
# DURATION
# ============================================================================


class Duration:
    """Duration/interval handling."""

    def __init__(self, delta: Optional[timedelta] = None):
        self._delta = delta if delta is not None else timedelta(0)

    @property
    def timedelta(self) -> timedelta:
        return self._delta

    @classmethod
    def days(cls, days: int) -> Duration:
        return cls(timedelta(days=days))

    @classmethod
    def hours(cls, hours: int) -> Duration:
        return cls(timedelta(hours=hours))

    @classmethod
    def minutes(cls, minutes: int) -> Duration:
        return cls(timedelta(minutes=minutes))

    @classmethod
    def seconds(cls, seconds: int) -> Duration:
        return cls(timedelta(seconds=seconds))

    @classmethod
    def milliseconds(cls, ms: int) -> Duration:
        return cls(timedelta(milliseconds=ms))

    @classmethod
    def microseconds(cls, us: int) -> Duration:
        return cls(timedelta(microseconds=us))

    def total_seconds(self) -> float:
        return self._delta.total_seconds()

    def total_milliseconds(self) -> float:
        return self._delta.total_seconds() * 1000

    def total_microseconds(self) -> float:
        return self._delta.total_seconds() * 1000000

    def in_days(self) -> int:
        return self._delta.days

    def in_hours(self) -> int:
        return self._delta.seconds // 3600

    def in_minutes(self) -> int:
        return (self._delta.seconds % 3600) // 60

    def in_seconds(self) -> int:
        return self._delta.seconds + (self._delta.days * 86400)

    def __str__(self) -> str:
        total = self.total_seconds()
        if total >= 86400:
            days = int(total // 86400)
            remaining = int(total % 86400)
            hours = remaining // 3600
            minutes = (remaining % 3600) // 60
            seconds = remaining % 60
            return f"{days}d {hours}h {minutes}m {seconds}s"
        elif total >= 3600:
            hours = int(total // 3600)
            minutes = int((total % 3600) // 60)
            seconds = int(total % 60)
            return f"{hours}h {minutes}m {seconds}s"
        elif total >= 60:
            minutes = int(total // 60)
            seconds = int(total % 60)
            return f"{minutes}m {seconds}s"
        else:
            return f"{int(total)}s"

    def __repr__(self) -> str:
        return f"Duration({self._delta!r})"

    def __add__(self, other: Duration) -> Duration:
        return Duration(self._delta + other._delta)

    def __sub__(self, other: Duration) -> Duration:
        return Duration(self._delta - other._delta)

    def __mul__(self, n: int) -> Duration:
        return Duration(self._delta * n)

    def __truediv__(self, n: float) -> Duration:
        return Duration(self._delta / n)


# ============================================================================
# PARSING
# ============================================================================


def parse_datetime(date_string: str, formats: Optional[list[str]] = None) -> DateTime:
    if formats is None:
        formats = [
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
            "%d/%m/%Y %H:%M:%S",
            "%m/%d/%Y %H:%M:%S",
            "%Y-%m-%d",
            "%d-%m-%Y",
            "%m-%d-%Y",
        ]

    for fmt in formats:
        try:
            return DateTime(datetime.strptime(date_string, fmt))
        except ValueError:
            continue

    raise ValueError(f"Unable to parse datetime from: {date_string}")


def parse_date(date_string: str, formats: Optional[list[str]] = None) -> Date:
    if formats is None:
        formats = [
            "%Y-%m-%d",
            "%d/%m/%Y",
            "%m/%d/%Y",
            "%Y/%m/%d",
            "%d-%m-%Y",
            "%m-%d-%Y",
            "%B %d, %Y",
            "%b %d, %Y",
        ]

    for fmt in formats:
        try:
            return Date(datetime.strptime(date_string, fmt).date())
        except ValueError:
            continue

    raise ValueError(f"Unable to parse date from: {date_string}")


def parse_time(time_string: str, formats: Optional[list[str]] = None) -> Time:
    if formats is None:
        formats = [
            "%H:%M:%S.%f",
            "%H:%M:%S",
            "%H:%M",
            "%I:%M:%S %p",
            "%I:%M %p",
        ]

    for fmt in formats:
        try:
            return Time(dt_time.strptime(time_string, fmt))
        except ValueError:
            continue

    raise ValueError(f"Unable to parse time from: {time_string}")


def format_datetime(dt: DateTime, format_string: str = "%Y-%m-%d %H:%M:%S") -> str:
    return dt._dt.strftime(format_string)


def format_date(d: Date, format_string: str = "%Y-%m-%d") -> str:
    return d._date.strftime(format_string)


def format_time(t: Time, format_string: str = "%H:%M:%S") -> str:
    return t._time.strftime(format_string)


# ============================================================================
# CALENDAR
# ============================================================================


class Calendar:
    """Calendar utilities."""

    @staticmethod
    def is_leap_year(year: int) -> bool:
        return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

    @staticmethod
    def days_in_month(year: int, month: int) -> int:
        if month in (1, 3, 5, 7, 8, 10, 12):
            return 31
        elif month in (4, 6, 9, 11):
            return 30
        elif month == 2:
            return 29 if Calendar.is_leap_year(year) else 28
        raise ValueError(f"Invalid month: {month}")

    @staticmethod
    def weekday(year: int, month: int, day: int) -> int:
        return date(year, month, day).weekday()

    @staticmethod
    def iso_calendar(year: int, month: int, day: int) -> tuple[int, int, int]:
        return date(year, month, day).isocalendar()

    @staticmethod
    def days_between(start: Date, end: Date) -> int:
        return (end._date - start._date).days

    @staticmethod
    def weeks_between(start: Date, end: Date) -> int:
        return (end._date - start._date).days // 7

    @staticmethod
    def month_range(year: int, month: int) -> tuple[Date, Date]:
        first_day = Date(date(year, month, 1))
        days = Calendar.days_in_month(year, month)
        last_day = Date(date(year, month, days))
        return first_day, last_day

    @staticmethod
    def year_days(year: int) -> int:
        return 366 if Calendar.is_leap_year(year) else 365


# ============================================================================
# UTILITIES
# ============================================================================


def timestamp_now() -> float:
    return time_module.time()


def timestamp_ms() -> int:
    return int(time_module.time() * 1000)


def timestamp_us() -> int:
    return int(time_module.time() * 1000000)


def sleep(seconds: float) -> None:
    time_module.sleep(seconds)


def sleep_ms(milliseconds: int) -> None:
    time_module.sleep(milliseconds / 1000.0)


class Timer:
    """Context manager for timing code execution."""

    def __init__(self, name: str = "Timer"):
        self.name = name
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.elapsed: Optional[float] = None

    def __enter__(self) -> Timer:
        self.start_time = time_module.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.end_time = time_module.perf_counter()
        self.elapsed = self.end_time - self.start_time

    def elapsed_ms(self) -> float:
        if self.elapsed is None:
            raise RuntimeError("Timer has not been run")
        return self.elapsed * 1000

    def elapsed_us(self) -> float:
        if self.elapsed is None:
            raise RuntimeError("Timer has not been run")
        return self.elapsed * 1000000

    def __str__(self) -> str:
        if self.elapsed is not None:
            return f"{self.name}: {self.elapsed:.6f}s"
        return f"{self.name}: not started"

    def __repr__(self) -> str:
        return f"Timer(name={self.name!r}, elapsed={self.elapsed!r})"
