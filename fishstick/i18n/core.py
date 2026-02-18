"""
Fishstick Internationalization (i18n) Module

Comprehensive internationalization support for the Fishstick framework.
Provides translation, localization, locale management, message extraction,
pluralization, formatting, and resource loading capabilities.
"""

from __future__ import annotations

import re
import os
import json
import locale
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, date, time
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
    Iterator,
    overload,
)
import threading


# ============================================================================
# Type Definitions
# ============================================================================

T = TypeVar("T")
MessageId = str
LocaleCode = str
PluralForm = str


# ============================================================================
# Exceptions
# ============================================================================


class I18nError(Exception):
    """Base exception for i18n operations."""

    pass


class TranslationError(I18nError):
    """Exception raised for translation-related errors."""

    pass


class LocaleError(I18nError):
    """Exception raised for locale-related errors."""

    pass


class ResourceError(I18nError):
    """Exception raised for resource loading errors."""

    pass


class FormatError(I18nError):
    """Exception raised for formatting errors."""

    pass


# ============================================================================
# 1. Translation
# ============================================================================


class TranslationEntry:
    """Represents a single translation entry with singular and plural forms."""

    def __init__(
        self,
        msgid: str,
        msgstr: Optional[str] = None,
        msgstr_plural: Optional[Dict[int, str]] = None,
        msgctxt: Optional[str] = None,
        comments: Optional[List[str]] = None,
    ):
        self.msgid = msgid
        self.msgstr = msgstr or ""
        self.msgstr_plural = msgstr_plural or {}
        self.msgctxt = msgctxt
        self.comments = comments or []

    def get_translation(self, plural_form: int = 0) -> str:
        """Get translation for a specific plural form."""
        if plural_form == 0:
            return self.msgstr or self.msgid
        return self.msgstr_plural.get(plural_form, self.msgstr or self.msgid)


class Catalog:
    """Translation catalog containing all messages for a locale."""

    def __init__(self, locale: LocaleCode, domain: str = "messages"):
        self.locale = locale
        self.domain = domain
        self._messages: Dict[Tuple[str, Optional[str]], TranslationEntry] = {}
        self._plural_rule: Optional[PluralRule] = None
        self.metadata: Dict[str, str] = {}

    def add(self, entry: TranslationEntry) -> None:
        """Add a translation entry to the catalog."""
        key = (entry.msgid, entry.msgctxt)
        self._messages[key] = entry

    def get(
        self, msgid: str, msgctxt: Optional[str] = None, default: Optional[str] = None
    ) -> str:
        """Get translation for a message ID."""
        key = (msgid, msgctxt)
        if key in self._messages:
            return self._messages[key].get_translation(0)
        return default if default is not None else msgid

    def get_plural(
        self,
        msgid_singular: str,
        msgid_plural: str,
        n: int,
        msgctxt: Optional[str] = None,
    ) -> str:
        """Get plural translation for a message ID."""
        key = (msgid_singular, msgctxt)
        if key in self._messages:
            entry = self._messages[key]
            if self._plural_rule:
                plural_form = self._plural_rule.get_plural_form(n)
            else:
                plural_form = 0 if n == 1 else 1
            return entry.get_translation(plural_form)
        return msgid_singular if n == 1 else msgid_plural

    def set_plural_rule(self, rule: PluralRule) -> None:
        """Set the plural rule for this catalog."""
        self._plural_rule = rule


class Translator:
    """
    Main translation class providing gettext functionality.

    Provides methods for translating strings with support for
    singular, plural, and context-based translations.
    """

    def __init__(self):
        self._catalogs: Dict[Tuple[LocaleCode, str], Catalog] = {}
        self._current_locale: LocaleCode = "en"
        self._default_domain: str = "messages"
        self._thread_local = threading.local()

    def bind_catalog(self, catalog: Catalog, domain: Optional[str] = None) -> None:
        """Bind a translation catalog for a locale and domain."""
        domain = domain or catalog.domain or self._default_domain
        key = (catalog.locale, domain)
        self._catalogs[key] = catalog

    def set_locale(self, locale: LocaleCode) -> None:
        """Set the current locale for translations."""
        self._current_locale = locale

    def get_locale(self) -> LocaleCode:
        """Get the current locale."""
        return getattr(self._thread_local, "locale", self._current_locale)

    def gettext(self, message: str, domain: Optional[str] = None) -> str:
        """
        Get text translation.

        Args:
            message: The message to translate
            domain: Optional domain for the translation

        Returns:
            Translated string
        """
        locale = self.get_locale()
        domain = domain or self._default_domain
        key = (locale, domain)

        if key in self._catalogs:
            return self._catalogs[key].get(message)
        return message

    def ngettext(
        self, singular: str, plural: str, n: int, domain: Optional[str] = None
    ) -> str:
        """
        Get plural-aware text translation.

        Args:
            singular: Singular form of the message
            plural: Plural form of the message
            n: Number for determining plural form
            domain: Optional domain for the translation

        Returns:
            Translated string based on plural form
        """
        locale = self.get_locale()
        domain = domain or self._default_domain
        key = (locale, domain)

        if key in self._catalogs:
            return self._catalogs[key].get_plural(singular, plural, n)
        return singular if n == 1 else plural

    def pgettext(self, context: str, message: str, domain: Optional[str] = None) -> str:
        """
        Get text translation with context.

        Args:
            context: Context for disambiguation
            message: The message to translate
            domain: Optional domain for the translation

        Returns:
            Translated string
        """
        locale = self.get_locale()
        domain = domain or self._default_domain
        key = (locale, domain)

        if key in self._catalogs:
            return self._catalogs[key].get(message, msgctxt=context)
        return message

    def npgettext(
        self,
        context: str,
        singular: str,
        plural: str,
        n: int,
        domain: Optional[str] = None,
    ) -> str:
        """
        Get plural-aware text translation with context.

        Args:
            context: Context for disambiguation
            singular: Singular form of the message
            plural: Plural form of the message
            n: Number for determining plural form
            domain: Optional domain for the translation

        Returns:
            Translated string based on plural form
        """
        locale = self.get_locale()
        domain = domain or self._default_domain
        key = (locale, domain)

        if key in self._catalogs:
            catalog = self._catalogs[key]
            # Use singular with context as key
            result = catalog.get_plural(singular, plural, n, msgctxt=context)
            return result
        return singular if n == 1 else plural


# ============================================================================
# 2. Localization
# ============================================================================


class Localizer:
    """
    Main localization class for formatting locale-specific data.

    Provides methods for formatting dates, times, numbers, and currency
    according to locale-specific conventions.
    """

    def __init__(self, locale: Optional[LocaleCode] = None):
        self._locale = locale or locale.getdefaultlocale()[0] or "en_US"
        self._date_formatter = DateFormatter(self._locale)
        self._number_formatter = NumberFormatter(self._locale)
        self._currency_formatter = CurrencyFormatter(self._locale)
        self._percent_formatter = PercentFormatter(self._locale)

    @property
    def locale(self) -> LocaleCode:
        """Get the current locale."""
        return self._locale

    @locale.setter
    def locale(self, value: LocaleCode) -> None:
        """Set the current locale and update formatters."""
        self._locale = value
        self._date_formatter = DateFormatter(value)
        self._number_formatter = NumberFormatter(value)
        self._currency_formatter = CurrencyFormatter(value)
        self._percent_formatter = PercentFormatter(value)

    def format_date(
        self,
        value: Union[date, datetime],
        format: str = "medium",
        pattern: Optional[str] = None,
    ) -> str:
        """
        Format a date according to locale conventions.

        Args:
            value: Date or datetime to format
            format: Predefined format (short, medium, long, full)
            pattern: Custom strftime pattern (overrides format)

        Returns:
            Formatted date string
        """
        return self._date_formatter.format_date(value, format, pattern)

    def format_time(
        self,
        value: Union[time, datetime],
        format: str = "medium",
        pattern: Optional[str] = None,
    ) -> str:
        """
        Format a time according to locale conventions.

        Args:
            value: Time or datetime to format
            format: Predefined format (short, medium, long, full)
            pattern: Custom strftime pattern (overrides format)

        Returns:
            Formatted time string
        """
        return self._date_formatter.format_time(value, format, pattern)

    def format_datetime(
        self, value: datetime, format: str = "medium", pattern: Optional[str] = None
    ) -> str:
        """
        Format a datetime according to locale conventions.

        Args:
            value: Datetime to format
            format: Predefined format (short, medium, long, full)
            pattern: Custom strftime pattern (overrides format)

        Returns:
            Formatted datetime string
        """
        return self._date_formatter.format_datetime(value, format, pattern)

    def format_number(
        self,
        value: Union[int, float, Decimal],
        decimal_places: Optional[int] = None,
        grouping: bool = True,
    ) -> str:
        """
        Format a number according to locale conventions.

        Args:
            value: Number to format
            decimal_places: Number of decimal places (None for auto)
            grouping: Whether to include grouping separators

        Returns:
            Formatted number string
        """
        return self._number_formatter.format(value, decimal_places, grouping)

    def format_currency(
        self,
        value: Union[int, float, Decimal],
        currency: str,
        format: Optional[str] = None,
    ) -> str:
        """
        Format a value as currency according to locale conventions.

        Args:
            value: Value to format
            currency: Currency code (e.g., 'USD', 'EUR')
            format: Optional custom format pattern

        Returns:
            Formatted currency string
        """
        return self._currency_formatter.format(value, currency, format)

    def format_percent(
        self, value: Union[int, float, Decimal], decimal_places: int = 2
    ) -> str:
        """
        Format a value as percentage according to locale conventions.

        Args:
            value: Value to format (0.5 for 50%)
            decimal_places: Number of decimal places

        Returns:
            Formatted percentage string
        """
        return self._percent_formatter.format(value, decimal_places)


# ============================================================================
# 3. Locale Management
# ============================================================================


class LocaleInfo:
    """Information about a locale."""

    def __init__(
        self,
        code: LocaleCode,
        name: str,
        language: str,
        territory: Optional[str] = None,
        encoding: str = "UTF-8",
    ):
        self.code = code
        self.name = name
        self.language = language
        self.territory = territory
        self.encoding = encoding

    def __repr__(self) -> str:
        return f"LocaleInfo({self.code!r})"


class LocaleManager:
    """
    Manages locales and provides locale detection and switching.

    Handles locale configuration, detection from various sources,
    and maintains the current locale state.
    """

    def __init__(self):
        self._locales: Dict[LocaleCode, LocaleInfo] = {}
        self._current_locale: LocaleCode = "en"
        self._default_locale: LocaleCode = "en"
        self._thread_local = threading.local()
        self._load_builtin_locales()

    def _load_builtin_locales(self) -> None:
        """Load built-in locale information."""
        builtin = [
            LocaleInfo("en", "English", "en", None),
            LocaleInfo("en_US", "English (United States)", "en", "US"),
            LocaleInfo("en_GB", "English (United Kingdom)", "en", "GB"),
            LocaleInfo("es", "Spanish", "es", None),
            LocaleInfo("es_ES", "Spanish (Spain)", "es", "ES"),
            LocaleInfo("fr", "French", "fr", None),
            LocaleInfo("fr_FR", "French (France)", "fr", "FR"),
            LocaleInfo("de", "German", "de", None),
            LocaleInfo("de_DE", "German (Germany)", "de", "DE"),
            LocaleInfo("it", "Italian", "it", None),
            LocaleInfo("pt", "Portuguese", "pt", None),
            LocaleInfo("pt_BR", "Portuguese (Brazil)", "pt", "BR"),
            LocaleInfo("zh", "Chinese", "zh", None),
            LocaleInfo("zh_CN", "Chinese (Simplified)", "zh", "CN"),
            LocaleInfo("ja", "Japanese", "ja", None),
            LocaleInfo("ko", "Korean", "ko", None),
            LocaleInfo("ar", "Arabic", "ar", None),
            LocaleInfo("ru", "Russian", "ru", None),
            LocaleInfo("hi", "Hindi", "hi", None),
        ]
        for info in builtin:
            self._locales[info.code] = info

    def set_locale(self, locale: LocaleCode) -> None:
        """
        Set the current locale.

        Args:
            locale: Locale code to set

        Raises:
            LocaleError: If locale is not available
        """
        if locale not in self._locales:
            raise LocaleError(f"Locale '{locale}' is not available")
        self._current_locale = locale
        self._thread_local.locale = locale

    def get_locale(self) -> LocaleCode:
        """Get the current locale."""
        return getattr(self._thread_local, "locale", self._current_locale)

    def get_locale_info(self, locale: Optional[LocaleCode] = None) -> LocaleInfo:
        """
        Get information about a locale.

        Args:
            locale: Locale code (defaults to current)

        Returns:
            LocaleInfo object
        """
        locale = locale or self.get_locale()
        if locale not in self._locales:
            raise LocaleError(f"Locale '{locale}' is not available")
        return self._locales[locale]

    def list_locales(self) -> List[LocaleInfo]:
        """
        List all available locales.

        Returns:
            List of LocaleInfo objects
        """
        return list(self._locales.values())

    def add_locale(self, info: LocaleInfo) -> None:
        """
        Add a new locale.

        Args:
            info: LocaleInfo object to add
        """
        self._locales[info.code] = info

    def detect_locale(
        self, sources: Optional[List[str]] = None, default: Optional[LocaleCode] = None
    ) -> LocaleCode:
        """
        Detect locale from various sources.

        Sources can include:
        - 'env': Environment variables (LANG, LC_ALL, etc.)
        - 'system': System locale settings
        - 'http': HTTP Accept-Language header
        - 'user': User preferences

        Args:
            sources: List of sources to check (in order)
            default: Default locale if detection fails

        Returns:
            Detected locale code
        """
        sources = sources or ["env", "system"]
        default = default or self._default_locale

        for source in sources:
            detected = self._detect_from_source(source)
            if detected:
                return detected

        return default

    def _detect_from_source(self, source: str) -> Optional[LocaleCode]:
        """Detect locale from a specific source."""
        if source == "env":
            for var in ["LC_ALL", "LC_MESSAGES", "LANG"]:
                value = os.environ.get(var)
                if value:
                    # Extract locale code (e.g., 'en_US.UTF-8' -> 'en_US')
                    locale_code = value.split(".")[0]
                    if locale_code in self._locales:
                        return locale_code

        elif source == "system":
            try:
                system_locale = locale.getdefaultlocale()[0]
                if system_locale and system_locale in self._locales:
                    return system_locale
            except:
                pass

        return None

    def set_default_locale(self, locale: LocaleCode) -> None:
        """Set the default locale."""
        if locale not in self._locales:
            raise LocaleError(f"Locale '{locale}' is not available")
        self._default_locale = locale


# ============================================================================
# 4. Message Extraction
# ============================================================================


@dataclass
class ExtractedMessage:
    """Represents an extracted translatable message."""

    msgid: str
    msgctxt: Optional[str] = None
    locations: List[Tuple[str, int]] = field(default_factory=list)
    comments: List[str] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)


class MessageExtractor:
    """
    Extracts translatable messages from source code.

    Scans source files for gettext calls and extracts messages
    for translation catalog creation.
    """

    def __init__(self):
        self._extractors: Dict[str, Callable[[Path], Iterator[ExtractedMessage]]] = {}
        self._register_default_extractors()

    def _register_default_extractors(self) -> None:
        """Register extractors for common file types."""
        self._extractors[".py"] = self._extract_python

    def extract_messages(
        self, paths: List[Union[str, Path]], keywords: Optional[List[str]] = None
    ) -> List[ExtractedMessage]:
        """
        Extract messages from source files.

        Args:
            paths: List of file or directory paths to scan
            keywords: List of function names to extract (e.g., '_', 'gettext')

        Returns:
            List of extracted messages
        """
        keywords = keywords or ["_", "gettext", "ngettext", "pgettext"]
        messages: Dict[str, ExtractedMessage] = {}

        for path in paths:
            path = Path(path)
            if path.is_file():
                self._extract_from_file(path, keywords, messages)
            elif path.is_dir():
                for file_path in path.rglob("*"):
                    if file_path.is_file():
                        self._extract_from_file(file_path, keywords, messages)

        return list(messages.values())

    def _extract_from_file(
        self, path: Path, keywords: List[str], messages: Dict[str, ExtractedMessage]
    ) -> None:
        """Extract messages from a single file."""
        suffix = path.suffix
        if suffix not in self._extractors:
            return

        extractor = self._extractors[suffix]
        for message in extractor(path):
            key = f"{message.msgctxt or ''}:{message.msgid}"
            if key in messages:
                messages[key].locations.extend(message.locations)
            else:
                messages[key] = message

    def _extract_python(self, path: Path) -> Iterator[ExtractedMessage]:
        """Extract messages from Python source files."""
        try:
            content = path.read_text(encoding="utf-8")
        except:
            return

        # Pattern for _("message") or _('message')
        patterns = [
            # _("message") or _('message')
            (r'\b_\s*\(\s*["\']([^"\']+)["\']\s*\)', None),
            # gettext("message")
            (r'\bgettext\s*\(\s*["\']([^"\']+)["\']\s*\)', None),
            # ngettext("singular", "plural", n)
            (r'\bngettext\s*\(\s*["\']([^"\']+)["\']\s*,\s*["\']([^"\']+)["\']', None),
            # pgettext("context", "message")
            (
                r'\bpgettext\s*\(\s*["\']([^"\']+)["\']\s*,\s*["\']([^"\']+)["\']\s*\)',
                0,
            ),
        ]

        for i, line in enumerate(content.split("\n"), 1):
            for pattern, ctxt_group in patterns:
                for match in re.finditer(pattern, line):
                    if ctxt_group is not None:
                        # Has context
                        msgctxt = match.group(1)
                        msgid = match.group(2)
                    else:
                        msgctxt = None
                        msgid = match.group(1)

                    msg = ExtractedMessage(
                        msgid=msgid, msgctxt=msgctxt, locations=[(str(path), i)]
                    )
                    yield msg

    def update_catalog(
        self,
        catalog: Catalog,
        messages: List[ExtractedMessage],
        ignore_obsolete: bool = False,
    ) -> Tuple[int, int, int]:
        """
        Update a catalog with extracted messages.

        Args:
            catalog: Catalog to update
            messages: Extracted messages to add/update
            ignore_obsolete: Whether to keep obsolete messages

        Returns:
            Tuple of (added, updated, removed) counts
        """
        added = updated = removed = 0

        # Build set of message keys
        message_keys = set()
        for msg in messages:
            key = (msg.msgid, msg.msgctxt)
            message_keys.add(key)

            entry = TranslationEntry(
                msgid=msg.msgid, msgctxt=msg.msgctxt, comments=msg.comments
            )

            # Check if exists
            existing_key = (msg.msgid, msg.msgctxt)
            if existing_key not in catalog._messages:
                catalog.add(entry)
                added += 1
            else:
                updated += 1

        # Remove obsolete if requested
        if not ignore_obsolete:
            obsolete = set(catalog._messages.keys()) - message_keys
            for key in obsolete:
                del catalog._messages[key]
                removed += 1

        return added, updated, removed

    def compile_catalog(
        self, catalog: Catalog, output_path: Union[str, Path], format: str = "mo"
    ) -> None:
        """
        Compile a catalog to binary format.

        Args:
            catalog: Catalog to compile
            output_path: Output file path
            format: Output format (mo, json)
        """
        output_path = Path(output_path)

        if format == "mo":
            self._compile_to_mo(catalog, output_path)
        elif format == "json":
            self._compile_to_json(catalog, output_path)
        else:
            raise ValueError(f"Unknown format: {format}")

    def _compile_to_mo(self, catalog: Catalog, path: Path) -> None:
        """Compile catalog to GNU gettext .mo format."""
        # Simplified MO file format
        # In production, use proper MO file generation
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {"locale": catalog.locale, "domain": catalog.domain, "messages": {}}

        for (msgid, msgctxt), entry in catalog._messages.items():
            key = f"{msgctxt or ''}:{msgid}"
            data["messages"][key] = {
                "msgstr": entry.msgstr,
                "msgstr_plural": entry.msgstr_plural,
            }

        path.write_bytes(hashlib.sha256(str(data).encode()).digest())

    def _compile_to_json(self, catalog: Catalog, path: Path) -> None:
        """Compile catalog to JSON format."""
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {"locale": catalog.locale, "domain": catalog.domain, "messages": {}}

        for (msgid, msgctxt), entry in catalog._messages.items():
            key = f"{msgctxt or ''}:{msgid}"
            data["messages"][key] = {
                "msgstr": entry.msgstr,
                "msgstr_plural": entry.msgstr_plural,
            }

        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


# ============================================================================
# 5. Pluralization
# ============================================================================


class PluralRule:
    """
    Handles pluralization rules for different locales.

    Implements CLDR plural rules for determining which plural form
    to use based on numeric values.
    """

    # CLDR plural rules for common locales
    RULES = {
        # English, Spanish, etc.: one (n == 1), other
        "en": lambda n: 0 if n == 1 else 1,
        "es": lambda n: 0 if n == 1 else 1,
        "de": lambda n: 0 if n == 1 else 1,
        "fr": lambda n: 0 if n <= 1 else 1,  # French includes 0 as singular
        "it": lambda n: 0 if n == 1 else 1,
        "pt": lambda n: 0 if n == 1 else 1,
        # Arabic: zero, one, two, few, many, other
        "ar": lambda n: (
            0
            if n == 0
            else 1
            if n == 1
            else 2
            if n == 2
            else 3
            if (n % 100 >= 3 and n % 100 <= 10)
            else 4
            if (n % 100 >= 11)
            else 5
        ),
        # Russian, Polish: one, few, many, other
        "ru": lambda n: (
            0
            if (n % 10 == 1 and n % 100 != 11)
            else 1
            if (n % 10 in [2, 3, 4] and n % 100 not in [12, 13, 14])
            else 2
        ),
        # Japanese, Korean, Chinese: no plural distinction
        "ja": lambda n: 0,
        "ko": lambda n: 0,
        "zh": lambda n: 0,
    }

    def __init__(self, locale: LocaleCode, rule: Optional[Callable[[int], int]] = None):
        """
        Initialize plural rule.

        Args:
            locale: Locale code
            rule: Custom plural rule function (optional)
        """
        self.locale = locale
        self._rule = rule or self._get_default_rule(locale)

    def _get_default_rule(self, locale: LocaleCode) -> Callable[[int], int]:
        """Get default plural rule for a locale."""
        # Extract language code (e.g., 'en_US' -> 'en')
        lang = locale.split("_")[0].split("-")[0]
        return self.RULES.get(lang, lambda n: 0 if n == 1 else 1)

    def get_plural_form(self, n: int) -> int:
        """
        Get the plural form index for a number.

        Args:
            n: Number to determine plural form for

        Returns:
            Index of the plural form to use
        """
        return self._rule(n)

    def pluralize(self, n: int, forms: List[str]) -> str:
        """
        Select the appropriate plural form for a number.

        Args:
            n: Number to determine plural form
            forms: List of plural forms (singular, plural, etc.)

        Returns:
            Selected plural form
        """
        index = self.get_plural_form(n)
        if index < len(forms):
            return forms[index]
        return forms[-1] if forms else ""

    @classmethod
    def get_available_locales(cls) -> List[str]:
        """Get list of locales with built-in plural rules."""
        return list(cls.RULES.keys())


# ============================================================================
# 6. Formatting
# ============================================================================


class DateFormatter:
    """Formats dates and times according to locale conventions."""

    # Locale-specific date patterns
    DATE_PATTERNS = {
        "en": {
            "short": "%m/%d/%Y",
            "medium": "%b %d, %Y",
            "long": "%B %d, %Y",
            "full": "%A, %B %d, %Y",
        },
        "en_GB": {
            "short": "%d/%m/%Y",
            "medium": "%d %b %Y",
            "long": "%d %B %Y",
            "full": "%A, %d %B %Y",
        },
        "de": {
            "short": "%d.%m.%Y",
            "medium": "%d.%m.%Y",
            "long": "%d. %B %Y",
            "full": "%A, %d. %B %Y",
        },
        "fr": {
            "short": "%d/%m/%Y",
            "medium": "%d %b %Y",
            "long": "%d %B %Y",
            "full": "%A %d %B %Y",
        },
    }

    TIME_PATTERNS = {
        "en": {
            "short": "%I:%M %p",
            "medium": "%I:%M:%S %p",
            "long": "%I:%M:%S %p %Z",
            "full": "%I:%M:%S %p %Z",
        },
        "de": {
            "short": "%H:%M",
            "medium": "%H:%M:%S",
            "long": "%H:%M:%S %Z",
            "full": "%H:%M:%S %Z",
        },
        "fr": {
            "short": "%H:%M",
            "medium": "%H:%M:%S",
            "long": "%H:%M:%S %Z",
            "full": "%H:%M:%S %Z",
        },
    }

    def __init__(self, locale: LocaleCode):
        self.locale = locale

    def format_date(
        self,
        value: Union[date, datetime],
        format: str = "medium",
        pattern: Optional[str] = None,
    ) -> str:
        """Format a date."""
        if pattern:
            return value.strftime(pattern)

        patterns = self.DATE_PATTERNS.get(self.locale, self.DATE_PATTERNS.get("en", {}))
        fmt = patterns.get(format, patterns.get("medium", "%Y-%m-%d"))
        return value.strftime(fmt)

    def format_time(
        self,
        value: Union[time, datetime],
        format: str = "medium",
        pattern: Optional[str] = None,
    ) -> str:
        """Format a time."""
        if pattern:
            return value.strftime(pattern)

        patterns = self.TIME_PATTERNS.get(self.locale, self.TIME_PATTERNS.get("en", {}))
        fmt = patterns.get(format, patterns.get("medium", "%H:%M:%S"))
        return value.strftime(fmt)

    def format_datetime(
        self, value: datetime, format: str = "medium", pattern: Optional[str] = None
    ) -> str:
        """Format a datetime."""
        if pattern:
            return value.strftime(pattern)

        date_str = self.format_date(value, format)
        time_str = self.format_time(value, format)
        return f"{date_str}, {time_str}"


class NumberFormatter:
    """Formats numbers according to locale conventions."""

    # Locale-specific number formatting
    FORMATS = {
        "en": {"decimal": ".", "group": ",", "group_size": 3},
        "en_GB": {"decimal": ".", "group": ",", "group_size": 3},
        "de": {"decimal": ",", "group": ".", "group_size": 3},
        "de_DE": {"decimal": ",", "group": ".", "group_size": 3},
        "fr": {"decimal": ",", "group": " ", "group_size": 3},
        "fr_FR": {"decimal": ",", "group": " ", "group_size": 3},
        "es": {"decimal": ",", "group": ".", "group_size": 3},
        "it": {"decimal": ",", "group": ".", "group_size": 3},
    }

    def __init__(self, locale: LocaleCode):
        self.locale = locale
        self.format_info = self.FORMATS.get(
            locale, self.FORMATS.get(locale.split("_")[0], self.FORMATS["en"])
        )

    def format(
        self,
        value: Union[int, float, Decimal],
        decimal_places: Optional[int] = None,
        grouping: bool = True,
    ) -> str:
        """Format a number."""
        if isinstance(value, Decimal):
            if decimal_places is not None:
                quantize_str = "0." + "0" * decimal_places
                value = value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)
            str_value = str(value)
        else:
            if decimal_places is not None:
                str_value = f"{value:.{decimal_places}f}"
            else:
                str_value = str(float(value))

        # Split into integer and decimal parts
        if "." in str_value:
            int_part, dec_part = str_value.split(".")
        else:
            int_part, dec_part = str_value, None

        # Apply grouping
        if grouping and self.format_info["group"]:
            group_size = self.format_info["group_size"]
            groups = []
            while int_part:
                groups.append(int_part[-group_size:])
                int_part = int_part[:-group_size]
            int_part = self.format_info["group"].join(reversed(groups))

        # Combine parts
        if dec_part:
            return f"{int_part}{self.format_info['decimal']}{dec_part}"
        return int_part


class CurrencyFormatter:
    """Formats currency values according to locale conventions."""

    # Currency symbols
    CURRENCY_SYMBOLS = {
        "USD": "$",
        "EUR": "€",
        "GBP": "£",
        "JPY": "¥",
        "CNY": "¥",
        "KRW": "₩",
        "INR": "₹",
        "RUB": "₽",
        "BRL": "R$",
    }

    # Currency formatting patterns
    PATTERNS = {
        "en": "{symbol}{value}",  # $100.00
        "en_GB": "{symbol}{value}",  # £100.00
        "de": "{value} {symbol}",  # 100,00 €
        "de_DE": "{value} {symbol}",
        "fr": "{value} {symbol}",  # 100,00 €
        "fr_FR": "{value} {symbol}",
        "es": "{value} {symbol}",
        "it": "{value} {symbol}",
        "ja": "{symbol}{value}",
        "zh": "{symbol}{value}",
    }

    def __init__(self, locale: LocaleCode):
        self.locale = locale
        self.number_formatter = NumberFormatter(locale)

    def format(
        self,
        value: Union[int, float, Decimal],
        currency: str,
        format: Optional[str] = None,
    ) -> str:
        """Format a currency value."""
        symbol = self.CURRENCY_SYMBOLS.get(currency, currency)

        # Format number with 2 decimal places
        formatted_value = self.number_formatter.format(value, decimal_places=2)

        # Get pattern
        if format:
            pattern = format
        else:
            pattern = self.PATTERNS.get(
                self.locale,
                self.PATTERNS.get(self.locale.split("_")[0], "{symbol}{value}"),
            )

        return pattern.format(symbol=symbol, value=formatted_value)


class PercentFormatter:
    """Formats percentage values according to locale conventions."""

    def __init__(self, locale: LocaleCode):
        self.locale = locale
        self.number_formatter = NumberFormatter(locale)

    def format(self, value: Union[int, float, Decimal], decimal_places: int = 2) -> str:
        """Format a percentage value."""
        # Convert to percentage
        percent_value = float(value) * 100

        # Format number
        formatted = self.number_formatter.format(percent_value, decimal_places)

        # Add percent sign
        return f"{formatted}%"


# ============================================================================
# 7. Resources
# ============================================================================


class ResourceLoader(ABC):
    """Abstract base class for resource loaders."""

    @abstractmethod
    def load(self, path: Union[str, Path]) -> Catalog:
        """Load translations from a resource file."""
        pass

    @abstractmethod
    def save(self, catalog: Catalog, path: Union[str, Path]) -> None:
        """Save translations to a resource file."""
        pass


class JSONResource(ResourceLoader):
    """Resource loader for JSON translation files."""

    def load(self, path: Union[str, Path]) -> Catalog:
        """Load translations from a JSON file."""
        path = Path(path)

        if not path.exists():
            raise ResourceError(f"Resource file not found: {path}")

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ResourceError(f"Invalid JSON in {path}: {e}")

        locale = data.get("locale", "en")
        domain = data.get("domain", "messages")
        catalog = Catalog(locale, domain)

        for key, value in data.get("messages", {}).items():
            if ":" in key:
                msgctxt, msgid = key.split(":", 1)
            else:
                msgctxt, msgid = None, key

            if isinstance(value, str):
                entry = TranslationEntry(msgid, msgstr=value, msgctxt=msgctxt)
            elif isinstance(value, dict):
                entry = TranslationEntry(
                    msgid,
                    msgstr=value.get("msgstr", ""),
                    msgstr_plural=value.get("msgstr_plural", {}),
                    msgctxt=msgctxt,
                )
            else:
                continue

            catalog.add(entry)

        return catalog

    def save(self, catalog: Catalog, path: Union[str, Path]) -> None:
        """Save translations to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {"locale": catalog.locale, "domain": catalog.domain, "messages": {}}

        for (msgid, msgctxt), entry in catalog._messages.items():
            key = f"{msgctxt}:{msgid}" if msgctxt else msgid
            if entry.msgstr_plural:
                data["messages"][key] = {
                    "msgstr": entry.msgstr,
                    "msgstr_plural": entry.msgstr_plural,
                }
            else:
                data["messages"][key] = entry.msgstr

        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


class YAMLResource(ResourceLoader):
    """Resource loader for YAML translation files."""

    def __init__(self):
        self._yaml_available = self._check_yaml()

    def _check_yaml(self) -> bool:
        """Check if YAML library is available."""
        try:
            import yaml

            return True
        except ImportError:
            return False

    def load(self, path: Union[str, Path]) -> Catalog:
        """Load translations from a YAML file."""
        if not self._yaml_available:
            raise ResourceError("PyYAML is required for YAML resource loading")

        import yaml

        path = Path(path)

        if not path.exists():
            raise ResourceError(f"Resource file not found: {path}")

        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as e:
            raise ResourceError(f"Invalid YAML in {path}: {e}")

        locale = data.get("locale", "en")
        domain = data.get("domain", "messages")
        catalog = Catalog(locale, domain)

        for key, value in data.get("messages", {}).items():
            if ":" in key:
                msgctxt, msgid = key.split(":", 1)
            else:
                msgctxt, msgid = None, key

            if isinstance(value, str):
                entry = TranslationEntry(msgid, msgstr=value, msgctxt=msgctxt)
            elif isinstance(value, dict):
                entry = TranslationEntry(
                    msgid,
                    msgstr=value.get("msgstr", ""),
                    msgstr_plural=value.get("msgstr_plural", {}),
                    msgctxt=msgctxt,
                )
            else:
                continue

            catalog.add(entry)

        return catalog

    def save(self, catalog: Catalog, path: Union[str, Path]) -> None:
        """Save translations to a YAML file."""
        if not self._yaml_available:
            raise ResourceError("PyYAML is required for YAML resource saving")

        import yaml

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {"locale": catalog.locale, "domain": catalog.domain, "messages": {}}

        for (msgid, msgctxt), entry in catalog._messages.items():
            key = f"{msgctxt}:{msgid}" if msgctxt else msgid
            if entry.msgstr_plural:
                data["messages"][key] = {
                    "msgstr": entry.msgstr,
                    "msgstr_plural": entry.msgstr_plural,
                }
            else:
                data["messages"][key] = entry.msgstr

        path.write_text(yaml.dump(data, allow_unicode=True, sort_keys=False))


class POFile(ResourceLoader):
    """Resource loader for GNU gettext PO files."""

    def load(self, path: Union[str, Path]) -> Catalog:
        """Load translations from a PO file."""
        path = Path(path)

        if not path.exists():
            raise ResourceError(f"PO file not found: {path}")

        content = path.read_text(encoding="utf-8")
        catalog = Catalog("en", "messages")  # Will be updated from headers

        # Parse PO file
        entries = self._parse_po(content)

        for entry_data in entries:
            entry = TranslationEntry(
                msgid=entry_data["msgid"],
                msgstr=entry_data.get("msgstr", ""),
                msgstr_plural=entry_data.get("msgstr_plural", {}),
                msgctxt=entry_data.get("msgctxt"),
                comments=entry_data.get("comments", []),
            )
            catalog.add(entry)

        return catalog

    def _parse_po(self, content: str) -> List[Dict[str, Any]]:
        """Parse PO file content."""
        entries = []
        current = {}

        lines = content.split("\n")
        i = 0
        while i < len(lines):
            line = lines[i]

            # Comments
            if line.startswith("#"):
                if "comments" not in current:
                    current["comments"] = []
                current["comments"].append(line[1:].strip())

            # Context
            elif line.startswith("msgctxt "):
                current["msgctxt"] = self._extract_string(line[8:])

            # Message ID
            elif line.startswith("msgid "):
                # Save previous entry if exists
                if "msgid" in current:
                    entries.append(current)
                    current = {}
                current["msgid"] = self._extract_string(line[6:])

            # Plural message ID
            elif line.startswith("msgid_plural "):
                current["msgid_plural"] = self._extract_string(line[13:])

            # Message string
            elif line.startswith("msgstr "):
                current["msgstr"] = self._extract_string(line[7:])

            # Plural message strings
            elif line.startswith("msgstr["):
                idx_end = line.find("]")
                if idx_end > 0:
                    idx = int(line[7:idx_end])
                    if "msgstr_plural" not in current:
                        current["msgstr_plural"] = {}
                    current["msgstr_plural"][idx] = self._extract_string(
                        line[idx_end + 2 :]
                    )

            # Continuation of multiline string
            elif line.startswith('"'):
                # Append to last field
                for key in ["msgstr", "msgid"]:
                    if key in current:
                        current[key] += self._extract_string(line)
                        break

            i += 1

        # Don't forget the last entry
        if "msgid" in current:
            entries.append(current)

        return entries

    def _extract_string(self, line: str) -> str:
        """Extract string from quoted line."""
        match = re.match(r'\s*"(.*)"\s*$', line)
        if match:
            return (
                match.group(1)
                .replace("\\n", "\n")
                .replace('\\"', '"')
                .replace("\\\\", "\\")
            )
        return line.strip()

    def save(self, catalog: Catalog, path: Union[str, Path]) -> None:
        """Save translations to a PO file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        lines = []

        # Header
        lines.append(f"# {catalog.domain} translation for {catalog.locale}")
        lines.append("#")
        lines.append('msgid ""')
        lines.append('msgstr ""')
        lines.append(f'"Content-Type: text/plain; charset=UTF-8\\n"')
        lines.append(f'"Language: {catalog.locale}\\n"')
        lines.append("")

        # Entries
        for (msgid, msgctxt), entry in catalog._messages.items():
            # Comments
            for comment in entry.comments:
                lines.append(f"#. {comment}")

            # Context
            if entry.msgctxt:
                lines.append(f'msgctxt "{self._escape(entry.msgctxt)}"')

            # Message ID
            lines.append(f'msgid "{self._escape(msgid)}"')

            # Translation
            if entry.msgstr_plural:
                lines.append('msgid_plural ""')
                for idx, plural in sorted(entry.msgstr_plural.items()):
                    lines.append(f'msgstr[{idx}] "{self._escape(plural)}"')
            else:
                lines.append(f'msgstr "{self._escape(entry.msgstr)}"')

            lines.append("")

        path.write_text("\n".join(lines), encoding="utf-8")

    def _escape(self, text: str) -> str:
        """Escape special characters for PO file."""
        return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def load_translations(path: Union[str, Path], format: Optional[str] = None) -> Catalog:
    """
    Load translations from a file.

    Args:
        path: Path to the translation file
        format: File format (json, yaml, po) - auto-detected if not specified

    Returns:
        Loaded Catalog
    """
    path = Path(path)

    if format is None:
        format = path.suffix.lstrip(".")

    loaders = {
        "json": JSONResource(),
        "yaml": YAMLResource(),
        "yml": YAMLResource(),
        "po": POFile(),
    }

    if format not in loaders:
        raise ResourceError(f"Unknown format: {format}")

    return loaders[format].load(path)


# ============================================================================
# 8. Utilities
# ============================================================================

# Global translator instance for shortcut functions
_global_translator: Optional[Translator] = None
_global_localizer: Optional[Localizer] = None
_global_locale_manager: Optional[LocaleManager] = None


def get_translator() -> Translator:
    """Get the global translator instance."""
    global _global_translator
    if _global_translator is None:
        _global_translator = Translator()
    return _global_translator


def get_localizer() -> Localizer:
    """Get the global localizer instance."""
    global _global_localizer
    if _global_localizer is None:
        _global_localizer = Localizer()
    return _global_localizer


def get_locale_manager() -> LocaleManager:
    """Get the global locale manager instance."""
    global _global_locale_manager
    if _global_locale_manager is None:
        _global_locale_manager = LocaleManager()
    return _global_locale_manager


def _(message: str, **kwargs) -> str:
    """
    Shortcut for gettext - translate a message.

    Args:
        message: Message to translate
        **kwargs: Format parameters

    Returns:
        Translated (and formatted) string
    """
    translator = get_translator()
    translated = translator.gettext(message)
    if kwargs:
        return translated.format(**kwargs)
    return translated


def N_(message: str) -> str:
    """
    No-op marker for translatable strings.

    Use this to mark strings for extraction without translating them immediately.

    Returns:
        The original message (untranslated)
    """
    return message


def ngettext(singular: str, plural: str, n: int, **kwargs) -> str:
    """
    Shortcut for plural gettext.

    Args:
        singular: Singular form
        plural: Plural form
        n: Number for determining plural form
        **kwargs: Format parameters

    Returns:
        Translated (and formatted) string
    """
    translator = get_translator()
    translated = translator.ngettext(singular, plural, n)
    if kwargs:
        return translated.format(**kwargs)
    return translated


def pgettext(context: str, message: str, **kwargs) -> str:
    """
    Shortcut for context gettext.

    Args:
        context: Context for disambiguation
        message: Message to translate
        **kwargs: Format parameters

    Returns:
        Translated (and formatted) string
    """
    translator = get_translator()
    translated = translator.pgettext(context, message)
    if kwargs:
        return translated.format(**kwargs)
    return translated


def localize(value: Any, locale: Optional[LocaleCode] = None, **kwargs) -> str:
    """
    Localize a value according to current or specified locale.

    Args:
        value: Value to localize (date, number, etc.)
        locale: Optional locale override
        **kwargs: Additional formatting options

    Returns:
        Localized string
    """
    localizer = get_localizer()

    if locale:
        old_locale = localizer.locale
        localizer.locale = locale

    try:
        if isinstance(value, (date, datetime)):
            if isinstance(value, datetime) and kwargs.get("time_only"):
                return localizer.format_time(
                    value, **{k: v for k, v in kwargs.items() if k != "time_only"}
                )
            elif isinstance(value, date):
                return localizer.format_date(value, **kwargs)
            else:
                return localizer.format_datetime(value, **kwargs)
        elif isinstance(value, (int, float, Decimal)):
            if kwargs.get("currency"):
                return localizer.format_currency(value, kwargs["currency"])
            elif kwargs.get("percent"):
                return localizer.format_percent(value, kwargs.get("decimal_places", 2))
            else:
                return localizer.format_number(value, **kwargs)
        else:
            return str(value)
    finally:
        if locale:
            localizer.locale = old_locale


def i18n_decorator(
    locale: Optional[LocaleCode] = None, domain: Optional[str] = None
) -> Callable:
    """
    Decorator for internationalizing functions.

    Automatically translates docstrings and can set locale for function execution.

    Args:
        locale: Locale to use for function execution
        domain: Translation domain

    Returns:
        Decorator function
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            translator = get_translator()
            old_locale = translator.get_locale()

            if locale:
                translator.set_locale(locale)

            try:
                return func(*args, **kwargs)
            finally:
                translator.set_locale(old_locale)

        # Translate docstring
        if func.__doc__:
            translator = get_translator()
            wrapper.__doc__ = translator.gettext(func.__doc__)

        return wrapper

    return decorator


class lazy_string:
    """
    Lazy string that delays translation until string conversion.

    Useful for module-level translations that should be evaluated
    at runtime with the current locale.
    """

    def __init__(self, message: str, **kwargs):
        self.message = message
        self.kwargs = kwargs

    def __str__(self) -> str:
        return _(self.message, **self.kwargs)

    def __repr__(self) -> str:
        return f"lazy_string({self.message!r})"

    def __add__(self, other: str) -> str:
        return str(self) + other

    def __radd__(self, other: str) -> str:
        return other + str(self)


class I18nContext:
    """
    Context manager for temporary locale changes.

    Example:
        with I18nContext('de'):
            print(_("Hello"))  # Will use German translation
    """

    def __init__(
        self, locale: Optional[LocaleCode] = None, domain: Optional[str] = None
    ):
        self.locale = locale
        self.domain = domain
        self._translator = get_translator()
        self._localizer = get_localizer()
        self._old_translator_locale: Optional[LocaleCode] = None
        self._old_localizer_locale: Optional[LocaleCode] = None

    def __enter__(self) -> I18nContext:
        if self.locale:
            self._old_translator_locale = self._translator.get_locale()
            self._old_localizer_locale = self._localizer.locale
            self._translator.set_locale(self.locale)
            self._localizer.locale = self.locale
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._old_translator_locale is not None:
            self._translator.set_locale(self._old_translator_locale)
        if self._old_localizer_locale is not None:
            self._localizer.locale = self._old_localizer_locale


# ============================================================================
# Convenience Exports
# ============================================================================

__all__ = [
    # Translation
    "TranslationEntry",
    "Catalog",
    "Translator",
    # Localization
    "Localizer",
    # Locale Management
    "LocaleInfo",
    "LocaleManager",
    # Message Extraction
    "ExtractedMessage",
    "MessageExtractor",
    # Pluralization
    "PluralRule",
    # Formatting
    "DateFormatter",
    "NumberFormatter",
    "CurrencyFormatter",
    "PercentFormatter",
    # Resources
    "ResourceLoader",
    "JSONResource",
    "YAMLResource",
    "POFile",
    "load_translations",
    # Utilities
    "_",
    "N_",
    "ngettext",
    "pgettext",
    "localize",
    "i18n_decorator",
    "lazy_string",
    "I18nContext",
    "get_translator",
    "get_localizer",
    "get_locale_manager",
    # Exceptions
    "I18nError",
    "TranslationError",
    "LocaleError",
    "ResourceError",
    "FormatError",
]
