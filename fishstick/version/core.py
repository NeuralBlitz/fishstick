"""
Fishstick Version Module

Comprehensive version management, comparison, compatibility checking,
changelog parsing, upgrade handling, and deprecation utilities.
"""

from __future__ import annotations

import functools
import json
import re
import sys
import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


# =============================================================================
# Version Info
# =============================================================================

__version__ = "1.0.0"
VERSION_INFO = (1, 0, 0, "final", 0)


def get_version() -> str:
    """Get the current version string."""
    return __version__


def parse_version(version: str) -> VersionInfo:
    """
    Parse a version string into a VersionInfo object.

    Args:
        version: Version string (e.g., "1.0.0", "2.0.0a1", "3.0.0b2")

    Returns:
        VersionInfo object containing parsed version components

    Raises:
        ValueError: If version string is invalid
    """
    return VersionInfo.from_string(version)


# =============================================================================
# Version Comparison
# =============================================================================


class Version:
    """
    Semantic version class with comparison support.

    Supports standard semantic versioning (MAJOR.MINOR.PATCH) with
    pre-release identifiers (alpha, beta, rc) and build metadata.
    """

    def __init__(
        self,
        major: int,
        minor: int = 0,
        patch: int = 0,
        prerelease: Optional[str] = None,
        build: Optional[str] = None,
    ):
        self.major = major
        self.minor = minor
        self.patch = patch
        self.prerelease = prerelease
        self.build = build

    @classmethod
    def from_string(cls, version: str) -> Version:
        """Parse version string into Version object."""
        pattern = r"^(\d+)(?:\.(\d+))?(?:\.(\d+))?(?:-?([a-zA-Z0-9.]+))?(?:\+([a-zA-Z0-9.]+))?$"
        match = re.match(pattern, version.strip())

        if not match:
            raise ValueError(f"Invalid version string: {version}")

        major = int(match.group(1))
        minor = int(match.group(2)) if match.group(2) else 0
        patch = int(match.group(3)) if match.group(3) else 0
        prerelease = match.group(4)
        build = match.group(5)

        return cls(major, minor, patch, prerelease, build)

    def to_tuple(self) -> Tuple[int, int, int, Optional[str], Optional[str]]:
        """Convert version to tuple for comparison."""
        return (self.major, self.minor, self.patch, self.prerelease, self.build)

    def __str__(self) -> str:
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version += f"-{self.prerelease}"
        if self.build:
            version += f"+{self.build}"
        return version

    def __repr__(self) -> str:
        return f"Version({self.major}, {self.minor}, {self.patch}, {self.prerelease!r}, {self.build!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Version):
            return NotImplemented
        return self.to_tuple()[:4] == other.to_tuple()[:4]

    def __lt__(self, other: Version) -> bool:
        if not isinstance(other, Version):
            return NotImplemented

        # Compare major, minor, patch
        for s, o in zip(
            (self.major, self.minor, self.patch),
            (other.major, other.minor, other.patch),
        ):
            if s != o:
                return s < o

        # Handle prerelease comparison
        self_pre = self.prerelease or ""
        other_pre = other.prerelease or ""

        if self_pre and not other_pre:
            return True  # prerelease < release
        if not self_pre and other_pre:
            return False  # release > prerelease

        return self_pre < other_pre

    def __le__(self, other: Version) -> bool:
        return self == other or self < other

    def __gt__(self, other: Version) -> bool:
        return not self <= other

    def __ge__(self, other: Version) -> bool:
        return not self < other

    def __hash__(self) -> int:
        return hash(self.to_tuple()[:4])

    def is_prerelease(self) -> bool:
        """Check if this is a pre-release version."""
        return self.prerelease is not None

    def is_stable(self) -> bool:
        """Check if this is a stable release (not pre-release)."""
        return self.prerelease is None


@dataclass
class VersionInfo:
    """Detailed version information container."""

    major: int
    minor: int
    micro: int = 0
    releaselevel: str = "final"
    serial: int = 0

    @classmethod
    def from_string(cls, version: str) -> VersionInfo:
        """Parse version string into VersionInfo."""
        pattern = r"^(\d+)(?:\.(\d+))?(?:\.(\d+))?([a-z]+)?(\d*)$"
        match = re.match(pattern, version.strip())

        if not match:
            raise ValueError(f"Invalid version string: {version}")

        major = int(match.group(1))
        minor = int(match.group(2)) if match.group(2) else 0
        micro = int(match.group(3)) if match.group(3) else 0
        releaselevel = match.group(4) or "final"
        serial = int(match.group(5)) if match.group(5) else 0

        return cls(major, minor, micro, releaselevel, serial)

    def to_version(self) -> Version:
        """Convert to Version object."""
        prerelease = None
        if self.releaselevel != "final":
            prerelease = (
                f"{self.releaselevel}{self.serial}"
                if self.serial
                else self.releaselevel
            )
        return Version(self.major, self.minor, self.micro, prerelease)

    def __str__(self) -> str:
        version = f"{self.major}.{self.minor}.{self.micro}"
        if self.releaselevel != "final":
            version += self.releaselevel
            if self.serial:
                version += str(self.serial)
        return version


def compare_versions(
    version1: Union[str, Version], version2: Union[str, Version]
) -> int:
    """
    Compare two versions.

    Args:
        version1: First version
        version2: Second version

    Returns:
        -1 if version1 < version2, 0 if equal, 1 if version1 > version2
    """
    v1 = version1 if isinstance(version1, Version) else Version.from_string(version1)
    v2 = version2 if isinstance(version2, Version) else Version.from_string(version2)

    if v1 < v2:
        return -1
    elif v1 > v2:
        return 1
    else:
        return 0


def check_version(
    current: Union[str, Version], required: Union[str, Version], comparison: str = ">="
) -> bool:
    """
    Check if current version satisfies the requirement.

    Args:
        current: Current version
        required: Required version
        comparison: Comparison operator (==, !=, <, <=, >, >=)

    Returns:
        True if requirement is satisfied
    """
    v_current = (
        current if isinstance(current, Version) else Version.from_string(current)
    )
    v_required = (
        required if isinstance(required, Version) else Version.from_string(required)
    )

    result = compare_versions(v_current, v_required)

    if comparison == "==":
        return result == 0
    elif comparison == "!=":
        return result != 0
    elif comparison == "<":
        return result < 0
    elif comparison == "<=":
        return result <= 0
    elif comparison == ">":
        return result > 0
    elif comparison == ">=":
        return result >= 0
    else:
        raise ValueError(f"Invalid comparison operator: {comparison}")


def require_version(
    current: Union[str, Version],
    required: Union[str, Version],
    comparison: str = ">=",
    package: str = "fishstick",
) -> None:
    """
    Require a specific version, raise error if not satisfied.

    Args:
        current: Current version
        required: Required version
        comparison: Comparison operator
        package: Package name for error message

    Raises:
        RuntimeError: If version requirement is not satisfied
    """
    if not check_version(current, required, comparison):
        raise RuntimeError(
            f"{package} version {comparison} {required} is required, "
            f"but {current} is installed"
        )


# =============================================================================
# Compatibility
# =============================================================================


class CompatibilityLevel(Enum):
    """Compatibility levels between versions."""

    FULL = "full"  # Fully compatible
    BACKWARD = "backward"  # Backward compatible
    PARTIAL = "partial"  # Partially compatible
    BREAKING = "breaking"  # Breaking changes
    UNKNOWN = "unknown"  # Unknown compatibility


def check_compatibility(
    current: Union[str, Version], target: Union[str, Version]
) -> CompatibilityLevel:
    """
    Check compatibility between two versions.

    Args:
        current: Current version
        target: Target version to check against

    Returns:
        CompatibilityLevel indicating the level of compatibility
    """
    v_current = (
        current if isinstance(current, Version) else Version.from_string(current)
    )
    v_target = target if isinstance(target, Version) else Version.from_string(target)

    # Same version - fully compatible
    if v_current == v_target:
        return CompatibilityLevel.FULL

    # Major version change - breaking
    if v_current.major != v_target.major:
        return CompatibilityLevel.BREAKING

    # Downgrade - backward compatible (can load old data)
    if v_target < v_current:
        return CompatibilityLevel.BACKWARD

    # Same major, different minor - partially compatible
    if v_current.minor != v_target.minor:
        return CompatibilityLevel.PARTIAL

    return CompatibilityLevel.FULL


def is_compatible(
    current: Union[str, Version], target: Union[str, Version], strict: bool = False
) -> bool:
    """
    Check if versions are compatible.

    Args:
        current: Current version
        target: Target version
        strict: If True, only FULL compatibility is acceptable

    Returns:
        True if versions are compatible
    """
    level = check_compatibility(current, target)

    if strict:
        return level == CompatibilityLevel.FULL

    return level in (CompatibilityLevel.FULL, CompatibilityLevel.BACKWARD)


def get_min_version(versions: List[Union[str, Version]]) -> Version:
    """
    Get the minimum version from a list.

    Args:
        versions: List of versions

    Returns:
        Minimum version
    """
    parsed = [v if isinstance(v, Version) else Version.from_string(v) for v in versions]
    return min(parsed)


def get_max_version(versions: List[Union[str, Version]]) -> Version:
    """
    Get the maximum version from a list.

    Args:
        versions: List of versions

    Returns:
        Maximum version
    """
    parsed = [v if isinstance(v, Version) else Version.from_string(v) for v in versions]
    return max(parsed)


# =============================================================================
# Changelog
# =============================================================================


@dataclass
class ChangeEntry:
    """Single changelog entry."""

    version: str
    date: Optional[str]
    changes: List[str]
    category: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "date": self.date,
            "changes": self.changes,
            "category": self.category,
        }


class Changelog:
    """Changelog parser and container."""

    def __init__(self, entries: Optional[List[ChangeEntry]] = None):
        self.entries = entries or []

    @classmethod
    def parse(cls, content: str, format: str = "auto") -> Changelog:
        """
        Parse changelog content.

        Args:
            content: Changelog content
            format: Format type ("auto", "markdown", "rst", "json")

        Returns:
            Parsed Changelog object
        """
        if format == "auto":
            format = cls._detect_format(content)

        if format == "json":
            return cls._parse_json(content)
        elif format == "markdown":
            return cls._parse_markdown(content)
        elif format == "rst":
            return cls._parse_rst(content)
        else:
            return cls._parse_plain(content)

    @classmethod
    def from_file(cls, filepath: Union[str, Path], format: str = "auto") -> Changelog:
        """Load and parse changelog from file."""
        with open(filepath, "r") as f:
            content = f.read()
        return cls.parse(content, format)

    @staticmethod
    def _detect_format(content: str) -> str:
        """Detect changelog format."""
        content_lower = content.lower()
        if content.strip().startswith("{"):
            return "json"
        elif ".. version::" in content or ".. changelog::" in content:
            return "rst"
        elif "## " in content or content_lower.startswith("# changelog"):
            return "markdown"
        return "plain"

    @classmethod
    def _parse_json(cls, content: str) -> Changelog:
        """Parse JSON changelog."""
        data = json.loads(content)
        entries = []

        if isinstance(data, list):
            for item in data:
                entry = ChangeEntry(
                    version=item.get("version", ""),
                    date=item.get("date"),
                    changes=item.get("changes", []),
                    category=item.get("category"),
                )
                entries.append(entry)
        elif isinstance(data, dict):
            for version, info in data.items():
                entry = ChangeEntry(
                    version=version,
                    date=info.get("date") if isinstance(info, dict) else None,
                    changes=info.get("changes", [])
                    if isinstance(info, dict)
                    else [str(info)],
                )
                entries.append(entry)

        return cls(entries)

    @classmethod
    def _parse_markdown(cls, content: str) -> Changelog:
        """Parse Markdown changelog."""
        entries = []
        lines = content.split("\n")
        current_entry = None

        for line in lines:
            line = line.strip()

            # Match version headers (## [1.0.0] - 2024-01-01 or ## 1.0.0)
            match = re.match(
                r"^##\s+\[?([^\]]+)\]?\s*(?:-\s*(\d{4}-\d{2}-\d{2}))?", line
            )
            if match:
                if current_entry:
                    entries.append(current_entry)
                current_entry = ChangeEntry(
                    version=match.group(1), date=match.group(2), changes=[]
                )
            elif current_entry and line.startswith("- "):
                current_entry.changes.append(line[2:].strip())
            elif current_entry and line.startswith("* "):
                current_entry.changes.append(line[2:].strip())

        if current_entry:
            entries.append(current_entry)

        return cls(entries)

    @classmethod
    def _parse_rst(cls, content: str) -> Changelog:
        """Parse reStructuredText changelog."""
        entries = []
        lines = content.split("\n")
        current_entry = None

        for line in lines:
            line = line.strip()

            # Match version directives
            match = re.match(r"^..\s+version::\s+(.+)", line)
            if match:
                if current_entry:
                    entries.append(current_entry)
                current_entry = ChangeEntry(
                    version=match.group(1), date=None, changes=[]
                )
            elif current_entry and line.startswith("- "):
                current_entry.changes.append(line[2:].strip())

        if current_entry:
            entries.append(current_entry)

        return cls(entries)

    @classmethod
    def _parse_plain(cls, content: str) -> Changelog:
        """Parse plain text changelog."""
        entries = []
        lines = content.split("\n")
        current_entry = None

        for line in lines:
            line = line.strip()

            # Match version patterns
            match = re.match(
                r"^(?:version\s+)?(\d+\.\d+(?:\.\d+)?[a-z0-9]*)", line, re.I
            )
            if match:
                if current_entry:
                    entries.append(current_entry)
                current_entry = ChangeEntry(
                    version=match.group(1), date=None, changes=[]
                )
            elif current_entry and line.startswith(("- ", "* ", "+ ")):
                current_entry.changes.append(line[2:].strip())

        if current_entry:
            entries.append(current_entry)

        return cls(entries)

    def get_changes(
        self, from_version: Optional[str] = None, to_version: Optional[str] = None
    ) -> List[ChangeEntry]:
        """
        Get changes between versions.

        Args:
            from_version: Start version (inclusive)
            to_version: End version (inclusive)

        Returns:
            List of change entries
        """
        filtered = self.entries

        if from_version:
            v_from = Version.from_string(from_version)
            filtered = [e for e in filtered if Version.from_string(e.version) >= v_from]

        if to_version:
            v_to = Version.from_string(to_version)
            filtered = [e for e in filtered if Version.from_string(e.version) <= v_to]

        return filtered

    def get_latest_changes(self, count: int = 1) -> List[ChangeEntry]:
        """
        Get latest N changes.

        Args:
            count: Number of entries to return

        Returns:
            List of latest change entries
        """
        sorted_entries = sorted(
            self.entries, key=lambda e: Version.from_string(e.version), reverse=True
        )
        return sorted_entries[:count]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {"entries": [e.to_dict() for e in self.entries]}

    def __iter__(self):
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)


def get_changelog(filepath: Optional[Union[str, Path]] = None) -> Changelog:
    """
    Get changelog from file or default location.

    Args:
        filepath: Path to changelog file (auto-detected if None)

    Returns:
        Changelog object
    """
    if filepath is None:
        # Try common changelog locations
        possible_paths = [
            "CHANGELOG.md",
            "CHANGELOG.rst",
            "CHANGELOG.txt",
            "CHANGELOG",
            "HISTORY.md",
            "HISTORY.rst",
            "CHANGES.md",
        ]

        for path in possible_paths:
            if Path(path).exists():
                filepath = path
                break

        if filepath is None:
            raise FileNotFoundError("No changelog file found in default locations")

    return Changelog.from_file(filepath)


def parse_changelog(content: str, format: str = "auto") -> Changelog:
    """
    Parse changelog content.

    Args:
        content: Changelog content string
        format: Format type ("auto", "markdown", "rst", "json")

    Returns:
        Parsed Changelog object
    """
    return Changelog.parse(content, format)


def get_changes(
    changelog: Union[Changelog, str, Path],
    from_version: Optional[str] = None,
    to_version: Optional[str] = None,
) -> List[ChangeEntry]:
    """
    Get changes from changelog.

    Args:
        changelog: Changelog object or path to file
        from_version: Start version
        to_version: End version

    Returns:
        List of change entries
    """
    if isinstance(changelog, (str, Path)):
        changelog = Changelog.from_file(changelog)

    return changelog.get_changes(from_version, to_version)


def get_latest_changes(
    changelog: Union[Changelog, str, Path], count: int = 1
) -> List[ChangeEntry]:
    """
    Get latest changes from changelog.

    Args:
        changelog: Changelog object or path to file
        count: Number of entries to return

    Returns:
        List of latest change entries
    """
    if isinstance(changelog, (str, Path)):
        changelog = Changelog.from_file(changelog)

    return changelog.get_latest_changes(count)


# =============================================================================
# Upgrade
# =============================================================================


class UpdateInfo:
    """Information about available updates."""

    def __init__(
        self,
        current: str,
        latest: str,
        available: bool,
        url: Optional[str] = None,
        changelog: Optional[str] = None,
        upgrade_type: str = "patch",
    ):
        self.current = current
        self.latest = latest
        self.available = available
        self.url = url
        self.changelog = changelog
        self.upgrade_type = upgrade_type

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current": self.current,
            "latest": self.latest,
            "available": self.available,
            "url": self.url,
            "changelog": self.changelog,
            "upgrade_type": self.upgrade_type,
        }


def check_for_updates(
    current_version: Optional[str] = None,
    package_name: str = "fishstick",
    index_url: Optional[str] = None,
) -> UpdateInfo:
    """
    Check for available updates.

    Args:
        current_version: Current version (uses __version__ if None)
        package_name: Package name to check
        index_url: Package index URL

    Returns:
        UpdateInfo object with update information

    Note:
        This is a placeholder implementation. In production, this would
        query PyPI or another package index.
    """
    if current_version is None:
        current_version = __version__

    # Placeholder: In real implementation, query PyPI API
    # For now, return dummy data
    latest = "1.1.0"  # Would come from API

    v_current = Version.from_string(current_version)
    v_latest = Version.from_string(latest)

    available = v_latest > v_current

    # Determine upgrade type
    if v_current.major != v_latest.major:
        upgrade_type = "major"
    elif v_current.minor != v_latest.minor:
        upgrade_type = "minor"
    else:
        upgrade_type = "patch"

    return UpdateInfo(
        current=current_version,
        latest=latest,
        available=available,
        url=index_url or f"https://pypi.org/project/{package_name}/",
        changelog=None,
        upgrade_type=upgrade_type,
    )


def upgrade_available(
    current_version: Optional[str] = None, package_name: str = "fishstick"
) -> bool:
    """
    Check if an upgrade is available.

    Args:
        current_version: Current version
        package_name: Package name

    Returns:
        True if upgrade is available
    """
    info = check_for_updates(current_version, package_name)
    return info.available


def get_upgrade_instructions(
    current: Optional[str] = None,
    target: Optional[str] = None,
    package_name: str = "fishstick",
) -> str:
    """
    Get upgrade instructions.

    Args:
        current: Current version
        target: Target version (latest if None)
        package_name: Package name

    Returns:
        Upgrade instructions string
    """
    if current is None:
        current = __version__

    update_info = check_for_updates(current, package_name)
    target = target or update_info.latest

    instructions = f"""
Upgrade Instructions for {package_name}
{"=" * 50}

Current version: {current}
Target version: {target}

To upgrade, run one of the following commands:

Using pip:
  pip install --upgrade {package_name}=={target}

Using pip with specific index:
  pip install --upgrade --index-url {update_info.url or "https://pypi.org/simple/"} {package_name}=={target}

Using conda (if available):
  conda update {package_name}

Using poetry:
  poetry add {package_name}@{target}

Post-upgrade steps:
1. Review the changelog for breaking changes
2. Run your test suite
3. Check for deprecated features
4. Update any configuration files if needed

For more information, visit:
{update_info.url or f"https://pypi.org/project/{package_name}/"}
"""

    return instructions.strip()


def migrate_version(
    from_version: str, to_version: str, config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Migrate configuration/data between versions.

    Args:
        from_version: Source version
        to_version: Target version
        config: Configuration to migrate

    Returns:
        Migrated configuration

    Raises:
        ValueError: If migration is not possible
    """
    config = config or {}
    v_from = Version.from_string(from_version)
    v_to = Version.from_string(to_version)

    if v_to < v_from:
        raise ValueError("Cannot migrate to an older version")

    migrated = config.copy()

    # Major version migrations
    if v_from.major < v_to.major:
        # Example migrations - customize for actual breaking changes
        if "old_setting" in migrated:
            migrated["new_setting"] = migrated.pop("old_setting")

    # Minor version migrations
    if v_from.minor < v_to.minor:
        # Add any minor version migrations here
        pass

    migrated["_migrated_from"] = from_version
    migrated["_migrated_to"] = to_version

    return migrated


# =============================================================================
# Deprecation
# =============================================================================


class FishstickDeprecationWarning(DeprecationWarning):
    """Custom deprecation warning for fishstick."""

    pass


class DeprecationTracker:
    """Track deprecated features."""

    def __init__(self):
        self._deprecated: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        name: str,
        version: str,
        removal_version: Optional[str] = None,
        alternative: Optional[str] = None,
        message: Optional[str] = None,
    ) -> None:
        """Register a deprecated feature."""
        self._deprecated[name] = {
            "version": version,
            "removal_version": removal_version,
            "alternative": alternative,
            "message": message,
        }

    def check(self, name: str) -> Optional[Dict[str, Any]]:
        """Check if a feature is deprecated."""
        return self._deprecated.get(name)

    def is_deprecated(self, name: str) -> bool:
        """Check if a feature name is deprecated."""
        return name in self._deprecated


# Global deprecation tracker
_deprecation_tracker = DeprecationTracker()


def warn_deprecated(
    name: str,
    version: Optional[str] = None,
    removal_version: Optional[str] = None,
    alternative: Optional[str] = None,
    message: Optional[str] = None,
    stacklevel: int = 2,
) -> None:
    """
    Emit a deprecation warning.

    Args:
        name: Name of deprecated feature
        version: Version when deprecated
        removal_version: Version when it will be removed
        alternative: Alternative to use instead
        message: Custom message
        stacklevel: Stack level for warning
    """
    if message is None:
        message = f"{name} is deprecated"
        if version:
            message += f" since version {version}"
        if removal_version:
            message += f" and will be removed in version {removal_version}"
        if alternative:
            message += f". Use {alternative} instead"
        message += "."

    warnings.warn(message, FishstickDeprecationWarning, stacklevel=stacklevel)

    # Register in tracker
    _deprecation_tracker.register(
        name, version or __version__, removal_version, alternative, message
    )


def deprecated(
    version: Optional[str] = None,
    removal_version: Optional[str] = None,
    alternative: Optional[str] = None,
    message: Optional[str] = None,
) -> Callable:
    """
    Decorator to mark functions/classes as deprecated.

    Args:
        version: Version when deprecated
        removal_version: Version when it will be removed
        alternative: Alternative to use instead
        message: Custom message

    Returns:
        Decorator function

    Example:
        @deprecated(version="1.0.0", alternative="new_function")
        def old_function():
            pass
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warn_deprecated(
                name=func.__name__,
                version=version,
                removal_version=removal_version,
                alternative=alternative,
                message=message,
                stacklevel=3,
            )
            return func(*args, **kwargs)

        # Mark the wrapper
        wrapper._deprecated = True  # type: ignore
        wrapper._deprecated_version = version  # type: ignore
        wrapper._removal_version = removal_version  # type: ignore

        return wrapper

    return decorator


# =============================================================================
# Release
# =============================================================================


def is_release(version: Optional[str] = None) -> bool:
    """
    Check if version is a release (not dev).

    Args:
        version: Version string (uses __version__ if None)

    Returns:
        True if it's a release version
    """
    if version is None:
        version = __version__

    v = Version.from_string(version)

    # Check for dev markers in prerelease
    if v.prerelease:
        return "dev" not in v.prerelease.lower()

    return True


def is_dev(version: Optional[str] = None) -> bool:
    """
    Check if version is a development version.

    Args:
        version: Version string

    Returns:
        True if it's a dev version
    """
    if version is None:
        version = __version__

    return not is_release(version)


def is_alpha(version: Optional[str] = None) -> bool:
    """
    Check if version is an alpha release.

    Args:
        version: Version string

    Returns:
        True if it's an alpha version
    """
    if version is None:
        version = __version__

    v = Version.from_string(version)

    if v.prerelease:
        return v.prerelease.lower().startswith("a") or "alpha" in v.prerelease.lower()

    return False


def is_beta(version: Optional[str] = None) -> bool:
    """
    Check if version is a beta release.

    Args:
        version: Version string

    Returns:
        True if it's a beta version
    """
    if version is None:
        version = __version__

    v = Version.from_string(version)

    if v.prerelease:
        return v.prerelease.lower().startswith("b") or "beta" in v.prerelease.lower()

    return False


def is_rc(version: Optional[str] = None) -> bool:
    """
    Check if version is a release candidate.

    Args:
        version: Version string

    Returns:
        True if it's an RC version
    """
    if version is None:
        version = __version__

    v = Version.from_string(version)

    if v.prerelease:
        prerelease_lower = v.prerelease.lower()
        return prerelease_lower.startswith("rc") or "rc" in prerelease_lower

    return False


# =============================================================================
# Utilities
# =============================================================================


def print_version(
    verbose: bool = False, output: Optional[Callable[[str], None]] = None
) -> None:
    """
    Print version information.

    Args:
        verbose: Print detailed information
        output: Output function (defaults to print)
    """
    output = output or print

    output(f"fishstick version {__version__}")

    if verbose:
        output(f"Version info: {VERSION_INFO}")
        output(f"Release: {'Yes' if is_release() else 'No'}")
        output(f"Development: {'Yes' if is_dev() else 'No'}")
        output(f"Python: {sys.version}")
        output(f"Platform: {sys.platform}")


def version_info() -> Dict[str, Any]:
    """
    Get comprehensive version information.

    Returns:
        Dictionary with version information
    """
    return {
        "version": __version__,
        "version_info": VERSION_INFO,
        "version_tuple": {
            "major": VERSION_INFO[0],
            "minor": VERSION_INFO[1],
            "micro": VERSION_INFO[2],
            "releaselevel": VERSION_INFO[3],
            "serial": VERSION_INFO[4],
        },
        "is_release": is_release(),
        "is_dev": is_dev(),
        "is_alpha": is_alpha(),
        "is_beta": is_beta(),
        "is_rc": is_rc(),
        "python_version": sys.version,
        "python_version_info": sys.version_info[:5],
        "platform": sys.platform,
    }


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Version Info
    "__version__",
    "VERSION_INFO",
    "get_version",
    "parse_version",
    # Version Comparison
    "compare_versions",
    "Version",
    "VersionInfo",
    "check_version",
    "require_version",
    # Compatibility
    "check_compatibility",
    "is_compatible",
    "CompatibilityLevel",
    "get_min_version",
    "get_max_version",
    # Changelog
    "get_changelog",
    "parse_changelog",
    "get_changes",
    "get_latest_changes",
    "Changelog",
    "ChangeEntry",
    # Upgrade
    "check_for_updates",
    "upgrade_available",
    "get_upgrade_instructions",
    "migrate_version",
    "UpdateInfo",
    # Deprecation
    "deprecated",
    "warn_deprecated",
    "FishstickDeprecationWarning",
    "DeprecationTracker",
    # Release
    "is_release",
    "is_dev",
    "is_alpha",
    "is_beta",
    "is_rc",
    # Utilities
    "print_version",
    "version_info",
]
