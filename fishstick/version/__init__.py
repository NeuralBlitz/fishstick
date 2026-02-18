"""Fishstick Version Module

Comprehensive version management, comparison, compatibility checking,
changelog parsing, upgrade handling, and deprecation utilities.
"""

from .core import (
    # Version Info
    __version__,
    VERSION_INFO,
    get_version,
    parse_version,
    # Version Comparison
    compare_versions,
    Version,
    VersionInfo,
    check_version,
    require_version,
    # Compatibility
    check_compatibility,
    is_compatible,
    CompatibilityLevel,
    get_min_version,
    get_max_version,
    # Changelog
    get_changelog,
    parse_changelog,
    get_changes,
    get_latest_changes,
    Changelog,
    ChangeEntry,
    # Upgrade
    check_for_updates,
    upgrade_available,
    get_upgrade_instructions,
    migrate_version,
    UpdateInfo,
    # Deprecation
    deprecated,
    warn_deprecated,
    FishstickDeprecationWarning,
    DeprecationTracker,
    # Release
    is_release,
    is_dev,
    is_alpha,
    is_beta,
    is_rc,
    # Utilities
    print_version,
    version_info,
)

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
