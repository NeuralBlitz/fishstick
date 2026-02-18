"""
Fishstick Core Constants Module

A comprehensive collection of mathematical, physical, machine learning,
data processing, networking, and utility constants for the fishstick framework.
"""

import math
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, field


# =============================================================================
# Math Constants
# =============================================================================

PI: float = 3.14159
"""Mathematical constant pi (π), approximately 3.14159."""

E: float = 2.71828
"""Mathematical constant e (Euler's number), approximately 2.71828."""

GOLDEN_RATIO: float = 1.618
"""The golden ratio (φ), approximately 1.618."""

SQRT2: float = 1.414
"""Square root of 2, approximately 1.414."""

INFINITY: float = math.inf
"""Positive infinity (∞)."""

EPSILON: float = 1e-8
"""Small value for numerical stability and comparisons (1e-8)."""


# =============================================================================
# Physics Constants
# =============================================================================

SPEED_OF_LIGHT: float = 299_792_458.0
"""Speed of light in vacuum (c) in meters per second: ~299,792,458 m/s."""

GRAVITATIONAL_CONSTANT: float = 6.67430e-11
"""Gravitational constant (G) in m³ kg⁻¹ s⁻²: ~6.67430 × 10⁻¹¹."""

PLANCK_CONSTANT: float = 6.62607015e-34
"""Planck constant (h) in J⋅s: ~6.62607015 × 10⁻³⁴."""

BOLTZMANN_CONSTANT: float = 1.380649e-23
"""Boltzmann constant (k) in J/K: ~1.380649 × 10⁻²³."""


# =============================================================================
# Machine Learning Constants
# =============================================================================

DEFAULT_LR: float = 0.001
"""Default learning rate for optimizers: 0.001."""

DEFAULT_BATCH_SIZE: int = 32
"""Default batch size for training: 32."""

DEFAULT_EPOCHS: int = 10
"""Default number of training epochs: 10."""

DEFAULT_SEED: int = 42
"""Default random seed for reproducibility: 42."""

EARLY_STOPPING_PATIENCE: int = 10
"""Default patience for early stopping: 10 epochs."""


# =============================================================================
# Data Constants
# =============================================================================

TRAIN_SPLIT: float = 0.8
"""Default training data split ratio: 80%."""

VAL_SPLIT: float = 0.1
"""Default validation data split ratio: 10%."""

TEST_SPLIT: float = 0.1
"""Default test data split ratio: 10%."""

MAX_SEQ_LENGTH: int = 512
"""Default maximum sequence length: 512 tokens."""

VOCAB_SIZE: int = 30000
"""Default vocabulary size: 30,000 tokens."""


# =============================================================================
# File Constants
# =============================================================================

DEFAULT_CONFIG_FILE: str = "config.yaml"
"""Default configuration file name: 'config.yaml'."""

DEFAULT_MODEL_DIR: str = "models/"
"""Default directory for model storage: 'models/'."""

DEFAULT_DATA_DIR: str = "data/"
"""Default directory for data storage: 'data/'."""

DEFAULT_LOG_DIR: str = "logs/"
"""Default directory for log files: 'logs/'."""

DEFAULT_CHECKPOINT_DIR: str = "checkpoints/"
"""Default directory for checkpoints: 'checkpoints/'."""


# =============================================================================
# Network Constants
# =============================================================================

DEFAULT_PORT: int = 8080
"""Default network port: 8080."""

DEFAULT_HOST: str = "localhost"
"""Default network host: 'localhost'."""

TIMEOUT: int = 30
"""Default network timeout in seconds: 30."""

MAX_RETRIES: int = 3
"""Default maximum number of retry attempts: 3."""


# =============================================================================
# Color Constants (ANSI Color Codes for CLI)
# =============================================================================

# Basic Colors
RED: str = "\033[91m"
"""ANSI color code for red text."""

GREEN: str = "\033[92m"
"""ANSI color code for green text."""

BLUE: str = "\033[94m"
"""ANSI color code for blue text."""

YELLOW: str = "\033[93m"
"""ANSI color code for yellow text."""

CYAN: str = "\033[96m"
"""ANSI color code for cyan text."""

MAGENTA: str = "\033[95m"
"""ANSI color code for magenta text."""

BLACK: str = "\033[90m"
"""ANSI color code for black (gray) text."""

WHITE: str = "\033[97m"
"""ANSI color code for white text."""

# Color Modifiers
RESET: str = "\033[0m"
"""ANSI reset code to return to default text formatting."""

BOLD: str = "\033[1m"
"""ANSI code for bold text."""

UNDERLINE: str = "\033[4m"
"""ANSI code for underlined text."""

# Background Colors
BG_RED: str = "\033[41m"
"""ANSI background color code for red."""

BG_GREEN: str = "\033[42m"
"""ANSI background color code for green."""

BG_BLUE: str = "\033[44m"
"""ANSI background color code for blue."""

BG_YELLOW: str = "\033[43m"
"""ANSI background color code for yellow."""

BG_CYAN: str = "\033[46m"
"""ANSI background color code for cyan."""

BG_MAGENTA: str = "\033[45m"
"""ANSI background color code for magenta."""

BG_BLACK: str = "\033[40m"
"""ANSI background color code for black."""

BG_WHITE: str = "\033[47m"
"""ANSI background color code for white."""


# =============================================================================
# Constants Container Class
# =============================================================================


@dataclass
class Constants:
    """
    A container class for managing constants dynamically.

    This class provides a structured way to organize and access constants
    by category. It also supports dynamic constant addition and modification.

    Attributes:
        math: Mathematical constants
        physics: Physics constants
        ml: Machine learning constants
        data: Data processing constants
        file: File system constants
        network: Network constants
        colors: Color constants

    Example:
        >>> consts = Constants()
        >>> consts.math.pi
        3.14159
        >>> consts.ml.default_lr
        0.001
    """

    # Math constants namespace
    math: "MathConstants" = field(default_factory=lambda: MathConstants())

    # Physics constants namespace
    physics: "PhysicsConstants" = field(default_factory=lambda: PhysicsConstants())

    # ML constants namespace
    ml: "MLConstants" = field(default_factory=lambda: MLConstants())

    # Data constants namespace
    data: "DataConstants" = field(default_factory=lambda: DataConstants())

    # File constants namespace
    file: "FileConstants" = field(default_factory=lambda: FileConstants())

    # Network constants namespace
    network: "NetworkConstants" = field(default_factory=lambda: NetworkConstants())

    # Color constants namespace
    colors: "ColorConstants" = field(default_factory=lambda: ColorConstants())

    # Storage for custom constants
    _custom: Dict[str, Any] = field(default_factory=dict, repr=False)

    def get(self, name: str, default: Optional[Any] = None) -> Any:
        """
        Get a constant value by name.

        Args:
            name: The name of the constant (supports dot notation like 'math.pi')
            default: Default value if constant not found

        Returns:
            The constant value or default if not found

        Example:
            >>> consts = Constants()
            >>> consts.get('math.pi')
            3.14159
            >>> consts.get('custom.value', 100)
            100
        """
        try:
            return get_constant(self, name)
        except (AttributeError, KeyError):
            return default

    def set(self, name: str, value: Any) -> None:
        """
        Set a constant value by name.

        Args:
            name: The name of the constant (supports dot notation)
            value: The value to set

        Example:
            >>> consts = Constants()
            >>> consts.set('custom.threshold', 0.5)
            >>> consts.get('custom.threshold')
            0.5
        """
        set_constant(self, name, value)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert all constants to a dictionary.

        Returns:
            Dictionary containing all constant values organized by category
        """
        return {
            "math": self.math.to_dict(),
            "physics": self.physics.to_dict(),
            "ml": self.ml.to_dict(),
            "data": self.data.to_dict(),
            "file": self.file.to_dict(),
            "network": self.network.to_dict(),
            "colors": self.colors.to_dict(),
            "custom": self._custom.copy(),
        }


@dataclass
class MathConstants:
    """Mathematical constants namespace."""

    pi: float = PI
    e: float = E
    golden_ratio: float = GOLDEN_RATIO
    sqrt2: float = SQRT2
    infinity: float = INFINITY
    epsilon: float = EPSILON

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pi": self.pi,
            "e": self.e,
            "golden_ratio": self.golden_ratio,
            "sqrt2": self.sqrt2,
            "infinity": self.infinity,
            "epsilon": self.epsilon,
        }


@dataclass
class PhysicsConstants:
    """Physics constants namespace."""

    speed_of_light: float = SPEED_OF_LIGHT
    gravitational_constant: float = GRAVITATIONAL_CONSTANT
    planck_constant: float = PLANCK_CONSTANT
    boltzmann_constant: float = BOLTZMANN_CONSTANT

    # Aliases
    c: float = field(default=SPEED_OF_LIGHT)
    G: float = field(default=GRAVITATIONAL_CONSTANT)
    h: float = field(default=PLANCK_CONSTANT)
    k: float = field(default=BOLTZMANN_CONSTANT)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "speed_of_light": self.speed_of_light,
            "gravitational_constant": self.gravitational_constant,
            "planck_constant": self.planck_constant,
            "boltzmann_constant": self.boltzmann_constant,
            "c": self.c,
            "G": self.G,
            "h": self.h,
            "k": self.k,
        }


@dataclass
class MLConstants:
    """Machine learning constants namespace."""

    default_lr: float = DEFAULT_LR
    default_batch_size: int = DEFAULT_BATCH_SIZE
    default_epochs: int = DEFAULT_EPOCHS
    default_seed: int = DEFAULT_SEED
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_lr": self.default_lr,
            "default_batch_size": self.default_batch_size,
            "default_epochs": self.default_epochs,
            "default_seed": self.default_seed,
            "early_stopping_patience": self.early_stopping_patience,
        }


@dataclass
class DataConstants:
    """Data processing constants namespace."""

    train_split: float = TRAIN_SPLIT
    val_split: float = VAL_SPLIT
    test_split: float = TEST_SPLIT
    max_seq_length: int = MAX_SEQ_LENGTH
    vocab_size: int = VOCAB_SIZE

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "train_split": self.train_split,
            "val_split": self.val_split,
            "test_split": self.test_split,
            "max_seq_length": self.max_seq_length,
            "vocab_size": self.vocab_size,
        }


@dataclass
class FileConstants:
    """File system constants namespace."""

    default_config_file: str = DEFAULT_CONFIG_FILE
    default_model_dir: str = DEFAULT_MODEL_DIR
    default_data_dir: str = DEFAULT_DATA_DIR
    default_log_dir: str = DEFAULT_LOG_DIR
    default_checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_config_file": self.default_config_file,
            "default_model_dir": self.default_model_dir,
            "default_data_dir": self.default_data_dir,
            "default_log_dir": self.default_log_dir,
            "default_checkpoint_dir": self.default_checkpoint_dir,
        }


@dataclass
class NetworkConstants:
    """Network constants namespace."""

    default_port: int = DEFAULT_PORT
    default_host: str = DEFAULT_HOST
    timeout: int = TIMEOUT
    max_retries: int = MAX_RETRIES

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_port": self.default_port,
            "default_host": self.default_host,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }


@dataclass
class ColorConstants:
    """Color constants namespace for CLI styling."""

    red: str = RED
    green: str = GREEN
    blue: str = BLUE
    yellow: str = YELLOW
    cyan: str = CYAN
    magenta: str = MAGENTA
    black: str = BLACK
    white: str = WHITE
    reset: str = RESET
    bold: str = BOLD
    underline: str = UNDERLINE
    bg_red: str = BG_RED
    bg_green: str = BG_GREEN
    bg_blue: str = BG_BLUE
    bg_yellow: str = BG_YELLOW
    bg_cyan: str = BG_CYAN
    bg_magenta: str = BG_MAGENTA
    bg_black: str = BG_BLACK
    bg_white: str = BG_WHITE

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "red": self.red,
            "green": self.green,
            "blue": self.blue,
            "yellow": self.yellow,
            "cyan": self.cyan,
            "magenta": self.magenta,
            "black": self.black,
            "white": self.white,
            "reset": self.reset,
            "bold": self.bold,
            "underline": self.underline,
            "bg_red": self.bg_red,
            "bg_green": self.bg_green,
            "bg_blue": self.bg_blue,
            "bg_yellow": self.bg_yellow,
            "bg_cyan": self.bg_cyan,
            "bg_magenta": self.bg_magenta,
            "bg_black": self.bg_black,
            "bg_white": self.bg_white,
        }

    def colorize(self, text: str, color: str, bold: bool = False) -> str:
        """
        Apply color to text for CLI output.

        Args:
            text: The text to colorize
            color: The color name (red, green, blue, etc.)
            bold: Whether to make the text bold

        Returns:
            Colorized text string with ANSI codes

        Example:
            >>> colors = ColorConstants()
            >>> colors.colorize("Success!", "green", bold=True)
            '\\x1b[1m\\x1b[92mSuccess!\\x1b[0m'
        """
        color_code = getattr(self, color.lower(), self.white)
        bold_code = self.bold if bold else ""
        return f"{bold_code}{color_code}{text}{self.reset}"


# =============================================================================
# Utility Functions
# =============================================================================


def get_constant(constants_obj: Union[Constants, Any], name: str) -> Any:
    """
    Get a constant value by name from a constants object.

    Supports dot notation for nested constants (e.g., 'math.pi', 'ml.default_lr').

    Args:
        constants_obj: A Constants object or any object with attributes
        name: The name of the constant, can use dot notation for nested access

    Returns:
        The constant value

    Raises:
        AttributeError: If the constant is not found
        KeyError: If accessing a dictionary key that doesn't exist

    Example:
        >>> consts = Constants()
        >>> get_constant(consts, 'math.pi')
        3.14159
        >>> get_constant(consts, 'physics.c')
        299792458.0
    """
    parts = name.split(".")
    current = constants_obj

    for part in parts:
        if hasattr(current, part):
            current = getattr(current, part)
        elif isinstance(current, dict) and part in current:
            current = current[part]
        else:
            raise AttributeError(f"Constant '{name}' not found")

    return current


def set_constant(constants_obj: Constants, name: str, value: Any) -> None:
    """
    Set a constant value by name on a constants object.

    Supports dot notation for nested constants. If the path doesn't exist,
    the constant will be stored in the custom constants namespace.

    Args:
        constants_obj: A Constants object
        name: The name of the constant, can use dot notation
        value: The value to set

    Example:
        >>> consts = Constants()
        >>> set_constant(consts, 'custom.threshold', 0.5)
        >>> set_constant(consts, 'ml.default_lr', 0.01)
    """
    parts = name.split(".")

    if len(parts) == 1:
        # Direct attribute set
        constants_obj._custom[name] = value
    else:
        namespace = parts[0]
        attr_name = parts[1]

        # Map namespace names to actual attributes
        namespace_map = {
            "math": constants_obj.math,
            "physics": constants_obj.physics,
            "ml": constants_obj.ml,
            "data": constants_obj.data,
            "file": constants_obj.file,
            "network": constants_obj.network,
            "colors": constants_obj.colors,
            "custom": constants_obj._custom,
        }

        if namespace in namespace_map:
            target = namespace_map[namespace]
            if hasattr(target, attr_name):
                setattr(target, attr_name, value)
            elif isinstance(target, dict):
                target[attr_name] = value
            else:
                # Store in custom if attribute doesn't exist
                constants_obj._custom[f"{namespace}.{attr_name}"] = value
        else:
            # Unknown namespace, store in custom
            constants_obj._custom[name] = value


def print_constant(constants_obj: Constants, name: str) -> None:
    """
    Print a constant value with its name for debugging.

    Args:
        constants_obj: A Constants object
        name: The name of the constant

    Example:
        >>> consts = Constants()
        >>> print_constant(consts, 'math.pi')
        math.pi = 3.14159
    """
    value = get_constant(constants_obj, name)
    print(f"{name} = {value}")


def list_constants(constants_obj: Constants) -> None:
    """
    Print all available constants organized by category.

    Args:
        constants_obj: A Constants object

    Example:
        >>> consts = Constants()
        >>> list_constants(consts)
        # Displays all constants organized by category
    """
    colors = ColorConstants()

    print(colors.bold + colors.underline + "FISHSTICK CONSTANTS" + colors.reset)
    print()

    sections = [
        ("Math", consts.math),
        ("Physics", consts.physics),
        ("ML", consts.ml),
        ("Data", consts.data),
        ("File", consts.file),
        ("Network", consts.network),
        ("Colors", consts.colors),
    ]

    for section_name, section_obj in sections:
        print(colors.bold + colors.cyan + f"{section_name} Constants:" + colors.reset)
        if hasattr(section_obj, "to_dict"):
            for key, value in section_obj.to_dict().items():
                if isinstance(value, str) and value.startswith("\033"):
                    # Skip raw ANSI codes in display
                    continue
                print(f"  {key}: {value}")
        print()

    if constants_obj._custom:
        print(colors.bold + colors.yellow + "Custom Constants:" + colors.reset)
        for key, value in constants_obj._custom.items():
            print(f"  {key}: {value}")


# =============================================================================
# Module-Level Singleton Instance
# =============================================================================

# Create a default constants instance for convenient access
consts = Constants()
"""
Global singleton instance of Constants for convenient module-level access.

Example:
    >>> from fishstick.constants.core import consts
    >>> consts.math.pi
    3.14159
    >>> consts.ml.default_lr
    0.001
"""


# =============================================================================
# __all__ Definition
# =============================================================================

__all__ = [
    # Math constants
    "PI",
    "E",
    "GOLDEN_RATIO",
    "SQRT2",
    "INFINITY",
    "EPSILON",
    # Physics constants
    "SPEED_OF_LIGHT",
    "GRAVITATIONAL_CONSTANT",
    "PLANCK_CONSTANT",
    "BOLTZMANN_CONSTANT",
    # ML constants
    "DEFAULT_LR",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_EPOCHS",
    "DEFAULT_SEED",
    "EARLY_STOPPING_PATIENCE",
    # Data constants
    "TRAIN_SPLIT",
    "VAL_SPLIT",
    "TEST_SPLIT",
    "MAX_SEQ_LENGTH",
    "VOCAB_SIZE",
    # File constants
    "DEFAULT_CONFIG_FILE",
    "DEFAULT_MODEL_DIR",
    "DEFAULT_DATA_DIR",
    "DEFAULT_LOG_DIR",
    "DEFAULT_CHECKPOINT_DIR",
    # Network constants
    "DEFAULT_PORT",
    "DEFAULT_HOST",
    "TIMEOUT",
    "MAX_RETRIES",
    # Color constants
    "RED",
    "GREEN",
    "BLUE",
    "YELLOW",
    "CYAN",
    "MAGENTA",
    "BLACK",
    "WHITE",
    "RESET",
    "BOLD",
    "UNDERLINE",
    "BG_RED",
    "BG_GREEN",
    "BG_BLUE",
    "BG_YELLOW",
    "BG_CYAN",
    "BG_MAGENTA",
    "BG_BLACK",
    "BG_WHITE",
    # Classes
    "Constants",
    "MathConstants",
    "PhysicsConstants",
    "MLConstants",
    "DataConstants",
    "FileConstants",
    "NetworkConstants",
    "ColorConstants",
    # Utility functions
    "get_constant",
    "set_constant",
    "print_constant",
    "list_constants",
    # Singleton instance
    "consts",
]
