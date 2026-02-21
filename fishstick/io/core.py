"""
Comprehensive File I/O Module for Fishstick.

This module provides a complete file I/O infrastructure including:
- File operations (read, write, copy, move, delete, find)
- Path operations (resolve, expand, normalize, absolute)
- Directory operations (list, create, watch, walk)
- File type handlers (image, video, audio, text, binary)
- Temporary file management
- File locking mechanisms
- File monitoring (inotify, FSEvents)
- Utility functions
"""

from __future__ import annotations

import asyncio
import binascii
import contextlib
import errno
import fcntl
import hashlib
import io
import json
import mimetypes
import mmap
import os
import pickle
import platform
import shutil
import socket
import stat
import struct
import sys
import tempfile
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from fnmatch import fnmatch
from functools import partial
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    BinaryIO,
    Callable,
    Dict,
    Generator,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)

T = TypeVar("T")
PathLike = Union[str, Path]


class FileError(Exception):
    """Base exception for file operations."""

    pass


class FileNotFoundError(FileError):
    """Raised when a file is not found."""

    pass


class FileExistsError(FileError):
    """Raised when a file already exists."""

    pass


class FilePermissionError(FileError):
    """Raised when there's a permission error."""

    pass


class FileLockError(FileError):
    """Raised when file locking fails."""

    pass


class FileWatchError(FileError):
    """Raised when file watching fails."""

    pass


class FileOperation(Enum):
    READ = auto()
    WRITE = auto()
    APPEND = auto()
    CREATE = auto()
    DELETE = auto()
    RENAME = auto()
    COPY = auto()
    MOVE = auto()
    LOCK = auto()
    UNLOCK = auto()


@dataclass
class FileInfo:
    path: Path
    size: int = 0
    created: Optional[datetime] = None
    modified: Optional[datetime] = None
    accessed: Optional[datetime] = None
    is_dir: bool = False
    is_file: bool = False
    is_symlink: bool = False
    mode: int = 0
    uid: int = 0
    gid: int = 0

    @classmethod
    def from_path(cls, path: PathLike) -> FileInfo:
        path = Path(path)
        st = path.stat()
        return cls(
            path=path,
            size=st.st_size,
            created=datetime.fromtimestamp(st.st_ctime),
            modified=datetime.fromtimestamp(st.st_mtime),
            accessed=datetime.fromtimestamp(st.st_atime),
            is_dir=path.is_dir(),
            is_file=path.is_file(),
            is_symlink=path.is_symlink(),
            mode=st.st_mode,
            uid=st.st_uid,
            gid=st.st_gid,
        )


@dataclass
class FileMetadata:
    mime_type: Optional[str] = None
    encoding: Optional[str] = None
    extension: Optional[str] = None
    size: int = 0
    checksum: Optional[str] = None


class FileReader:
    __slots__ = ("_buffer_size", "_encoding", "_errors")

    def __init__(
        self, buffer_size: int = 8192, encoding: str = "utf-8", errors: str = "strict"
    ):
        self._buffer_size = buffer_size
        self._encoding = encoding
        self._errors = errors

    def read(self, path: PathLike, size: int = -1) -> str:
        path = Path(path)
        with open(path, "r", encoding=self._encoding, errors=self._errors) as f:
            return f.read(size)

    def read_binary(self, path: PathLike, size: int = -1) -> bytes:
        path = Path(path)
        with open(path, "rb") as f:
            return f.read(size)

    def read_lines(self, path: PathLike, strip: bool = True) -> List[str]:
        path = Path(path)
        with open(path, "r", encoding=self._encoding, errors=self._errors) as f:
            lines = f.readlines()
        if strip:
            return [line.rstrip("\n\r") for line in lines]
        return lines

    def read_line_iter(self, path: PathLike) -> Iterator[str]:
        path = Path(path)
        with open(path, "r", encoding=self._encoding, errors=self._errors) as f:
            for line in f:
                yield line.rstrip("\n\r")

    def read_chunks(self, path: PathLike) -> Iterator[bytes]:
        path = Path(path)
        with open(path, "rb") as f:
            while chunk := f.read(self._buffer_size):
                yield chunk

    def read_mmap(self, path: PathLike, access: int = mmap.ACCESS_READ) -> mmap.mmap:
        path = Path(path)
        f = open(path, "rb")
        return mmap.mmap(f.fileno(), 0, access=access)

    def read_json(self, path: PathLike) -> Dict[str, Any]:
        path = Path(path)
        with open(path, "r", encoding=self._encoding) as f:
            return json.load(f)

    def read_pickle(self, path: PathLike) -> Any:
        path = Path(path)
        with open(path, "rb") as f:
            return pickle.load(f)


class FileWriter:
    __slots__ = ("_buffer_size", "_encoding", "_errors", "_newline")

    def __init__(
        self,
        buffer_size: int = 8192,
        encoding: str = "utf-8",
        errors: str = "strict",
        newline: Optional[str] = None,
    ):
        self._buffer_size = buffer_size
        self._encoding = encoding
        self._errors = errors
        self._newline = newline

    def write(self, path: PathLike, content: str, mode: str = "w") -> int:
        path = Path(path)
        with open(
            path,
            mode,
            encoding=self._encoding,
            errors=self._errors,
            newline=self._newline,
        ) as f:
            return f.write(content)

    def write_binary(self, path: PathLike, content: bytes) -> int:
        path = Path(path)
        with open(path, "wb") as f:
            return f.write(content)

    def write_lines(
        self, path: PathLike, lines: List[str], append_newline: bool = True
    ) -> int:
        path = Path(path)
        with open(
            path,
            "w",
            encoding=self._encoding,
            errors=self._errors,
            newline=self._newline,
        ) as f:
            return f.writelines(
                line + "\n" if append_newline else line for line in lines
            )

    def write_chunks(self, path: PathLike, chunks: Iterator[bytes]) -> int:
        path = Path(path)
        total = 0
        with open(path, "wb") as f:
            for chunk in chunks:
                total += f.write(chunk)
        return total

    def write_json(
        self, path: PathLike, data: Dict[str, Any], indent: Optional[int] = 2
    ) -> None:
        path = Path(path)
        with open(path, "w", encoding=self._encoding) as f:
            json.dump(data, f, indent=indent)

    def write_pickle(
        self, path: PathLike, obj: Any, protocol: int = pickle.HIGHEST_PROTOCOL
    ) -> None:
        path = Path(path)
        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol=protocol)

    def append(self, path: PathLike, content: str) -> int:
        return self.write(path, content, mode="a")

    def append_lines(self, path: PathLike, lines: List[str]) -> int:
        path = Path(path)
        with open(
            path,
            "a",
            encoding=self._encoding,
            errors=self._errors,
            newline=self._newline,
        ) as f:
            return f.writelines(line + "\n" for line in lines)


class FileCopier:
    __slots__ = ("_buffer_size", "_preserve_metadata")

    def __init__(self, buffer_size: int = 131072, preserve_metadata: bool = True):
        self._buffer_size = buffer_size
        self._preserve_metadata = preserve_metadata

    def copy(self, src: PathLike, dst: PathLike, follow_symlinks: bool = True) -> Path:
        src = Path(src)
        dst = Path(dst)
        if src.is_dir():
            return self.copy_tree(src, dst)
        return self.copy_file(src, dst, follow_symlinks)

    def copy_file(
        self, src: PathLike, dst: PathLike, follow_symlinks: bool = True
    ) -> Path:
        src = Path(src)
        dst = Path(dst)
        if not src.exists():
            raise FileNotFoundError(f"Source file not found: {src}")

        dst.parent.mkdir(parents=True, exist_ok=True)

        if self._preserve_metadata:
            shutil.copy2(src, dst, follow_symlinks=follow_symlinks)
        else:
            shutil.copy(src, dst, follow_symlinks=follow_symlinks)

        return dst

    def copy_tree(self, src: PathLike, dst: PathLike) -> Path:
        src = Path(src)
        dst = Path(dst)
        if not src.is_dir():
            raise FileError(f"Source is not a directory: {src}")

        dst.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            src,
            dst,
            dirs_exist_ok=True,
            copy_function=shutil.copy2 if self._preserve_metadata else shutil.copy,
        )

        return dst

    def copy_with_progress(
        self,
        src: PathLike,
        dst: PathLike,
        callback: Optional[Callable[[int, int], None]] = None,
    ) -> Path:
        src = Path(src)
        dst = Path(dst)
        total_size = src.stat().st_size
        copied = 0

        dst.parent.mkdir(parents=True, exist_ok=True)

        with open(src, "rb") as fsrc:
            with open(dst, "wb") as fdst:
                while True:
                    chunk = fsrc.read(self._buffer_size)
                    if not chunk:
                        break
                    fdst.write(chunk)
                    copied += len(chunk)
                    if callback:
                        callback(copied, total_size)

        if self._preserve_metadata:
            shutil.copystat(src, dst)

        return dst


class FileMover:
    __slots__ = ()

    def move(self, src: PathLike, dst: PathLike, overwrite: bool = False) -> Path:
        src = Path(src)
        dst = Path(dst)

        if not src.exists():
            raise FileNotFoundError(f"Source not found: {src}")

        if dst.exists() and not overwrite:
            raise FileExistsError(f"Destination already exists: {dst}")

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))

        return dst

    def rename(self, src: PathLike, dst: PathLike, overwrite: bool = False) -> Path:
        src = Path(src)
        dst = Path(dst)

        if not src.exists():
            raise FileNotFoundError(f"Source not found: {src}")

        if dst.exists() and not overwrite:
            raise FileExistsError(f"Destination already exists: {dst}")

        src.rename(dst)
        return dst


class FileDeleter:
    __slots__ = ("_trash", "_trash_path")

    def __init__(self, use_trash: bool = True, trash_path: Optional[PathLike] = None):
        self._trash = use_trash
        self._trash_path = (
            Path(trash_path)
            if trash_path
            else Path.home() / ".local" / "share" / "Trash" / "files"
        )

    def delete(self, path: PathLike, use_trash: Optional[bool] = None) -> None:
        path = Path(path)
        if not path.exists():
            return

        use_trash = use_trash if use_trash is not None else self._trash

        if use_trash and path.is_file():
            self._move_to_trash(path)
        elif path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

    def _move_to_trash(self, path: PathLike) -> None:
        path = Path(path)
        self._trash_path.mkdir(parents=True, exist_ok=True)

        name = path.name
        trash_name = f"{name}_{int(time.time())}"
        trash_path = self._trash_path / trash_name

        counter = 1
        while trash_path.exists():
            trash_path = self._trash_path / f"{name}_{int(time.time())}_{counter}"
            counter += 1

        shutil.move(str(path), str(trash_path))

    def delete_if_exists(self, path: PathLike) -> bool:
        path = Path(path)
        if path.exists():
            self.delete(path)
            return True
        return False

    def wipe(self, path: PathLike, passes: int = 3) -> None:
        path = Path(path)
        if not path.exists():
            return

        if path.is_dir():
            shutil.rmtree(path)
            return

        size = path.stat().st_size

        with open(path, "ba+") as f:
            for _ in range(passes):
                f.seek(0)
                f.write(os.urandom(size))
                f.flush()
                os.fsync(f.fileno())

        path.unlink()


class FileFinder:
    __slots__ = ("_case_sensitive",)

    def __init__(self, case_sensitive: bool = True):
        self._case_sensitive = case_sensitive

    def find(
        self,
        path: PathLike,
        pattern: str,
        recursive: bool = True,
        type_filter: Optional[str] = None,
    ) -> List[Path]:
        path = Path(path)
        results = []

        if recursive:
            for root, dirs, files in os.walk(path):
                root_path = Path(root)
                for name in files + dirs:
                    if self._match(name, pattern):
                        full_path = root_path / name
                        if self._check_type(full_path, type_filter):
                            results.append(full_path)
        else:
            for item in path.iterdir():
                if self._match(item.name, pattern):
                    if self._check_type(item, type_filter):
                        results.append(item)

        return sorted(results)

    def find_by_name(
        self, path: PathLike, name: str, recursive: bool = True
    ) -> List[Path]:
        return self.find(path, f"*{name}*" if "*" not in name else name, recursive)

    def find_by_extension(
        self, path: PathLike, extensions: Union[str, List[str]], recursive: bool = True
    ) -> List[Path]:
        if isinstance(extensions, str):
            extensions = [extensions]

        extensions = [ext if ext.startswith(".") else f".{ext}" for ext in extensions]

        results = []
        for ext in extensions:
            results.extend(self.find(path, f"*{ext}", recursive))

        return sorted(set(results))

    def find_by_size(
        self,
        path: PathLike,
        min_size: int = 0,
        max_size: Optional[int] = None,
        recursive: bool = True,
    ) -> List[Path]:
        path = Path(path)
        results = []

        iterator = path.rglob("*") if recursive else path.iterdir()

        for item in iterator:
            if not item.is_file():
                continue

            size = item.stat().st_size
            if size >= min_size and (max_size is None or size <= max_size):
                results.append(item)

        return sorted(results, key=lambda p: p.stat().st_size)

    def find_by_date(
        self,
        path: PathLike,
        after: Optional[datetime] = None,
        before: Optional[datetime] = None,
        recursive: bool = True,
    ) -> List[Path]:
        path = Path(path)
        results = []

        iterator = path.rglob("*") if recursive else path.iterdir()

        for item in iterator:
            if not item.is_file():
                continue

            mtime = datetime.fromtimestamp(item.stat().st_mtime)
            if (after is None or mtime >= after) and (
                before is None or mtime <= before
            ):
                results.append(item)

        return sorted(results, key=lambda p: p.stat().st_mtime)

    def _match(self, name: str, pattern: str) -> bool:
        if not self._case_sensitive:
            name = name.lower()
            pattern = pattern.lower()
        return fnmatch(name, pattern)

    def _check_type(self, path: Path, type_filter: Optional[str]) -> bool:
        if type_filter is None:
            return True
        if type_filter == "f" or type_filter == "file":
            return path.is_file()
        if type_filter == "d" or type_filter == "dir":
            return path.is_dir()
        if type_filter == "l" or type_filter == "link":
            return path.is_symlink()
        return True


class PathResolver:
    __slots__ = ()

    @staticmethod
    def resolve(path: PathLike) -> Path:
        return Path(path).resolve()

    @staticmethod
    def expand_path(path: PathLike) -> Path:
        return Path(os.path.expanduser(os.path.expandvars(str(path))))

    @staticmethod
    def normalize_path(path: PathLike) -> Path:
        return Path(os.path.normpath(str(path)))

    @staticmethod
    def make_absolute(path: PathLike, base: Optional[PathLike] = None) -> Path:
        path = Path(path)
        if path.is_absolute():
            return path
        if base:
            return (Path(base) / path).resolve()
        return path.resolve()

    @staticmethod
    def relative(path: PathLike, start: Optional[PathLike] = None) -> Path:
        return Path(path).relative_to(start if start else Path.cwd())

    @staticmethod
    def common_path(paths: List[PathLike]) -> Path:
        return Path(os.path.commonpath([str(p) for p in paths]))

    @staticmethod
    def split_path(path: PathLike) -> Tuple[Path, Path]:
        path = Path(path)
        return path.parent, path.name

    @staticmethod
    def join(*paths: PathLike) -> Path:
        return Path(*paths) if paths else Path()

    @staticmethod
    def get_extension(path: PathLike) -> str:
        return Path(path).suffix

    @staticmethod
    def get_stem(path: PathLike) -> str:
        return Path(path).stem

    @staticmethod
    def get_name(path: PathLike) -> str:
        return Path(path).name


def expand_path(path: PathLike) -> Path:
    """Expand user home directory and environment variables in path."""
    return PathResolver.expand_path(path)


def normalize_path(path: PathLike) -> Path:
    """Normalize path by resolving . and .. components."""
    return PathResolver.normalize_path(path)


def make_absolute(path: PathLike, base: Optional[PathLike] = None) -> Path:
    """Convert path to absolute path."""
    return PathResolver.make_absolute(path, base)


class DirectoryLister:
    __slots__ = ("_sort_key", "_reverse")

    def __init__(self, sort_key: str = "name", reverse: bool = False):
        self._sort_key = sort_key
        self._reverse = reverse

    def list(
        self, path: PathLike, pattern: Optional[str] = None, show_hidden: bool = False
    ) -> List[Path]:
        path = Path(path)
        items = []

        for item in path.iterdir():
            if not show_hidden and item.name.startswith("."):
                continue
            if pattern and not fnmatch(item.name, pattern):
                continue
            items.append(item)

        return self._sort(items)

    def list_dirs(self, path: PathLike) -> List[Path]:
        path = Path(path)
        return self._sort([p for p in path.iterdir() if p.is_dir()])

    def list_files(self, path: PathLike) -> List[Path]:
        path = Path(path)
        return self._sort([p for p in path.iterdir() if p.is_file()])

    def list_all(self, path: PathLike, recursive: bool = False) -> List[Path]:
        path = Path(path)
        if recursive:
            return self._sort(list(path.rglob("*")))
        return self.list(path)

    def _sort(self, items: List[Path]) -> List[Path]:
        if self._sort_key == "name":
            return sorted(items, key=lambda p: p.name, reverse=self._reverse)
        elif self._sort_key == "size":
            return sorted(
                items,
                key=lambda p: p.stat().st_size if p.exists() else 0,
                reverse=self._reverse,
            )
        elif self._sort_key == "modified":
            return sorted(
                items,
                key=lambda p: p.stat().st_mtime if p.exists() else 0,
                reverse=self._reverse,
            )
        return items


class DirectoryCreator:
    __slots__ = ("_mode", "_exist_ok")

    def __init__(self, mode: int = 0o755, exist_ok: bool = True):
        self._mode = mode
        self._exist_ok = exist_ok

    def create(self, path: PathLike, parents: bool = True) -> Path:
        path = Path(path)
        path.mkdir(mode=self._mode, parents=parents, exist_ok=self._exist_ok)
        return path

    def create_temp(self, prefix: str = "fishstick_", suffix: str = "") -> Path:
        return Path(tempfile.mkdtemp(prefix=prefix, suffix=suffix))

    def create_nested(self, path: PathLike) -> Path:
        return self.create(path, parents=True)


class DirectoryWatcher:
    __slots__ = ("_path", "_recursive", "_callback", "_watcher", "_running")

    def __init__(self, path: PathLike, recursive: bool = False):
        self._path = Path(path)
        self._recursive = recursive
        self._callback: Optional[Callable[[str, str], None]] = None
        self._watcher: Optional[Any] = None
        self._running = False

    def watch(self, callback: Callable[[str, str], None]) -> None:
        self._callback = callback

        if sys.platform == "darwin":
            self._watch_macos()
        elif sys.platform == "linux":
            self._watch_inotify()
        else:
            self._watch_polling()

    def _watch_macos(self) -> None:
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler

            class Handler(FileSystemEventHandler):
                def __init__(watcher):
                    self._watcher = watcher

                def on_any_event(self, event):
                    if self._watcher._callback and not event.is_directory:
                        self._watcher._callback(event.event_type, event.src_path)

            self._watcher = Observer()
            handler = Handler(self)
            self._watcher.schedule(handler, str(self._path), recursive=self._recursive)
            self._watcher.start()
            self._running = True
        except ImportError:
            self._watch_polling()

    def _watch_inotify(self) -> None:
        try:
            import inotify.adapters

            self._watcher = inotify.adapters.Inotify()
            self._running = True
            threading.Thread(target=self._inotify_loop, daemon=True).start()
        except ImportError:
            self._watch_polling()

    def _inotify_loop(self) -> None:
        if not self._watcher:
            return
        try:
            for event in self._watcher.event_gen():
                if event:
                    (_, type_names, path, _) = event
                    if self._callback:
                        for type_name in type_names:
                            self._callback(type_name, path)
        except Exception:
            pass

    def _watch_polling(self) -> None:
        self._running = True
        self._last_state: Dict[str, float] = {}
        threading.Thread(target=self._polling_loop, daemon=True).start()

    def _polling_loop(self) -> None:
        while self._running:
            try:
                for item in (
                    self._path.rglob("*") if self._recursive else self._path.iterdir()
                ):
                    mtime = item.stat().st_mtime
                    key = str(item)
                    if key not in self._last_state:
                        self._last_state[key] = mtime
                    elif self._last_state[key] != mtime:
                        self._last_state[key] = mtime
                        if self._callback:
                            self._callback("modified", key)
            except Exception:
                pass
            time.sleep(1)

    def stop(self) -> None:
        self._running = False
        if self._watcher and hasattr(self._watcher, "stop"):
            self._watcher.stop()
        if self._watcher and hasattr(self._watcher, "join"):
            self._watcher.join(timeout=5)


class DirectoryWalker:
    __slots__ = ("_follow_symlinks", "_filter")

    def __init__(
        self,
        follow_symlinks: bool = False,
        filter: Optional[Callable[[Path], bool]] = None,
    ):
        self._follow_symlinks = follow_symlinks
        self._filter = filter

    def walk(
        self, path: PathLike, topdown: bool = True
    ) -> Generator[Tuple[Path, List[Path], List[Path]], None, None]:
        path = Path(path)
        for root, dirs, files in os.walk(
            path, topdown=topdown, followlinks=self._follow_symlinks
        ):
            root_path = Path(root)
            if self._filter:
                dirs[:] = [d for d in dirs if self._filter(root_path / d)]
                files = [f for f in files if self._filter(root_path / f)]
            yield (
                root_path,
                [root_path / d for d in dirs],
                [root_path / f for f in files],
            )

    def walk_files(self, path: PathLike) -> Iterator[Path]:
        path = Path(path)
        for root, _, files in os.walk(path, followlinks=self._follow_symlinks):
            for f in files:
                yield Path(root) / f

    def walk_dirs(self, path: PathLike) -> Iterator[Path]:
        path = Path(path)
        for root, dirs, _ in os.walk(path, followlinks=self._follow_symlinks):
            for d in dirs:
                yield Path(root) / d

    def flat_walk(self, path: PathLike) -> Iterator[Path]:
        path = Path(path)
        for item in path.rglob("*"):
            if self._filter is None or self._filter(item):
                yield item


class ImageFile:
    SUPPORTED_FORMATS = {
        "jpg",
        "jpeg",
        "png",
        "gif",
        "bmp",
        "webp",
        "tiff",
        "ico",
        "svg",
    }

    @classmethod
    def is_image(cls, path: PathLike) -> bool:
        return Path(path).suffix.lower().lstrip(".") in cls.SUPPORTED_FORMATS

    @classmethod
    def get_info(cls, path: PathLike) -> FileMetadata:
        path = Path(path)
        ext = path.suffix.lower().lstrip(".")

        mime_type, _ = mimetypes.guess_type(str(path))

        return FileMetadata(
            mime_type=mime_type,
            extension=path.suffix,
            size=path.stat().st_size if path.exists() else 0,
        )

    @classmethod
    def get_dimensions(cls, path: PathLike) -> Optional[Tuple[int, int]]:
        path = Path(path)
        try:
            from PIL import Image

            with Image.open(path) as img:
                return img.size
        except ImportError:
            return cls._get_dimensions_raw(path)
        except Exception:
            return None

    @classmethod
    def _get_dimensions_raw(cls, path: PathLike) -> Optional[Tuple[int, int]]:
        path = Path(path)

        def get_png_dimensions():
            with open(path, "rb") as f:
                f.seek(16)
                return struct.unpack(">ii", f.read(8))

        def get_jpeg_dimensions():
            with open(path, "rb") as f:
                f.seek(2)
                while True:
                    marker, length = struct.unpack(">2H", f.read(4))
                    if marker & 0xFF00 != 0xFF00:
                        break
                    if marker == 0xFFC0 or marker == 0xFFC1:
                        f.read(1)
                        return struct.unpack(">HH", f.read(4))
                    f.seek(length - 2, 1)

        ext = path.suffix.lower()
        if ext in (".jpg", ".jpeg"):
            return get_jpeg_dimensions()
        elif ext == ".png":
            return get_png_dimensions()

        return None


class VideoFile:
    SUPPORTED_FORMATS = {
        "mp4",
        "avi",
        "mkv",
        "mov",
        "wmv",
        "flv",
        "webm",
        "m4v",
        "mpeg",
        "mpg",
    }

    @classmethod
    def is_video(cls, path: PathLike) -> bool:
        return Path(path).suffix.lower().lstrip(".") in cls.SUPPORTED_FORMATS

    @classmethod
    def get_info(cls, path: PathLike) -> FileMetadata:
        path = Path(path)
        mime_type, _ = mimetypes.guess_type(str(path))

        return FileMetadata(
            mime_type=mime_type,
            extension=path.suffix,
            size=path.stat().st_size if path.exists() else 0,
        )

    @classmethod
    def get_duration(cls, path: PathLike) -> Optional[float]:
        path = Path(path)
        try:
            import mutagen
            from mutagen.mp4 import MP4
            from mutagen.mkv import MKV
            from mutagen.flac import FLAC

            ext = path.suffix.lower()
            if ext == ".mp4":
                return MP4(str(path)).info.length
            elif ext in (".mkv", ".webm"):
                return MKV(str(path)).info.length
            elif ext == ".flac":
                return FLAC(str(path)).info.length
        except ImportError:
            pass
        except Exception:
            pass
        return None


class AudioFile:
    SUPPORTED_FORMATS = {
        "mp3",
        "wav",
        "flac",
        "aac",
        "ogg",
        "wma",
        "m4a",
        "aiff",
        "opus",
    }

    @classmethod
    def is_audio(cls, path: PathLike) -> bool:
        return Path(path).suffix.lower().lstrip(".") in cls.SUPPORTED_FORMATS

    @classmethod
    def get_info(cls, path: PathLike) -> FileMetadata:
        path = Path(path)
        mime_type, _ = mimetypes.guess_type(str(path))

        return FileMetadata(
            mime_type=mime_type,
            extension=path.suffix,
            size=path.stat().st_size if path.exists() else 0,
        )

    @classmethod
    def get_metadata(cls, path: PathLike) -> Dict[str, Any]:
        path = Path(path)
        metadata = {}

        try:
            import mutagen
            from mutagen.mp3 import MP3
            from mutagen.flac import FLAC
            from mutagen.oggvorbis import OggVorbis

            ext = path.suffix.lower()
            if ext == ".mp3":
                audio = MP3(str(path))
                metadata = dict(audio.tags) if audio.tags else {}
                metadata["duration"] = audio.info.length
                metadata["bitrate"] = audio.info.bitrate
            elif ext == ".flac":
                audio = FLAC(str(path))
                metadata = dict(audio.tags) if audio.tags else {}
                metadata["duration"] = audio.info.length
                metadata["sample_rate"] = audio.info.sample_rate
            elif ext == ".ogg":
                audio = OggVorbis(str(path))
                metadata = dict(audio.tags) if audio.tags else {}
                metadata["duration"] = audio.info.length
        except ImportError:
            pass
        except Exception:
            pass

        return metadata


class TextFile:
    ENCODINGS = ("utf-8", "utf-16", "utf-32", "ascii", "iso-8859-1", "cp1252")

    @classmethod
    def is_text(cls, path: PathLike) -> bool:
        path = Path(path)
        if path.suffix.lower() in (".txt", ".text", ".log"):
            return True

        try:
            with open(path, "rb") as f:
                chunk = f.read(1024)
            return cls._is_text_chunk(chunk)
        except Exception:
            return False

    @classmethod
    def _is_text_chunk(cls, chunk: bytes) -> bool:
        if not chunk:
            return True

        text_chars = bytearray(
            {7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x100)) - {0x7F}
        )
        return bool(chunk.translate(None, text_chars))

    @classmethod
    def detect_encoding(cls, path: PathLike) -> str:
        path = Path(path)

        import chardet

        with open(path, "rb") as f:
            result = chardet.detect(f.read(8192))
            return result.get("encoding", "utf-8") or "utf-8"

        return "utf-8"

    @classmethod
    def get_line_count(cls, path: PathLike) -> int:
        path = Path(path)
        count = 0
        with open(path, "rb") as f:
            for _ in f:
                count += 1
        return count

    @classmethod
    def get_word_count(cls, path: PathLike) -> int:
        path = Path(path)
        count = 0
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                count += len(line.split())
        return count


class BinaryFile:
    @classmethod
    def is_binary(cls, path: PathLike) -> bool:
        return not TextFile.is_text(path)

    @classmethod
    def get_hex_dump(cls, path: PathLike, offset: int = 0, length: int = 256) -> str:
        path = Path(path)

        with open(path, "rb") as f:
            f.seek(offset)
            data = f.read(length)

        lines = []
        for i in range(0, len(data), 16):
            chunk = data[i : i + 16]
            hex_part = " ".join(f"{b:02x}" for b in chunk)
            ascii_part = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
            lines.append(f"{i:08x}  {hex_part:<48}  {ascii_part}")

        return "\n".join(lines)

    @classmethod
    def calculate_checksum(cls, path: PathLike, algorithm: str = "sha256") -> str:
        path = Path(path)

        hash_func = getattr(hashlib, algorithm)()

        with open(path, "rb") as f:
            for chunk in iter(partial(f.read, 8192), b""):
                hash_func.update(chunk)

        return hash_func.hexdigest()

    @classmethod
    def get_magic_number(cls, path: PathLike, length: int = 16) -> str:
        path = Path(path)

        with open(path, "rb") as f:
            magic = f.read(length)

        return binascii.hexlify(magic).decode("ascii")


class TemporaryFile:
    def __init__(
        self,
        mode: str = "w+b",
        encoding: str = "utf-8",
        suffix: Optional[str] = None,
        prefix: Optional[str] = None,
        dir: Optional[PathLike] = None,
        delete: bool = True,
    ):
        self._mode = mode
        self._encoding = encoding
        self._suffix = suffix
        self._prefix = prefix
        self._dir = dir
        self._delete = delete
        self._file: Optional[io.IOBase] = None
        self._path: Optional[Path] = None

    def __enter__(self) -> io.IOBase:
        self._file = tempfile.NamedTemporaryFile(
            mode=self._mode,
            encoding=self._encoding if "b" not in self._mode else None,
            suffix=self._suffix,
            prefix=self._prefix,
            dir=self._dir,
            delete=self._delete,
        )
        self._path = Path(self._file.name)
        return self._file

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._file:
            self._file.close()

    @property
    def path(self) -> Optional[Path]:
        return self._path

    @property
    def name(self) -> Optional[str]:
        return self._path.name if self._path else None


class TemporaryDirectory:
    def __init__(
        self,
        suffix: Optional[str] = None,
        prefix: Optional[str] = None,
        dir: Optional[PathLike] = None,
    ):
        self._suffix = suffix
        self._prefix = prefix
        self._dir = dir
        self._path: Optional[Path] = None

    def __enter__(self) -> Path:
        self._path = Path(
            tempfile.mkdtemp(suffix=self._suffix, prefix=self._prefix, dir=self._dir)
        )
        return self._path

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._path and self._path.exists():
            shutil.rmtree(self._path)

    @property
    def path(self) -> Optional[Path]:
        return self._path


class NamedTemporaryFile:
    def __init__(
        self,
        mode: str = "w+b",
        encoding: str = "utf-8",
        suffix: Optional[str] = None,
        prefix: Optional[str] = None,
        dir: Optional[PathLike] = None,
        delete: bool = True,
        permanent: bool = False,
    ):
        self._mode = mode
        self._encoding = encoding
        self._suffix = suffix
        self._prefix = prefix
        self._dir = dir
        self._delete = delete
        self._permanent = permanent
        self._file: Optional[tempfile.NamedTemporaryFile] = None

    def __enter__(self) -> tempfile.NamedTemporaryFile:
        kwargs = {
            "mode": self._mode,
            "suffix": self._suffix,
            "prefix": self._prefix,
            "dir": self._dir,
            "delete": self._delete,
        }

        if "b" not in self._mode:
            kwargs["encoding"] = self._encoding

        self._file = tempfile.NamedTemporaryFile(**kwargs)

        if self._permanent:
            self._path = Path(self._file.name)
            self._file_path = self._path.with_suffix(self._path.suffix + ".tmp")
            self._file.close()
            self._path.rename(self._file_path)
            self._file = open(self._file_path, self._mode)

        return self._file

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._file:
            self._file.close()

    @property
    def name(self) -> Optional[str]:
        return self._file.name if self._file else None

    @property
    def path(self) -> Optional[Path]:
        return Path(self._file.name) if self._file else None


class SpooledTemporaryFile:
    def __init__(
        self,
        max_size: int = 1048576,
        mode: str = "w+b",
        encoding: str = "utf-8",
        suffix: Optional[str] = None,
        prefix: Optional[str] = None,
        dir: Optional[PathLike] = None,
    ):
        self._max_size = max_size
        self._mode = mode
        self._encoding = encoding
        self._suffix = suffix
        self._prefix = prefix
        self._dir = dir
        self._file: Optional[tempfile.SpooledTemporaryFile] = None

    def __enter__(self) -> tempfile.SpooledTemporaryFile:
        kwargs = {
            "max_size": self._max_size,
            "mode": self._mode,
            "suffix": self._suffix,
            "prefix": self._prefix,
            "dir": self._dir,
        }

        if "b" not in self._mode:
            kwargs["encoding"] = self._encoding

        self._file = tempfile.SpooledTemporaryFile(**kwargs)
        return self._file

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._file:
            self._file.close()

    @property
    def is_disk(self) -> bool:
        return self._file._rolled if self._file else False


class FileLock:
    def __init__(self, path: PathLike, timeout: float = 10.0):
        self._path = Path(path)
        self._timeout = timeout
        self._lock_file: Optional[int] = None
        self._acquired = False

    def acquire(self, blocking: bool = True) -> bool:
        self._lock_file = os.open(str(self._path), os.O_CREAT | os.O_RDWR)

        try:
            if blocking:
                fcntl.flock(self._lock_file, fcntl.LOCK_EX)
            else:
                fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._acquired = True
            return True
        except (IOError, OSError):
            os.close(self._lock_file)
            self._lock_file = None
            return False

    def release(self) -> None:
        if self._lock_file is not None and self._acquired:
            fcntl.flock(self._lock_file, fcntl.LOCK_UN)
            os.close(self._lock_file)
            self._lock_file = None
            self._acquired = False

    def __enter__(self) -> "FileLock":
        start_time = time.time()
        while True:
            if self.acquire(blocking=False):
                return self
            if time.time() - start_time >= self._timeout:
                raise FileLockError(
                    f"Could not acquire lock within {self._timeout} seconds"
                )
            time.sleep(0.1)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    @property
    def is_locked(self) -> bool:
        return self._acquired


class SharedLock:
    def __init__(self, path: PathLike, timeout: float = 10.0):
        self._path = Path(path)
        self._timeout = timeout
        self._lock_file: Optional[int] = None
        self._acquired = False

    def acquire(self, blocking: bool = True) -> bool:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_file = os.open(str(self._path), os.O_CREAT | os.O_RDWR)

        try:
            if blocking:
                fcntl.flock(self._lock_file, fcntl.LOCK_SH)
            else:
                fcntl.flock(self._lock_file, fcntl.LOCK_SH | fcntl.LOCK_NB)
            self._acquired = True
            return True
        except (IOError, OSError):
            os.close(self._lock_file)
            self._lock_file = None
            return False

    def release(self) -> None:
        if self._lock_file is not None and self._acquired:
            fcntl.flock(self._lock_file, fcntl.LOCK_UN)
            os.close(self._lock_file)
            self._lock_file = None
            self._acquired = False

    def __enter__(self) -> "SharedLock":
        start_time = time.time()
        while True:
            if self.acquire(blocking=False):
                return self
            if time.time() - start_time >= self._timeout:
                raise FileLockError(
                    f"Could not acquire lock within {self._timeout} seconds"
                )
            time.sleep(0.1)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()


class ExclusiveLock:
    def __init__(self, path: PathLike, timeout: float = 10.0):
        self._path = Path(path)
        self._timeout = timeout
        self._lock_file: Optional[int] = None
        self._acquired = False

    def acquire(self, blocking: bool = True) -> bool:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_file = os.open(str(self._path), os.O_CREAT | os.O_RDWR)

        try:
            if blocking:
                fcntl.flock(self._lock_file, fcntl.LOCK_EX)
            else:
                fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._acquired = True
            return True
        except (IOError, OSError):
            os.close(self._lock_file)
            self._lock_file = None
            return False

    def release(self) -> None:
        if self._lock_file is not None and self._acquired:
            fcntl.flock(self._lock_file, fcntl.LOCK_UN)
            os.close(self._lock_file)
            self._lock_file = None
            self._acquired = False

    def __enter__(self) -> "ExclusiveLock":
        start_time = time.time()
        while True:
            if self.acquire(blocking=False):
                return self
            if time.time() - start_time >= self._timeout:
                raise FileLockError(
                    f"Could not acquire lock within {self._timeout} seconds"
                )
            time.sleep(0.1)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()


class WatchEvent:
    def __init__(self, event_type: str, path: str, is_directory: bool = False):
        self.event_type = event_type
        self.path = path
        self.is_directory = is_directory
        self.timestamp = datetime.now()

    def __repr__(self) -> str:
        return f"WatchEvent(type={self.event_type}, path={self.path}, time={self.timestamp})"


class FileWatcher:
    def __init__(self, path: PathLike, recursive: bool = False, debounce: float = 0.5):
        self._path = Path(path)
        self._recursive = recursive
        self._debounce = debounce
        self._running = False
        self._callbacks: List[Callable[[WatchEvent], None]] = []
        self._thread: Optional[threading.Thread] = None
        self._last_events: Dict[str, float] = {}

    def add_callback(self, callback: Callable[[WatchEvent], None]) -> None:
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[WatchEvent], None]) -> None:
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def start(self) -> None:
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _watch_loop(self) -> None:
        if sys.platform == "linux":
            self._watch_inotify()
        elif sys.platform == "darwin":
            self._watch_fsevents()
        else:
            self._watch_polling()

    def _watch_inotify(self) -> None:
        try:
            import inotify.adapters

            i = inotify.adapters.Inotify()
            i.add_watch(str(self._path))

            for event in i.event_gen():
                if not self._running:
                    break
                if event:
                    (_, type_names, path, _) = event
                    for type_name in type_names:
                        if self._should_notify(path):
                            self._notify(WatchEvent(type_name, path))
        except ImportError:
            self._watch_polling()

    def _watch_fsevents(self) -> None:
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler

            class Handler(FileSystemEventHandler):
                def __init__(watcher):
                    self._watcher = watcher

                def on_any_event(self, event):
                    if self._watcher._should_notify(event.src_path):
                        self._watcher._notify(
                            WatchEvent(
                                event.event_type, event.src_path, event.is_directory
                            )
                        )

            observer = Observer()
            handler = Handler(self)
            observer.schedule(handler, str(self._path), recursive=self._recursive)
            observer.start()

            while self._running:
                time.sleep(0.1)

            observer.stop()
            observer.join()
        except ImportError:
            self._watch_polling()

    def _watch_polling(self) -> None:
        last_states: Dict[str, float] = {}

        def get_files():
            if self._recursive:
                return list(self._path.rglob("*"))
            return list(self._path.iterdir())

        while self._running:
            try:
                current_files = get_files()
                for f in current_files:
                    if not f.exists():
                        continue
                    mtime = f.stat().st_mtime
                    key = str(f)

                    if key not in last_states:
                        last_states[key] = mtime
                        if self._should_notify(key):
                            self._notify(WatchEvent("created", key))
                    elif last_states[key] != mtime:
                        last_states[key] = mtime
                        if self._should_notify(key):
                            self._notify(WatchEvent("modified", key))

                for key in list(last_states.keys()):
                    if not any(str(f) == key for f in current_files):
                        del last_states[key]
                        if self._should_notify(key):
                            self._notify(WatchEvent("deleted", key))
            except Exception:
                pass

            time.sleep(self._debounce)

    def _should_notify(self, path: str) -> bool:
        current_time = time.time()
        key = path

        if key in self._last_events:
            if current_time - self._last_events[key] < self._debounce:
                return False

        self._last_events[key] = current_time
        return True

    def _notify(self, event: WatchEvent) -> None:
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception:
                pass


class InotifyWatcher:
    def __init__(self, path: PathLike, mask: Optional[int] = None):
        self._path = Path(path)
        self._mask = mask or (
            0x00000001
            | 0x00000002
            | 0x00000004
            | 0x00000008
            | 0x00000010
            | 0x00000020
            | 0x00000080
            | 0x00000100
        )
        self._fd: Optional[int] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        import inotify.adapters

        self._adapter = inotify.adapters.Inotify()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _loop(self) -> None:
        import inotify.adapters

        try:
            for event in self._adapter.event_gen():
                if not self._running:
                    break
                if event:
                    yield event
        except Exception:
            pass

    def events(self) -> Iterator[Tuple[str, List[str], str]]:
        import inotify.adapters

        i = inotify.adapters.Inotify()
        i.add_watch(str(self._path))

        for event in i.event_gen():
            if event:
                (_, type_names, path, _) = event
                yield _, type_names, path


class FSEventsWatcher:
    def __init__(self, path: PathLike, latency: float = 1.0):
        self._path = Path(path)
        self._latency = latency
        self._running = False

    def watch(self, callback: Callable[[WatchEvent], None]) -> None:
        try:
            import fsevents

            stream = fsevents.Stream(self._callback, str(self._path), recursive=True)
            observer = fsevents.Observer()
            observer.schedule(stream)
            observer.start()
            self._running = True
            self._observer = observer
            self._callback = callback
        except ImportError:
            raise FileWatchError(
                "fsevents not available. Install with: pip install fsevents"
            )

    def stop(self) -> None:
        self._running = False
        if hasattr(self, "_observer"):
            self._observer.stop()

    @property
    def is_running(self) -> bool:
        return self._running


def read_file(
    path: PathLike, binary: bool = False, encoding: str = "utf-8"
) -> Union[str, bytes]:
    """Read file content."""
    if binary:
        with open(path, "rb") as f:
            return f.read()
    with open(path, "r", encoding=encoding) as f:
        return f.read()


def write_file(
    path: PathLike, content: Union[str, bytes], encoding: str = "utf-8"
) -> int:
    """Write content to file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(content, bytes):
        with open(path, "wb") as f:
            return f.write(content)
    with open(path, "w", encoding=encoding) as f:
        return f.write(content)


def copy_file(src: PathLike, dst: PathLike, preserve_metadata: bool = True) -> Path:
    """Copy file from source to destination."""
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    if preserve_metadata:
        shutil.copy2(src, dst)
    else:
        shutil.copy(src, dst)

    return dst


def move_file(src: PathLike, dst: PathLike) -> Path:
    """Move file from source to destination."""
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    return dst


def delete_file(path: PathLike) -> None:
    """Delete a file."""
    Path(path).unlink()


def file_exists(path: PathLike) -> bool:
    """Check if file exists."""
    return Path(path).exists()


def is_file(path: PathLike) -> bool:
    """Check if path is a file."""
    return Path(path).is_file()


def is_directory(path: PathLike) -> bool:
    """Check if path is a directory."""
    return Path(path).is_dir()


def get_file_size(path: PathLike) -> int:
    """Get file size in bytes."""
    return Path(path).stat().st_size


def get_file_mtime(path: PathLike) -> datetime:
    """Get file modification time."""
    return datetime.fromtimestamp(Path(path).stat().st_mtime)


def get_file_ctime(path: PathLike) -> datetime:
    """Get file creation time."""
    return datetime.fromtimestamp(Path(path).stat().st_ctime)


def chmod(path: PathLike, mode: int) -> None:
    """Change file mode."""
    os.chmod(path, mode)


def chown(path: PathLike, uid: int, gid: int) -> None:
    """Change file owner."""
    os.chown(path, uid, gid)


def create_symlink(src: PathLike, dst: PathLike) -> None:
    """Create symbolic link."""
    Path(dst).symlink_to(src)


def read_symlink(path: PathLike) -> Path:
    """Read symbolic link target."""
    return Path(path).resolve()


__all__ = [
    "FileError",
    "FileNotFoundError",
    "FileExistsError",
    "FilePermissionError",
    "FileLockError",
    "FileWatchError",
    "FileOperation",
    "FileInfo",
    "FileMetadata",
    "FileReader",
    "FileWriter",
    "FileCopier",
    "FileMover",
    "FileDeleter",
    "FileFinder",
    "PathResolver",
    "expand_path",
    "normalize_path",
    "make_absolute",
    "DirectoryLister",
    "DirectoryCreator",
    "DirectoryWatcher",
    "DirectoryWalker",
    "ImageFile",
    "VideoFile",
    "AudioFile",
    "TextFile",
    "BinaryFile",
    "TemporaryFile",
    "TemporaryDirectory",
    "NamedTemporaryFile",
    "SpooledTemporaryFile",
    "FileLock",
    "SharedLock",
    "ExclusiveLock",
    "FileWatcher",
    "WatchEvent",
    "InotifyWatcher",
    "FSEventsWatcher",
    "read_file",
    "write_file",
    "copy_file",
    "move_file",
    "delete_file",
    "file_exists",
    "is_file",
    "is_directory",
    "get_file_size",
    "get_file_mtime",
    "get_file_ctime",
    "chmod",
    "chown",
    "create_symlink",
    "read_symlink",
]
