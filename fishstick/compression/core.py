"""
Fishstick - Comprehensive Compression Module
A high-performance, feature-rich compression library for Python.
"""

from __future__ import annotations

import abc
import bz2
import gzip
import io
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
    runtime_checkable,
)

try:
    import lz4.frame

    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False

try:
    import zstandard as zstd

    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False

try:
    import brotli

    HAS_BROTLI = True
except ImportError:
    HAS_BROTLI = False

try:
    import lzma

    HAS_LZMA = True
except ImportError:
    HAS_LZMA = False

try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import av

    HAS_AV = True
except ImportError:
    HAS_AV = False

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

T = TypeVar("T")


# ============================================================================
# Base Compression Interfaces
# ============================================================================


@runtime_checkable
class Compressor(Protocol):
    """Base protocol for compression operations."""

    def compress(self, data: bytes) -> bytes: ...
    def decompress(self, data: bytes) -> bytes: ...


@runtime_checkable
class ArchiveHandler(Protocol):
    """Base protocol for archive operations."""

    def create(self, files: List[Tuple[str, bytes]], output_path: str) -> None: ...
    def extract(self, archive_path: str, output_dir: str) -> List[str]: ...
    def list_contents(self, archive_path: str) -> List[str]: ...


class BaseCompressor(abc.ABC):
    """Abstract base class for all compressor implementations."""

    @abc.abstractmethod
    def compress(self, data: bytes) -> bytes:
        """Compress data."""
        pass

    @abc.abstractmethod
    def decompress(self, data: bytes) -> bytes:
        """Decompress data."""
        pass

    def compress_file(self, input_path: str, output_path: Optional[str] = None) -> str:
        """Compress a file."""
        if output_path is None:
            output_path = input_path + self.get_extension()
        with open(input_path, "rb") as f:
            data = f.read()
        compressed = self.compress(data)
        with open(output_path, "wb") as f:
            f.write(compressed)
        return output_path

    def decompress_file(
        self, input_path: str, output_path: Optional[str] = None
    ) -> str:
        """Decompress a file."""
        with open(input_path, "rb") as f:
            data = f.read()
        decompressed = self.decompress(data)
        if output_path is None:
            output_path = input_path.replace(self.get_extension(), "")
        with open(output_path, "wb") as f:
            f.write(decompressed)
        return output_path

    @abc.abstractmethod
    def get_extension(self) -> str:
        """Return file extension for this compression format."""
        pass


# ============================================================================
# Lossless Compression
# ============================================================================


class GZIPCompress(BaseCompressor):
    """GZIP compression."""

    def __init__(self, compresslevel: int = 9):
        self.compresslevel = compresslevel

    def compress(self, data: bytes) -> bytes:
        return gzip.compress(data, compresslevel=self.compresslevel)

    def decompress(self, data: bytes) -> bytes:
        return gzip.decompress(data)

    def get_extension(self) -> str:
        return ".gz"


class ZLIBCompress(BaseCompressor):
    """ZLIB compression (raw deflate)."""

    def __init__(self, level: int = 9):
        self.level = level

    def compress(self, data: bytes) -> bytes:
        import zlib

        return zlib.compress(data, level=self.level)

    def decompress(self, data: bytes) -> bytes:
        import zlib

        return zlib.decompress(data)

    def get_extension(self) -> str:
        return ".zz"


class BZ2Compress(BaseCompressor):
    """BZ2 compression."""

    def __init__(self, compresslevel: int = 9):
        self.compresslevel = compresslevel

    def compress(self, data: bytes) -> bytes:
        return bz2.compress(data, compresslevel=self.compresslevel)

    def decompress(self, data: bytes) -> bytes:
        return bz2.decompress(data)

    def get_extension(self) -> str:
        return ".bz2"


class LZMACompress(BaseCompressor):
    """LZMA compression."""

    def __init__(self, preset: int = 6):
        if not HAS_LZMA:
            raise ImportError("lzma module is required for LZMACompress")
        self.preset = preset

    def compress(self, data: bytes) -> bytes:
        return lzma.compress(data, preset=self.preset)

    def decompress(self, data: bytes) -> bytes:
        return lzma.decompress(data)

    def get_extension(self) -> str:
        return ".xz"


class LZ4Compress(BaseCompressor):
    """LZ4 compression."""

    def __init__(self, compression_level: int = 0):
        if not HAS_LZ4:
            raise ImportError("lz4.frame package is required for LZ4Compress")
        self.compression_level = compression_level

    def compress(self, data: bytes) -> bytes:
        return lz4.frame.compress(data, compression_level=self.compression_level)

    def decompress(self, data: bytes) -> bytes:
        return lz4.frame.decompress(data)

    def get_extension(self) -> str:
        return ".lz4"


class ZstandardCompress(BaseCompressor):
    """Zstandard (Zstd) compression."""

    def __init__(self, level: int = 3):
        if not HAS_ZSTD:
            raise ImportError("zstandard package is required for ZstandardCompress")
        self.level = level

    def compress(self, data: bytes) -> bytes:
        cctx = zstd.ZstdCompress(level=self.level)
        return cctx.compress(data)

    def decompress(self, data: bytes) -> bytes:
        dctx = zstd.ZstdDecompress()
        return dctx.decompress(data)

    def get_extension(self) -> str:
        return ".zst"


class BrotliCompress(BaseCompressor):
    """Brotli compression."""

    def __init__(self, quality: int = 6):
        if not HAS_BROTLI:
            raise ImportError("brotli package is required for BrotliCompress")
        self.quality = quality

    def compress(self, data: bytes) -> bytes:
        return brotli.compress(data, quality=self.quality)

    def decompress(self, data: bytes) -> bytes:
        return brotli.decompress(data)

    def get_extension(self) -> str:
        return ".br"


# ============================================================================
# Archive Formats
# ============================================================================


class ZIPArchive:
    """ZIP archive handler."""

    def __init__(self, compression: int = zipfile.ZIP_DEFLATED, compresslevel: int = 9):
        self.compression = compression
        self.compresslevel = compresslevel

    def create(self, files: List[Tuple[str, bytes]], output_path: str) -> None:
        with zipfile.ZipFile(
            output_path,
            "w",
            compression=self.compression,
            compresslevel=self.compresslevel,
        ) as zf:
            for name, data in files:
                zf.writestr(name, data)

    def extract(self, archive_path: str, output_dir: str) -> List[str]:
        extracted = []
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(output_dir)
            extracted = zf.namelist()
        return extracted

    def list_contents(self, archive_path: str) -> List[str]:
        with zipfile.ZipFile(archive_path, "r") as zf:
            return zf.namelist()

    def add_file(
        self, archive_path: str, file_path: str, arcname: Optional[str] = None
    ) -> None:
        with zipfile.ZipFile(archive_path, "a", compression=self.compression) as zf:
            zf.write(file_path, arcname=arcname or os.path.basename(file_path))


class TARArchive:
    """TAR archive handler."""

    def __init__(self, compression: Optional[str] = "gz"):
        self.compression = compression

    def _get_mode(self) -> str:
        mode = "w"
        if self.compression == "gz":
            mode += ":gz"
        elif self.compression == "bz2":
            mode += ":bz2"
        elif self.compression == "xz":
            mode += ":xz"
        elif self.compression == "lzma":
            mode += ":lzma"
        return mode

    def create(self, files: List[Tuple[str, bytes]], output_path: str) -> None:
        mode = self._get_mode()
        with tarfile.open(output_path, mode) as tf:
            for name, data in files:
                tf.writestr(name, data)

    def extract(self, archive_path: str, output_dir: str) -> List[str]:
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(output_dir)
            return [m.name for m in tf.getmembers()]

    def list_contents(self, archive_path: str) -> List[str]:
        with tarfile.open(archive_path, "r:*") as tf:
            return [m.name for m in tf.getmembers()]

    def add_file(
        self, archive_path: str, file_path: str, arcname: Optional[str] = None
    ) -> None:
        mode = self._get_mode()
        with tarfile.open(archive_path, mode) as tf:
            tf.add(file_path, arcname=arcname or os.path.basename(file_path))


class SevenZipArchive:
    """7z archive handler (requires p7zip or 7z command)."""

    def __init__(self, compression_level: int = 9):
        self.compression_level = compression_level

    def _run_7z(self, args: List[str]) -> subprocess.CompletedProcess:
        return subprocess.run(["7z"] + args, capture_output=True, check=False)

    def create(self, files: List[Tuple[str, bytes]], output_path: str) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            for name, data in files:
                path = Path(tmpdir) / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            result = self._run_7z(
                ["a", f"-mx={self.compression_level}", output_path, str(tmpdir) + "/*"]
            )
            if result.returncode != 0:
                raise RuntimeError(f"7z failed: {result.stderr.decode()}")

    def extract(self, archive_path: str, output_dir: str) -> List[str]:
        result = self._run_7z(["x", archive_path, f"-o{output_dir}", "-y"])
        if result.returncode != 0:
            raise RuntimeError(f"7z extraction failed: {result.stderr.decode()}")
        return self.list_contents(archive_path)

    def list_contents(self, archive_path: str) -> List[str]:
        result = self._run_7z(["l", archive_path])
        if result.returncode != 0:
            raise RuntimeError(f"7z list failed: {result.stderr.decode()}")
        lines = result.stdout.decode().split("\n")
        return [line.split()[-1] for line in lines[7:-3] if line.strip()]


class RARArchive:
    """RAR archive handler (requires rar command)."""

    def __init__(self, compression_level: int = 5):
        self.compression_level = compression_level

    def _run_rar(self, args: List[str]) -> subprocess.CompletedProcess:
        return subprocess.run(["rar"] + args, capture_output=True, check=False)

    def create(self, files: List[Tuple[str, bytes]], output_path: str) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            for name, data in files:
                path = Path(tmpdir) / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            result = self._run_rar(
                ["a", f"-m{self.compression_level}", output_path, str(tmpdir) + "/*"]
            )
            if result.returncode != 0:
                raise RuntimeError(f"rar failed: {result.stderr.decode()}")

    def extract(self, archive_path: str, output_dir: str) -> List[str]:
        result = self._run_rar(["x", archive_path, output_dir])
        if result.returncode != 0:
            raise RuntimeError(f"rar extraction failed: {result.stderr.decode()}")
        return []

    def list_contents(self, archive_path: str) -> List[str]:
        result = self._run_rar(["l", archive_path])
        if result.returncode != 0:
            raise RuntimeError(f"rar list failed: {result.stderr.decode()}")
        return []


# ============================================================================
# Image Compression
# ============================================================================


class BaseImageCompressor(abc.ABC):
    """Base class for image compression."""

    @abc.abstractmethod
    def compress(
        self, input_path: str, output_path: str, quality: Optional[int] = None, **kwargs
    ) -> None:
        """Compress an image."""
        pass

    @abc.abstractmethod
    def get_extension(self) -> str:
        """Return output file extension."""
        pass

    def compress_buffer(
        self, data: bytes, quality: Optional[int] = None, **kwargs
    ) -> bytes:
        """Compress image from buffer."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_in:
            tmp_in.write(data)
            tmp_in.flush()
            tmp_out = tempfile.NamedTemporaryFile(
                suffix=self.get_extension(), delete=False
            )
            tmp_out.close()
            try:
                self.compress(tmp_in.name, tmp_out.name, quality, **kwargs)
                return Path(tmp_out.name).read_bytes()
            finally:
                Path(tmp_in.name).unlink(missing_ok=True)
                Path(tmp_out.name).unlink(missing_ok=True)


class JPEGCompress(BaseImageCompressor):
    """JPEG image compression."""

    def __init__(self):
        if not HAS_PIL:
            raise ImportError("Pillow package is required for JPEGCompress")

    def compress(
        self, input_path: str, output_path: str, quality: int = 85, **kwargs
    ) -> None:
        with Image.open(input_path) as img:
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGB")
            img.save(output_path, "JPEG", quality=quality, **kwargs)

    def get_extension(self) -> str:
        return ".jpg"


class PNGCompress(BaseImageCompressor):
    """PNG image compression."""

    def __init__(self):
        if not HAS_PIL:
            raise ImportError("Pillow package is required for PNGCompress")

    def compress(
        self, input_path: str, output_path: str, compress_level: int = 9, **kwargs
    ) -> None:
        with Image.open(input_path) as img:
            img.save(output_path, "PNG", compress_level=compress_level, **kwargs)

    def get_extension(self) -> str:
        return ".png"


class WebPCompress(BaseImageCompressor):
    """WebP image compression."""

    def __init__(self):
        if not HAS_PIL:
            raise ImportError("Pillow package is required for WebPCompress")

    def compress(
        self, input_path: str, output_path: str, quality: int = 80, **kwargs
    ) -> None:
        with Image.open(input_path) as img:
            if img.mode in ("RGBA", "LA") and "lossless" not in kwargs:
                img = img.convert("RGBA")
            img.save(output_path, "WEBP", quality=quality, **kwargs)

    def get_extension(self) -> str:
        return ".webp"


class AVIFCompress(BaseImageCompressor):
    """AVIF image compression (requires pillow-avif-plugin)."""

    def __init__(self):
        if not HAS_PIL:
            raise ImportError("Pillow package is required for AVIFCompress")

    def compress(
        self, input_path: str, output_path: str, quality: int = 50, **kwargs
    ) -> None:
        with Image.open(input_path) as img:
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGB")
            img.save(output_path, "AVIF", quality=quality, **kwargs)

    def get_extension(self) -> str:
        return ".avif"


class JPEGXLCompress(BaseImageCompressor):
    """JPEG-XL image compression (requires pillow-jxl-plugin or cjxl)."""

    def __init__(self):
        self._has_pil_jxl = False
        try:
            from PIL import JxlImagePlugin

            self._has_pil_jxl = True
        except ImportError:
            pass

    def compress(
        self, input_path: str, output_path: str, quality: int = 80, **kwargs
    ) -> None:
        if self._has_pil_jxl:
            with Image.open(input_path) as img:
                if img.mode in ("RGBA", "LA", "P"):
                    img = img.convert("RGB")
                img.save(output_path, "JPEG XL", quality=quality, **kwargs)
        else:
            result = subprocess.run(
                ["cjxl", input_path, output_path, f"-q{quality}"],
                capture_output=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"cjxl failed: {result.stderr.decode()}")

    def get_extension(self) -> str:
        return ".jxl"


# ============================================================================
# Video Compression
# ============================================================================


class BaseVideoCompressor(abc.ABC):
    """Base class for video compression."""

    @abc.abstractmethod
    def compress(
        self, input_path: str, output_path: str, quality: int = 23, **kwargs
    ) -> None:
        """Compress a video."""
        pass

    @abc.abstractmethod
    def get_extension(self) -> str:
        """Return output file extension."""
        pass


class VideoEncoder:
    """Base video encoder using ffmpeg."""

    def __init__(self, codec: str):
        self.codec = codec
        self._check_ffmpeg()

    def _check_ffmpeg(self) -> None:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True)
        if result.returncode != 0:
            raise RuntimeError("ffmpeg is required for video encoding")

    def encode(
        self,
        input_path: str,
        output_path: str,
        quality: int = 23,
        preset: str = "medium",
        extra_args: Optional[List[str]] = None,
    ) -> None:
        args = [
            "ffmpeg",
            "-i",
            input_path,
            "-c:v",
            self.codec,
            "-crf",
            str(quality),
            "-preset",
            preset,
            "-y",
            output_path,
        ]
        if extra_args:
            args = args[:-1] + extra_args + [output_path]
        result = subprocess.run(args, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg encoding failed: {result.stderr.decode()}")


class H264Encode(VideoEncoder):
    """H.264/AVC video compression."""

    def __init__(self):
        super().__init__("libx264")

    def compress(
        self, input_path: str, output_path: str, quality: int = 23, **kwargs
    ) -> None:
        self.encode(input_path, output_path, quality, **kwargs)

    def get_extension(self) -> str:
        return ".mp4"


class H265Encode(VideoEncoder):
    """H.265/HEVC video compression."""

    def __init__(self):
        super().__init__("libx265")

    def compress(
        self, input_path: str, output_path: str, quality: int = 28, **kwargs
    ) -> None:
        self.encode(input_path, output_path, quality, **kwargs)

    def get_extension(self) -> str:
        return ".mp4"


class VP9Encode(VideoEncoder):
    """VP9 video compression."""

    def __init__(self):
        super().__init__("libvpx-vp9")

    def compress(
        self, input_path: str, output_path: str, quality: int = 31, **kwargs
    ) -> None:
        self.encode(input_path, output_path, quality, **kwargs)

    def get_extension(self) -> str:
        return ".webm"


class AV1Encode(VideoEncoder):
    """AV1 video compression."""

    def __init__(self):
        super().__init__("libaom-av1")

    def compress(
        self, input_path: str, output_path: str, quality: int = 35, **kwargs
    ) -> None:
        self.encode(input_path, output_path, quality, **kwargs)

    def get_extension(self) -> str:
        return ".mp4"


# ============================================================================
# Audio Compression
# ============================================================================


class BaseAudioCompressor(abc.ABC):
    """Base class for audio compression."""

    @abc.abstractmethod
    def compress(
        self, input_path: str, output_path: str, quality: Optional[int] = None, **kwargs
    ) -> None:
        """Compress an audio file."""
        pass

    @abc.abstractmethod
    def get_extension(self) -> str:
        """Return output file extension."""
        pass


class AudioEncoder:
    """Base audio encoder using ffmpeg."""

    def __init__(self, codec: str):
        self.codec = codec

    def encode(
        self,
        input_path: str,
        output_path: str,
        bitrate: Optional[int] = None,
        extra_args: Optional[List[str]] = None,
    ) -> None:
        args = ["ffmpeg", "-i", input_path, "-c:a", self.codec, "-y", output_path]
        if bitrate:
            args.insert(-1, "-b:a")
            args.insert(-1, f"{bitrate}k")
        if extra_args:
            args = args[:-1] + extra_args + [output_path]
        result = subprocess.run(args, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg encoding failed: {result.stderr.decode()}")


class MP3Encode(AudioEncoder):
    """MP3 audio compression."""

    def __init__(self):
        super().__init__("libmp3lame")

    def compress(
        self, input_path: str, output_path: str, quality: int = 192, **kwargs
    ) -> None:
        self.encode(input_path, output_path, bitrate=quality, **kwargs)

    def get_extension(self) -> str:
        return ".mp3"


class AACEncode(AudioEncoder):
    """AAC audio compression."""

    def __init__(self):
        super().__init__("aac")

    def compress(
        self, input_path: str, output_path: str, quality: int = 192, **kwargs
    ) -> None:
        self.encode(input_path, output_path, bitrate=quality, **kwargs)

    def get_extension(self) -> str:
        return ".m4a"


class OpusEncode(AudioEncoder):
    """Opus audio compression."""

    def __init__(self):
        super().__init__("libopus")

    def compress(
        self, input_path: str, output_path: str, quality: int = 128, **kwargs
    ) -> None:
        self.encode(input_path, output_path, bitrate=quality, **kwargs)

    def get_extension(self) -> str:
        return ".opus"


class FLACCompress(BaseAudioCompressor):
    """FLAC lossless audio compression."""

    def __init__(self):
        pass

    def compress(
        self, input_path: str, output_path: str, compression_level: int = 8, **kwargs
    ) -> None:
        if not HAS_AV:
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-i",
                    input_path,
                    "-c:a",
                    "flac",
                    f"-compression_level{compression_level}",
                    "-y",
                    output_path,
                ],
                capture_output=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg encoding failed: {result.stderr.decode()}")
        else:
            with av.open(input_path) as inp:
                out = av.open(output_path, "w")
                stream = inp.streams.audio[0]
                ostream = out.add_stream("flac", template=stream)
                ostream.options = {"compression_level": str(compression_level)}
                for frame in inp.decode(stream):
                    frame = frame.reformat(ostream.layout)
                    for p in ostream.encode(frame):
                        out.mux(p)
                for p in ostream.encode():
                    out.mux(p)
                out.close()

    def get_extension(self) -> str:
        return ".flac"


# ============================================================================
# Neural Compression
# ============================================================================


class NeuralCompress:
    """Neural compression using pre-trained models."""

    def __init__(self, model_name: str = "bjpeg"):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for NeuralCompress")
        self.model_name = model_name
        self.model = self._load_model(model_name)

    def _load_model(self, model_name: str) -> nn.Module:
        try:
            import torchvision.models as models

            if model_name == "bjpeg":
                model = models.resnet50(pretrained=False)
            else:
                model = models.resnet34(pretrained=False)
            model.eval()
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_name}: {e}")

    def compress(self, data: bytes) -> bytes:
        import numpy as np
        import torch

        arr = np.frombuffer(data, dtype=np.uint8)
        tensor = torch.from_numpy(arr).float() / 255.0
        tensor = tensor.view(1, -1)
        with torch.no_grad():
            encoded = self.model(tensor)
        return encoded.numpy().tobytes()

    def decompress(self, data: bytes) -> bytes:
        import numpy as np
        import torch

        arr = np.frombuffer(data, dtype=np.float32)
        tensor = torch.from_numpy(arr).view(1, -1)
        with torch.no_grad():
            decoded = self.model(tensor)
        result = (decoded.clamp(0, 1) * 255).round().byte()
        return result.numpy().tobytes()


class CompressionAutoencoder(nn.Module):
    """Autoencoder for neural compression."""

    def __init__(
        self, input_dim: int = 784, hidden_dim: int = 256, latent_dim: int = 64
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def compress(self, data: bytes) -> bytes:
        import numpy as np
        import torch

        arr = np.frombuffer(data, dtype=np.uint8)
        tensor = torch.from_numpy(arr).float() / 255.0
        tensor = tensor.view(1, -1)
        with torch.no_grad():
            z = self.encode(tensor)
        return z.numpy().tobytes()

    def decompress(self, data: bytes) -> bytes:
        import numpy as np
        import torch

        arr = np.frombuffer(data, dtype=np.float32)
        z = torch.from_numpy(arr).view(1, -1)
        with torch.no_grad():
            x = self.decode(z)
        result = (x * 255).round().byte()
        return result.numpy().tobytes()


# ============================================================================
# Utility Functions
# ============================================================================


def compress(
    data: bytes,
    method: str = "gzip",
    level: Optional[int] = None,
    **kwargs,
) -> bytes:
    """Compress data using specified method.

    Args:
        data: Data to compress
        method: Compression method ('gzip', 'zlib', 'bz2', 'lzma', 'lz4', 'zstd', 'brotli')
        level: Compression level (method-specific)
        **kwargs: Additional parameters

    Returns:
        Compressed data
    """
    compressors = {
        "gzip": GZIPCompress(level or 9),
        "zlib": ZLIBCompress(level or 9),
        "bz2": BZ2Compress(level or 9),
        "lzma": LZMACompress(level or 6),
        "lz4": LZ4Compress(level or 0),
        "zstd": ZstandardCompress(level or 3),
        "brotli": BrotliCompress(level or 6),
    }
    if method not in compressors:
        raise ValueError(f"Unknown compression method: {method}")
    return compressors[method].compress(data)


def decompress(
    data: bytes,
    method: str = "gzip",
    **kwargs,
) -> bytes:
    """Decompress data using specified method.

    Args:
        data: Compressed data
        method: Decompression method ('gzip', 'zlib', 'bz2', 'lzma', 'lz4', 'zstd', 'brotli')
        **kwargs: Additional parameters

    Returns:
        Decompressed data
    """
    decompressors = {
        "gzip": GZIPCompress(),
        "zlib": ZLIBCompress(),
        "bz2": BZ2Compress(),
        "lzma": LZMACompress(),
        "lz4": LZ4Compress(),
        "zstd": ZstandardCompress(),
        "brotli": BrotliCompress(),
    }
    if method not in decompressors:
        raise ValueError(f"Unknown decompression method: {method}")
    return decompressors[method].decompress(data)


def get_best_codec(
    data: bytes,
    target_size: Optional[int] = None,
    speed_priority: bool = False,
) -> Tuple[str, bytes]:
    """Find the best compression codec for given data.

    Args:
        data: Data to compress
        target_size: Target compressed size (optional)
        speed_priority: Prefer faster codecs over compression ratio

    Returns:
        Tuple of (codec_name, compressed_data)
    """
    codecs = ["gzip", "bz2", "zstd", "lz4", "brotli"]
    if HAS_LZMA:
        codecs.append("lzma")

    results = []
    for codec in codecs:
        try:
            compressed = compress(data, method=codec)
            ratio = len(data) / len(compressed)
            results.append((codec, compressed, ratio))
        except Exception:
            continue

    if not results:
        raise RuntimeError("No compression codec available")

    if speed_priority:
        results.sort(key=lambda x: len(x[1]))
    else:
        results.sort(key=lambda x: x[2], reverse=True)

    if target_size:
        for codec, compressed, ratio in results:
            if len(compressed) <= target_size:
                return codec, compressed
        return results[-1][:2]

    return results[0][0], results[0][1]


# ============================================================================
# Registry
# ============================================================================


COMPRESSION_CODECS = {
    "gzip": GZIPCompress,
    "zlib": ZLIBCompress,
    "bz2": BZ2Compress,
    "lzma": LZMACompress,
    "lz4": LZ4Compress,
    "zstd": ZstandardCompress,
    "brotli": BrotliCompress,
}

ARCHIVE_FORMATS = {
    "zip": ZIPArchive,
    "tar": TARArchive,
    "7z": SevenZipArchive,
    "rar": RARArchive,
}

IMAGE_CODECS = {
    "jpeg": JPEGCompress,
    "png": PNGCompress,
    "webp": WebPCompress,
    "avif": AVIFCompress,
    "jxl": JPEGXLCompress,
}

VIDEO_CODECS = {
    "h264": H264Encode,
    "h265": H265Encode,
    "vp9": VP9Encode,
    "av1": AV1Encode,
}

AUDIO_CODECS = {
    "mp3": MP3Encode,
    "aac": AACEncode,
    "opus": OpusEncode,
    "flac": FLACCompress,
}


def get_codec(
    codec_name: str,
) -> Union[BaseCompressor, BaseImageCompressor, BaseAudioCompressor]:
    """Get a compression codec by name."""
    all_codecs = {**COMPRESSION_CODECS, **IMAGE_CODECS, **AUDIO_CODECS}
    if codec_name in all_codecs:
        return all_codecs[codec_name]()
    if codec_name in VIDEO_CODECS:
        return VIDEO_CODECS[codec_name]()
    raise ValueError(f"Unknown codec: {codec_name}")


__all__ = [
    "BaseCompressor",
    "Compressor",
    "ArchiveHandler",
    "GZIPCompress",
    "ZLIBCompress",
    "BZ2Compress",
    "LZMACompress",
    "LZ4Compress",
    "ZstandardCompress",
    "BrotliCompress",
    "ZIPArchive",
    "TARArchive",
    "SevenZipArchive",
    "RARArchive",
    "JPEGCompress",
    "PNGCompress",
    "WebPCompress",
    "AVIFCompress",
    "JPEGXLCompress",
    "H264Encode",
    "H265Encode",
    "VP9Encode",
    "AV1Encode",
    "MP3Encode",
    "AACEncode",
    "OpusEncode",
    "FLACCompress",
    "NeuralCompress",
    "CompressionAutoencoder",
    "compress",
    "decompress",
    "get_best_codec",
    "get_codec",
    "COMPRESSION_CODECS",
    "ARCHIVE_FORMATS",
    "IMAGE_CODECS",
    "VIDEO_CODECS",
    "AUDIO_CODECS",
]
