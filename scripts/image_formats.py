# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""Shared image/video extension constants and image encoder settings."""

from pathlib import Path

SUPPORTED_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")
SUPPORTED_VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi")


class ImageFormatError(ValueError):
    """An option was given for a format that cannot honour it."""


class ImageQualityError(ImageFormatError):
    """A quality was given for a format that cannot honour it."""


def check_bit_depth(ext, bit_depth):
    """Validate a per-channel bit depth for the extension ``ext``: 8 anywhere, 16 only for PNG
    (WebP is 8-bit by format, JPEG is not offered at 16). None means "not given"."""
    if bit_depth in (None, 8):
        return bit_depth
    if bit_depth != 16:
        raise ImageFormatError(f"bit depth must be 8 or 16, got {bit_depth!r}")
    if ext.lower() != ".png":
        raise ImageFormatError(f"16-bit output is available for .png only, not {ext} (WebP and JPEG are 8-bit)")
    return bit_depth


def png_bit_depth(path):
    """Bits per channel of a PNG file, read from its IHDR chunk (8 or 16); 8 for anything that is not a PNG."""
    try:
        with open(path, "rb") as f:
            head = f.read(25)
    except OSError:
        return 8
    if len(head) < 25 or head[:8] != b"\x89PNG\r\n\x1a\n" or head[12:16] != b"IHDR":
        return 8
    return head[24]


def add_bit_depth_argument(parser, flag="--bit_depth", default=8, help=None):
    """Register an 8/16 per-channel bit depth flag on an argparse parser."""
    parser.add_argument(
        flag, type=int, choices=(8, 16), default=default,
        help=help or "Bits per channel. 16 is for .png only (about 4x the disk and slower to write); "
                     "default 8. Nothing downstream of the colour step reads more than 8 bits.")


def check_image_quality(ext, quality):
    """Validate an optional encoder quality for the extension ``ext``.

    ``quality`` is None (the default: lossless wherever the format allows it)
    or an int in 1..100. Only ``.webp`` takes a quality (lossy WebP). ``.png``
    is always lossless, and ``.jpg``/``.jpeg`` carry no alpha channel, which the
    RGBA training images need, so a quality with either is an error rather than
    being ignored silently. Returns the validated quality (or None).
    """
    if quality is None:
        return None
    ext = ext.lower()
    if isinstance(quality, bool) or not isinstance(quality, int) or not 1 <= quality <= 100:
        raise ImageQualityError(f"image quality must be an integer from 1 to 100, got {quality!r}")
    if ext != ".webp":
        raise ImageQualityError(
            f"an image quality applies only to .webp output, not {ext}: .png is always "
            "lossless, and .jpg has no alpha channel")
    return quality


LOSSLESS_WEBP_METHOD = 2   # encoder effort 0-6; lossless WebP at the default 4 is ~10x slower than PNG level 1


def pil_save_kwargs(ext, quality=None, png_compress_level=None, webp_method=None):
    """Keyword arguments for ``PIL.Image.save`` for an output with extension ``ext``.

    ``.webp`` with no quality is LOSSLESS (PIL's own default is lossy, quality
    80) at encoder effort ``webp_method`` (default ``LOSSLESS_WEBP_METHOD``;
    measured on a 5312x4648 frame method 0 is 5.5 s, 4 is 9.3 s, 6 is 47.6 s, against
    0.77 s for PNG level 1). ``.webp`` with a quality is lossy (method 4 unless
    ``webp_method`` is given), and the alpha channel stays lossless in either
    case. ``.png`` keeps ``png_compress_level`` when given. Lossy encoding
    happens only when a quality was passed.
    """
    ext = ext.lower()
    quality = check_image_quality(ext, quality)
    if ext == ".webp":
        if quality is None:
            return {"lossless": True, "quality": 100,
                    "method": LOSSLESS_WEBP_METHOD if webp_method is None else webp_method}
        return {"lossless": False, "quality": quality, "method": 4 if webp_method is None else webp_method}
    if ext == ".png" and png_compress_level is not None:
        return {"compress_level": png_compress_level}
    return {}


def cv2_write_params(ext, quality=None, png_compress_level=None):
    """The ``params`` list for ``cv2.imwrite`` for an output with extension ``ext``.

    OpenCV's WebP encoder is lossless when the quality is above 100, and lossy
    otherwise; with no quality given this returns the lossless setting. That
    setting runs the encoder at maximum effort (7.6 s on a 5312x4648 frame), so
    write lossless WebP through ``save_image`` instead, which can pick a faster
    method.
    """
    import cv2

    ext = ext.lower()
    quality = check_image_quality(ext, quality)
    if ext == ".webp":
        return [cv2.IMWRITE_WEBP_QUALITY, 101 if quality is None else quality]
    if ext == ".png" and png_compress_level is not None:
        return [cv2.IMWRITE_PNG_COMPRESSION, png_compress_level]
    return []


def save_image(image, path, quality=None, png_compress_level=None, webp_method=None):
    """Save a PIL image to ``path``, choosing encoder settings from its extension."""
    image.save(path, **pil_save_kwargs(Path(path).suffix, quality, png_compress_level, webp_method))


def add_quality_argument(parser, flag="--image_quality"):
    """Register a 1-100 quality flag (lossy WebP) on an argparse parser."""
    parser.add_argument(
        flag, type=int, default=None, metavar="1-100",
        help="Lossy WebP quality for .webp output (RGBA alpha stays lossless). "
             "Unset (default) is lossless. An error for .png and .jpg.")
