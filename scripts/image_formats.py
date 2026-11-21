# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""Shared image/video extension constants and image encoder settings."""

from pathlib import Path

SUPPORTED_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")
SUPPORTED_VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi")


class ImageQualityError(ValueError):
    """A quality was given for a format that cannot honour it."""


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


def pil_save_kwargs(ext, quality=None, png_compress_level=None):
    """Keyword arguments for ``PIL.Image.save`` for an output with extension ``ext``.

    ``.webp`` with no quality is LOSSLESS (PIL's own default is lossy, quality
    80). ``.webp`` with a quality is lossy (method 4), and the alpha channel
    stays lossless in either case. ``.png`` keeps ``png_compress_level`` when
    given. Lossy encoding happens only when a quality was passed.
    """
    ext = ext.lower()
    quality = check_image_quality(ext, quality)
    if ext == ".webp":
        if quality is None:
            return {"lossless": True, "quality": 100, "method": 4}
        return {"lossless": False, "quality": quality, "method": 4}
    if ext == ".png" and png_compress_level is not None:
        return {"compress_level": png_compress_level}
    return {}


def cv2_write_params(ext, quality=None, png_compress_level=None):
    """The ``params`` list for ``cv2.imwrite`` for an output with extension ``ext``.

    OpenCV's WebP encoder is lossless when the quality is above 100, and lossy
    otherwise; with no quality given this returns the lossless setting.
    """
    import cv2

    ext = ext.lower()
    quality = check_image_quality(ext, quality)
    if ext == ".webp":
        return [cv2.IMWRITE_WEBP_QUALITY, 101 if quality is None else quality]
    if ext == ".png" and png_compress_level is not None:
        return [cv2.IMWRITE_PNG_COMPRESSION, png_compress_level]
    return []


def save_image(image, path, quality=None, png_compress_level=None):
    """Save a PIL image to ``path``, choosing encoder settings from its extension."""
    image.save(path, **pil_save_kwargs(Path(path).suffix, quality, png_compress_level))


def add_quality_argument(parser, flag="--image_quality"):
    """Register a 1-100 quality flag (lossy WebP) on an argparse parser."""
    parser.add_argument(
        flag, type=int, default=None, metavar="1-100",
        help="Lossy WebP quality for .webp output (RGBA alpha stays lossless). "
             "Unset (default) is lossless. An error for .png and .jpg.")
