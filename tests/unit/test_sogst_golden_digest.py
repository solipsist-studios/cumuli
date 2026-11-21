# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""Golden digests of the reference encoder's output.

Every other sogst test checks decoded values within a tolerance, which is the
right bar between implementations (spec section 10) and the wrong one for
catching an unintended change in our own encoder: a reordering, a codebook
tweak or a manifest change can stay inside every tolerance while changing
every byte.  These digests pin the bytes.

When a digest changes on purpose, update it here and say why in the commit.
Lossless WebP output depends on the libwebp build, so the digests are
recorded with the Pillow and libwebp versions that produced them, and the
test skips (rather than fails) under different versions.  The fixture has no
higher-order SH, because the SH palette is fitted with torch and its result
depends on the device.
"""

import hashlib

import PIL
import PIL.features
import pytest

import make_sogst_fixture as fixture
from sogst_pack import compute_sogst_order, pack_sogst

RECORDED_WITH = ("12.2.0", "1.6.0")   # (Pillow, libwebp)

GOLDEN = {
    "full": "c83e1a597b3109099c2c05c764426daa2ae149b6c9e7a4fe46c58782e2352e35",
    "coarse": "2c0d3c316def13a0f838371f3d9bdfae2afad839833a8240edd114ec37ab9334",
}
CASES = {"full": {}, "coarse": {"drop_bits": {"means": 1, "motion": 6, "accel": 6}}}


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_encoder_bytes_match_the_golden_digest(tmp_path, name):
    versions = (PIL.__version__, PIL.features.version("webp"))
    if versions != RECORDED_WITH:
        pytest.skip(f"digests recorded with Pillow/libwebp {RECORDED_WITH}, running {versions}")
    fields, meta = fixture.build_fixture(count=2048, degree=2, include_sh=False, seed=11)
    order = compute_sogst_order(fields, meta["time_min"], meta["time_max"], 0.1)
    path = tmp_path / f"{name}.sogst"
    pack_sogst(str(path), fields, meta["time_min"], meta["time_max"], meta["fps"],
               shn_count=0, order_segments=order, **CASES[name])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == GOLDEN[name]
