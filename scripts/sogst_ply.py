#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""sogst_ply.py - the 4D interchange PLY: reader, writer, CLI.

The implementation moved to cumuli_core.ply (deps/cumuli-core).  This file
re-exports it so existing `from sogst_ply import ...` lines keep working,
and keeps the CLI:

    python sogst_ply.py --input scene.sogst --output scene.ply [--sidecar]

which is the same as `python -m cumuli_core.ply ...`.  The format is
docs/sogst-format.md section 7.  Read cumuli_core/ply.py before changing
anything about it.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cumuli_core_path  # noqa: E402,F401  (adds deps/cumuli-core/src if needed)
from cumuli_core.ply import (  # noqa: E402,F401
    PLY_ACCEL_COLUMNS,
    PLY_BASE_COLUMNS,
    PLY_COMMENT_PREFIX,
    PLY_REQUIRED_COMMENTS,
    PLY_SH_COLUMNS,
    SIDECAR_SUFFIX,
    SOGST_FIELDS,
    SOGST_PLY_VERSION,
    _collect_columns,
    main,
    ply_vertex_count,
    read_sogst_ply,
    sh_columns_present,
    write_sogst_ply,
    write_sogst_sidecar,
)

if __name__ == '__main__':
    main()
