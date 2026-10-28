# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""cumuli_core_path.py - make `import cumuli_core` work from scripts/.

cumuli_core is the shared library in the deps/cumuli-core submodule.
scripts/setup_cumuli_env.sh installs it editable into the cumuli env, and
then this module does nothing.  When it is not installed (a fresh clone, a
bare CI python) and the submodule is checked out, importing this module
puts deps/cumuli-core/src on sys.path.  When neither holds, the later
`import cumuli_core` fails with the usual ImportError, and this module's
hint says how to fix it.
"""

import importlib.util
import os
import sys

CORE_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'deps', 'cumuli-core', 'src')
HINT = ('cumuli_core is not importable. Run `git submodule update --init '
        'deps/cumuli-core` and `pip install -e deps/cumuli-core` (the setup '
        'script does both).')

if importlib.util.find_spec('cumuli_core') is None and os.path.isdir(CORE_SRC):
    sys.path.insert(0, CORE_SRC)
