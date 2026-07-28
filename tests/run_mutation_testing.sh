#!/bin/sh
# Mutation-test the unit suite with mutmut (quality check on the tests
# themselves: mutmut mutates the code and re-runs the suite; every mutant no
# test kills is a concrete gap where a bug could slip through unnoticed).
#
# mutmut 3.x hardcodes support for ./, src/, and source/ layouts only, and this
# repo's tests import top-level modules via pytest's `pythonpath = ["scripts"]`
# -- running `mutmut run` in the repo root aborts with a trampoline-key
# mismatch. So this script stages a src/-layout copy of scripts/ + tests/unit/
# and runs mutmut there. An unmutated scripts/ copy is also staged
# (`also_copy`) because render_frame_sequence.py resolves sibling scripts by
# on-disk path relative to itself.
#
# Usage:  tests/run_mutation_testing.sh [extra mutmut-run args]
# Results: $STAGE/full-results.txt, or `mutmut results` / `mutmut browse`
# run from inside $STAGE. Full run is ~9,500 mutants; expect it to take a
# while. Survivor triage: argparse-cosmetic survivors (description=None etc.)
# are noise; surviving logic mutants (flipped comparisons, off-by-ones,
# dropped calls) are real test gaps.
set -eu

REPO="$(cd "$(dirname "$0")/.." && pwd)"
STAGE="${MUTATION_STAGE:-${TMPDIR:-/tmp}/mutation-staging-$(basename "$REPO")}"
MUTMUT="$REPO/.venv/bin/mutmut"

[ -x "$MUTMUT" ] || { echo "mutmut not found at $MUTMUT (pip install mutmut)"; exit 1; }

rm -rf "$STAGE"
mkdir -p "$STAGE/src" "$STAGE/tests" "$STAGE/scripts"
cp "$REPO"/scripts/*.py "$STAGE/src/"
cp "$REPO"/scripts/*.py "$STAGE/scripts/"
cp "$REPO"/tests/unit/test_*.py "$STAGE/tests/"

cat > "$STAGE/pyproject.toml" <<'EOF'
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --strict-markers --strict-config"
filterwarnings = ["error"]

[tool.mutmut]
source_paths = ["src"]
tests_dir = ["tests"]
also_copy = ["scripts"]
EOF

echo "Staging: $STAGE"
cd "$STAGE"
"$MUTMUT" run "$@"
"$MUTMUT" results > full-results.txt || true
echo "Results written to $STAGE/full-results.txt (or 'cd $STAGE && $MUTMUT browse')"
