# Contributing

Thanks for your interest. Please read this before opening a pull request --
there are two things about this project that differ from the norm and will
save you wasted effort.

## Read this first

**1. This project is source-available, not open source.** It is licensed
under [PolyForm Noncommercial 1.0.0](LICENSE.md), which is not an
OSI-approved license. Any noncommercial use is free, and use by charities,
educational institutions, public research organizations, and government
institutions is free regardless of how that work is funded. Commercial use
requires a separate license. 

**2. Contributions require a signed CLA.** Because commercial licenses are
sold separately, we need the right to license your contribution under terms
other than PolyForm -- which your copyright, held by you, does not
otherwise permit. A Contributor License Agreement is how that gets granted. 
We say this plainly because some people would rather not contribute under 
those terms, and you deserve to know before you write code, not after.

See [Contributor License Agreement](#contributor-license-agreement) below
for what it grants and how to sign.

## Before you write code

**Open an issue first** for anything beyond a typo or an obvious bug fix.
This pipeline has a lot of implicit coupling between stages (see below),
and a change that looks local often is not. A short conversation up front
is cheaper than a rewritten PR.

## Setting up

```bash
git submodule update --init --recursive
conda env create -f envs/<name>.yml
```

[docs/environment.md](docs/environment.md) is the precise reference and
lists known-good pinned versions. [docs/pipeline.md](docs/pipeline.md) is
the authoritative stage-by-stage walkthrough -- **read it before changing
pipeline order or data conventions.** You will need Linux and an NVIDIA
GPU; there is no way to meaningfully test most of this without one.

## How this codebase is shaped

There is no build system, no package, and no framework. Every stage is a
standalone Python CLI script in `scripts/`, run with `python3`, and stages
communicate **only through files and directory conventions**. Each script's
module docstring is the flag reference for that script -- if you add a
flag, document it there.

Things that are easy to break without noticing:

- **`run_unified_pipeline.py` and `render_frame_sequence.py` share stage
  functions.** `render_frame_sequence.py` imports the orchestrator as a
  module and calls its stage functions directly. A change to a stage
  function affects both. Check both before you submit.
- **Stage output layout is the next stage's input contract.** Renaming a
  directory or an output file is an interface change, not a cleanup.
- **Camera labels.** `build_flat_dataset.py` is the single place where
  HLOC's `Camera_undistorted_0001`-style labels become the plain
  zero-padded two-digit labels (`"00"`, `"01"`, ...) that everything after
  it looks up by literal string. Do not introduce a second place that does
  this.
- **Extension constants live in `scripts/image_formats.py`.** Use them
  rather than hardcoding another list of image extensions.
- **Keypoints are `goliath308`.** `refine_poses_with_keypoints.py` hard
  assumes the 308-keypoint layout with no backward compatibility.
  `coco_wholebody133` is a legacy fallback in `predict_keypoints_2d.py`
  only.
- **Never dilate masks.** Dilated alpha supervision teaches the splat a
  white silhouette fringe. This has been learned the hard way.

New files need the license header, matching the existing scripts:

```python
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
```

Match the surrounding style otherwise. `ruff` and `mypy` run in CI but are
advisory, not gates (see [Testing](#testing)); the existing scripts are
still the practical style guide. `ruff format` is deliberately not enforced
-- roughly 40 files would need reformatting first, and we would rather not
bury real changes under a reformat commit.

## Testing

```bash
pip install -r requirements-dev.txt
pytest tests/unit                # the whole unit suite
pytest tests/unit -n auto        # parallel, as CI runs it
pytest tests/unit/test_clean_masks.py -k dilate    # one file, one pattern
```

**You do not need a GPU or any capture footage to run the unit suite.** The
tests monkeypatch `subprocess.run` before any wrapped tool is invoked, so
nothing in `deps/` is executed and no submodules are needed -- CI
deliberately checks out without them. This is the cheapest way to catch a
regression, so please run it before opening a PR.

A few things about the setup that will save you confusion:

- `pyproject.toml` puts `scripts/` on `pythonpath`, so tests import pipeline
  modules directly by name (`import clean_masks as cm`), not by path.
- `filterwarnings = ["error"]` -- **warnings are errors** in the test run. A
  new `DeprecationWarning` from a dependency will fail the suite. That is
  intentional; it is how the pins stay honest.
- Property-based tests use [Hypothesis][hyp] and are guarded with
  `pytest.importorskip("hypothesis")`, so they skip cleanly rather than
  failing collection if it is not installed.
- `tests/run_mutation_testing.sh` runs mutmut against a staged `src/`-layout
  copy of the tree (mutmut 3.x only supports certain layouts). It is not
  part of CI -- run it when you want to check whether a test actually
  asserts anything.

[hyp]: https://hypothesis.readthedocs.io/

### CI

`.github/workflows/tests.yml` runs on every pull request:

- **unit-tests** -- `pytest tests/unit -n auto`. This gates the merge.
- **lint** -- `ruff check` and `mypy`, both **advisory** (`continue-on-error`).
  They will not fail your build. Please still read the annotations; we would
  like to gate these eventually and every new warning makes that harder.

### What the unit tests do and do not cover

They cover the plumbing: argument construction, file and directory
conventions, label mapping, error handling. Because they mock out every
wrapped tool, they **cannot** tell you whether a change produces a better
splat, a cleaner mask, or a more accurate pose. Passing CI is necessary, not
sufficient.

So for changes to pose estimation, masking, or training quality, we still
need evidence in the PR description:

1. The exact command you ran.
2. The data: how many cameras, what resolution, single frame or a sequence.
3. A before/after on the same input.
4. A number. Subject-space median reprojection error is the metric this
   project already tracks, and `run_pose_refinement.py --report_only` prints
   it without modifying anything. "Looks the same" is not evidence; "median
   reprojection error went from 5.1px to 5.0px" is.

If you cannot run an end-to-end check -- no GPU, no multi-camera footage --
say so plainly rather than implying you did. A PR that passes the unit
suite and says "I could not verify the splat quality" is welcome and
honest. We would rather know.

### Known gaps

`tests/integration/` exists but is empty. There is no end-to-end test,
because there is no redistributable sample capture to run one against.
Contributions toward either are especially welcome. Note that sample
footage containing identifiable people needs signed likeness releases
before it can be published.

## Contributor License Agreement

> **Not yet being collected.** The agreements below are final in substance
> but are still awaiting sign-off from our lawyer. Until that lands we
> cannot merge external contributions -- please still open issues, and
> still open pull requests if you want to discuss an approach. Delete this
> note once review is complete.

We use the [Project Harmony](https://www.harmonyagreements.org/) contributor
agreements, version 1.0, unmodified except for the choices every project
using them has to make:

- **[cla/HA-CLA-I.md](cla/HA-CLA-I.md)** -- sign this if you are
  contributing **on your own behalf**.
- **[cla/HA-CLA-E.md](cla/HA-CLA-E.md)** -- sign this if you are
  contributing **on behalf of a company**, which includes the common case
  where your employment contract assigns your work to your employer.

If you are employed and unsure which applies, read your employment
agreement's IP clause, or ask your employer. Getting this wrong is the most
common way a CLA ends up worthless.

### What it actually grants

Worth reading rather than skimming, but in short:

- **You keep your copyright.** Section 2.1(a) is explicit: you retain
  ownership and keep every right you had before signing. This is a license
  to us, not an assignment.
- **We can license your contribution commercially.** We selected Harmony's
  Section 2.3 **Option Five**, which lets us license contributions under
  any terms, including proprietary ones. This is the point of the
  agreement -- it is what makes selling commercial licenses possible, and
  we would rather state it plainly than bury it.
- **But we cannot take it private.** Option Five is conditional: whatever
  else we do, we must *also* keep licensing your contribution under the
  license the project was using when you submitted it. We cannot accept
  your work and then make the project proprietary-only.
- **You waive moral rights** where they apply (Section 2.4), and grant a
  patent license (Section 2.2) limited to what your contribution
  necessarily infringes.
- **You warrant that you have the right to contribute** (Section 3) -- that
  you own the work and that granting these rights does not violate an
  agreement with someone else, typically an employer.

Governing law is Quebec, Canada, where Solipsist Studios Inc. is
incorporated.

### Signing the CLA

1. Read the applicable agreement in full.
2. Email **jeff@solipsist.studio** with the subject "CLA" and tell us which
   one applies and your GitHub username. We will send it for e-signature.
3. Sign. We countersign and keep the executed copy on file.
4. Mention in your pull request that you have signed.

This is deliberately a manual process. We are not running a CLA bot: the
maintained options in that space are thin right now, and your signature is
the record that establishes our right to license the project commercially,
so we would rather hold it ourselves than in a third-party service. You
only sign once; it covers all your future contributions.

### If you do not own the whole contribution

If any part of what you are submitting was written by someone else -- code
copied from another project, a snippet from a colleague, output you are not
sure about -- **email jeff@solipsist.studio before opening the pull
request** and tell us what it is and where it came from. Do not sign the
CLA asserting ownership you do not have.

This is not a formality. Because we sell commercial licenses, third-party
code entering the tree under the wrong terms is a real problem for us and
potentially for you. Flagging it costs a message; discovering it later
costs a lot more.

## Scope

**In scope:** the pipeline scripts in `scripts/`, the orchestrator, the
docs, the conda environment definitions, and packaging or install
ergonomics.

**Out of scope:**

- Bugs in the wrapped tools -- COLMAP, HLOC, Brush, LichtFeld-Studio,
  Sapiens, BiRefNet, Diffuman4D. Report those upstream. We will bump the
  submodule or pin once fixed.
- Debugging your local GPU, CUDA, or conda installation. We are happy to
  fix genuinely misleading documentation, but we cannot triage individual
  machine setups.
- Support for platforms other than Linux with an NVIDIA GPU.

## What to expect from us

This is maintained by a very small team alongside other work. Honestly:

- We aim to respond to issues and PRs within about two weeks.
- There is no SLA, and quiet periods happen. A polite ping on a PR after
  two weeks is welcome, not annoying.
- We may decline changes that are correct but that we do not want to
  maintain. That is not a judgment of the code.

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).
Security issues go to [SECURITY.md](SECURITY.md), not the public tracker.
