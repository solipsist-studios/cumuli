<!--
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
-->

# Security Policy

## Reporting a vulnerability

Email **jeff@solipsist.studio** with "SECURITY" in the subject line, or
use GitHub's private [Report a vulnerability][ghsa] button on the Security
tab. Please do not open a public issue for a security problem.

[ghsa]: https://github.com/solipsist-studios/volumetric-capture-pipeline/security/advisories/new

Include the pipeline stage involved, the exact command you ran, the input
that triggered it, and what happened. A reproducer file is more useful than
a description, but do not send captures containing identifiable people --
describe the file's shape instead and we will construct our own.

## What to expect

This project is maintained by a very small team. Realistic commitments,
which we would rather state honestly than miss:

- **Acknowledgement within 5 business days.** If you have not heard back by
  then, assume the mail went astray and ping the issue tracker with no
  details beyond "sent a security report on <date>".
- **An assessment within 30 days**, including whether we consider it in
  scope and what we intend to do.
- We will credit you in the fix commit and any advisory unless you ask us
  not to.
- We have no bug bounty.

## Scope

This is a set of local command-line scripts. There is no server, no daemon,
no network listener, and no authentication or user data of our own. That
removes most of the usual attack surface, and it means "in scope" is
narrower here than for typical software.

**In scope** -- issues in this repository's own code:

- Command injection or path traversal reachable from filenames, camera
  labels, or config values (the pipeline builds `subprocess` command lines
  and derives output paths from input filenames).
- Unsafe deserialization -- the pipeline reads `.pkl` calibration files, and
  `pickle` is trivially exploitable if the file is untrusted. Treat
  calibration `.pkl` files as executable code and only use ones you
  produced. A report showing a path where we load a pickle we should not is
  in scope.
- Anything in a config or `transforms.json` file that escalates into
  arbitrary code execution.

**Out of scope**, though still worth telling us about as normal bugs:

- Vulnerabilities in the wrapped third-party tools -- COLMAP, HLOC,
  the 4D trainer stack (deps/4d-gaussian-splatting, deps/OMG4),
  Sapiens, BiRefNet, Diffuman4D, ffmpeg, RawTherapee.
  Report those upstream. We will bump pins once a fix is released.
- Vulnerabilities in the conda environments' Python packages. We audit the
  pins in `envs/*.yml` periodically, but those packages are not ours.
  Note that the pipeline feeds arbitrary user-supplied images and video
  through Pillow, ffmpeg, and torch, so a malicious *media file* is the most
  plausible real attack path -- and it lands in those dependencies rather
  than in our code.
- Denial of service through very large inputs. This is a batch tool that
  intentionally consumes all available GPU and disk; there is no resource
  limit to bypass.
- Anything requiring an attacker who already has local shell access as the
  user running the pipeline. At that point they can run the scripts anyway.

## Supported versions

Pre-1.0. Only the latest commit on `main` is supported. There are no
backports to tags.
