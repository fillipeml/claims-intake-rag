# Security policy

## Reporting a vulnerability

Report privately through GitHub's "Report a vulnerability" button on the Security tab. Do not
open a public issue.

Acknowledgement within 72 hours, and a fix or mitigation plan within 14 days for anything
confirmed.

## Scope

This repository reads documents and answers questions from them, so the failures worth
reporting are the ones that make an answer look better supported than it is.

- **An answer that is not grounded in what it cites.** The system refuses when retrieval is
  weak, at a threshold `claims calibrate` measured. A way to get a confident answer out of it
  when the corpus does not contain one is the defect this repository exists to prevent.
- **A retrieval figure that cannot be reproduced.** Every number in the report comes from
  committed fixtures and a fixed seed. If `claims evaluate` gives you a different table on the
  same commit, that is a finding.
- **A statistical error** — an interval computed by the wrong method, a paired comparison run
  as an independent one, a metric whose denominator is not what it says.
- **Real personal data** anywhere in `fixtures/`. See below.
- A path by which a document given to the service leaves it, or is written somewhere the
  caller did not ask for.

## Not in scope

The data. Every person, vehicle, workshop and municipality in `fixtures/` is invented, the
state does not exist, and every CPF is generated with **deliberately incorrect check digits**:
the two digits are computed properly and then replaced, so each one has the right shape, fails
every validator, and cannot collide with a real person's. The claim and policy numbers belong
to no scheme in use.

The rendered pages under `fixtures/pages/` are not committed. They are a deterministic
function of the seed in `scripts/make_fixtures.py`; regenerate them with that script.

The OCR error rates reported here are a property of this corpus, this renderer and this
engine. They are not a claim about how the engine performs on real scans, and the README says
so where they appear.

## Supported versions

The `main` branch.
