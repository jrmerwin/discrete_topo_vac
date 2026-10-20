# Addendum 1 to the Stage-2 Prediction Card — Coarse-Binning Scenario Layer

**Date:** 2026-10-19
**Author:** Jason Merwin (drafted with Claude, Anthropic)
**Status:** ADDENDUM. This document **adds** auxiliary scoreable targets for
one foreseeable binning scenario. It does not amend, reinterpret, or replace
any frozen artifact. The Stage-2 card, its mapping, its falsifiers, and the
protocol's VOID clause (§3.1) remain primary and untouched.

**References (frozen):**
- Prediction card: `prediction_card.json`, SHA-256
  `c5fadd91a7f76ef00fcb7990567c8513d0cf5711cd0dbb06216d615bba59ddeb`,
  git tag `run3-values-v1`, Zenodo 10.5281/zenodo.21464119
  (state/data deposit: 10.5281/zenodo.21464816).
- Protocol: `packet/run3_protocol_prereg_v1.md`, tag `run3-protocol-v1`.

## 1. Motivation and full disclosure

This addendum is written **after** examining the CMS full Run-2 dijet
angular measurement (arXiv:2603.25458; HEPData record ins3136278). That
deposit's highest-mass table (M(JJ) > 7.0 TeV) uses χ bins
[1,3], [3,6], [6,9], [9,12], [12,16]: **χ = 2 is not a bin edge**, so the
frozen card's pre-boundary (1 ≤ χ ≤ 2) and post-boundary (2 < χ ≤ 3)
regions cannot be separated there, and no aggregation can restore the
edge. If the Run-3 publication adopts the same coarsened binning at high
mass, frozen §3.1 scores the affected predictions VOID.

Two disclosures, stated plainly: (i) the decision to add this layer was
prompted by inspecting Run-2 data (a retrodiction analysis of ins3136278,
documented separately), i.e., after data in the same observable family was
known; (ii) the **values** below are not choices — they follow mechanically
from the frozen card's numbers under its own embedded mapping (rules
m4–m6), with one declared convention (§3). A reader can reproduce every
number from `prediction_card.json` in three lines of arithmetic.

We further note, and commit the values regardless: the derived [7–8] TeV
near-cancellation below stands in apparent **tension** with the Run-2
> 7 TeV merged-bin excess (data above NNLO⊗EW in χ ∈ [1,3]; see the
retrodiction analysis). A layer tuned for safety after seeing that data
would not look like this one. It is published because it is what the
frozen card implies, not because it is expected to win.

## 2. Scenario map (precedence order)

- **S1 — fine binning:** the Run-3 publication (or its HEPData deposit)
  resolves χ = 2 and χ = 3 in a scored mass bin → the **original card**
  scores there as frozen. This addendum is dormant for that bin.
- **S2 — Run-2-style coarse binning:** the lowest χ bin is [1, c] with
  2 < c ≤ 3 and no internal edge at 2 → the derived layer of §3 scores
  for that mass bin, **in addition to nothing** (the original quantities
  are VOID there per §3.1; the derived layer is the only scoreable
  content).
- **S3 — conformable binning:** edges at 2 and 3 recoverable by
  aggregation alone → original card after aggregation (already frozen
  §3.1 behavior).
- **S4 — anything else** (e.g., lowest bin wider than [1,3], for which
  the frozen mapping defines no population) → VOID, as frozen. No further
  layers will be added after any relevant Run-3 data exists.

## 3. Derived coarse-grained targets (mechanically forced)

Declared convention (the only choice in this document): the two frozen
wall regions, [1,2] and (2,3], have equal unit χ-width and are fused by
**equal (width) weighting**, the same integration-by-width used
throughout the frozen mapping's arithmetic.

From `prediction_card.json` (bridge population, mapping v1 rules m4–m6):

**[7.0–8.0] TeV** (frozen: p_L = 0.6552, pre = 19, post = 2, soft-wall):
r_L-value = +0.6552; deficit depth = (1 − 2/19) × 0.6552 = 0.5862
(rule m5); merged value = (0.6552 − 0.5862)/2 = **+0.0345** →
**sign +, magnitude class null** (|x| < 0.05). The scoreable statement
under S2 is therefore the near-cancellation itself: the merged χ ∈ [1, c]
residual is **consistent with zero at the null-class threshold**
(|residual| < 0.05).

**> 8.0 TeV** (frozen: p_L = 0.5294, p_R = 0.098, populated):
merged value = (0.5294 + 0.098)/2 = **+0.3137** →
**sign +, magnitude class strong** (0.15–0.35).

Not derivable, and therefore **not claimed** under S2: the Δ* edge-drop
(the frozen mapping defines no population beyond χ = 3), the R_R sign
inversion between the two mass bins, and the §3.3 ordinal trend (its
classes require pre/post separation). Under S2 these are VOID; this
addendum does not substitute weakened versions of them.

## 4. Scoring note

Under S2, each derived element is scored with the frozen two-part scheme
(§3.2/§7.2): the [7–8] element is a hit iff |merged residual| < 0.05
(the null-class claim; sign carries no additional credit there); the > 8
element is a directional hit on sign +, with magnitude class scored
separately against 0.15–0.35. Aggregation across however many elements
score follows frozen §7.2 (unweighted Stouffer) over that reduced set;
VOID elements contribute nothing in either direction.

## 5. Practical request (non-binding)

The decisive variable remains upstream of any scoring rule: whether the
Run-3 dijet angular results resolve χ = 2 above 7 TeV, in the paper or in
its HEPData deposit. We record here the standing request, made through
the appropriate reinterpretation channels, that fine-χ tables at high
mass be included in the Run-3 deposit; the Run-2 [1,3] merge is precisely
what erases the distinction between a pre-boundary pile-up and its
post-boundary response.
