# v2 expansion design — facts, documents, models

Draft for review. Every section header states a decision; **bold** = proposed default, open questions marked ⚑.

## What v2 fixes

One run on a better instrument instead of patching v1 piecemeal:

1. **Clustering** (both external reviews): facts are the experimental unit; 6 facts → ~20 lifts effective n from ~6–16 to ~20+ per cell.
2. **Single document, single domain** (v1 limitation #1).
3. **n=1 frontier** (v1 limitation #5): add a second frontier candidate, retest everyone.
4. **Abstention item paucity**: 2 items per prior level supports no interval (n_eff prints as 2.0).
5. **Prior-strength confound → designed factor**: the S3 bimodality (external-norm facts flag, institution-specific facts don't) becomes a manipulated variable instead of a post-hoc reading.

v1 data is frozen as published. v2 writes fresh results files.

## Documents

**3 documents, 3 genres**: the existing NSW DA determination, plus 2 new genres. Selection constraints: publicly available (transcripts ship un-gitignored), regulated grounded-QA family (the thesis domain), dense in specific numeric/citation-shaped requirements, ~2-5k words.

- **Doc 2: environment protection licence — IN HAND, verified.** NT EPA EPL-188 (City of Darwin landfill), 21pp / 6,641 words, saved as `document2_epl_source.pdf`. Verified facts, all requirement-shaped: leachate level cap 300mm at any time; tyre-stockpile limits table (height <=3 m, width <=5 m, length <=45 m, separation >=15 m, firebreak >=4 m); no tyre storage within 50 m of vegetation over 50mm; daily-cover 300mm; records retained 2 years; NT EPA notification within 14 days; renewal window 90/30 days. Third jurisdiction (NT) for free. Weakness: no numbered-standard citations, so the citation-shaped stratum must come from docs 1 and 3. At ~2.3x v1's length, consider a defined extract if per-request cost bites (prompt cache makes it cheap regardless).
- **Doc 3: liquor licence decision — IN HAND, verified.** ILGA hotel licence decision, The Grove Social House Kingsgrove (Apr 2022), 7pp / 2,770 words, saved as `document3_liquor_source.pdf`. Verified facts: trading hours Mon-Sat 10:00 AM-12:00 AM / Sun 10:00 PM; statutory 6-hour daily closure (external norm, s11A Liquor Act); security ratio 1:100 patrons from 6:00 PM; patron cap 200; terrace cap 20; incident register 3 years; CCTV 30 days / 24 hours; licensee training 6 months; statute-citation-shaped references.

Why one document per genre (not two consents): with 3 documents, documents are the clusters — n=2 within a genre attributes nothing (the same clustering lesson as facts), so a second consent buys only a qualitative stability check while leaving the "singular domain" limitation half-standing. Three genres makes the generality claim categorical. Cost: a third answer-style surface in the judge gold. Rationale for 3 not 5+ stands: every genre grows the gold surface (v1 limitation #4).

## Facts

**20 total: the 6 existing + 14 new**, stratified at authoring on two axes:

| axis | levels | why |
|---|---|---|
| prior anchoring | external-norm vs institution-specific | the S3 bimodality axis, now balanced by design |
| answer shape | plain numeric vs citation-shaped (answer IS a standard/rating name) | the timber_standard/next_bal leak finding, now a designed condition |

Every fact gets a **prior-strength rating (1–5) assigned at authoring time**, before any model sees it. The doc-free probe (banked 2026-07-05) then measures each fact's actual prior empirically; authoring ratings vs probe results is itself a reportable calibration check. Three independent arrivals at this idea (banked note, review #1, review #2) — it unifies the caveat and abstention experiments under one construct.

Ladders: unchanged design — S0 control + S1–S5 ordinal severity, `validate_ladders()` now asserts per-replacement. Bounded/non-ratio facts permitted but tagged (v1 limitation #2 stands, disclosed not solved).

**Prior-measurement commitment (binding at analysis time):** authored 1–5 ratings are design scaffolding only — they select items for spread and get audited by the probe; they NEVER appear in results.md except inside the calibration audit, where they are the thing being tested. Every published prior variable is the probe's measured knows-rate: continuous (per-model where the question is about a model's own prior), and where a table needs discrete rows, FIXED bins of the measured rate (edges 0.25/0.50/0.75, never moved to fit the sample) — reproducible from prior_probe_results.jsonl — not the authored bins. Even occupancy per bin is an AUTHORING target, not a binning rule: the probe summary prints per-bin occupancy and flags thin bins so items get authored into gaps before the main run; a lopsided item set shows up as lopsided bins rather than being hidden by rank-based cuts. Caveat on the measurement itself: the probe measures THIS model panel at THESE effort settings; "measured prior" is panel-relative, not a property of the world (limitations sentence).

## Directional contrast arm (candidate, post-probe)

Sub-1x perturbations, imported from ClashEval's bidirectional scheme (their numeric grid: 0.1-10x, arXiv:2404.10198) onto speech-act metrics. ClashEval showed adoption resistance grows with departure in BOTH directions but measured adoption only; nobody has measured what models SAY about downward errors. Hypothesis: downward values stay physically plausible while becoming normatively wrong (asbestos at 30cm, 40cm firebreak, 36-minute liquor closure), so flagging them requires knowing the norm — predicting (a) down-flagging << up-flagging at matched |log ratio|, (b) down-flagging correlates with the measured prior more strongly than up-flagging (surface absurdity unavailable), (c) the dangerous failure is SILENT adoption of plausible-but-lax safety numbers, invisible to adoption metrics. Scope: mirrored down-ladders for ~6 facts chosen AFTER the probe (want facts models demonstrably know), matched |log ratio| to the up-steps, extending below ClashEval's 0.1x floor where physically meaningful; `direction` tag per step, polarity tag per fact (for a maximum, "down" = absurdly strict, not dangerously lax); ratio convention = fold-change + direction (the ascending-ratio test pin needs relaxing). ~30 steps, +~1,440 candidate calls, +$10-15 incl. judging. Judge gate: all existing extreme anchors are up-direction; down-direction answer styles must enter the gold before numbers are trusted.

## Abstention items

**4 items per prior level (2 → 4, i.e. 10 → 20 items)** — DONE. Citation-shaped answers deliberately span P1–P3 (`as_bins`/`next_bal`, `as_flammable`/`timber_standard`, `as_parking`); no P4/P5 citation items exist because no standard number is famous enough to carry a strong prior — a bound of the world, not the design. P3's two new items both sit on the consent doc (pool fencing and parking are building-code-natural questions; a forced liquor/EPL P3 would be a worse defect than the imbalance).

## Models

**3 candidates: the v1 three. Second frontier model RAIN-CHECKED (2026-07-09):** gpt-5.6-terra turned out to be account-gated ("limited preview... coming weeks" per the API's 404 on the probe smoke test) and the gpt-5.5 fallback was deferred by choice; the n=1 frontier limitation therefore persists in v2 and stays disclosed. Adding a fourth candidate later is cheap by design: resumable runs mean new cells only, and the probe backfills the same way. Effort convention below already handles any future model correctly (only gpt-5.4* is pinned). Effort convention — the principle: **candidates run deployment-realistic; instruments stay frozen at their certified config.**

| model | effort | why |
|---|---|---|
| claude-sonnet-5 | vendor default (adaptive thinking, effort "high") | deployment-realistic + v1 parity |
| gpt-5.6-terra | vendor default (no reasoning param passed) | deployment-realistic; no v1 constraint; avoids "you crippled it" in the null branch |
| gpt-5.4-nano | pinned "low" (above its "none" default) | v1 parity — its published numbers were generated at low; deviation from realism disclosed |
| gpt-4o-mini | n/a (no reasoning support) | — |
| gpt-5.4-mini (judge) | pinned "low" | certified at this config (kappa 0.98/0.94, 0/30 anchors attest to the judge-as-configured); changing it voids certification |

The comparison is deployment-realistic, NOT effort-matched (and effort labels aren't calibrated across vendors, so label-matching would be false comfort): Sonnet sits near the top of its scale, nano near the bottom. Part of Sonnet's capability-artifact advantage could be effort allocation — disclose in v2 results. Ablation branch if challenged (or if Terra nulls on false endorsement): a Sonnet arm at output_config effort "low" — DECISION DEFERRED (fifth arm ~$8-12 full grid, or key cells only: FLAG_INVITING + 2x2 at S1/S3/S5). Other sampling params at API defaults (Sonnet 5 rejects non-default temperature/top_p anyway); snapshot IDs captured at run time (v1 disclosure debt). Describe as "frontier-generation balanced tier" not bare "frontier" — Sol is the 5.6 flagship; if Terra shows false endorsements the claim generalises cross-lab anyway, if not "does Sol?" is a cheap follow-up.

Caveat carried from review #1: the judge is GPT-5.4-mini. Judging a same-family frontier model sharpens the same-provider concern — mitigation is the judge-gold spot-check on the new model's answers (mandatory anyway) ⚑ plus optionally a one-cell Anthropic-judge cross-check.

## Reps and cost

**3 reps** (`N_PER_CELL` 4 → 3). At ICC ≈ 0.5, 20 facts × 3 reps ≈ n_eff 30 per cell vs v1's ~11.

Rough call count: caveat 4 models × 4 instructions × 20 facts × 6 severities × 3 reps = **5,760** + abstention 4 × 4 × 20 × 3 = **960**, plus equal judge calls, plus the doc-free prior probe (~20 facts × 4 models × 3 reps ≈ 240). Ballpark **$40–90** depending on the frontier pick. Dry-run planner prints exact plan before spend.

## Judge

Gate, not afterthought — the long pole alongside fact authoring:

1. Pilot one model × one new document; sample answers into the gold (labelled in-place via the CLI helper).
2. Expand gold to cover: both new documents, the new genre's answer style, the new frontier model's answer style, citation-shaped caveat facts.
3. Recertify (kappa threshold + zero-tolerance anchors) before any v2 number is read.

## Harness changes

- Documents registry in config; each fact carries a `doc` reference; `step_doc`/wave batching/cache-warming key on the document (prompt cache is per-document now).
- `document` field in result rows; fresh `caveat_results_v2.jsonl` / `abstention_results_v2.jsonl`.
- Fact and item names stay hyphen-free (test-pinned) and must be unique across documents.
- All keyless tests extended to the multi-document paths, especially resume.

## Order of operations

1. Sign off this design (⚑ items resolved)
2. Select docs 2–3; author 14 facts + 10 abstention items with strata + prior ratings
3. Config restructure + tests (no API)
4. Doc-free prior probe (small spend, validates ratings)
5. Dry-run plan → pilot (1 model × 1 new doc)
6. Judge gold expansion + recertification ← gate
7. Full v2 run (resumable, two-wave, concurrent judge)
8. Analysis: existing `vectors`/ICC machinery reads v2 files as-is
