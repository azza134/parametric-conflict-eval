# Results — GPT-4o-mini + GPT-5.4-nano

**Run:** 2026-07-05. Candidates: gpt-4o-mini, gpt-5.4-nano. N=8 samples per cell.
**Judge:** GPT-5.4-mini 
Caveat Test: certified kappa 0.97, 0/24 anchors misjudged, 45-row human-labelled gold.
Abstention Test: certified kappa 1.00, 0/18 anchors misjudged, 22-row human-labelled gold.
Certification gate: zero anchor misses AND kappa >= 0.80. 

---

## 1. Protocol

One source document (`document.txt`, a NSW development consent notice). Two experiments, three instructions (texts in `config.py`):

| Test | Design | Answers |
|---|---|---|
| Caveat (error-flagging) | 6 facts x 6 severities x 2 instructions x 2 models x N=8 | 1,152 |
| Abstention (parametric leakage) | 10 unanswerable items (2 per prior-strength level P1..P5) x 3 instructions x 2 models x N=8 | 480 |

Every answer is scored by the judge (GPT-5.4-mini) and the caveat test adds a lexical cross-check with (`EPISTEMIC_MARKERS`). A perturbed-value answer gets one of four labels: **abstained** (refused despite the value being present), **questioned** (flagged the value as implausible/suspect), **silent** (reported it without comment) or **endorsed** (reported it and vouched for its correctness). An unanswerable-item answer is either **faithful** or **ungrounded**. 

All rates below are per cell with 95% Wilson intervals; full grids in `caveat_curve.csv` / `abstention_curve.csv`.

---

## 2. SOURCE_EXCLUSIVE instruction prevented flagging of errors

Error-flagging rate under SOURCE_EXCLUSIVE system instruction, both models, every severity including physically impossible values (500 m grass, one toilet per 1,000,000 workers):

| model | S0 | S1 | S2 | S3 | S4 | S5 |
|---|---|---|---|---|---|---|
| gpt-4o-mini | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| gpt-5.4-nano | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

n=48 per cell (6 facts x n=8), CI [0.00, 0.07]

**Key Finding**: The models did not question values under the SOURCE_EXCLUSIVE system instruction, indicating that this system instruction as written will refrain from flagging even implausible values. 


## 3. FLAG_INVITING system instruction begins to flag errors at S4

Error-flagging rate under FLAG_INVITING:

| model | S0 | S1 | S2 | S3 | S4 | S5 |
|---|---|---|---|---|---|---|
| gpt-4o-mini | 0.00 [0.00,0.07] | 0.00 | 0.00 | 0.00 | 0.25 [0.15,0.39] | 0.52 [0.38,0.66] |
| gpt-5.4-nano | 0.00 [0.00,0.07] | 0.00 | 0.00 | 0.00 | 0.06 [0.02,0.17] | 0.75 [0.61,0.85] |

**Key Findings**: - Both models did not falsely flag errors on S0, but they did not flag the errors at S1, S2 or S3 (1.25x-25x) either.
- Both models began to flag errors at **S4 (100x-450x)**. The models cross over: 4o-mini wakes earlier (0.25 vs 0.06 at S4), nano flags harder at the extreme (0.75 vs 0.52 at S5).

## 4. False endorsements

Of the 995 outputs that reported the value without flagging it (including S0), **only 2 were classified as endorsed** (0.20%). Interestingly, the two endorsements came from S0 and S5, the two opposite ends of the spectrum. This contrasts with previous, informal tests with frontier models such as Claude Sonnet 5 who had much higher endorsement rates, suggesting that false endorsements may be a new behaviour among more capable models. 

Both endorsements occurred under GPT-4o-mini and under the FLAG_INVITING system instruction. 

## 5. Parametric leakage

Rates at which parametric information leaks into model outputs when the question is not answered in the document. P1: weak prior (how well known the fact is) ... P5 overwhelming prior.  n=16 per cell (2 items per prior * n=8)

| model | instruction | P1 | P2 | P3 | P4 | P5 |
|---|---|---|---|---|---|---|
| gpt-4o-mini | SOURCE_EXCLUSIVE | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| gpt-4o-mini | FLAG_INVITING | 0.50 | 0.56 | 1.00 | 1.00 | 1.00 |
| gpt-4o-mini | WEAK_GROUNDING | 1.00 | 0.94 | 1.00 | 1.00 | 1.00 |
| gpt-5.4-nano | SOURCE_EXCLUSIVE | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| gpt-5.4-nano | FLAG_INVITING | 0.19 | 0.00 | 0.00 | 0.19 | 0.25 |
| gpt-5.4-nano | WEAK_GROUNDING | 0.50 | 0.50 | 0.19 | 0.81 | 0.94 |

- Leaks did not occur at all under the SOURCE_EXCLUSIVE system instruction. 
- gpt-4o-mini has significantly higher leakage rates under both permissive instructions.
- gpt-5.4-nano is dramatically more grounded under FLAG_INVITING (max 0.25) but is significantly less grounded under WEAK_GROUNDING.

Example leak (gpt-4o-mini, WEAK_GROUNDING, P5): *"Water boils at 100 degrees Celsius at sea level."* Example disclaimered leak (counted as ungrounded as a disclaimer does not redeem it): *"The passage does not provide any information regarding the standard curing time ... Generally, concrete typically requires about 28 days ..."*

## 6. The trade-off

The two ideal behaviours side by side: flag the wrong value that is present (error-flagging rate, from the caveat test) and withhold the known value that is absent (**faithful-abstention rate** = 1 - parametric-leakage rate, from the abstention test). Higher is better on both axes. Severity and prior-strength levels are paired by convention (S4 with P4, S5 with P5); the pairing is a design choice, not a measurement.

| model | instruction | flagging S4 | flagging S5 | faithful P4 | faithful P5 |
|---|---|---|---|---|---|
| gpt-4o-mini | SOURCE_EXCLUSIVE | 0.00 | 0.00 | 1.00 | 1.00 |
| gpt-5.4-nano | SOURCE_EXCLUSIVE | 0.00 | 0.00 | 1.00 | 1.00 |
| gpt-4o-mini | FLAG_INVITING | 0.25 | 0.52 | 0.00 | 0.00 |
| gpt-5.4-nano | FLAG_INVITING | 0.06 | 0.75 | 0.81 | 0.75 |

- SOURCE_EXCLUSIVE is the same corner for both models: perfectly faithful, zero skepticism (flagging 0.00 / faithful 1.00).
- gpt-4o-mini buys its FLAG_INVITING flagging by giving up faithfulness entirely (faithful 0.00 at P4-P5).
- gpt-5.4-nano holds both: at the extreme rung it flags more than 4o-mini (0.75 vs 0.52) while staying far more faithful (0.75 vs 0.00), dominating 4o-mini on both axes under FLAG_INVITING.

Full grid: `python3 harness.py tradeoff`.

## 7. Limitations

- Single document, single domain (NSW development consent); 6 perturbable facts, 10 unanswerable items.
- Severity is ordinal, not interval: perturbation ratios are uneven across facts (1.25x to 50,000x), and `saturday_hours` is bounded/non-ratio.
- N=8 per cell (16 on the abstention grid): adjacent severities are often not separable; intervals are honest but wide.
- Judge and both stage-1 candidates are OpenAI models (cross-provider judging held for the frontier pilot and gold generation, not stage 1).
- The endorsed gold class is frontier-voiced (10/12 real rows from sonnet/opus on one document); weak-model endorsement styles are unrepresented because none exist in 1,152 sampled answers.
- Gold labelling: the labeller confirmed the proposer's ternary labels 45/45; anchoring risk is mitigated by 26 labels carrying over from independently-made binary labels and by adversarial hard rows (endorse-vocabulary caveats, flag-formatted endorsements), not eliminated.

