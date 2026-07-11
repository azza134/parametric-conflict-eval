# Results

**Models tested:** gpt-4o-mini (gpt-4o-mini-2024-07-18), gpt-5.4-nano (gpt-5.4-nano-2026-03-17), claude-sonnet-5 (resolved alias), N=3 per cell; the seeded consent cells are N=8 (pooling rules in the pre-registered plan). (legacy vs budget vs frontier)
**Judge:** GPT-5.4-mini (gpt-5.4-mini-2026-03-17): caveat stance kappa 0.98 / corroboration 0.91, 0/30 anchors misjudged, 138-row human-labelled gold; abstention kappa 0.97, 0/54 anchors misjudged, 80-row human-labelled gold.
**Certification gate**: zero anchor misses AND kappa >= 0.80.

The v1 single-document study (sections 1-7 of this file until 2026-07-11) lives verbatim in `archive/results-v1.md`; sections 2-13 supersede it wherever they overlap.

## 2. Key Terms

Definitions of the outcomes are as follows:

**1. Contradiction Sensitivity:** Rate at which the document catches a wrong value (severity 1 to 5, perturbed values)
**2. Clean Specificity:** Rate at which the model does not question or decline the document value at severity 0 (unperturbed)
**3. Absence Faithfulness:** Rate at which the model abstains from supplying an answer if the answer to a question about the document is removed. 
**4. False Endorsement:** Rate at which the model actively endorses perturbed values (S1 to S5)
**5. False Corroboration:** Rate at which the model actively endorses perturbed values (S1 to S5) AND cites external sources to justify it.
**6. Selective Success:** If the model does all three of the following, simulating ideal actions in each RAG scenario, it scores one point, with 24 repetitions for 24 facts

- reporting S0 correctly with no fuss
- reporting S3-S5 with what the document says AND raises a concern for plausibility
- reporting that the answer is not in a document for an unanswerable question for each fact,

System instructions are shortened as such:
SE = **SOURCE_EXCLUSIVE**
FI = **FLAG_INVITING**
WG = **WEAK_GROUNDING**
SE+FI = **SOURCE_EXCLUSIVE_FLAG_INVITING**
AUDIT = **SELECTIVE_AUDIT**

## 3. Contradiction sensitivity


| model           | SE               | FI               | WG               | SE+FI            | AUDIT            |
| --------------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- |
| gpt-4o-mini     | 0.00 [0.00,0.01] | 0.15 [0.12,0.18] | 0.00 [0.00,0.01] | 0.00 [0.00,0.01] | 0.00 [0.00,0.01] |
| gpt-5.4-nano    | 0.00 [0.00,0.01] | 0.15 [0.12,0.18] | 0.00 [0.00,0.01] | 0.04 [0.03,0.06] | 0.00 [0.00,0.01] |
| claude-sonnet-5 | 0.00 [0.00,0.01] | 0.70 [0.65,0.75] | 0.18 [0.15,0.23] | 0.55 [0.50,0.60] | 0.37 [0.32,0.42] |


**Key Findings:** 

- None of the models caught any errors at all on SE, and only Sonnet 5 caught any errors on WG and AUDIT, which were at lower rates than FI and SE+FI, implying that explicit invitation in the system instruction to flag errors raises the likelihood of the models trying to catch errors.
- Early indication that more advanced models like Sonnet 5 are more likely to catch errors, although this is based on n=1 (frontier models) and Sonnet 5 was the only model to catch errors on WG and AUDIT.

## 4. Clean specificity


| model           | SE               | FI               | WG               | SE+FI            | AUDIT            |
| --------------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- |
| gpt-4o-mini     | 0.94 [0.88,0.97] | 1.00 [0.96,1.00] | 1.00 [0.96,1.00] | 0.97 [0.92,0.99] | 0.97 [0.90,0.99] |
| gpt-5.4-nano    | 0.95 [0.89,0.98] | 1.00 [0.96,1.00] | 1.00 [0.96,1.00] | 0.93 [0.87,0.97] | 1.00 [0.95,1.00] |
| claude-sonnet-5 | 1.00 [0.95,1.00] | 0.85 [0.75,0.91] | 1.00 [0.95,1.00] | 0.99 [0.93,1.00] | 1.00 [0.95,1.00] |


**Key Findings:**

- All the models had a similar performance across all instructions in abstaining from raising doubts about the unperturbed value, with the exception of 0.85 on Sonnet 5 (FI), an early indication based on n=1 frontier models that more advanced models are more sensitive to the FI instruction when deciding whether to raise doubts and catch errors.

## 5. O3 Absence faithfulness


| model           | SE               | FI               | WG               | SE+FI            | AUDIT            |
| --------------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- |
| gpt-4o-mini     | 0.86 [0.76,0.92] | 0.60 [0.48,0.70] | 0.46 [0.35,0.57] | 0.81 [0.70,0.88] | 0.83 [0.73,0.90] |
| gpt-5.4-nano    | 0.75 [0.64,0.84] | 0.71 [0.59,0.80] | 0.53 [0.41,0.64] | 0.81 [0.70,0.88] | 0.75 [0.64,0.84] |
| claude-sonnet-5 | 0.96 [0.88,0.99] | 0.71 [0.59,0.80] | 0.81 [0.70,0.88] | 0.96 [0.88,0.99] | 0.94 [0.87,0.98] |


**Key Findings:** 

- Sonnet 5 recorded the highest faithfulness rates when the answer was not in the document, an early indication based on n=1 frontier models that more advanced models are more likely to stay faithful. 
- SE and SE+FI had the highest faithfulness rates across the models, closely trailed by AUDIT, indicating that the system instructions that more heavily suppressed the use of parametric knowledge were more effective at faithfulness.

## 6. O4 False endorsement and O5 false corroboration

Raw endorsement (O4) on perturbed values S1-S5: rate with Wilson 95% interval.


| model           | SE               | FI               | WG               | SE+FI            | AUDIT            |
| --------------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- |
| gpt-4o-mini     | 0.00 [0.00,0.01] | 0.00 [0.00,0.01] | 0.00 [0.00,0.01] | 0.00 [0.00,0.01] | 0.00 [0.00,0.01] |
| gpt-5.4-nano    | 0.00 [0.00,0.01] | 0.00 [0.00,0.01] | 0.00 [0.00,0.01] | 0.00 [0.00,0.01] | 0.00 [0.00,0.01] |
| claude-sonnet-5 | 0.00 [0.00,0.01] | 0.24 [0.20,0.29] | 0.00 [0.00,0.01] | 0.01 [0.00,0.02] | 0.00 [0.00,0.01] |




Key: False endorsement / false corroboration 


| model           | SE          | FI          | WG          | SE+FI       | AUDIT       |
| --------------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| gpt-4o-mini     | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| gpt-5.4-nano    | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| claude-sonnet-5 | 0.00 / 0.00 | 0.24 / 0.20 | 0.00 / 0.00 | 0.01 / 0.00 | 0.00 / 0.00 |


**Key Findings:** 

- 

## 7. Selective success


| model           | SE   | FI    | WG   | SE+FI | AUDIT |
| --------------- | ---- | ----- | ---- | ----- | ----- |
| gpt-4o-mini     | 0/24 | 2/24  | 0/24 | 0/24  | 0/24  |
| gpt-5.4-nano    | 0/24 | 1/24  | 0/24 | 0/24  | 0/24  |
| claude-sonnet-5 | 0/24 | 16/24 | 5/24 | 23/24 | 16/24 |


**Key Findings:** 

- Sonnet 5 paired with SE+FI was able to achieve the highest score of ~96%, a far lead over the ~67% score on Sonnet 5 with FI or AUDIT. 
- GPT-4o-mini and GPT-5.4-nano, the legacy and budget models, were unable to adequately address all three RAG scenarios at once under any of the system instructions.
- SE was unable to score any points because it would never flag errors, failing the RAG scenarios where the document has perturbed values or the answer is not in the document. 

## 8. The 2x2 factorial

**O1 contradiction sensitivity (questioned | S1–S5)**


| model           | source-exclusivity main              | flag-invitation main                  | interaction                           |
| --------------- | ------------------------------------ | ------------------------------------- | ------------------------------------- |
| gpt-4o-mini     | -0.07 [-0.10,-0.05], p<0.001 (1/18+) | +0.07 [+0.04,+0.10], p<0.001 (17/17+) | -0.14 [-0.19,-0.09], p<0.001 (1/17+)  |
| gpt-5.4-nano    | -0.05 [-0.07,-0.04], p<0.001 (0/17+) | +0.09 [+0.06,+0.12], p<0.001 (18/18+) | -0.10 [-0.13,-0.07], p<0.001 (0/17+)  |
| claude-sonnet-5 | -0.17 [-0.20,-0.13], p<0.001 (1/24+) | +0.54 [+0.48,+0.59], p<0.001 (24/24+) | +0.03 [-0.05,+0.11], p=0.523 (13/22+) |


**O3 absence faithfulness**


| model           | source-exclusivity main               | flag-invitation main                  | interaction                          |
| --------------- | ------------------------------------- | ------------------------------------- | ------------------------------------ |
| gpt-4o-mini     | +0.31 [+0.15,+0.48], p=0.035 (12/15+) | +0.04 [-0.06,+0.13], p=0.227 (8/11+)  | -0.19 [-0.35,-0.06], p=0.146 (3/12+) |
| gpt-5.4-nano    | +0.16 [+0.06,+0.27], p=0.039 (10/12+) | +0.12 [+0.03,+0.21], p=0.092 (10/13+) | -0.12 [-0.26,+0.01], p=0.227 (3/11+) |
| claude-sonnet-5 | +0.20 [+0.10,+0.31], p<0.001 (14/15+) | -0.05 [-0.13,+0.03], p=0.267 (4/13+)  | +0.10 [-0.07,+0.26], p=0.267 (9/13+) |


**O4 false endorsement (endorsed | S1–S5)**


| model           | source-exclusivity main              | flag-invitation main                  | interaction                          |
| --------------- | ------------------------------------ | ------------------------------------- | ------------------------------------ |
| gpt-4o-mini     | +0.00, p=1.000                       | +0.00, p=1.000                        | +0.00, p=1.000                       |
| gpt-5.4-nano    | -0.00 [-0.00,+0.00], p=1.000 (0/1+)  | -0.00 [-0.00,+0.00], p=1.000 (0/1+)   | +0.00 [+0.00,+0.01], p=1.000 (1/1+)  |
| claude-sonnet-5 | -0.12 [-0.15,-0.09], p<0.001 (0/22+) | +0.13 [+0.10,+0.16], p<0.001 (23/23+) | -0.24 [-0.29,-0.18], p<0.001 (0/22+) |


The O4 cell rates behind claude-sonnet-5's interaction: WEAK_GROUNDING 0.00, FLAG_INVITING 0.24, SOURCE_EXCLUSIVE 0.00, SOURCE_EXCLUSIVE_FLAG_INVITING 0.01 — false endorsement exists only in the invitation-without-exclusivity cell.

**Key Findings:** *(Andrew — to write)*

## 9. SELECTIVE_AUDIT existence test


| model           | best 2x2 cell                  | best-cell O1 / O3 / selective | SELECTIVE_AUDIT O1 / O3 / selective |
| --------------- | ------------------------------ | ----------------------------- | ----------------------------------- |
| gpt-4o-mini     | FLAG_INVITING                  | 0.15 / 0.60 / 2/24            | 0.00 / 0.83 / 0/24                  |
| gpt-5.4-nano    | FLAG_INVITING                  | 0.15 / 0.71 / 1/24            | 0.00 / 0.75 / 0/24                  |
| claude-sonnet-5 | SOURCE_EXCLUSIVE_FLAG_INVITING | 0.55 / 0.96 / 23/24           | 0.37 / 0.94 / 16/24                 |


**Key Findings:** *(Andrew — to write)*

## 10. Severity contrasts


| model           | instruction                    | Q S0 | Q S1-2 | Q S3-5 | E S0 | E S1-2 | E S3-5 |
| --------------- | ------------------------------ | ---- | ------ | ------ | ---- | ------ | ------ |
| gpt-4o-mini     | SOURCE_EXCLUSIVE               | 0.00 | 0.00   | 0.00   | 0.00 | 0.00   | 0.00   |
| gpt-4o-mini     | FLAG_INVITING                  | 0.00 | 0.00   | 0.24   | 0.00 | 0.00   | 0.00   |
| gpt-4o-mini     | WEAK_GROUNDING                 | 0.00 | 0.00   | 0.00   | 0.00 | 0.00   | 0.00   |
| gpt-4o-mini     | SOURCE_EXCLUSIVE_FLAG_INVITING | 0.00 | 0.00   | 0.00   | 0.00 | 0.00   | 0.00   |
| gpt-4o-mini     | SELECTIVE_AUDIT                | 0.00 | 0.00   | 0.00   | 0.00 | 0.00   | 0.00   |
| gpt-5.4-nano    | SOURCE_EXCLUSIVE               | 0.00 | 0.00   | 0.00   | 0.00 | 0.00   | 0.00   |
| gpt-5.4-nano    | FLAG_INVITING                  | 0.00 | 0.00   | 0.25   | 0.00 | 0.00   | 0.00   |
| gpt-5.4-nano    | WEAK_GROUNDING                 | 0.00 | 0.00   | 0.00   | 0.00 | 0.00   | 0.00   |
| gpt-5.4-nano    | SOURCE_EXCLUSIVE_FLAG_INVITING | 0.00 | 0.00   | 0.07   | 0.00 | 0.00   | 0.00   |
| gpt-5.4-nano    | SELECTIVE_AUDIT                | 0.00 | 0.00   | 0.00   | 0.00 | 0.00   | 0.00   |
| claude-sonnet-5 | SOURCE_EXCLUSIVE               | 0.00 | 0.00   | 0.00   | 0.00 | 0.00   | 0.00   |
| claude-sonnet-5 | FLAG_INVITING                  | 0.06 | 0.35   | 0.94   | 0.83 | 0.56   | 0.03   |
| claude-sonnet-5 | WEAK_GROUNDING                 | 0.00 | 0.00   | 0.31   | 0.00 | 0.00   | 0.00   |
| claude-sonnet-5 | SOURCE_EXCLUSIVE_FLAG_INVITING | 0.01 | 0.09   | 0.86   | 0.00 | 0.01   | 0.00   |
| claude-sonnet-5 | SELECTIVE_AUDIT                | 0.00 | 0.00   | 0.61   | 0.00 | 0.00   | 0.00   |


**Key Findings:** *(Andrew — to write)*

## 11. Per-document effects


| model           | instruction                    | O1 consent | O1 epl | O1 liquor | O3 consent | O3 epl | O3 liquor |
| --------------- | ------------------------------ | ---------- | ------ | --------- | ---------- | ------ | --------- |
| gpt-4o-mini     | SOURCE_EXCLUSIVE               | 0.00       | 0.00   | 0.00      | 0.96       | 0.71   | 0.89      |
| gpt-4o-mini     | FLAG_INVITING                  | 0.14       | 0.14   | 0.16      | 0.62       | 0.52   | 0.63      |
| gpt-4o-mini     | WEAK_GROUNDING                 | 0.00       | 0.00   | 0.01      | 0.67       | 0.38   | 0.33      |
| gpt-4o-mini     | SOURCE_EXCLUSIVE_FLAG_INVITING | 0.00       | 0.00   | 0.00      | 0.88       | 0.62   | 0.89      |
| gpt-4o-mini     | SELECTIVE_AUDIT                | 0.00       | 0.00   | 0.00      | 0.92       | 0.62   | 0.93      |
| gpt-5.4-nano    | SOURCE_EXCLUSIVE               | 0.00       | 0.00   | 0.00      | 0.96       | 0.67   | 0.63      |
| gpt-5.4-nano    | FLAG_INVITING                  | 0.17       | 0.17   | 0.10      | 0.96       | 0.67   | 0.52      |
| gpt-5.4-nano    | WEAK_GROUNDING                 | 0.00       | 0.01   | 0.00      | 0.67       | 0.62   | 0.33      |
| gpt-5.4-nano    | SOURCE_EXCLUSIVE_FLAG_INVITING | 0.06       | 0.05   | 0.00      | 0.96       | 0.71   | 0.74      |
| gpt-5.4-nano    | SELECTIVE_AUDIT                | 0.00       | 0.00   | 0.00      | 1.00       | 0.67   | 0.59      |
| claude-sonnet-5 | SOURCE_EXCLUSIVE               | 0.00       | 0.00   | 0.00      | 1.00       | 0.86   | 1.00      |
| claude-sonnet-5 | FLAG_INVITING                  | 0.67       | 0.69   | 0.75      | 0.54       | 0.81   | 0.78      |
| claude-sonnet-5 | WEAK_GROUNDING                 | 0.25       | 0.20   | 0.11      | 1.00       | 0.71   | 0.70      |
| claude-sonnet-5 | SOURCE_EXCLUSIVE_FLAG_INVITING | 0.57       | 0.57   | 0.52      | 1.00       | 0.86   | 1.00      |
| claude-sonnet-5 | SELECTIVE_AUDIT                | 0.40       | 0.38   | 0.33      | 1.00       | 0.86   | 0.96      |


**Key Findings:** *(Andrew — to write)*

## 12. Provenance, sensitivity and exclusions


| model        | instruction                    | seeded Q / E (n)    | fresh Q / E (n)     |
| ------------ | ------------------------------ | ------------------- | ------------------- |
| gpt-4o-mini  | SOURCE_EXCLUSIVE               | 0.000 / 0.000 (240) | 0.000 / 0.000 (270) |
| gpt-4o-mini  | FLAG_INVITING                  | 0.154 / 0.000 (240) | 0.137 / 0.000 (270) |
| gpt-4o-mini  | WEAK_GROUNDING                 | 0.000 / 0.000 (240) | 0.004 / 0.000 (270) |
| gpt-4o-mini  | SOURCE_EXCLUSIVE_FLAG_INVITING | 0.004 / 0.000 (240) | 0.000 / 0.000 (270) |
| gpt-5.4-nano | SOURCE_EXCLUSIVE               | 0.000 / 0.000 (240) | 0.000 / 0.000 (270) |
| gpt-5.4-nano | FLAG_INVITING                  | 0.163 / 0.000 (240) | 0.137 / 0.000 (270) |
| gpt-5.4-nano | WEAK_GROUNDING                 | 0.000 / 0.000 (240) | 0.004 / 0.004 (270) |
| gpt-5.4-nano | SOURCE_EXCLUSIVE_FLAG_INVITING | 0.046 / 0.000 (240) | 0.037 / 0.000 (270) |


## 13. Parametric leakage on the v2 grid (exploratory)

Ungrounded-answer rate on the 24 unanswerable items, by measured prior bin (each model's own doc-free recall of the item's answer, from the snapshot-aware probe; fixed bin edges 0.25/0.50/0.75). Exploratory per the pre-registered plan: the unanswerable-item sweep is not one of the six pre-registered outcomes. Per-cell n is uneven (9-44) because bin occupancy is per model and the consent cells carry seeded N=8 reps; full grid in `abstention_curve.csv`.


| model           | instruction                    | 0.00-0.25        | 0.25-0.50        | 0.50-0.75        | 0.75-1.00        |
| --------------- | ------------------------------ | ---------------- | ---------------- | ---------------- | ---------------- |
| gpt-4o-mini     | SOURCE_EXCLUSIVE               | 0.00 [0.00,0.30] | 0.00 [0.00,0.20] | 0.00 [0.00,0.14] | 0.00 [0.00,0.08] |
| gpt-4o-mini     | FLAG_INVITING                  | 0.00 [0.00,0.30] | 0.00 [0.00,0.20] | 0.08 [0.02,0.26] | 0.68 [0.53,0.80] |
| gpt-4o-mini     | WEAK_GROUNDING                 | 0.33 [0.12,0.65] | 0.80 [0.55,0.93] | 0.62 [0.43,0.79] | 0.98 [0.88,1.00] |
| gpt-4o-mini     | SOURCE_EXCLUSIVE_FLAG_INVITING | 0.00 [0.00,0.30] | 0.00 [0.00,0.20] | 0.00 [0.00,0.14] | 0.00 [0.00,0.08] |
| gpt-4o-mini     | SELECTIVE_AUDIT                | 0.00 [0.00,0.30] | 0.00 [0.00,0.20] | 0.00 [0.00,0.14] | 0.00 [0.00,0.14] |
| gpt-5.4-nano    | SOURCE_EXCLUSIVE               | 0.00 [0.00,0.30] | 0.00 [0.00,0.20] | 0.00 [0.00,0.14] | 0.00 [0.00,0.08] |
| gpt-5.4-nano    | FLAG_INVITING                  | 0.00 [0.00,0.30] | 0.00 [0.00,0.20] | 0.00 [0.00,0.14] | 0.16 [0.08,0.29] |
| gpt-5.4-nano    | WEAK_GROUNDING                 | 0.00 [0.00,0.30] | 0.20 [0.07,0.45] | 0.21 [0.09,0.40] | 0.70 [0.56,0.82] |
| gpt-5.4-nano    | SOURCE_EXCLUSIVE_FLAG_INVITING | 0.00 [0.00,0.30] | 0.00 [0.00,0.20] | 0.00 [0.00,0.14] | 0.00 [0.00,0.08] |
| gpt-5.4-nano    | SELECTIVE_AUDIT                | 0.00 [0.00,0.30] | 0.00 [0.00,0.20] | 0.00 [0.00,0.14] | 0.04 [0.01,0.20] |
| claude-sonnet-5 | SOURCE_EXCLUSIVE               | 0.00 [0.00,0.30] | 0.00 [0.00,0.20] | 0.00 [0.00,0.14] | 0.00 [0.00,0.14] |
| claude-sonnet-5 | FLAG_INVITING                  | 0.00 [0.00,0.30] | 0.00 [0.00,0.20] | 0.08 [0.02,0.26] | 0.38 [0.21,0.57] |
| claude-sonnet-5 | WEAK_GROUNDING                 | 0.00 [0.00,0.30] | 0.13 [0.04,0.38] | 0.00 [0.00,0.14] | 0.58 [0.39,0.76] |
| claude-sonnet-5 | SOURCE_EXCLUSIVE_FLAG_INVITING | 0.00 [0.00,0.30] | 0.00 [0.00,0.20] | 0.00 [0.00,0.14] | 0.08 [0.02,0.26] |
| claude-sonnet-5 | SELECTIVE_AUDIT                | 0.00 [0.00,0.30] | 0.00 [0.00,0.20] | 0.00 [0.00,0.14] | 0.00 [0.00,0.14] |


**Key Findings:** *(Andrew — to write)*