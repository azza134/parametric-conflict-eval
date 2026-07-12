# Results

**Models tested:** gpt-4o-mini (gpt-4o-mini-2024-07-18), gpt-5.4-nano (gpt-5.4-nano-2026-03-17), claude-sonnet-5 (resolved alias), N=3 per cell; the seeded consent cells are N=8 (pooling rules in the pre-registered plan). (legacy vs budget vs frontier)
**Judge:** GPT-5.4-mini (gpt-5.4-mini-2026-03-17): caveat stance kappa 0.98 / corroboration 0.91, 0/30 anchors misjudged, 138-row human-labelled gold; abstention kappa 0.97, 0/54 anchors misjudged, 80-row human-labelled gold.
**Certification gate**: zero anchor misses AND kappa >= 0.80.

The v1 single-document study (sections 1-7 of this file until 2026-07-11) lives verbatim in `archive/results-v1.md`; sections 1-11 supersede it wherever they overlap.

## 1. Key Terms

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

## 2. Contradiction sensitivity


| model           | SE               | FI               | WG               | SE+FI            | AUDIT            |
| --------------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- |
| gpt-4o-mini     | 0.00 [0.00,0.01] | 0.15 [0.12,0.18] | 0.00 [0.00,0.01] | 0.00 [0.00,0.01] | 0.00 [0.00,0.01] |
| gpt-5.4-nano    | 0.00 [0.00,0.01] | 0.15 [0.12,0.18] | 0.00 [0.00,0.01] | 0.04 [0.03,0.06] | 0.00 [0.00,0.01] |
| claude-sonnet-5 | 0.00 [0.00,0.01] | 0.70 [0.65,0.75] | 0.18 [0.15,0.23] | 0.55 [0.50,0.60] | 0.37 [0.32,0.42] |


**Key Findings:** 

- None of the models caught any errors at all on SE, and only Sonnet 5 caught any errors on WG and AUDIT, which were at lower rates than FI and SE+FI, implying that explicit invitation in the system instruction to flag errors raises the likelihood of the models trying to catch errors.
- Early indication that more advanced models like Sonnet 5 are more likely to catch errors, although this is based on n=1 (frontier models) and Sonnet 5 was the only model to catch errors on WG and AUDIT.

## 3. Clean specificity


| model           | SE               | FI               | WG               | SE+FI            | AUDIT            |
| --------------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- |
| gpt-4o-mini     | 0.94 [0.88,0.97] | 1.00 [0.96,1.00] | 1.00 [0.96,1.00] | 0.97 [0.92,0.99] | 0.97 [0.90,0.99] |
| gpt-5.4-nano    | 0.95 [0.89,0.98] | 1.00 [0.96,1.00] | 1.00 [0.96,1.00] | 0.93 [0.87,0.97] | 1.00 [0.95,1.00] |
| claude-sonnet-5 | 1.00 [0.95,1.00] | 0.85 [0.75,0.91] | 1.00 [0.95,1.00] | 0.99 [0.93,1.00] | 1.00 [0.95,1.00] |


**Key Findings:**

- All the models had a similar performance across all instructions in abstaining from raising doubts about the unperturbed value, with the exception of 0.85 on Sonnet 5 (FI), an early indication based on n=1 frontier models that more advanced models are more sensitive to the FI instruction when deciding whether to raise doubts and catch errors.

## 4. Absence faithfulness


| model           | SE               | FI               | WG               | SE+FI            | AUDIT            |
| --------------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- |
| gpt-4o-mini     | 0.86 [0.76,0.92] | 0.60 [0.48,0.70] | 0.46 [0.35,0.57] | 0.81 [0.70,0.88] | 0.83 [0.73,0.90] |
| gpt-5.4-nano    | 0.75 [0.64,0.84] | 0.71 [0.59,0.80] | 0.53 [0.41,0.64] | 0.81 [0.70,0.88] | 0.75 [0.64,0.84] |
| claude-sonnet-5 | 0.96 [0.88,0.99] | 0.71 [0.59,0.80] | 0.81 [0.70,0.88] | 0.96 [0.88,0.99] | 0.94 [0.87,0.98] |


**Key Findings:** 

- Sonnet 5 recorded the highest faithfulness rates when the answer was not in the document, an early indication based on n=1 frontier models that more advanced models are more likely to stay faithful. 
- SE and SE+FI had the highest faithfulness rates across the models, closely trailed by AUDIT, indicating that the system instructions that more heavily suppressed the use of parametric knowledge were more effective at faithfulness.

## 5. False endorsements and corroborations

Raw endorsement on perturbed values S1-S5: rate with Wilson 95% interval. (more detail in Section 9)


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


Sonnet 5 x FI endorsements per severity


| severity | endorsed | rate [Wilson 95%] |
| -------- | -------- | ----------------- |
| S0       | 60/72    | 0.83 [0.73,0.90]  |
| S1       | 56/72    | 0.78 [0.67,0.86]  |
| S2       | 25/72    | 0.35 [0.25,0.46]  |
| S3       | 7/72     | 0.10 [0.05,0.19]  |
| S4       | 0/72     | 0.00 [0.00,0.05]  |
| S5       | 0/72     | 0.00 [0.00,0.05]  |


**Key Findings:** 

- None of the GPT models (legacy and budget) recorded any endorsements at all, an early indication that endorsements are a new behaviour observed in frontier models like Sonnet 5 based on n=1 frontier models.
- Dangerously, 24% of answers from Sonnet 5 on the FI instruction were endorsements that were on perturbed values and 20% of the answers were endorsements that were corroborated with external standards that were fabricated in order to meet consistency, a failure of both RAG grounding and parametric knowledge accuracy.

## 6. Selective success


| model           | SE   | FI    | WG   | SE+FI | AUDIT |
| --------------- | ---- | ----- | ---- | ----- | ----- |
| gpt-4o-mini     | 0/24 | 2/24  | 0/24 | 0/24  | 0/24  |
| gpt-5.4-nano    | 0/24 | 1/24  | 0/24 | 0/24  | 0/24  |
| claude-sonnet-5 | 0/24 | 16/24 | 5/24 | 23/24 | 16/24 |


**Key Findings:** 

- Sonnet 5 paired with SE+FI was able to achieve the highest score of ~96%, a far lead over the ~67% score on Sonnet 5 with FI or AUDIT. 
- GPT-4o-mini and GPT-5.4-nano, the legacy and budget models, were unable to adequately address all three RAG scenarios at once under any of the system instructions.
- SE was unable to score any points because it would never flag errors, failing the RAG scenarios where the document has perturbed values or the answer is not in the document.

## 7. The 2x2 factorial

The four instructions other than AUDIT form a 2x2 grid: WG carries neither clause, SE adds source exclusivity, FI adds the flag invitation, SE+FI adds both. Each clause's main effect is the change in the outcome from adding that clause, averaged over both states of the other clause (source-exclusivity main = mean of SE − WG and SE+FI − FI; flag-invitation main likewise). The interaction measures what happens when the clauses are combined (SE+FI − SE − FI + WG): negative means combining loses part of what they deliver alone and vice versa for positive, whereas closer to zero means the effects of the clauses cancel each other out. 

All effects are computed per fact and averaged. [] is a 95% bootstrap interval over facts (10,000 resamples, fixed seed), p is an exact two-sided sign test measuring statistical significance of the numbers and the notation **(x/y+)** shows that out of **y** total facts that changed between the main effects, **x** of them moved in a positive direction. 

**Contradiction sensitivity** 


| model           | source-exclusivity main              | flag-invitation main                  | interaction                           |
| --------------- | ------------------------------------ | ------------------------------------- | ------------------------------------- |
| gpt-4o-mini     | -0.07 [-0.10,-0.05], p<0.001 (1/18+) | +0.07 [+0.04,+0.10], p<0.001 (17/17+) | -0.14 [-0.19,-0.09], p<0.001 (1/17+)  |
| gpt-5.4-nano    | -0.05 [-0.07,-0.04], p<0.001 (0/17+) | +0.09 [+0.06,+0.12], p<0.001 (18/18+) | -0.10 [-0.13,-0.07], p<0.001 (0/17+)  |
| claude-sonnet-5 | -0.17 [-0.20,-0.13], p<0.001 (1/24+) | +0.54 [+0.48,+0.59], p<0.001 (24/24+) | +0.03 [-0.05,+0.11], p=0.523 (13/22+) |


**Key Findings:** 

- Source exclusivity instructions lowered the rate at which errors were flagged, whereas flag invitation instructions increased this rate.
- In the case of contradiction sensitivity, Sonnet 5 was most sensitive to the change in instructions.

**Absence faithfulness**


| model           | source-exclusivity main               | flag-invitation main                  | interaction                          |
| --------------- | ------------------------------------- | ------------------------------------- | ------------------------------------ |
| gpt-4o-mini     | +0.31 [+0.15,+0.48], p=0.035 (12/15+) | +0.04 [-0.06,+0.13], p=0.227 (8/11+)  | -0.19 [-0.35,-0.06], p=0.146 (3/12+) |
| gpt-5.4-nano    | +0.16 [+0.06,+0.27], p=0.039 (10/12+) | +0.12 [+0.03,+0.21], p=0.092 (10/13+) | -0.12 [-0.26,+0.01], p=0.227 (3/11+) |
| claude-sonnet-5 | +0.20 [+0.10,+0.31], p<0.001 (14/15+) | -0.05 [-0.13,+0.03], p=0.267 (4/13+)  | +0.10 [-0.07,+0.26], p=0.267 (9/13+) |


**Key Findings:** 

- Both instruction sets increased faithfulness rates in comparison to WG but source exclusivity instructions were much more effective than flag invitation ones across the models. 
- On the weaker GPT models, the interaction between the two instructions worsened the faithfulness rates, while the opposite effect was surprisingly observed in Sonnet 5.

**False endorsement**


| model           | source-exclusivity main              | flag-invitation main                  | interaction                          |
| --------------- | ------------------------------------ | ------------------------------------- | ------------------------------------ |
| gpt-4o-mini     | +0.00, p=1.000                       | +0.00, p=1.000                        | +0.00, p=1.000                       |
| gpt-5.4-nano    | -0.00 [-0.00,+0.00], p=1.000 (0/1+)  | -0.00 [-0.00,+0.00], p=1.000 (0/1+)   | +0.00 [+0.00,+0.01], p=1.000 (1/1+)  |
| claude-sonnet-5 | -0.12 [-0.15,-0.09], p<0.001 (0/22+) | +0.13 [+0.10,+0.16], p<0.001 (23/23+) | -0.24 [-0.29,-0.18], p<0.001 (0/22+) |


**Key Findings:** 

- The GPT models observed negligible levels of false endorsements, contrary to Sonnet 5, early indications that according to n=1 frontier models that endorsements in general are a newer behaviour.
- Source exclusivity instructions lowered false endorsement rates while the opposite was observed in flag-inviting instructions and the interaction between the two significantly lowers false endorsement rates.

## 8. AUDIT test

The AUDIT system instruction was designed to give models a step-by-step process for navigating each RAG scenario and responding in the most beneficial way for the user. The following table compares the performance of the AUDIT instruction against the best performing system instruction for each model across three of the dependent variables. 

Key: contradiction sensitivity / absence faithfulness / selective success


| model           | best 2x2 cell | best-cell rates     | AUDIT rates         |
| --------------- | ------------- | ------------------- | ------------------- |
| gpt-4o-mini     | FI            | 0.15 / 0.60 / 2/24  | 0.00 / 0.83 / 0/24  |
| gpt-5.4-nano    | FI            | 0.15 / 0.71 / 1/24  | 0.00 / 0.75 / 0/24  |
| claude-sonnet-5 | SE+FI         | 0.55 / 0.96 / 23/24 | 0.37 / 0.94 / 16/24 |


**Key Findings:** 

- AUDIT is generally much worse at catching errors compared to both flag inviting instructions, because the FI instructions are much more forceful than AUDIT when ordering the behaviour of catching errors. 
- As the model got more advanced, the advantage that AUDIT had in faithfulness rates worsened compared to each of the model's best performing system instructions. 
- Surprisingly, AUDIT had lower selective success rates than all of the model's best performing system instructions. This is because it often failed the second requirement 'reporting S3-S5 with what the document says AND raises a concern for plausibility'. 
- Overall, the AUDIT instruction was written to give models permission to flag errors rather than actively look for them, and the result was that SE+FI outperforms AUDIT on Sonnet 5 in these three dependent variables.

## 9. Severity as an independent variable

Key: questioned = raised doubt about the value; endorsed = actively vouched for it; most answers are neither (reported silently or declined). Severity bands: unperturbed S0, plausible perturbations S1-2, extreme perturbations S3-S5.

**Questioned rates**


| instruction | model           | S0   | S1   | S2   | S3   | S4   | S5   |
| ----------- | --------------- | ---- | ---- | ---- | ---- | ---- | ---- |
| SE          | gpt-4o-mini     | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| SE          | gpt-5.4-nano    | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| SE          | claude-sonnet-5 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| FI          | gpt-4o-mini     | 0.00 | 0.00 | 0.00 | 0.01 | 0.18 | 0.54 |
| FI          | gpt-5.4-nano    | 0.00 | 0.00 | 0.00 | 0.00 | 0.12 | 0.63 |
| FI          | claude-sonnet-5 | 0.06 | 0.15 | 0.54 | 0.82 | 1.00 | 1.00 |
| WG          | gpt-4o-mini     | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 |
| WG          | gpt-5.4-nano    | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 |
| WG          | claude-sonnet-5 | 0.00 | 0.00 | 0.00 | 0.08 | 0.18 | 0.65 |
| SE+FI       | gpt-4o-mini     | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 |
| SE+FI       | gpt-5.4-nano    | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.21 |
| SE+FI       | claude-sonnet-5 | 0.01 | 0.03 | 0.15 | 0.60 | 1.00 | 0.99 |
| AUDIT       | gpt-4o-mini     | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| AUDIT       | gpt-5.4-nano    | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| AUDIT       | claude-sonnet-5 | 0.00 | 0.00 | 0.00 | 0.29 | 0.64 | 0.90 |


**Key Findings:**

- None of the models questioned any values on SE instruction regardless of severity
- On FI, the old GPT models only began questioning values from S4, whereas Sonnet 5 questioned some claims on every severity, including S0, a dangerous behaviour observed only in the n=1 frontier model where the value that was actually correct was flagged as potentially implausible. 
- On WG, the old GPT models rarely questioned any claims with the exception of an anomaly in S5, whereas Sonnet 5 questioned claims on S4 and flagged a majority of S5 perturbed facts, even without any encouragement to flag errors from the system instruction. 
- SE+FI was more conservative in questioning values than FI, but the overall trend was similar in both instructions across the models. 
- The old GPT models refrained from questioning any values at all under the AUDIT instruction, whereas Sonnet 5 began to flag errors from S3, which quickly climbed in S4 and S5.

**Endorsement rates**


| instruction | model           | S0   | S1   | S2   | S3   | S4   | S5   |
| ----------- | --------------- | ---- | ---- | ---- | ---- | ---- | ---- |
| SE          | gpt-4o-mini     | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| SE          | gpt-5.4-nano    | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| SE          | claude-sonnet-5 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| FI          | gpt-4o-mini     | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| FI          | gpt-5.4-nano    | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| FI          | claude-sonnet-5 | 0.83 | 0.78 | 0.35 | 0.10 | 0.00 | 0.00 |
| WG          | gpt-4o-mini     | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| WG          | gpt-5.4-nano    | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 |
| WG          | claude-sonnet-5 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| SE+FI       | gpt-4o-mini     | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| SE+FI       | gpt-5.4-nano    | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| SE+FI       | claude-sonnet-5 | 0.00 | 0.03 | 0.00 | 0.01 | 0.00 | 0.00 |
| AUDIT       | gpt-4o-mini     | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| AUDIT       | gpt-5.4-nano    | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| AUDIT       | claude-sonnet-5 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |


**Key Findings:** 

- The only instance in which significant levels of endorsements were observed was Sonnet 5 on the FI system instruction. Every other combination of model and instruction observed very little if any endorsements at all.
- Based on the extrapolation of limited data, endorsements are probably a newer behaviour in frontier models that appear when the system instruction explicitly invites models to look for errors. 
- As discussed before, arguably the most dangerous behaviour observed throughout the results can be seen here where values that were intentionally perturbed were actively endorsed by Sonnet 5, many of which were corroborated with external standards from the model's parametric knowledge that was not verified well enough. 
- This can lead to users of RAG systems being misled into using incorrect information, especially when the model uses external standards to justify its endorsements even when the standards are applied incorrectly.

## 10. Per-document effects

**Contradiction sensitivity**


| instruction | model           | consent (doc 1) | epl (doc 2) | liquor (doc 3) |
| ----------- | --------------- | --------------- | ----------- | -------------- |
| SE          | gpt-4o-mini     | 0.00            | 0.00        | 0.00           |
| SE          | gpt-5.4-nano    | 0.00            | 0.00        | 0.00           |
| SE          | claude-sonnet-5 | 0.00            | 0.00        | 0.00           |
| FI          | gpt-4o-mini     | 0.14            | 0.14        | 0.16           |
| FI          | gpt-5.4-nano    | 0.17            | 0.17        | 0.10           |
| FI          | claude-sonnet-5 | 0.67            | 0.69        | 0.75           |
| WG          | gpt-4o-mini     | 0.00            | 0.00        | 0.01           |
| WG          | gpt-5.4-nano    | 0.00            | 0.01        | 0.00           |
| WG          | claude-sonnet-5 | 0.25            | 0.20        | 0.11           |
| SE+FI       | gpt-4o-mini     | 0.00            | 0.00        | 0.00           |
| SE+FI       | gpt-5.4-nano    | 0.06            | 0.05        | 0.00           |
| SE+FI       | claude-sonnet-5 | 0.57            | 0.57        | 0.52           |
| AUDIT       | gpt-4o-mini     | 0.00            | 0.00        | 0.00           |
| AUDIT       | gpt-5.4-nano    | 0.00            | 0.00        | 0.00           |
| AUDIT       | claude-sonnet-5 | 0.40            | 0.38        | 0.33           |


**Absence faithfulness**


| instruction | model           | consent | epl  | liquor |
| ----------- | --------------- | ------- | ---- | ------ |
| SE          | gpt-4o-mini     | 0.96    | 0.71 | 0.89   |
| SE          | gpt-5.4-nano    | 0.96    | 0.67 | 0.63   |
| SE          | claude-sonnet-5 | 1.00    | 0.86 | 1.00   |
| FI          | gpt-4o-mini     | 0.62    | 0.52 | 0.63   |
| FI          | gpt-5.4-nano    | 0.96    | 0.67 | 0.52   |
| FI          | claude-sonnet-5 | 0.54    | 0.81 | 0.78   |
| WG          | gpt-4o-mini     | 0.67    | 0.38 | 0.33   |
| WG          | gpt-5.4-nano    | 0.67    | 0.62 | 0.33   |
| WG          | claude-sonnet-5 | 1.00    | 0.71 | 0.70   |
| SE+FI       | gpt-4o-mini     | 0.88    | 0.62 | 0.89   |
| SE+FI       | gpt-5.4-nano    | 0.96    | 0.71 | 0.74   |
| SE+FI       | claude-sonnet-5 | 1.00    | 0.86 | 1.00   |
| AUDIT       | gpt-4o-mini     | 0.92    | 0.62 | 0.93   |
| AUDIT       | gpt-5.4-nano    | 1.00    | 0.67 | 0.59   |
| AUDIT       | claude-sonnet-5 | 1.00    | 0.86 | 0.96   |


## 11. Provenance

In order to stay cost-efficient, some data was transferred from v1 to v2. The following table compares all the seeded data (transferred data from v1) to fresh data to illustrate how the effect of transferring was negligible to the overall results. 


| model        | instruction | seeded Q / E (n)    | fresh Q / E (n)     |
| ------------ | ----------- | ------------------- | ------------------- |
| gpt-4o-mini  | SE          | 0.000 / 0.000 (240) | 0.000 / 0.000 (270) |
| gpt-4o-mini  | FI          | 0.154 / 0.000 (240) | 0.137 / 0.000 (270) |
| gpt-4o-mini  | WG          | 0.000 / 0.000 (240) | 0.004 / 0.000 (270) |
| gpt-4o-mini  | SE+FI       | 0.004 / 0.000 (240) | 0.000 / 0.000 (270) |
| gpt-5.4-nano | SE          | 0.000 / 0.000 (240) | 0.000 / 0.000 (270) |
| gpt-5.4-nano | FI          | 0.163 / 0.000 (240) | 0.137 / 0.000 (270) |
| gpt-5.4-nano | WG          | 0.000 / 0.000 (240) | 0.004 / 0.004 (270) |
| gpt-5.4-nano | SE+FI       | 0.046 / 0.000 (240) | 0.037 / 0.000 (270) |


