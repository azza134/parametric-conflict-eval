# Results

**Models tested:** gpt-4o-mini (gpt-4o-mini-2024-07-18), gpt-5.4-nano (gpt-5.4-nano-2026-03-17), claude-sonnet-5 (resolved alias), gpt-5.6-terra (resolved alias), N=3 per cell; the seeded consent cells are N=8. (legacy vs budget vs two frontier)
**Judge:** GPT-5.4-mini (gpt-5.4-mini-2026-03-17): caveat stance kappa 0.97 / corroboration 0.92, 0/30 anchors misjudged, 168-row human-labelled gold; abstention kappa 0.94, 0/54 anchors misjudged, 110-row human-labelled gold.
**Certification gate**: zero anchor misses AND kappa >= 0.80.

## 1. Key Terms

Definitions of the outcomes are as follows:

**1. Contradiction Sensitivity:** Rate at which the document catches a wrong value (severity 1 to 5, perturbed values)
**2. Clean Specificity:** Rate at which the model does not question or decline the document value at severity 0 (unperturbed)
**3. Absence Faithfulness:** Rate at which the model abstains from supplying an answer if the answer to a question about the document is removed. 
**4. False Endorsement:** Rate at which the model actively endorses perturbed values (S1 to S5)
**5. False Corroboration:** Rate at which the model actively endorses perturbed values (S1 to S5) AND justifies the endorsement, either generically (the value "appears standard/reasonable") or by citing a named external authority. The two justification types are reported separately in Section 5.
**6. Situated Faithfulness:** One point per fact (24 facts) if the model takes the ideal speech act in all three grounding scenarios, each by majority over that fact's repetitions:

- does not question or decline the unperturbed value (S0)
- raises a plausibility concern on extreme perturbations (S3-S5)
- faithfully abstains when the answer is absent from the document

System instructions are shortened as such:
SE = **SOURCE_EXCLUSIVE**
FI = **FLAG_INVITING**
WG = **WEAK_GROUNDING**
SE+FI = **SOURCE_EXCLUSIVE_FLAG_INVITING**
AUDIT = **SELECTIVE_AUDIT**

One further term appears in the tables below: **parametric leakage**, the rate at which a model answers from its own pre-training data when the instruction requires grounding.

## 2. Contradiction sensitivity


| model           | SE                       | FI               | WG               | SE+FI            | AUDIT                    |
| --------------- | ------------------------ | ---------------- | ---------------- | ---------------- | ------------------------ |
| gpt-4o-mini     | 0.00 [0.00,0.01] (0/360) | 0.15 [0.12,0.18] | 0.00 [0.00,0.01] | 0.00 [0.00,0.01] | 0.00 [0.00,0.01]         |
| gpt-5.4-nano    | 0.00 [0.00,0.01]         | 0.15 [0.12,0.18] | 0.00 [0.00,0.01] | 0.04 [0.03,0.06] | 0.00 [0.00,0.01]         |
| claude-sonnet-5 | 0.00 [0.00,0.01]         | 0.70 [0.65,0.75] | 0.18 [0.15,0.23] | 0.55 [0.50,0.60] | 0.37 [0.32,0.42]         |
| gpt-5.6-terra   | 0.00 [0.00,0.01]         | 0.48 [0.43,0.53] | 0.00 [0.00,0.01] | 0.30 [0.25,0.35] | 0.00 [0.00,0.02] (1/360) |


**Key Findings:** 

- All the models caught the most errors on FI, with a lower but still significant amount of errors still being caught under the SE+FI instruction.
- Interestingly, only Sonnet 5 caught errors on the AUDIT instruction, a behavioural difference between Sonnet 5 and the OpenAI models in responding to this instruction in terms of catching errors.

## 3. Clean specificity


| model           | SE               | FI               | WG               | SE+FI            | AUDIT            |
| --------------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- |
| gpt-4o-mini     | 0.94 [0.88,0.97] | 1.00 [0.96,1.00] | 1.00 [0.96,1.00] | 0.97 [0.92,0.99] | 0.97 [0.90,0.99] |
| gpt-5.4-nano    | 0.95 [0.89,0.98] | 1.00 [0.96,1.00] | 1.00 [0.96,1.00] | 0.93 [0.87,0.97] | 1.00 [0.95,1.00] |
| claude-sonnet-5 | 1.00 [0.95,1.00] | 0.85 [0.75,0.91] | 1.00 [0.95,1.00] | 0.99 [0.93,1.00] | 1.00 [0.95,1.00] |
| gpt-5.6-terra   | 1.00 [0.95,1.00] | 1.00 [0.95,1.00] | 1.00 [0.95,1.00] | 1.00 [0.95,1.00] | 1.00 [0.95,1.00] |


**Key Findings:**

- All the models had a similar performance across all instructions in abstaining from raising doubts about the unperturbed value, with the anomaly of 0.85 on Sonnet 5 (FI). 
- Between the anomalies in Sonnet 5's clean specificity on FI and contradiction sensitivity on AUDIT, Sonnet 5 can be seen from the data as being more likely to flag errors if the system instruction invites such behaviour, even values that are correct and unperturbed (0.85 clean specificity).

## 4. Absence faithfulness


| model           | SE               | FI               | WG               | SE+FI            | AUDIT            |
| --------------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- |
| gpt-4o-mini     | 0.86 [0.76,0.92] | 0.60 [0.48,0.70] | 0.46 [0.35,0.57] | 0.81 [0.70,0.88] | 0.83 [0.73,0.90] |
| gpt-5.4-nano    | 0.75 [0.64,0.84] | 0.71 [0.59,0.80] | 0.53 [0.41,0.64] | 0.81 [0.70,0.88] | 0.75 [0.64,0.84] |
| claude-sonnet-5 | 0.96 [0.88,0.99] | 0.71 [0.59,0.80] | 0.81 [0.70,0.88] | 0.96 [0.88,0.99] | 0.94 [0.87,0.98] |
| gpt-5.6-terra   | 0.85 [0.75,0.91] | 0.71 [0.59,0.80] | 0.65 [0.54,0.75] | 0.82 [0.72,0.89] | 0.83 [0.73,0.90] |


**Key Findings:** 

- Sonnet 5 recorded significantly higher absence faithfulness rates than the GPT models (including 5.6 Terra), indicating that Sonnet 5 is generally more likely to abstain from falling back on parametric knowledge when the answer to a question is not provided in external context at all. Whether this extends to other Anthropic models is untested (one Anthropic model in the roster). 
- SE had the highest faithfulness rates across the models, closely trailed by SE+FI and AUDIT, indicating that the system instructions that more heavily suppressed the use of parametric knowledge were more effective at faithfulness. FI and WG instructions recorded much lower absence faithfulness rates.

## 5. False endorsements and corroborations

Raw endorsement on perturbed values S1-S5: rate with Wilson 95% interval. (more detail in Section 9)


| model           | SE               | FI               | WG               | SE+FI            | AUDIT            |
| --------------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- |
| gpt-4o-mini     | 0.00 [0.00,0.01] | 0.00 [0.00,0.01] | 0.00 [0.00,0.01] | 0.00 [0.00,0.01] | 0.00 [0.00,0.01] |
| gpt-5.4-nano    | 0.00 [0.00,0.01] | 0.00 [0.00,0.01] | 0.00 [0.00,0.01] | 0.00 [0.00,0.01] | 0.00 [0.00,0.01] |
| claude-sonnet-5 | 0.00 [0.00,0.01] | 0.24 [0.20,0.29] | 0.00 [0.00,0.01] | 0.01 [0.00,0.02] | 0.00 [0.00,0.01] |
| gpt-5.6-terra   | 0.00 [0.00,0.01] | 0.00 [0.00,0.01] | 0.00 [0.00,0.01] | 0.00 [0.00,0.01] | 0.00 [0.00,0.01] |


Key: False endorsement / false corroboration 


| model           | SE          | FI          | WG          | SE+FI       | AUDIT       |
| --------------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| gpt-4o-mini     | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| gpt-5.4-nano    | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| claude-sonnet-5 | 0.00 / 0.00 | 0.24 / 0.20 | 0.00 / 0.00 | 0.01 / 0.00 | 0.00 / 0.00 |
| gpt-5.6-terra   | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |


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

- Sonnet 5 is the only model that recorded any endorsements in 1,800 perturbed answers. 
- Under FI, 24% of Sonnet 5's perturbed answers were endorsements (88/360) and 20% were *corroborated* endorsements (73/360). Of the 73, 63 were justified generically (the value "appears standard/reasonable") and 10 cited a named external authority (2.8% of perturbed answers). The latter cases are the most serious ones; this involves the model invoking real standards (e.g. "Planning for Bushfire Protection 2019") to vouch for a perturbed value, a failure of both document grounding and parametric knowledge accuracy. 
- The endorsements are concentrated at low severity: 0.78 at S1 and 0.35 at S2, falling to 0.10 at S3 and zero at S4-S5 (per-severity table above). Sonnet 5 also endorses the unperturbed value 83% of the time under FI, so the behaviour is better described as vouching for any value it deems plausible when the instruction invites a plausibility assessment. S1-S2 perturbations (1.25x to 3.5x) sit inside that plausibility window and are not reliably detectable by any model.
  -  The failure is not endorsement of absurd values, which never happened in the data. It is confident corroboration language, sometimes citing named authorities, applied to errors the model cannot actually rule out.
- This behaviour was only observed in Sonnet 5. Whether it is Anthropic-family-wide or idiosyncratic to Sonnet 5 is untested (one Anthropic model in the roster).

## 6. Situated faithfulness


| model           | SE   | FI    | WG   | SE+FI | AUDIT |
| --------------- | ---- | ----- | ---- | ----- | ----- |
| gpt-4o-mini     | 0/24 | 2/24  | 0/24 | 0/24  | 0/24  |
| gpt-5.4-nano    | 0/24 | 1/24  | 0/24 | 0/24  | 0/24  |
| claude-sonnet-5 | 0/24 | 16/24 | 5/24 | 23/24 | 16/24 |
| gpt-5.6-terra   | 0/24 | 16/24 | 0/24 | 12/24 | 0/24  |


**Key Findings:** 

- Sonnet 5 paired with SE+FI was able to achieve the highest score of ~96%, a far lead over the ~67% score observed in Sonnet 5 (FI and AUDIT) and GPT-5.6 Terra (FI). 
- Unlike Sonnet 5, GPT-5.6-terra's best instruction is FI (16/24) rather than SE+FI (12/24). It was observed that Sonnet 5 is able to use SE+FI to correctly decide when to ground its answers and when to flag implausible values, whereas GPT-5.6-terra treats the two instructions as conflicting and flags implausible values significantly less.
  - To illustrate this, GPT-5.6-terra's ability to pass the requirement 'raises a plausibility concern on extreme perturbations (S3-S5)' was 22/24 on FI and 12/24 on SE+FI.
- GPT-4o-mini and GPT-5.4-nano, the legacy and budget models, were unable to adequately address all three grounding scenarios at once under any of the system instructions.
- SE was unable to score any points because it would never flag errors, failing the grounding scenarios where the document has perturbed values or the answer is not in the document.

## 7. The 2x2 factorial

The four instructions other than AUDIT form a 2x2 grid: WG carries neither clause, SE adds source exclusivity, FI adds the flag invitation, SE+FI adds both. Each clause's main effect is the change in the outcome from adding that clause, averaged over both states of the other clause (source-exclusivity main = mean of SE − WG and SE+FI − FI; flag-invitation main likewise). The interaction measures what happens when the clauses are combined (SE+FI − SE − FI + WG): negative means combining loses part of what they deliver alone and vice versa for positive, whereas closer to zero means the effects of the clauses cancel each other out. 

All effects are computed per fact and averaged. [] is a 95% bootstrap interval over facts (10,000 resamples, fixed seed), p is an exact two-sided sign test measuring statistical significance of the numbers and the notation **(x/y+)** shows that out of **y** total facts that changed between the main effects, **x** of them moved in a positive direction. 

**Contradiction sensitivity** 


| model           | source-exclusivity main              | flag-invitation main                  | interaction                           |
| --------------- | ------------------------------------ | ------------------------------------- | ------------------------------------- |
| gpt-4o-mini     | -0.07 [-0.10,-0.05], p<0.001 (1/18+) | +0.07 [+0.04,+0.10], p<0.001 (17/17+) | -0.14 [-0.19,-0.09], p<0.001 (1/17+)  |
| gpt-5.4-nano    | -0.05 [-0.07,-0.04], p<0.001 (0/17+) | +0.09 [+0.06,+0.12], p<0.001 (18/18+) | -0.10 [-0.13,-0.07], p<0.001 (0/17+)  |
| claude-sonnet-5 | -0.17 [-0.20,-0.13], p<0.001 (1/24+) | +0.54 [+0.48,+0.59], p<0.001 (24/24+) | +0.03 [-0.05,+0.11], p=0.523 (13/22+) |
| gpt-5.6-terra   | -0.09 [-0.12,-0.07], p<0.001 (0/19+) | +0.39 [+0.34,+0.44], p<0.001 (24/24+) | -0.18 [-0.24,-0.13], p<0.001 (0/19+)  |


**Key Findings:** 

- Source exclusivity instructions lowered the rate at which errors were flagged, whereas flag invitation instructions increased this rate.
- In the case of contradiction sensitivity, Sonnet 5 was most sensitive to the change in instructions.
- The two mains interacted additively under Sonnet 5, whereas all the GPT models suffered from the interaction in the case of contradiction sensitivity.

**Absence faithfulness**


| model           | source-exclusivity main               | flag-invitation main                  | interaction                          |
| --------------- | ------------------------------------- | ------------------------------------- | ------------------------------------ |
| gpt-4o-mini     | +0.31 [+0.15,+0.48], p=0.035 (12/15+) | +0.04 [-0.06,+0.13], p=0.227 (8/11+)  | -0.19 [-0.35,-0.06], p=0.146 (3/12+) |
| gpt-5.4-nano    | +0.16 [+0.06,+0.27], p=0.039 (10/12+) | +0.12 [+0.03,+0.21], p=0.092 (10/13+) | -0.12 [-0.26,+0.01], p=0.227 (3/11+) |
| claude-sonnet-5 | +0.20 [+0.10,+0.31], p<0.001 (14/15+) | -0.05 [-0.13,+0.03], p=0.267 (4/13+)  | +0.10 [-0.07,+0.26], p=0.267 (9/13+) |
| gpt-5.6-terra   | +0.15 [+0.00,+0.31], p=0.146 (9/12+)  | +0.01 [-0.04,+0.08], p=1.000 (6/11+)  | -0.08 [-0.19,+0.01], p=0.344 (3/10+) |


**Key Findings:** 

- Both instruction sets increased faithfulness rates in comparison to WG (with the exception of Sonnet 5 on flag-invitation main), but source exclusivity instructions were much more effective than flag invitation ones across the models. 
- On the GPT models, the interaction between the two instructions worsened the faithfulness rates, while the opposite effect was observed in Sonnet 5. 
- Between absence faithfulness and contradiction sensitivity, GPT models appeared to have worsened performance in handling document-grounded QA under the interaction between the mains as opposed to Sonnet 5, suggesting that Sonnet 5 is more capable of interpreting more complex system instructions than the GPT models tested.

**False endorsement**


| model           | source-exclusivity main              | flag-invitation main                  | interaction                          |
| --------------- | ------------------------------------ | ------------------------------------- | ------------------------------------ |
| gpt-4o-mini     | +0.00, p=1.000                       | +0.00, p=1.000                        | +0.00, p=1.000                       |
| gpt-5.4-nano    | -0.00 [-0.00,+0.00], p=1.000 (0/1+)  | -0.00 [-0.00,+0.00], p=1.000 (0/1+)   | +0.00 [+0.00,+0.01], p=1.000 (1/1+)  |
| claude-sonnet-5 | -0.12 [-0.15,-0.09], p<0.001 (0/22+) | +0.13 [+0.10,+0.16], p<0.001 (23/23+) | -0.24 [-0.29,-0.18], p<0.001 (0/22+) |
| gpt-5.6-terra   | +0.00, p=1.000                       | +0.00, p=1.000                        | +0.00, p=1.000                       |


**Key Findings:** 

- Sonnet 5 is the only model with any false endorsements. 
- Source exclusivity instructions lowered false endorsement rates while the opposite was observed in flag-inviting instructions, and the interaction between the two significantly lowers false endorsement rates.

## 8. AUDIT test

The AUDIT system instruction was designed to give models a step-by-step process for navigating each grounding scenario and responding in the most beneficial way for the user. The following table compares the performance of the AUDIT instruction against the best performing system instruction for each model across three of the dependent variables. 

Key: contradiction sensitivity / absence faithfulness / situated faithfulness


| model           | best 2x2 cell | best-cell rates     | AUDIT rates         |
| --------------- | ------------- | ------------------- | ------------------- |
| gpt-4o-mini     | FI            | 0.15 / 0.60 / 2/24  | 0.00 / 0.83 / 0/24  |
| gpt-5.4-nano    | FI            | 0.15 / 0.71 / 1/24  | 0.00 / 0.75 / 0/24  |
| claude-sonnet-5 | SE+FI         | 0.55 / 0.96 / 23/24 | 0.37 / 0.94 / 16/24 |
| gpt-5.6-terra   | FI            | 0.48 / 0.71 / 16/24 | 0.00 / 0.83 / 0/24  |


**Key Findings:** 

- AUDIT is generally much worse at catching errors compared to both flag inviting instructions, because the FI instructions are much more forceful than AUDIT when ordering the behaviour of catching errors. 
- Overall, the AUDIT instruction was written to give models permission to flag errors rather than actively look for them, and the result was that SE+FI outperforms AUDIT on Sonnet 5 in these three dependent variables.

## 9. Severity as an independent variable

Key: questioned = raised doubt about the value; endorsed = actively vouched for it; most answers are neither (reported silently or declined). Severity bands: unperturbed S0, plausible perturbations S1-2, extreme perturbations S3-S5.

**Questioned rates**


| instruction | model           | S0   | S1   | S2   | S3   | S4   | S5   |
| ----------- | --------------- | ---- | ---- | ---- | ---- | ---- | ---- |
| SE          | gpt-4o-mini     | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| SE          | gpt-5.4-nano    | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| SE          | claude-sonnet-5 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| SE          | gpt-5.6-terra   | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| FI          | gpt-4o-mini     | 0.00 | 0.00 | 0.00 | 0.01 | 0.18 | 0.54 |
| FI          | gpt-5.4-nano    | 0.00 | 0.00 | 0.00 | 0.00 | 0.12 | 0.63 |
| FI          | claude-sonnet-5 | 0.06 | 0.15 | 0.54 | 0.82 | 1.00 | 1.00 |
| FI          | gpt-5.6-terra   | 0.00 | 0.00 | 0.12 | 0.42 | 0.92 | 0.96 |
| WG          | gpt-4o-mini     | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 |
| WG          | gpt-5.4-nano    | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 |
| WG          | claude-sonnet-5 | 0.00 | 0.00 | 0.00 | 0.08 | 0.18 | 0.65 |
| WG          | gpt-5.6-terra   | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| SE+FI       | gpt-4o-mini     | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 |
| SE+FI       | gpt-5.4-nano    | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.21 |
| SE+FI       | claude-sonnet-5 | 0.01 | 0.03 | 0.15 | 0.60 | 1.00 | 0.99 |
| SE+FI       | gpt-5.6-terra   | 0.00 | 0.00 | 0.00 | 0.07 | 0.51 | 0.92 |
| AUDIT       | gpt-4o-mini     | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| AUDIT       | gpt-5.4-nano    | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| AUDIT       | claude-sonnet-5 | 0.00 | 0.00 | 0.00 | 0.29 | 0.64 | 0.90 |
| AUDIT       | gpt-5.6-terra   | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 |


**Key Findings:**

- None of the models questioned any values on SE instruction regardless of severity
- On FI, questioned rates appeared to scale with model capability, with the old GPT models only starting to question at meaningful rates from S4, while GPT-5.6-terra started as S2. Sonnet 5 even questioned unperturbed values 6% of the time, which could be perceived as potentially concerning and reveals that Sonnet 5 is much more likely to question values if explicitly invited by the system instruction as opposed to OpenAI models. 
- On WG, the GPT models rarely questioned any claims with the exception of an anomaly in S5, whereas Sonnet 5 questioned claims on S4 and flagged a majority of S5 perturbed facts, even without any encouragement to flag errors from the system instruction. 
- SE+FI was more conservative in questioning values than FI, with questioning starting later on all the models, but the overall trend was similar in both instructions across the models. 
- The GPT models refrained from questioning any values at all under the AUDIT instruction, with the exception of an anomaly on S5 for GPT-5.6-terra, whereas Sonnet 5 began to flag errors from S3 under AUDIT, which quickly climbed in S4 and S5.

**Endorsement rates**


| instruction | model           | S0   | S1   | S2   | S3   | S4   | S5   |
| ----------- | --------------- | ---- | ---- | ---- | ---- | ---- | ---- |
| SE          | gpt-4o-mini     | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| SE          | gpt-5.4-nano    | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| SE          | claude-sonnet-5 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| SE          | gpt-5.6-terra   | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| FI          | gpt-4o-mini     | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| FI          | gpt-5.4-nano    | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| FI          | claude-sonnet-5 | 0.83 | 0.78 | 0.35 | 0.10 | 0.00 | 0.00 |
| FI          | gpt-5.6-terra   | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| WG          | gpt-4o-mini     | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| WG          | gpt-5.4-nano    | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 |
| WG          | claude-sonnet-5 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| WG          | gpt-5.6-terra   | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| SE+FI       | gpt-4o-mini     | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| SE+FI       | gpt-5.4-nano    | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| SE+FI       | claude-sonnet-5 | 0.00 | 0.03 | 0.00 | 0.01 | 0.00 | 0.00 |
| SE+FI       | gpt-5.6-terra   | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| AUDIT       | gpt-4o-mini     | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| AUDIT       | gpt-5.4-nano    | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| AUDIT       | claude-sonnet-5 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| AUDIT       | gpt-5.6-terra   | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |


**Key Findings:** 

- The only instance in which significant levels of endorsements were observed was Sonnet 5 on the FI system instruction. Every other combination of model and instruction observed very little if any endorsements at all.
- GPT-5.6-terra, the second frontier model, endorsed nothing at any severity under any instruction. Thus, endorsement is a behaviour that appears to be specific to Sonnet 5 and possibly other Anthropic models rather than one that arrives with frontier capability. What remains open is whether it is model-family-specific or idiosyncratic. 
- As discussed in Section 5, arguably the most dangerous behaviour observed throughout the results can be seen here, where values that were intentionally perturbed were actively endorsed by Sonnet 5. Most were justified generically as appearing standard, but 10 instances cited named standards that are real but whose claimed support for the perturbed value is hallucinated. The endorsements sit almost entirely at S1-S2, the perturbations small enough to pass a plausibility check.
  - This can lead to document-grounded QA misleading users into applying incorrect information, especially when the model uses external standards to justify its endorsements even when the standards are applied incorrectly. Users are less likely to verify information if an external standard is confidently cited by a model.

## 10. Per-document effects

**Contradiction sensitivity**


| instruction | model           | consent (doc 1) | epl (doc 2) | liquor (doc 3) |
| ----------- | --------------- | --------------- | ----------- | -------------- |
| SE          | gpt-4o-mini     | 0.00            | 0.00        | 0.00           |
| SE          | gpt-5.4-nano    | 0.00            | 0.00        | 0.00           |
| SE          | claude-sonnet-5 | 0.00            | 0.00        | 0.00           |
| SE          | gpt-5.6-terra   | 0.00            | 0.00        | 0.00           |
| FI          | gpt-4o-mini     | 0.14            | 0.14        | 0.16           |
| FI          | gpt-5.4-nano    | 0.17            | 0.17        | 0.10           |
| FI          | claude-sonnet-5 | 0.67            | 0.69        | 0.75           |
| FI          | gpt-5.6-terra   | 0.51            | 0.48        | 0.47           |
| WG          | gpt-4o-mini     | 0.00            | 0.00        | 0.01           |
| WG          | gpt-5.4-nano    | 0.00            | 0.01        | 0.00           |
| WG          | claude-sonnet-5 | 0.25            | 0.20        | 0.11           |
| WG          | gpt-5.6-terra   | 0.00            | 0.00        | 0.00           |
| SE+FI       | gpt-4o-mini     | 0.00            | 0.00        | 0.00           |
| SE+FI       | gpt-5.4-nano    | 0.06            | 0.05        | 0.00           |
| SE+FI       | claude-sonnet-5 | 0.57            | 0.57        | 0.52           |
| SE+FI       | gpt-5.6-terra   | 0.35            | 0.26        | 0.29           |
| AUDIT       | gpt-4o-mini     | 0.00            | 0.00        | 0.00           |
| AUDIT       | gpt-5.4-nano    | 0.00            | 0.00        | 0.00           |
| AUDIT       | claude-sonnet-5 | 0.40            | 0.38        | 0.33           |
| AUDIT       | gpt-5.6-terra   | 0.00            | 0.01        | 0.00           |


**Absence faithfulness**


| instruction | model           | consent | epl  | liquor |
| ----------- | --------------- | ------- | ---- | ------ |
| SE          | gpt-4o-mini     | 0.96    | 0.71 | 0.89   |
| SE          | gpt-5.4-nano    | 0.96    | 0.67 | 0.63   |
| SE          | claude-sonnet-5 | 1.00    | 0.86 | 1.00   |
| SE          | gpt-5.6-terra   | 0.88    | 0.86 | 0.81   |
| FI          | gpt-4o-mini     | 0.62    | 0.52 | 0.63   |
| FI          | gpt-5.4-nano    | 0.96    | 0.67 | 0.52   |
| FI          | claude-sonnet-5 | 0.54    | 0.81 | 0.78   |
| FI          | gpt-5.6-terra   | 0.92    | 0.67 | 0.56   |
| WG          | gpt-4o-mini     | 0.67    | 0.38 | 0.33   |
| WG          | gpt-5.4-nano    | 0.67    | 0.62 | 0.33   |
| WG          | claude-sonnet-5 | 1.00    | 0.71 | 0.70   |
| WG          | gpt-5.6-terra   | 0.83    | 0.57 | 0.56   |
| SE+FI       | gpt-4o-mini     | 0.88    | 0.62 | 0.89   |
| SE+FI       | gpt-5.4-nano    | 0.96    | 0.71 | 0.74   |
| SE+FI       | claude-sonnet-5 | 1.00    | 0.86 | 1.00   |
| SE+FI       | gpt-5.6-terra   | 0.88    | 0.86 | 0.74   |
| AUDIT       | gpt-4o-mini     | 0.92    | 0.62 | 0.93   |
| AUDIT       | gpt-5.4-nano    | 1.00    | 0.67 | 0.59   |
| AUDIT       | claude-sonnet-5 | 1.00    | 0.86 | 0.96   |
| AUDIT       | gpt-5.6-terra   | 0.88    | 0.86 | 0.78   |


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


