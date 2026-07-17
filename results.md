# Results

**Models tested:** gpt-4o-mini (gpt-4o-mini-2024-07-18), gpt-5.4-nano (gpt-5.4-nano-2026-03-17), claude-sonnet-5 (resolved alias), gpt-5.6-terra (resolved alias), claude-haiku-4-5 (claude-haiku-4-5-20251001), N=3 per cell; the seeded consent cells are N=8. (legacy, budget, two frontier, and a second Anthropic model at the small tier) claude-opus-4-8 (resolved alias) joins under FLAG_INVITING only at N=1 -- Section 14.
**Judge:** GPT-5.4-mini (gpt-5.4-mini-2026-03-17): caveat stance kappa 0.97 / corroboration 0.90, 0/30 anchors misjudged, 228-row human-labelled gold; abstention kappa 0.97, 0/54 anchors misjudged, 140-row human-labelled gold.
**Certification gate**: zero anchor misses AND kappa >= 0.80.

Error-flagging rate vs perturbation severity, five instruction lines faceted by model
*Regenerate offline with `python3 plot_results.py`.*

## 1. Key Terms

Definitions of the outcomes are as follows:

**1. Contradiction Sensitivity:** The error-flagging rate on perturbed rows (S1-S5): the rate at which the model flags a planted wrong value in the document.
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


| model            | SE                       | FI               | WG               | SE+FI            | AUDIT                    |
| ---------------- | ------------------------ | ---------------- | ---------------- | ---------------- | ------------------------ |
| gpt-4o-mini      | 0.00 [0.00,0.01] (0/360) | 0.15 [0.12,0.18] | 0.00 [0.00,0.01] | 0.00 [0.00,0.01] | 0.00 [0.00,0.01]         |
| gpt-5.4-nano     | 0.00 [0.00,0.01]         | 0.15 [0.12,0.18] | 0.00 [0.00,0.01] | 0.04 [0.03,0.06] | 0.00 [0.00,0.01]         |
| claude-haiku-4-5 | 0.08 [0.05,0.11]         | 0.33 [0.28,0.38] | 0.14 [0.11,0.18] | 0.24 [0.20,0.29] | 0.14 [0.11,0.19]         |
| claude-sonnet-5  | 0.00 [0.00,0.01]         | 0.70 [0.65,0.75] | 0.18 [0.15,0.23] | 0.55 [0.50,0.60] | 0.37 [0.32,0.42]         |
| gpt-5.6-terra    | 0.00 [0.00,0.01]         | 0.48 [0.43,0.53] | 0.00 [0.00,0.01] | 0.30 [0.25,0.35] | 0.00 [0.00,0.02] (1/360) |


**Key Findings:** 

- All the models caught the most errors on FI, with a lower but still significant amount of errors still being caught under the SE+FI instruction.
- Interestingly, only the Anthropic models caught errors on the AUDIT instruction (Sonnet 5 0.37, Haiku 4.5 0.14), a behavioural difference between the Anthropic and OpenAI models in responding to this instruction in terms of catching errors.
- Haiku 4.5 is the only model in the study that flags any errors under SE. For every other model, SE suppresses flagging to zero.

**Detection-mechanism note:** contradiction sensitivity pools two ways a model can catch a perturbation — world-knowledge implausibility, and document-internal contradiction. One fact (minors_section) is internally detectable at every severity: the licence's legislative header enumerates "...51 and 121 of the Liquor Act 2007", which the perturbation does not touch, so the perturbed bullet contradicts the document itself and no parametric knowledge is required to flag it. A programmatic check (reported under `analysis`) confirms no other fact leaves an answer-bearing site intact under perturbation. Excluding minors_section changes every model x instruction flag rate by at most 0.008 — no table cell above moves at two decimal places — so the reported rates are not driven by internal cross-referencing. Softer internal cues (a section-number format range, or the surviving Christmas Day "12:00 noon" beside a perturbed Good Friday time) remain and are not controlled for.

## 3. Clean specificity


| model            | SE               | FI               | WG               | SE+FI            | AUDIT            |
| ---------------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- |
| gpt-4o-mini      | 0.94 [0.88,0.97] | 1.00 [0.96,1.00] | 1.00 [0.96,1.00] | 0.97 [0.92,0.99] | 0.97 [0.90,0.99] |
| gpt-5.4-nano     | 0.95 [0.89,0.98] | 1.00 [0.96,1.00] | 1.00 [0.96,1.00] | 0.93 [0.87,0.97] | 1.00 [0.95,1.00] |
| claude-haiku-4-5 | 1.00 [0.95,1.00] | 1.00 [0.95,1.00] | 1.00 [0.95,1.00] | 1.00 [0.95,1.00] | 1.00 [0.95,1.00] |
| claude-sonnet-5  | 1.00 [0.95,1.00] | 0.85 [0.75,0.91] | 1.00 [0.95,1.00] | 0.99 [0.93,1.00] | 1.00 [0.95,1.00] |
| gpt-5.6-terra    | 1.00 [0.95,1.00] | 1.00 [0.95,1.00] | 1.00 [0.95,1.00] | 1.00 [0.95,1.00] | 1.00 [0.95,1.00] |


**Key Findings:**

- All the models had a similar performance across all instructions in abstaining from raising doubts about the unperturbed value, with the anomaly of 0.85 on Sonnet 5 (FI). 
- Between the anomalies in Sonnet 5's clean specificity on FI and contradiction sensitivity on AUDIT, Sonnet 5 can be seen from the data as being more likely to flag errors if the system instruction invites such behaviour, even values that are correct and unperturbed (0.85 clean specificity).

## 4. Absence faithfulness


| model            | SE               | FI               | WG               | SE+FI            | AUDIT            |
| ---------------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- |
| gpt-4o-mini      | 0.86 [0.76,0.92] | 0.60 [0.48,0.70] | 0.46 [0.35,0.57] | 0.81 [0.70,0.88] | 0.83 [0.73,0.90] |
| gpt-5.4-nano     | 0.75 [0.64,0.84] | 0.71 [0.59,0.80] | 0.53 [0.41,0.64] | 0.81 [0.70,0.88] | 0.75 [0.64,0.84] |
| claude-haiku-4-5 | 0.93 [0.85,0.97] | 0.78 [0.67,0.86] | 0.82 [0.72,0.89] | 0.94 [0.87,0.98] | 0.90 [0.81,0.95] |
| claude-sonnet-5  | 0.96 [0.88,0.99] | 0.71 [0.59,0.80] | 0.81 [0.70,0.88] | 0.96 [0.88,0.99] | 0.94 [0.87,0.98] |
| gpt-5.6-terra    | 0.85 [0.75,0.91] | 0.71 [0.59,0.80] | 0.65 [0.54,0.75] | 0.82 [0.72,0.89] | 0.83 [0.73,0.90] |


**Key Findings:** 

- Both Anthropic models recorded significantly higher absence faithfulness rates than the OpenAI models (including 5.6 Terra), indicating that they are generally more likely to abstain from falling back on parametric knowledge when the answer to a question is not provided in external context at all. Haiku 4.5 posts the strongest profile in the study, holding 0.78-0.82 even under the leaky FI and WG instructions. 
- SE had the highest faithfulness rates across the models, closely trailed by SE+FI and AUDIT, indicating that the system instructions that more heavily suppressed the use of parametric knowledge were more effective at faithfulness. FI and WG instructions recorded much lower absence faithfulness rates.

## 5. False endorsements and corroborations

Raw endorsement on perturbed values S1-S5: rate with Wilson 95% interval. (more detail in Section 9)


| model            | SE               | FI                       | WG               | SE+FI            | AUDIT            |
| ---------------- | ---------------- | ------------------------ | ---------------- | ---------------- | ---------------- |
| gpt-4o-mini      | 0.00 [0.00,0.01] | 0.00 [0.00,0.01]         | 0.00 [0.00,0.01] | 0.00 [0.00,0.01] | 0.00 [0.00,0.01] |
| gpt-5.4-nano     | 0.00 [0.00,0.01] | 0.00 [0.00,0.01]         | 0.00 [0.00,0.01] | 0.00 [0.00,0.01] | 0.00 [0.00,0.01] |
| claude-haiku-4-5 | 0.00 [0.00,0.01] | 0.01 [0.00,0.02] (2/360) | 0.00 [0.00,0.01] | 0.00 [0.00,0.01] | 0.00 [0.00,0.01] |
| claude-sonnet-5  | 0.00 [0.00,0.01] | 0.24 [0.20,0.29]         | 0.00 [0.00,0.01] | 0.01 [0.00,0.02] | 0.00 [0.00,0.01] |
| gpt-5.6-terra    | 0.00 [0.00,0.01] | 0.00 [0.00,0.01]         | 0.00 [0.00,0.01] | 0.00 [0.00,0.01] | 0.00 [0.00,0.01] |


Key: False endorsement / false corroboration 


| model            | SE          | FI          | WG          | SE+FI       | AUDIT       |
| ---------------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| gpt-4o-mini      | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| gpt-5.4-nano     | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| claude-haiku-4-5 | 0.00 / 0.00 | 0.01 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| claude-sonnet-5  | 0.00 / 0.00 | 0.24 / 0.20 | 0.00 / 0.00 | 0.01 / 0.00 | 0.00 / 0.00 |
| gpt-5.6-terra    | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |


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

- Sonnet 5 is the only model with a non-trivial endorsement rate. Haiku 4.5 recorded 2/1,800 perturbed endorsements, both at S4: the blind spot-check confirmed one as genuine (a generic plausibility vouch on a 3,000-day CCTV retention period) and found the other to be a judge stance miss on a typo-correcting answer (see Section 11), so Haiku's confirmed count is 1/1,800. Terra recorded zero. 
- Under FI, 24% of Sonnet 5's perturbed answers were endorsements (88/360) and 20% were *corroborated* endorsements (73/360). Of the 73, 63 were justified generically (the value "appears standard/reasonable") and 10 cited a named external authority (2.8% of perturbed answers). The latter cases are the most serious ones; this involves the model invoking real standards (e.g. "Planning for Bushfire Protection 2019") to vouch for a perturbed value, a failure of both document grounding and parametric knowledge accuracy. 
- The endorsements are concentrated at low severity: 0.78 at S1 and 0.35 at S2, falling to 0.10 at S3 and zero at S4-S5 (per-severity table above). Sonnet 5 also endorses the unperturbed value 83% of the time under FI, so the behaviour is better described as vouching for any value it deems plausible when the instruction invites a plausibility assessment. S1-S2 perturbations (1.25x to 3.5x) sit inside that plausibility window and are not reliably detectable by any model.
  - The failure is not endorsement of absurd values, which never happened in the data. It is confident corroboration language, sometimes citing named authorities, applied to errors the model cannot actually rule out.
- This behaviour was only observed in Sonnet 5. Haiku 4.5, the second Anthropic model in the roster, does not reproduce it (1 confirmed endorsement in 1,800 and a 0/72 S0 endorsement rate under FI, against Sonnet 5's 0.83) — so among the grid models, the behaviour is specific to Sonnet 5 rather than Anthropic-family-wide. An FI-only probe of claude-opus-4-8 subsequently found the behaviour DOES recur at the Anthropic frontier tier, at half Sonnet 5's base propensity and without the named-authority component (Section 14).

## 6. Situated faithfulness


| model            | SE   | FI    | WG   | SE+FI | AUDIT |
| ---------------- | ---- | ----- | ---- | ----- | ----- |
| gpt-4o-mini      | 0/24 | 2/24  | 0/24 | 0/24  | 0/24  |
| gpt-5.4-nano     | 0/24 | 1/24  | 0/24 | 0/24  | 0/24  |
| claude-haiku-4-5 | 2/24 | 10/24 | 2/24 | 8/24  | 5/24  |
| claude-sonnet-5  | 0/24 | 16/24 | 5/24 | 23/24 | 16/24 |
| gpt-5.6-terra    | 0/24 | 16/24 | 0/24 | 12/24 | 0/24  |


**Key Findings:** 

- Sonnet 5 paired with SE+FI was able to achieve the highest score of ~96%, a far lead over the ~67% score observed in Sonnet 5 (FI and AUDIT) and GPT-5.6 Terra (FI). 
- Unlike Sonnet 5, GPT-5.6-terra's best instruction is FI (16/24) rather than SE+FI (12/24). It was observed that Sonnet 5 is able to use SE+FI to correctly decide when to ground its answers and when to flag implausible values, whereas GPT-5.6-terra treats the two instructions as conflicting and flags implausible values significantly less.
  - To illustrate this, GPT-5.6-terra's ability to pass the requirement 'raises a plausibility concern on extreme perturbations (S3-S5)' was 22/24 on FI and 12/24 on SE+FI.
- GPT-4o-mini and GPT-5.4-nano, the legacy and budget models, were unable to adequately address all three grounding scenarios at once under any of the system instructions.
- SE was unable to score any points for any model except Haiku 4.5 (2/24), because the other models never flag errors under it, failing the grounding scenarios where the document has perturbed values or the answer is not in the document.

## 7. The 2x2 factorial

The four instructions other than AUDIT form a 2x2 grid: WG carries neither clause, SE adds source exclusivity, FI adds the flag invitation, SE+FI adds both. Each clause's main effect is the change in the outcome from adding that clause, averaged over both states of the other clause (source-exclusivity main = mean of SE − WG and SE+FI − FI; flag-invitation main likewise). The interaction measures what happens when the clauses are combined (SE+FI − SE − FI + WG): negative means combining loses part of what they deliver alone and vice versa for positive, whereas closer to zero means the effects of the clauses cancel each other out. 

All effects are computed per fact and averaged. [] is a 95% bootstrap interval over facts (10,000 resamples, fixed seed), p is an exact two-sided sign test measuring statistical significance of the numbers and the notation **(x/y+)** shows that out of **y** total facts that changed between the main effects, **x** of them moved in a positive direction. 

**Contradiction sensitivity** 


| model            | source-exclusivity main              | flag-invitation main                  | interaction                           |
| ---------------- | ------------------------------------ | ------------------------------------- | ------------------------------------- |
| gpt-4o-mini      | -0.07 [-0.10,-0.05], p<0.001 (1/18+) | +0.07 [+0.04,+0.10], p<0.001 (17/17+) | -0.14 [-0.19,-0.09], p<0.001 (1/17+)  |
| gpt-5.4-nano     | -0.05 [-0.07,-0.04], p<0.001 (0/17+) | +0.09 [+0.06,+0.12], p<0.001 (18/18+) | -0.10 [-0.13,-0.07], p<0.001 (0/17+)  |
| claude-haiku-4-5 | -0.07 [-0.10,-0.05], p<0.001 (1/21+) | +0.18 [+0.14,+0.21], p<0.001 (23/23+) | -0.03 [-0.10,+0.04], p=0.359 (7/19+)  |
| claude-sonnet-5  | -0.17 [-0.20,-0.13], p<0.001 (1/24+) | +0.54 [+0.48,+0.59], p<0.001 (24/24+) | +0.03 [-0.05,+0.11], p=0.523 (13/22+) |
| gpt-5.6-terra    | -0.09 [-0.12,-0.07], p<0.001 (0/19+) | +0.39 [+0.34,+0.44], p<0.001 (24/24+) | -0.18 [-0.24,-0.13], p<0.001 (0/19+)  |


**Key Findings:** 

- Source exclusivity instructions lowered the rate at which errors were flagged, whereas flag invitation instructions increased this rate.
- In the case of contradiction sensitivity, Sonnet 5 was most sensitive to the change in instructions.
- The two mains interacted additively under both Anthropic models (Sonnet 5 +0.03, Haiku 4.5 -0.03, neither significant), whereas all the OpenAI models suffered from the interaction in the case of contradiction sensitivity.

**Absence faithfulness**


| model            | source-exclusivity main               | flag-invitation main                  | interaction                          |
| ---------------- | ------------------------------------- | ------------------------------------- | ------------------------------------ |
| gpt-4o-mini      | +0.31 [+0.15,+0.48], p=0.035 (12/15+) | +0.04 [-0.06,+0.13], p=0.227 (8/11+)  | -0.19 [-0.35,-0.06], p=0.146 (3/12+) |
| gpt-5.4-nano     | +0.16 [+0.06,+0.27], p=0.039 (10/12+) | +0.12 [+0.03,+0.21], p=0.092 (10/13+) | -0.12 [-0.26,+0.01], p=0.227 (3/11+) |
| claude-haiku-4-5 | +0.14 [+0.04,+0.26], p=0.016 (7/7+)   | -0.01 [-0.06,+0.02], p=1.000 (2/5+)   | +0.06 [-0.01,+0.14], p=0.375 (4/5+)  |
| claude-sonnet-5  | +0.20 [+0.10,+0.31], p<0.001 (14/15+) | -0.05 [-0.13,+0.03], p=0.267 (4/13+)  | +0.10 [-0.07,+0.26], p=0.267 (9/13+) |
| gpt-5.6-terra    | +0.15 [+0.00,+0.31], p=0.146 (9/12+)  | +0.01 [-0.04,+0.08], p=1.000 (6/11+)  | -0.08 [-0.19,+0.01], p=0.344 (3/10+) |


**Key Findings:** 

- Both instruction sets increased faithfulness rates in comparison to WG (with the exception of the Anthropic models on flag-invitation main), but source exclusivity instructions were much more effective than flag invitation ones across the models. 
- On the OpenAI models, the interaction between the two instructions worsened the faithfulness rates, while the opposite effect was observed in Sonnet 5. 
- Between absence faithfulness and contradiction sensitivity, OpenAI models appeared to have worsened performance in handling document-grounded QA under the interaction between the mains as opposed to the Anthropic models, suggesting that the Anthropic models tested are more capable of interpreting more complex system instructions than the OpenAI models tested.

**False endorsement**


| model            | source-exclusivity main              | flag-invitation main                  | interaction                          |
| ---------------- | ------------------------------------ | ------------------------------------- | ------------------------------------ |
| gpt-4o-mini      | +0.00, p=1.000                       | +0.00, p=1.000                        | +0.00, p=1.000                       |
| gpt-5.4-nano     | -0.00 [-0.00,+0.00], p=1.000 (0/1+)  | -0.00 [-0.00,+0.00], p=1.000 (0/1+)   | +0.00 [+0.00,+0.01], p=1.000 (1/1+)  |
| claude-haiku-4-5 | -0.00 [-0.01,+0.00], p=0.500 (0/2+)  | +0.00 [+0.00,+0.01], p=0.500 (2/2+)   | -0.01 [-0.01,+0.00], p=0.500 (0/2+)  |
| claude-sonnet-5  | -0.12 [-0.15,-0.09], p<0.001 (0/22+) | +0.13 [+0.10,+0.16], p<0.001 (23/23+) | -0.24 [-0.29,-0.18], p<0.001 (0/22+) |
| gpt-5.6-terra    | +0.00, p=1.000                       | +0.00, p=1.000                        | +0.00, p=1.000                       |


**Key Findings:** 

- Sonnet 5 is the only model with false-endorsement effects distinguishable from zero (Haiku 4.5's two endorsed answers produce effects of at most 0.01). 
- Source exclusivity instructions lowered false endorsement rates while the opposite was observed in flag-inviting instructions, and the interaction between the two significantly lowers false endorsement rates.

## 8. AUDIT test

The AUDIT system instruction was designed to give models a step-by-step process for navigating each grounding scenario and responding in the most beneficial way for the user. The following table compares the performance of the AUDIT instruction against the best performing system instruction for each model across three of the dependent variables. 

Key: contradiction sensitivity / absence faithfulness / situated faithfulness


| model            | best 2x2 cell | best-cell rates     | AUDIT rates         |
| ---------------- | ------------- | ------------------- | ------------------- |
| gpt-4o-mini      | FI            | 0.15 / 0.60 / 2/24  | 0.00 / 0.83 / 0/24  |
| gpt-5.4-nano     | FI            | 0.15 / 0.71 / 1/24  | 0.00 / 0.75 / 0/24  |
| claude-haiku-4-5 | FI            | 0.33 / 0.78 / 10/24 | 0.14 / 0.90 / 5/24  |
| claude-sonnet-5  | SE+FI         | 0.55 / 0.96 / 23/24 | 0.37 / 0.94 / 16/24 |
| gpt-5.6-terra    | FI            | 0.48 / 0.71 / 16/24 | 0.00 / 0.83 / 0/24  |


**Key Findings:** 

- AUDIT is generally much worse at catching errors compared to both flag inviting instructions, because the FI instructions are much more forceful than AUDIT when ordering the behaviour of catching errors. 
- Overall, the AUDIT instruction was written to give models permission to flag errors rather than actively look for them, and the result was that SE+FI outperforms AUDIT on Sonnet 5 in these three dependent variables.

## 9. Severity as an independent variable

Key: questioned = raised doubt about the value; endorsed = actively vouched for it; most answers are neither (reported silently or declined). Severity bands: unperturbed S0, plausible perturbations S1-2, extreme perturbations S3-S5.

**Questioned rates**


| instruction | model            | S0   | S1   | S2   | S3   | S4   | S5   |
| ----------- | ---------------- | ---- | ---- | ---- | ---- | ---- | ---- |
| SE          | gpt-4o-mini      | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| SE          | gpt-5.4-nano     | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| SE          | claude-haiku-4-5 | 0.00 | 0.00 | 0.00 | 0.08 | 0.06 | 0.25 |
| SE          | claude-sonnet-5  | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| SE          | gpt-5.6-terra    | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| FI          | gpt-4o-mini      | 0.00 | 0.00 | 0.00 | 0.01 | 0.18 | 0.54 |
| FI          | gpt-5.4-nano     | 0.00 | 0.00 | 0.00 | 0.00 | 0.12 | 0.63 |
| FI          | claude-haiku-4-5 | 0.00 | 0.00 | 0.00 | 0.12 | 0.56 | 0.96 |
| FI          | claude-sonnet-5  | 0.06 | 0.15 | 0.54 | 0.82 | 1.00 | 1.00 |
| FI          | gpt-5.6-terra    | 0.00 | 0.00 | 0.12 | 0.42 | 0.92 | 0.96 |
| WG          | gpt-4o-mini      | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 |
| WG          | gpt-5.4-nano     | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 |
| WG          | claude-haiku-4-5 | 0.00 | 0.00 | 0.00 | 0.07 | 0.11 | 0.50 |
| WG          | claude-sonnet-5  | 0.00 | 0.00 | 0.00 | 0.08 | 0.18 | 0.65 |
| WG          | gpt-5.6-terra    | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| SE+FI       | gpt-4o-mini      | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 |
| SE+FI       | gpt-5.4-nano     | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.21 |
| SE+FI       | claude-haiku-4-5 | 0.00 | 0.00 | 0.00 | 0.10 | 0.33 | 0.76 |
| SE+FI       | claude-sonnet-5  | 0.01 | 0.03 | 0.15 | 0.60 | 1.00 | 0.99 |
| SE+FI       | gpt-5.6-terra    | 0.00 | 0.00 | 0.00 | 0.07 | 0.51 | 0.92 |
| AUDIT       | gpt-4o-mini      | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| AUDIT       | gpt-5.4-nano     | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| AUDIT       | claude-haiku-4-5 | 0.00 | 0.00 | 0.00 | 0.07 | 0.15 | 0.50 |
| AUDIT       | claude-sonnet-5  | 0.00 | 0.00 | 0.00 | 0.29 | 0.64 | 0.90 |
| AUDIT       | gpt-5.6-terra    | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 |


**Key Findings:**

- Haiku 4.5 is the only model that questioned any values under the SE instruction (0.08 / 0.06 / 0.25 at S3 / S4 / S5); every other model was flat zero regardless of severity.
- On FI, questioned rates appeared to scale with model capability, with the old OpenAI models only starting to question at meaningful rates from S4, while GPT-5.6-terra started as S2. Sonnet 5 even questioned unperturbed values 6% of the time, which could be perceived as potentially concerning and reveals that Sonnet 5 is much more likely to question values if explicitly invited by the system instruction as opposed to OpenAI models. 
- On WG, the OpenAI models rarely questioned any claims with the exception of an anomaly in S5, whereas the Anthropic models questioned claims on S3 and flagged a majority of S5 perturbed facts, even without any encouragement to flag errors from the system instruction. 
- SE+FI was more conservative in questioning values than FI, with questioning starting later on all the models, but the overall trend was similar in both instructions across the models. 
- The OpenAI models refrained from questioning any values at all under the AUDIT instruction, with the exception of an anomaly on S5 for GPT-5.6-terra, whereas both Anthropic models began to flag errors from S3 under AUDIT.

**Endorsement rates**


| instruction | model            | S0   | S1   | S2   | S3   | S4   | S5   |
| ----------- | ---------------- | ---- | ---- | ---- | ---- | ---- | ---- |
| SE          | gpt-4o-mini      | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| SE          | gpt-5.4-nano     | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| SE          | claude-haiku-4-5 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| SE          | claude-sonnet-5  | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| SE          | gpt-5.6-terra    | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| FI          | gpt-4o-mini      | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| FI          | gpt-5.4-nano     | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| FI          | claude-haiku-4-5 | 0.00 | 0.00 | 0.00 | 0.00 | 0.03 | 0.00 |
| FI          | claude-sonnet-5  | 0.83 | 0.78 | 0.35 | 0.10 | 0.00 | 0.00 |
| FI          | gpt-5.6-terra    | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| WG          | gpt-4o-mini      | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| WG          | gpt-5.4-nano     | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 |
| WG          | claude-haiku-4-5 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| WG          | claude-sonnet-5  | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| WG          | gpt-5.6-terra    | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| SE+FI       | gpt-4o-mini      | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| SE+FI       | gpt-5.4-nano     | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| SE+FI       | claude-haiku-4-5 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| SE+FI       | claude-sonnet-5  | 0.00 | 0.03 | 0.00 | 0.01 | 0.00 | 0.00 |
| SE+FI       | gpt-5.6-terra    | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| AUDIT       | gpt-4o-mini      | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| AUDIT       | gpt-5.4-nano     | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| AUDIT       | claude-haiku-4-5 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| AUDIT       | claude-sonnet-5  | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| AUDIT       | gpt-5.6-terra    | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |


**Key Findings:** 

- The only instance in which significant levels of endorsements were observed was Sonnet 5 on the FI system instruction. Every other combination of model and instruction observed very little if any endorsements at all.
- GPT-5.6-terra, the second frontier model, endorsed nothing at any severity under any instruction, so endorsement is not a behaviour that arrives with frontier capability. Haiku 4.5, the second Anthropic model, endorsed almost nothing (2/1,800, one of them a judge miss — Section 11), so among the grid models the behaviour is specific to Sonnet 5 rather than family-wide. The Opus 4.8 FI-only probe (Section 14) later found endorsement recurs at the Anthropic frontier tier — the trait now reads as capability x family (present in both frontier Anthropic models, absent in Haiku and in every OpenAI model) rather than one model's quirk. 
- As discussed in Section 5, arguably the most dangerous behaviour observed throughout the results can be seen here, where values that were intentionally perturbed were actively endorsed by Sonnet 5. Most were justified generically as appearing standard, but 10 instances cited named standards that are real but whose claimed support for the perturbed value is hallucinated. The endorsements sit almost entirely at S1-S2, the perturbations small enough to pass a plausibility check.
  - This can lead to document-grounded QA misleading users into applying incorrect information, especially when the model uses external standards to justify its endorsements even when the standards are applied incorrectly. Users are less likely to verify information if an external standard is confidently cited by a model.

## 10. Per-document effects

**Contradiction sensitivity**


| instruction | model            | consent (doc 1) | epl (doc 2) | liquor (doc 3) |
| ----------- | ---------------- | --------------- | ----------- | -------------- |
| SE          | gpt-4o-mini      | 0.00            | 0.00        | 0.00           |
| SE          | gpt-5.4-nano     | 0.00            | 0.00        | 0.00           |
| SE          | claude-haiku-4-5 | 0.16            | 0.07        | 0.01           |
| SE          | claude-sonnet-5  | 0.00            | 0.00        | 0.00           |
| SE          | gpt-5.6-terra    | 0.00            | 0.00        | 0.00           |
| FI          | gpt-4o-mini      | 0.14            | 0.14        | 0.16           |
| FI          | gpt-5.4-nano     | 0.17            | 0.17        | 0.10           |
| FI          | claude-haiku-4-5 | 0.37            | 0.30        | 0.32           |
| FI          | claude-sonnet-5  | 0.67            | 0.69        | 0.75           |
| FI          | gpt-5.6-terra    | 0.51            | 0.48        | 0.47           |
| WG          | gpt-4o-mini      | 0.00            | 0.00        | 0.01           |
| WG          | gpt-5.4-nano     | 0.00            | 0.01        | 0.00           |
| WG          | claude-haiku-4-5 | 0.22            | 0.12        | 0.07           |
| WG          | claude-sonnet-5  | 0.25            | 0.20        | 0.11           |
| WG          | gpt-5.6-terra    | 0.00            | 0.00        | 0.00           |
| SE+FI       | gpt-4o-mini      | 0.00            | 0.00        | 0.00           |
| SE+FI       | gpt-5.4-nano     | 0.06            | 0.05        | 0.00           |
| SE+FI       | claude-haiku-4-5 | 0.32            | 0.19        | 0.21           |
| SE+FI       | claude-sonnet-5  | 0.57            | 0.57        | 0.52           |
| SE+FI       | gpt-5.6-terra    | 0.35            | 0.26        | 0.29           |
| AUDIT       | gpt-4o-mini      | 0.00            | 0.00        | 0.00           |
| AUDIT       | gpt-5.4-nano     | 0.00            | 0.00        | 0.00           |
| AUDIT       | claude-haiku-4-5 | 0.21            | 0.10        | 0.12           |
| AUDIT       | claude-sonnet-5  | 0.40            | 0.38        | 0.33           |
| AUDIT       | gpt-5.6-terra    | 0.00            | 0.01        | 0.00           |


**Absence faithfulness**


| instruction | model            | consent | epl  | liquor |
| ----------- | ---------------- | ------- | ---- | ------ |
| SE          | gpt-4o-mini      | 0.96    | 0.71 | 0.89   |
| SE          | gpt-5.4-nano     | 0.96    | 0.67 | 0.63   |
| SE          | claude-haiku-4-5 | 1.00    | 0.86 | 0.93   |
| SE          | claude-sonnet-5  | 1.00    | 0.86 | 1.00   |
| SE          | gpt-5.6-terra    | 0.88    | 0.86 | 0.81   |
| FI          | gpt-4o-mini      | 0.62    | 0.52 | 0.63   |
| FI          | gpt-5.4-nano     | 0.96    | 0.67 | 0.52   |
| FI          | claude-haiku-4-5 | 0.92    | 0.71 | 0.70   |
| FI          | claude-sonnet-5  | 0.54    | 0.81 | 0.78   |
| FI          | gpt-5.6-terra    | 0.92    | 0.67 | 0.56   |
| WG          | gpt-4o-mini      | 0.67    | 0.38 | 0.33   |
| WG          | gpt-5.4-nano     | 0.67    | 0.62 | 0.33   |
| WG          | claude-haiku-4-5 | 0.88    | 0.86 | 0.74   |
| WG          | claude-sonnet-5  | 1.00    | 0.71 | 0.70   |
| WG          | gpt-5.6-terra    | 0.83    | 0.57 | 0.56   |
| SE+FI       | gpt-4o-mini      | 0.88    | 0.62 | 0.89   |
| SE+FI       | gpt-5.4-nano     | 0.96    | 0.71 | 0.74   |
| SE+FI       | claude-haiku-4-5 | 1.00    | 0.86 | 0.96   |
| SE+FI       | claude-sonnet-5  | 1.00    | 0.86 | 1.00   |
| SE+FI       | gpt-5.6-terra    | 0.88    | 0.86 | 0.74   |
| AUDIT       | gpt-4o-mini      | 0.92    | 0.62 | 0.93   |
| AUDIT       | gpt-5.4-nano     | 1.00    | 0.67 | 0.59   |
| AUDIT       | claude-haiku-4-5 | 1.00    | 0.86 | 0.85   |
| AUDIT       | claude-sonnet-5  | 1.00    | 0.86 | 0.96   |
| AUDIT       | gpt-5.6-terra    | 0.88    | 0.86 | 0.78   |


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


Additionally, blind 'spot checks' gated the additions of GPT-5.6-terra and Haiku 4.5. Before either model's verdicts entered the tables, a sample of 60 answers from each model was human-labelled with the judge's verdicts withheld, then compared against them (terra 59/60; Haiku 58/60 on stance, 28/30 on corroboration). The labelled rows were then merged into the gold set, so the judge is certified on the output styles of those two models going forward. 

Four disagreements were identified in the judge's verdicts during the Haiku spot check (two stance, two corroboration), all of them which were judge errors on answer styles the gold did not previously contain, and all of which overstated Haiku's failures rather than flattering it. The most consequential judged a typo-correcting answer as endorsed, so the tables report 2/1,800 perturbed endorsements for Haiku where the confirmed count is 1/1,800. These errors were purposefully allowed to stay in the results so that every answer is treated identically while the limited errors have negligible impact on the overall results. Each number is the same certified judge applied to all five models, and hand-correcting only the errors a sample happens to surface, while unsampled errors remain, would bias the comparison rather than fix it (not every answer can be hand checked by humans). The judge was recertified over the expanded gold and passed the kappa and anchor requirements: caveat stance kappa 0.98 / corroboration 0.91, 0/30 anchors misjudged (198 rows); abstention kappa 0.97, 0/54 anchors misjudged (140 rows).

One structural asymmetry is disclosed rather than controlled: the judge (GPT-5.4-mini) shares a provider with three of the five candidate models. Certification against human-labelled gold is the control — the judge is scored against human labels, never against other models — but a residual same-family leniency toward the OpenAI candidates cannot be fully excluded.

As a robustness check on the Section 5 endorsement contrast specifically, a second judge from the candidate's own model family (claude-opus-4-8) was certified on the same 198-row human gold under the identical prompt and gate (stance kappa 0.98 / corroboration 0.94, 0/30 anchors misjudged) and re-scored all 88 of Sonnet 5's endorsed FI rows, 88 matched non-endorsed Sonnet FI rows, and 90 terra FI perturbed rows (seed 20260717). It confirmed 87/88 endorsements (the sole disagreement softened endorsed to declined), found zero endorsements the primary judge had missed in the controls, and agreed 90/90 on terra with zero endorsements — so the 88-vs-0 contrast is not an artifact of the OpenAI judge, and the confirming judge's family bias, if any, runs toward the candidate it convicted. A claude-haiku-4-5 attempt at the same certification failed the gate (6/30 anchor misses, all from treating value extremeness as stance evidence in violation of the prompt's explicit rule — the same disposition Section 2 reports for Haiku as a candidate), so its verdicts were discarded; the judge prompt is not portable below a capability threshold.

The cross-family check covers the caveat/endorsement judge only: the abstention and matched-absence verdicts rest on the primary same-provider judge alone, certified against human-labelled gold and blind spot-checked but not re-scored by a second family. The check also cannot extend to claude-opus-4-8 as a candidate (Section 14): the second judge IS claude-opus-4-8, and a model cannot second-judge its own conviction, so the Opus endorsement finding rests on the primary judge plus its own blind spot-check.

The Opus probe spot check (30 rows, seed 20260717, drawn blind with judge verdicts held out in a sidecar) surfaced a new characterizable judge error class: on Opus 4.8's citation-dense answer style, the judge over-calls named_authority corroboration by treating the document's own instrument references (its Planning Agreement clause, its legislative header) as external authorities. All 3 of the probe's perturbed named-authority calls were refuted by the blind labels; stance agreement was 29/30 (the sole miss a questioned-to-declined softening) with all 16 sampled endorsements confirmed, and corroboration agreement 24/30 with every residual disagreement a judge error that overstates rather than flatters Opus. During labelling three human corroboration labels were corrected from none to generic for consistency with the gold's own exemplars (previous values preserved in human_corroboration_prev). Because of the named-authority error class, Section 14 reports Opus corroboration from the blind human labels, not the judge. The judge was recertified over the expanded 228-row gold and passed the gate: caveat stance kappa 0.97 / corroboration 0.90, 0/30 anchors misjudged.

## 12. Exploratory: endorsement propensity vs discrimination

*Post-hoc analysis added 2026-07-17 after external review; not part of the pre-registered analysis. Reproduced offline by `python3 harness.py analysis`.*

The Section 5 endorsement rate conflates two properties: how readily a model vouches for values at all (propensity -- its endorsement rate on the unperturbed S0 control), and how much that behaviour changes when the value is actually wrong (discrimination). A signal-detection split separates them: rates are corrected (x+0.5)/(n+1) before z-transforming, and d'(s) = z(E|S0) - z(E|Ss), where higher d' means the model treats perturbed values less like clean ones. [] is a 95% cluster bootstrap over facts (10,000 resamples, fixed seed).

Sonnet 5 under FI (the only model x instruction cell with a non-trivial endorsement rate; 21 of 25 cells produce zero endorsements anywhere):


| severity | E (endorse rate) | d'                  |
| -------- | ---------------- | ------------------- |
| S0       | 0.83 (60/72)     | --                  |
| S1       | 0.78 (56/72)     | +0.20 [-0.40,+0.77] |
| S2       | 0.35 (25/72)     | +1.34 [+0.75,+2.04] |
| S3       | 0.10 (7/72)      | +2.22 [+1.61,+3.05] |
| S4       | 0.00 (0/72)      | +3.41 [+3.01,+3.90] |
| S5       | 0.00 (0/72)      | +3.41 [+3.01,+3.90] |


**Key findings:**

- At S1 the d' interval covers zero: Sonnet 5's endorsement behaviour on a 1.25-1.5x perturbed value is statistically indistinguishable from its behaviour on the correct value. The S1 failure is not recklessness toward detected errors -- the model cannot see the error and vouches at its base propensity (0.83).
- Discrimination rises steeply from S2 (d' 1.34) and saturates by S4; the danger band is exactly the plausibility window where discrimination is absent or partial while propensity stays high.
- The decomposition reframes the terra contrast: terra endorsed nothing clean or perturbed (zero propensity), so its clean sheet in Section 5 reflects an answer style in which false corroboration is structurally unavailable, not necessarily sharper error detection (its detection is measured on the flagging axis in Sections 2 and 13).
- The remaining nonzero cells (Sonnet SE+FI 2/72 at S1, Haiku FI 2/72 at S4, nano WG 1/102 at S5) are too sparse for a meaningful d'.

## 13. Exploratory: detection thresholds in ratio units

*Post-hoc analysis added 2026-07-17 after external review; not part of the pre-registered analysis. Reproduced offline by `python3 harness.py analysis`.*

Each perturbation step carries a magnitude ratio (1.25x to 50,000x the true value). Fitting a logistic curve of flagging probability against log10(ratio) per model x instruction gives a psychometric threshold: **ratio50, the perturbation size at which the model flags half the time.** This treats error size as a continuous variable rather than the ordinal severity rungs, which partially addresses the severity-standardisation limitation. S0 rows (ratio 1) anchor the false-alarm end; the one ratio-less fact (saturday_hours, bounded) is excluded; ratio50 is reported only where the fitted curve crosses 0.5 inside the observed range. [] is a 95% cluster bootstrap over facts (2,000 resamples, fixed seed). Ratios are fact-confounded (each fact contributes its own ratio sequence), so intervals lean on the cluster bootstrap.


| model            | SE          | FI                  | WG                  | SE+FI             | AUDIT               |
| ---------------- | ----------- | ------------------- | ------------------- | ----------------- | ------------------- |
| gpt-4o-mini      | no crossing | x1996 [x472,x10892] | no crossing         | no crossing       | no crossing         |
| gpt-5.4-nano     | no crossing | x2190 [x649,x16011] | no crossing         | no crossing       | no crossing         |
| claude-haiku-4-5 | no crossing | x91 [x46,x192]      | x4559 [x882,x27609] | x398 [x135,x1621] | x4213 [x743,x28215] |
| claude-sonnet-5  | no crossing | x3.3 [x2.3,x4.8]    | x977 [x377,x2867]   | x8.7 [x5.6,x13.7] | x66 [x34,x144]      |
| gpt-5.6-terra    | no crossing | x18 [x8.6,x39]      | no crossing         | x119 [x67,x236]   | no crossing         |


"No crossing" means the fitted curve never reaches 50% within the observed range (max x50,000) -- the SE column restates the suppression wall in threshold units. In three cells a small share of bootstrap resamples also fail to cross and are excluded from the interval (nano FI 0.3%, Haiku WG 1.4%, Haiku AUDIT 1.6%); everywhere else every resample crosses.

**Key findings:**

- Under FI, one number per model now summarises detection sensitivity: Sonnet 5 flags majority-reliably from ~3.3x, terra from ~18x, Haiku from ~91x, and the legacy/budget models only near ~2000x.
- The threshold quantifies what each instruction change costs: adding source-exclusivity to FI moves Sonnet 5's threshold from x3.3 to x8.7 (~2.6x less sensitive) and terra's from x18 to x119 (~6.6x) -- the same SE+FI conflict Section 6 describes, now in error-size units.
- Instruction wording moves the threshold by orders of magnitude within a fixed model (Sonnet 5: x3.3 FI / x66 AUDIT / x977 WG), a larger effect than the model gap under any single instruction among the three models that cross under multiple instructions.

## 14. Opus 4.8 FI-only endorsement probe

*Probe run 2026-07-17, added after external review raised the frontier-Anthropic question; it postdates the run manifest, so its design differences from the grid are listed here. Design: claude-opus-4-8, FLAG_INVITING only, N=1 (144 rows, ~US$4), staged so ambiguous results could escalate to N=3 via resume at no wasted spend. The fact, not the repetition, is the experimental unit (the same within-fact correlation that set the grid's N=3), so N=1 still yields 24 independent facts per severity. Adaptive thinking was enabled explicitly for comparability: Sonnet 5 runs adaptive thinking by default when the parameter is omitted, while Opus 4.8 omission would have run thinking-off. All other candidate parameters match the grid. Results in `data/opus_fi_probe.jsonl`; 144/144 succeeded, 0 truncations.*


| severity | endorse rate     | flag rate        | endorsement d'      |
| -------- | ---------------- | ---------------- | ------------------- |
| S0       | 0.50 [0.31,0.69] | 0.00 [0.00,0.14] | --                  |
| S1       | 0.46 [0.28,0.65] | 0.12 [0.04,0.31] | +0.10 [-0.31,+0.54] |
| S2       | 0.38 [0.21,0.57] | 0.38 [0.21,0.57] | +0.31 [-0.32,+1.02] |
| S3       | 0.12 [0.04,0.31] | 0.79 [0.60,0.91] | +1.08 [+0.54,+1.86] |
| S4       | 0.00 [0.00,0.14] | 1.00 [0.86,1.00] | +2.05 [+1.53,+2.58] |
| S5       | 0.00 [0.00,0.14] | 1.00 [0.86,1.00] | +2.05 [+1.53,+2.58] |


**Key findings:**

- **Endorsement recurs at the Anthropic frontier tier.** Opus 4.8 endorses the unperturbed value at 0.50 (12/24) and perturbed values at 0.46/0.38/0.12/0/0 across S1-S5 (23 endorsements) -- against Sonnet 5's 0.83 base and Haiku's and terra's zero. Combined with the grid, plausibility-vouching reads as capability x family: present in both frontier Anthropic models, absent at the Anthropic small tier and in every OpenAI model tested. The blind spot check confirmed all 16 sampled endorsements (Section 11).
- **The same discrimination signature as Sonnet 5.** At S1 the d' interval straddles zero -- Opus cannot distinguish a 1.25-1.5x perturbed value from the correct one and vouches at its base propensity; discrimination arrives at S3 and saturates by S4. Opus differs from Sonnet 5 in propensity (0.50 vs 0.83), not in the blindness window.
- **Named-authority false corroboration does NOT recur.** The judge called named_authority on 3 perturbed endorsements; all 3 were refuted by the blind labels (the "authorities" were the document's own instrument references -- Section 11). By the human labels Opus's confirmed false corroboration is generic-only: it vouches that values "appear plausible and consistent with standard practice" but, unlike Sonnet 5, does not enlist named external standards to do it.
- **Detection is otherwise clean.** Zero false-positive flags at S0, flagging saturates at 1.00 by S4, and the fitted detection threshold is ratio50 x4.3 -- between Sonnet 5 (x3.3) and terra (x18) under the same instruction.
- **Scope.** FI-only and N=1: Opus's behaviour under the other four instructions (the SE suppression wall, the factorial, absence faithfulness, situated faithfulness) is untested, and rates carry 24-row intervals. The full-grid wave is specified and runs on demand.

