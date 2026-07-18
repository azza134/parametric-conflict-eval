# Results

**Models tested:** gpt-4o-mini (gpt-4o-mini-2024-07-18), gpt-5.4-nano (gpt-5.4-nano-2026-03-17), gpt-5.6-terra (resolved alias), claude-haiku-4-5 (claude-haiku-4-5-20251001), claude-sonnet-5 (resolved alias), claude-opus-4-8 (resolved alias) under FLAG_INVITING only at N=1.  
**Judge:** GPT-5.4-mini (gpt-5.4-mini-2026-03-17): caveat stance kappa 0.97 / corroboration 0.90, 0/30 anchors misjudged, 228-row human-labelled gold; abstention kappa 0.97, 0/54 anchors misjudged, 140-row human-labelled gold.  
**Certification gate**: zero anchor misses AND kappa >= 0.80.  

*Regenerate every table offline with `python3 harness.py analysis`, and the severity figure with `python3 plot_results.py`.*

## 1. Key Terms

Definitions of the outcomes are as follows:

**1. Error Flagging:** Rate at which the model flags a planted wrong value in the document, on perturbed rows (S1-S5).
**2. Clean Specificity:** Rate at which the model does not question or decline the document value at severity 0 (unperturbed).  
**3. Absence Faithfulness:** Rate at which the model abstains from supplying an answer if the answer to a question about the document is removed.  
**4. False Endorsement:** Rate at which the model actively endorses perturbed values (S1 to S5).  
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

Other terms:

- **parametric leakage**, the rate at which a model answers from its own pre-training data when the instruction requires grounding.
- **adoption**, which value the answer contains, matched lexically against the document's 'target' perturbed value and the true value: target_only, true_only, both or neither. 
  - A **silent override** is true_only on a perturbed row where the model did not question or decline, correcting the document without saying so.

## 2.1 Error flagging


| model            | SE                       | FI               | WG               | SE+FI            | AUDIT                    |
| ---------------- | ------------------------ | ---------------- | ---------------- | ---------------- | ------------------------ |
| gpt-4o-mini      | 0.00 [0.00,0.01] (0/360) | 0.15 [0.12,0.18] | 0.00 [0.00,0.01] | 0.00 [0.00,0.01] | 0.00 [0.00,0.01]         |
| gpt-5.4-nano     | 0.00 [0.00,0.01]         | 0.15 [0.12,0.18] | 0.00 [0.00,0.01] | 0.04 [0.03,0.06] | 0.00 [0.00,0.01]         |
| gpt-5.6-terra    | 0.00 [0.00,0.01]         | 0.48 [0.43,0.53] | 0.00 [0.00,0.01] | 0.30 [0.25,0.35] | 0.00 [0.00,0.02] (1/360) |
| claude-haiku-4-5 | 0.08 [0.05,0.11]         | 0.33 [0.28,0.38] | 0.14 [0.11,0.18] | 0.24 [0.20,0.29] | 0.14 [0.11,0.19]         |
| claude-sonnet-5  | 0.00 [0.00,0.01]         | 0.70 [0.65,0.75] | 0.18 [0.15,0.23] | 0.55 [0.50,0.60] | 0.37 [0.32,0.42]         |
| claude-opus-4-8  | --                       | 0.66 [0.57,0.74] | --               | --               | --                       |


**Key Findings:** 

- SE suppresses error flagging for every model except Haiku 4.5.
- All the models caught the most errors on FI, with a lower but still significant amount of errors still being caught under the SE+FI instruction.
- Error-flagging rates are significantly higher in Anthropic models than OpenAI models. 
- Interestingly, only the Anthropic models caught errors on the AUDIT instruction, a behavioural difference between the Anthropic and OpenAI models in responding to this instruction in terms of catching errors.

## 2.2 Adoption

2.1 tells you whether the model raised concerns about the implausibility of a value. 2.2 tells you which of the target or true value was actually used in the answer. This is an important metric because a model could flag nothing yet quietly insert the real value anyway. The closed-book probe does not cover Opus 4.8. 

**Key:** 

any-rep / prior known = model produced true value on at least one of three closed book attempts

any-rep / prior absent = model never produced true value on any of the three closed book attempts

majority / prior known = model produced true value on at least two of three closed book attempts

majority / prior absent = model produced true value on at most one of three closed book attempts 

Reproduce with `python3 harness.py adoption`; per-row output in `data/adoption_v2.jsonl`.


| aggregation | prior status | n    | true_only  | silent_override |
| ----------- | ------------ | ---- | ---------- | --------------- |
| any-rep     | prior known  | 2500 | 0.000 (0)  | 0.000 (0)       |
| any-rep     | prior absent | 7700 | 0.009 (70) | 0.009 (67)      |
| majority    | prior known  | 1800 | 0.000 (0)  | 0.000 (0)       |
| majority    | prior absent | 8400 | 0.008 (70) | 0.008 (67)      |


**Key findings:**

- Models never silently corrected an error that they provably could, as evidenced by the zero on either of the prior known rows. 
- All 70 true_only instances occurred on facts that the models did not have in parametric knowledge (prior absent). 
  - A majority of these instances were silent overrides.
  - A majority of these instances were outputs where the model could infer the true value from the passage itself without having to fall back on parametric knowledge anyways. 
    - This is because some of the perturbations resulted in obvious inconsistencies in the coherence of the document that the model could infer. These were the perturbations that formed a majority of the true_only instances.
- Under SE, GPT-5.4-nano, GPT-4o-mini and Sonnet 5 had zero silent overrides, meaning that the reason for the zero error-flagging on SE was **not** the result of quietly fixing the errors. GPT-5.6-terra and Haiku 4.5 only had five and six silent overrides on this instruction respectively as well, a negligible figure in the total n.

## 3. Clean specificity


| model            | SE               | FI               | WG               | SE+FI            | AUDIT            |
| ---------------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- |
| gpt-4o-mini      | 0.94 [0.88,0.97] | 1.00 [0.96,1.00] | 1.00 [0.96,1.00] | 0.97 [0.92,0.99] | 0.97 [0.90,0.99] |
| gpt-5.4-nano     | 0.95 [0.89,0.98] | 1.00 [0.96,1.00] | 1.00 [0.96,1.00] | 0.93 [0.87,0.97] | 1.00 [0.95,1.00] |
| gpt-5.6-terra    | 1.00 [0.95,1.00] | 1.00 [0.95,1.00] | 1.00 [0.95,1.00] | 1.00 [0.95,1.00] | 1.00 [0.95,1.00] |
| claude-haiku-4-5 | 1.00 [0.95,1.00] | 1.00 [0.95,1.00] | 1.00 [0.95,1.00] | 1.00 [0.95,1.00] | 1.00 [0.95,1.00] |
| claude-sonnet-5  | 1.00 [0.95,1.00] | 0.85 [0.75,0.91] | 1.00 [0.95,1.00] | 0.99 [0.93,1.00] | 1.00 [0.95,1.00] |
| claude-opus-4-8  | --               | 1.00 [0.86,1.00] | --               | --               | --               |


**Key Findings:**

- All the models had a similar performance across all instructions in abstaining from raising doubts about the unperturbed value, with the anomaly of 0.85 on Sonnet 5 (FI). 
- Between the anomalies in Sonnet 5's clean specificity on FI and error flagging on AUDIT, Sonnet 5 can be seen from the data as being more likely to flag errors if the system instruction invites such behaviour, even values that are correct and unperturbed (0.85 clean specificity).

## 4. Absence faithfulness


| model            | SE               | FI               | WG               | SE+FI            | AUDIT            |
| ---------------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- |
| gpt-4o-mini      | 0.86 [0.76,0.92] | 0.60 [0.48,0.70] | 0.46 [0.35,0.57] | 0.81 [0.70,0.88] | 0.83 [0.73,0.90] |
| gpt-5.4-nano     | 0.75 [0.64,0.84] | 0.71 [0.59,0.80] | 0.53 [0.41,0.64] | 0.81 [0.70,0.88] | 0.75 [0.64,0.84] |
| gpt-5.6-terra    | 0.85 [0.75,0.91] | 0.71 [0.59,0.80] | 0.65 [0.54,0.75] | 0.82 [0.72,0.89] | 0.83 [0.73,0.90] |
| claude-haiku-4-5 | 0.93 [0.85,0.97] | 0.78 [0.67,0.86] | 0.82 [0.72,0.89] | 0.94 [0.87,0.98] | 0.90 [0.81,0.95] |
| claude-sonnet-5  | 0.96 [0.88,0.99] | 0.71 [0.59,0.80] | 0.81 [0.70,0.88] | 0.96 [0.88,0.99] | 0.94 [0.87,0.98] |


**Key Findings:** 

- Both Anthropic models recorded significantly higher absence faithfulness rates than the OpenAI models (including 5.6 Terra), indicating that they are generally more likely to abstain from falling back on parametric knowledge when the answer to a question is not provided in external context at all. Haiku 4.5 posts the strongest profile in the study, holding 0.78-0.82 even under the leaky FI and WG instructions. 
- SE had the highest faithfulness rates across the models, closely trailed by SE+FI and AUDIT, indicating that the system instructions that more heavily suppressed the use of parametric knowledge were more effective at faithfulness. FI and WG instructions recorded much lower absence faithfulness rates.

## 5. False endorsements and corroborations

Raw endorsement on perturbed values S1-S5.


| model            | SE               | FI                        | WG               | SE+FI            | AUDIT            |
| ---------------- | ---------------- | ------------------------- | ---------------- | ---------------- | ---------------- |
| gpt-4o-mini      | 0.00 [0.00,0.01] | 0.00 [0.00,0.01]          | 0.00 [0.00,0.01] | 0.00 [0.00,0.01] | 0.00 [0.00,0.01] |
| gpt-5.4-nano     | 0.00 [0.00,0.01] | 0.00 [0.00,0.01]          | 0.00 [0.00,0.01] | 0.00 [0.00,0.01] | 0.00 [0.00,0.01] |
| gpt-5.6-terra    | 0.00 [0.00,0.01] | 0.00 [0.00,0.01]          | 0.00 [0.00,0.01] | 0.00 [0.00,0.01] | 0.00 [0.00,0.01] |
| claude-haiku-4-5 | 0.00 [0.00,0.01] | 0.01 [0.00,0.02] (2/360)  | 0.00 [0.00,0.01] | 0.00 [0.00,0.01] | 0.00 [0.00,0.01] |
| claude-sonnet-5  | 0.00 [0.00,0.01] | 0.24 [0.20,0.29]          | 0.00 [0.00,0.01] | 0.01 [0.00,0.02] | 0.00 [0.00,0.01] |
| claude-opus-4-8  | --               | 0.19 [0.13,0.27] (23/120) | --               | --               | --               |


Key: False endorsement / false corroboration 


| model            | SE          | FI          | WG          | SE+FI       | AUDIT       |
| ---------------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| gpt-4o-mini      | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| gpt-5.4-nano     | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| gpt-5.6-terra    | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| claude-haiku-4-5 | 0.00 / 0.00 | 0.01 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| claude-sonnet-5  | 0.00 / 0.00 | 0.24 / 0.20 | 0.00 / 0.00 | 0.01 / 0.00 | 0.00 / 0.00 |
| claude-opus-4-8  | --          | 0.19 / 0.12 | --          | --          | --          |


Endorsements per severity under FI, for the two models that endorse: Sonnet 5 (n=72 per severity) and the Opus 4.8 probe (n=24 per severity). The probe's design is recorded in the `probes` block of `data/run_manifest.json`; adaptive thinking was enabled explicitly for comparability, since Sonnet 5 runs it by default while an Opus 4.8 call omitting the parameter would run thinking-off. 

The d' column is a signal-detection decomposition: rates are corrected (x+0.5)/(n+1) before z-transforming, d'(s) = z(E|S0) - z(E|Ss) and [] is a 95% cluster bootstrap over facts (10,000 resamples, fixed seed). In simple terms, d' measures whether a model can differentiate between a perturbed value and an unperturbed one, where higher d' means it has a higher ability to differentiate between the two. 


| severity | Sonnet 5 endorsed | Sonnet 5 rate [Wilson 95%] | Sonnet 5 d'         | Opus 4.8 endorsed | Opus 4.8 rate [Wilson 95%] | Opus 4.8 d'         |
| -------- | ----------------- | -------------------------- | ------------------- | ----------------- | -------------------------- | ------------------- |
| S0       | 60/72             | 0.83 [0.73,0.90]           | --                  | 12/24             | 0.50 [0.31,0.69]           | --                  |
| S1       | 56/72             | 0.78 [0.67,0.86]           | +0.20 [-0.40,+0.77] | 11/24             | 0.46 [0.28,0.65]           | +0.10 [-0.31,+0.54] |
| S2       | 25/72             | 0.35 [0.25,0.46]           | +1.34 [+0.75,+2.04] | 9/24              | 0.38 [0.21,0.57]           | +0.31 [-0.32,+1.02] |
| S3       | 7/72              | 0.10 [0.05,0.19]           | +2.22 [+1.61,+3.05] | 3/24              | 0.12 [0.04,0.31]           | +1.08 [+0.54,+1.86] |
| S4       | 0/72              | 0.00 [0.00,0.05]           | +3.41 [+3.01,+3.90] | 0/24              | 0.00 [0.00,0.14]           | +2.05 [+1.53,+2.58] |
| S5       | 0/72              | 0.00 [0.00,0.05]           | +3.41 [+3.01,+3.90] | 0/24              | 0.00 [0.00,0.14]           | +2.05 [+1.53,+2.58] |


**Key Findings:** 

- Sonnet 5 and Opus 4.8 are the only models with non-trivial endorsement rates, indicating that this is a behaviour that surfaces among Anthropic's frontier models, almost exclusively under the FI instruction (Haiku 4.5's confirmed count is 1 in 1,800; Sonnet 5's only off-FI endorsements are 2/72 under SE+FI). 
- False corroborations are the most dangerous behaviour. Of Sonnet 5's 73 corroborated endorsements, 63 were generic ("appears standard/reasonable") and 10 cited a named real standard (e.g. "Planning for Bushfire Protection 2019") whose support for the perturbed value is hallucinated. 
  - All of Opus 4.8's 15 recorded corroborations were generic (the judge called 3 of them named-authority, but a blind spot-check refuted all 3 — the "authorities" were the document's own instrument references), indicating that citing real standards incorrectly was a behaviour exclusive to Sonnet 5.

**Disclaimer:**

The endorsement rate includes not just false endorsements but also when the model vouches for the correct, unperturbed value (S0). Sonnet and Opus endorse at similar rates between S0 and S1, which means they cannot see the error. In contrast, GPT-5.6-Terra doesn't exhibit endorsements as a behaviour at all, which means it is not necessarily better at error detection, it is just not prone to this specific failure mode. 

To verify that the false endorsements were not the result of a cross-provider judge (an OpenAI model judging Anthropic models), Opus 4.8 acted as a second judge, certified on the same gold, and agreed with the OpenAI judge on 87 of Sonnet 5's 88 false endorsements while finding none the primary judge missed. 

## 6. Situated faithfulness


| model            | SE   | FI    | WG   | SE+FI | AUDIT |
| ---------------- | ---- | ----- | ---- | ----- | ----- |
| gpt-4o-mini      | 0/24 | 2/24  | 0/24 | 0/24  | 0/24  |
| gpt-5.4-nano     | 0/24 | 1/24  | 0/24 | 0/24  | 0/24  |
| gpt-5.6-terra    | 0/24 | 16/24 | 0/24 | 12/24 | 0/24  |
| claude-haiku-4-5 | 2/24 | 10/24 | 2/24 | 8/24  | 5/24  |
| claude-sonnet-5  | 0/24 | 16/24 | 5/24 | 23/24 | 16/24 |


**Key Findings:** 

- Sonnet 5 paired with SE+FI was able to achieve the highest score of ~96%, a far lead over the ~67% score observed in Sonnet 5 (FI and AUDIT) and GPT-5.6 Terra (FI). 
- Unlike Sonnet 5, GPT-5.6-terra's best instruction is FI (16/24) rather than SE+FI (12/24). It was observed that Sonnet 5 is able to use SE+FI to correctly decide when to ground its answers and when to flag implausible values, whereas GPT-5.6-terra treats the two instructions as conflicting and flags implausible values significantly less.
  - To illustrate this, GPT-5.6-terra's ability to pass the requirement 'raises a plausibility concern on extreme perturbations (S3-S5)' was 22/24 on FI and 12/24 on SE+FI.
- GPT-4o-mini and GPT-5.4-nano, the legacy and budget models, were unable to adequately address all three grounding scenarios at once under any of the system instructions.
- SE was unable to score any points for any model except Haiku 4.5 (2/24), because the other models never flag errors under it, failing the requirement 'raises a plausibility concern on extreme perturbations (S3-S5)'.

## 7. The 2x2 factorial

The four instructions other than AUDIT form a 2x2 grid: WG carries neither clause, SE adds source exclusivity, FI adds the flag invitation, SE+FI adds both. Each clause's main effect is the change in the outcome from adding that clause, averaged over both states of the other clause (source-exclusivity main = mean of SE − WG and SE+FI − FI; flag-invitation main likewise). The interaction measures what happens when the clauses are combined (SE+FI − SE − FI + WG): negative means combining loses part of what they deliver alone and vice versa for positive, whereas closer to zero means the effects of the clauses cancel each other out. 

All effects are computed per fact and averaged. [] is a 95% bootstrap interval over facts (10,000 resamples, fixed seed), p is an exact two-sided sign test measuring statistical significance of the numbers and the notation **(x/y+)** shows that out of **y** total facts that changed between the main effects, **x** of them moved in a positive direction. 

**Error flagging** 


| model            | source-exclusivity main              | flag-invitation main                  | interaction                           |
| ---------------- | ------------------------------------ | ------------------------------------- | ------------------------------------- |
| gpt-4o-mini      | -0.07 [-0.10,-0.05], p<0.001 (1/18+) | +0.07 [+0.04,+0.10], p<0.001 (17/17+) | -0.14 [-0.19,-0.09], p<0.001 (1/17+)  |
| gpt-5.4-nano     | -0.05 [-0.07,-0.04], p<0.001 (0/17+) | +0.09 [+0.06,+0.12], p<0.001 (18/18+) | -0.10 [-0.13,-0.07], p<0.001 (0/17+)  |
| gpt-5.6-terra    | -0.09 [-0.12,-0.07], p<0.001 (0/19+) | +0.39 [+0.34,+0.44], p<0.001 (24/24+) | -0.18 [-0.24,-0.13], p<0.001 (0/19+)  |
| claude-haiku-4-5 | -0.07 [-0.10,-0.05], p<0.001 (1/21+) | +0.18 [+0.14,+0.21], p<0.001 (23/23+) | -0.03 [-0.10,+0.04], p=0.359 (7/19+)  |
| claude-sonnet-5  | -0.17 [-0.20,-0.13], p<0.001 (1/24+) | +0.54 [+0.48,+0.59], p<0.001 (24/24+) | +0.03 [-0.05,+0.11], p=0.523 (13/22+) |


**Key Findings:** 

- Source exclusivity instructions lowered the rate at which errors were flagged, whereas flag invitation instructions increased this rate.
- In the case of error flagging, Sonnet 5 was most sensitive to the change in instructions.
- The two mains interacted additively under both Anthropic models (Sonnet 5 +0.03, Haiku 4.5 -0.03, neither significant), whereas all the OpenAI models suffered from the interaction in the case of error flagging.

**Absence faithfulness**


| model            | source-exclusivity main               | flag-invitation main                  | interaction                          |
| ---------------- | ------------------------------------- | ------------------------------------- | ------------------------------------ |
| gpt-4o-mini      | +0.31 [+0.15,+0.48], p=0.035 (12/15+) | +0.04 [-0.06,+0.13], p=0.227 (8/11+)  | -0.19 [-0.35,-0.06], p=0.146 (3/12+) |
| gpt-5.4-nano     | +0.16 [+0.06,+0.27], p=0.039 (10/12+) | +0.12 [+0.03,+0.21], p=0.092 (10/13+) | -0.12 [-0.26,+0.01], p=0.227 (3/11+) |
| gpt-5.6-terra    | +0.15 [+0.00,+0.31], p=0.146 (9/12+)  | +0.01 [-0.04,+0.08], p=1.000 (6/11+)  | -0.08 [-0.19,+0.01], p=0.344 (3/10+) |
| claude-haiku-4-5 | +0.14 [+0.04,+0.26], p=0.016 (7/7+)   | -0.01 [-0.06,+0.02], p=1.000 (2/5+)   | +0.06 [-0.01,+0.14], p=0.375 (4/5+)  |
| claude-sonnet-5  | +0.20 [+0.10,+0.31], p<0.001 (14/15+) | -0.05 [-0.13,+0.03], p=0.267 (4/13+)  | +0.10 [-0.07,+0.26], p=0.267 (9/13+) |


**Key Findings:** 

- Both instruction sets increased faithfulness rates in comparison to WG (with the exception of the Anthropic models on flag-invitation main), but source exclusivity instructions were much more effective than flag invitation ones across the models. 
- On the OpenAI models, the interaction between the two instructions worsened the faithfulness rates, while the opposite effect was observed in Sonnet 5. 
- Between absence faithfulness and error flagging, OpenAI models appeared to have worsened performance in handling document-grounded QA under the interaction between the mains as opposed to the Anthropic models, suggesting that the Anthropic models tested are more capable of interpreting more complex system instructions than the OpenAI models tested.

**False endorsement**


| model            | source-exclusivity main              | flag-invitation main                  | interaction                          |
| ---------------- | ------------------------------------ | ------------------------------------- | ------------------------------------ |
| gpt-4o-mini      | +0.00, p=1.000                       | +0.00, p=1.000                        | +0.00, p=1.000                       |
| gpt-5.4-nano     | -0.00 [-0.00,+0.00], p=1.000 (0/1+)  | -0.00 [-0.00,+0.00], p=1.000 (0/1+)   | +0.00 [+0.00,+0.01], p=1.000 (1/1+)  |
| gpt-5.6-terra    | +0.00, p=1.000                       | +0.00, p=1.000                        | +0.00, p=1.000                       |
| claude-haiku-4-5 | -0.00 [-0.01,+0.00], p=0.500 (0/2+)  | +0.00 [+0.00,+0.01], p=0.500 (2/2+)   | -0.01 [-0.01,+0.00], p=0.500 (0/2+)  |
| claude-sonnet-5  | -0.12 [-0.15,-0.09], p<0.001 (0/22+) | +0.13 [+0.10,+0.16], p<0.001 (23/23+) | -0.24 [-0.29,-0.18], p<0.001 (0/22+) |


**Key Findings:** 

- Sonnet 5 is the only model with false-endorsement effects distinguishable from zero (Haiku 4.5's two endorsed answers produce effects of at most 0.01). 
- Source exclusivity instructions lowered false endorsement rates while the opposite was observed in flag-inviting instructions, and the interaction between the two significantly lowers false endorsement rates.

## 8. AUDIT test

The AUDIT system instruction was designed to give models a step-by-step process for navigating each grounding scenario and responding in the most beneficial way for the user. The following table compares the performance of the AUDIT instruction against the best performing system instruction for each model across three of the dependent variables. 

Key: error flagging / absence faithfulness / situated faithfulness


| model            | best 2x2 cell | best-cell rates     | AUDIT rates         |
| ---------------- | ------------- | ------------------- | ------------------- |
| gpt-4o-mini      | FI            | 0.15 / 0.60 / 2/24  | 0.00 / 0.83 / 0/24  |
| gpt-5.4-nano     | FI            | 0.15 / 0.71 / 1/24  | 0.00 / 0.75 / 0/24  |
| gpt-5.6-terra    | FI            | 0.48 / 0.71 / 16/24 | 0.00 / 0.83 / 0/24  |
| claude-haiku-4-5 | FI            | 0.33 / 0.78 / 10/24 | 0.14 / 0.90 / 5/24  |
| claude-sonnet-5  | SE+FI         | 0.55 / 0.96 / 23/24 | 0.37 / 0.94 / 16/24 |


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
| SE          | gpt-5.6-terra    | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| SE          | claude-haiku-4-5 | 0.00 | 0.00 | 0.00 | 0.08 | 0.06 | 0.25 |
| SE          | claude-sonnet-5  | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| FI          | gpt-4o-mini      | 0.00 | 0.00 | 0.00 | 0.01 | 0.18 | 0.54 |
| FI          | gpt-5.4-nano     | 0.00 | 0.00 | 0.00 | 0.00 | 0.12 | 0.63 |
| FI          | gpt-5.6-terra    | 0.00 | 0.00 | 0.12 | 0.42 | 0.92 | 0.96 |
| FI          | claude-haiku-4-5 | 0.00 | 0.00 | 0.00 | 0.12 | 0.56 | 0.96 |
| FI          | claude-sonnet-5  | 0.06 | 0.15 | 0.54 | 0.82 | 1.00 | 1.00 |
| FI          | claude-opus-4-8  | 0.00 | 0.12 | 0.38 | 0.79 | 1.00 | 1.00 |
| WG          | gpt-4o-mini      | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 |
| WG          | gpt-5.4-nano     | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 |
| WG          | gpt-5.6-terra    | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| WG          | claude-haiku-4-5 | 0.00 | 0.00 | 0.00 | 0.07 | 0.11 | 0.50 |
| WG          | claude-sonnet-5  | 0.00 | 0.00 | 0.00 | 0.08 | 0.18 | 0.65 |
| SE+FI       | gpt-4o-mini      | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 |
| SE+FI       | gpt-5.4-nano     | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.21 |
| SE+FI       | gpt-5.6-terra    | 0.00 | 0.00 | 0.00 | 0.07 | 0.51 | 0.92 |
| SE+FI       | claude-haiku-4-5 | 0.00 | 0.00 | 0.00 | 0.10 | 0.33 | 0.76 |
| SE+FI       | claude-sonnet-5  | 0.01 | 0.03 | 0.15 | 0.60 | 1.00 | 0.99 |
| AUDIT       | gpt-4o-mini      | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| AUDIT       | gpt-5.4-nano     | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| AUDIT       | gpt-5.6-terra    | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 |
| AUDIT       | claude-haiku-4-5 | 0.00 | 0.00 | 0.00 | 0.07 | 0.15 | 0.50 |
| AUDIT       | claude-sonnet-5  | 0.00 | 0.00 | 0.00 | 0.29 | 0.64 | 0.90 |


**Key Findings:**

- On FI, questioned rates appeared to scale with model capability, with the old OpenAI models only starting to question at meaningful rates from S4, while GPT-5.6-terra started at S2. Sonnet 5 even questioned unperturbed values 6% of the time, which could be perceived as potentially concerning and reveals that Sonnet 5 is much more likely to question values if explicitly invited by the system instruction as opposed to OpenAI models. 
- On WG, the OpenAI models rarely questioned any claims with the exception of an anomaly in S5, whereas the Anthropic models questioned claims on S3 and flagged a majority of S5 perturbed facts, even without any encouragement to flag errors from the system instruction. 
- SE+FI was more conservative in questioning values than FI, with questioning starting later on all the models, but the overall trend was similar in both instructions across the models. 
- The OpenAI models refrained from questioning any values at all under the AUDIT instruction, with the exception of an anomaly on S5 for GPT-5.6-terra, whereas both Anthropic models began to flag errors from S3 under AUDIT.

**Endorsement rates**


| instruction | model            | S0   | S1   | S2   | S3   | S4   | S5   |
| ----------- | ---------------- | ---- | ---- | ---- | ---- | ---- | ---- |
| SE          | gpt-4o-mini      | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| SE          | gpt-5.4-nano     | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| SE          | gpt-5.6-terra    | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| SE          | claude-haiku-4-5 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| SE          | claude-sonnet-5  | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| FI          | gpt-4o-mini      | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| FI          | gpt-5.4-nano     | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| FI          | gpt-5.6-terra    | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| FI          | claude-haiku-4-5 | 0.00 | 0.00 | 0.00 | 0.00 | 0.03 | 0.00 |
| FI          | claude-sonnet-5  | 0.83 | 0.78 | 0.35 | 0.10 | 0.00 | 0.00 |
| FI          | claude-opus-4-8  | 0.50 | 0.46 | 0.38 | 0.12 | 0.00 | 0.00 |
| WG          | gpt-4o-mini      | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| WG          | gpt-5.4-nano     | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 |
| WG          | gpt-5.6-terra    | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| WG          | claude-haiku-4-5 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| WG          | claude-sonnet-5  | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| SE+FI       | gpt-4o-mini      | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| SE+FI       | gpt-5.4-nano     | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| SE+FI       | gpt-5.6-terra    | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| SE+FI       | claude-haiku-4-5 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| SE+FI       | claude-sonnet-5  | 0.00 | 0.03 | 0.00 | 0.01 | 0.00 | 0.00 |
| AUDIT       | gpt-4o-mini      | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| AUDIT       | gpt-5.4-nano     | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| AUDIT       | gpt-5.6-terra    | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| AUDIT       | claude-haiku-4-5 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| AUDIT       | claude-sonnet-5  | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |


**Key Findings:** 

- Endorsements as a behaviour occur from S0 to S3, which is evidently where Opus 4.8 and Sonnet 5 perceive the perturbations to remain plausible. Endorsements do not occur on S4 and S5, so these are the points of perturbation where the models do not believe the perturbations to be plausible.

**Detection thresholds** 

This table displays at which perturbation multiplier the model begins to flag errors more than 50% of the time. 

Calculation: a logistic curve of flagging probability against log10(perturbation ratio) is fitted per model x instruction, and **ratio50** is where the fitted curve crosses 0.5. "no crossing" means it never reaches 0.5 inside the observed range (up to x50,000). [] is a 95% cluster bootstrap over facts.


| model            | SE          | FI                  | WG                  | SE+FI             | AUDIT               |
| ---------------- | ----------- | ------------------- | ------------------- | ----------------- | ------------------- |
| gpt-4o-mini      | no crossing | x1996 [x472,x10892] | no crossing         | no crossing       | no crossing         |
| gpt-5.4-nano     | no crossing | x2190 [x649,x16011] | no crossing         | no crossing       | no crossing         |
| gpt-5.6-terra    | no crossing | x18 [x8.6,x39]      | no crossing         | x119 [x67,x236]   | no crossing         |
| claude-haiku-4-5 | no crossing | x91 [x46,x192]      | x4559 [x882,x27609] | x398 [x135,x1621] | x4213 [x743,x28215] |
| claude-sonnet-5  | no crossing | x3.3 [x2.3,x4.8]    | x977 [x377,x2867]   | x8.7 [x5.6,x13.7] | x66 [x34,x144]      |
| claude-opus-4-8  | --          | x4.3 [x2.9,x6.7]    | --                  | --                | --                  |


## 10. Per-document effects

**Error flagging**


| instruction | model            | consent (doc 1) | epl (doc 2) | liquor (doc 3) |
| ----------- | ---------------- | --------------- | ----------- | -------------- |
| SE          | gpt-4o-mini      | 0.00            | 0.00        | 0.00           |
| SE          | gpt-5.4-nano     | 0.00            | 0.00        | 0.00           |
| SE          | gpt-5.6-terra    | 0.00            | 0.00        | 0.00           |
| SE          | claude-haiku-4-5 | 0.16            | 0.07        | 0.01           |
| SE          | claude-sonnet-5  | 0.00            | 0.00        | 0.00           |
| FI          | gpt-4o-mini      | 0.14            | 0.14        | 0.16           |
| FI          | gpt-5.4-nano     | 0.17            | 0.17        | 0.10           |
| FI          | gpt-5.6-terra    | 0.51            | 0.48        | 0.47           |
| FI          | claude-haiku-4-5 | 0.37            | 0.30        | 0.32           |
| FI          | claude-sonnet-5  | 0.67            | 0.69        | 0.75           |
| FI          | claude-opus-4-8  | 0.68            | 0.60        | 0.69           |
| WG          | gpt-4o-mini      | 0.00            | 0.00        | 0.01           |
| WG          | gpt-5.4-nano     | 0.00            | 0.01        | 0.00           |
| WG          | gpt-5.6-terra    | 0.00            | 0.00        | 0.00           |
| WG          | claude-haiku-4-5 | 0.22            | 0.12        | 0.07           |
| WG          | claude-sonnet-5  | 0.25            | 0.20        | 0.11           |
| SE+FI       | gpt-4o-mini      | 0.00            | 0.00        | 0.00           |
| SE+FI       | gpt-5.4-nano     | 0.06            | 0.05        | 0.00           |
| SE+FI       | gpt-5.6-terra    | 0.35            | 0.26        | 0.29           |
| SE+FI       | claude-haiku-4-5 | 0.32            | 0.19        | 0.21           |
| SE+FI       | claude-sonnet-5  | 0.57            | 0.57        | 0.52           |
| AUDIT       | gpt-4o-mini      | 0.00            | 0.00        | 0.00           |
| AUDIT       | gpt-5.4-nano     | 0.00            | 0.00        | 0.00           |
| AUDIT       | gpt-5.6-terra    | 0.00            | 0.01        | 0.00           |
| AUDIT       | claude-haiku-4-5 | 0.21            | 0.10        | 0.12           |
| AUDIT       | claude-sonnet-5  | 0.40            | 0.38        | 0.33           |


**Absence faithfulness**


| instruction | model            | consent | epl  | liquor |
| ----------- | ---------------- | ------- | ---- | ------ |
| SE          | gpt-4o-mini      | 0.96    | 0.71 | 0.89   |
| SE          | gpt-5.4-nano     | 0.96    | 0.67 | 0.63   |
| SE          | gpt-5.6-terra    | 0.88    | 0.86 | 0.81   |
| SE          | claude-haiku-4-5 | 1.00    | 0.86 | 0.93   |
| SE          | claude-sonnet-5  | 1.00    | 0.86 | 1.00   |
| FI          | gpt-4o-mini      | 0.62    | 0.52 | 0.63   |
| FI          | gpt-5.4-nano     | 0.96    | 0.67 | 0.52   |
| FI          | gpt-5.6-terra    | 0.92    | 0.67 | 0.56   |
| FI          | claude-haiku-4-5 | 0.92    | 0.71 | 0.70   |
| FI          | claude-sonnet-5  | 0.54    | 0.81 | 0.78   |
| WG          | gpt-4o-mini      | 0.67    | 0.38 | 0.33   |
| WG          | gpt-5.4-nano     | 0.67    | 0.62 | 0.33   |
| WG          | gpt-5.6-terra    | 0.83    | 0.57 | 0.56   |
| WG          | claude-haiku-4-5 | 0.88    | 0.86 | 0.74   |
| WG          | claude-sonnet-5  | 1.00    | 0.71 | 0.70   |
| SE+FI       | gpt-4o-mini      | 0.88    | 0.62 | 0.89   |
| SE+FI       | gpt-5.4-nano     | 0.96    | 0.71 | 0.74   |
| SE+FI       | gpt-5.6-terra    | 0.88    | 0.86 | 0.74   |
| SE+FI       | claude-haiku-4-5 | 1.00    | 0.86 | 0.96   |
| SE+FI       | claude-sonnet-5  | 1.00    | 0.86 | 1.00   |
| AUDIT       | gpt-4o-mini      | 0.92    | 0.62 | 0.93   |
| AUDIT       | gpt-5.4-nano     | 1.00    | 0.67 | 0.59   |
| AUDIT       | gpt-5.6-terra    | 0.88    | 0.86 | 0.78   |
| AUDIT       | claude-haiku-4-5 | 1.00    | 0.86 | 0.85   |
| AUDIT       | claude-sonnet-5  | 1.00    | 0.86 | 0.96   |


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


Before a new model's results entered the tables, a sample of its answers was labelled by hand with the judge's verdicts hidden, then compared against them. The few disagreements were all judge errors that overstated the model's failures (Haiku 4.5's table shows 2/1,800 endorsements; the confirmed count is 1/1,800). These errors were left in the results on purpose: every answer is scored by the same judge, and correcting only the errors a sample happens to catch would bias the comparison. The judge was recertified on the expanded gold after each check. Either way, leaving the errors in the results would have had a negligible impact on the results. 

The judge (GPT-5.4-mini) shares a provider with three of the candidate models. It is certified against human labels, never against other models, but some leniency toward the OpenAI candidates cannot be fully ruled out. As a check, claude-opus-4-8 was certified on the same 140-row abstention gold (kappa 0.98, 0/54 anchors misjudged) and re-judged a 91-row sample (60 OpenAI faithful-labelled rows, 13 OpenAI ungrounded-labelled, 18 Anthropic-model controls): it agreed with the primary judge on all 91, overturning none.

The Opus 4.8 spot-check (30 rows, all 16 sampled endorsements confirmed) found one repeatable judge error. A named authority can appear in an answer in two ways: the model vouching from its own knowledge ("3 metres is consistent with Planning for Bushfire Protection 2019"), or the model pointing at the document's own references ("per the Planning Agreement referenced in the passage"). The judge cannot reliably tell these apart. Opus 4.8's citation-heavy answer style produces the second case constantly, and all 3 of the judge's named-authority calls on it were this mistake, so Section 5 reports Opus 4.8's corroboration from the human labels instead, the only time this is done. Sonnet 5's 10 named-authority rows are the first case: the human-labelled gold contains endorsed answers in this exact style and agrees with the judge, and Sonnet 5 calls a different perturbed value "consistent with" the same standard at each severity, which only vouching from its own knowledge can produce.

The adoption classifier is lexical, not judge-based. Its false-negative rate is estimated by an S0 canary — answered unperturbed rows should contain the document's value — at 17/2,029 = 0.008 (0.060 before format-variant handling). A 30-row blind spot-check scored 24/30, four formatting fixes were applied, and it scores 29/30 on the same sample; the one remaining miss class is fully verbalized numbers ("one per one million").