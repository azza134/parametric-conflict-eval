# How System Instructions Affect LLM Behaviour in Document-Grounded QA

## The dilemma

Documents used by small businesses, enterprises and governments are not immune from errors, large or small. And with rapid increases in AI adoption and integration across the workflows of these institutions, hallucinations as a business and institutional risk are surfacing quickly. 

What should a model do when it encounters an error in a document? Should it reach into pretraining data? Pretraining data is sometimes not specific enough to answer the question. So should the model not bring it up at all? Then the likelihood is that the error goes unnoticed. 

## What it measures

This harness tests the ability of Anthropic and OpenAI LLMs to react appropriately to different grounding scenarios, using system instructions as an independent variable. In a RAG system, this repo measures only the generation step and assumes that all previous steps, including retrieval, were done perfectly.

These were the system instructions tested: 

```text
SOURCE_EXCLUSIVE: "Answer using ONLY the passage. If the passage does not contain the answer, reply exactly: NOT IN DOCUMENT. Never use outside knowledge."

FLAG_INVITING: "Answer the question using the passage. Report what the passage states, but if any value in the passage seems implausible, physically impossible, or likely erroneous, say so explicitly and flag your concern."

WEAK_GROUNDING: "Base your answer on the passage provided."

SOURCE_EXCLUSIVE_FLAG_INVITING: "Answer using ONLY the passage. If the passage does not contain the answer, reply exactly: NOT IN DOCUMENT. Never use outside knowledge. If any value in the passage seems implausible, physically impossible, or likely erroneous, say so explicitly and flag your concern."

SELECTIVE_AUDIT: "Answer using the passage. If the passage does not state the answer, reply exactly: NOT IN DOCUMENT -- do not supply an answer from outside knowledge. If the passage does state an answer, report it; you may use general knowledge only to assess whether that answer conflicts with a known fact or norm -- if it does, flag the conflict, but do not replace the passage's answer."
```

## Findings so far

- The error-flagging and faithfulness rates were generally proportional to the capabilities of the models.
- Under SOURCE_EXCLUSIVE (a strict-grounding system instruction standard in RAG deployments), absence faithfulness rates are at or near the top of every instruction (statistically tied with SOURCE_EXCLUSIVE_FLAG_INVITING), but the error-flagging rate is zero at every severity for every model except Haiku 4.5 (0.08), a clear trade-off. The other four models repeat physically impossible values (500-metre grass, one toilet per 1,000,000 workers) without comment under this system instruction. 
- The WEAK_GROUNDING instruction is very ineffective, recording the worst absence faithfulness of any instruction for the OpenAI models (the Anthropic models' worst is FLAG_INVITING) and error-flagging rates no higher than 0.18.
- The FLAG_INVITING instruction had the highest observed error-flagging but is extremely prone to false endorsements on Sonnet 5 specifically, most dangerously endorsing perturbed values with reference to external authorities, a behaviour labelled as false corroboration. No other model produced any false endorsements under the same instruction, including gpt-5.6-terra, a second frontier model that flags at 0.48 while endorsing nothing in 1,800 perturbed answers, making false endorsement absent from every OpenAI model and from the Anthropic small tier. An Opus 4.8 FI-only probe (N=1; results.md Section 12) found the endorsement behaviour recurs at the Anthropic frontier tier at half Sonnet 5's base propensity, though without the named-authority component -- so plausibility-vouching reads as an Anthropic-frontier-tier answer style rather than a Sonnet 5 quirk or a behaviour that emerges with capability generally. 
- The SOURCE_EXCLUSIVE_FLAG_INVITING instruction near-matched SOURCE_EXCLUSIVE's zero parametric leakage (2/72 leaks on Sonnet 5, zero on every other model) with near-identical absence faithfulness while generally avoiding endorsements, but all models recorded lower error-flagging rates under the SOURCE_EXCLUSIVE_FLAG_INVITING instruction than its FLAG_INVITING counterpart.
- Sonnet 5 on SOURCE_EXCLUSIVE_FLAG_INVITING had the highest success in navigating each grounding scenario (situated-faithfulness rate, see results.md).

## Quickstart

```
pip install -r requirements.txt
cp .env.example .env        # add OPENAI_API_KEY and ANTHROPIC_API_KEY -- the default MODELS roster in config.py includes both providers
python3 harness.py          # dry-run: prints the full design + call-count estimate, costs nothing
```

Real runs are explicit and resumable (partial results persist to disk after every call):

```
python3 harness.py caveat [N]       # error-flagging sweep
python3 harness.py abstention [N]   # parametric-leakage sweep
python3 harness.py absence [N]      # matched-absence sweep
python3 harness.py probe [N]        # closed-book prior-strength probe
python3 harness.py analysis         # full pre-registered readout (every table in results.md), no API calls
python3 harness.py vectors          # per-fact/per-item vectors + ICC per cell, no API calls
python3 judge.py caveat             # re-certify the caveat judge against its gold
python3 judge.py abstention         # re-certify the abstention judge
```

Sweeps default to N=3, matching the published v2 results. N=8 was used in v1, but reps within a fact were observed to be strongly correlated, making extra repetitions redundant, which is why the analysis reports cluster-adjusted intervals alongside Wilson). 

It is also advised to verify your judge first before running the harness. 

## Reproducing the published results

Every table in `results.md` and every figure regenerates offline from the committed result files, with no API keys:

```
pip install -r requirements.txt
python3 -m unittest test_logic     # 214 tests over the scoring and analysis logic
python3 harness.py analysis        # regenerates the published statistics from data/
python3 plot_results.py            # regenerates the figures
```

CI runs this exact sequence (fresh install, tests, full analysis readout) on every push, so a clean clone rebuilding the published numbers is continuously verified, not a one-time claim. API keys are only needed to run new sweeps, and re-running a sweep is not expected to reproduce the saved transcripts bit-for-bit: several roster entries are floating aliases and provider models drift over time (disclosed in `results.md`). The saved transcripts in `data/` are the frozen record; everything downstream of them is deterministic.

## Customisation

The benchmark can be customised in `config.py`, including models tested, number of repetitions (n) and the system instructions. 

Documents are config-only too: each document is a text file plus a JSON spec (`documents/<name>_spec.json`) holding its perturbation ladders (which facts to perturb, at which severities, with the exact find/replace strings), each fact's absence deletion, and its unanswerable items. Adding a document = drop in the two files and register both paths in `DOCUMENTS` and `DOCUMENT_SPECS` in `config.py`. Specs are shape-checked at load (missing keys and malformed replace pairs fail immediately with a per-fact message), and the dry-run verifies every find-string and target string against the actual document text before any API call. 

## Why trust the numbers

Every answer is scored by an LLM judge, and no judge scores anything before being certified against a human-labelled gold set with a zero-tolerance anchor check (obvious cases must all be judged correctly) plus a Cohen's kappa threshold >= 0.80 against the human labels. The certifications for the judge can be observed in `results.md`.

To show the headline endorsement finding is not a judge-family artifact, a second judge from the convicted candidate's own family (claude-opus-4-8) was certified on the same human gold under the identical prompt and gate, and upheld the endorsement contrast 264/266, showing that the cause for the endorsements should not attributed to self-preference bias.

## Repo map


| File                                                                             | Role                                                                                                                                          |
| -------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `config.py`                                                                      | every customisable setting and shared functions                                                                                               |
| `harness.py`                                                                     | the three sweeps, the prior-strength probe and the pre-registered analysis                                                                    |
| `judge.py`                                                                       | both judges and their certification pipeline                                                                                                  |
| `second_judge.py`                                                                | the cross-family second judge (claude-opus-4-8) that re-scored the endorsement rows                                                           |
| `spotcheck_sampler.py`                                                           | draws blind spot-check samples from newly added models and compares human labels against judge verdicts to be added to the gold set           |
| `test_logic.py`                                                                  | offline tests verifying the functions                                                                                                         |
| `documents/document1_consent.txt` / `document2_epl.txt` / `document3_liquor.txt` | the three source documents: a NSW development consent, an environment protection licence, a liquor licence (`*_source.pdf` are the originals) |
| `documents/*_spec.json`                                                          | per-document specs: perturbation ladders, absence deletions and unanswerable items                                                            |
| `plot_results.py`                                                                | regenerates the flag-rate-vs-severity figure in `figures/` from the committed results                                                         |
| `data/caveat_gold.json` / `data/abstention_gold.json`                            | human-labelled gold sets the judges are certified against                                                                                     |
| `data/*_results_v2.jsonl`                                                        | all v2 model outputs in full detail                                                                                                           |
| `data/prior_probe_results.jsonl`                                                 | measures what each model recalls without the document                                                                                         |
| `data/run_manifest.json`                                                         | the pre-registered run manifest                                                                                                               |
| `results.md`                                                                     | most recent published findings (v2)                                                                                                           |
| `archive/results-v1.md`                                                          | the v1 results                                                                                                                                |


## Status / limitations

- All three documents are Australian regulatory instruments from one broad domain. 
- Multi-document conflict and position effects are out of scope. 
- Every result is conditioned on an explicit system instruction. 
- Model coverage is limited to two providers (Anthropic and OpenAI). 
- The cross-family second judge covers the caveat/endorsement verdicts only; the abstention and matched-absence numbers rest on the primary same-provider judge (certified against human gold and blind spot-checked, but not second-judged). 
- The ratios for each severity are hard to standardise due to different units having to be perturbed by different values to achieve similar levels of implausibility, which is a subjective process. The detection-threshold analysis (results.md Section 9, ratio50) partially addresses this by treating the ratio itself as the continuous independent variable.

## Related work

- The closest prior work is [ClashEval](https://arxiv.org/abs/2404.10198) (Wu, Wu & Zou, NeurIPS 2024 Datasets & Benchmarks), which perturbs retrieved context at graded severities and finds adoption of the wrong value falls as implausibility rises. This harness independently converged on the same severity-ladder design and the overlap was only found after the design was implemented. 
  - However, the dependent variable differs: ClashEval measures which answer the model selects, while this harness builds on that by measuring what the model *says* about the value; flagging implausibility, staying silent, endorsing values or corroborating it with an external authority. 
  - The false-corroboration metric and the instruction factorial on speech-act outcomes appear in none of the knowledge-conflict work surveyed.
- The repo also sits adjacent and builds on other works in the knowledge-conflict field.
  - [Longpre et al. 2021](https://arxiv.org/abs/2109.05052) took QA examples where the external context mentioned an entity such as 'founded by Steve Jobs' and perturbed it with another entity such as Elon Musk to see if the model would fall back on its parametric knowledge or the external context, to which Longpre et al. found the model stubbornly preferred the former. 
    - [Xie et al. 2024](https://arxiv.org/abs/2305.13300) (ICLR) built on Longpre's work by hypothesising that this stubborness was because the perturbed evidence in isolation was unconvincing and created an incoherent external context, so instead of swapping one name, they generated whole coherent passages that conflicted with the model's parametric knowledge. The stubborness identified in Longpre's work was subsequently suppressed once the whole passage was made coherent. These two works laid the foundation for this repo's contradiction sensitivity test. Furthermore, Xie et al.'s independent variable was the quality of the conflicting evidence, that is to say the counter-memory passages had varying levels of conflict with the model's parametric knowledge, which retroactively parallels this repo's severity-ladder design (see section 9 of results.md). 
    - [ConflictBank](https://arxiv.org/abs/2408.12076) (NeurIPS 2024) scales this design into a large-scale benchmark, with up to 7.4M claim-evidence pairs across three conflict causes (misinformation, temporal drift, semantic ambiguity). This repo focuses mainly on misinformation perturbations. However, ConflictBank's dependent variable is ultimately also concerned with whether the model falls back on parametric knowledge or external context, which is one of the points where this repo differentiates itself.
  - [WikiContradict](https://arxiv.org/abs/2406.13805) (NeurIPS 2024) measures how models behave when presented with Wikipedia pages contradicting each other, collecting 253 contradictions. It was found that most of the time, models will commit to one answer and will not admit that there is conflicting information. It is one of the only papers in the field with a dependent variable that is not simply whether the model answers with its parametric knowledge or external context. However, it differs from this repo in that as far as WikiContradict is concerned, it does not matter which of the conflicting external sources is ground truth, whereas much of this repo is concerned how a model behaves when faced with ground truth or a perturbed value.
- [FACTS Grounding](https://arxiv.org/abs/2501.03200) (Google DeepMind, 2025) is a benchmark for how well a model sticks to a document that is presumed to be correct, using 1700 real-world documents of up to tens of thousands of tokens. All documents have different system instructions but a shared theme in that they all require the model to stay within the bounds of the document, similar to the SOURCE_EXCLUSIVE instruction in this repo, although FACTS Grounding has varying forcefulness in their instructions with regards to the model having to ground their responses in the documents. However, this repo is more multidimensional than FACTS Grounding in the sense that this repo accounts for the possibility of perturbed or insufficient contexts. 
- [RGB](https://arxiv.org/abs/2309.01431) (Chen et al., AAAI 2024) is a RAG benchmark that builds QAs on news that are past the training cutoffs of the tested models and retrieves real web documents for each question, monitoring noise robustness, information integration, negative rejection and counterfactual robustness, the last two of which are adjacent to this repo's absence faithfulness and contradiction sensitivity metrics. RGB is a broader benchmark on RAG systems, but as this repo focuses on the generation step of RAG, it uncovers findings that cannot be replicated in RGB, such as SOURCE_EXCLUSIVE actively suppressing error-flagging, the concept of false endorsements and corroborations and the severity-ladder which observes the point at which models begin to start detecting errors. 
  - The key difference between negative rejection and absence faithfulness is that negative rejection provides a completely irrelevant document, whereas absence faithfulness provides the relevant document but removes the answer to the question asked.

