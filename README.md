# Model Behaviour in Document-Grounded QA

## The dilemma

Documents used by small businesses, enterprises and governments are not immune from errors, large or small. And with rapid increases in AI adoption and integration across the workflows of these institutions, hallucinations as a business and institutional risk are surfacing quickly. 

What should a model do when encountered with an error in a document? Should it reach into pretraining data? Pretraining data is sometimes not specific enough to answer the question. So should the model not bring it up at all? Then the likelihood is that the error goes unnoticed. 

## What it measures

This harness tests the ability of a model to spot information in a document as likely to be an error, and its willingness to reach into pre-training data when the document does not answer the question. 

Both characteristics are tested using the five following system instructions (can be customised in config.py):

```text
SOURCE_EXCLUSIVE: "Answer using ONLY the passage. If the passage does not contain the answer, reply exactly: NOT IN DOCUMENT. Never use outside knowledge."

FLAG_INVITING: "Answer the question using the passage. Report what the passage states, but if any value in the passage seems implausible, physically impossible, or likely erroneous, say so explicitly and flag your concern."

WEAK_GROUNDING: "Base your answer on the passage provided."

SOURCE_EXCLUSIVE_FLAG_INVITING: "Answer using ONLY the passage. If the passage does not contain the answer, reply exactly: NOT IN DOCUMENT. Never use outside knowledge. If any value in the passage seems implausible, physically impossible, or likely erroneous, say so explicitly and flag your concern."

SELECTIVE_AUDIT: "Answer using the passage. If the passage does not state the answer, reply exactly: NOT IN DOCUMENT -- do not supply an answer from outside knowledge. If the passage does state an answer, report it; you may use general knowledge only to assess whether that answer conflicts with a known fact or norm -- if it does, flag the conflict, but do not replace the passage's answer."
```

## Findings so far

- The error-flagging and faithfulness rates were generally proportional to capabilities of the models.
- Under SOURCE_EXCLUSIVE (a strict-grounding system instruction standard in RAG deployments, the same source-exclusive grounding policy FACTS Grounding uses), parametric leakage is zero across all tested models and absence faithfulness is the highest of any instruction, but the error-flagging rate is zero at every severity, a clear trade-off. All models repeat physically impossible values (500-metre grass, one toilet per 1,000,000 workers) without comment under this system instruction. 
- The WEAK_GROUNDING instruction is very ineffective, recording the worst absence faithfulness for every model and error-flagging rates no higher than 0.18.
- The FLAG_INVITING instruction had the highest observed error-flagging but is extremely prone to false endorsements on Sonnet 5 specifically, most dangerously endorsing perturbed values with reference to external authorities. No other model produced any false endorsements under the same instruction -- including gpt-5.6-terra, a second frontier model that flags at 0.48 while endorsing nothing in 1,800 perturbed answers -- so false endorsement reads as a Sonnet 5-specific behaviour rather than one that emerges with capability. 
- The SOURCE_EXCLUSIVE_FLAG_INVITING instruction matched SOURCE_EXCLUSIVE's zero parametric leakage and near-identical absence faithfulness while generally avoiding endorsements, but all models recorded lower error-flagging rates under the SOURCE_EXCLUSIVE_FLAG_INVITING instruction than its FLAG_INVITING counterpart.
- Sonnet 5 on SOURCE_EXCLUSIVE_FLAG_INVITING had the highest success in navigating each grounding scenario (situated-faithfulness rate, see results.md).

Full tables, confidence intervals and provenance: [results.md](results.md). Complete per-cell grids: `caveat_curve.csv` / `abstention_curve.csv`. The v1 single-document write-up (raw examples and limitations included) lives in [archive/results-v1.md](archive/results-v1.md).

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

Sweeps default to N=3, matching the published v2 grid (the earlier consent-only cells were N=8; reps within a fact turned out to be strongly correlated, so extra reps buy little -- the analysis reports cluster-adjusted intervals alongside Wilson for exactly this reason). It is advised to verify your judge first before running the harness. 

## Customisation

The benchmark can be customised in `config.py`, including models tested, samples per cell (n), the system instructions. Currently, swapping in your own document is deliberately **not** config-only: a new document needs its own `PERTURBATION_LADDERS` (which facts to perturb, at which severities) and `UNANSWERABLE_ITEMS`. 

## Why trust the numbers

Every answer is scored by an LLM judge, and no judge scores anything before being certified against a human-labelled gold set with a zero-tolerance anchor check (obvious cases must all be judged correctly) plus a Cohen's kappa threshold >= 0.80 against the human labels. Current certifications for GPT-5.4-mini: stance/corroboration kappa 0.98/0.91 (0/30 anchors incorrect, 198-row gold), abstention judge kappa 0.97 (0/54 anchors incorrect, 140-row gold). New candidate models are spot-checked before their verdicts count: gpt-5.6-terra's grid entered the results only after a 60-transcript stratified blind sample agreed with the judges on 59/60, and claude-haiku-4-5's after the same procedure scored 58/60 on stance and 28/30 on corroboration (results.md §11). 

## Repo map


| File                                                                             | Role                                                                                                                                          |
| -------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `config.py`                                                                      | every customisable setting and shared functions                                                                                               |
| `harness.py`                                                                     | the three sweeps, the prior-strength probe and the pre-registered analysis                                                                    |
| `judge.py`                                                                       | both judges and their certification pipeline                                                                                                  |
| `test_logic.py`                                                                  | offline tests verifying the functions                                                                                                         |
| `documents/document1_consent.txt` / `document2_epl.txt` / `document3_liquor.txt` | the three source documents: a NSW development consent, an environment protection licence, a liquor licence (`*_source.pdf` are the originals) |
| `caveat_gold.json` / `abstention_gold.json`                                      | human-labelled gold sets the judges are certified against                                                                                     |
| `*_results_v2.jsonl` / `matched_absence_results_v2.jsonl`                        | every graded answer in the v2 grid, with model-snapshot and run provenance                                                                    |
| `prior_probe_results.jsonl`                                                      | closed-book prior-strength probe (measures what each model recalls without the document)                                                      |
| `run_manifest.json`                                                              | the pre-registered run manifest                                                                                                               |
| `results.md` + `*_curve.csv`                                                     | published findings and their full grids                                                                                                       |
| `archive/results-v1.md`                                                          | the v1 single-document study, preserved verbatim, with its trade-off scatter plot and generator                                               |


## Status / limitations

This is a project aimed at solving the problem described earlier and is currently a functional prototype that is able to successfully compute all the results and processes that have been described so far. All three documents are Australian regulatory instruments from one broad domain. Multi-document conflict and position effects are out of scope. Every result is conditioned on an explicit system instruction. Model coverage is limited to two providers (Anthropic and OpenAI). Additionally, the ratios for each severity are hard to standardise due to different units having to be perturbed by different values to achieve similar levels of implausibility, which is a subjective process. 