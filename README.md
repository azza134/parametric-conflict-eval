# Faithfulness Evaluation Harness

## The dilemma

Documents used by small businesses, enterprises and governments are not immune from errors, large or small. And with rapid increases in AI adoption and integration across the workflows of these institutions, hallucinations as a business and institutional risk are surfacing quickly. 

What should a model do when encountered with an error in a document? Should it reach into pretraining data? Pretraining data is sometimes not specific enough to answer the question. So should the model not bring it up at all? Then the likelihood is that the error goes unnoticed. 

## What it measures

This harness tests the ability of a model to spot information in a document as likely to be an error. It also tests the willingness of a model to reach into its pretraining data to answer a question a user may ask that is not answered in the document. This is an important relationship because the aforementioned entities are deploying AI across their documents to provide specific responses that navigate gaps in pretrained data. However, the inevitability of errors in documents will result in a scenario where someone has to decide whether they would rather have the model spot errors or stick strictly to document-based retrieval. 

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
- Under SOURCE_EXCLUSIVE (a standard strict-grounding RAG system instruction), faithfulness rates are 100% across all tested models, but the error-flagging rate is zero at every severity, a clear trade-off. All models repeat physically impossible values (500-metre grass, one toilet per 1,000,000 workers) without comment under this system instruction. 
- The WEAK_GROUNDING instruction is very ineffective with no model achieving an average error-flagging OR faithfulness rate of over 50%.
- The FLAG_INVITING instruction had the highest observed error-flagging but is extremely prone to false endorsements on Sonnet 5 specifically, most dangerously endorsing perturbed values with reference to external authorities. On the other hand, the older GPT models tested on the FLAG_INVITING instruction did not generate any false endorsements at all at the cost of significantly lower error-flagging rates. This provides an early indication that false endorsements are a new behaviour in frontier models, although only one frontier model was tested. 
- The SOURCE_EXCLUSIVE_FLAG_INVITING instruction also recorded 100% faithfulness rates similar to SOURCE_EXCLUSIVE and generally avoided endorsements, but all models recorded lower error-flagging rates under the SOURCE_EXCLUSIVE_FLAG_INVITING instruction than its FLAG_INVITING counterpart. 

Full tables, confidence intervals, raw examples and limitations: [results.md](results.md). Complete per-cell grids: `caveat_curve.csv` / `abstention_curve.csv`.

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
python3 harness.py tradeoff         # joint readout, no API calls
python3 harness.py vectors          # per-fact/per-item vectors + ICC per cell, no API calls
python3 judge.py caveat             # re-certify the caveat judge against its gold
python3 judge.py abstention         # re-certify the abstention judge
```

Both sweeps at the default N=4 (the published runs used N=8; reps within a fact turned out to be strongly correlated, so extra reps buy little -- see the independence limitation in results.md). It is advised to verify your judge first before running the harness. 

## Customisation

The benchmark can be customised in `config.py`, including models tested, samples per cell (n), the system instructions. Currently, swapping in your own document is deliberately **not** config-only: a new `document1_consent.txt` needs its own `PERTURBATION_LADDERS` (which facts to perturb, at which severities) and `UNANSWERABLE_ITEMS`. 

## Why trust the numbers

Every answer is scored by an LLM judge, and no judge scores anything before being certified against a human-labelled gold set with a zero-tolerance anchor check (obvious cases must all be judged correctly) plus a Cohen's kappa threshold >= 0.80 against the human labels. Current certifications for GPT-5.4-mini: stance/corroboration kappa 0.98/0.94 (0/30 anchors incorrect, 92-row gold), abstention judge kappa 1.00 (22-row gold). 


## Repo map

| File | Role |
|---|---|
| `config.py` | every customisable setting and shared functions |
| `harness.py` | the two experiments |
| `judge.py` | both judges and their certification pipeline |
| `test_logic.py` | 93 offline tests verifying the functions |
| `document1_consent.txt` | the source document (currently a NSW development consent) |
| `caveat_gold.json` / `abstention_gold.json` | human-labelled gold sets the judges are certified against |
| `results.md` + `*_curve.csv` | published findings and their full grids |
| `plot_tradeoff.py` + `tradeoff_scatter.png` | the trade-off scatter plot in results.md and its generator |

## Status / limitations

This is a project aimed at solving the problem described earlier and is currently a functional prototype that is able to successfully compute all the results and processes that have been described so far. However, the scope and complexity is limited to just two different model providers (Anthropic and OpenAI) and one grounded document. Additionally, the ratios for each severity are hard to standardise due to different units having to be perturbed by different values to achieve similar levels of implausibility, which is a subjective process. 