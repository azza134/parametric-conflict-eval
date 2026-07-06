# Faithfulness Evaluation Harness

## The dilemma

Documents used by small businesses, enterprises and governments are not immune from errors, large or small. And with rapid increases in AI adoption and integration across the workflows of these institutions, hallucinations as a business and institutional risk are surfacing quickly. 

What should a model do when encountered with an error in a document? Should it reach into pretraining data? Pretraining data is sometimes not specific enough to answer the question. So should the model not bring it up at all? Then the likelihood is that the error goes unnoticed. 

## What it measures

This harness tests the ability of a model to spot information in a document as likely to be an error. It also tests the willingness of a model to reach into its pretraining data to answer a question a user may ask that is not answered in the document. This is an important relationship because the aforementioned entities are deploying AI across their documents to provide specific responses that navigate gaps in pretrained data. However, the inevitability of errors in documents will result in a scenario where someone has to decide whether they would rather have the model spot errors or stick strictly to document-based retrieval. 

Both characteristics are tested using the three following system instructions (can be customised in config.py):

```text
SOURCE_EXCLUSIVE: "Answer using ONLY the passage. If the passage does not contain the answer, reply exactly: NOT IN DOCUMENT. Never use outside knowledge."

FLAG_INVITING: "Answer the question using the passage. Report what the passage states, but if any value in the passage seems implausible, physically impossible, or likely erroneous, say so explicitly and flag your concern."

WEAK_GROUNDING: "Base your answer on the passage provided."
```

## Findings so far

- Under SOURCE_EXCLUSIVE (the standard strict-grounding RAG system instruction), error-flagging is zero at every severity. Both models repeat physically impossible values (500-metre grass, one toilet per 1,000,000 workers) without comment.
- Under FLAG_INVITING, flagging only switches on at roughly 100x errors; perturbations up to 25x pass unflagged. False positives on correct values: zero. Under WEAK_GROUNDING (no invitation to flag), flagging is zero at every severity -- the skepticism is elicited by the invitation, not present by default.
- The weak models' failure mode is silence, not false endorsements: zero endorsements in 1,728 answers from GPT-4o-mini and GPT-5.4-nano across all three instructions. Fluent false verification has so far only been observed informally in frontier models.

Full tables, confidence intervals, raw examples and limitations: [results.md](results.md). Complete per-cell grids: `caveat_curve.csv` / `abstention_curve.csv`.

## Quickstart

```
pip install -r requirements.txt
cp .env.example .env        # add OPENAI_API_KEY (and ANTHROPIC_API_KEY for Anthropic candidates)
python3 harness.py          # dry-run: prints the full design + call-count estimate, costs nothing
```

Real runs are explicit and resumable (partial results persist to disk after every call):

```
python3 harness.py caveat [N]       # error-flagging sweep
python3 harness.py abstention [N]   # parametric-leakage sweep
python3 harness.py tradeoff         # joint readout, no API calls
python3 judge.py caveat             # re-certify the caveat judge against its gold
python3 judge.py abstention         # re-certify the abstention judge
```

Both sweeps at the default N=8. It is advised to verify your judge first before running the harness. 

## Customisation

The benchmark can be customised in `config.py`, including models tested, samples per cell (n), the system instructions. Currently, swapping in your own document is deliberately **not** config-only: a new `document.txt` needs its own `PERTURBATION_LADDERS` (which facts to perturb, at which severities) and `UNANSWERABLE_ITEMS`. 

## Why trust the numbers

Every answer is scored by an LLM judge, and no judge scores anything before being certified against a human-labelled gold set with a zero-tolerance anchor check (obvious cases must all be judged correctly) plus a Cohen's kappa threshold >= 0.80 against the human labels. Current certifications: caveat judge kappa 0.97 (3-class, 45-row gold), abstention judge kappa 1.00 (22-row gold). 

## Repo map

| File | Role |
|---|---|
| `config.py` | every customisable setting and shared functions |
| `harness.py` | the two experiments |
| `judge.py` | both judges and their certification pipeline |
| `test_logic.py` | 73 offline tests verifying the functions |
| `document.txt` | the source document (currently a NSW development consent) |
| `caveat_gold.json` / `abstention_gold.json` | human-labelled gold sets the judges are certified against |
| `results.md` + `*_curve.csv` | published findings and their full grids |

## Status / limitations

This is a project aimed at solving the problem described earlier and is currently a functional prototype that is able to successfully compute all the results and processes that have been described so far. However, the scope and complexity is limited to just two different model providers (Anthropic and OpenAI) and one grounded document. Additionally, the ratios for each severity are hard to standardise due to different units having to be perturbed by different values to achieve similar levels of implausibility, which is a subjective process. 