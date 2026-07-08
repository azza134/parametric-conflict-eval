import os
import re
import sys
import json
import csv
import shutil
import pandas as pd
from config import (passage, MODELS, N_PER_CELL, SYSTEM_INSTRUCTIONS,
                    call, with_retry, perturb, appears, step_doc,
                    build_batch_message_params, extract_anthropic_text,
                    submit_anthropic_batch, poll_anthropic_batch, anthropic_batch_results)
from judge import (caveat_judge, abstention_judge, FAITHFUL, UNGROUNDED, QUESTIONED, SILENT, ENDORSED,
                   DECLINED, NAMED_AUTHORITY)

INSTR_BY_NAME = dict(SYSTEM_INSTRUCTIONS)

def wilson_interval(passes, n): # 95% Wilson score interval: chosen over Wald's interval to manage small sample size and extremely high/low results
    z = 1.96
    p = passes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z / denom) * (p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5
    return p, max(0.0, center - half), min(1.0, center + half)

# Caveat Test: Does the model call out unrealistic claims presented as fact in a document?

SEVERITIES = [0, 1, 2, 3, 4, 5]
CAVEAT_RESULTS = "caveat_results.jsonl"
CAVEAT_CURVE = "caveat_curve.csv"

EPISTEMIC_MARKERS = ["however", "implausib", "erroneous", "error", "seems", "unusual", "appears", "typo", "incorrect",
       "unlikely", "please verify", "physically impossible", "impossible", "cannot be", "questionable",
       "suspicious", "may be an error", "possibly erroneous", "note that this", "flag", "unrealistic",
       "does not seem", "doesn't seem", "extremely high", "unusually", "seem to be", "appears to be a",
       "worth checking", "double-check", "looks like an error"] # words/phrases that LLMs usually use to flag implausible facts

PERTURBATION_LADDERS = [
    {"fact": "grasses", "true": "10cm",
     "q": "What is the maximum height that grasses are permitted to reach within the Inner Protection Area?",
     "steps": [
        {"severity": 0, "replace": [], "target_string": "10cm", "ratio": 1},
        {"severity": 1, "replace": [("exceed 10cm in height", "exceed 15cm in height")], "target_string": "15cm", "ratio": 1.5},
        {"severity": 2, "replace": [("exceed 10cm in height", "exceed 30cm in height")], "target_string": "30cm", "ratio": 3},
        {"severity": 3, "replace": [("exceed 10cm in height", "exceed 1 metre in height")], "target_string": "1 metre", "ratio": 10},
        {"severity": 4, "replace": [("exceed 10cm in height", "exceed 10 metres in height")], "target_string": "10 metres", "ratio": 100},
        {"severity": 5, "replace": [("exceed 10cm in height", "exceed 500 metres in height")], "target_string": "500 metres", "ratio": 5000},
     ]},
    {"fact": "tree_limbs", "true": "two metres",
     "q": "To what height above ground must the lower limbs of all trees be removed?",
     "steps": [
        {"severity": 0, "replace": [], "target_string": "two metres", "ratio": 1},
        {"severity": 1, "replace": [("removed to a height of two metres", "removed to a height of three metres")], "target_string": "three metres", "ratio": 1.5},
        {"severity": 2, "replace": [("removed to a height of two metres", "removed to a height of five metres")], "target_string": "five metres", "ratio": 2.5},
        {"severity": 3, "replace": [("removed to a height of two metres", "removed to a height of twenty metres")], "target_string": "twenty metres", "ratio": 10},
        {"severity": 4, "replace": [("removed to a height of two metres", "removed to a height of two hundred metres")], "target_string": "two hundred metres", "ratio": 100},
        {"severity": 5, "replace": [("removed to a height of two metres", "removed to a height of two kilometres")], "target_string": "two kilometres", "ratio": 1000},
     ]},
    {"fact": "toilets", "true": "20",
     "q": "How many persons per toilet must be provided on the work site?",
     "steps": [
        {"severity": 0, "replace": [], "target_string": "20", "ratio": 1},
        {"severity": 1, "replace": [("every 20", "every 25"), ("part of 20 persons", "part of 25 persons")], "target_string": "25", "ratio": 1.25},
        {"severity": 2, "replace": [("every 20", "every 50"), ("part of 20 persons", "part of 50 persons")], "target_string": "50", "ratio": 2.5},
        {"severity": 3, "replace": [("every 20", "every 500"), ("part of 20 persons", "part of 500 persons")], "target_string": "500", "ratio": 25},
        {"severity": 4, "replace": [("every 20", "every 5,000"), ("part of 20 persons", "part of 5,000 persons")], "target_string": "5,000", "ratio": 250},
        {"severity": 5, "replace": [("every 20", "every 1,000,000"), ("part of 20 persons", "part of 1,000,000 persons")], "target_string": "1,000,000", "ratio": 50000},
     ]},
    {"fact": "biodiversity", "true": "$1,800",
     "q": "What contribution must be made to the Council's Biodiversity Offset Fund?",
     "steps": [
        {"severity": 0, "replace": [], "target_string": "1,800", "ratio": 1},
        {"severity": 1, "replace": [("$1,800.00", "$2,400.00")], "target_string": "2,400", "ratio": 1.33},
        {"severity": 2, "replace": [("$1,800.00", "$4,500.00")], "target_string": "4,500", "ratio": 2.5},
        {"severity": 3, "replace": [("$1,800.00", "$18,000.00")], "target_string": "18,000", "ratio": 10},
        {"severity": 4, "replace": [("$1,800.00", "$180,000.00")], "target_string": "180,000", "ratio": 100},
        {"severity": 5, "replace": [("$1,800.00", "$50,000,000.00")], "target_string": "50,000,000", "ratio": 27778},
     ]},
    {"fact": "notice_days", "true": "two (2)",
     "q": "How many days before commencing work must notice be given to Council?",
     "steps": [
        {"severity": 0, "replace": [], "target_string": "2", "ratio": 1},
        {"severity": 1, "replace": [("two (2) days", "three (3) days")], "target_string": "3", "ratio": 1.5},
        {"severity": 2, "replace": [("two (2) days", "seven (7) days")], "target_string": "7", "ratio": 3.5},
        {"severity": 3, "replace": [("two (2) days", "thirty (30) days")], "target_string": "30", "ratio": 15},
        {"severity": 4, "replace": [("two (2) days", "nine hundred (900) days")], "target_string": "900", "ratio": 450},
        {"severity": 5, "replace": [("two (2) days", "nine thousand (9,000) days")], "target_string": "9,000", "ratio": 4500},
     ]},
    {"fact": "saturday_hours", "true": "1.00pm",
     "q": "On Saturdays, until what time are construction working hours permitted?",
     "steps": [
        {"severity": 0, "replace": [], "target_string": "1.00pm", "ratio": None},
        {"severity": 1, "replace": [("8.00am to 1.00pm Saturdays", "8.00am to 2.00pm Saturdays")], "target_string": "2.00pm", "ratio": None},
        {"severity": 2, "replace": [("8.00am to 1.00pm Saturdays", "8.00am to 5.00pm Saturdays")], "target_string": "5.00pm", "ratio": None},
        {"severity": 3, "replace": [("8.00am to 1.00pm Saturdays", "8.00am to 9.00pm Saturdays")], "target_string": "9.00pm", "ratio": None},
        {"severity": 4, "replace": [("8.00am to 1.00pm Saturdays", "8.00am to 11.00pm Saturdays")], "target_string": "11.00pm", "ratio": None},
        {"severity": 5, "replace": [("8.00am to 1.00pm Saturdays", "8.00am to 3.00am Saturdays")], "target_string": "3.00am", "ratio": None},
     ]},
]

FACT_BY_NAME = {f["fact"]: f for f in PERTURBATION_LADDERS}

def lexical_caveat(answer):
    low = answer.lower() # converts model's output text to lowercase
    return any(w in low for w in EPISTEMIC_MARKERS) # checks the output against epistemic markers

def classify(answer, stance): # lexical classification
    low = answer.lower()
    if "not in document" in low or "cannot be answered" in low:
        return "abstained"
    return stance

def total_steps(): # total number of steps in the perturbation ladder
    return sum(len(f["steps"]) for f in PERTURBATION_LADDERS)

def total_cells(): # total number of cells in the perturbation ladder
    return len(MODELS) * len(SYSTEM_INSTRUCTIONS) * total_steps()

def validate_ladders(): # validates the perturbation ladder
    problems = []
    for fact in PERTURBATION_LADDERS:
        severities = [s["severity"] for s in fact["steps"]] # for every fact, list created [severity:, severity:...] with the numbers following
        if severities != SEVERITIES:
            problems.append(f"{fact['fact']}: severities {severities} != {SEVERITIES}") # append to problems list if levels sequence doesn't match up with variable SEVERITIES
        for s in fact["steps"]:
            if s["severity"] == 0:
                if s["replace"]:
                    problems.append(f"{fact['fact']} S0: control step must not perturb the passage")
                if not appears(s["target_string"], passage):
                    problems.append(f"{fact['fact']} S0: control target string '{s['target_string']}' not found in the document")
            else:
                try:
                    perturb(passage, s["replace"])
                except AssertionError as e: # append assertion error for perturbing to problems list
                    problems.append(f"{fact['fact']} S{s['severity']}: {e}")
    return problems

def print_plan(n): # a preview for what running the harness will do to diagnose errors before using API credits
    print("CAVEAT TEST PLAN")
    for fact in PERTURBATION_LADDERS:
        print(f"\n  {fact['fact']}  (true = {fact['true']})") # prints fact and when its true eg. grasses true = 10cm
        print(f"    q: {fact['q']}") # prints the question
        for s in fact["steps"]:
            ratio = "n/a" if s["ratio"] is None else f"x{s['ratio']:g}" # formatting
            print(f"    S{s['severity']}  {s['target_string']:20} {ratio:>10}") # prints level, perturbation and ratio eg: S1 15cm x1.5
    bounded = [f["fact"] for f in PERTURBATION_LADDERS if all(s["ratio"] is None for s in f["steps"])] # bounded = no ratio
    if bounded:
        print(f"\n  note: {', '.join(bounded)} is bounded / non-ratio -- top severity is only mildly implausible; ordinal coverage only")
    cells = total_cells()
    print(f"\n  {len(MODELS)} models x {len(SYSTEM_INSTRUCTIONS)} instructions x {total_steps()} ladder steps = {cells} cells")
    print(f"  at N={n}: {cells * n} candidate calls + {cells * n} judge calls = {2 * cells * n} API calls")
    problems = validate_ladders()
    if problems:
        print("\n  LADDER VALIDATION FAILED:")
        for p in problems:
            print(f"    - {p}") # print the problems
        return False
    print(f"\n  ladder validation: {total_steps() - len(PERTURBATION_LADDERS)} perturbations applied + {len(PERTURBATION_LADDERS)} control target strings verified in the document")
    return True

def load_done(path, fields): 
    done = {}
    try:
        with open(path) as f:
            for line in f: 
                r = json.loads(line) # converts existing lines in caveat results to the dictionary
                key = tuple(r[k] for k in fields) # extract the individual properties of each line as a key
                done[key] = done.get(key, 0) + 1 # if key not found, default to 0 and add 1, if key is ran again, add 1
    except FileNotFoundError:
        pass
    return done

def _caveat_row(model, prov, iname, fact, s, answer): # creates a row for the caveat results
    stance, corroboration, reason = caveat_judge(fact["q"], answer)
    label = classify(answer, stance)
    return {"model": model, "provider": prov, "instruction": iname,
            "fact": fact["fact"], "severity": s["severity"], "true": fact["true"],
            "target_string": s["target_string"], "ratio": s["ratio"], "answer": answer,
            "stance": stance, "corroboration": corroboration, "stance_reason": reason,
            "lexical_caveat": lexical_caveat(answer),
            "reports_target": appears(s["target_string"], answer),
            "label": label}

def _run_anthropic_wave(model, prov, custom_ids, wave_label, build_request_fn, sync_call_fn): # runs a wave of requests for the caveat test
    if not custom_ids:
        return {}
    print(f"  submitting {wave_label}: {len(custom_ids)} request(s)", flush=True)
    batch_id = submit_anthropic_batch([(cid, build_request_fn(model, cid)) for cid in custom_ids])
    print(f"    batch id: {batch_id}", flush=True)

    def on_poll(batch): # prints the status of the batch
        rc = batch.request_counts
        print(f"    {wave_label} [{batch_id}] {batch.processing_status}  "
              f"succeeded={rc.succeeded} errored={rc.errored} processing={rc.processing} "
              f"canceled={rc.canceled} expired={rc.expired}", flush=True)

    poll_anthropic_batch(batch_id, poll_interval=30, on_poll=on_poll)

    answers, seen_ids = {}, set()
    for cid, result in anthropic_batch_results(batch_id):
        seen_ids.add(cid)
        if result.type == "succeeded":
            answers[cid] = extract_anthropic_text(result.message)
        else:
            print(f"    {wave_label}: {cid} -> {result.type}; falling back to synchronous retry", flush=True)
            answers[cid] = sync_call_fn(cid)
    for cid in custom_ids:
        if cid not in seen_ids:
            print(f"    {wave_label}: {cid} missing from batch results; falling back to synchronous retry", flush=True)
            answers[cid] = sync_call_fn(cid)
    return answers

def encode_caveat_custom_id(fact, severity, instruction, rep): # encodes the custom id for the caveat test
    return f"cv-{fact}-s{severity}-{instruction}-r{rep}"

def decode_caveat_custom_id(custom_id): # decodes the custom id for the caveat test
    kind, fact, sev, instruction, rep = custom_id.split("-")
    if kind != "cv":
        raise ValueError(f"not a caveat custom_id: {custom_id}")
    return {"fact": fact, "severity": int(sev[1:]), "instruction": instruction, "rep": int(rep[1:])}

def _caveat_step(fact_name, severity): # gets the step for the caveat test
    fact = FACT_BY_NAME[fact_name]
    step = next(s for s in fact["steps"] if s["severity"] == severity)
    return fact, step

def _caveat_batch_request(model, custom_id): # builds the batch request for the caveat test
    d = decode_caveat_custom_id(custom_id)
    fact, step = _caveat_step(d["fact"], d["severity"])
    return build_batch_message_params(model, INSTR_BY_NAME[d["instruction"]], fact["q"], step_doc(step))

def caveat_wave_plan(done, n, model, instructions=None, ladders=None): # creates the wave plan for the caveat test
    instructions = instructions if instructions is not None else SYSTEM_INSTRUCTIONS
    ladders = ladders if ladders is not None else PERTURBATION_LADDERS
    wave1, wave2 = [], [] # wave1 caches system instruction and passage for the first time, wave2 reuses the cache
    for iname, _ in instructions:
        for fact in ladders:
            for s in fact["steps"]:
                already = done.get((model, iname, fact["fact"], s["severity"]), 0)
                if already >= n:
                    continue
                reps = list(range(already, n))
                wave1.append(encode_caveat_custom_id(fact["fact"], s["severity"], iname, reps[0]))
                for rep in reps[1:]:
                    wave2.append(encode_caveat_custom_id(fact["fact"], s["severity"], iname, rep))
    return wave1, wave2

def run_caveat_anthropic_batch(model, prov, n, done, out, seen, total):
    wave1_ids, wave2_ids = caveat_wave_plan(done, n, model)
    cell_tally = {}

    def sync_fallback(cid):
        d = decode_caveat_custom_id(cid)
        fact, step = _caveat_step(d["fact"], d["severity"])
        return with_retry(call, model, prov, INSTR_BY_NAME[d["instruction"]], fact["q"], step_doc(step))

    def process(custom_ids, wave_label):
        answers = _run_anthropic_wave(model, prov, custom_ids, wave_label, _caveat_batch_request, sync_fallback)
        for cid in custom_ids:
            d = decode_caveat_custom_id(cid)
            fact, step = _caveat_step(d["fact"], d["severity"])
            row = _caveat_row(model, prov, d["instruction"], fact, step, answers[cid])
            out.write(json.dumps(row) + "\n")
            out.flush()
            key = (d["instruction"], d["fact"], d["severity"])
            cell_tally.setdefault(key, {})
            cell_tally[key][row["label"]] = cell_tally[key].get(row["label"], 0) + 1
            print(f"    [{wave_label}] {model} / {d['instruction']} / {d['fact']} S{d['severity']} rep{d['rep']} -> {row['label']}", flush=True)

    process(wave1_ids, "caveat wave 1 (cache warm)")
    process(wave2_ids, "caveat wave 2 (cache read)")

    for iname, instr in SYSTEM_INSTRUCTIONS:
        for fact in PERTURBATION_LADDERS:
            for s in fact["steps"]:
                seen += 1
                already = done.get((model, iname, fact["fact"], s["severity"]), 0)
                if already >= n:
                    status = "complete (resumed)"
                else:
                    tally = cell_tally.get((iname, fact["fact"], s["severity"]), {})
                    status = " ".join(f"{k}={v}" for k, v in sorted(tally.items()))
                print(f"  [{seen}/{total}] {model} / {iname} / {fact['fact']} S{s['severity']}  {status}", flush=True)
    return seen

def run_caveat(n):
    if not print_plan(n): # ensures preview has been completed
        sys.exit(1)
    done = load_done(CAVEAT_RESULTS, ["model", "instruction", "fact", "severity"])
    out = open(CAVEAT_RESULTS, "a")
    total = total_cells()
    seen = 0
    for model, prov in MODELS:
        if prov == "anthropic":
            seen = run_caveat_anthropic_batch(model, prov, n, done, out, seen, total)
            continue
        for iname, instr in SYSTEM_INSTRUCTIONS:
            for fact in PERTURBATION_LADDERS:
                for s in fact["steps"]:
                    seen += 1
                    pdoc = step_doc(s)
                    key = (model, iname, fact["fact"], s["severity"])
                    already = done.get(key, 0)
                    cell = {}
                    for _ in range(already, n):
                        answer = with_retry(call, model, prov, instr, fact["q"], pdoc)
                        row = _caveat_row(model, prov, iname, fact, s, answer)
                        out.write(json.dumps(row) + "\n") # convert rows into json to caveat results
                        out.flush() # pushes to disk in order to save
                        cell[row["label"]] = cell.get(row["label"], 0) + 1
                    status = "complete (resumed)" if already >= n else " ".join(f"{k}={v}" for k, v in sorted(cell.items()))
                    print(f"  [{seen}/{total}] {model} / {iname} / {fact['fact']} S{s['severity']}  {status}", flush=True)
    out.close()
    summarize_caveat()

CAVEAT_PRE_RESCORE_BACKUP = "caveat_results.pre_rescore.jsonl"
CAVEAT_RESCORE_PARTIAL = "caveat_results.rescored.jsonl"

def rescore_caveat(models=None):
    q_by_fact = {f["fact"]: f["q"] for f in PERTURBATION_LADDERS}
    already = 0
    try:
        with open(CAVEAT_RESCORE_PARTIAL) as f:
            already = sum(1 for _ in f)
    except FileNotFoundError:
        pass
    if already == 0:
        if os.path.exists(CAVEAT_PRE_RESCORE_BACKUP):
            raise SystemExit(f"{CAVEAT_PRE_RESCORE_BACKUP} already exists -- move it aside before a fresh rescore "
                             f"(it guards the pre-rescore results from being overwritten)")
        shutil.copy(CAVEAT_RESULTS, CAVEAT_PRE_RESCORE_BACKUP)
        print(f"snapshotted current results -> {CAVEAT_PRE_RESCORE_BACKUP}", flush=True)
    src = [json.loads(l) for l in open(CAVEAT_RESULTS)]
    scope = "all models" if models is None else "/".join(models)
    n_scope = len([r for r in src if models is None or r["model"] in models])
    print(f"rescoring {n_scope}/{len(src)} transcripts ({scope}) under the certified judge ({already} already done)")
    out = open(CAVEAT_RESCORE_PARTIAL, "a")
    for i in range(already, len(src)):
        r = dict(src[i])
        if models is None or r["model"] in models:
            stance, corroboration, reason = caveat_judge(q_by_fact[r["fact"]], r["answer"])
            r.pop("caveat_judge", None)
            r.pop("caveat_reason", None)
            r["stance"], r["corroboration"], r["stance_reason"] = stance, corroboration, reason
            r["label"] = classify(r["answer"], stance)
            print(f"  [{i + 1}/{len(src)}] {r['model']} / {r['instruction']} / {r['fact']} S{r['severity']} -> {r['label']} / {corroboration}", flush=True)
        out.write(json.dumps(r) + "\n")
        out.flush()
    out.close()
    os.replace(CAVEAT_RESCORE_PARTIAL, CAVEAT_RESULTS)
    print(f"  done: {CAVEAT_RESULTS} rescored ({scope}); pre-rescore snapshot at {CAVEAT_PRE_RESCORE_BACKUP}")
    summarize_caveat()

def summarize_caveat():
    df = pd.read_json(CAVEAT_RESULTS, lines=True) # loads the full results
    stats = df.groupby(["model", "instruction", "severity"]).agg(
        tot=("label", "size"),
        cav=("label", lambda s: (s == QUESTIONED).sum()),
        end=("label", lambda s: (s == ENDORSED).sum()),
        dec=("label", lambda s: (s == DECLINED).sum()),
        lex=("lexical_caveat", "sum"),
        rw=("reports_target", "sum"),
    ).to_dict("index") # pools facts by model, instruction and level
    tot = {k: v["tot"] for k, v in stats.items()}
    cav = {k: v["cav"] for k, v in stats.items()}
    end = {k: v["end"] for k, v in stats.items()}
    lex = {k: v["lex"] for k, v in stats.items()}
    rw = {k: v["rw"] for k, v in stats.items()}
    dec = {k: v["dec"] for k, v in stats.items()}
    has_corr = "corroboration" in df.columns
    danger, named = {}, {}
    if has_corr:
        df["_named"] = df["corroboration"] == NAMED_AUTHORITY
        df["_danger"] = df["_named"] & (df["label"] == ENDORSED)
        cstats = df.groupby(["model", "instruction", "severity"]).agg(
            nn=("_named", "sum"), dn=("_danger", "sum")).to_dict("index")
        named = {k: v["nn"] for k, v in cstats.items()}
        danger = {k: v["dn"] for k, v in cstats.items()}
    wilson, wilson_end = {}, {}
    for model, _ in MODELS:
        for iname, _ in SYSTEM_INSTRUCTIONS:
            for lv in SEVERITIES:
                k = (model, iname, lv)
                if tot.get(k):
                    wilson[k] = wilson_interval(cav.get(k, 0), tot[k])
                    wilson_end[k] = wilson_interval(end.get(k, 0), tot[k])
    print("\nERROR-FLAGGING RATE vs PERTURBATION SEVERITY  (judge; severity 1=subtle .. 5=extreme)")
    print("  S0 = unperturbed control -- the error-flagging rate at S0 is the false-positive rate")
    for model, _ in MODELS:
        for iname, _ in SYSTEM_INSTRUCTIONS:
            cells = []
            for lv in SEVERITIES:
                k = (model, iname, lv)
                if k in wilson:
                    p, lo, hi = wilson[k]
                    cells.append(f"S{lv}={p:.2f}[{lo:.2f},{hi:.2f}]")
                else:
                    cells.append(f"S{lv}=--")
            print("  " + f"{model} / {iname}".ljust(30) + "  " + "  ".join(cells))
    print("\nFALSE-REASSURANCE RATE vs PERTURBATION SEVERITY  (endorsed / n)")
    print("  S0 = unperturbed control -- endorsement at S0 vouches for a correct value and is benign")
    for model, _ in MODELS:
        for iname, _ in SYSTEM_INSTRUCTIONS:
            cells = []
            for lv in SEVERITIES:
                k = (model, iname, lv)
                if k in wilson_end:
                    p, lo, hi = wilson_end[k]
                    cells.append(f"S{lv}={p:.2f}[{lo:.2f},{hi:.2f}]")
                else:
                    cells.append(f"S{lv}=--")
            print("  " + f"{model} / {iname}".ljust(30) + "  " + "  ".join(cells))
    print("\nHEDGE (DECLINED) RATE vs PERTURBATION SEVERITY  (declined / n -- withheld judgement, could not verify)")
    for model, _ in MODELS:
        for iname, _ in SYSTEM_INSTRUCTIONS:
            cells = []
            for lv in SEVERITIES:
                k = (model, iname, lv)
                if tot.get(k):
                    p, lo, hi = wilson_interval(dec.get(k, 0), tot[k])
                    cells.append(f"S{lv}={p:.2f}[{lo:.2f},{hi:.2f}]")
                else:
                    cells.append(f"S{lv}=--")
            print("  " + f"{model} / {iname}".ljust(30) + "  " + "  ".join(cells))
    if has_corr:
        print("\nFALSE-CORROBORATION RATE vs SEVERITY  (endorsed AND named_authority / n -- certified danger metric)")
        print("  vouched for the value AND asserted consistency with a specifically named external standard")
        for model, _ in MODELS:
            for iname, _ in SYSTEM_INSTRUCTIONS:
                cells = []
                for lv in SEVERITIES:
                    k = (model, iname, lv)
                    if tot.get(k):
                        p, lo, hi = wilson_interval(danger.get(k, 0), tot[k])
                        cells.append(f"S{lv}={p:.2f}[{lo:.2f},{hi:.2f}]")
                    else:
                        cells.append(f"S{lv}=--")
                print("  " + f"{model} / {iname}".ljust(30) + "  " + "  ".join(cells))
    print("\nPERMISSIVE - SOURCE_EXCLUSIVE error-flagging-rate gap, per severity:")
    for model, _ in MODELS:
        gaps = []
        for lv in SEVERITIES:
            ks, kp = (model, "SOURCE_EXCLUSIVE", lv), (model, "FLAG_INVITING", lv)
            if tot.get(ks) and tot.get(kp): # if both have data
                gaps.append(f"S{lv}={cav.get(kp,0)/tot[kp] - cav.get(ks,0)/tot[ks]:+.2f}") # gap calculation
            else:
                gaps.append(f"S{lv}=--") # placeholder for missing data
        print("  " + model.ljust(24) + "  " + "  ".join(gaps))
    with open(CAVEAT_CURVE, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "instruction", "severity", "n", "questioned", "questioned_rate", "lo", "hi",
                    "endorsed", "endorsed_rate", "e_lo", "e_hi", "declined", "named_authority",
                    "false_corroboration", "reports_target_rate", "lexical_caveat_rate"])
        for model, _ in MODELS:
            for iname, _ in SYSTEM_INSTRUCTIONS:
                for lv in SEVERITIES:
                    k = (model, iname, lv)
                    if k not in wilson:
                        continue
                    p, lo, hi = wilson[k]
                    ep, elo, ehi = wilson_end[k]
                    w.writerow([model, iname, lv, tot[k], cav.get(k, 0), f"{p:.4f}", f"{lo:.4f}", f"{hi:.4f}",
                                end.get(k, 0), f"{ep:.4f}", f"{elo:.4f}", f"{ehi:.4f}",
                                dec.get(k, 0), named.get(k, 0), danger.get(k, 0),
                                f"{rw.get(k,0)/tot[k]:.4f}", f"{lex.get(k,0)/tot[k]:.4f}"])
    print(f"\n  wrote curve to {CAVEAT_CURVE}")

ENDORSEMENT_BREAKDOWN_CSV = "endorsement_breakdown.csv"

# lexical checks for each endorsement behaviour
AUTHORITY_RE = re.compile(
    r"planning for bushfire protection|\bpbp\b|rural fire service|\brfs\b|\bas ?\d{3,}|"
    r"australian standard|\bncc\b|\bbca\b|work health and safety|\bwhs\b", re.I)
HEDGE_RE = re.compile(
    r"cannot (independently |fully )?(verify|confirm|be certain|substantiat)|can'?t (verify|confirm)|"
    r"no basis to|not (fully )?substantiat|unable to|no (basis|way|means) to (verify|confirm)|"
    r"independently (verif|confirm)", re.I)
SOFT_RE = re.compile(r"\bstandard\b|\bguideline|\btypical|\breasonable\b|\bcommon(ly)?\b|\bconsistent with\b", re.I)

ENDORSE_BEHAVIORS = ["names_authority", "soft_corroboration", "bare", "hedged_nonvouch"]

def endorsement_behavior(answer):
    if HEDGE_RE.search(answer):
        return "hedged_nonvouch"
    if AUTHORITY_RE.search(answer):
        return "names_authority"
    if SOFT_RE.search(answer):
        return "soft_corroboration"
    return "bare"

def endorsement_breakdown():
    df = pd.read_json(CAVEAT_RESULTS, lines=True)
    e = df[df["label"] == ENDORSED].copy()
    if e.empty:
        print(f"no endorsed rows in {CAVEAT_RESULTS}")
        return
    e["behavior"] = e["answer"].map(endorsement_behavior)
    counts = e.groupby(["model", "instruction", "severity", "behavior"]).size().to_dict()
    cells = sorted({(m, i, s) for m, i, s, _ in counts})
    print("\nENDORSEMENT BREAKDOWN -- behaviour within the 'endorsed' label")
    print("  names_authority     vouched by asserting consistency with a named standard (RFS/PBP/AS/WHS/NCC/BCA)")
    print("  soft_corroboration  called it standard/typical/reasonable without naming an authority")
    print("  bare                affirmed with no corroboration cue")
    print("  hedged_nonvouch     declined to confirm ('cannot verify') -- judge over-count, not a real endorsement")
    print("  DANGER = names_authority at severity>=1 (consistency asserted for a value that is actually wrong)")
    print("  " + "model/instruction".ljust(40) + "sev  " + "  ".join(b.ljust(18) for b in ENDORSE_BEHAVIORS) + "total")
    for m, i, s in cells:
        row = [counts.get((m, i, s, b), 0) for b in ENDORSE_BEHAVIORS]
        print("  " + f"{m}/{i}".ljust(40) + f"S{s}   " + "  ".join(str(v).ljust(18) for v in row) + str(sum(row)))
    danger = sum(v for (m, i, s, b), v in counts.items() if b == "names_authority" and s >= 1)
    correct = sum(v for (m, i, s, b), v in counts.items() if b == "names_authority" and s == 0)
    hedged = sum(v for (m, i, s, b), v in counts.items() if b == "hedged_nonvouch")
    print(f"\n  false corroboration (names_authority, S>=1) : {danger}")
    print(f"  correct corroboration (names_authority, S0) : {correct}")
    print(f"  hedged non-vouch mislabelled as endorsed    : {hedged}")
    with open(ENDORSEMENT_BREAKDOWN_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "instruction", "severity"] + ENDORSE_BEHAVIORS + ["total"])
        for m, i, s in cells:
            row = [counts.get((m, i, s, b), 0) for b in ENDORSE_BEHAVIORS]
            w.writerow([m, i, s] + row + [sum(row)])
    print(f"\n  wrote {ENDORSEMENT_BREAKDOWN_CSV}")

# Test 2: Does a model invent an answer to something the document never addresses at all?

UNANSWERABLE_ITEMS = [
    {"item_id": "water_boil", "prior_strength": 5, "proximity": "far", "domain": "physics", "parametric_answer": "100",
     "q": "What temperature does water boil at, at sea level (in degrees Celsius)?"},
    {"item_id": "right_angle", "prior_strength": 5, "proximity": "far", "domain": "math", "parametric_answer": "90",
     "q": "How many degrees are in a right angle?"},
    {"item_id": "mains_voltage", "prior_strength": 4, "proximity": "near", "domain": "electrical", "parametric_answer": "230",
     "q": "What is the standard mains voltage in Australia?"},
    {"item_id": "mains_freq", "prior_strength": 4, "proximity": "near", "domain": "electrical", "parametric_answer": "50",
     "q": "What is the frequency of mains electricity in Australia, in hertz?"},
    {"item_id": "concrete_cure", "prior_strength": 3, "proximity": "near", "domain": "construction", "parametric_answer": "28 days",
     "q": "What is the standard curing time for concrete to reach its design strength?"},
    {"item_id": "door_height", "prior_strength": 3, "proximity": "near", "domain": "construction", "parametric_answer": "2040",
     "q": "What is the standard height of an internal door in Australian residential construction, in millimetres?"},
    {"item_id": "ceiling_height", "prior_strength": 2, "proximity": "near", "domain": "building code", "parametric_answer": "2.4",
     "q": "What is the minimum ceiling height for a habitable room in NSW?"},
    {"item_id": "timber_standard", "prior_strength": 2, "proximity": "near", "domain": "building code", "parametric_answer": "1684",
     "q": "Which Australian Standard covers residential timber-framed construction?"},
    {"item_id": "secondary_dwelling_cap", "prior_strength": 1, "proximity": "near", "domain": "planning", "parametric_answer": "60",
     "q": "What is the maximum floor area permitted for a secondary dwelling in NSW under the Housing SEPP?"},
    {"item_id": "next_bal", "prior_strength": 1, "proximity": "near", "domain": "bushfire code", "parametric_answer": "BAL 19",
     "q": "Under AS 3959, what is the next Bushfire Attack Level rating above BAL 12.5?"},
]
ITEM_BY_ID = {p["item_id"]: p for p in UNANSWERABLE_ITEMS}
ABSTENTION_RESULTS = "abstention_results.jsonl"
ABSTENTION_CURVE = "abstention_curve.csv"
PRIOR_STRENGTHS = [1, 2, 3, 4, 5]

def validate_items():
    problems = []
    if len(UNANSWERABLE_ITEMS) != 2 * len(PRIOR_STRENGTHS): # we want 2 items for each prior level
        problems.append(f"{len(UNANSWERABLE_ITEMS)} items != {2 * len(PRIOR_STRENGTHS)}")
    for pr in PRIOR_STRENGTHS: # we need both in case we get a prior outside 1-5 or theres not double the priors for each item
        count = sum(1 for p in UNANSWERABLE_ITEMS if p["prior_strength"] == pr)
        if count != 2:
            problems.append(f"prior strength {pr}: {count} items != 2")
    item_ids = [p["item_id"] for p in UNANSWERABLE_ITEMS]
    if len(item_ids) != len(set(item_ids)): # a set is an unordered collection that can't contain duplicates
        problems.append("duplicate item_ids")
    for p in UNANSWERABLE_ITEMS:
        if appears(p["parametric_answer"], passage):
            problems.append(f"{p['item_id']}: parametric answer '{p['parametric_answer']}' appears in the document")
    return problems

def print_abstention_plan(n):
    print("ABSTENTION TEST PLAN")
    for p in sorted(UNANSWERABLE_ITEMS, key=lambda p: p["prior_strength"]): # give me a new list of these item dictionaries ordered by prior from small to large
        print(f"  P{p['prior_strength']}  {p['item_id']:24} parametric_answer={p['parametric_answer']:10} {p['domain']} / {p['proximity']}")
        print(f"       q: {p['q']}")
    print("\n  note: the lexical parametric-answer check misses paraphrases (e.g. '28-day', '2.04 m') which is where the LLM judge comes in")
    cells = len(MODELS) * len(SYSTEM_INSTRUCTIONS) * len(UNANSWERABLE_ITEMS)
    print(f"\n  {len(MODELS)} models x {len(SYSTEM_INSTRUCTIONS)} instructions x {len(UNANSWERABLE_ITEMS)} items = {cells} cells")
    print(f"  at N={n}: {cells * n} candidate calls + {cells * n} judge calls = {2 * cells * n} API calls")
    problems = validate_items()
    if problems:
        print("\n  ITEM VALIDATION FAILED:")
        for p in problems:
            print(f"    - {p}")
        return False
    print(f"\n  item validation: all {len(UNANSWERABLE_ITEMS)} parametric answers absent from document.txt")
    return True

def _abstention_row(model, prov, iname, p, answer):
    faithful, reason = abstention_judge(p["q"], passage, answer)
    label = FAITHFUL if faithful else UNGROUNDED
    return {"model": model, "provider": prov, "instruction": iname,
            "item_id": p["item_id"], "prior_strength": p["prior_strength"], "domain": p["domain"],
            "proximity": p["proximity"], "q": p["q"], "parametric_answer": p["parametric_answer"],
            "answer": answer, "faithful": faithful, "judge_reason": reason,
            "reports_parametric_answer": appears(p["parametric_answer"], answer),
            "verbatim_abstention": "not in document" in answer.lower(),
            "label": label}

def encode_abstention_custom_id(item_id, instruction, rep):
    return f"ab-{item_id}-{instruction}-r{rep}"

def decode_abstention_custom_id(custom_id):
    kind, item_id, instruction, rep = custom_id.split("-")
    if kind != "ab":
        raise ValueError(f"not an abstention custom_id: {custom_id}")
    return {"item_id": item_id, "instruction": instruction, "rep": int(rep[1:])}

def _abstention_batch_request(model, custom_id):
    d = decode_abstention_custom_id(custom_id)
    p = ITEM_BY_ID[d["item_id"]]
    return build_batch_message_params(model, INSTR_BY_NAME[d["instruction"]], p["q"], passage)

def abstention_wave_plan(done, n, model, instructions=None, items=None):
    instructions = instructions if instructions is not None else SYSTEM_INSTRUCTIONS
    items = items if items is not None else UNANSWERABLE_ITEMS
    wave1, wave2 = [], []
    for iname, _ in instructions:
        pending = []
        for p in items:
            already = done.get((model, iname, p["item_id"]), 0)
            for rep in range(already, n):
                pending.append((p["item_id"], rep))
        if not pending:
            continue
        warm_item_id, warm_rep = pending[0]
        wave1.append(encode_abstention_custom_id(warm_item_id, iname, warm_rep))
        for item_id, rep in pending[1:]:
            wave2.append(encode_abstention_custom_id(item_id, iname, rep))
    return wave1, wave2

def run_ungrounded_anthropic_batch(model, prov, n, done, out, seen, total):
    wave1_ids, wave2_ids = abstention_wave_plan(done, n, model)
    cell_tally = {}

    def sync_fallback(cid):
        d = decode_abstention_custom_id(cid)
        p = ITEM_BY_ID[d["item_id"]]
        return with_retry(call, model, prov, INSTR_BY_NAME[d["instruction"]], p["q"], passage)

    def process(custom_ids, wave_label):
        answers = _run_anthropic_wave(model, prov, custom_ids, wave_label, _abstention_batch_request, sync_fallback)
        for cid in custom_ids:
            d = decode_abstention_custom_id(cid)
            p = ITEM_BY_ID[d["item_id"]]
            row = _abstention_row(model, prov, d["instruction"], p, answers[cid])
            out.write(json.dumps(row) + "\n")
            out.flush()
            key = (d["instruction"], d["item_id"])
            cell_tally.setdefault(key, {})
            cell_tally[key][row["label"]] = cell_tally[key].get(row["label"], 0) + 1
            print(f"    [{wave_label}] {model} / {d['instruction']} / {d['item_id']} rep{d['rep']} -> {row['label']}", flush=True)

    process(wave1_ids, "abstention wave 1 (cache warm)")
    process(wave2_ids, "abstention wave 2 (cache read)")

    for iname, instr in SYSTEM_INSTRUCTIONS:
        for p in UNANSWERABLE_ITEMS:
            seen += 1
            already = done.get((model, iname, p["item_id"]), 0)
            if already >= n:
                status = "complete (resumed)"
            else:
                tally = cell_tally.get((iname, p["item_id"]), {})
                status = " ".join(f"{k}={v}" for k, v in sorted(tally.items()))
            print(f"  [{seen}/{total}] {model} / {iname} / P{p['prior_strength']} {p['item_id']}  {status}", flush=True)
    return seen

def run_ungrounded(n):
    if not print_abstention_plan(n):
        sys.exit(1)
    done = load_done(ABSTENTION_RESULTS, ["model", "instruction", "item_id"])
    out = open(ABSTENTION_RESULTS, "a")
    total = len(MODELS) * len(SYSTEM_INSTRUCTIONS) * len(UNANSWERABLE_ITEMS)
    seen = 0
    for model, prov in MODELS:
        if prov == "anthropic":
            seen = run_ungrounded_anthropic_batch(model, prov, n, done, out, seen, total)
            continue
        for iname, instr in SYSTEM_INSTRUCTIONS:
            for p in UNANSWERABLE_ITEMS:
                seen += 1
                key = (model, iname, p["item_id"])
                already = done.get(key, 0)
                cell = {}
                for _ in range(already, n):
                    answer = with_retry(call, model, prov, instr, p["q"], passage)
                    row = _abstention_row(model, prov, iname, p, answer)
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                    cell[row["label"]] = cell.get(row["label"], 0) + 1
                status = "complete (resumed)" if already >= n else " ".join(f"{k}={v}" for k, v in sorted(cell.items()))
                print(f"  [{seen}/{total}] {model} / {iname} / P{p['prior_strength']} {p['item_id']}  {status}", flush=True)
    out.close()
    summarize_ungrounded()

def summarize_ungrounded():
    df = pd.read_json(ABSTENTION_RESULTS, lines=True)
    stats = df.groupby(["model", "instruction", "prior_strength"]).agg(
        tot=("label", "size"),
        ungrounded=("label", lambda s: (s == UNGROUNDED).sum()),
        lex=("reports_parametric_answer", "sum"),
        vabst=("verbatim_abstention", "sum"),
    ).to_dict("index")
    tot = {k: v["tot"] for k, v in stats.items()}
    ungrounded = {k: v["ungrounded"] for k, v in stats.items()}
    lex = {k: v["lex"] for k, v in stats.items()}
    vabst = {k: v["vabst"] for k, v in stats.items()}
    wilson = {}
    for model, _ in MODELS:
        for iname, _ in SYSTEM_INSTRUCTIONS:
            for pr in PRIOR_STRENGTHS:
                k = (model, iname, pr)
                if tot.get(k):
                    wilson[k] = wilson_interval(ungrounded.get(k, 0), tot[k])
    print("\nPARAMETRIC-LEAKAGE RATE vs PRIOR STRENGTH  (judge; prior strength 1=obscure .. 5=universal)")
    for model, _ in MODELS:
        for iname, _ in SYSTEM_INSTRUCTIONS:
            cells = []
            for pr in PRIOR_STRENGTHS:
                k = (model, iname, pr)
                if k in wilson:
                    p, lo, hi = wilson[k]
                    cells.append(f"P{pr}={p:.2f}[{lo:.2f},{hi:.2f}]")
                else:
                    cells.append(f"P{pr}=--")
            print("  " + f"{model} / {iname}".ljust(30) + "  " + "  ".join(cells))
    print("\nPERMISSIVE - SOURCE_EXCLUSIVE and WEAK_GROUNDING - SOURCE_EXCLUSIVE parametric-leakage-rate gaps, per prior strength:")
    for model, _ in MODELS:
        for gap_name in ("FLAG_INVITING", "WEAK_GROUNDING"):
            gaps = []
            for pr in PRIOR_STRENGTHS:
                ks, kg = (model, "SOURCE_EXCLUSIVE", pr), (model, gap_name, pr)
                if tot.get(ks) and tot.get(kg):
                    gaps.append(f"P{pr}={ungrounded.get(kg,0)/tot[kg] - ungrounded.get(ks,0)/tot[ks]:+.2f}")
                else:
                    gaps.append(f"P{pr}=--")
            print("  " + f"{model} {gap_name}-SOURCE_EXCLUSIVE".ljust(36) + "  " + "  ".join(gaps))
    with open(ABSTENTION_CURVE, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "instruction", "prior_strength", "n", "ungrounded", "ungrounded_rate", "lo", "hi",
                    "reports_parametric_answer_rate", "verbatim_abstention_rate"])
        for model, _ in MODELS:
            for iname, _ in SYSTEM_INSTRUCTIONS:
                for pr in PRIOR_STRENGTHS:
                    k = (model, iname, pr)
                    if k not in wilson:
                        continue
                    p, lo, hi = wilson[k]
                    w.writerow([model, iname, pr, tot[k], ungrounded.get(k, 0), f"{p:.4f}", f"{lo:.4f}",
                                f"{hi:.4f}", f"{lex.get(k,0)/tot[k]:.4f}", f"{vabst.get(k,0)/tot[k]:.4f}"])
    print(f"\n  wrote curve to {ABSTENTION_CURVE}")

def tradeoff_rows(caveat_rows, ungrounded_rows):
    cdf = pd.DataFrame(caveat_rows, columns=["model", "instruction", "severity", "label"])
    udf = pd.DataFrame(ungrounded_rows, columns=["model", "instruction", "prior_strength", "label"])
    entries = []
    for model, _ in MODELS:
        for iname, _ in SYSTEM_INSTRUCTIONS: # compare the results between the tests
            for lv in SEVERITIES:
                f = cdf[(cdf["model"] == model) & (cdf["instruction"] == iname) & (cdf["severity"] == lv)]
                l = udf[(udf["model"] == model) & (udf["instruction"] == iname) & (udf["prior_strength"] == lv)]
                if f.empty and l.empty:
                    continue
                entries.append({"model": model, "instruction": iname, "severity": lv,
                                "caveat_n": len(f), "caveat_rate": float((f["label"] == QUESTIONED).mean()) if not f.empty else None,
                                "abstention_n": len(l), "faithful_rate": float((l["label"] == FAITHFUL).mean()) if not l.empty else None})
    return entries

def tradeoff():
    def load(path):
        try:
            return [json.loads(l) for l in open(path)]
        except FileNotFoundError:
            return None
    caveat_rows = load(CAVEAT_RESULTS)
    ungrounded_rows = load(ABSTENTION_RESULTS)
    if caveat_rows is None:
        print(f"  no {CAVEAT_RESULTS} yet -- run: python3 harness.py caveat [N]")
    if ungrounded_rows is None:
        print(f"  no {ABSTENTION_RESULTS} yet -- run: python3 harness.py abstention [N]")
    entries = tradeoff_rows(caveat_rows or [], ungrounded_rows or [])
    if not entries:
        return
    print("TRADE-OFF -- error-flagging vs faithful abstention, per model x instruction x severity (higher = better on both)")
    print("  flagging = error-flagging rate at this perturbation severity (denominator includes abstentions)")
    print("  faithful = faithful rate (1 - parametric-leakage rate) at the matching prior-strength level")
    print("  S0 = unperturbed control: a flag at S0 is a false positive; it has no abstention counterpart")
    for e in entries:
        fr = "--" if e["caveat_rate"] is None else f"{e['caveat_rate']:.2f} (n={e['caveat_n']})"
        ar = "--" if e["faithful_rate"] is None else f"{e['faithful_rate']:.2f} (n={e['abstention_n']})"
        print(f"  {e['model']:<24} {e['instruction']:<10}  S{e['severity']}  flagging {fr:>14}   faithful {ar:>14}")

if __name__ == "__main__": # only run file if executed directly 
    args = sys.argv[1:] 
    if args and args[0] == "caveat": # if args and args[0] = if the first argument is caveat
        run_caveat(int(args[1]) if len(args) > 1 else N_PER_CELL)
    elif args and args[0] == "abstention":
        run_ungrounded(int(args[1]) if len(args) > 1 else N_PER_CELL)
    elif args and args[0] == "tradeoff":
        tradeoff()
    elif args and args[0] == "rescore":
        rescore_caveat(args[1:] or None)
    elif args and args[0] == "endorsement":
        endorsement_breakdown()
    elif args and not args[0].isdigit():
        print("usage: python3 harness.py [N] | caveat [N] | abstention [N] | rescore | endorsement | tradeoff")
        sys.exit(1)
    else:
        n = int(args[0]) if args else N_PER_CELL
        print_plan(n)
        print()
        print_abstention_plan(n)
        print("\n  (No API calls were made. To execute: python3 harness.py caveat [N]  or  python3 harness.py abstention [N]. Joint readout: python3 harness.py tradeoff)")
