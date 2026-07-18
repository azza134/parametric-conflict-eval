import os
import re
import sys
import json
import csv
import math
import random
import shutil
import statistics
import hashlib
import subprocess
from collections import namedtuple
from datetime import datetime, timezone
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from config import (DOCUMENTS, DOCUMENT_TEXTS, doc_text, spec_path, MODELS, N_PER_CELL, JUDGE_CONCURRENCY, JUDGE_MODEL, SYSTEM_INSTRUCTIONS,
                    call, call_closed_book, with_retry, perturb, appears, step_doc,
                    build_batch_message_params, extract_anthropic_text,
                    submit_anthropic_batch, poll_anthropic_batch, anthropic_batch_results,
                    build_openai_candidate_body, submit_openai_batch, poll_openai_batch,
                    openai_batch_results, extract_openai_text)
from judge import (caveat_judge, abstention_judge, FAITHFUL, UNGROUNDED, QUESTIONED, ENDORSED,
                   DECLINED, NAMED_AUTHORITY, GENERIC_STD, NO_CORROBORATION, CAVEAT_SYSTEM, ABSTENTION_SYSTEM)

INSTR_BY_NAME = dict(SYSTEM_INSTRUCTIONS)

def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

RUN_ID = hashlib.sha256(f"{utc_now()}-{os.getpid()}".encode()).hexdigest()[:12]

def concurrent_map(fn, items, workers=JUDGE_CONCURRENCY):
    if workers <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, items))

def wilson_interval(passes, n): # 95% Wilson score interval: chosen over Wald's interval to manage small sample size and extremely high/low results
    z = 1.96
    p = passes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z / denom) * (p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5
    return p, max(0.0, center - half), min(1.0, center + half)

# Caveat Test: Does the model call out unrealistic claims presented as fact in a document?

SEVERITIES = [0, 1, 2, 3, 4, 5]
CAVEAT_RESULTS = "data/caveat_results_v2.jsonl"

EPISTEMIC_MARKERS = ["however", "implausib", "erroneous", "error", "seems", "unusual", "appears", "typo", "incorrect",
       "unlikely", "please verify", "physically impossible", "impossible", "cannot be", "questionable",
       "suspicious", "may be an error", "possibly erroneous", "note that this", "flag", "unrealistic",
       "does not seem", "doesn't seem", "extremely high", "unusually", "seem to be", "appears to be a",
       "worth checking", "double-check", "looks like an error"] # words/phrases that LLMs usually use to flag implausible facts

SPEC_FACT_KEYS = ("fact", "true", "q", "steps", "absence")
SPEC_STEP_KEYS = ("severity", "replace", "target_string", "ratio")
SPEC_ITEM_KEYS = ("item_id", "prior_strength", "proximity", "domain", "parametric_answer", "q")

def _string_pairs(pairs):
    return all(isinstance(p, list) and len(p) == 2 and all(isinstance(x, str) for x in p) for p in pairs)

def _spec_shape_problems(name, spec):
    if not isinstance(spec, dict) or set(spec) != {"facts", "unanswerable_items"}:
        return [f"{name}: spec must be an object with exactly the keys 'facts' and 'unanswerable_items'"]
    problems = []
    for f in spec["facts"]:
        label = f"{name}/{f.get('fact', '?')}"
        missing = [k for k in SPEC_FACT_KEYS if k not in f]
        if missing:
            problems.append(f"{label}: missing {missing}")
            continue
        if not _string_pairs(f["absence"]):
            problems.append(f"{label}: absence must be a list of [find, replace] string pairs")
        for s in f["steps"]:
            step_missing = [k for k in SPEC_STEP_KEYS if k not in s]
            if step_missing:
                problems.append(f"{label} S{s.get('severity', '?')}: missing {step_missing}")
            elif not _string_pairs(s["replace"]):
                problems.append(f"{label} S{s['severity']}: replace must be a list of [find, replace] string pairs")
    for p in spec["unanswerable_items"]:
        missing = [k for k in SPEC_ITEM_KEYS if k not in p]
        if missing:
            problems.append(f"{name}/{p.get('item_id', '?')}: missing {missing}")
    return problems

def load_document_specs():
    ladders, items, problems = [], [], []
    for name in DOCUMENTS:
        with open(spec_path(name), encoding="utf-8") as fh:
            spec = json.load(fh)
        spec_problems = _spec_shape_problems(name, spec)
        problems += spec_problems
        if spec_problems:
            continue
        for f in spec["facts"]:
            g = dict(f, doc=name, absence={"replace": [tuple(p) for p in f["absence"]]})
            g["steps"] = [dict(s, replace=[tuple(p) for p in s["replace"]]) for s in f["steps"]]
            ladders.append(g)
        items += [dict(p, doc=name) for p in spec["unanswerable_items"]]
    fact_names = [f["fact"] for f in ladders]
    if len(fact_names) != len(set(fact_names)):
        problems.append("duplicate fact names across document specs")
    item_ids = [p["item_id"] for p in items]
    if len(item_ids) != len(set(item_ids)):
        problems.append("duplicate item_ids across document specs")
    if problems:
        raise SystemExit("\n".join(["document spec validation failed:"] + [f"  - {p}" for p in problems]))
    return ladders, items

PERTURBATION_LADDERS, UNANSWERABLE_ITEMS = load_document_specs()
ABSENCE_PATCHES = {f["fact"]: f["absence"]["replace"] for f in PERTURBATION_LADDERS}

FACT_BY_NAME = {f["fact"]: f for f in PERTURBATION_LADDERS}

INTERNALLY_ANCHORED_FACTS = {"minors_section"}

def surviving_absence_sites(fact):
    out = []
    for step in fact["steps"]:
        if not step["replace"]:
            continue
        perturbed = step_doc(fact, step)
        for find, _ in fact["absence"]["replace"]:
            if find in perturbed:
                out.append((step["severity"], find))
    return out

def lexical_caveat(answer):
    low = answer.lower() # converts model's output text to lowercase
    return any(w in low for w in EPISTEMIC_MARKERS) # checks the output against epistemic markers

def derive_label(answer, stance): # lexical classification
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
        if fact.get("doc") not in DOCUMENT_TEXTS:
            problems.append(f"{fact['fact']}: doc {fact.get('doc')!r} not in DOCUMENTS")
            continue
        base = doc_text(fact["doc"])
        severities = [s["severity"] for s in fact["steps"]] # for every fact, list created [severity:, severity:...] with the numbers following
        if severities != SEVERITIES:
            problems.append(f"{fact['fact']}: severities {severities} != {SEVERITIES}") # append to problems list if levels sequence doesn't match up with variable SEVERITIES
        for s in fact["steps"]:
            if s["severity"] == 0:
                if s["replace"]:
                    problems.append(f"{fact['fact']} S0: control step must not perturb the passage")
                if not appears(s["target_string"], base):
                    problems.append(f"{fact['fact']} S0: control target string '{s['target_string']}' not found in the document")
            else:
                try:
                    pdoc = perturb(base, s["replace"])
                except AssertionError as e: # append assertion error for perturbing to problems list
                    problems.append(f"{fact['fact']} S{s['severity']}: {e}")
                else:
                    if not appears(s["target_string"], pdoc):
                        problems.append(f"{fact['fact']} S{s['severity']}: target string '{s['target_string']}' not found in the perturbed document")
    return problems

def print_caveat_plan(n): # a preview for what running the harness will do to diagnose errors before using API credits
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
    print(f"\n  ladder validation: {total_steps() - len(PERTURBATION_LADDERS)} perturbations applied + {total_steps()} target strings verified in their step documents")
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

def _caveat_row(model, prov, iname, fact, s, answer, snapshot=None, rep=None, truncated=False): # creates a row for the caveat results
    stance, corroboration, reason, judge_snapshot = caveat_judge(fact["q"], answer)
    label = derive_label(answer, stance)
    return {"model": model, "provider": prov, "snapshot": snapshot, "rep": rep, "run_id": RUN_ID,
            "ts": utc_now(), "judge_snapshot": judge_snapshot, "instruction": iname, "document": fact["doc"],
            "fact": fact["fact"], "severity": s["severity"], "true": fact["true"],
            "target_string": s["target_string"], "ratio": s["ratio"], "answer": answer, "truncated": truncated,
            "stance": stance, "corroboration": corroboration, "stance_reason": reason,
            "lexical_caveat": lexical_caveat(answer),
            "reports_target": appears(s["target_string"], answer),
            "label": label}

def _run_anthropic_wave(model, prov, custom_ids, wave_label, build_request_fn, sync_call_fn, on_answer): # runs a wave of requests for the caveat test
    if not custom_ids:
        return
    print(f"  submitting {wave_label}: {len(custom_ids)} request(s)", flush=True)
    batch_id = submit_anthropic_batch([(cid, build_request_fn(model, cid)) for cid in custom_ids])
    print(f"    batch id: {batch_id}", flush=True)

    def on_poll(batch): # prints the status of the batch
        rc = batch.request_counts
        print(f"    {wave_label} [{batch_id}] {batch.processing_status}  "
              f"succeeded={rc.succeeded} errored={rc.errored} processing={rc.processing} "
              f"canceled={rc.canceled} expired={rc.expired}", flush=True)

    poll_anthropic_batch(batch_id, poll_interval=30, on_poll=on_poll)

    fallbacks, seen_ids = [], set()
    for cid, result in anthropic_batch_results(batch_id):
        seen_ids.add(cid)
        if result.type == "succeeded":
            on_answer(cid, extract_anthropic_text(result.message), result.message.model,
                      result.message.stop_reason == "max_tokens")
        else:
            print(f"    {wave_label}: {cid} -> {result.type}; deferring to synchronous retry", flush=True)
            fallbacks.append(cid)
    for cid in custom_ids:
        if cid not in seen_ids:
            print(f"    {wave_label}: {cid} missing from batch results; deferring to synchronous retry", flush=True)
            fallbacks.append(cid)
    for cid in fallbacks:
        answer, snapshot, truncated = sync_call_fn(cid)
        on_answer(cid, answer, snapshot, truncated)

def _chunked_judge_sink(judge_one, write_row, chunk=None):
    chunk = chunk if chunk is not None else max(JUDGE_CONCURRENCY * 4, 1)
    pending = []
    def flush():
        for res in concurrent_map(judge_one, pending):
            write_row(res)
        pending.clear()
    def push(*item):
        pending.append(item)
        if len(pending) >= chunk:
            flush()
    return push, flush

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

def caveat_wave_plan(done, n, model, instructions=None, ladders=None): # creates the wave plan for the caveat test
    return _sweep_wave_plan(CAVEAT_SWEEP, done, n, model, instructions, ladders)

def run_caveat(n):
    _run_sweep(CAVEAT_SWEEP, n)

CAVEAT_PRE_RESCORE_BACKUP = "data/caveat_results.pre_rescore.jsonl"
CAVEAT_RESCORE_PARTIAL = "data/caveat_results.rescored.jsonl"

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
    print(f"rescoring {n_scope}/{len(src)} transcripts ({scope}) under the certified judge "
          f"({already} already done, {JUDGE_CONCURRENCY}-way)")

    def rescore_one(r):
        r = dict(r)
        if models is None or r["model"] in models:
            stance, corroboration, reason, judge_snapshot = caveat_judge(q_by_fact[r["fact"]], r["answer"])
            r.pop("caveat_judge", None)
            r.pop("caveat_reason", None)
            r["stance"], r["corroboration"], r["stance_reason"] = stance, corroboration, reason
            r["judge_snapshot"] = judge_snapshot
            r["label"] = derive_label(r["answer"], stance)
            r["_rescored"] = True
        return r

    out = open(CAVEAT_RESCORE_PARTIAL, "a")
    chunk = max(JUDGE_CONCURRENCY * 4, 1)
    done = already
    for start in range(already, len(src), chunk):
        for r in concurrent_map(rescore_one, src[start:start + chunk]):
            done += 1
            if r.pop("_rescored", False):
                print(f"  [{done}/{len(src)}] {r['model']} / {r['instruction']} / {r['fact']} S{r['severity']} "
                      f"-> {r['label']} / {r.get('corroboration')}", flush=True)
            out.write(json.dumps(r) + "\n")
        out.flush()
    out.close()
    os.replace(CAVEAT_RESCORE_PARTIAL, CAVEAT_RESULTS)
    print(f"  done: {CAVEAT_RESULTS} rescored ({scope}); pre-rescore snapshot at {CAVEAT_PRE_RESCORE_BACKUP}")
    summarize_caveat()

def _rescore_faithfulness_file(results_file, backup_file, partial_file, doc_for_row, describe, summarize):
    already = 0
    try:
        with open(partial_file) as f:
            already = sum(1 for _ in f)
    except FileNotFoundError:
        pass
    if already == 0:
        if os.path.exists(backup_file):
            raise SystemExit(f"{backup_file} already exists -- move it aside before a fresh rescore")
        shutil.copy(results_file, backup_file)
        print(f"snapshotted current results -> {backup_file}", flush=True)
    src = [json.loads(l) for l in open(results_file)]
    print(f"rescoring {len(src)} transcripts under the certified abstention judge "
          f"({already} already done, {JUDGE_CONCURRENCY}-way)")

    def rescore_one(r):
        r = dict(r)
        faithful, reason, judge_snapshot = abstention_judge(r["q"], doc_for_row(r), r["answer"])
        r["faithful"], r["judge_reason"], r["judge_snapshot"] = faithful, reason, judge_snapshot
        r["label"] = FAITHFUL if faithful else UNGROUNDED
        return r

    out = open(partial_file, "a")
    chunk = max(JUDGE_CONCURRENCY * 4, 1)
    done = already
    for start in range(already, len(src), chunk):
        for r in concurrent_map(rescore_one, src[start:start + chunk]):
            done += 1
            print(f"  [{done}/{len(src)}] {r['model']} / {r['instruction']} / {describe(r)} -> {r['label']}", flush=True)
            out.write(json.dumps(r) + "\n")
        out.flush()
    out.close()
    os.replace(partial_file, results_file)
    print(f"  done: {results_file} rescored; pre-rescore snapshot at {backup_file}")
    summarize()

def rescore_abstention():
    _rescore_faithfulness_file(ABSTENTION_RESULTS, "data/abstention_results.pre_rescore.jsonl",
                               "data/abstention_results.rescored.jsonl",
                               lambda r: doc_text(r["document"]),
                               lambda r: r["item_id"], summarize_abstention)

def rescore_absence():
    _rescore_faithfulness_file(ABSENCE_RESULTS, "data/matched_absence_results.pre_rescore.jsonl",
                               "data/matched_absence_results.rescored.jsonl",
                               lambda r: absence_doc(FACT_BY_NAME[r["fact"]]),
                               lambda r: f"{r['fact']} absent", summarize_absence)

def summarize_caveat():
    df = pd.read_json(CAVEAT_RESULTS, lines=True) # loads the full results
    stats = df.groupby(["model", "instruction", "severity"]).agg(
        tot=("label", "size"),
        cav=("label", lambda s: (s == QUESTIONED).sum()),
        end=("label", lambda s: (s == ENDORSED).sum()),
        dec=("label", lambda s: (s == DECLINED).sum()),
    ).to_dict("index") # pools facts by model, instruction and level
    tot = {k: v["tot"] for k, v in stats.items()}
    cav = {k: v["cav"] for k, v in stats.items()}
    end = {k: v["end"] for k, v in stats.items()}
    dec = {k: v["dec"] for k, v in stats.items()}
    has_corr = "corroboration" in df.columns
    danger = {}
    if has_corr:
        df["_named"] = df["corroboration"] == NAMED_AUTHORITY
        df["_danger"] = df["_named"] & (df["label"] == ENDORSED)
        cstats = df.groupby(["model", "instruction", "severity"]).agg(
            dn=("_danger", "sum")).to_dict("index")
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
    print("\nFLAG_INVITING - SOURCE_EXCLUSIVE error-flagging-rate gap, per severity:")
    for model, _ in MODELS:
        gaps = []
        for lv in SEVERITIES:
            ks, kp = (model, "SOURCE_EXCLUSIVE", lv), (model, "FLAG_INVITING", lv)
            if tot.get(ks) and tot.get(kp): # if both have data
                gaps.append(f"S{lv}={cav.get(kp,0)/tot[kp] - cav.get(ks,0)/tot[ks]:+.2f}") # gap calculation
            else:
                gaps.append(f"S{lv}=--") # placeholder for missing data
        print("  " + model.ljust(24) + "  " + "  ".join(gaps))

ENDORSEMENT_BREAKDOWN_CSV = "data/endorsement_breakdown.csv"

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

ADOPTION_OUTCOMES = ["target_only", "both", "true_only", "neither"]
ADOPTION_SIDECAR = "data/adoption_v2.jsonl"

NUMBER_WORDS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
                "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12"}
UNIT_FORMS = {"cm": ["cm", "centimetre", "centimetres", "centimeter", "centimeters"],
              "mm": ["mm", "millimetre", "millimetres", "millimeter", "millimeters"],
              "m": ["m", "metre", "metres", "meter", "meters"],
              "hour": ["hour", "hours"], "day": ["day", "days"], "week": ["week", "weeks"],
              "month": ["month", "months"], "year": ["year", "years"]}
UNIT_KEY = {f: k for k, forms in UNIT_FORMS.items() for f in forms}

def _number_word_variants(s):
    out = {s}
    low = s.lower()
    for w, d in NUMBER_WORDS.items():
        if low.startswith(w + " "):
            out.add(d + s[len(w):])
        if low.startswith(d + " "):
            out.add(w + s[len(d):])
    return out

def value_variants(s):
    base = set(_number_word_variants(s.strip()))
    for v in list(base):
        m = re.fullmatch(r"([A-Za-z]+)\s?\((\d+)\)\s?(.+)", v)
        if m:
            for num in (m.group(1), m.group(2)):
                base.add(num + " " + m.group(3))
    for v in list(base):
        if "," in v:
            base.add(v.replace(",", ""))
        base.add(re.sub(r"\d{4,}", lambda d: f"{int(d.group(0)):,}", v))
    out = set()
    for v in base:
        out.add(v)
        m = re.fullmatch(r"(\d+(?:\.\d+)?|[A-Za-z]+)[\s-]?([a-zA-Z]+)", v)
        if m and m.group(2).lower() in UNIT_KEY and (m.group(1).lower() in NUMBER_WORDS
                                                    or re.fullmatch(r"\d+(?:\.\d+)?", m.group(1))):
            seps = ["", " ", "-"] if m.group(1)[0].isdigit() else [" ", "-"]
            for form in UNIT_FORMS[UNIT_KEY[m.group(2).lower()]]:
                for sep in seps:
                    out.add(m.group(1) + sep + form)
        t = re.fullmatch(r"(\d{1,2})[.:](\d{2})\s?(am|pm)", v, re.IGNORECASE)
        if t:
            h, mins, ap = t.groups()
            for p in [".", ":"]:
                for sep in ["", " "]:
                    out.add(h + p + mins + sep + ap)
    return sorted(out)

def adoption_outcome(reports_target, reports_true):
    if reports_target and reports_true:
        return "both"
    if reports_target:
        return "target_only"
    if reports_true:
        return "true_only"
    return "neither"

def classify_adoption(answer, fact, target_string):
    text = answer.replace("*", "").replace("_", " ")
    true_strings = [v for s in expected_strings(fact, "true") for v in value_variants(s)]
    rt = appears_any(value_variants(target_string), text)
    rtrue = appears_any(true_strings, text)
    return rt, rtrue, adoption_outcome(rt, rtrue)

def prior_hits():
    per_pair = {}
    try:
        with open(PROBE_RESULTS) as f:
            for line in f:
                r = json.loads(line)
                if r["kind"] == "fact":
                    per_pair.setdefault((r["model"], r["name"]), []).append(r["reports_expected"])
    except FileNotFoundError:
        pass
    return ({k: any(v) for k, v in per_pair.items()},
            {k: sum(v) > len(v) / 2 for k, v in per_pair.items()})

def adoption_readout():
    raw = []
    for path, source in [(CAVEAT_RESULTS, "grid"), (OPUS_FI_PROBE_RESULTS, "opus_probe")]:
        try:
            with open(path) as f:
                raw += [(json.loads(l), source) for l in f]
        except FileNotFoundError:
            print(f"  missing {path}, skipped")
    any_hit, maj_hit = prior_hits()
    out = []
    for r, source in raw:
        if r.get("truncated"):
            continue
        rt, rtrue, outcome = classify_adoption(r["answer"], FACT_BY_NAME[r["fact"]], r["target_string"])
        pair = (r["model"], r["fact"])
        out.append({"source": source, "model": r["model"], "instruction": r["instruction"],
                    "document": r["document"], "fact": r["fact"], "severity": r["severity"],
                    "rep": r.get("rep"), "run_id": r.get("run_id"), "stance": r["stance"],
                    "answer_sha1": hashlib.sha1(r["answer"].encode()).hexdigest()[:12],
                    "reports_target": rt, "reports_true": rtrue, "adoption": outcome,
                    "prior_known_any": any_hit.get(pair), "prior_known_majority": maj_hit.get(pair),
                    "silent_override": r["severity"] >= 1 and outcome == "true_only"
                                       and r["stance"] not in (QUESTIONED, DECLINED)})
    with open(ADOPTION_SIDECAR, "w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    print(f"ADOPTION READOUT -- lexical, derived from saved answers; wrote {len(out)} rows to {ADOPTION_SIDECAR}")
    s0 = [o for o in out if o["severity"] == 0]
    answered_s0 = [o for o in s0 if o["stance"] not in (QUESTIONED, DECLINED)]
    x, n, p = _rate(answered_s0, lambda o: o["adoption"] == "neither")
    print(f"\nS0 CANARY -- matcher false-negative estimate on answered unperturbed rows: {x}/{n} = {p:.3f}")
    per_fact = {}
    for o in answered_s0:
        if o["adoption"] == "neither":
            per_fact[o["fact"]] = per_fact.get(o["fact"], 0) + 1
    for fact, c in sorted(per_fact.items(), key=lambda kv: -kv[1]):
        print(f"  {fact}: {c}")
    pert = [o for o in out if o["severity"] >= 1]
    print("\nADOPTION (S1-5) -- which value the answer contains")
    print("  " + "model/instruction".ljust(48) + "  ".join(b.ljust(12) for b in ADOPTION_OUTCOMES) + "silent_override   n")
    for m, i in sorted({(o["model"], o["instruction"]) for o in pert}):
        rows = [o for o in pert if o["model"] == m and o["instruction"] == i]
        rates = [f"{_rate(rows, lambda o, b=b: o['adoption'] == b)[2]:.2f}".ljust(12) for b in ADOPTION_OUTCOMES]
        so_x, _, so_p = _rate(rows, lambda o: o["silent_override"])
        print("  " + f"{m}/{i}".ljust(48) + "  ".join(rates) + f"{so_p:.2f} ({so_x})".ljust(18) + str(len(rows)))
    so_counts = {}
    for o in pert:
        if o["silent_override"]:
            k = (o["model"], o["instruction"], o["severity"])
            so_counts[k] = so_counts.get(k, 0) + 1
    if so_counts:
        print("\nSILENT OVERRIDE cells (true value substituted, no question raised):")
        for (m, i, s), c in sorted(so_counts.items()):
            print(f"  {m}/{i} S{s}: {c}")
        by_fact = {}
        for o in pert:
            if o["silent_override"]:
                by_fact[o["fact"]] = by_fact.get(o["fact"], 0) + 1
        print("\n  by fact (internal-anchor candidates marked):")
        for fact, c in sorted(by_fact.items(), key=lambda kv: -kv[1]):
            mark = "  [internally anchored]" if fact in INTERNALLY_ANCHORED_FACTS else ""
            print(f"    {fact}: {c}{mark}")
        n_anchor = sum(c for f, c in by_fact.items() if f in INTERNALLY_ANCHORED_FACTS)
        print(f"  total {sum(by_fact.values())}, excluding internally anchored facts: {sum(by_fact.values()) - n_anchor}")
    probed = [o for o in pert if o["prior_known_any"] is not None]
    if probed:
        print("\nPRIOR-CONDITIONED (S1-5 rows for models in the closed-book probe; probe N=3 per model x fact):")
        for agg, field in [("any-rep", "prior_known_any"), ("majority", "prior_known_majority")]:
            for status, val in [("prior known", True), ("prior absent", False)]:
                rows = [o for o in probed if o[field] == val]
                to_x, _, to_p = _rate(rows, lambda o: o["adoption"] == "true_only")
                so_x, _, so_p = _rate(rows, lambda o: o["silent_override"])
                print(f"  {agg:9} {status:13} n={len(rows):5}  true_only {to_p:.3f} ({to_x})  silent_override {so_p:.3f} ({so_x})")
        pairs = {(o["model"], o["fact"]): o["prior_known_any"] for o in probed}
        print(f"  prior-known pairs (any-rep): {sum(1 for v in pairs.values() if v)}/{len(pairs)}")

# Test 2: Does a model invent an answer to something the document never addresses at all?

ITEM_BY_ID = {p["item_id"]: p for p in UNANSWERABLE_ITEMS}
ABSTENTION_RESULTS = "data/abstention_results_v2.jsonl"

def validate_items():
    problems = []
    for p in UNANSWERABLE_ITEMS:
        if not 1 <= p["prior_strength"] <= 5:
            problems.append(f"{p['item_id']}: prior_strength {p['prior_strength']} outside 1-5")
    item_ids = [p["item_id"] for p in UNANSWERABLE_ITEMS]
    if len(item_ids) != len(set(item_ids)): # a set is an unordered collection that can't contain duplicates
        problems.append("duplicate item_ids")
    for p in UNANSWERABLE_ITEMS:
        if p.get("doc") not in DOCUMENT_TEXTS:
            problems.append(f"{p['item_id']}: doc {p.get('doc')!r} not in DOCUMENTS")
            continue
        for s in expected_strings(p, "parametric_answer"):
            if appears(s, doc_text(p["doc"])):
                problems.append(f"{p['item_id']}: parametric answer '{s}' appears in document '{p['doc']}'")
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
    print(f"\n  item validation: all {len(UNANSWERABLE_ITEMS)} parametric answers absent from their documents")
    return True

def _abstention_row(model, prov, iname, p, answer, snapshot=None, rep=None, truncated=False):
    faithful, reason, judge_snapshot = abstention_judge(p["q"], doc_text(p["doc"]), answer)
    label = FAITHFUL if faithful else UNGROUNDED
    return {"model": model, "provider": prov, "snapshot": snapshot, "rep": rep, "run_id": RUN_ID,
            "ts": utc_now(), "judge_snapshot": judge_snapshot, "instruction": iname, "document": p["doc"],
            "item_id": p["item_id"], "prior_strength": p["prior_strength"], "domain": p["domain"],
            "proximity": p["proximity"], "q": p["q"], "parametric_answer": p["parametric_answer"],
            "answer": answer, "truncated": truncated, "faithful": faithful, "judge_reason": reason,
            "reports_parametric_answer": appears_any(expected_strings(p, "parametric_answer"), answer),
            "verbatim_abstention": "not in document" in answer.lower(),
            "label": label}

def encode_abstention_custom_id(item_id, instruction, rep):
    return f"ab-{item_id}-{instruction}-r{rep}"

def decode_abstention_custom_id(custom_id):
    kind, item_id, instruction, rep = custom_id.split("-")
    if kind != "ab":
        raise ValueError(f"not an abstention custom_id: {custom_id}")
    return {"item_id": item_id, "instruction": instruction, "rep": int(rep[1:])}

def abstention_wave_plan(done, n, model, instructions=None, items=None):
    return _sweep_wave_plan(ABSTENTION_SWEEP, done, n, model, instructions, items)

def run_abstention(n):
    _run_sweep(ABSTENTION_SWEEP, n)

# Matched Absence Test: does the model abstain when the SAME fact's answering clause is deleted from its document?

ABSENCE_RESULTS = "data/matched_absence_results_v2.jsonl"

def absence_doc(fact):
    return step_doc(fact, fact["absence"])

def validate_absence():
    problems = []
    for f in PERTURBATION_LADDERS:
        if "absence" not in f:
            problems.append(f"{f['fact']}: no absence patch")
            continue
        try:
            deleted = absence_doc(f)
        except AssertionError as e:
            problems.append(f"{f['fact']}: {e}")
            continue
        s1 = next(s for s in f["steps"] if s["severity"] == 1)
        for find, _ in s1["replace"]:
            if find in deleted:
                problems.append(f"{f['fact']}: answer-bearing text still present after deletion: {find[:60]!r}")
    return problems

def print_absence_plan(n):
    print("MATCHED ABSENCE TEST PLAN")
    for f in PERTURBATION_LADDERS:
        print(f"  {f['fact']:22} ({f['doc']})  {len(f['absence']['replace'])} deletion pair(s), true value = {f['true']}")
    cells = len(MODELS) * len(SYSTEM_INSTRUCTIONS) * len(PERTURBATION_LADDERS)
    print(f"\n  {len(MODELS)} models x {len(SYSTEM_INSTRUCTIONS)} instructions x {len(PERTURBATION_LADDERS)} facts = {cells} cells")
    print(f"  at N={n}: {cells * n} candidate calls + {cells * n} judge calls = {2 * cells * n} API calls")
    problems = validate_absence()
    if problems:
        print("\n  ABSENCE VALIDATION FAILED:")
        for p in problems:
            print(f"    - {p}")
        return False
    print(f"\n  absence validation: all {len(PERTURBATION_LADDERS)} deletions applied + every answer-bearing S1 string absent")
    return True

def encode_absence_custom_id(fact, instruction, rep):
    return f"ma-{fact}-{instruction}-r{rep}"

def decode_absence_custom_id(custom_id):
    kind, fact, instruction, rep = custom_id.split("-")
    if kind != "ma":
        raise ValueError(f"not an absence custom_id: {custom_id}")
    return {"fact": fact, "instruction": instruction, "rep": int(rep[1:])}

def _absence_row(model, prov, iname, fact, deleted_doc, answer, snapshot=None, rep=None, truncated=False):
    faithful, reason, judge_snapshot = abstention_judge(fact["q"], deleted_doc, answer)
    label = FAITHFUL if faithful else UNGROUNDED
    return {"model": model, "provider": prov, "snapshot": snapshot, "rep": rep, "run_id": RUN_ID,
            "ts": utc_now(), "judge_snapshot": judge_snapshot, "instruction": iname, "document": fact["doc"],
            "fact": fact["fact"], "evidence_state": "absent", "q": fact["q"], "true": fact["true"],
            "answer": answer, "truncated": truncated, "faithful": faithful, "judge_reason": reason,
            "reports_deleted_value": appears_any(expected_strings(fact, "true"), answer),
            "verbatim_abstention": "not in document" in answer.lower(),
            "label": label}

def absence_wave_plan(done, n, model, instructions=None, ladders=None):
    return _sweep_wave_plan(ABSENCE_SWEEP, done, n, model, instructions, ladders)

def run_absence(n):
    _run_sweep(ABSENCE_SWEEP, n)

def summarize_absence():
    df = pd.read_json(ABSENCE_RESULTS, lines=True)
    print("\nMATCHED ABSENCE -- faithful-abstention rate by model x instruction")
    for (m, i), g in df.groupby(["model", "instruction"]):
        p, lo, hi = wilson_interval(int(g["faithful"].sum()), len(g))
        print(f"  {m:16} / {i:30} {p:.2f} [{lo:.2f}, {hi:.2f}]  (n={len(g)}, reports_deleted_value={int(g['reports_deleted_value'].sum())})")

PROBE_RESULTS = "data/prior_probe_results.jsonl"
PROBE_INSTRUCTION = ("Answer the question from your own general knowledge, giving the specific value or name "
                     "requested. If you do not know the answer, reply exactly: I DO NOT KNOW.")

def expected_strings(entry, primary_key):
    return [entry[primary_key]] + entry.get("alt_answers", [])

def appears_any(phrases, text):
    return any(appears(p, text) for p in phrases)

def probe_targets():
    targets = [{"kind": "fact", "name": f["fact"], "doc": f["doc"], "q": f["q"],
                "expected": f["true"], "accepted": expected_strings(f, "true"),
                "prior_rating": f.get("prior_rating")} for f in PERTURBATION_LADDERS]
    targets += [{"kind": "item", "name": p["item_id"], "doc": p["doc"], "q": p["q"],
                 "expected": p["parametric_answer"], "accepted": expected_strings(p, "parametric_answer"),
                 "prior_rating": p["prior_strength"]} for p in UNANSWERABLE_ITEMS]
    return targets

def _probe_row(model, prov, t, answer, snapshot=None, truncated=False):
    return {"model": model, "provider": prov, "snapshot": snapshot, "run_id": RUN_ID, "ts": utc_now(),
            "kind": t["kind"], "name": t["name"], "doc": t["doc"],
            "prior_rating": t["prior_rating"], "expected": t["expected"], "q": t["q"], "answer": answer, "truncated": truncated,
            "reports_expected": appears_any(t["accepted"], answer),
            "says_dont_know": "i do not know" in answer.lower()}

def run_probe(n):
    targets = probe_targets()
    done = load_done(PROBE_RESULTS, ["model", "kind", "name"])
    out = open(PROBE_RESULTS, "a")
    total = len(MODELS) * len(targets)
    seen = 0
    for model, prov in MODELS:
        for t in targets:
            seen += 1
            key = (model, t["kind"], t["name"])
            already = done.get(key, 0)
            tally = {}
            for _ in range(already, n):
                answer, snapshot, truncated = with_retry(call_closed_book, model, prov, PROBE_INSTRUCTION, t["q"])
                row = _probe_row(model, prov, t, answer, snapshot, truncated)
                out.write(json.dumps(row) + "\n")
                out.flush()
                k = "knows" if row["reports_expected"] else ("dontknow" if row["says_dont_know"] else "other")
                tally[k] = tally.get(k, 0) + 1
            status = "complete (resumed)" if already >= n else " ".join(f"{k}={v}" for k, v in sorted(tally.items()))
            print(f"  [{seen}/{total}] {model} / {t['kind']} {t['name']} (rated P{t['prior_rating']})  {status}", flush=True)
    out.close()
    summarize_probe()

def summarize_probe():
    rows = [json.loads(l) for l in open(PROBE_RESULTS)]
    accepted_by_target = {(t["kind"], t["name"]): t["accepted"] for t in probe_targets()}
    by_target = {}
    for r in rows:
        k = (r["kind"], r["name"])
        accepted = accepted_by_target.get(k)
        hit = appears_any(accepted, r["answer"]) if accepted else bool(r["reports_expected"])
        by_target.setdefault(k, {"rating": r["prior_rating"], "knows": 0, "dontknow": 0, "n": 0})
        by_target[k]["n"] += 1
        by_target[k]["knows"] += hit
        by_target[k]["dontknow"] += r["says_dont_know"]
    current = set(accepted_by_target)
    print("\nDOC-FREE PRIOR PROBE -- measured knows-rate vs authored prior rating (lexical match on expected value)")
    print("  a high knows-rate on a low-rated target (or vice versa) means the authored rating is wrong")
    for (kind, name), v in sorted(by_target.items(), key=lambda kv: (kv[1]["rating"] is None, kv[1]["rating"], kv[0])):
        if (kind, name) not in current:
            continue
        rating = "--" if v["rating"] is None else f"P{v['rating']}"
        print(f"  {rating:>3}  {kind:<5} {name:<24} knows {v['knows']}/{v['n']}   dontknow {v['dontknow']}/{v['n']}")

def summarize_abstention():
    df = pd.read_json(ABSTENTION_RESULTS, lines=True)
    stats = df.groupby(["model", "instruction", "item_id"]).agg(
        tot=("label", "size"),
        ungrounded=("label", lambda s: (s == UNGROUNDED).sum()),
    ).to_dict("index")
    items = sorted({k[2] for k in stats})
    print("\nPARAMETRIC-LEAKAGE RATE by model x instruction (judge)")
    for model, _ in MODELS:
        for iname, _ in SYSTEM_INSTRUCTIONS:
            ks = [(model, iname, it) for it in items if (model, iname, it) in stats]
            if not ks:
                continue
            n = sum(stats[k]["tot"] for k in ks)
            x = sum(stats[k]["ungrounded"] for k in ks)
            p, lo, hi = wilson_interval(x, n)
            print(f"  {model:16} / {iname:30} {p:.2f} [{lo:.2f}, {hi:.2f}]  (n={n})")

SweepSpec = namedtuple("SweepSpec", ["name", "results", "done_fields", "plan", "dataset", "units", "warm",
                                     "encode", "decode", "prompt", "row", "wave_label", "cell_label", "summarize"])

def _caveat_unit_decode(cid):
    d = decode_caveat_custom_id(cid)
    return (d["fact"], d["severity"]), d["instruction"], d["rep"]

def _caveat_prompt(unit):
    fact, step = _caveat_step(*unit)
    return fact["q"], step_doc(fact, step)

def _caveat_spec_row(model, prov, iname, unit, doc, answer, snapshot, rep, truncated=False):
    fact, step = _caveat_step(*unit)
    return _caveat_row(model, prov, iname, fact, step, answer, snapshot, rep, truncated)

def _abstention_unit_decode(cid):
    d = decode_abstention_custom_id(cid)
    return (d["item_id"],), d["instruction"], d["rep"]

def _abstention_prompt(unit):
    p = ITEM_BY_ID[unit[0]]
    return p["q"], doc_text(p["doc"])

def _abstention_spec_row(model, prov, iname, unit, doc, answer, snapshot, rep, truncated=False):
    return _abstention_row(model, prov, iname, ITEM_BY_ID[unit[0]], answer, snapshot, rep, truncated)

def _absence_unit_decode(cid):
    d = decode_absence_custom_id(cid)
    return (d["fact"],), d["instruction"], d["rep"]

def _absence_prompt(unit):
    fact = FACT_BY_NAME[unit[0]]
    return fact["q"], absence_doc(fact)

def _absence_spec_row(model, prov, iname, unit, doc, answer, snapshot, rep, truncated=False):
    return _absence_row(model, prov, iname, FACT_BY_NAME[unit[0]], doc, answer, snapshot, rep, truncated)

CAVEAT_SWEEP = SweepSpec("caveat", CAVEAT_RESULTS, ["model", "instruction", "fact", "severity"], print_caveat_plan,
                         lambda: PERTURBATION_LADDERS,
                         lambda ds: [(f["fact"], s["severity"]) for f in ds for s in f["steps"]], "cell",
                         lambda u, i, r: encode_caveat_custom_id(u[0], u[1], i, r), _caveat_unit_decode,
                         _caveat_prompt, _caveat_spec_row,
                         lambda u: f"{u[0]} S{u[1]}", lambda u: f"{u[0]} S{u[1]}", summarize_caveat)

ABSTENTION_SWEEP = SweepSpec("abstention", ABSTENTION_RESULTS, ["model", "instruction", "item_id"],
                             print_abstention_plan, lambda: UNANSWERABLE_ITEMS,
                             lambda ds: [(p["item_id"],) for p in ds], "instruction",
                             lambda u, i, r: encode_abstention_custom_id(u[0], i, r), _abstention_unit_decode,
                             _abstention_prompt, _abstention_spec_row,
                             lambda u: u[0], lambda u: f"P{ITEM_BY_ID[u[0]]['prior_strength']} {u[0]}",
                             summarize_abstention)

ABSENCE_SWEEP = SweepSpec("absence", ABSENCE_RESULTS, ["model", "instruction", "fact"], print_absence_plan,
                          lambda: PERTURBATION_LADDERS,
                          lambda ds: [(f["fact"],) for f in ds], "cell",
                          lambda u, i, r: encode_absence_custom_id(u[0], i, r), _absence_unit_decode,
                          _absence_prompt, _absence_spec_row,
                          lambda u: f"{u[0]} absent", lambda u: f"{u[0]} absent", summarize_absence)

def _sweep_wave_plan(spec, done, n, model, instructions=None, dataset=None):
    instructions = instructions if instructions is not None else SYSTEM_INSTRUCTIONS
    dataset = dataset if dataset is not None else spec.dataset()
    units = spec.units(dataset)
    wave1, wave2 = [], []
    for iname, _ in instructions:
        if spec.warm == "instruction":
            pending = []
            for u in units:
                already = done.get((model, iname) + u, 0)
                for rep in range(already, n):
                    pending.append((u, rep))
            if not pending:
                continue
            warm_unit, warm_rep = pending[0]
            wave1.append(spec.encode(warm_unit, iname, warm_rep))
            for u, rep in pending[1:]:
                wave2.append(spec.encode(u, iname, rep))
        else:
            for u in units:
                already = done.get((model, iname) + u, 0)
                if already >= n:
                    continue
                reps = list(range(already, n))
                wave1.append(spec.encode(u, iname, reps[0]))
                for rep in reps[1:]:
                    wave2.append(spec.encode(u, iname, rep))
    return wave1, wave2

def _run_sweep_anthropic_batch(spec, model, prov, n, done, out, seen, total):
    wave1_ids, wave2_ids = _sweep_wave_plan(spec, done, n, model)
    cell_tally = {}

    def sync_fallback(cid):
        unit, iname, rep = spec.decode(cid)
        q, doc = spec.prompt(unit)
        return with_retry(call, model, prov, INSTR_BY_NAME[iname], q, doc)

    def batch_request(req_model, cid):
        unit, iname, rep = spec.decode(cid)
        q, doc = spec.prompt(unit)
        return build_batch_message_params(req_model, INSTR_BY_NAME[iname], q, doc)

    def process(custom_ids, wave_label):
        def judge_one(item):
            cid, answer, snapshot, truncated = item
            unit, iname, rep = spec.decode(cid)
            q, doc = spec.prompt(unit)
            return (unit, iname, rep), spec.row(model, prov, iname, unit, doc, answer, snapshot, rep, truncated)
        def write_row(res):
            (unit, iname, rep), row = res
            out.write(json.dumps(row) + "\n")
            out.flush()
            key = (iname,) + unit
            cell_tally.setdefault(key, {})
            cell_tally[key][row["label"]] = cell_tally[key].get(row["label"], 0) + 1
            print(f"    [{wave_label}] {model} / {iname} / {spec.wave_label(unit)} rep{rep} -> {row['label']}", flush=True)
        push, flush = _chunked_judge_sink(judge_one, write_row)
        _run_anthropic_wave(model, prov, custom_ids, wave_label, batch_request, sync_fallback, push)
        flush()

    process(wave1_ids, f"{spec.name} wave 1 (cache warm)")
    process(wave2_ids, f"{spec.name} wave 2 (cache read)")

    for iname, instr in SYSTEM_INSTRUCTIONS:
        for u in spec.units(spec.dataset()):
            seen += 1
            already = done.get((model, iname) + u, 0)
            if already >= n:
                status = "complete (resumed)"
            else:
                tally = cell_tally.get((iname,) + u, {})
                status = " ".join(f"{k}={v}" for k, v in sorted(tally.items()))
            print(f"  [{seen}/{total}] {model} / {iname} / {spec.cell_label(u)}  {status}", flush=True)
    return seen

OPENAI_BATCH_TOKEN_BUDGET = 1_000_000
OPENAI_BATCH_MAX_REQUESTS = 2000

def _openai_batch_chunks(requests):
    chunk, size = [], 0
    for cid, body in requests:
        t = (len(body.get("instructions", "")) + len(body.get("input", ""))) // 4 + body.get("max_output_tokens", 0)
        if chunk and (size + t > OPENAI_BATCH_TOKEN_BUDGET or len(chunk) >= OPENAI_BATCH_MAX_REQUESTS):
            yield chunk
            chunk, size = [], 0
        chunk.append((cid, body))
        size += t
    if chunk:
        yield chunk

def _run_openai_wave(model, prov, custom_ids, wave_label, build_request_fn, sync_call_fn, on_answer):
    if not custom_ids:
        return
    requests = [(cid, build_request_fn(model, cid)) for cid in custom_ids]
    chunks = list(_openai_batch_chunks(requests))
    for idx, chunk in enumerate(chunks, 1):
        label = wave_label if len(chunks) == 1 else f"{wave_label} chunk {idx}/{len(chunks)}"
        print(f"  submitting {label}: {len(chunk)} request(s)", flush=True)
        batch_id = submit_openai_batch(chunk)
        print(f"    batch id: {batch_id}", flush=True)

        def on_poll(batch, label=label, batch_id=batch_id):
            rc = batch.request_counts
            print(f"    {label} [{batch_id}] {batch.status}  "
                  f"completed={getattr(rc, 'completed', 0)} failed={getattr(rc, 'failed', 0)} "
                  f"total={getattr(rc, 'total', 0)}", flush=True)

        final = poll_openai_batch(batch_id, poll_interval=30, on_poll=on_poll)
        if final.status == "failed":
            errs = getattr(final, "errors", None)
            detail = "; ".join(e.message for e in errs.data) if errs and errs.data else "no error detail"
            raise SystemExit(f"{label}: batch {batch_id} failed wholesale, aborting instead of "
                             f"sync-retrying {len(chunk)} request(s) at full price -- {detail}")

        submitted = {cid for cid, _ in chunk}
        fallbacks, seen_ids = [], set()
        for cid, rec in openai_batch_results(batch_id):
            if cid not in submitted or cid in seen_ids:
                continue
            seen_ids.add(cid)
            resp = rec.get("response")
            if resp and resp.get("status_code") == 200 and rec.get("error") is None:
                body = resp["body"]
                on_answer(cid, extract_openai_text(body), body.get("model"), body.get("status") == "incomplete")
            else:
                print(f"    {label}: {cid} -> error; deferring to synchronous retry", flush=True)
                fallbacks.append(cid)
        for cid in submitted:
            if cid not in seen_ids:
                print(f"    {label}: {cid} missing from batch results; deferring to synchronous retry", flush=True)
                fallbacks.append(cid)
        for cid in fallbacks:
            answer, snapshot, truncated = sync_call_fn(cid)
            on_answer(cid, answer, snapshot, truncated)

def _run_sweep_openai_batch(spec, model, prov, n, done, out, seen, total):
    wave1_ids, wave2_ids = _sweep_wave_plan(spec, done, n, model)
    custom_ids = wave1_ids + wave2_ids
    cell_tally = {}

    def sync_fallback(cid):
        unit, iname, rep = spec.decode(cid)
        q, doc = spec.prompt(unit)
        return with_retry(call, model, prov, INSTR_BY_NAME[iname], q, doc)

    def batch_request(req_model, cid):
        unit, iname, rep = spec.decode(cid)
        q, doc = spec.prompt(unit)
        return build_openai_candidate_body(req_model, INSTR_BY_NAME[iname], q, doc)

    def judge_one(item):
        cid, answer, snapshot, truncated = item
        unit, iname, rep = spec.decode(cid)
        q, doc = spec.prompt(unit)
        return (unit, iname, rep), spec.row(model, prov, iname, unit, doc, answer, snapshot, rep, truncated)

    def write_row(res):
        (unit, iname, rep), row = res
        out.write(json.dumps(row) + "\n")
        out.flush()
        key = (iname,) + unit
        cell_tally.setdefault(key, {})
        cell_tally[key][row["label"]] = cell_tally[key].get(row["label"], 0) + 1
        print(f"    {model} / {iname} / {spec.wave_label(unit)} rep{rep} -> {row['label']}", flush=True)

    push, flush = _chunked_judge_sink(judge_one, write_row)
    _run_openai_wave(model, prov, custom_ids, f"{spec.name} batch", batch_request, sync_fallback, push)
    flush()

    for iname, instr in SYSTEM_INSTRUCTIONS:
        for u in spec.units(spec.dataset()):
            seen += 1
            already = done.get((model, iname) + u, 0)
            if already >= n:
                status = "complete (resumed)"
            else:
                tally = cell_tally.get((iname,) + u, {})
                status = " ".join(f"{k}={v}" for k, v in sorted(tally.items()))
            print(f"  [{seen}/{total}] {model} / {iname} / {spec.cell_label(u)}  {status}", flush=True)
    return seen

def _run_sweep(spec, n):
    if not spec.plan(n):
        sys.exit(1)
    done = load_done(spec.results, spec.done_fields)
    out = open(spec.results, "a")
    units = spec.units(spec.dataset())
    total = len(MODELS) * len(SYSTEM_INSTRUCTIONS) * len(units)
    seen = 0
    for model, prov in MODELS:
        if prov == "anthropic":
            seen = _run_sweep_anthropic_batch(spec, model, prov, n, done, out, seen, total)
        else:
            seen = _run_sweep_openai_batch(spec, model, prov, n, done, out, seen, total)
    out.close()
    spec.summarize()

def _load_jsonl(path, quiet=False):
    try:
        return [json.loads(l) for l in open(path)]
    except FileNotFoundError:
        if not quiet:
            print(f"  no {path} yet")
        return None

def cluster_icc(counts):
    m = len(counts)
    N = sum(n for _, n in counts)
    x = sum(xi for xi, _ in counts)
    p = x / N
    if m < 2 or x == 0 or x == N:
        return p, None, None
    msb = sum(n * (xi / n - p) ** 2 for xi, n in counts) / (m - 1)
    msw = sum(xi * (n - xi) / n for xi, n in counts) / (N - m)
    k0 = (N - sum(n * n for _, n in counts) / N) / (m - 1)
    denom = msb + (k0 - 1) * msw
    icc = 0.0 if denom <= 0 else max(0.0, min(1.0, (msb - msw) / denom))
    return p, icc, N / (1 + (N / m - 1) * icc)

def vector_cells(rows, unit_field, level_field, positive_label):
    cells = {}
    for r in rows:
        key = (r["model"], r["instruction"], r[level_field])
        per = cells.setdefault(key, {})
        xi, n = per.get(r[unit_field], (0, 0))
        per[r[unit_field]] = (xi + (r["label"] == positive_label), n + 1)
    return cells

def _print_vector_section(title, cells, level_prefix):
    print(title)
    for key in sorted(cells):
        model, iname, lv = key
        per = cells[key]
        p, icc, n_eff = cluster_icc(list(per.values()))
        vec = "  ".join(f"{u}:{xi}/{n}" for u, (xi, n) in sorted(per.items()))
        tail = "" if icc is None else f"   ICC {icc:.2f}  n_eff {n_eff:.1f}"
        label = f"{level_prefix}{lv}"
        print(f"  {model:<24} {iname:<30} {label}  rate {p:.2f}   {vec}{tail}")

def vectors():
    print("PER-UNIT VECTORS -- the fact/item, not the rep, is the experimental unit: reps within a unit are correlated")
    print("  ICC = within-unit correlation (ANOVA method-of-moments); n_eff = design-effect-adjusted sample size")
    print("  ICC is unidentifiable in all-zero/all-one cells; no ICC shown there")
    print()
    caveat_rows = _load_jsonl(CAVEAT_RESULTS)
    if caveat_rows:
        _print_vector_section("CAVEAT -- questioned x/n per fact, per model x instruction x severity",
                              vector_cells(caveat_rows, "fact", "severity", QUESTIONED), "S")
        print()
    abstention_rows = _load_jsonl(ABSTENTION_RESULTS)
    if abstention_rows:
        _print_vector_section("ABSTENTION -- faithful x/n per item, per model x instruction x authored prior strength",
                              vector_cells(abstention_rows, "item_id", "prior_strength", FAITHFUL), "P")
        print()
    absence_rows = _load_jsonl(ABSENCE_RESULTS)
    if absence_rows:
        _print_vector_section("MATCHED ABSENCE -- faithful x/n per fact, per model x instruction",
                              vector_cells(absence_rows, "fact", "evidence_state", FAITHFUL), "")

def matched_readout():
    caveat_rows, absence_rows = _load_jsonl(CAVEAT_RESULTS), _load_jsonl(ABSENCE_RESULTS)
    if not caveat_rows or not absence_rows:
        return
    print("MATCHED EVIDENCE-STATE READOUT -- same facts, three states; situated = per-fact majority on all three")
    cv, ab = pd.DataFrame(caveat_rows), pd.DataFrame(absence_rows)
    for (m, i), g in cv.groupby(["model", "instruction"]):
        s0 = g[g.severity == 0]
        pert = g[g.severity >= 1]
        absent = ab[(ab.model == m) & (ab.instruction == i)]
        if absent.empty:
            continue
        accept = (~s0.stance.isin([QUESTIONED, DECLINED])).mean()
        flag = (pert.stance == QUESTIONED).mean()
        abstain = absent.faithful.mean()
        strict = 0
        facts = sorted(set(absent.fact))
        for fname in facts:
            a_ok = (~s0[s0.fact == fname].stance.isin([QUESTIONED, DECLINED])).mean() > 0.5
            f_ok = (pert[(pert.fact == fname) & (pert.severity >= 3)].stance == QUESTIONED).mean() > 0.5
            b_ok = absent[absent.fact == fname].faithful.mean() > 0.5
            strict += a_ok and f_ok and b_ok
        print(f"  {m:16} / {i:30} accept_S0={accept:.2f} flag_perturbed={flag:.2f} abstain_absent={abstain:.2f}  situated {strict}/{len(facts)}")

FACTORIAL_ARMS = ["WEAK_GROUNDING", "FLAG_INVITING", "SOURCE_EXCLUSIVE", "SOURCE_EXCLUSIVE_FLAG_INVITING"]
ANALYSIS_SEED = 20260711

def sign_test(diffs):
    nonzero = [d for d in diffs if abs(d) > 1e-12]
    if not nonzero:
        return 1.0, 0, 0
    pos = sum(d > 0 for d in nonzero)
    n = len(nonzero)
    p = sum(math.comb(n, k) for k in range(min(pos, n - pos) + 1)) / 2 ** n * 2
    return min(1.0, p), pos, n

def bootstrap_ci(values, iters=10000, seed=ANALYSIS_SEED):
    rng = random.Random(seed)
    means = sorted(sum(values[rng.randrange(len(values))] for _ in values) / len(values) for _ in range(iters))
    return means[int(0.025 * iters)], means[int(0.975 * iters)]

def unit_counts(rows, pred, unit_field="fact"):
    per = {}
    for r in rows:
        x, n = per.get(r[unit_field], (0, 0))
        per[r[unit_field]] = (x + bool(pred(r)), n + 1)
    return per

def unit_rate_map(rows, pred, units, unit_field="fact"):
    per = unit_counts(rows, pred, unit_field)
    return {u: (per[u][0] / per[u][1] if u in per and per[u][1] else None) for u in units}

def factorial_effects(arm_rates, units):
    usable = [u for u in units if all(arm_rates[a][u] is not None for a in FACTORIAL_ARMS)]
    effects = {"SE_main": [], "FI_main": [], "interaction": []}
    for u in usable:
        wg, fi = arm_rates["WEAK_GROUNDING"][u], arm_rates["FLAG_INVITING"][u]
        se, sefi = arm_rates["SOURCE_EXCLUSIVE"][u], arm_rates["SOURCE_EXCLUSIVE_FLAG_INVITING"][u]
        effects["SE_main"].append(((se + sefi) - (fi + wg)) / 2)
        effects["FI_main"].append(((fi + sefi) - (se + wg)) / 2)
        effects["interaction"].append(sefi - se - fi + wg)
    return effects, usable

def _rate(rows, pred):
    x = sum(1 for r in rows if pred(r))
    return x, len(rows), (x / len(rows) if rows else float("nan"))

def _bracket(rows, pred, unit_field="fact"):
    x, n, p = _rate(rows, pred)
    if n == 0:
        return "n=0"
    _, lo, hi = wilson_interval(x, n)
    per = unit_counts(rows, pred, unit_field)
    _, icc, n_eff = cluster_icc(list(per.values()))
    if icc is None:
        m = len(per)
        _, clo, chi = wilson_interval(round(p * m), m)
        return f"{p:.3f} [{lo:.3f},{hi:.3f}] (n={n}); cluster [{clo:.3f},{chi:.3f}] (m={m}, ICC n/a degenerate)"
    _, clo, chi = wilson_interval(p * n_eff, n_eff)
    return f"{p:.3f} [{lo:.3f},{hi:.3f}] (n={n}); cluster [{clo:.3f},{chi:.3f}] (n_eff={n_eff:.1f}, ICC={icc:.2f})"

def _situated_faithfulness(cav_rows, ab_rows, model, iname):
    s0 = [r for r in cav_rows if r["model"] == model and r["instruction"] == iname and r["severity"] == 0]
    pert = [r for r in cav_rows if r["model"] == model and r["instruction"] == iname and r["severity"] >= 1]
    absent = [r for r in ab_rows if r["model"] == model and r["instruction"] == iname]
    ok, facts = 0, sorted(set(r["fact"] for r in absent))
    for f in facts:
        a = [r for r in s0 if r["fact"] == f]
        p3 = [r for r in pert if r["fact"] == f and r["severity"] >= 3]
        b = [r for r in absent if r["fact"] == f]
        a_ok = a and sum(r["stance"] not in (QUESTIONED, DECLINED) for r in a) / len(a) > 0.5
        f_ok = p3 and sum(r["stance"] == QUESTIONED for r in p3) / len(p3) > 0.5
        b_ok = b and sum(bool(r["faithful"]) for r in b) / len(b) > 0.5
        ok += bool(a_ok and f_ok and b_ok)
    return ok, len(facts)

def _print_effects(title, effects, usable):
    print(f"  {title}")
    for name, vals in effects.items():
        mean = sum(vals) / len(vals)
        lo, hi = bootstrap_ci(vals)
        p, pos, n = sign_test(vals)
        print(f"    {name:<12} {mean:+.3f}  boot95[{lo:+.3f},{hi:+.3f}]  sign-test p={p:.4f} ({pos}/{n} facts +, {len(usable)} usable)")

def _analysis_dataset(tag, cav_rows, ab_rows, models, facts, fact_doc):
    is_q = lambda r: r["stance"] == QUESTIONED
    is_e = lambda r: r["stance"] == ENDORSED
    is_fc = lambda r: r["stance"] == ENDORSED and r["corroboration"] in ("generic", "named_authority")
    accepts = lambda r: r["stance"] not in (QUESTIONED, DECLINED)
    is_f = lambda r: bool(r["faithful"])
    arms = [n for n, _ in SYSTEM_INSTRUCTIONS]
    print("=" * 100)
    print(f"DATASET: {tag}  (caveat rows={len(cav_rows)}, absence rows={len(ab_rows)})")
    print("=" * 100)
    print("\n--- PRIMARY OUTCOMES 1-6, per model x instruction ---")
    print("O1 error flagging (questioned | S1-5); O2 clean specificity (1 - questioned-or-declined | S0)")
    print("O3 absence faithfulness; O4 false endorsement (endorsed | S1-5); O5 false corroboration (endorsed & generic/named | S1-5)")
    for m in models:
        for i in arms:
            g = [r for r in cav_rows if r["model"] == m and r["instruction"] == i]
            if not g:
                continue
            pert = [r for r in g if r["severity"] >= 1]
            s0 = [r for r in g if r["severity"] == 0]
            absn = [r for r in ab_rows if r["model"] == m and r["instruction"] == i]
            k, nf = _situated_faithfulness(cav_rows, ab_rows, m, i)
            print(f"\n  {m} / {i}")
            print(f"    O1 flag_perturbed   {_bracket(pert, is_q)}")
            print(f"    O2 clean_specific   {_bracket(s0, accepts)}")
            if absn:
                print(f"    O3 abstain_absent   {_bracket(absn, is_f)}")
            print(f"    O4 false_endorse    {_bracket(pert, is_e)}")
            print(f"    O5 false_corrob     {_bracket(pert, is_fc)}")
            print(f"    O6 situated         {k}/{nf}")
    print("\n--- 2x2 FACTORIAL (fact-level, paired; effects on outcome rates in [0,1]) ---")
    for m in models:
        print(f"\n {m}")
        for title, rows, pred in (("O1 error flagging", cav_rows, is_q),
                                  ("O3 absence faithfulness", ab_rows, is_f),
                                  ("O4 false endorsement", cav_rows, is_e)):
            arm_rates = {a: unit_rate_map([r for r in rows if r["model"] == m and r["instruction"] == a
                                           and (rows is ab_rows or r["severity"] >= 1)], pred, facts)
                         for a in FACTORIAL_ARMS}
            _print_effects(title, *factorial_effects(arm_rates, facts))
    print("\n--- SELECTIVE_AUDIT EXISTENCE TEST (vs best 2x2 cell per model) ---")
    for m in models:
        best, bk, bn = None, -1, 0
        for i in FACTORIAL_ARMS:
            k, nf = _situated_faithfulness(cav_rows, ab_rows, m, i)
            if k > bk:
                best, bk, bn = i, k, nf
        ka, na = _situated_faithfulness(cav_rows, ab_rows, m, "SELECTIVE_AUDIT")
        cells = {}
        for i in (best, "SELECTIVE_AUDIT"):
            p = [r for r in cav_rows if r["model"] == m and r["instruction"] == i and r["severity"] >= 1]
            a = [r for r in ab_rows if r["model"] == m and r["instruction"] == i]
            cells[i] = (_rate(p, is_q)[2], _rate(a, is_f)[2])
        print(f"  {m}: best 2x2 = {best} situated {bk}/{bn} (O1 {cells[best][0]:.2f}, O3 {cells[best][1]:.2f})"
              f"  |  SELECTIVE_AUDIT {ka}/{na} (O1 {cells['SELECTIVE_AUDIT'][0]:.2f}, O3 {cells['SELECTIVE_AUDIT'][1]:.2f})")
    for metric, pred in (("QUESTIONED", is_q), ("ENDORSED", is_e)):
        print(f"\n--- SEVERITY CONTRASTS ({metric}): rate at S0 / S1 / S2 / S3 / S4 / S5 ---")
        for i in arms:
            for m in models:
                g = [r for r in cav_rows if r["model"] == m and r["instruction"] == i]
                if not g:
                    continue
                cells = []
                for s in SEVERITIES:
                    b = [r for r in g if r["severity"] == s]
                    cells.append(f"{_rate(b, pred)[2]:.2f}")
                print(f"  {i:32} {m:16} " + " / ".join(cells))
    print("\n--- PER-DOCUMENT: O1 (S1-5 questioned) and O3 (absence faithful) per doc, instruction x model ---")
    for i in arms:
        for m in models:
            parts = []
            for d in DOCUMENTS:
                p = [r for r in cav_rows if r["model"] == m and r["instruction"] == i and r["severity"] >= 1 and fact_doc[r["fact"]] == d]
                a = [r for r in ab_rows if r["model"] == m and r["instruction"] == i and fact_doc[r["fact"]] == d]
                if p or a:
                    o3 = f"{_rate(a, is_f)[2]:.2f}" if a else "--"
                    parts.append(f"{d}: O1 {_rate(p, is_q)[2]:.2f}({len(p)}) O3 {o3}({len(a)})")
            if parts:
                print(f"  {i:32} {m:16} " + "  ".join(parts))

def analysis():
    cav_all = [json.loads(l) for l in open(CAVEAT_RESULTS)]
    ab_all = [json.loads(l) for l in open(ABSENCE_RESULTS)]
    n_flagless = sum(1 for r in cav_all + ab_all if "truncated" not in r)
    cav = [r for r in cav_all if not r.get("truncated")]
    ab = [r for r in ab_all if not r.get("truncated")]
    n_excluded = len(cav_all) - len(cav) + len(ab_all) - len(ab)
    fact_doc = {f["fact"]: f["doc"] for f in PERTURBATION_LADDERS}
    facts = sorted(fact_doc)
    present = set(r["model"] for r in cav)
    models = [m for m, _ in MODELS if m in present]
    seeded = [r for r in cav if "seeded_from" in r]
    fresh = [r for r in cav if "seeded_from" not in r]
    no_prov = [r for r in fresh if "ts" not in r]
    print("SECTION 8 PRE-REGISTERED ANALYSIS -- run against the current result files")
    judges = sorted(set(r.get("judge_snapshot", "unrecorded") for r in cav + ab))
    print(f"judge snapshots in files: {judges}")
    print(f"caveat rows {len(cav)} (seeded {len(seeded)}); absence rows {len(ab)}")
    print(f"fresh rows without ts provenance: {len(no_prov)}")
    print(f"truncation exclusions: {n_excluded} applied via the truncated flag; "
          f"{n_flagless} rows predate the flag and are retained")
    _analysis_dataset("POOLED (fresh + seeded)", cav, ab, models, facts, fact_doc)
    _analysis_dataset("FRESH ONLY (seeded v1 rows excluded -- sensitivity)", fresh, ab, models, facts, fact_doc)
    if seeded:
        print("\n--- SEEDED vs FRESH side-by-side (models with seeded cells; perturbed severities S1-5) ---")
        is_q = lambda r: r["stance"] == QUESTIONED
        is_e = lambda r: r["stance"] == ENDORSED
        for m in models:
            for i in [n for n, _ in SYSTEM_INSTRUCTIONS]:
                sd = [r for r in seeded if r["model"] == m and r["instruction"] == i and r["severity"] >= 1]
                fr = [r for r in fresh if r["model"] == m and r["instruction"] == i and r["severity"] >= 1]
                if sd:
                    print(f"  {m:14} {i:32} seeded Q {_rate(sd, is_q)[2]:.3f} E {_rate(sd, is_e)[2]:.3f} (n={len(sd)})"
                          f"   fresh Q {_rate(fr, is_q)[2]:.3f} E {_rate(fr, is_e)[2]:.3f} (n={len(fr)})")
    _mechanism_split(cav, models)
    try:
        opus = [r for r in (json.loads(l) for l in open(OPUS_FI_PROBE_RESULTS)) if not r.get("truncated")]
    except FileNotFoundError:
        opus = []
    if opus:
        opus_probe_readout(opus)
    sdt_models = models + ([OPUS_PROBE_MODEL[0]] if opus else [])
    endorsement_sdt(cav + opus, sdt_models)
    detection_thresholds(cav + opus, sdt_models)

def opus_probe_readout(opus):
    print(f"\n--- OPUS 4.8 FI PROBE ({OPUS_PROBE_MODEL[0]}, FLAG_INVITING only, N=1) ---")
    pert = [r for r in opus if r["severity"] >= 1]
    s0 = [r for r in opus if r["severity"] == 0]
    for label, rows, pred in [
            ("S1-5 flag", pert, lambda r: r["stance"] == QUESTIONED),
            ("S0 clean specificity", s0, lambda r: r["stance"] not in (QUESTIONED, DECLINED)),
            ("S1-5 endorsement", pert, lambda r: r["stance"] == ENDORSED),
            ("S1-5 corroborated endorsement", pert,
             lambda r: r["stance"] == ENDORSED and r["corroboration"] != NO_CORROBORATION)]:
        x, n, _ = _rate(rows, pred)
        p, lo, hi = wilson_interval(x, n)
        print(f"  {label:30} {p:.2f} [{lo:.2f},{hi:.2f}] ({x}/{n})")
    for c in (GENERIC_STD, NAMED_AUTHORITY):
        x = sum(1 for r in pert if r["stance"] == ENDORSED and r["corroboration"] == c)
        print(f"    corroboration {c}: {x}")
    print("  questioned by severity: " + "  ".join(
        f"S{s} {_rate([r for r in opus if r['severity'] == s], lambda r: r['stance'] == QUESTIONED)[2]:.2f}"
        for s in SEVERITIES))
    print("  per-document flag: " + "  ".join(
        f"{d} {_rate([r for r in pert if r['document'] == d], lambda r: r['stance'] == QUESTIONED)[2]:.2f}"
        for d in sorted(set(r["document"] for r in pert))))

def _mechanism_split(cav, models):
    print("\n--- DETECTION MECHANISM: internal-anchor split (questioned rate on perturbed rows) ---")
    leftovers = sorted(f["fact"] for f in PERTURBATION_LADDERS if surviving_absence_sites(f))
    print(f"  facts leaving an answer-bearing site intact under perturbation: {leftovers or 'none'}")
    print(f"  hand-classified internally anchored: {sorted(INTERNALLY_ANCHORED_FACTS)}")
    worst = 0.0
    for m in models:
        for i in [n for n, _ in SYSTEM_INSTRUCTIONS]:
            rows = [r for r in cav if r["model"] == m and r["instruction"] == i and r["severity"] >= 1]
            clean = [r for r in rows if r["fact"] not in INTERNALLY_ANCHORED_FACTS]
            if not rows or not clean:
                continue
            a = sum(r["stance"] == QUESTIONED for r in rows) / len(rows)
            b = sum(r["stance"] == QUESTIONED for r in clean) / len(clean)
            worst = max(worst, abs(a - b))
            if abs(a - b) >= 0.005:
                print(f"  {m:16} {i:32} all {a:.3f}  excl-anchored {b:.3f}  delta {a - b:+.3f}")
    print(f"  max |delta| across model x instruction: {worst:.3f}")

SDT_BOOTSTRAP_ITERS = 10000
THRESHOLD_BOOTSTRAP_ITERS = 2000

def _corrected_rate(x, n):
    return (x + 0.5) / (n + 1)

def _zscore(p):
    return statistics.NormalDist().inv_cdf(p)

def _endorse_counts(rows):
    counts = {}
    for r in rows:
        d = counts.setdefault(r["fact"], {})
        x, n = d.get(r["severity"], (0, 0))
        d[r["severity"]] = (x + (r["stance"] == ENDORSED), n + 1)
    return counts

def _pooled_dprime(counts, facts, severity):
    x0 = n0 = xs = ns = 0
    for f in facts:
        a, b = counts[f].get(0, (0, 0))
        x0 += a; n0 += b
        a, b = counts[f].get(severity, (0, 0))
        xs += a; ns += b
    if n0 == 0 or ns == 0:
        return None
    return _zscore(_corrected_rate(x0, n0)) - _zscore(_corrected_rate(xs, ns))

def _dprime_ci(counts, severity, iters=SDT_BOOTSTRAP_ITERS, seed=ANALYSIS_SEED):
    facts = sorted(counts)
    rng = random.Random(seed)
    draws = []
    for _ in range(iters):
        sample = [facts[rng.randrange(len(facts))] for _ in facts]
        d = _pooled_dprime(counts, sample, severity)
        if d is not None:
            draws.append(d)
    draws.sort()
    return draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws))]

def endorsement_sdt(cav, models):
    print("\n--- endorsement propensity vs discrimination ---")
    print(f"  rates corrected (x+0.5)/(n+1) before z; d'(s) = z(E|S0) - z(E|Ss); "
          f"[] = 95% cluster bootstrap over facts ({SDT_BOOTSTRAP_ITERS} resamples, fixed seed)")
    for m in models:
        for i in [n for n, _ in SYSTEM_INSTRUCTIONS]:
            rows = [r for r in cav if r["model"] == m and r["instruction"] == i]
            if not rows:
                continue
            endorsed = sum(r["stance"] == ENDORSED for r in rows)
            if endorsed == 0:
                continue
            counts = _endorse_counts(rows)
            facts = sorted(counts)
            x0 = sum(counts[f].get(0, (0, 0))[0] for f in facts)
            n0 = sum(counts[f].get(0, (0, 0))[1] for f in facts)
            print(f"  {m:16} {i:32} E|S0 {x0 / n0:.2f} ({x0}/{n0})")
            for s in SEVERITIES[1:]:
                xs = sum(counts[f].get(s, (0, 0))[0] for f in facts)
                ns = sum(counts[f].get(s, (0, 0))[1] for f in facts)
                d = _pooled_dprime(counts, facts, s)
                lo, hi = _dprime_ci(counts, s)
                print(f"      S{s}  E {xs / ns:.2f} ({xs}/{ns})   d' {d:+.2f} [{lo:+.2f},{hi:+.2f}]")
    zero = [(m, i) for m in models for i in [n for n, _ in SYSTEM_INSTRUCTIONS]
            if any(r["model"] == m and r["instruction"] == i for r in cav)
            and not any(r["model"] == m and r["instruction"] == i and r["stance"] == ENDORSED for r in cav)]
    tested = sum(1 for m in models for i in [n for n, _ in SYSTEM_INSTRUCTIONS]
                 if any(r["model"] == m and r["instruction"] == i for r in cav))
    print(f"  cells with zero endorsements anywhere (d' n/a, zero propensity): {len(zero)} of {tested}")
    for m in models:
        insts = [i for mm, i in zero if mm == m]
        if len(insts) == len(SYSTEM_INSTRUCTIONS):
            print(f"    {m}: all instructions")
        elif insts:
            print(f"    {m}: {', '.join(insts)}")

def _sigmoid(z):
    return 1 / (1 + math.exp(-max(-35.0, min(35.0, z))))

def logistic_fit(points, ridge=1e-6, iters=50):
    b0 = b1 = 0.0
    for _ in range(iters):
        g0 = g1 = h00 = h01 = h11 = 0.0
        for x, y in points:
            p = _sigmoid(b0 + b1 * x)
            g0 += y - p
            g1 += (y - p) * x
            w = max(p * (1 - p), 1e-12)
            h00 += w; h01 += w * x; h11 += w * x * x
        g0 -= ridge * b0
        g1 -= ridge * b1
        h00 += ridge; h11 += ridge
        det = h00 * h11 - h01 * h01
        if det <= 0:
            break
        db0 = (h11 * g0 - h01 * g1) / det
        db1 = (h00 * g1 - h01 * g0) / det
        b0 += db0; b1 += db1
        if abs(db0) < 1e-9 and abs(db1) < 1e-9:
            break
    return b0, b1

def ratio50(points, max_log_ratio):
    b0, b1 = logistic_fit(points)
    if b1 <= 0:
        return None
    x50 = -b0 / b1
    if not 0 <= x50 <= max_log_ratio:
        return None
    return 10 ** x50

def _threshold_points(rows):
    ratio_by_step = {(f["fact"], s["severity"]): s["ratio"] for f in PERTURBATION_LADDERS for s in f["steps"]}
    per_fact = {}
    for r in rows:
        ratio = ratio_by_step.get((r["fact"], r["severity"]))
        if ratio is None:
            continue
        per_fact.setdefault(r["fact"], []).append((math.log10(ratio), r["stance"] == QUESTIONED))
    return per_fact

def detection_thresholds(cav, models):
    print("\n--- detection threshold in ratio units ---")
    bounded = sorted(f["fact"] for f in PERTURBATION_LADDERS if all(s["ratio"] is None for s in f["steps"]))
    print("  logistic fit of questioned rate on log10(perturbation ratio), S0 (ratio 1) included as the false-alarm anchor;")
    print("  ratio50 = ratio at 50% flagging, reported only when the fitted curve crosses 0.5 inside the observed range;")
    print(f"  [] = 95% cluster bootstrap over facts ({THRESHOLD_BOOTSTRAP_ITERS} resamples, fixed seed), on resamples that cross;")
    print(f"  ratio-less facts excluded: {bounded or 'none'}")
    for m in models:
        for i in [n for n, _ in SYSTEM_INSTRUCTIONS]:
            rows = [r for r in cav if r["model"] == m and r["instruction"] == i]
            if not rows:
                continue
            per_fact = _threshold_points(rows)
            facts = sorted(per_fact)
            points = [p for f in facts for p in per_fact[f]]
            max_log = max(x for x, _ in points)
            est = ratio50(points, max_log)
            if est is None:
                print(f"  {m:16} {i:32} no 50% crossing <= x{10 ** max_log:g}")
                continue
            rng = random.Random(ANALYSIS_SEED)
            crossings = []
            misses = 0
            for _ in range(THRESHOLD_BOOTSTRAP_ITERS):
                sample = [facts[rng.randrange(len(facts))] for _ in facts]
                bpts = [p for f in sample for p in per_fact[f]]
                b = ratio50(bpts, max_log)
                if b is None:
                    misses += 1
                else:
                    crossings.append(math.log10(b))
            crossings.sort()
            lo = 10 ** crossings[int(0.025 * len(crossings))]
            hi = 10 ** crossings[int(0.975 * len(crossings))]
            miss_note = f", {100 * misses / THRESHOLD_BOOTSTRAP_ITERS:.1f}% of resamples no crossing" if misses else ""
            print(f"  {m:16} {i:32} ratio50 x{est:.1f} [x{lo:.1f},x{hi:.1f}]{miss_note}")

MANIFEST_FILE = "data/run_manifest.json"

def _sha256(text):
    return hashlib.sha256(text.encode()).hexdigest()

def build_manifest():
    git_sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    absence_facts = [f for f in PERTURBATION_LADDERS if "absence" in f]
    return json.loads(json.dumps({
        "generated_at": utc_now(),
        "git_sha": git_sha,
        "run_id": RUN_ID,
        "documents": {name: _sha256(DOCUMENT_TEXTS[name]) for name in DOCUMENTS},
        "absence_documents": {f["fact"]: _sha256(step_doc(f, f["absence"])) for f in absence_facts},
        "instructions": [{"name": n, "text": t, "sha256": _sha256(t)} for n, t in SYSTEM_INSTRUCTIONS],
        "models": MODELS,
        "judge": {"model": JUDGE_MODEL, "caveat_system_sha256": _sha256(CAVEAT_SYSTEM),
                  "abstention_system_sha256": _sha256(ABSTENTION_SYSTEM)},
        "n_per_cell": N_PER_CELL,
        "judge_concurrency": JUDGE_CONCURRENCY,
        "candidate_params": {"anthropic_max_tokens": 1200, "openai_max_output_tokens": 2000,
                             "gpt54_reasoning_effort": "low", "temperature": "API default"},
        "expected_cells": {"caveat": len(MODELS) * len(SYSTEM_INSTRUCTIONS) * total_steps(),
                           "abstention": len(MODELS) * len(SYSTEM_INSTRUCTIONS) * len(UNANSWERABLE_ITEMS),
                           "absence": len(MODELS) * len(SYSTEM_INSTRUCTIONS) * len(absence_facts)},
        "probes": {"opus_fi": {"model": OPUS_PROBE_MODEL[0], "provider": OPUS_PROBE_MODEL[1],
                               "instruction": "FLAG_INVITING", "n_per_cell": 1,
                               "expected_rows": total_steps(), "results_file": OPUS_FI_PROBE_RESULTS,
                               "anthropic_thinking": "adaptive, explicitly enabled",
                               "candidate_params_otherwise": "match the grid",
                               "run_date": "2026-07-17",
                               "provenance": "probe ran after the grid manifest was generated; entry added 2026-07-17"}},
        "perturbation_ladders": PERTURBATION_LADDERS,
        "unanswerable_items": UNANSWERABLE_ITEMS,
    }))

def write_manifest():
    m = build_manifest()
    with open(MANIFEST_FILE, "w") as f:
        json.dump(m, f, indent=2)
    e = m["expected_cells"]
    print(f"{MANIFEST_FILE}: git {m['git_sha'][:12]} run {m['run_id']} -- "
          f"{len(m['perturbation_ladders'])} facts / {len(m['unanswerable_items'])} items / "
          f"{len(m['instructions'])} instructions; cells caveat={e['caveat']} abstention={e['abstention']} absence={e['absence']}")

OPUS_FI_PROBE_RESULTS = "data/opus_fi_probe.jsonl"
OPUS_PROBE_MODEL = ("claude-opus-4-8", "anthropic")

def run_opus_fi_probe(n):
    global CAVEAT_RESULTS
    MODELS[:] = [OPUS_PROBE_MODEL]
    SYSTEM_INSTRUCTIONS[:] = [(name, text) for name, text in SYSTEM_INSTRUCTIONS if name == "FLAG_INVITING"]
    CAVEAT_RESULTS = OPUS_FI_PROBE_RESULTS
    print(f"OPUS FI PROBE: {OPUS_PROBE_MODEL[0]} x FLAG_INVITING only, N={n}, "
          f"adaptive thinking explicit, results -> {OPUS_FI_PROBE_RESULTS}\n")
    _run_sweep(CAVEAT_SWEEP._replace(results=OPUS_FI_PROBE_RESULTS), n)

def pilot_selection(model_name, doc):
    models = [(m, p) for m, p in MODELS if m == model_name]
    if not models:
        raise SystemExit(f"unknown model {model_name!r} -- roster: {[m for m, _ in MODELS]}")
    if doc not in DOCUMENTS:
        raise SystemExit(f"unknown document {doc!r} -- registry: {list(DOCUMENTS)}")
    ladders = [f for f in PERTURBATION_LADDERS if f["doc"] == doc]
    items = [p for p in UNANSWERABLE_ITEMS if p["doc"] == doc]
    if not ladders and not items:
        raise SystemExit(f"document {doc!r} has no facts and no items")
    return models, ladders, items

def run_pilot(model_name, doc, n):
    models, ladders, items = pilot_selection(model_name, doc)
    MODELS[:] = models
    PERTURBATION_LADDERS[:] = ladders
    UNANSWERABLE_ITEMS[:] = items
    print(f"PILOT: {model_name} x {doc} -- {len(ladders)} facts, {len(items)} items, N={n}\n")
    run_caveat(n)
    print()
    run_abstention(n)
    print()
    run_absence(n)

if __name__ == "__main__": # only run file if executed directly
    args = sys.argv[1:]
    if args and args[0] == "caveat": # if args and args[0] = if the first argument is caveat
        run_caveat(int(args[1]) if len(args) > 1 else N_PER_CELL)
    elif args and args[0] == "abstention":
        run_abstention(int(args[1]) if len(args) > 1 else N_PER_CELL)
    elif args and args[0] == "absence":
        run_absence(int(args[1]) if len(args) > 1 else N_PER_CELL)
    elif args and args[0] == "vectors":
        vectors()
    elif args and args[0] == "probe":
        run_probe(int(args[1]) if len(args) > 1 else N_PER_CELL)
    elif args and args[0] == "opusprobe":
        run_opus_fi_probe(int(args[1]) if len(args) > 1 else 1)
    elif args and args[0] == "rescore":
        if args[1:2] == ["abstention"]:
            rescore_abstention()
        elif args[1:2] == ["absence"]:
            rescore_absence()
        else:
            rescore_caveat(args[1:] or None)
    elif args and args[0] == "endorsement":
        endorsement_breakdown()
    elif args and args[0] == "adoption":
        adoption_readout()
    elif args and args[0] == "manifest":
        write_manifest()
    elif args and args[0] == "matched":
        matched_readout()
    elif args and args[0] == "analysis":
        analysis()
    elif args and args[0] == "pilot":
        if len(args) < 3:
            print("usage: python3 harness.py pilot <model> <doc> [N]")
            sys.exit(1)
        run_pilot(args[1], args[2], int(args[3]) if len(args) > 3 else N_PER_CELL)
    elif args and not args[0].isdigit():
        print("usage: python3 harness.py [N] | caveat [N] | abstention [N] | absence [N] | probe [N] | rescore | endorsement | adoption | vectors | matched | analysis | manifest | pilot <model> <doc> [N]")
        sys.exit(1)
    else:
        n = int(args[0]) if args else N_PER_CELL
        print_caveat_plan(n)
        print()
        print_abstention_plan(n)
        print()
        print_absence_plan(n)
        print("\n  (No API calls were made. To execute: python3 harness.py caveat [N] | abstention [N] | absence [N])")
