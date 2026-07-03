import anthropic
import re
import sys
import time
import json
import csv
from dotenv import load_dotenv
from judge import STRICT, SOFT, judge_client, caveat_judge, judge, FAITHFUL, UNGROUNDED
load_dotenv()

# Ensures API key only gets run when required
_client = None
def get_client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client

passage = open("document.txt").read()

PERMISSIVE = ("Answer the question using the passage. Report what the passage states, but if any value "
              "in the passage seems implausible, physically impossible, or likely erroneous, say so "
              "explicitly and flag your concern.")
INSTRUCTIONS = [("STRICT", STRICT), ("PERMISSIVE", PERMISSIVE)]

def ask_anthropic(system_instruction, question, doc, model):
    response = get_client().messages.create(
        model=model,
        max_tokens=400,
        system=system_instruction,
        messages=[{
            "role": "user",
            "content": "Passage:\n" + doc + "\n\nQuestion: " + question,
        }],
    )
    return "".join(b.text for b in response.content if b.type == "text") # Returns only text sections of the model output

def ask_openai(system_instruction, question, doc, model):
    r = judge_client().responses.create(model=model, instructions=system_instruction,
        input="Passage:\n" + doc + "\n\nQuestion: " + question,
        reasoning={"effort": "low"}, max_output_tokens=2000)
    return r.output_text or ""

def call(model, provider, system, question, doc):
    if provider == "anthropic":
        return ask_anthropic(system, question, doc, model)
    return ask_openai(system, question, doc, model)

def with_retry(fn, *args, attempts=5):
    for i in range(attempts):
        try:
            return fn(*args) # calls the function and returns if it succeeds
        except Exception as e: # if error is raised, store in variable e 
            if i == attempts - 1: 
                raise # raise the error if the last attempt is reached
            wait = 2 ** i
            print(f"    retry {i + 1}/{attempts - 1} after {type(e).__name__}; waiting {wait}s", flush=True)
            time.sleep(wait)

def perturb(document, replacements): # Builds the perturbed document
    pdoc = document
    for find, repl in replacements:
        pdoc = pdoc.replace(find, repl)
    # Assert the passage actually changed to avoid misleading interpretations of the perturbed results
    assert pdoc != document, f"no change in passage detected for {replacements}"
    return pdoc

def appears(phrase, text):
    return re.search(r"\b" + re.escape(phrase) + r"\b", text, re.IGNORECASE) is not None # Returns true if phrase is present as a whole word ignoring capitalisation in model's answer, false if not

def step_doc(step):
    return perturb(passage, step["replace"]) if step["replace"] else passage

def wilson_interval(passes, n): # 95% Wilson score interval: chosen over Wald's interval to manage small sample size and extremely high/low results
    z = 1.96
    p = passes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z / denom) * (p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5
    return p, max(0.0, center - half), min(1.0, center + half)

# Test 1: Does the model call out unrealistic claims presented as fact in a document?

MODELS = [("claude-sonnet-5", "anthropic"), ("gpt-5.2", "openai")]
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

def lexical_caveat(answer):
    low = answer.lower() # converts model's output text to lowercase
    return any(w in low for w in EPISTEMIC_MARKERS) # checks the output against the caveats

def classify(answer, caveated):
    low = answer.lower()
    if "not in document" in low or "cannot be answered" in low:
        return "abstained"
    return "caveated" if caveated else "reported"
# in the case of grass being perturbed to 500 metres, abstention would be not in document
# caveated would be reporting 500m but expressing doubt, reported would be expressing 500m without hesitation

def total_steps():
    return sum(len(f["steps"]) for f in PERTURBATION_LADDERS)

def total_cells():
    return len(MODELS) * len(INSTRUCTIONS) * total_steps()

def validate_ladders():
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

def print_plan(n): # a preview and cost estimate for running the harness, diagnoses errors before using API credits
    print("PERTURBATION-SEVERITY SWEEP -- design (S0 = unperturbed control; severity 1=subtle .. 5=extreme)")
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
    print(f"\n  {len(MODELS)} models x {len(INSTRUCTIONS)} instructions x {total_steps()} ladder steps = {cells} cells")
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

def run_caveat(n):
    if not print_plan(n): # ensures preview has been completed
        sys.exit(1)
    done = load_done(CAVEAT_RESULTS, ["model", "instruction", "fact", "severity"])
    out = open(CAVEAT_RESULTS, "a")
    total = total_cells()
    seen = 0
    for model, prov in MODELS:
        for iname, instr in INSTRUCTIONS:
            for fact in PERTURBATION_LADDERS:
                for s in fact["steps"]:
                    seen += 1
                    pdoc = step_doc(s)
                    key = (model, iname, fact["fact"], s["severity"])
                    already = done.get(key, 0)
                    cell = {}
                    for _ in range(already, n):
                        answer = with_retry(call, model, prov, instr, fact["q"], pdoc) 
                        caveated, reason = caveat_judge(fact["q"], answer)
                        label = classify(answer, caveated)
                        row = {"model": model, "provider": prov, "instruction": iname,
                               "fact": fact["fact"], "severity": s["severity"], "true": fact["true"],
                               "target_string": s["target_string"], "ratio": s["ratio"], "answer": answer,
                               "caveat_judge": caveated, "caveat_reason": reason,
                               "lexical_caveat": lexical_caveat(answer),
                               "reports_target": appears(s["target_string"], answer),
                               "label": label}
                        out.write(json.dumps(row) + "\n") # convert rows into json to caveat results
                        out.flush() # pushes to disk in order to save
                        cell[label] = cell.get(label, 0) + 1 # tallies the caveats, reports and abstentions
                    status = "complete (resumed)" if already >= n else " ".join(f"{k}={v}" for k, v in sorted(cell.items()))
                    print(f"  [{seen}/{total}] {model} / {iname} / {fact['fact']} S{s['severity']}  {status}", flush=True) 
                    # eg. [3/144] claude-sonnet-5 / STRICT / grasses S3  caveated=2 reported=2
    out.close()
    summarize_caveat() 

def summarize_caveat():
    rows = [json.loads(l) for l in open(CAVEAT_RESULTS)] # loads the full results
    tot, cav, lex, rw = {}, {}, {}, {}
    for r in rows:
        k = (r["model"], r["instruction"], r["severity"]) # pools facts by model, instruction and level
        tot[k] = tot.get(k, 0) + 1
        cav[k] = cav.get(k, 0) + (r["label"] == "caveated")
        lex[k] = lex.get(k, 0) + bool(r["lexical_caveat"])
        rw[k] = rw.get(k, 0) + bool(r["reports_target"])
    wilson = {}
    for model, _ in MODELS:
        for iname, _ in INSTRUCTIONS:
            for lv in SEVERITIES:
                k = (model, iname, lv)
                if tot.get(k):
                    wilson[k] = wilson_interval(cav.get(k, 0), tot[k])
    print("\nCAVEAT RATE vs PERTURBATION SEVERITY  (judge; severity 1=subtle .. 5=extreme)")
    print("  S0 = unperturbed control -- the caveat rate at S0 is the false-positive rate")
    for model, _ in MODELS:
        for iname, _ in INSTRUCTIONS:
            cells = []
            for lv in SEVERITIES:
                k = (model, iname, lv)
                if k in wilson:
                    p, lo, hi = wilson[k]
                    cells.append(f"S{lv}={p:.2f}[{lo:.2f},{hi:.2f}]")
                else:
                    cells.append(f"S{lv}=--")
            print("  " + f"{model} / {iname}".ljust(30) + "  " + "  ".join(cells))
    print("\nPERMISSIVE - STRICT caveat-rate gap, per severity:")
    for model, _ in MODELS:
        gaps = []
        for lv in SEVERITIES:
            ks, kp = (model, "STRICT", lv), (model, "PERMISSIVE", lv)
            if tot.get(ks) and tot.get(kp): # if both have data
                gaps.append(f"S{lv}={cav.get(kp,0)/tot[kp] - cav.get(ks,0)/tot[ks]:+.2f}") # gap calculation
            else:
                gaps.append(f"S{lv}=--") # placeholder for missing data
        print("  " + model.ljust(24) + "  " + "  ".join(gaps))
    with open(CAVEAT_CURVE, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "instruction", "severity", "n", "caveated", "caveat_rate", "lo", "hi",
                    "reports_target_rate", "lexical_caveat_rate"])
        for model, _ in MODELS:
            for iname, _ in INSTRUCTIONS:
                for lv in SEVERITIES:
                    k = (model, iname, lv)
                    if k not in wilson:
                        continue
                    p, lo, hi = wilson[k]
                    w.writerow([model, iname, lv, tot[k], cav.get(k, 0), f"{p:.4f}", f"{lo:.4f}",
                                f"{hi:.4f}", f"{rw.get(k,0)/tot[k]:.4f}", f"{lex.get(k,0)/tot[k]:.4f}"])
    print(f"\n  wrote curve to {CAVEAT_CURVE}")

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
ABSTENTION_INSTRUCTIONS = [("STRICT", STRICT), ("PERMISSIVE", PERMISSIVE), ("SOFT", SOFT)]
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
    print("ABSTENTION SWEEP -- design (prior strength 1=obscure .. 5=universal; the document answers NONE of these)")
    for p in sorted(UNANSWERABLE_ITEMS, key=lambda p: p["prior_strength"]): # give me a new list of these item dictionaries ordered by prior from small to large
        print(f"  P{p['prior_strength']}  {p['item_id']:24} parametric_answer={p['parametric_answer']:10} {p['domain']} / {p['proximity']}")
        print(f"       q: {p['q']}")
    print("\n  note: the lexical parametric-answer check misses paraphrases (e.g. '28-day', '2.04 m') -- the judge is primary, lexical is a cross-check")
    cells = len(MODELS) * len(ABSTENTION_INSTRUCTIONS) * len(UNANSWERABLE_ITEMS)
    print(f"\n  {len(MODELS)} models x {len(ABSTENTION_INSTRUCTIONS)} instructions x {len(UNANSWERABLE_ITEMS)} items = {cells} cells")
    print(f"  at N={n}: {cells * n} candidate calls + {cells * n} judge calls = {2 * cells * n} API calls")
    problems = validate_items()
    if problems:
        print("\n  ITEM VALIDATION FAILED:")
        for p in problems:
            print(f"    - {p}")
        return False
    print(f"\n  item validation: all {len(UNANSWERABLE_ITEMS)} parametric answers absent from document.txt")
    return True

def run_ungrounded(n):
    if not print_abstention_plan(n):
        sys.exit(1)
    done = load_done(ABSTENTION_RESULTS, ["model", "instruction", "item_id"])
    out = open(ABSTENTION_RESULTS, "a")
    total = len(MODELS) * len(ABSTENTION_INSTRUCTIONS) * len(UNANSWERABLE_ITEMS)
    seen = 0
    for model, prov in MODELS:
        for iname, instr in ABSTENTION_INSTRUCTIONS:
            for p in UNANSWERABLE_ITEMS:
                seen += 1
                key = (model, iname, p["item_id"])
                already = done.get(key, 0)
                cell = {}
                for _ in range(already, n):
                    answer = with_retry(call, model, prov, instr, p["q"], passage)
                    faithful, reason = judge(p["q"], passage, answer)
                    label = FAITHFUL if faithful else UNGROUNDED
                    row = {"model": model, "provider": prov, "instruction": iname,
                           "item_id": p["item_id"], "prior_strength": p["prior_strength"], "domain": p["domain"],
                           "proximity": p["proximity"], "q": p["q"], "parametric_answer": p["parametric_answer"],
                           "answer": answer, "faithful": faithful, "judge_reason": reason,
                           "reports_parametric_answer": appears(p["parametric_answer"], answer),
                           "verbatim_abstention": "not in document" in answer.lower(),
                           "label": label}
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                    cell[label] = cell.get(label, 0) + 1
                status = "complete (resumed)" if already >= n else " ".join(f"{k}={v}" for k, v in sorted(cell.items()))
                print(f"  [{seen}/{total}] {model} / {iname} / P{p['prior_strength']} {p['item_id']}  {status}", flush=True)
    out.close()
    summarize_ungrounded()

def summarize_ungrounded():
    rows = [json.loads(l) for l in open(ABSTENTION_RESULTS)]
    tot, ungrounded, lex, vabst = {}, {}, {}, {}
    for r in rows:
        k = (r["model"], r["instruction"], r["prior_strength"])
        tot[k] = tot.get(k, 0) + 1
        ungrounded[k] = ungrounded.get(k, 0) + (r["label"] == UNGROUNDED)
        lex[k] = lex.get(k, 0) + bool(r["reports_parametric_answer"])
        vabst[k] = vabst.get(k, 0) + bool(r["verbatim_abstention"])
    wilson = {}
    for model, _ in MODELS:
        for iname, _ in ABSTENTION_INSTRUCTIONS:
            for pr in PRIOR_STRENGTHS:
                k = (model, iname, pr)
                if tot.get(k):
                    wilson[k] = wilson_interval(ungrounded.get(k, 0), tot[k])
    print("\nUNGROUNDED-ANSWER RATE vs PRIOR STRENGTH  (judge; prior strength 1=obscure .. 5=universal)")
    for model, _ in MODELS:
        for iname, _ in ABSTENTION_INSTRUCTIONS:
            cells = []
            for pr in PRIOR_STRENGTHS:
                k = (model, iname, pr)
                if k in wilson:
                    p, lo, hi = wilson[k]
                    cells.append(f"P{pr}={p:.2f}[{lo:.2f},{hi:.2f}]")
                else:
                    cells.append(f"P{pr}=--")
            print("  " + f"{model} / {iname}".ljust(30) + "  " + "  ".join(cells))
    print("\nPERMISSIVE - STRICT and SOFT - STRICT ungrounded-answer-rate gaps, per prior strength:")
    for model, _ in MODELS:
        for gap_name in ("PERMISSIVE", "SOFT"):
            gaps = []
            for pr in PRIOR_STRENGTHS:
                ks, kg = (model, "STRICT", pr), (model, gap_name, pr)
                if tot.get(ks) and tot.get(kg):
                    gaps.append(f"P{pr}={ungrounded.get(kg,0)/tot[kg] - ungrounded.get(ks,0)/tot[ks]:+.2f}")
                else:
                    gaps.append(f"P{pr}=--")
            print("  " + f"{model} {gap_name}-STRICT".ljust(36) + "  " + "  ".join(gaps))
    with open(ABSTENTION_CURVE, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "instruction", "prior_strength", "n", "ungrounded", "ungrounded_rate", "lo", "hi",
                    "reports_parametric_answer_rate", "verbatim_abstention_rate"])
        for model, _ in MODELS:
            for iname, _ in ABSTENTION_INSTRUCTIONS:
                for pr in PRIOR_STRENGTHS:
                    k = (model, iname, pr)
                    if k not in wilson:
                        continue
                    p, lo, hi = wilson[k]
                    w.writerow([model, iname, pr, tot[k], ungrounded.get(k, 0), f"{p:.4f}", f"{lo:.4f}",
                                f"{hi:.4f}", f"{lex.get(k,0)/tot[k]:.4f}", f"{vabst.get(k,0)/tot[k]:.4f}"])
    print(f"\n  wrote curve to {ABSTENTION_CURVE}")

def tradeoff_rows(caveat_rows, ungrounded_rows):
    entries = []
    for model, _ in MODELS:
        for iname, _ in INSTRUCTIONS: # compare the results between the tests
            for lv in SEVERITIES:
                f = [r for r in caveat_rows if r["model"] == model and r["instruction"] == iname and r["severity"] == lv]
                l = [r for r in ungrounded_rows if r["model"] == model and r["instruction"] == iname and r["prior_strength"] == lv]
                if not f and not l:
                    continue
                entries.append({"model": model, "instruction": iname, "severity": lv,
                                "caveat_n": len(f), "caveat_rate": sum(r["label"] == "caveated" for r in f) / len(f) if f else None,
                                "ungrounded_n": len(l), "ungrounded_rate": sum(r["label"] == UNGROUNDED for r in l) / len(l) if l else None})
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
    print("TRADE-OFF -- error-caveating vs abstention, per model x instruction x severity")
    print("  caveat = caveat-rate at this perturbation severity (denominator includes abstentions)")
    print("  ungrounded = ungrounded-answer rate at the matching prior-strength level")
    print("  S0 = unperturbed control: a caveat at S0 is a false positive; it has no abstention counterpart")
    for e in entries:
        fr = "--" if e["caveat_rate"] is None else f"{e['caveat_rate']:.2f} (n={e['caveat_n']})"
        lr = "--" if e["ungrounded_rate"] is None else f"{e['ungrounded_rate']:.2f} (n={e['ungrounded_n']})"
        print(f"  {e['model']:<24} {e['instruction']:<10}  S{e['severity']}  caveat {fr:>14}   ungrounded {lr:>14}")

if __name__ == "__main__": # only run file if executed directly 
    args = sys.argv[1:] 
    if args and args[0] == "caveat": # if args and args[0] = if the first argument is caveat
        run_caveat(int(args[1]) if len(args) > 1 else 8)
    elif args and args[0] == "abstention":
        run_ungrounded(int(args[1]) if len(args) > 1 else 8)
    elif args and args[0] == "tradeoff":
        tradeoff()
    elif args and not args[0].isdigit():
        print("usage: python3 harness.py [N] | caveat [N] | abstention [N] | tradeoff")
        sys.exit(1)
    else:
        n = int(args[0]) if args else 8
        print_plan(n)
        print()
        print_abstention_plan(n)
        print("\n  (dry run -- no API calls. To execute: python3 harness.py caveat [N]  or  python3 harness.py abstention [N]. Joint readout: python3 harness.py tradeoff)")
