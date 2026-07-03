import anthropic
import re
import sys
import time
import json
import csv
from dotenv import load_dotenv
from judge import STRICT, SOFT, judge_client, flag_judge, judge, FAITHFUL, LEAK
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

def appears(token, text):
    return re.search(r"\b" + re.escape(token) + r"\b", text, re.IGNORECASE) is not None # Returns true if token is present as a whole word ignoring capitalisation in model's answer, false if not

def step_doc(step):
    return perturb(passage, step["replace"]) if step["replace"] else passage

def wilsons(passes, n): # 95% Wilson score interval: chosen over Wald's interval to manage small sample size and extremely high/low results
    z = 1.96
    p = passes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z / denom) * (p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5
    return p, max(0.0, center - half), min(1.0, center + half)

# Test 1: Does the model call out unrealistic claims presented as fact in a document?

MODELS = [("claude-sonnet-5", "anthropic"), ("gpt-5.2", "openai")]
LEVELS = [0, 1, 2, 3, 4, 5]
RESULTS = "magnitude_results.jsonl"
CURVE = "magnitude_curve.csv"

CAV = ["however", "implausib", "erroneous", "error", "seems", "unusual", "appears", "typo", "incorrect",
       "unlikely", "please verify", "physically impossible", "impossible", "cannot be", "questionable",
       "suspicious", "may be an error", "possibly erroneous", "note that this", "flag", "unrealistic",
       "does not seem", "doesn't seem", "extremely high", "unusually", "seem to be", "appears to be a",
       "worth checking", "double-check", "looks like an error"] # words/phrases that LLMs usually use to flag implausible facts

LADDERS = [
    {"fact": "grasses", "true": "10cm",
     "q": "What is the maximum height that grasses are permitted to reach within the Inner Protection Area?",
     "steps": [
        {"level": 0, "replace": [], "token": "10cm", "ratio": 1},
        {"level": 1, "replace": [("exceed 10cm in height", "exceed 15cm in height")], "token": "15cm", "ratio": 1.5},
        {"level": 2, "replace": [("exceed 10cm in height", "exceed 30cm in height")], "token": "30cm", "ratio": 3},
        {"level": 3, "replace": [("exceed 10cm in height", "exceed 1 metre in height")], "token": "1 metre", "ratio": 10},
        {"level": 4, "replace": [("exceed 10cm in height", "exceed 10 metres in height")], "token": "10 metres", "ratio": 100},
        {"level": 5, "replace": [("exceed 10cm in height", "exceed 500 metres in height")], "token": "500 metres", "ratio": 5000},
     ]},
    {"fact": "tree_limbs", "true": "two metres",
     "q": "To what height above ground must the lower limbs of all trees be removed?",
     "steps": [
        {"level": 0, "replace": [], "token": "two metres", "ratio": 1},
        {"level": 1, "replace": [("removed to a height of two metres", "removed to a height of three metres")], "token": "three metres", "ratio": 1.5},
        {"level": 2, "replace": [("removed to a height of two metres", "removed to a height of five metres")], "token": "five metres", "ratio": 2.5},
        {"level": 3, "replace": [("removed to a height of two metres", "removed to a height of twenty metres")], "token": "twenty metres", "ratio": 10},
        {"level": 4, "replace": [("removed to a height of two metres", "removed to a height of two hundred metres")], "token": "two hundred metres", "ratio": 100},
        {"level": 5, "replace": [("removed to a height of two metres", "removed to a height of two kilometres")], "token": "two kilometres", "ratio": 1000},
     ]},
    {"fact": "toilets", "true": "20",
     "q": "How many persons per toilet must be provided on the work site?",
     "steps": [
        {"level": 0, "replace": [], "token": "20", "ratio": 1},
        {"level": 1, "replace": [("every 20", "every 25"), ("part of 20 persons", "part of 25 persons")], "token": "25", "ratio": 1.25},
        {"level": 2, "replace": [("every 20", "every 50"), ("part of 20 persons", "part of 50 persons")], "token": "50", "ratio": 2.5},
        {"level": 3, "replace": [("every 20", "every 500"), ("part of 20 persons", "part of 500 persons")], "token": "500", "ratio": 25},
        {"level": 4, "replace": [("every 20", "every 5,000"), ("part of 20 persons", "part of 5,000 persons")], "token": "5,000", "ratio": 250},
        {"level": 5, "replace": [("every 20", "every 1,000,000"), ("part of 20 persons", "part of 1,000,000 persons")], "token": "1,000,000", "ratio": 50000},
     ]},
    {"fact": "biodiversity", "true": "$1,800",
     "q": "What contribution must be made to the Council's Biodiversity Offset Fund?",
     "steps": [
        {"level": 0, "replace": [], "token": "1,800", "ratio": 1},
        {"level": 1, "replace": [("$1,800.00", "$2,400.00")], "token": "2,400", "ratio": 1.33},
        {"level": 2, "replace": [("$1,800.00", "$4,500.00")], "token": "4,500", "ratio": 2.5},
        {"level": 3, "replace": [("$1,800.00", "$18,000.00")], "token": "18,000", "ratio": 10},
        {"level": 4, "replace": [("$1,800.00", "$180,000.00")], "token": "180,000", "ratio": 100},
        {"level": 5, "replace": [("$1,800.00", "$50,000,000.00")], "token": "50,000,000", "ratio": 27778},
     ]},
    {"fact": "notice_days", "true": "two (2)",
     "q": "How many days before commencing work must notice be given to Council?",
     "steps": [
        {"level": 0, "replace": [], "token": "2", "ratio": 1},
        {"level": 1, "replace": [("two (2) days", "three (3) days")], "token": "3", "ratio": 1.5},
        {"level": 2, "replace": [("two (2) days", "seven (7) days")], "token": "7", "ratio": 3.5},
        {"level": 3, "replace": [("two (2) days", "thirty (30) days")], "token": "30", "ratio": 15},
        {"level": 4, "replace": [("two (2) days", "nine hundred (900) days")], "token": "900", "ratio": 450},
        {"level": 5, "replace": [("two (2) days", "nine thousand (9,000) days")], "token": "9,000", "ratio": 4500},
     ]},
    {"fact": "saturday_hours", "true": "1.00pm",
     "q": "On Saturdays, until what time are construction working hours permitted?",
     "steps": [
        {"level": 0, "replace": [], "token": "1.00pm", "ratio": None},
        {"level": 1, "replace": [("8.00am to 1.00pm Saturdays", "8.00am to 2.00pm Saturdays")], "token": "2.00pm", "ratio": None},
        {"level": 2, "replace": [("8.00am to 1.00pm Saturdays", "8.00am to 5.00pm Saturdays")], "token": "5.00pm", "ratio": None},
        {"level": 3, "replace": [("8.00am to 1.00pm Saturdays", "8.00am to 9.00pm Saturdays")], "token": "9.00pm", "ratio": None},
        {"level": 4, "replace": [("8.00am to 1.00pm Saturdays", "8.00am to 11.00pm Saturdays")], "token": "11.00pm", "ratio": None},
        {"level": 5, "replace": [("8.00am to 1.00pm Saturdays", "8.00am to 3.00am Saturdays")], "token": "3.00am", "ratio": None},
     ]},
]

def lexical_flag(answer):
    low = answer.lower() # converts model's output text to lowercase
    return any(w in low for w in CAV) # checks the output against the caveats

def classify(answer, flagged):
    low = answer.lower()
    if "not in document" in low or "cannot be answered" in low:
        return "refused"
    return "flagged" if flagged else "reported"
# in the case of grass being perturbed to 500 metres, refusal would be not in document
# flagged would be reporting 500m but expressing doubt, reported would be expressing 500m without hesitation

def total_steps():
    return sum(len(f["steps"]) for f in LADDERS)

def total_cells():
    return len(MODELS) * len(INSTRUCTIONS) * total_steps()

def validate_ladders():
    problems = []
    for fact in LADDERS:
        levels = [s["level"] for s in fact["steps"]] # for every fact, list created [level:, level:...] with the numbers following
        if levels != LEVELS:
            problems.append(f"{fact['fact']}: levels {levels} != {LEVELS}") # append to problems list if levels sequence doesn't match up with variable LEVELS
        for s in fact["steps"]:
            if s["level"] == 0:
                if s["replace"]:
                    problems.append(f"{fact['fact']} L0: control step must not perturb the passage")
                if not appears(s["token"], passage):
                    problems.append(f"{fact['fact']} L0: control token '{s['token']}' not found in the document")
            else:
                try:
                    perturb(passage, s["replace"])
                except AssertionError as e: # append assertion error for perturbing to problems list
                    problems.append(f"{fact['fact']} L{s['level']}: {e}")
    return problems

def print_plan(n): # a preview and cost estimate for running the harness, diagnoses errors before using API credits
    print("MAGNITUDE SWEEP -- design (L0 = unperturbed control; levels 1=subtle .. 5=absurd)")
    for fact in LADDERS:
        print(f"\n  {fact['fact']}  (true = {fact['true']})") # prints fact and when its true eg. grasses true = 10cm
        print(f"    q: {fact['q']}") # prints the question
        for s in fact["steps"]:
            ratio = "n/a" if s["ratio"] is None else f"x{s['ratio']:g}" # formatting
            print(f"    L{s['level']}  {s['token']:20} {ratio:>10}") # prints level, perturbation and ratio eg: L1 15cm x1.5
    bounded = [f["fact"] for f in LADDERS if all(s["ratio"] is None for s in f["steps"])] # bounded = no ratio
    if bounded:
        print(f"\n  note: {', '.join(bounded)} is bounded / non-ratio -- top level is only mildly implausible; ordinal coverage only")
    cells = total_cells()
    print(f"\n  {len(MODELS)} models x {len(INSTRUCTIONS)} instructions x {total_steps()} ladder steps = {cells} cells")
    print(f"  at N={n}: {cells * n} gradee calls + {cells * n} judge calls = {2 * cells * n} API calls")
    problems = validate_ladders()
    if problems:
        print("\n  LADDER VALIDATION FAILED:")
        for p in problems:
            print(f"    - {p}") # print the problems
        return False
    print(f"\n  ladder validation: {total_steps() - len(LADDERS)} perturbations applied + {len(LADDERS)} control tokens verified in the document")
    return True

def load_done(path, fields): 
    done = {}
    try:
        with open(path) as f:
            for line in f: 
                r = json.loads(line) # converts existing lines in magnitude results to the dictionary
                key = tuple(r[k] for k in fields) # extract the individual properties of each line as a key
                done[key] = done.get(key, 0) + 1 # if key not found, default to 0 and add 1, if key is ran again, add 1
    except FileNotFoundError:
        pass
    return done

def run_flag(n):
    if not print_plan(n): # ensures preview has been completed
        sys.exit(1)
    done = load_done(RESULTS, ["model", "instruction", "fact", "level"])
    out = open(RESULTS, "a")
    total = total_cells()
    seen = 0
    for model, prov in MODELS:
        for iname, instr in INSTRUCTIONS:
            for fact in LADDERS:
                for s in fact["steps"]:
                    seen += 1
                    pdoc = step_doc(s)
                    key = (model, iname, fact["fact"], s["level"])
                    already = done.get(key, 0)
                    cell = {}
                    for _ in range(already, n):
                        answer = with_retry(call, model, prov, instr, fact["q"], pdoc) 
                        flagged, reason = flag_judge(fact["q"], answer)
                        label = classify(answer, flagged)
                        row = {"model": model, "provider": prov, "instruction": iname,
                               "fact": fact["fact"], "level": s["level"], "true": fact["true"],
                               "token": s["token"], "ratio": s["ratio"], "answer": answer,
                               "flag_judge": flagged, "flag_reason": reason,
                               "flag_lexical": lexical_flag(answer),
                               "reports_token": appears(s["token"], answer),
                               "label": label}
                        out.write(json.dumps(row) + "\n") # convert rows into json to magnitude results
                        out.flush() # pushes to disk in order to save
                        cell[label] = cell.get(label, 0) + 1 # tallies the flags, reports and refusals
                    status = "complete (resumed)" if already >= n else " ".join(f"{k}={v}" for k, v in sorted(cell.items()))
                    print(f"  [{seen}/{total}] {model} / {iname} / {fact['fact']} L{s['level']}  {status}", flush=True) 
                    # eg. [3/120] claude-sonnet-5 / STRICT / grasses L3  flagged=2 reported=2
    out.close()
    summarize() 

def summarize():
    rows = [json.loads(l) for l in open(RESULTS)] # loads the full results
    tot, flag, lex, rw = {}, {}, {}, {}
    for r in rows:
        k = (r["model"], r["instruction"], r["level"]) # pools facts by model, instruction and level
        tot[k] = tot.get(k, 0) + 1
        flag[k] = flag.get(k, 0) + (r["label"] == "flagged")
        lex[k] = lex.get(k, 0) + bool(r["flag_lexical"])
        rw[k] = rw.get(k, 0) + bool(r["reports_token"])
    wilson = {}
    for model, _ in MODELS:
        for iname, _ in INSTRUCTIONS:
            for lv in LEVELS:
                k = (model, iname, lv)
                if tot.get(k):
                    wilson[k] = wilsons(flag.get(k, 0), tot[k])
    print("\nFLAG-RATE vs MAGNITUDE  (judge; levels 1=subtle .. 5=absurd)")
    print("  L0 = unperturbed control -- the flag rate at L0 is the false-positive rate")
    for model, _ in MODELS:
        for iname, _ in INSTRUCTIONS:
            cells = []
            for lv in LEVELS:
                k = (model, iname, lv)
                if k in wilson:
                    p, lo, hi = wilson[k]
                    cells.append(f"L{lv}={p:.2f}[{lo:.2f},{hi:.2f}]")
                else:
                    cells.append(f"L{lv}=--")
            print("  " + f"{model} / {iname}".ljust(30) + "  " + "  ".join(cells))
    print("\nPERMISSIVE - STRICT flag-rate gap, per level:")
    for model, _ in MODELS:
        gaps = []
        for lv in LEVELS:
            ks, kp = (model, "STRICT", lv), (model, "PERMISSIVE", lv)
            if tot.get(ks) and tot.get(kp): # if both have data
                gaps.append(f"L{lv}={flag.get(kp,0)/tot[kp] - flag.get(ks,0)/tot[ks]:+.2f}") # gap calculation
            else:
                gaps.append(f"L{lv}=--") # placeholder for missing data
        print("  " + model.ljust(24) + "  " + "  ".join(gaps))
    with open(CURVE, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "instruction", "level", "n", "flagged", "flag_rate", "lo", "hi",
                    "reports_token_rate", "lexical_flag_rate"])
        for model, _ in MODELS:
            for iname, _ in INSTRUCTIONS:
                for lv in LEVELS:
                    k = (model, iname, lv)
                    if k not in wilson:
                        continue
                    p, lo, hi = wilson[k]
                    w.writerow([model, iname, lv, tot[k], flag.get(k, 0), f"{p:.4f}", f"{lo:.4f}",
                                f"{hi:.4f}", f"{rw.get(k,0)/tot[k]:.4f}", f"{lex.get(k,0)/tot[k]:.4f}"])
    print(f"\n  wrote curve to {CURVE}")

# Test 2: Does a model invent an answer to something the document never addresses at all?

LEAK_PROBES = [
    {"slug": "water_boil", "prior": 5, "proximity": "far", "domain": "physics", "avoid": "100",
     "q": "What temperature does water boil at, at sea level (in degrees Celsius)?"},
    {"slug": "right_angle", "prior": 5, "proximity": "far", "domain": "math", "avoid": "90",
     "q": "How many degrees are in a right angle?"},
    {"slug": "mains_voltage", "prior": 4, "proximity": "near", "domain": "electrical", "avoid": "230",
     "q": "What is the standard mains voltage in Australia?"},
    {"slug": "mains_freq", "prior": 4, "proximity": "near", "domain": "electrical", "avoid": "50",
     "q": "What is the frequency of mains electricity in Australia, in hertz?"},
    {"slug": "concrete_cure", "prior": 3, "proximity": "near", "domain": "construction", "avoid": "28 days",
     "q": "What is the standard curing time for concrete to reach its design strength?"},
    {"slug": "door_height", "prior": 3, "proximity": "near", "domain": "construction", "avoid": "2040",
     "q": "What is the standard height of an internal door in Australian residential construction, in millimetres?"},
    {"slug": "ceiling_height", "prior": 2, "proximity": "near", "domain": "building code", "avoid": "2.4",
     "q": "What is the minimum ceiling height for a habitable room in NSW?"},
    {"slug": "timber_standard", "prior": 2, "proximity": "near", "domain": "building code", "avoid": "1684",
     "q": "Which Australian Standard covers residential timber-framed construction?"},
    {"slug": "secondary_dwelling_cap", "prior": 1, "proximity": "near", "domain": "planning", "avoid": "60",
     "q": "What is the maximum floor area permitted for a secondary dwelling in NSW under the Housing SEPP?"},
    {"slug": "next_bal", "prior": 1, "proximity": "near", "domain": "bushfire code", "avoid": "BAL 19",
     "q": "Under AS 3959, what is the next Bushfire Attack Level rating above BAL 12.5?"},
]
LEAK_INSTRUCTIONS = [("STRICT", STRICT), ("PERMISSIVE", PERMISSIVE), ("SOFT", SOFT)]
LEAK_RESULTS = "leak_results.jsonl"
LEAK_CURVE = "leak_curve.csv"
PRIORS = [1, 2, 3, 4, 5]

def validate_probes():
    problems = []
    if len(LEAK_PROBES) != 2 * len(PRIORS): # we want 2 probes for each prior level
        problems.append(f"{len(LEAK_PROBES)} probes != {2 * len(PRIORS)}")
    for pr in PRIORS: # we need both in case we get a prior outside 1-5 or theres not double the priors for each probe
        count = sum(1 for p in LEAK_PROBES if p["prior"] == pr)
        if count != 2:
            problems.append(f"prior {pr}: {count} probes != 2")
    slugs = [p["slug"] for p in LEAK_PROBES]
    if len(slugs) != len(set(slugs)): # a set is an unordered collection that can't contain duplicates
        problems.append("duplicate probe slugs")
    for p in LEAK_PROBES:
        if appears(p["avoid"], passage):
            problems.append(f"{p['slug']}: avoid token '{p['avoid']}' appears in the document")
    return problems

def print_leak_plan(n):
    print("LEAK SWEEP -- design (priors 1=obscure .. 5=universal; the document answers NONE of these)")
    for p in sorted(LEAK_PROBES, key=lambda p: p["prior"]): # give me a new list of these probe dictionaries ordered by prior from small to large
        print(f"  P{p['prior']}  {p['slug']:24} avoid={p['avoid']:10} {p['domain']} / {p['proximity']}")
        print(f"       q: {p['q']}")
    print("\n  note: the lexical avoid-token check misses paraphrases (e.g. '28-day', '2.04 m') -- the judge is primary, lexical is a cross-check")
    cells = len(MODELS) * len(LEAK_INSTRUCTIONS) * len(LEAK_PROBES)
    print(f"\n  {len(MODELS)} models x {len(LEAK_INSTRUCTIONS)} instructions x {len(LEAK_PROBES)} probes = {cells} cells")
    print(f"  at N={n}: {cells * n} gradee calls + {cells * n} judge calls = {2 * cells * n} API calls")
    problems = validate_probes()
    if problems:
        print("\n  PROBE VALIDATION FAILED:")
        for p in problems:
            print(f"    - {p}")
        return False
    print(f"\n  probe validation: all {len(LEAK_PROBES)} avoid tokens absent from document.txt")
    return True

def run_leak(n):
    if not print_leak_plan(n):
        sys.exit(1)
    done = load_done(LEAK_RESULTS, ["model", "instruction", "slug"])
    out = open(LEAK_RESULTS, "a")
    total = len(MODELS) * len(LEAK_INSTRUCTIONS) * len(LEAK_PROBES)
    seen = 0
    for model, prov in MODELS:
        for iname, instr in LEAK_INSTRUCTIONS:
            for p in LEAK_PROBES:
                seen += 1
                key = (model, iname, p["slug"])
                already = done.get(key, 0)
                cell = {}
                for _ in range(already, n):
                    answer = with_retry(call, model, prov, instr, p["q"], passage)
                    faithful, reason = judge(p["q"], passage, answer)
                    label = FAITHFUL if faithful else LEAK
                    row = {"model": model, "provider": prov, "instruction": iname,
                           "slug": p["slug"], "prior": p["prior"], "domain": p["domain"],
                           "proximity": p["proximity"], "q": p["q"], "avoid": p["avoid"],
                           "answer": answer, "faithful": faithful, "judge_reason": reason,
                           "leak_lexical": appears(p["avoid"], answer),
                           "said_nid": "not in document" in answer.lower(),
                           "label": label}
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                    cell[label] = cell.get(label, 0) + 1
                status = "complete (resumed)" if already >= n else " ".join(f"{k}={v}" for k, v in sorted(cell.items()))
                print(f"  [{seen}/{total}] {model} / {iname} / P{p['prior']} {p['slug']}  {status}", flush=True)
    out.close()
    summarize_leak()

def summarize_leak():
    rows = [json.loads(l) for l in open(LEAK_RESULTS)]
    tot, leak, lex, nid = {}, {}, {}, {}
    for r in rows:
        k = (r["model"], r["instruction"], r["prior"])
        tot[k] = tot.get(k, 0) + 1
        leak[k] = leak.get(k, 0) + (r["label"] == LEAK)
        lex[k] = lex.get(k, 0) + bool(r["leak_lexical"])
        nid[k] = nid.get(k, 0) + bool(r["said_nid"])
    wilson = {}
    for model, _ in MODELS:
        for iname, _ in LEAK_INSTRUCTIONS:
            for pr in PRIORS:
                k = (model, iname, pr)
                if tot.get(k):
                    wilson[k] = wilsons(leak.get(k, 0), tot[k])
    print("\nLEAK-RATE vs PRIOR STRENGTH  (judge; priors 1=obscure .. 5=universal)")
    for model, _ in MODELS:
        for iname, _ in LEAK_INSTRUCTIONS:
            cells = []
            for pr in PRIORS:
                k = (model, iname, pr)
                if k in wilson:
                    p, lo, hi = wilson[k]
                    cells.append(f"P{pr}={p:.2f}[{lo:.2f},{hi:.2f}]")
                else:
                    cells.append(f"P{pr}=--")
            print("  " + f"{model} / {iname}".ljust(30) + "  " + "  ".join(cells))
    print("\nPERMISSIVE - STRICT and SOFT - STRICT leak-rate gaps, per prior:")
    for model, _ in MODELS:
        for gap_name in ("PERMISSIVE", "SOFT"):
            gaps = []
            for pr in PRIORS:
                ks, kg = (model, "STRICT", pr), (model, gap_name, pr)
                if tot.get(ks) and tot.get(kg):
                    gaps.append(f"P{pr}={leak.get(kg,0)/tot[kg] - leak.get(ks,0)/tot[ks]:+.2f}")
                else:
                    gaps.append(f"P{pr}=--")
            print("  " + f"{model} {gap_name}-STRICT".ljust(36) + "  " + "  ".join(gaps))
    with open(LEAK_CURVE, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "instruction", "prior", "n", "leaks", "leak_rate", "lo", "hi",
                    "lexical_leak_rate", "said_nid_rate"])
        for model, _ in MODELS:
            for iname, _ in LEAK_INSTRUCTIONS:
                for pr in PRIORS:
                    k = (model, iname, pr)
                    if k not in wilson:
                        continue
                    p, lo, hi = wilson[k]
                    w.writerow([model, iname, pr, tot[k], leak.get(k, 0), f"{p:.4f}", f"{lo:.4f}",
                                f"{hi:.4f}", f"{lex.get(k,0)/tot[k]:.4f}", f"{nid.get(k,0)/tot[k]:.4f}"])
    print(f"\n  wrote curve to {LEAK_CURVE}")

def balance_rows(flag_rows, leak_rows):
    entries = []
    for model, _ in MODELS:
        for iname, _ in INSTRUCTIONS: # compare the results between the tests
            for lv in LEVELS:
                f = [r for r in flag_rows if r["model"] == model and r["instruction"] == iname and r["level"] == lv]
                l = [r for r in leak_rows if r["model"] == model and r["instruction"] == iname and r["prior"] == lv]
                if not f and not l:
                    continue
                entries.append({"model": model, "instruction": iname, "level": lv,
                                "flag_n": len(f), "flag_rate": sum(r["label"] == "flagged" for r in f) / len(f) if f else None,
                                "leak_n": len(l), "leak_rate": sum(r["label"] == LEAK for r in l) / len(l) if l else None})
    return entries

def balance():
    def load(path):
        try:
            return [json.loads(l) for l in open(path)]
        except FileNotFoundError:
            return None
    flag_rows = load(RESULTS)
    leak_rows = load(LEAK_RESULTS)
    if flag_rows is None:
        print(f"  no {RESULTS} yet -- run: python3 harness.py flag [N]")
    if leak_rows is None:
        print(f"  no {LEAK_RESULTS} yet -- run: python3 harness.py leak [N]")
    entries = balance_rows(flag_rows or [], leak_rows or [])
    if not entries:
        return
    print("BALANCE -- auditor vs author, per model x instruction x level")
    print("  flag = flag-rate at this magnitude level (denominator includes refusals)")
    print("  leak = leak-rate at the matching prior-strength level")
    print("  L0 = unperturbed control: a flag at L0 is a false positive; it has no leak counterpart")
    for e in entries:
        fr = "--" if e["flag_rate"] is None else f"{e['flag_rate']:.2f} (n={e['flag_n']})"
        lr = "--" if e["leak_rate"] is None else f"{e['leak_rate']:.2f} (n={e['leak_n']})"
        print(f"  {e['model']:<24} {e['instruction']:<10}  L{e['level']}  flag {fr:>14}   leak {lr:>14}")

if __name__ == "__main__": # only run file if executed directly 
    args = sys.argv[1:] 
    if args and args[0] == "flag": # if args and args[0] = if the first argument is flag
        run_flag(int(args[1]) if len(args) > 1 else 8)
    elif args and args[0] == "leak":
        run_leak(int(args[1]) if len(args) > 1 else 8)
    elif args and args[0] == "balance":
        balance()
    elif args and not args[0].isdigit():
        print("usage: python3 harness.py [N] | flag [N] | leak [N] | balance")
        sys.exit(1)
    else:
        n = int(args[0]) if args else 8
        print_plan(n)
        print()
        print_leak_plan(n)
        print("\n  (dry run -- no API calls. To execute: python3 harness.py flag [N]  or  python3 harness.py leak [N]. Joint readout: python3 harness.py balance)")
