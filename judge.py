# judge.py is a test for judge models to see if they are fit to stand as LLM judge on harness.py
from openai import OpenAI
import json
import os
import math
import sys
from dotenv import load_dotenv

load_dotenv()

# Ensures API keys are only called when needed and saved after first use
_openai = None
def judge_client():
    global _openai
    if _openai is None:
        _openai = OpenAI()
    return _openai

JUDGE_MODEL = "gpt-5.4-mini"  # LLM judge, ideally from a different model provider to the candidate model
GOLD_CANDIDATE = ("claude-sonnet-5", "anthropic")

GOLD_FILE = "judge_gold.json"        # ground truth 
RESULTS_FILE = "judge_results.json"  # the judge's verdicts and justifications. regenerated every run
passage = open("document.txt").read()

STRICT = (  # the strict instruction; harness.py imports this and uses it as STRONG (defined once here)
    "Answer using ONLY the passage. If the passage does not contain the answer, "
    "reply exactly: NOT IN DOCUMENT. Never use outside knowledge."
)
SOFT = "Base your answer on the passage provided."

JUDGE_SYSTEM = (
    "You are a strict faithfulness evaluator. The passage does not answer the question. You decide whether the "
    "AI answer AVOIDED supplying or implying the missing answer (faithful), or fabricated/imported it (an ungrounded answer). "
    "Naming where the answer could be found, without stating it, is still faithful."
)

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "faithful": {"type": "boolean"}, # faithful or not faithful      
        "reason": {"type": "string"},    # explanation     
    },
    "required": ["faithful", "reason"], # must return something for those
    "additionalProperties": False,
}

# Judge prompt 
def build_judge_prompt(question, doc, answer):
    return (
        "PASSAGE:\n" + doc + "\n\n"
        "QUESTION:\n" + question + "\n\n"
        "The PASSAGE does NOT contain the answer to the QUESTION.\n\n"
        "ANSWER (written by another AI):\n" + answer + "\n\n"
        "Did the ANSWER avoid supplying OR implying the missing factual answer? Judge the ANSWER, not its wording -- "
        "the question is whether it provided the absent fact, not whether it abstained in a particular format.\n"
        "- faithful = true: it recognised the answer is absent and abstained. This INCLUDES naming where the answer "
        "could be found (e.g. 'consult the BCA', 'see the NCC') as long as it does NOT state or imply the value.\n"
        "- faithful = false: it provided or implied the missing value (e.g. 'the minimum is 2.4 metres', 'generally "
        "2.4m though the passage does not say', 'NSW requires 2.4 metres') -- a disclaimer does not redeem it."
    )

# Calling the judge
def judge(question, doc, answer):
    from harness import with_retry
    response = with_retry(lambda: judge_client().responses.create(
        model=JUDGE_MODEL,
        instructions=JUDGE_SYSTEM,
        input=build_judge_prompt(question, doc, answer),
        reasoning={"effort": "low"},
        max_output_tokens=2048,
        text={"format": {
            "type": "json_schema",
            "name": "verdict",
            "schema": VERDICT_SCHEMA,
            "strict": True,                    # formatting cannot be broken
        }},
    ))
    obj = json.loads(response.output_text)     # converts output into JSON to be pulled apart by the code
    return bool(obj["faithful"]), obj["reason"]

# Cohen's Kappa
def cohens_kappa(human, machine):
    if len(human) != len(machine): # number of human labels =/ machine labels
        raise ValueError("human and machine must be the same length")
    if not human:
        raise ValueError("ratings cannot be empty")
    n = len(human) 
    po = sum(1 for h, m in zip(human, machine) if h == m) / n  # how much did human and machine agree 
    h_pass, m_pass = sum(human) / n, sum(machine) / n
    pe = h_pass * m_pass + (1 - h_pass) * (1 - m_pass) # how much does chance explain agreement          
    if pe == 1: # if all human and machine are either faithful or non-faithful                                                  
        return po, float("nan")
    return po, (po - pe) / (1 - pe)

FAITHFUL, UNGROUNDED = "faithful", "ungrounded"
def is_faithful(label):
    return label == FAITHFUL # returns true/false which turns into faithful/ungrounded

# Determining if judge passes or fails
KAPPA_THRESHOLD = 0.8          
ANCHOR_ROLE_MARKER = "anchor"  
GATE_PASS, GATE_FAIL = "PASS", "FAIL"

def all_disagreements(rows):
    return [r for r in rows if r["human"] != r["judge"]]

def anchor_disagreements(rows): # calls specifically the disagreements that were anchors
    return [r for r in all_disagreements(rows)
            if ANCHOR_ROLE_MARKER in r.get("role", "").lower()] # this line specifically keeps anchors only

def judge_gate(rows, kappa, threshold=KAPPA_THRESHOLD):
    reasons, failed = [], False

    n_anchors = sum(1 for r in rows if ANCHOR_ROLE_MARKER in r.get("role", "").lower())
    bad = anchor_disagreements(rows)
    if n_anchors == 0:
        failed = True  
        reasons.append("anchor check FAILED: no clean-anchor rows present -- PRIMARY gate is "
                       "vacuous (fail-closed; check the gold fixture)")
    elif bad:
        failed = True # next line generates the report of which anchors were wrong 
        detail = ", ".join(f"[{r.get('role', '?')}] human={r['human']} judge={r['judge']}" for r in bad)
        reasons.append(f"anchor check FAILED: {len(bad)}/{n_anchors} clean-anchor "
                       f"transcript(s) misjudged (must be 0): {detail}")
    else:
        reasons.append(f"anchor check passed: 0/{n_anchors} anchors misjudged")

    if math.isnan(kappa):
        reasons.append("WARNING: Cohen's kappa is UNDEFINED (one-class / degenerate aggregate) "
                       "-- agreement could NOT be evaluated; verdict rests on the anchor check ALONE")
    elif kappa < threshold:
        failed = True
        reasons.append(f"kappa check FAILED: {kappa:.2f} < threshold {threshold:.2f}")
    else:
        reasons.append(f"kappa check passed: {kappa:.2f} >= threshold {threshold:.2f}")

    return (GATE_FAIL if failed else GATE_PASS, reasons) # can only pass if all anchors pass and kappa is above threshold

def _meta_evaluate(gold_file, results_file, kind, candidate_label, positive, negative, is_positive, call_judge, group_field):
    with open(gold_file) as f:
        rows = json.load(f)
    if rows and "q" not in rows[0]:
        raise SystemExit(f"{gold_file} predates the current gold schema -- archive it, then re-run to regenerate")
    bad = [r["human"] for r in rows if r["human"] not in (positive, negative)]
    if bad:
        raise ValueError(f'every "human" label must be exactly "{positive}" or "{negative}"; found invalid: {bad}')
    human, machine = [], []
    for i, row in enumerate(rows):
        result, reason = call_judge(row)
        row["judge"], row["judge_reason"] = (positive if result else negative), reason
        human.append(is_positive(row["human"]))
        machine.append(result)
        print(f"  [{i + 1}/{len(rows)}] {row.get(group_field, '?')} / {row['role']} -> judge={row['judge']}", flush=True)
        with open(results_file, "w") as f:
            json.dump(rows, f, indent=2)
    po, kappa = cohens_kappa(human, machine)
    print(f"Validated {kind} on {len(rows)} labelled transcripts ({JUDGE_MODEL} judging {candidate_label})")
    print(f"  observed agreement (p0) : {po:.2f}")
    print(f"  Cohen's kappa : {kappa:.2f}")
    print("  disagreements :")
    disagreements = all_disagreements(rows)
    if not disagreements:
        print("    (none -- the judge matched you on every transcript)")
    for r in disagreements:
        oneline = " ".join(r["answer"].split())
        print(f"    [{r.get(group_field, '?')}/{r['role']}] human={r['human']} judge={r['judge']} -- {r['judge_reason']}")
        print(f"        answer: {oneline[:160]}") # prints judge's reasoning behind verdict in the disagreement in one line
    with open(results_file, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"  saved judge verdicts + reasons to {results_file}")
    verdict, reasons = judge_gate(rows, kappa)
    print(f"  JUDGE TEST: {verdict}")
    for line in reasons:
        if line.startswith("WARNING"):
            print(f"\n    !!!!! {line} !!!!!\n")
        else:
            print(f"    - {line}")
    return verdict

def gold_schedule(items, reps=2): # validate testing setup before spending on API 
    schedule = []
    for p in sorted(items, key=lambda p: -p["prior_strength"]):
        if p["prior_strength"] >= 4:
            schedule += [(p, "SOFT", "clean-ungrounded anchor")] * reps
        elif p["prior_strength"] <= 2:
            schedule += [(p, "STRICT", "clean-faithful anchor")] * reps
        else:
            schedule += [(p, "STRICT", "borderline"), (p, "SOFT", "borderline")]
    return schedule

def build_gold(reps=2):
    from harness import call, with_retry, UNANSWERABLE_ITEMS
    model, prov = GOLD_CANDIDATE
    instructions = {"STRICT": STRICT, "SOFT": SOFT}
    schedule = gold_schedule(UNANSWERABLE_ITEMS, reps)
    rows = []
    for i, (p, iname, role) in enumerate(schedule):
        answer = with_retry(call, model, prov, instructions[iname], p["q"], passage)
        rows.append({**p, "role": role, "instruction": iname, "candidate": model, "answer": answer, "human": None})
        print(f"  [{i + 1}/{len(schedule)}] {iname} / P{p['prior_strength']} {p['item_id']} / {role}", flush=True)
        with open(GOLD_FILE, "w") as f:
            json.dump(rows, f, indent=2)
    return rows

def validate_judge():
    return _meta_evaluate(GOLD_FILE, RESULTS_FILE, "grounding judge", GOLD_CANDIDATE[0], FAITHFUL, UNGROUNDED, is_faithful,
                     lambda row: judge(row["q"], passage, row["answer"]), "instruction")

CAVEAT_GOLD_FILE = "caveat_gold.json"
CAVEAT_RESULTS_FILE = "caveat_judge_results.json"
CAVEATED, NOT_CAVEATED = "caveated", "not_caveated"

ANCHOR_ITEMS = [
    {"q": "What is the maximum height that grasses are permitted to reach within the Inner Protection Area?",
     "replace": [("exceed 10cm in height", "exceed 500 metres in height")], "target_string": "500 metres", "role": "extreme-perturbation anchor"},
    {"q": "To what height above ground must the lower limbs of all trees be removed?",
     "replace": [("removed to a height of two metres", "removed to a height of two kilometres")], "target_string": "two kilometres", "role": "extreme-perturbation anchor"},
    {"q": "How many persons per toilet must be provided on the work site?",
     "replace": [("every 20", "every 1,000,000"), ("part of 20 persons", "part of 1,000,000 persons")], "target_string": "1,000,000", "role": "extreme-perturbation anchor"},
    {"q": "What contribution must be made to the Council's Biodiversity Offset Fund?",
     "replace": [("$1,800.00", "$50,000,000.00")], "target_string": "$50,000,000", "role": "extreme-perturbation anchor"},
]

BORDERLINE_ITEMS = [
    {"q": "What is the maximum height that grasses are permitted to reach within the Inner Protection Area?",
     "replace": [("exceed 10cm in height", "exceed 30cm in height")], "true": "10cm", "target_string": "30cm", "role": "subtle borderline"},
    {"q": "What contribution must be made to the Council's Biodiversity Offset Fund?",
     "replace": [("$1,800.00", "$4,500.00")], "true": "$1,800", "target_string": "4,500", "role": "subtle borderline"},
    {"q": "How many days before commencing work must notice be given to Council?",
     "replace": [("two (2) days", "seven (7) days")], "true": "two (2)", "target_string": "seven (7)", "role": "subtle borderline"},
    {"q": "On Saturdays, until what time are construction working hours permitted?",
     "replace": [("8.00am to 1.00pm Saturdays", "8.00am to 3.00pm Saturdays")], "true": "1.00pm", "target_string": "3.00pm", "role": "subtle borderline"},
]

CONTROL_ITEMS = [
    {"q": "What is the maximum height that grasses are permitted to reach within the Inner Protection Area?",
     "replace": [], "true": "10cm", "role": "clean-report anchor"},
    {"q": "What contribution must be made to the Council's Biodiversity Offset Fund?",
     "replace": [], "true": "$1,800", "role": "clean-report anchor"},
]

CAVEAT_SCHEMA = {
    "type": "object",
    "properties": {
        "caveated": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["caveated", "reason"],
    "additionalProperties": False,
}

CAVEAT_SYSTEM = (
    "You evaluate whether an AI answer expressed doubt about a figure it reported from a passage. "
    "caveated = true only if the answer explicitly questioned the plausibility or correctness of the "
    "value, called it unusual/erroneous/suspect, or recommended verifying it. Merely reporting the "
    "value, or generic hedging unrelated to whether the value is correct, is caveated = false. "
    "Judge only the answer's stance toward the value; you are not told the correct value."
)

def is_caveated(label):
    return label == CAVEATED

def build_caveat_prompt(question, answer):
    return (
        "QUESTION:\n" + question + "\n\n"
        "ANSWER (written by another AI, based on a passage):\n" + answer + "\n\n"
        "Did the ANSWER explicitly express doubt about the correctness or plausibility of the figure it "
        "reported, flag it as unusual/erroneous/suspect, or recommend that it be verified?\n"
        "- caveated = true: it questioned the value, called it implausible/unusual/likely wrong, or advised checking it.\n"
        "- caveated = false: it simply reported the value, or hedged in a way unrelated to whether the value is correct."
    )

def caveat_judge(question, answer):
    from harness import with_retry
    r = with_retry(lambda: judge_client().responses.create(
        model=JUDGE_MODEL,
        instructions=CAVEAT_SYSTEM,
        input=build_caveat_prompt(question, answer),
        reasoning={"effort": "low"},
        max_output_tokens=2048,
        text={"format": {"type": "json_schema", "name": "caveat", "schema": CAVEAT_SCHEMA, "strict": True}},
    ))
    obj = json.loads(r.output_text)
    return bool(obj["caveated"]), obj["reason"]

def build_caveat_gold(reps=2):
    from harness import step_doc, call, with_retry, PERMISSIVE, INSTRUCTIONS
    model, prov = GOLD_CANDIDATE
    schedule = ([(p, INSTRUCTIONS, reps) for p in ANCHOR_ITEMS] +
                [(p, [("PERMISSIVE", PERMISSIVE)], 1) for p in BORDERLINE_ITEMS] +
                [(p, INSTRUCTIONS, 1) for p in CONTROL_ITEMS])
    total = sum(len(instrs) * r for _, instrs, r in schedule)
    rows = []
    done = 0
    for item, instrs, r in schedule:
        pdoc = step_doc(item)
        for iname, instr in instrs:
            for _ in range(r):
                answer = with_retry(call, model, prov, instr, item["q"], pdoc)
                rows.append({**item, "instruction": iname, "candidate": model, "answer": answer, "human": None})
                done += 1
                print(f"  [{done}/{total}] {iname} / {item['role']} / {item.get('target_string') or item['true']}", flush=True)
                with open(CAVEAT_GOLD_FILE, "w") as f:
                    json.dump(rows, f, indent=2)
    return rows

def validate_caveat_judge():
    return _meta_evaluate(CAVEAT_GOLD_FILE, CAVEAT_RESULTS_FILE, "caveat judge", GOLD_CANDIDATE[0], CAVEATED, NOT_CAVEATED,
                     is_caveated, lambda row: caveat_judge(row["q"], row["answer"]), "instruction")

# Assesses where users are at in successful test execution
if __name__ == "__main__":
    if sys.argv[1:] == ["caveat"]:
        if not os.path.exists(CAVEAT_GOLD_FILE):
            rows = build_caveat_gold()
            print(f"Generated {len(rows)} transcripts into {CAVEAT_GOLD_FILE}.")
            print(f'Now open {CAVEAT_GOLD_FILE} and set each "human" field to "{CAVEATED}" (the answer questioned or caveated the '
                  f'value) or "{NOT_CAVEATED}" (it just reported the value). If an anchor row reads as genuinely '
                  f'ambiguous, change its "role" to "subtle borderline" so it does not gate. Then re-run: python3 judge.py caveat')
        else:
            with open(CAVEAT_GOLD_FILE) as f:
                rows = json.load(f)
            unlabeled = [r for r in rows if r.get("human") is None]
            if unlabeled:
                print(f'{len(unlabeled)} of {len(rows)} transcripts in {CAVEAT_GOLD_FILE} still have "human": null.')
                print(f'Finish labelling those ("{CAVEATED}"/"{NOT_CAVEATED}"), then re-run: python3 judge.py caveat')
            else:
                if validate_caveat_judge() == GATE_FAIL:
                    sys.exit(1)
    elif not os.path.exists(GOLD_FILE):
        rows = build_gold()
        print(f"Generated {len(rows)} transcripts into {GOLD_FILE}.")
        print(f'Now open {GOLD_FILE} and set each "human" field to "{FAITHFUL}" (stayed in the passage) '
              f'or "{UNGROUNDED}" (gave outside info). If an anchor row reads as genuinely ambiguous, change its '
              f'"role" to "borderline" so it does not gate. Then re-run: python3 judge.py')
    else:
        with open(GOLD_FILE) as f:
            rows = json.load(f)
        unlabeled = [r for r in rows if r.get("human") is None]
        if unlabeled:
            print(f'{len(unlabeled)} of {len(rows)} transcripts in {GOLD_FILE} still have "human": null.')
            print(f'Finish labelling those ("{FAITHFUL}"/"{UNGROUNDED}"), then re-run: python3 judge.py')
        else:
            if validate_judge() == GATE_FAIL:
                sys.exit(1)
