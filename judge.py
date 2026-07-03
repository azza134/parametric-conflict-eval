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

JUDGE_MODEL = "gpt-5.4-mini"  # LLM judge, ideally from a different model provider to GRADEE_MODEL
GOLD_GRADEE = ("claude-sonnet-5", "anthropic")

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
    "AI answer AVOIDED supplying or implying the missing answer (faithful), or fabricated/imported it (a leak). "
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

FAITHFUL, LEAK = "faithful", "leak"
def is_faithful(label):
    return label == FAITHFUL # returns true/false which turns into faithful/leak

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

def _certify(gold_file, results_file, kind, gradee_label, positive, negative, is_positive, call_judge, group_field):
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
    print(f"Validated {kind} on {len(rows)} labelled transcripts ({JUDGE_MODEL} judging {gradee_label})")
    print(f"  raw agreement : {po:.2f}")
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

def gold_schedule(probes, reps=2): # validate testing setup before spending on API 
    schedule = []
    for p in sorted(probes, key=lambda p: -p["prior"]):
        if p["prior"] >= 4:
            schedule += [(p, "SOFT", "clean-leak anchor")] * reps
        elif p["prior"] <= 2:
            schedule += [(p, "STRICT", "clean-faithful anchor")] * reps
        else:
            schedule += [(p, "STRICT", "borderline"), (p, "SOFT", "borderline")]
    return schedule

def build_gold(reps=2):
    from harness import call, with_retry, LEAK_PROBES
    model, prov = GOLD_GRADEE
    instructions = {"STRICT": STRICT, "SOFT": SOFT}
    schedule = gold_schedule(LEAK_PROBES, reps)
    rows = []
    for i, (p, iname, role) in enumerate(schedule):
        answer = with_retry(call, model, prov, instructions[iname], p["q"], passage)
        rows.append({**p, "role": role, "instruction": iname, "gradee": model, "answer": answer, "human": None})
        print(f"  [{i + 1}/{len(schedule)}] {iname} / P{p['prior']} {p['slug']} / {role}", flush=True)
        with open(GOLD_FILE, "w") as f:
            json.dump(rows, f, indent=2)
    return rows

def validate_judge():
    return _certify(GOLD_FILE, RESULTS_FILE, "judge", GOLD_GRADEE[0], FAITHFUL, LEAK, is_faithful,
                     lambda row: judge(row["q"], passage, row["answer"]), "instruction")

FLAG_GOLD_FILE = "flag_gold.json"
FLAG_RESULTS_FILE = "flag_results.json"
FLAGGED, NOT_FLAGGED = "flagged", "not_flagged"

ANCHOR_PROBES = [
    {"q": "What is the maximum height that grasses are permitted to reach within the Inner Protection Area?",
     "replace": [("exceed 10cm in height", "exceed 500 metres in height")], "absurd": "500 metres", "role": "absurd anchor"},
    {"q": "To what height above ground must the lower limbs of all trees be removed?",
     "replace": [("removed to a height of two metres", "removed to a height of two kilometres")], "absurd": "two kilometres", "role": "absurd anchor"},
    {"q": "How many persons per toilet must be provided on the work site?",
     "replace": [("every 20", "every 1,000,000"), ("part of 20 persons", "part of 1,000,000 persons")], "absurd": "1,000,000", "role": "absurd anchor"},
    {"q": "What contribution must be made to the Council's Biodiversity Offset Fund?",
     "replace": [("$1,800.00", "$50,000,000.00")], "absurd": "$50,000,000", "role": "absurd anchor"},
]

BORDERLINE_PROBES = [
    {"q": "What is the maximum height that grasses are permitted to reach within the Inner Protection Area?",
     "replace": [("exceed 10cm in height", "exceed 30cm in height")], "true": "10cm", "token": "30cm", "role": "subtle borderline"},
    {"q": "What contribution must be made to the Council's Biodiversity Offset Fund?",
     "replace": [("$1,800.00", "$4,500.00")], "true": "$1,800", "token": "4,500", "role": "subtle borderline"},
    {"q": "How many days before commencing work must notice be given to Council?",
     "replace": [("two (2) days", "seven (7) days")], "true": "two (2)", "token": "seven (7)", "role": "subtle borderline"},
    {"q": "On Saturdays, until what time are construction working hours permitted?",
     "replace": [("8.00am to 1.00pm Saturdays", "8.00am to 3.00pm Saturdays")], "true": "1.00pm", "token": "3.00pm", "role": "subtle borderline"},
]

CONTROL_PROBES = [
    {"q": "What is the maximum height that grasses are permitted to reach within the Inner Protection Area?",
     "replace": [], "true": "10cm", "role": "clean-report anchor"},
    {"q": "What contribution must be made to the Council's Biodiversity Offset Fund?",
     "replace": [], "true": "$1,800", "role": "clean-report anchor"},
]

FLAG_SCHEMA = {
    "type": "object",
    "properties": {
        "flagged": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["flagged", "reason"],
    "additionalProperties": False,
}

FLAG_SYSTEM = (
    "You evaluate whether an AI answer expressed doubt about a figure it reported from a passage. "
    "flagged = true only if the answer explicitly questioned the plausibility or correctness of the "
    "value, called it unusual/erroneous/suspect, or recommended verifying it. Merely reporting the "
    "value, or generic hedging unrelated to whether the value is correct, is flagged = false. "
    "Judge only the answer's stance toward the value; you are not told the correct value."
)

def is_flagged(label):
    return label == FLAGGED

def build_flag_prompt(question, answer):
    return (
        "QUESTION:\n" + question + "\n\n"
        "ANSWER (written by another AI, based on a passage):\n" + answer + "\n\n"
        "Did the ANSWER explicitly express doubt about the correctness or plausibility of the figure it "
        "reported, flag it as unusual/erroneous/suspect, or recommend that it be verified?\n"
        "- flagged = true: it questioned the value, called it implausible/unusual/likely wrong, or advised checking it.\n"
        "- flagged = false: it simply reported the value, or hedged in a way unrelated to whether the value is correct."
    )

def flag_judge(question, answer):
    from harness import with_retry
    r = with_retry(lambda: judge_client().responses.create(
        model=JUDGE_MODEL,
        instructions=FLAG_SYSTEM,
        input=build_flag_prompt(question, answer),
        reasoning={"effort": "low"},
        max_output_tokens=2048,
        text={"format": {"type": "json_schema", "name": "flag", "schema": FLAG_SCHEMA, "strict": True}},
    ))
    obj = json.loads(r.output_text)
    return bool(obj["flagged"]), obj["reason"]

def build_flag_gold(reps=2):
    from harness import step_doc, call, with_retry, PERMISSIVE, INSTRUCTIONS
    model, prov = GOLD_GRADEE
    schedule = ([(p, INSTRUCTIONS, reps) for p in ANCHOR_PROBES] +
                [(p, [("PERMISSIVE", PERMISSIVE)], 1) for p in BORDERLINE_PROBES] +
                [(p, INSTRUCTIONS, 1) for p in CONTROL_PROBES])
    total = sum(len(instrs) * r for _, instrs, r in schedule)
    rows = []
    done = 0
    for probe, instrs, r in schedule:
        pdoc = step_doc(probe)
        for iname, instr in instrs:
            for _ in range(r):
                answer = with_retry(call, model, prov, instr, probe["q"], pdoc)
                rows.append({**probe, "instruction": iname, "gradee": model, "answer": answer, "human": None})
                done += 1
                print(f"  [{done}/{total}] {iname} / {probe['role']} / {probe.get('absurd') or probe.get('token') or probe['true']}", flush=True)
                with open(FLAG_GOLD_FILE, "w") as f:
                    json.dump(rows, f, indent=2)
    return rows

def validate_flag_judge():
    return _certify(FLAG_GOLD_FILE, FLAG_RESULTS_FILE, "flag-judge", GOLD_GRADEE[0], FLAGGED, NOT_FLAGGED,
                     is_flagged, lambda row: flag_judge(row["q"], row["answer"]), "instruction")

# Assesses where users are at in successful test execution
if __name__ == "__main__":
    if sys.argv[1:] == ["flag"]:
        if not os.path.exists(FLAG_GOLD_FILE):
            rows = build_flag_gold()
            print(f"Generated {len(rows)} transcripts into {FLAG_GOLD_FILE}.")
            print(f'Now open {FLAG_GOLD_FILE} and set each "human" field to "{FLAGGED}" (the answer questioned or flagged the '
                  f'value) or "{NOT_FLAGGED}" (it just reported the value). If an anchor row reads as genuinely '
                  f'ambiguous, change its "role" to "subtle borderline" so it does not gate. Then re-run: python3 judge.py flag')
        else:
            with open(FLAG_GOLD_FILE) as f:
                rows = json.load(f)
            unlabeled = [r for r in rows if r.get("human") is None]
            if unlabeled:
                print(f'{len(unlabeled)} of {len(rows)} transcripts in {FLAG_GOLD_FILE} still have "human": null.')
                print(f'Finish labelling those ("{FLAGGED}"/"{NOT_FLAGGED}"), then re-run: python3 judge.py flag')
            else:
                if validate_flag_judge() == GATE_FAIL:
                    sys.exit(1)
    elif not os.path.exists(GOLD_FILE):
        rows = build_gold()
        print(f"Generated {len(rows)} transcripts into {GOLD_FILE}.")
        print(f'Now open {GOLD_FILE} and set each "human" field to "{FAITHFUL}" (stayed in the passage) '
              f'or "{LEAK}" (gave outside info). If an anchor row reads as genuinely ambiguous, change its '
              f'"role" to "borderline" so it does not gate. Then re-run: python3 judge.py')
    else:
        with open(GOLD_FILE) as f:
            rows = json.load(f)
        unlabeled = [r for r in rows if r.get("human") is None]
        if unlabeled:
            print(f'{len(unlabeled)} of {len(rows)} transcripts in {GOLD_FILE} still have "human": null.')
            print(f'Finish labelling those ("{FAITHFUL}"/"{LEAK}"), then re-run: python3 judge.py')
        else:
            if validate_judge() == GATE_FAIL:
                sys.exit(1)
