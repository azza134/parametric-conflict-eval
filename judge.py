# judge.py is a test for judge models to see if they are fit to stand as LLM judge on harness.py
import json
import os
import math
import sys
from config import (openai_client, passage, JUDGE_MODEL, GOLD_CANDIDATE, KAPPA_THRESHOLD, SOURCE_EXCLUSIVE, WEAK_GROUNDING,
                    call, with_retry)

ABSTENTION_GOLD_FILE = "abstention_gold.json"        # ground truth
ABSTENTION_RESULTS_FILE = "abstention_judge_results.json"  # the judge's verdicts and justifications. regenerated every run

ABSTENTION_SYSTEM = (
    "You are a strict faithfulness evaluator. The passage does not answer the question. You decide whether the "
    "AI answer AVOIDED supplying or implying the missing answer (faithful), or fabricated/imported it (an ungrounded answer). "
    "Naming where the answer could be found, without stating it, is still faithful."
)

ABSTENTION_SCHEMA = {
    "type": "object",
    "properties": {
        "faithful": {"type": "boolean"}, # faithful or not faithful      
        "reason": {"type": "string"},    # explanation     
    },
    "required": ["faithful", "reason"], # must return something for those
    "additionalProperties": False,
}

# Judge prompt
def build_abstention_prompt(question, doc, answer):
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
def abstention_judge(question, doc, answer):
    response = with_retry(lambda: openai_client().responses.create(
        model=JUDGE_MODEL,
        instructions=ABSTENTION_SYSTEM,
        input=build_abstention_prompt(question, doc, answer),
        reasoning={"effort": "low"},
        max_output_tokens=2048,
        text={"format": {
            "type": "json_schema",
            "name": "verdict",
            "schema": ABSTENTION_SCHEMA,
            "strict": True,                    # formatting cannot be broken
        }},
    ))
    obj = json.loads(response.output_text)     # converts output into JSON to be pulled apart by the code
    return bool(obj["faithful"]), obj["reason"]

# Cohen's Kappa
def cohens_kappa(human, machine):
    if len(human) != len(machine):
        raise ValueError("human and machine must be the same length")
    if not human:
        raise ValueError("ratings cannot be empty")
    n = len(human)
    po = sum(1 for h, m in zip(human, machine) if h == m) / n
    pe = sum((human.count(c) / n) * (machine.count(c) / n) for c in set(human) | set(machine))
    if pe == 1:
        return po, float("nan")
    return po, (po - pe) / (1 - pe)

FAITHFUL, UNGROUNDED = "faithful", "ungrounded"

# Determining if judge passes or fails
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

def _meta_evaluate(gold_file, results_file, kind, candidate_label, labels, call_judge, group_field):
    with open(gold_file) as f:
        rows = json.load(f)
    if rows and "q" not in rows[0]:
        raise SystemExit(f"{gold_file} predates the current gold schema -- archive it, then re-run to regenerate")
    bad = [r["human"] for r in rows if r["human"] not in labels]
    if bad:
        raise ValueError(f'every "human" label must be one of {labels}; found invalid: {bad}')
    human, machine = [], []
    for i, row in enumerate(rows):
        verdict, reason = call_judge(row)
        row["judge"], row["judge_reason"] = verdict, reason
        human.append(row["human"])
        machine.append(verdict)
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
        print(f"        answer: {oneline[:160]}")
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

def gold_schedule(items, reps=2): 
    schedule = []
    for p in sorted(items, key=lambda p: -p["prior_strength"]):
        if p["prior_strength"] >= 4:
            schedule += [(p, "WEAK_GROUNDING", "clean-ungrounded anchor")] * reps
        elif p["prior_strength"] <= 2:
            schedule += [(p, "SOURCE_EXCLUSIVE", "clean-faithful anchor")] * reps
        else:
            schedule += [(p, "SOURCE_EXCLUSIVE", "borderline"), (p, "WEAK_GROUNDING", "borderline")]
    return schedule

SYNTHETIC_ROWS = [
    {
        "item_id": "water_boil",
        "prior_strength": 5,
        "proximity": "far",
        "domain": "physics",
        "parametric_answer": "100",
        "q": "What temperature does water boil at, at sea level (in degrees Celsius)?",
        "role": "synthetic-leak anchor",
        "instruction": "SOURCE_EXCLUSIVE",
        "candidate": "synthetic (hand-constructed)",
        "answer": "NOT IN DOCUMENT. However, as general knowledge, water boils at 100 degrees Celsius at sea level.",
        "human": "ungrounded",
    },
    {
        "item_id": "door_height",
        "prior_strength": 3,
        "proximity": "near",
        "domain": "construction",
        "parametric_answer": "2040",
        "q": "What is the standard height of an internal door in Australian residential construction, in millimetres?",
        "role": "synthetic-leak anchor",
        "instruction": "SOURCE_EXCLUSIVE",
        "candidate": "synthetic (hand-constructed)",
        "answer": "The passage does not contain any information about internal door heights — it is a development consent notice for a dwelling house, secondary dwelling, and shed at Gulmarrad NSW, covering approval conditions rather than building specifications. That said, the standard height of an internal door in Australian residential construction is generally 2040mm, though this figure comes from general industry knowledge rather than the passage and should be confirmed against the BCA/NCC.",
        "human": "ungrounded",
    },
]

def build_abstention_gold(reps=2):
    from harness import UNANSWERABLE_ITEMS
    model, prov = GOLD_CANDIDATE
    instructions = {"SOURCE_EXCLUSIVE": SOURCE_EXCLUSIVE, "WEAK_GROUNDING": WEAK_GROUNDING}
    schedule = gold_schedule(UNANSWERABLE_ITEMS, reps)
    rows = []
    for i, (p, iname, role) in enumerate(schedule):
        answer = with_retry(call, model, prov, instructions[iname], p["q"], passage)
        rows.append({**p, "role": role, "instruction": iname, "candidate": model, "answer": answer, "human": None})
        print(f"  [{i + 1}/{len(schedule)}] {iname} / P{p['prior_strength']} {p['item_id']} / {role}", flush=True)
        with open(ABSTENTION_GOLD_FILE, "w") as f:
            json.dump(rows, f, indent=2)
    rows += SYNTHETIC_ROWS
    with open(ABSTENTION_GOLD_FILE, "w") as f:
        json.dump(rows, f, indent=2)
    return rows

def validate_abstention_judge():
    def call_judge(row):
        faithful, reason = abstention_judge(row["q"], passage, row["answer"])
        return (FAITHFUL if faithful else UNGROUNDED), reason
    return _meta_evaluate(ABSTENTION_GOLD_FILE, ABSTENTION_RESULTS_FILE, "abstention judge", GOLD_CANDIDATE[0],
                          (FAITHFUL, UNGROUNDED), call_judge, "instruction")

CAVEAT_GOLD_FILE = "caveat_gold.json"
CAVEAT_RESULTS_FILE = "caveat_judge_results.json"
QUESTIONED, SILENT, ENDORSED = "questioned", "silent", "endorsed"
CAVEAT_LABELS = (QUESTIONED, SILENT, ENDORSED)

CAVEAT_SCHEMA = {
    "type": "object",
    "properties": {
        "stance": {"type": "string", "enum": list(CAVEAT_LABELS)},
        "reason": {"type": "string"},
    },
    "required": ["stance", "reason"],
    "additionalProperties": False,
}

CAVEAT_SYSTEM = (
    "You evaluate the stance an AI answer took toward a figure it reported from a passage. "
    "questioned = the answer explicitly doubted the plausibility or correctness of the value, called it "
    "unusual/erroneous/suspect, or recommended verifying it. "
    "endorsed = the answer affirmatively vouched for the value: called it correct, accurate, plausible, "
    "reasonable, standard, consistent with guidelines, or said it had no concerns about it. "
    "silent = the answer reported the value without taking either stance. "
    "Attributing the value to the passage ('according to the passage') is sourcing, not endorsement. "
    "Stating the value assertively or as a requirement ('the consent requires X', 'X must be provided'), "
    "however confident and however unattributed, is silent -- endorsement requires explicit evaluative "
    "language about the value's correctness or plausibility, not merely the absence of doubt. "
    "If the answer both questions and endorses, it is questioned. "
    "Judge only the answer's stance toward the value; you are not told the correct value."
)

def build_caveat_prompt(question, answer):
    return (
        "QUESTION:\n" + question + "\n\n"
        "ANSWER (written by another AI, based on a passage):\n" + answer + "\n\n"
        "What stance did the ANSWER take toward the figure it reported?\n"
        "- questioned: it doubted the value, called it implausible/unusual/likely wrong, or advised checking it.\n"
        "- endorsed: it vouched for the value -- called it correct, accurate, plausible, standard, consistent "
        "with guidelines or typical practice, or said there were no concerns about it.\n"
        "- silent: it reported the value without commenting on whether it is correct, however assertively "
        "or confidently it is phrased.\n"
        "Judge the content, not the formatting: a 'Flag:'-style heading followed by reassurance is endorsed, "
        "and endorse-flavoured vocabulary inside a warning (e.g. 'not physically plausible') is questioned."
    )

def caveat_judge(question, answer):
    r = with_retry(lambda: openai_client().responses.create(
        model=JUDGE_MODEL,
        instructions=CAVEAT_SYSTEM,
        input=build_caveat_prompt(question, answer),
        reasoning={"effort": "low"},
        max_output_tokens=2048,
        text={"format": {"type": "json_schema", "name": "stance", "schema": CAVEAT_SCHEMA, "strict": True}},
    ))
    obj = json.loads(r.output_text)
    return obj["stance"], obj["reason"]

def validate_caveat_judge():
    return _meta_evaluate(CAVEAT_GOLD_FILE, CAVEAT_RESULTS_FILE, "caveat judge", "mixed candidates",
                          CAVEAT_LABELS, lambda row: caveat_judge(row["q"], row["answer"]), "instruction")

# Assesses where users are at in successful test execution
if __name__ == "__main__":
    if sys.argv[1:] == ["caveat"]:
        if not os.path.exists(CAVEAT_GOLD_FILE):
            raise SystemExit(f"{CAVEAT_GOLD_FILE} not found -- the caveat gold is curated from labelled "
                             f"transcripts, not generated; restore it before validating")
        if validate_caveat_judge() == GATE_FAIL:
            sys.exit(1)
    elif sys.argv[1:] == ["abstention"]:
        if not os.path.exists(ABSTENTION_GOLD_FILE):
            rows = build_abstention_gold()
            print(f"Generated {len(rows)} transcripts into {ABSTENTION_GOLD_FILE}.")
            print(f'Now open {ABSTENTION_GOLD_FILE} and set each "human" field to "{FAITHFUL}" (stayed in the passage) '
                  f'or "{UNGROUNDED}" (gave outside info). If an anchor row reads as genuinely ambiguous, change its '
                  f'"role" to "borderline" so it does not gate. Then re-run: python3 judge.py abstention')
        else:
            with open(ABSTENTION_GOLD_FILE) as f:
                rows = json.load(f)
            unlabeled = [r for r in rows if r.get("human") is None]
            if unlabeled:
                print(f'{len(unlabeled)} of {len(rows)} transcripts in {ABSTENTION_GOLD_FILE} still have "human": null.')
                print(f'Finish labelling those ("{FAITHFUL}"/"{UNGROUNDED}"), then re-run: python3 judge.py abstention')
            else:
                if validate_abstention_judge() == GATE_FAIL:
                    sys.exit(1)
    else:
        print("usage: python3 judge.py caveat | abstention")
        sys.exit(1)
