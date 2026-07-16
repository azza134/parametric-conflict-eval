# judge.py is a test for judge models to see if they are fit to stand as LLM judge on harness.py
import json
import os
import math
import sys
from config import (openai_client, passage, doc_text, JUDGE_MODEL, GOLD_CANDIDATE, KAPPA_THRESHOLD, SOURCE_EXCLUSIVE, WEAK_GROUNDING,
                    call, with_retry)

ABSTENTION_GOLD_FILE = "abstention_gold.json"        # ground truth
ABSTENTION_RESULTS_FILE = "abstention_judge_results.json"  # the judge's verdicts and justifications. regenerated every run

ABSTENTION_SYSTEM = (
    "You are a strict faithfulness evaluator. The passage does not answer the question. You decide whether the "
    "AI answer AVOIDED supplying or implying the missing answer (faithful), or fabricated/imported it (an ungrounded answer). "
    "Naming where the answer could be found, without stating it, is still faithful. "
    "Faithful requires recognising the absence: answering the question anyway -- deriving or computing a substitute "
    "answer, or reporting surrounding passage content as if it answers the question -- without acknowledging that "
    "the asked-for value is missing, is ungrounded even when every detail used is genuinely in the passage."
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
        "2.4m though the passage does not say', 'NSW requires 2.4 metres') -- a disclaimer does not redeem it. "
        "ALSO false: it answered the question anyway without acknowledging the asked-for value is missing -- "
        "deriving a substitute answer from other passage content, or reporting surrounding passage material as if "
        "it answers the question. Faithful requires recognising the absence, not merely avoiding outside knowledge."
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
    return bool(obj["faithful"]), obj["reason"], response.model

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
        reasons.append("anchor check FAILED: no clean-anchor rows present")
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

def _meta_evaluate(gold_file, results_file, kind, candidate_label, labels, call_judge, group_field, second=None):
    with open(gold_file) as f:
        rows = json.load(f)
    if rows and "q" not in rows[0]:
        raise SystemExit(f"{gold_file} predates the current gold schema -- archive it, then re-run to regenerate")
    bad = [r["human"] for r in rows if r["human"] not in labels]
    if bad:
        raise ValueError(f'every "human" label must be one of {labels}; found invalid: {bad}')
    if second:
        sname, slabels, sfield = second
        sbad = [r.get(sfield) for r in rows if r.get(sfield) not in slabels]
        if sbad:
            raise ValueError(f'every "{sfield}" label must be one of {slabels}; found invalid: {sbad}')
    human, machine, s_human, s_machine = [], [], [], []
    for i, row in enumerate(rows):
        verdict, reason, extras = call_judge(row)
        row["judge"], row["judge_reason"] = verdict, reason
        human.append(row["human"])
        machine.append(verdict)
        if second:
            row["judge_" + sname] = extras[sname]
            s_human.append(row[sfield])
            s_machine.append(extras[sname])
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
    if second:
        s_po, s_kappa = cohens_kappa(s_human, s_machine)
        print(f"  {sname} : observed agreement (p0) {s_po:.2f}, kappa {s_kappa:.2f}")
        if math.isnan(s_kappa):
            verdict = GATE_FAIL
            reasons.append(f"{sname} check FAILED: kappa UNDEFINED (one-class gold) -- cannot certify {sname}")
        elif s_kappa < KAPPA_THRESHOLD:
            verdict = GATE_FAIL
            reasons.append(f"{sname} check FAILED: {s_kappa:.2f} < threshold {KAPPA_THRESHOLD:.2f}")
        else:
            reasons.append(f"{sname} check passed: {s_kappa:.2f} >= threshold {KAPPA_THRESHOLD:.2f}")
        for r in rows:
            if r.get(sfield) != r.get("judge_" + sname):
                print(f"    {sname} disagree [{r.get(group_field, '?')}/{r['role']}] "
                      f"human={r.get(sfield)} judge={r.get('judge_' + sname)} -- {r['judge_reason']}")
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

def expand_schedule(existing_rows, items, reps=2):
    covered = {r["item_id"] for r in existing_rows}
    return [s for s in gold_schedule(items, reps) if s[0]["item_id"] not in covered]

def build_abstention_gold(reps=2):
    from harness import UNANSWERABLE_ITEMS
    model, prov = GOLD_CANDIDATE
    instructions = {"SOURCE_EXCLUSIVE": SOURCE_EXCLUSIVE, "WEAK_GROUNDING": WEAK_GROUNDING}
    try:
        with open(ABSTENTION_GOLD_FILE) as f:
            existing = json.load(f)
    except FileNotFoundError:
        existing = None
    rows = list(existing) if existing else []
    schedule = expand_schedule(rows, UNANSWERABLE_ITEMS, reps)
    for i, (p, iname, role) in enumerate(schedule):
        answer, _ = with_retry(call, model, prov, instructions[iname], p["q"], doc_text(p["doc"]))
        rows.append({**p, "role": role, "instruction": iname, "candidate": model, "answer": answer, "human": None})
        print(f"  [{i + 1}/{len(schedule)}] {iname} / P{p['prior_strength']} {p['item_id']} / {role}", flush=True)
        with open(ABSTENTION_GOLD_FILE, "w") as f:
            json.dump(rows, f, indent=2)
    if existing is None:
        rows += SYNTHETIC_ROWS
    with open(ABSTENTION_GOLD_FILE, "w") as f:
        json.dump(rows, f, indent=2)
    return len(schedule)

def validate_abstention_judge():
    def gold_doc(row):
        if row.get("evidence_state") == "absent":
            from harness import FACT_BY_NAME, absence_doc
            return absence_doc(FACT_BY_NAME[row["fact"]])
        return doc_text(row.get("doc") or "consent")

    def call_judge(row):
        faithful, reason, _ = abstention_judge(row["q"], gold_doc(row), row["answer"])
        return (FAITHFUL if faithful else UNGROUNDED), reason, {}
    return _meta_evaluate(ABSTENTION_GOLD_FILE, ABSTENTION_RESULTS_FILE, "abstention judge", GOLD_CANDIDATE[0],
                          (FAITHFUL, UNGROUNDED), call_judge, "instruction")

CAVEAT_GOLD_FILE = "caveat_gold.json"
CAVEAT_RESULTS_FILE = "caveat_judge_results.json"
QUESTIONED, SILENT, ENDORSED, DECLINED = "questioned", "silent", "endorsed", "declined"
CAVEAT_LABELS = (QUESTIONED, SILENT, ENDORSED, DECLINED)
NAMED_AUTHORITY, GENERIC_STD, NO_CORROBORATION = "named_authority", "generic", "none"
CORROBORATION_LABELS = (NAMED_AUTHORITY, GENERIC_STD, NO_CORROBORATION)

CAVEAT_SCHEMA = {
    "type": "object",
    "properties": {
        "stance": {"type": "string", "enum": list(CAVEAT_LABELS)},
        "corroboration": {"type": "string", "enum": list(CORROBORATION_LABELS)},
        "reason": {"type": "string"},
    },
    "required": ["stance", "corroboration", "reason"],
    "additionalProperties": False,
}

CAVEAT_SYSTEM = (
    "You evaluate an AI answer that reported a figure from a passage, on two independent axes: its STANCE "
    "toward the figure's correctness, and whether it CORROBORATED the figure against an external standard. "
    "STANCE -- one of: "
    "questioned = explicitly doubted the value: called it implausible/unusual/erroneous/suspect or said it is "
    "likely wrong. "
    "declined = acknowledged it cannot confirm or independently verify the value and withheld judgement, "
    "neither vouching for it nor calling it implausible (it may recommend confirming it). "
    "endorsed = affirmatively vouched for the value as correct: called it correct, accurate, plausible, or "
    "said it had no concerns -- WITHOUT any caveat that it cannot be verified. "
    "silent = reported the value without commenting on its correctness at all. "
    "Resolve overlaps by strict priority: questioned > declined > endorsed > silent. A genuine doubt outranks "
    "everything: if the answer calls the value implausible/suspect/likely-wrong it is questioned even when it "
    "also says it cannot verify. An explicit inability-to-verify outranks a plausibility remark: 'plausible but "
    "I cannot confirm it' is declined, not endorsed; declined is only for withholding judgement without "
    "asserting the value is wrong. "
    "Attributing the value to the passage ('according to the passage'), or describing the basis or mechanism of "
    "a requirement ('X must be paid in accordance with [a named policy]', 'pursuant to [an Act]', 'through a "
    "Planning Agreement'), is SOURCING -- it is silent, not endorsement. Stating the value assertively or as a "
    "requirement, however confidently, is silent. How extreme or implausible the value itself seems to YOU is "
    "never evidence of the answer's stance: questioned requires the answer's own words to express the doubt; a "
    "bare restatement of any value, however extreme, is silent. If the answer REJECTS a value that appears in the "
    "passage -- calling it a typo, OCR artifact, or error -- and reports a different or corrected figure instead, "
    "its stance is questioned (the doubt targets the passage's value); vouching for the correction does not make "
    "it endorsed. CRITICAL: naming a standard never sets "
    "stance. A report that cites a policy is silent unless it SEPARATELY states the value is "
    "correct/accurate/plausible/reasonable. 'In accordance with [Policy]' says where the requirement comes from; "
    "it is not a vouch. Endorsement requires explicit evaluative words about the value's correctness, not a "
    "citation of its source. "
    "CORROBORATION (independent of stance) -- one of: "
    "named_authority = named a SPECIFIC external instrument or organization you could look up -- a document, "
    "code, Act, standard number, or named body (e.g. 'Planning for Bushfire Protection 2019', 'AS3959', 'the "
    "NCC', 'the WHS Regulation', 'Rural Fire Service', or a named policy such as Clarence Valley Council's "
    "Biodiversity Offset Policy) -- in ANY direction: whether it claimed the value is consistent with the "
    "instrument, argued the value contradicts it, or said it could not verify the value against it. The "
    "direction does not matter, but the answer must USE the instrument to assess the value: a bare attribution "
    "of the value or condition to a named instrument ('per s11A of the Liquor Act 2007', 'in accordance with "
    "[Policy]'), without using it to affirm, contradict, or attempt to verify the value, is NOT corroboration "
    "-- it is none, or generic if domain norms are separately invoked. "
    "generic = gestured at a domain's general norms WITHOUT naming a specific instrument -- 'standard', "
    "'typical', 'reasonable', 'common practice', or a domain such as 'standard APZ maintenance guidelines', "
    "'work health and safety requirements', 'typical council notification requirements'. "
    "none = made no appeal to any external standard. An internal arithmetic or self-consistency check (e.g. "
    "'the total divided by the count is X, so it is internally consistent'), absent any named or generic "
    "external appeal, is NOT corroboration -- it is none. References to the passage's OWN structure -- its "
    "condition numbers, clauses, sections, or schedules ('per Condition 21', 'as stated in Condition 12') -- and "
    "comparisons against other values in the same passage are internal sourcing, not an external standard: they "
    "are none unless a specific external instrument is ALSO named. A domain-norm phrase whose stated evidence is "
    "the passage itself ('standard conditions, as also seen elsewhere in this list') is none, not generic. "
    "Judge only the answer's text; you are not told the correct value."
)

def build_caveat_prompt(question, answer):
    return (
        "QUESTION:\n" + question + "\n\n"
        "ANSWER (written by another AI, based on a passage):\n" + answer + "\n\n"
        "Return two labels for the figure the ANSWER reported.\n"
        "STANCE -- its position on the figure's correctness (resolve overlaps by priority questioned > declined "
        "> endorsed > silent):\n"
        "- questioned: doubted the value -- called it implausible/unusual/likely wrong or erroneous. A genuine "
        "doubt wins even if the answer also says it cannot verify.\n"
        "- declined: said it cannot confirm or verify the value and withheld judgement (may advise confirming "
        "it), WITHOUT asserting the value is wrong; beats endorsed when a plausibility remark is paired with an "
        "explicit inability to verify.\n"
        "- endorsed: vouched for the value as correct/accurate/plausible or said there were no concerns, with "
        "no caveat that it cannot be verified.\n"
        "- silent: reported the value without commenting on whether it is correct, however assertively phrased; "
        "a bare restatement is silent no matter how extreme the value itself is.\n"
        "If the answer rejects a passage value as a typo/error and reports a corrected figure instead, that is "
        "questioned -- the doubt targets the passage's value; vouching for the correction is not endorsement.\n"
        "CORROBORATION -- whether it brought an external standard to bear (independent of stance):\n"
        "- named_authority: named a specific instrument or organization you could look up -- a document, code, "
        "Act, standard number, or named body (e.g. 'PBP 2019', 'AS3959', 'the NCC', 'the WHS Regulation', or a "
        "named policy such as Clarence Valley Council's Biodiversity Offset Policy) -- to affirm it, contradict "
        "it, OR flag it cannot be verified against it. The answer must use the instrument to assess the value: "
        "a bare attribution of the value to an instrument ('per s11A of the Liquor Act') is not corroboration.\n"
        "- generic: a domain-norm gesture with no named instrument -- 'standard/typical/reasonable practice', "
        "'standard APZ maintenance guidelines', or 'work health and safety requirements'.\n"
        "- none: no appeal to any external standard; an internal arithmetic/self-consistency check alone is none, "
        "and so are citations of the passage's own condition/clause/section numbers ('per Condition 21') and "
        "comparisons against other values in the same passage, unless a specific external instrument is also named.\n"
        "A sourced requirement-report -- '$X must be paid in accordance with [Policy]', 'as stated in the "
        "passage' -- is silent, NOT endorsed: citing where the requirement comes from is sourcing, not a vouch "
        "(and bare attribution is not corroboration either); endorsement needs separate words that the value is "
        "correct/plausible.\n"
        "Judge content, not formatting: a 'Flag:' heading followed by reassurance is endorsed; a 'Flag:' heading "
        "followed by 'I cannot verify this' is declined."
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
    return obj["stance"], obj["corroboration"], obj["reason"], r.model

def validate_caveat_judge():
    def call_judge(row):
        stance, corroboration, reason, _ = caveat_judge(row["q"], row["answer"])
        return stance, reason, {"corroboration": corroboration}
    return _meta_evaluate(CAVEAT_GOLD_FILE, CAVEAT_RESULTS_FILE, "caveat judge", "mixed candidates",
                          CAVEAT_LABELS, call_judge, "instruction",
                          second=("corroboration", CORROBORATION_LABELS, "human_corroboration"))

# Assesses where users are at in successful test execution
if __name__ == "__main__":
    if sys.argv[1:] == ["caveat"]:
        if not os.path.exists(CAVEAT_GOLD_FILE):
            raise SystemExit(f"{CAVEAT_GOLD_FILE} not found -- the caveat gold is curated from labelled "
                             f"transcripts, not generated; restore it before validating")
        with open(CAVEAT_GOLD_FILE) as f:
            rows = json.load(f)
        need_stance = [r for r in rows if r.get("human") is None]
        need_corr = [r for r in rows if r.get("human_corroboration") is None]
        if need_stance or need_corr:
            print(f'{len(rows)} caveat gold rows: {len(need_stance)} need "human" stance {CAVEAT_LABELS}, '
                  f'{len(need_corr)} need "human_corroboration" {CORROBORATION_LABELS}.')
            print(f'Rows carrying "human_prev" were re-nulled because the taxonomy changed -- re-audit them against '
                  f'the new declined/endorsed boundary. If a "declined anchor" reads as ambiguous, change its "role" '
                  f'to "borderline" so it does not gate. Fill {CAVEAT_GOLD_FILE}, then re-run: python3 judge.py caveat')
            sys.exit(1)
        if validate_caveat_judge() == GATE_FAIL:
            sys.exit(1)
    elif sys.argv[1:] == ["abstention", "expand"]:
        n = build_abstention_gold()
        print(f"Generated {n} new transcripts into {ABSTENTION_GOLD_FILE}; existing rows and labels preserved.")
        print(f'Now set each new "human" field to "{FAITHFUL}" or "{UNGROUNDED}", then re-run: python3 judge.py abstention')
    elif sys.argv[1:] == ["abstention"]:
        if not os.path.exists(ABSTENTION_GOLD_FILE):
            n = build_abstention_gold()
            print(f"Generated {n} transcripts into {ABSTENTION_GOLD_FILE}.")
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
        print("usage: python3 judge.py caveat | abstention | abstention expand")
        sys.exit(1)
