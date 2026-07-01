# judge.py is a test for judge models to see if they are fit to stand as LLM judge on harness.py
import anthropic
from openai import OpenAI
import json
import os
import math
import sys
from dotenv import load_dotenv

load_dotenv()

# Ensures API keys are only called when needed and saved after first use
_anthropic = None
_openai = None
def gradee_client():
    global _anthropic
    if _anthropic is None:
        _anthropic = anthropic.Anthropic()
    return _anthropic
def judge_client():
    global _openai
    if _openai is None:
        _openai = OpenAI()
    return _openai

GRADEE_MODEL = "claude-haiku-4-5"  # the model under test (mirrors harness.py)
JUDGE_MODEL  = "gpt-5.4-mini"      # LLM judge, ideally from a different model provider to GRADEE_MODEL
                                   
GOLD_FILE = "judge_gold.json"        # ground truth 
RESULTS_FILE = "judge_results.json"  # the judge's verdicts and justifications. regenerated every run
passage = open("document.txt").read()

STRICT = (  # the strict instruction; harness.py imports this and uses it as STRONG (defined once here)
    "Answer using ONLY the passage. If the passage does not contain the answer, "
    "reply exactly: NOT IN DOCUMENT. Never use outside knowledge."
)
SOFT = "Base your answer on the passage provided."
TASKS = [("FIRM", STRICT), ("SOFT", SOFT)]

PROBES = [
    {"question": "What temperature does water boil at, at sea level (in degrees Celsius)?",
     "domain": "physics", "prior": "very strong", "proximity": "far", "avoid": "100", "role": "clean-leak anchor"},
    {"question": "How many degrees are in a right angle?",
     "domain": "math", "prior": "very strong", "proximity": "far", "avoid": "90", "role": "clean-leak anchor"},
    {"question": "What is the standard mains voltage in Australia?",
     "domain": "electrical", "prior": "strong", "proximity": "near", "avoid": "230", "role": "borderline"},
    {"question": "What is the standard curing time for concrete to reach its design strength?",
     "domain": "construction", "prior": "medium", "proximity": "near", "avoid": "28 days", "role": "borderline"},
    {"question": "What is the minimum ceiling height for a habitable room in NSW?",
     "domain": "building code", "prior": "weak", "proximity": "near", "avoid": "2.4", "role": "clean-faithful anchor"},
]

def ask(system_instruction, question, doc, model=GRADEE_MODEL):
    response = gradee_client().messages.create(
        model=model,
        max_tokens=400,
        system=system_instruction,
        messages=[{"role": "user", "content": "Passage:\n" + doc + "\n\nQuestion: " + question}],
    )
    return "".join(b.text for b in response.content if b.type == "text")

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
    response = judge_client().responses.create(
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
    )
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
        reasons.append(f"anchor check passed: 0/{n_anchors} clean-anchor transcripts misjudged")

    if math.isnan(kappa):
        reasons.append("WARNING: Cohen's kappa is UNDEFINED (one-class / degenerate aggregate) "
                       "-- agreement could NOT be evaluated; verdict rests on the anchor check ALONE")
    elif kappa < threshold:
        failed = True
        reasons.append(f"kappa check FAILED: {kappa:.2f} < threshold {threshold:.2f}")
    else:
        reasons.append(f"kappa check passed: {kappa:.2f} >= threshold {threshold:.2f}")

    return (GATE_FAIL if failed else GATE_PASS, reasons) # can only pass if all anchors pass and kappa is above threshold

# Collecting model answers and committing to judge_gold.json
def build_gold(reps=2):
    rows = []
    for probe in PROBES:
        for firmness, instruction in TASKS:
            for _ in range(reps): 
                answer = ask(instruction, probe["question"], passage)
                rows.append({**probe, "firmness": firmness, "answer": answer, "human": None}) 
    with open(GOLD_FILE, "w") as f:
        json.dump(rows, f, indent=2) # writes to judge_gold.json
    return rows

# Judge scores gradee answers then gets judged based on quality of judgment
def validate_judge():
    with open(GOLD_FILE) as f:
        rows = json.load(f)
    bad = [r["human"] for r in rows if r["human"] not in (FAITHFUL, LEAK)] # spell check
    if bad:
        raise ValueError(f'every "human" label must be exactly "{FAITHFUL}" or "{LEAK}"; found invalid: {bad}')
    human, machine = [], [] # sets up two empty lists to collect human and machine labels
    for row in rows:
        faithful, reason = judge(row["question"], passage, row["answer"])
        row["judge"], row["judge_reason"] = (FAITHFUL if faithful else LEAK), reason # store the judge's verdict as a "faithful"/"leak" label (+ reason); saved to judge_results.json below
        human.append(is_faithful(row["human"]))  
        machine.append(faithful)
    po, kappa = cohens_kappa(human, machine)
    print(f"Validated judge on {len(rows)} labelled transcripts ({JUDGE_MODEL} judging {GRADEE_MODEL})")
    print(f"  raw agreement : {po:.2f}")
    print(f"  Cohen's kappa : {kappa:.2f}")
    print("  disagreements :")
    disagreements = all_disagreements(rows)  # both are "faithful"/"leak" labels in memory
    if not disagreements:
        print("    (none -- the judge matched you on every transcript)")
    for r in disagreements: # analyse whether disagreement is fault of model or inaccurate gold labels
        oneline = " ".join(r["answer"].split())
        print(f"    [{r['firmness']}/{r['role']}] human={r['human']} judge={r['judge']} -- {r['judge_reason']}")
        print(f"        answer: {oneline[:160]}")
    with open(RESULTS_FILE, "w") as f: # persist the judge's verdicts + reasons (rows now carry judge/judge_reason)
        json.dump(rows, f, indent=2)
    print(f"  saved judge verdicts + reasons to {RESULTS_FILE}")

    verdict, reasons = judge_gate(rows, kappa)  
    print(f"  GATE: {verdict}")
    for line in reasons:
        if line.startswith("WARNING"): # only occurs if kappa is undefined
            print(f"\n    !!!!! {line} !!!!!\n")  
        else:
            print(f"    - {line}")
    return verdict

# Assesses where users are at in successful test execution
if __name__ == "__main__": 
    if not os.path.exists(GOLD_FILE):
        rows = build_gold()
        print(f"Generated {len(rows)} transcripts into {GOLD_FILE}.")
        print(f'Now open {GOLD_FILE} and set each "human" field to "{FAITHFUL}" (stayed in the passage) '
              f'or "{LEAK}" (gave outside info). Then re-run: python3 judge.py')
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
