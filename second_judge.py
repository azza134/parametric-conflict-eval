import json
import random
import sys
from config import anthropic_client, with_retry
from judge import (CAVEAT_SYSTEM, build_caveat_prompt, CAVEAT_SCHEMA, CAVEAT_LABELS,
                   CORROBORATION_LABELS, CAVEAT_GOLD_FILE, _meta_evaluate, ENDORSED)

SECOND_JUDGE_MODEL = "claude-opus-4-8"
SECOND_CERT_RESULTS_FILE = "data/second_judge_certification.json"
SECOND_CHECK_RESULTS_FILE = "data/second_judge_check.jsonl"
CAVEAT_RESULTS = "data/caveat_results_v2.jsonl"
CHECK_SEED = 20260717
CONTROL_N = 88
TERRA_N = 90


def anthropic_caveat_judge(question, answer):
    r = with_retry(lambda: anthropic_client().messages.create(
        model=SECOND_JUDGE_MODEL,
        max_tokens=1024,
        system=CAVEAT_SYSTEM,
        tools=[{"name": "stance", "input_schema": CAVEAT_SCHEMA}],
        tool_choice={"type": "tool", "name": "stance"},
        messages=[{"role": "user", "content": build_caveat_prompt(question, answer)}],
    ))
    obj = next(b.input for b in r.content if b.type == "tool_use")
    return obj["stance"], obj["corroboration"], obj["reason"], r.model


def certify():
    def call_judge(row):
        stance, corroboration, reason, _ = anthropic_caveat_judge(row["q"], row["answer"])
        return stance, reason, {"corroboration": corroboration}
    return _meta_evaluate(CAVEAT_GOLD_FILE, SECOND_CERT_RESULTS_FILE,
                          f"second caveat judge ({SECOND_JUDGE_MODEL})", "mixed candidates",
                          CAVEAT_LABELS, call_judge, "instruction",
                          second=("corroboration", CORROBORATION_LABELS, "human_corroboration"))


def check_selection():
    rows = [json.loads(l) for l in open(CAVEAT_RESULTS)]
    fact_q = {}
    from harness import FACT_BY_NAME
    for name, f in FACT_BY_NAME.items():
        fact_q[name] = f["q"]
    sonnet_fi = [r for r in rows if r["model"] == "claude-sonnet-5"
                 and r["instruction"] == "FLAG_INVITING" and r["severity"]]
    endorsed = [r for r in sonnet_fi if r["stance"] == ENDORSED]
    non_endorsed = [r for r in sonnet_fi if r["stance"] != ENDORSED]
    terra_fi = [r for r in rows if r["model"] == "gpt-5.6-terra"
                and r["instruction"] == "FLAG_INVITING" and r["severity"]]
    rng = random.Random(CHECK_SEED)
    controls = rng.sample(non_endorsed, CONTROL_N)
    terra = rng.sample(terra_fi, TERRA_N)
    sel = [("sonnet-endorsed", r) for r in endorsed] + \
          [("sonnet-control", r) for r in controls] + \
          [("terra-zero-side", r) for r in terra]
    return [(i, arm, r, fact_q[r["fact"]]) for i, (arm, r) in enumerate(sel)]


def check():
    sel = check_selection()
    done = 0
    try:
        done = sum(1 for _ in open(SECOND_CHECK_RESULTS_FILE))
    except FileNotFoundError:
        pass
    out = open(SECOND_CHECK_RESULTS_FILE, "a")
    for i, arm, r, q in sel:
        if i < done:
            continue
        stance, corroboration, reason, snapshot = anthropic_caveat_judge(q, r["answer"])
        json.dump({"idx": i, "arm": arm, "model": r["model"], "fact": r["fact"],
                   "severity": r["severity"], "certified_stance": r["stance"],
                   "certified_corroboration": r["corroboration"], "second_stance": stance,
                   "second_corroboration": corroboration, "second_reason": reason,
                   "second_judge_snapshot": snapshot}, out)
        out.write("\n")
        out.flush()
        print(f"  [{i + 1}/{len(sel)}] {arm} {r['fact']} S{r['severity']} "
              f"certified={r['stance']} second={stance}", flush=True)
    out.close()
    summarize()


def summarize():
    rows = [json.loads(l) for l in open(SECOND_CHECK_RESULTS_FILE)]
    for arm in ("sonnet-endorsed", "sonnet-control", "terra-zero-side"):
        sub = [r for r in rows if r["arm"] == arm]
        agree = sum(r["second_stance"] == r["certified_stance"] for r in sub)
        second_endorsed = sum(r["second_stance"] == ENDORSED for r in sub)
        print(f"{arm}: n={len(sub)}  stance agreement {agree}/{len(sub)}  "
              f"second-judge endorsed count {second_endorsed}")
        for r in sub:
            if r["second_stance"] != r["certified_stance"]:
                print(f"    {r['fact']} S{r['severity']}: certified={r['certified_stance']} "
                      f"second={r['second_stance']}")


if __name__ == "__main__":
    if sys.argv[1:] == ["certify"]:
        certify()
    elif sys.argv[1:] == ["check"]:
        check()
    elif sys.argv[1:] == ["summarize"]:
        summarize()
    else:
        print("usage: python3 second_judge.py certify | check | summarize")
        sys.exit(1)
