import json
import random
import sys
from harness import FACT_BY_NAME, CAVEAT_RESULTS, ABSTENTION_RESULTS, ABSENCE_RESULTS

CAVEAT_GOLD = "data/caveat_gold.json"
ABSTENTION_GOLD = "data/abstention_gold.json"
COUNTS = {"cv": 30, "ab": 18, "ma": 12}
STANCE_CAPS = [("endorsed", 8), ("questioned", 10), ("declined", 4)]
UNGROUNDED_CAPS = {"ab": 6, "ma": 4}


def load_results(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def coverage_pick(pool, k, keys, rng):
    pool = pool[:]
    rng.shuffle(pool)
    picked = []
    seen = {key: set() for key in keys}
    while pool and len(picked) < k:
        best = max(pool, key=lambda r: sum(r[key] not in seen[key] for key in keys))
        pool.remove(best)
        picked.append(best)
        for key in keys:
            seen[key].add(best[key])
    return picked


def sample_caveat(rows, n, rng):
    keys = ["instruction", "severity", "document"]
    picked = []
    for stance, cap in STANCE_CAPS:
        picked += coverage_pick([r for r in rows if r["stance"] == stance], cap, keys, rng)
    picked += coverage_pick([r for r in rows if r["stance"] == "silent"], n - len(picked), keys, rng)
    return picked[:n]


def sample_binary(rows, n, ungrounded_cap, keys, rng):
    picked = coverage_pick([r for r in rows if r["label"] == "ungrounded"], ungrounded_cap, keys, rng)
    picked += coverage_pick([r for r in rows if r["label"] == "faithful"], n - len(picked), keys, rng)
    return picked[:n]


def caveat_gold_row(r, role):
    fact = FACT_BY_NAME[r["fact"]]
    step = next(s for s in fact["steps"] if s["severity"] == r["severity"])
    return {"q": fact["q"], "replace": [list(pair) for pair in step["replace"]],
            "target_string": r["target_string"], "severity": r["severity"], "document": r["document"],
            "role": role, "instruction": r["instruction"], "candidate": r["model"],
            "human": "", "human_corroboration": "", "answer": r["answer"]}


def abstention_gold_row(r, role):
    return {"item_id": r["item_id"], "doc": r["document"], "prior_strength": r["prior_strength"],
            "proximity": r["proximity"], "domain": r["domain"], "parametric_answer": r["parametric_answer"],
            "q": r["q"], "role": role, "instruction": r["instruction"], "candidate": r["model"],
            "human": "", "answer": r["answer"]}


def absence_gold_row(r, role):
    return {"item_id": "absent " + r["fact"], "doc": r["document"], "fact": r["fact"],
            "evidence_state": r["evidence_state"], "prior_strength": None, "proximity": "matched",
            "domain": "matched-absence", "parametric_answer": r["true"], "q": r["q"], "role": role,
            "instruction": r["instruction"], "candidate": r["model"], "human": "", "answer": r["answer"]}


def sidecar_entry(kind, r, role):
    if kind == "cv":
        return {"role": role, "stance": r["stance"], "corroboration": r["corroboration"], "label": r["label"]}
    return {"role": role, "label": r["label"], "verbatim_abstention": r["verbatim_abstention"]}


def sidecar_path(tag, seed):
    return f"data/spotcheck_sidecar_{tag}_{seed}.json"


def role_prefix(tag, seed):
    return f"{tag}-spotcheck seed{seed} "


def draw(model, tag, seed):
    rng = random.Random(seed)
    samples = {
        "cv": sample_caveat([r for r in load_results(CAVEAT_RESULTS) if r["model"] == model], COUNTS["cv"], rng),
        "ab": sample_binary([r for r in load_results(ABSTENTION_RESULTS) if r["model"] == model],
                            COUNTS["ab"], UNGROUNDED_CAPS["ab"], ["instruction", "item_id"], rng),
        "ma": sample_binary([r for r in load_results(ABSENCE_RESULTS) if r["model"] == model],
                            COUNTS["ma"], UNGROUNDED_CAPS["ma"], ["instruction", "fact"], rng),
    }
    builders = {"cv": caveat_gold_row, "ab": abstention_gold_row, "ma": absence_gold_row}
    gold = {"cv": [], "ab": [], "ma": []}
    sidecar = []
    for kind, rows in samples.items():
        for i, r in enumerate(rows):
            role = f"{role_prefix(tag, seed)}{kind}{i:02d}"
            gold[kind].append(builders[kind](r, role))
            sidecar.append(sidecar_entry(kind, r, role))
    return gold, sidecar


def summarize(gold):
    for kind, rows in gold.items():
        by_instruction = {}
        for r in rows:
            by_instruction[r["instruction"]] = by_instruction.get(r["instruction"], 0) + 1
        print(f"{kind}: {len(rows)} rows  {by_instruction}")
        if kind == "cv":
            by_severity = {}
            for r in rows:
                by_severity[r["severity"]] = by_severity.get(r["severity"], 0) + 1
            print(f"    severities: {dict(sorted(by_severity.items()))}")


def merge(gold, sidecar, tag, seed):
    prefix = role_prefix(tag, seed)
    for path, new_rows in [(CAVEAT_GOLD, gold["cv"]), (ABSTENTION_GOLD, gold["ab"] + gold["ma"])]:
        existing = json.load(open(path))
        clashes = [g["role"] for g in existing if str(g.get("role", "")).startswith(prefix)]
        if clashes:
            raise SystemExit(f"{path} already contains {len(clashes)} rows for {prefix!r} -- refusing to merge twice")
        with open(path, "w") as f:
            json.dump(existing + new_rows, f, indent=2)
        print(f"{path}: +{len(new_rows)} unlabeled rows (total {len(existing) + len(new_rows)})")
    with open(sidecar_path(tag, seed), "w") as f:
        json.dump(sidecar, f, indent=2)
    print(f"{sidecar_path(tag, seed)}: judge verdicts held out here -- do not open until labelling is done")


def compare(tag, seed):
    verdicts = {e["role"]: e for e in json.load(open(sidecar_path(tag, seed)))}
    prefix = role_prefix(tag, seed)
    rows = [g for path in (CAVEAT_GOLD, ABSTENTION_GOLD) for g in json.load(open(path))
            if str(g.get("role", "")).startswith(prefix)]
    if len(rows) != len(verdicts):
        print(f"WARNING: {len(rows)} gold rows vs {len(verdicts)} sidecar entries")
    agree = 0
    misses = []
    for g in rows:
        v = verdicts[g["role"]]
        judge_call = v["stance"] if "stance" in v else v["label"]
        if not g["human"]:
            misses.append((g["role"], "UNLABELLED", judge_call))
            continue
        if g["human"] == judge_call:
            agree += 1
        else:
            misses.append((g["role"], g["human"], judge_call))
    print(f"stance/label agreement: {agree}/{len(rows)}")
    for role, human, judge_call in misses:
        print(f"  {role}: human={human} judge={judge_call}")
    corro = [(g, verdicts[g["role"]]) for g in rows
             if "human_corroboration" in g and g.get("human_corroboration", "") != ""]
    if corro:
        c_agree = sum(g["human_corroboration"] == v["corroboration"] for g, v in corro)
        print(f"corroboration agreement: {c_agree}/{len(corro)}")
        for g, v in corro:
            if g["human_corroboration"] != v["corroboration"]:
                print(f"  {g['role']}: human={g['human_corroboration']} judge={v['corroboration']}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) >= 4 and args[0] == "sample":
        model, tag, seed = args[1], args[2], int(args[3])
        gold, sidecar = draw(model, tag, seed)
        summarize(gold)
        if "--merge" in args:
            merge(gold, sidecar, tag, seed)
        else:
            print("\n  (dry run -- add --merge to append unlabeled rows to the gold files and write the sidecar)")
    elif len(args) == 3 and args[0] == "compare":
        compare(args[1], int(args[2]))
    else:
        print("usage: python3 spotcheck_sampler.py sample <model> <tag> <seed> [--merge] | compare <tag> <seed>")
        sys.exit(1)
