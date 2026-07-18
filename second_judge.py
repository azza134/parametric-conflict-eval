import json
import random
import sys
from collections import Counter
from config import (anthropic_client, with_retry, doc_text,
                    submit_anthropic_batch, poll_anthropic_batch, anthropic_batch_results)
from judge import (CAVEAT_SYSTEM, build_caveat_prompt, CAVEAT_SCHEMA, CAVEAT_LABELS,
                   CORROBORATION_LABELS, CAVEAT_GOLD_FILE, _meta_evaluate, ENDORSED,
                   ABSTENTION_SYSTEM, ABSTENTION_SCHEMA, ABSTENTION_GOLD_FILE,
                   build_abstention_prompt, FAITHFUL, UNGROUNDED)

SECOND_JUDGE_MODEL = "claude-opus-4-8"
SECOND_CERT_RESULTS_FILE = "data/second_judge_certification.json"
SECOND_CHECK_RESULTS_FILE = "data/second_judge_check.jsonl"
CAVEAT_RESULTS = "data/caveat_results_v2.jsonl"
CHECK_SEED = 20260717
CONTROL_N = 88
TERRA_N = 90

ABSTENTION_RESULTS = "data/abstention_results_v2.jsonl"
SECOND_ABSTENTION_CERT_FILE = "data/second_judge_abstention_certification.json"
SECOND_ABSTENTION_CHECK_FILE = "data/second_judge_abstention_check.jsonl"
SECOND_ABSTENTION_RAW_FILE = "data/second_judge_abstention_raw.json"
ABSTENTION_CHECK_SEED = 20260718
ABSTENTION_BUDGET_USD = 2.54
ABSTENTION_SAMPLE_LADDER = [(4, 2, 6, 3), (3, 2, 6, 3), (3, 2, 4, 2), (2, 1, 4, 2), (2, 1, 2, 1)]
CACHE_MIN_TOKENS = 4096
CACHE_MIN_JOBS = 2
CACHE_TTL = "5m"
BATCH_IN_PER_M, BATCH_OUT_PER_M = 2.50, 12.50
CACHE_WRITE_MULT, CACHE_READ_MULT = 1.25, 0.1
EST_OUTPUT_TOKENS = 130
SUFFIX_CHARS_PER_TOKEN = 3.8
ANSWER_MARKER = "ANSWER (written by another AI):\n"


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


def abstention_prompt_blocks(question, doc, answer):
    full = build_abstention_prompt(question, doc, answer)
    i = full.index(ANSWER_MARKER)
    return full[:i], full[i:]


def abstention_judge_params(question, doc, answer, cached):
    prefix, suffix = abstention_prompt_blocks(question, doc, answer)
    prefix_block = {"type": "text", "text": prefix}
    if cached:
        prefix_block["cache_control"] = {"type": "ephemeral", "ttl": CACHE_TTL}
    return {
        "model": SECOND_JUDGE_MODEL,
        "max_tokens": 1024,
        "system": ABSTENTION_SYSTEM,
        "tools": [{"name": "verdict", "input_schema": ABSTENTION_SCHEMA}],
        "tool_choice": {"type": "tool", "name": "verdict"},
        "messages": [{"role": "user", "content": [prefix_block, {"type": "text", "text": suffix}]}],
    }


def _abstention_gold_doc(row):
    if row.get("evidence_state") == "absent":
        from harness import FACT_BY_NAME, absence_doc
        return absence_doc(FACT_BY_NAME[row["fact"]])
    return doc_text(row.get("doc") or "consent")


def abstention_cert_jobs():
    with open(ABSTENTION_GOLD_FILE) as f:
        gold = json.load(f)
    return [{"cid": f"acert-{i}", "q": row["q"], "doc": _abstention_gold_doc(row),
             "answer": row["answer"]} for i, row in enumerate(gold)]


def abstention_check_selection(openai_faithful, openai_ungrounded, control_faithful, control_ungrounded):
    rows = [json.loads(l) for l in open(ABSTENTION_RESULTS)]
    rng = random.Random(ABSTENTION_CHECK_SEED)
    sel = []
    gpt = [(i, r) for i, r in enumerate(rows) if r["model"].startswith("gpt")]
    for model in sorted({r["model"] for _, r in gpt}):
        for instruction in sorted({r["instruction"] for _, r in gpt}):
            cell = [(i, r) for i, r in gpt if r["model"] == model and r["instruction"] == instruction]
            faith = [(i, r) for i, r in cell if r["faithful"]]
            unfaith = [(i, r) for i, r in cell if not r["faithful"]]
            sel += [("openai-faithful-side", i, r)
                    for i, r in rng.sample(faith, min(openai_faithful, len(faith)))]
            sel += [("openai-ungrounded-side", i, r)
                    for i, r in rng.sample(unfaith, min(openai_ungrounded, len(unfaith)))]
    claude = [(i, r) for i, r in enumerate(rows) if r["model"].startswith("claude")]
    for model in sorted({r["model"] for _, r in claude}):
        mrows = [(i, r) for i, r in claude if r["model"] == model]
        faith = [(i, r) for i, r in mrows if r["faithful"]]
        unfaith = [(i, r) for i, r in mrows if not r["faithful"]]
        sel += [("anthropic-control-faithful", i, r)
                for i, r in rng.sample(faith, min(control_faithful, len(faith)))]
        sel += [("anthropic-control-ungrounded", i, r)
                for i, r in rng.sample(unfaith, min(control_ungrounded, len(unfaith)))]
    return sel


def abstention_check_jobs(selection):
    return [{"cid": f"achk-{i}", "q": r["q"], "doc": doc_text(r["document"]),
             "answer": r["answer"]} for _, i, r in selection]


def cached_prefix_keys(jobs, prefix_tokens):
    counts = Counter((j["q"], j["doc"]) for j in jobs)
    return {k for k, n in counts.items()
            if n >= CACHE_MIN_JOBS and prefix_tokens[k] >= CACHE_MIN_TOKENS}


def estimate_abstention_cost(jobs, prefix_tokens):
    cached = cached_prefix_keys(jobs, prefix_tokens)
    counts = Counter((j["q"], j["doc"]) for j in jobs)
    expected_in = worst_in = 0.0
    for key, n in counts.items():
        pt = prefix_tokens[key]
        if key in cached:
            expected_in += pt * (CACHE_WRITE_MULT + (n - 1) * CACHE_READ_MULT)
            worst_in += pt * n * CACHE_WRITE_MULT
        else:
            expected_in += pt * n
            worst_in += pt * n
    suffix = sum(len(abstention_prompt_blocks(j["q"], j["doc"], j["answer"])[1]) / SUFFIX_CHARS_PER_TOKEN
                 for j in jobs)
    out = len(jobs) * EST_OUTPUT_TOKENS
    return {
        "n_jobs": len(jobs),
        "n_prefixes": len(counts),
        "n_cached_prefixes": len(cached),
        "expected_usd": (expected_in + suffix) * BATCH_IN_PER_M / 1e6 + out * BATCH_OUT_PER_M / 1e6,
        "worst_usd": (worst_in + suffix) * BATCH_IN_PER_M / 1e6 + out * BATCH_OUT_PER_M / 1e6,
    }


def count_prefix_tokens(jobs, tokens=None):
    tokens = {} if tokens is None else tokens
    for j in jobs:
        key = (j["q"], j["doc"])
        if key in tokens:
            continue
        prefix, _ = abstention_prompt_blocks(j["q"], j["doc"], j["answer"])
        r = with_retry(lambda p=prefix: anthropic_client().messages.count_tokens(
            model=SECOND_JUDGE_MODEL,
            system=ABSTENTION_SYSTEM,
            tools=[{"name": "verdict", "input_schema": ABSTENTION_SCHEMA}],
            messages=[{"role": "user", "content": [{"type": "text", "text": p}]}],
        ))
        tokens[key] = r.input_tokens
    return tokens


def abstention_plan():
    cert = abstention_cert_jobs()
    print(f"counting prefix tokens ({SECOND_JUDGE_MODEL}, free) ...", flush=True)
    prefix_tokens = {}
    for config_ in ABSTENTION_SAMPLE_LADDER:
        sel = abstention_check_selection(*config_)
        jobs = cert + abstention_check_jobs(sel)
        count_prefix_tokens(jobs, prefix_tokens)
        est = estimate_abstention_cost(jobs, prefix_tokens)
        print(f"  sample config {config_}: {est['n_jobs']} jobs "
              f"({len(cert)} cert + {len(sel)} check), {est['n_prefixes']} prefixes "
              f"({est['n_cached_prefixes']} cached), expected ${est['expected_usd']:.2f}, "
              f"worst-case ${est['worst_usd']:.2f}", flush=True)
        if est["expected_usd"] <= ABSTENTION_BUDGET_USD:
            return config_, sel, jobs, prefix_tokens, est
    raise SystemExit(f"no sample config fits the ${ABSTENTION_BUDGET_USD:.2f} budget")


def _load_abstention_raw():
    try:
        with open(SECOND_ABSTENTION_RAW_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def run_abstention_batches(jobs, prefix_tokens):
    raw = _load_abstention_raw()
    cached = cached_prefix_keys(jobs, prefix_tokens)
    pending = [j for j in jobs if j["cid"] not in raw]
    spend = Counter()
    for round_ in range(3):
        if not pending:
            break
        seen, warm, main = set(), [], []
        for j in pending:
            key = (j["q"], j["doc"])
            if key in cached and key not in seen:
                warm.append(j)
                seen.add(key)
            else:
                main.append(j)
        for wave_name, wave in (("warm", warm), ("main", main)):
            if not wave:
                continue
            reqs = [(j["cid"], abstention_judge_params(j["q"], j["doc"], j["answer"],
                                                       (j["q"], j["doc"]) in cached)) for j in wave]
            batch_id = submit_anthropic_batch(reqs)
            print(f"round {round_ + 1} {wave_name} wave: {len(wave)} requests, batch {batch_id}", flush=True)
            poll_anthropic_batch(batch_id, on_poll=lambda b: print(
                f"  {b.processing_status}: {b.request_counts.succeeded} ok / "
                f"{b.request_counts.errored} err / {b.request_counts.processing} processing", flush=True))
            for cid, result in anthropic_batch_results(batch_id):
                if result.type != "succeeded":
                    print(f"  {cid}: {result.type}", flush=True)
                    continue
                m = result.message
                obj = next(b.input for b in m.content if b.type == "tool_use")
                u = m.usage
                spend.update({"input": u.input_tokens, "output": u.output_tokens,
                              "cache_write": u.cache_creation_input_tokens or 0,
                              "cache_read": u.cache_read_input_tokens or 0})
                raw[cid] = {"faithful": bool(obj["faithful"]), "reason": obj["reason"], "snapshot": m.model}
            with open(SECOND_ABSTENTION_RAW_FILE, "w") as f:
                json.dump(raw, f, indent=2)
        pending = [j for j in jobs if j["cid"] not in raw]
    if pending:
        raise SystemExit(f"{len(pending)} jobs still unresolved after 3 rounds: "
                         + ", ".join(j["cid"] for j in pending[:10]))
    usd = ((spend["input"] * 1.0 + spend["cache_write"] * CACHE_WRITE_MULT
            + spend["cache_read"] * CACHE_READ_MULT) * BATCH_IN_PER_M
           + spend["output"] * BATCH_OUT_PER_M) / 1e6
    print(f"this run: {spend['input']} uncached-in, {spend['cache_write']} cache-write, "
          f"{spend['cache_read']} cache-read, {spend['output']} out tokens -> ${usd:.2f}", flush=True)
    return raw


def certify_abstention(raw):
    state = {"i": -1}

    def call_judge(row):
        state["i"] += 1
        v = raw[f"acert-{state['i']}"]
        return (FAITHFUL if v["faithful"] else UNGROUNDED), v["reason"], {}

    return _meta_evaluate(ABSTENTION_GOLD_FILE, SECOND_ABSTENTION_CERT_FILE,
                          f"second abstention judge ({SECOND_JUDGE_MODEL})", "mixed candidates",
                          (FAITHFUL, UNGROUNDED), call_judge, "instruction")


def abstention_run():
    config_, sel, jobs, prefix_tokens, est = abstention_plan()
    print(f"running with sample config {config_} (expected ${est['expected_usd']:.2f})", flush=True)
    raw = run_abstention_batches(jobs, prefix_tokens)
    verdict = certify_abstention(raw)
    with open(SECOND_ABSTENTION_CHECK_FILE, "w") as f:
        for arm, i, r in sel:
            v = raw[f"achk-{i}"]
            json.dump({"idx": i, "arm": arm, "model": r["model"], "instruction": r["instruction"],
                       "item_id": r["item_id"], "document": r["document"],
                       "primary_faithful": r["faithful"],
                       "second_faithful": v["faithful"], "second_reason": v["reason"],
                       "second_judge_snapshot": v["snapshot"]}, f)
            f.write("\n")
    summarize_abstention()
    print(f"certification gate: {verdict}", flush=True)


def summarize_abstention():
    rows = [json.loads(l) for l in open(SECOND_ABSTENTION_CHECK_FILE)]
    for arm in ("openai-faithful-side", "openai-ungrounded-side",
                "anthropic-control-faithful", "anthropic-control-ungrounded"):
        sub = [r for r in rows if r["arm"] == arm]
        if not sub:
            continue
        agree = sum(r["second_faithful"] == r["primary_faithful"] for r in sub)
        print(f"{arm}: n={len(sub)}  agreement {agree}/{len(sub)}")
        for r in sub:
            if r["second_faithful"] != r["primary_faithful"]:
                print(f"    {r['model']} {r['instruction']} {r['item_id']}: "
                      f"primary={'faithful' if r['primary_faithful'] else 'ungrounded'} "
                      f"second={'faithful' if r['second_faithful'] else 'ungrounded'}")
    gpt_faith = [r for r in rows if r["arm"] == "openai-faithful-side"]
    ctl_faith = [r for r in rows if r["arm"] == "anthropic-control-faithful"]
    if gpt_faith and ctl_faith:
        gpt_flip = sum(not r["second_faithful"] for r in gpt_faith)
        ctl_flip = sum(not r["second_faithful"] for r in ctl_faith)
        print(f"faithful-side overturn rate: openai {gpt_flip}/{len(gpt_faith)} "
              f"vs anthropic control {ctl_flip}/{len(ctl_faith)}")


if __name__ == "__main__":
    if sys.argv[1:] == ["certify"]:
        certify()
    elif sys.argv[1:] == ["check"]:
        check()
    elif sys.argv[1:] == ["summarize"]:
        summarize()
    elif sys.argv[1:] == ["abstention"]:
        abstention_plan()
    elif sys.argv[1:] == ["abstention", "run"]:
        abstention_run()
    elif sys.argv[1:] == ["abstention", "summarize"]:
        summarize_abstention()
    else:
        print("usage: python3 second_judge.py certify | check | summarize | "
              "abstention [run | summarize]")
        sys.exit(1)
