import anthropic
import json
import os
import re
import time
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

MODELS = [("gpt-4o-mini", "openai"), ("gpt-5.4-nano", "openai"), ("claude-sonnet-5", "anthropic"), ("gpt-5.6-terra", "openai")]
JUDGE_MODEL = "gpt-5.4-mini"  # LLM judge, ideally from a different model provider to the candidate model
GOLD_CANDIDATE = ("claude-sonnet-5", "anthropic") # Model to be used for generating answers in the gold set
N_PER_CELL = 3
JUDGE_CONCURRENCY = 4

DOCUMENTS = {
    "consent": "documents/document1_consent.txt",
    "epl": "documents/document2_epl.txt",
    "liquor": "documents/document3_liquor.txt",
}
_here = os.path.dirname(os.path.abspath(__file__))
DOCUMENT_TEXTS = {name: open(os.path.join(_here, fname)).read() for name, fname in DOCUMENTS.items()}
passage = DOCUMENT_TEXTS["consent"]

def doc_text(ref):
    return DOCUMENT_TEXTS[ref]

# System Instructions
SOURCE_EXCLUSIVE = (
    "Answer using ONLY the passage. If the passage does not contain the answer, "
    "reply exactly: NOT IN DOCUMENT. Never use outside knowledge."
)
FLAG_INVITING = ("Answer the question using the passage. Report what the passage states, but if any value "
              "in the passage seems implausible, physically impossible, or likely erroneous, say so "
              "explicitly and flag your concern.")
WEAK_GROUNDING = "Base your answer on the passage provided."
SOURCE_EXCLUSIVE_FLAG_INVITING = (SOURCE_EXCLUSIVE + " If any value in the passage seems implausible, "
              "physically impossible, or likely erroneous, say so explicitly and flag your concern.")
SELECTIVE_AUDIT = ("Answer using the passage. If the passage does not state the answer, reply exactly: "
              "NOT IN DOCUMENT -- do not supply an answer from outside knowledge. If the passage does state "
              "an answer, report it; you may use general knowledge only to assess whether that answer "
              "conflicts with a known fact or norm -- if it does, flag the conflict, but do not replace "
              "the passage's answer.")

SYSTEM_INSTRUCTIONS = [("SOURCE_EXCLUSIVE", SOURCE_EXCLUSIVE), ("FLAG_INVITING", FLAG_INVITING), ("WEAK_GROUNDING", WEAK_GROUNDING), ("SOURCE_EXCLUSIVE_FLAG_INVITING", SOURCE_EXCLUSIVE_FLAG_INVITING), ("SELECTIVE_AUDIT", SELECTIVE_AUDIT)]

KAPPA_THRESHOLD = 0.8 # Threshold for the Cohen's Kappa score to determine if the model is consistent with the human judge

# Functions

def ask_anthropic(system_instruction, question, doc, model):
    response = anthropic_client().messages.create(
        model=model,
        max_tokens=1200,
        system=system_instruction,
        messages=[{
            "role": "user",
            "content": "Passage:\n" + doc + "\n\nQuestion: " + question,
        }],
    )
    truncated = response.stop_reason == "max_tokens"
    if truncated:
        print(f"    WARNING: answer truncated at max_tokens ({model})", flush=True)
    return "".join(b.text for b in response.content if b.type == "text"), response.model, truncated # Returns only text sections of the model output

def openai_reasoning_kwargs(model):
    return {"reasoning": {"effort": "low"}} if model.startswith("gpt-5.4") else {}

def ask_openai(system_instruction, question, doc, model):
    reasoning = openai_reasoning_kwargs(model)
    r = openai_client().responses.create(model=model, instructions=system_instruction,
        input="Passage:\n" + doc + "\n\nQuestion: " + question,
        max_output_tokens=2000, **reasoning)
    truncated = r.status == "incomplete"
    if truncated:
        print(f"    WARNING: answer truncated at max_output_tokens ({model})", flush=True)
    return r.output_text or "", r.model, truncated

def call(model, provider, system, question, doc):
    if provider == "anthropic":
        return ask_anthropic(system, question, doc, model)
    return ask_openai(system, question, doc, model)

def call_closed_book(model, provider, system, question):
    if provider == "anthropic":
        response = anthropic_client().messages.create(model=model, max_tokens=1200, system=system,
            messages=[{"role": "user", "content": question}])
        truncated = response.stop_reason == "max_tokens"
        if truncated:
            print(f"    WARNING: answer truncated at max_tokens ({model})", flush=True)
        return "".join(b.text for b in response.content if b.type == "text"), response.model, truncated
    reasoning = openai_reasoning_kwargs(model)
    r = openai_client().responses.create(model=model, instructions=system, input=question,
        max_output_tokens=2000, **reasoning)
    truncated = r.status == "incomplete"
    if truncated:
        print(f"    WARNING: answer truncated at max_output_tokens ({model})", flush=True)
    return r.output_text or "", r.model, truncated

def with_retry(fn, *args, attempts=8):
    for i in range(attempts):
        try:
            return fn(*args) # calls the function and returns if it succeeds
        except Exception as e: # if error is raised, store in variable e
            if i == attempts - 1:
                raise # raise the error if the last attempt is reached
            wait = 2 ** i
            print(f"    retry {i + 1}/{attempts - 1} after {type(e).__name__}; waiting {wait}s", flush=True)
            time.sleep(wait)

# Ensures API keys are only called when needed and saved after first use
_anthropic = None
def anthropic_client():
    global _anthropic
    if _anthropic is None:
        _anthropic = anthropic.Anthropic()
    return _anthropic

_openai = None
def openai_client():
    global _openai
    if _openai is None:
        _openai = OpenAI()
    return _openai

def perturb(document, replacements): # Builds the perturbed document
    pdoc = document
    for find, repl in replacements:
        assert find in pdoc, f"replacement find-string not present in passage: {find!r}"
        pdoc = pdoc.replace(find, repl)
    # Assert the passage actually changed to avoid misleading interpretations of the perturbed results
    assert pdoc != document, f"no change in passage detected for {replacements}"
    return pdoc

def appears(phrase, text):
    return re.search(r"\b" + re.escape(phrase) + r"\b", text, re.IGNORECASE) is not None # Returns true if phrase is present as a whole word ignoring capitalisation in model's answer, false if not

def step_doc(fact, step): # perturbs the fact's document based on the step
    base = doc_text(fact["doc"])
    return perturb(base, step["replace"]) if step["replace"] else base

def build_batch_message_params(model, system_instruction, question, doc, max_tokens=1200, cache_ttl="1h"):
    return {
        "model": model,
        "max_tokens": max_tokens,
        "system": system_instruction,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Passage:\n" + doc,
                 "cache_control": {"type": "ephemeral", "ttl": cache_ttl}},
                {"type": "text", "text": "\n\nQuestion: " + question},
            ],
        }],
    }

def extract_anthropic_text(message):
    return "".join(b.text for b in message.content if b.type == "text")

def submit_anthropic_batch(requests):
    reqs = [Request(custom_id=cid, params=MessageCreateParamsNonStreaming(**params)) for cid, params in requests]
    batch = with_retry(lambda: anthropic_client().messages.batches.create(requests=reqs))
    return batch.id

def poll_anthropic_batch(batch_id, poll_interval=30, on_poll=None):
    while True:
        batch = with_retry(lambda: anthropic_client().messages.batches.retrieve(batch_id))
        if on_poll:
            on_poll(batch)
        if batch.processing_status == "ended":
            return batch
        time.sleep(poll_interval)

def anthropic_batch_results(batch_id):
    for r in anthropic_client().messages.batches.results(batch_id):
        yield r.custom_id, r.result

def build_openai_batch_body(model, instructions, input_text, max_output_tokens=2000, text_format=None):
    body = {"model": model, "instructions": instructions, "input": input_text,
            "max_output_tokens": max_output_tokens, **openai_reasoning_kwargs(model)}
    if text_format is not None:
        body["text"] = {"format": text_format}
    return body

def submit_openai_batch(requests, endpoint="/v1/responses"):
    lines = "".join(json.dumps({"custom_id": cid, "method": "POST", "url": endpoint, "body": body}) + "\n"
                    for cid, body in requests)
    upload = with_retry(lambda: openai_client().files.create(file=("batch.jsonl", lines.encode()), purpose="batch"))
    batch = with_retry(lambda: openai_client().batches.create(
        input_file_id=upload.id, endpoint=endpoint, completion_window="24h"))
    return batch.id

OPENAI_BATCH_TERMINAL = {"completed", "failed", "expired", "cancelled"}

def poll_openai_batch(batch_id, poll_interval=30, on_poll=None):
    while True:
        batch = with_retry(lambda: openai_client().batches.retrieve(batch_id))
        if on_poll:
            on_poll(batch)
        if batch.status in OPENAI_BATCH_TERMINAL:
            return batch
        time.sleep(poll_interval)

def openai_batch_results(batch_id):
    batch = with_retry(lambda: openai_client().batches.retrieve(batch_id))
    for fid in (batch.output_file_id, batch.error_file_id):
        if not fid:
            continue
        content = with_retry(lambda: openai_client().files.content(fid)).text
        for line in content.splitlines():
            if line.strip():
                rec = json.loads(line)
                yield rec["custom_id"], rec

def extract_openai_text(body):
    parts = []
    for item in body.get("output", []):
        if item.get("type") == "message":
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    parts.append(c.get("text", ""))
    return "".join(parts)

def build_openai_candidate_body(model, system_instruction, question, doc, max_output_tokens=2000):
    return build_openai_batch_body(model, system_instruction,
        "Passage:\n" + doc + "\n\nQuestion: " + question, max_output_tokens)
