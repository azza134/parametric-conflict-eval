import anthropic
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
    "consent": "document1_consent.txt",
    "epl": "document2_epl.txt",
    "liquor": "document3_liquor.txt",
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

SYSTEM_INSTRUCTIONS = [("SOURCE_EXCLUSIVE", SOURCE_EXCLUSIVE), ("FLAG_INVITING", FLAG_INVITING), ("WEAK_GROUNDING", WEAK_GROUNDING), ("SOURCE_EXCLUSIVE_FLAG_INVITING", SOURCE_EXCLUSIVE_FLAG_INVITING)]

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
    if response.stop_reason == "max_tokens":
        print(f"    WARNING: answer truncated at max_tokens ({model})", flush=True)
    return "".join(b.text for b in response.content if b.type == "text") # Returns only text sections of the model output

def openai_reasoning(model):
    return {"reasoning": {"effort": "low"}} if model.startswith("gpt-5.4") else {}

def ask_openai(system_instruction, question, doc, model):
    reasoning = openai_reasoning(model)
    r = openai_client().responses.create(model=model, instructions=system_instruction,
        input="Passage:\n" + doc + "\n\nQuestion: " + question,
        max_output_tokens=2000, **reasoning)
    if r.status == "incomplete":
        print(f"    WARNING: answer truncated at max_output_tokens ({model})", flush=True)
    return r.output_text or ""

def call(model, provider, system, question, doc):
    if provider == "anthropic":
        return ask_anthropic(system, question, doc, model)
    return ask_openai(system, question, doc, model)

def call_docfree(model, provider, system, question):
    if provider == "anthropic":
        response = anthropic_client().messages.create(model=model, max_tokens=1200, system=system,
            messages=[{"role": "user", "content": question}])
        return "".join(b.text for b in response.content if b.type == "text")
    reasoning = openai_reasoning(model)
    r = openai_client().responses.create(model=model, instructions=system, input=question,
        max_output_tokens=2000, **reasoning)
    if r.status == "incomplete":
        print(f"    WARNING: answer truncated at max_output_tokens ({model})", flush=True)
    return r.output_text or ""

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

# Ensures API keys are only called when needed and saved after first use
_client = None
def anthropic_client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client

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
