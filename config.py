import anthropic
import re
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

MODELS = [("gpt-4o-mini", "openai"), ("gpt-5.4-nano", "openai")]
JUDGE_MODEL = "gpt-5.4-mini"  # LLM judge, ideally from a different model provider to the candidate model
GOLD_CANDIDATE = ("claude-sonnet-5", "anthropic") # Model to be used for generating answers in the gold set
N_PER_CELL = 8

passage = open("document.txt").read()

# System Instructions
SOURCE_EXCLUSIVE = (
    "Answer using ONLY the passage. If the passage does not contain the answer, "
    "reply exactly: NOT IN DOCUMENT. Never use outside knowledge."
)
FLAG_INVITING = ("Answer the question using the passage. Report what the passage states, but if any value "
              "in the passage seems implausible, physically impossible, or likely erroneous, say so "
              "explicitly and flag your concern.")
WEAK_GROUNDING = "Base your answer on the passage provided."

SYSTEM_INSTRUCTIONS = [("SOURCE_EXCLUSIVE", SOURCE_EXCLUSIVE), ("FLAG_INVITING", FLAG_INVITING), ("WEAK_GROUNDING", WEAK_GROUNDING)]

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

def ask_openai(system_instruction, question, doc, model):
    reasoning = {"reasoning": {"effort": "low"}} if model.startswith("gpt-5") else {}
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
        pdoc = pdoc.replace(find, repl)
    # Assert the passage actually changed to avoid misleading interpretations of the perturbed results
    assert pdoc != document, f"no change in passage detected for {replacements}"
    return pdoc

def appears(phrase, text):
    return re.search(r"\b" + re.escape(phrase) + r"\b", text, re.IGNORECASE) is not None # Returns true if phrase is present as a whole word ignoring capitalisation in model's answer, false if not

def step_doc(step):
    return perturb(passage, step["replace"]) if step["replace"] else passage
