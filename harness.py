import json
def score_examples(examples):
    correct = 0
    for ex in examples:
        if ex["answer"] == ex["gold"]:
            correct = correct + 1
    total = len(examples)
    return correct / total

# test: 1 of 2 answers match, so the score must be 0.5
test_examples = [
    {"answer": "yes", "gold": "yes"},
    {"answer": "no", "gold": "yes"},
]
assert score_examples(test_examples) == 0.5
print("Test passed")
f = open("examples.json")
examples = json.load(f)
score = score_examples(examples)
print("Score:", round(score, 2))