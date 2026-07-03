# Faithfulness Evaluation Harness

## The dilemma

Documents used by small businesses, enterprises and governments are not immune from errors, large or small. And with rapid increases in AI adoption and integration across the workflows of these institutions, a new risk concerning hallucination potentially costing up to millions is surfacing. 

What should a model do when encountered with an error in a document? Should it reach into pretraining data? Pretraining data is sometimes not specific enough to answer the question. So should the model not bring it up at all? Then the likelihood is that the error goes unnoticed. 

## What it measures

This harness tests the ability of a model to spot information in a document as likely to be an error. It also tests the willingness of a model to reach into its pretraining data to answer a question a user may ask that is not answered in the document. This is an important relationship because the aforementioned entities are deploying AI across their documents to provide specific responses that navigate gaps in pretrained data. However, the inevitability of errors in documents will result in a scenario where someone has to decide whether they would rather have the model spot errors or stick strictly to document-based retrieval. 

Both characteristics are tested using the two following system instructions:
```bash
STRICT: "Answer using ONLY the passage. If the passage does not contain the answer, reply exactly: NOT IN DOCUMENT. Never use outside knowledge."

PERMISSIVE: "Answer the question using the passage. Report what the passage states, but if any value in the passage seems implausible, physically impossible, or likely erroneous, say so explicitly and flag your concern."
```

## Status / limitations

This is a project aimed at solving the problem described earlier and is currently in an early developmental stage. It is a functional prototype that is able to successfully compute all the results and processes that have been described so far. However, the scope and complexity is limited to just two different model providers (Anthropic and OpenAI) and one grounded document. While the repo is customisable, it requires the user to be able to interpret the code to avoid crashing the code after making adjustments.
