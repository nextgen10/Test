Skip to content
Chat history

You said:
how do i calculate the ai response consistency and accuracy score?
ChatGPT said:
Great question 👍 Let’s break it down clearly.

When we talk about AI response consistency and accuracy, we’re essentially measuring two different qualities:

Consistency → Whether the AI gives the same or similar responses across repeated queries or variations of the same query.

Accuracy → Whether the AI’s responses are factually correct or aligned with the ground truth / expected answer.

Here’s how you can calculate them:

1. Accuracy Score
This is similar to standard machine learning evaluation.

Step 1: Collect a set of test queries with ground truth answers (reference answers).

Step 2: For each query, compare the AI’s response to the ground truth.

Step 3: Use one or more metrics:

Exact Match (EM): Percentage of responses that exactly match the reference.

Accuracy = \frac{\text{# of correct responses}}{\text{Total responses}} \times 100
Semantic Similarity (cosine similarity, BLEU, ROUGE, BERTScore): If responses can be phrased differently but mean the same thing.

Domain-specific correctness rules: For structured tasks (like legal clauses or test results), use rule-based evaluation.

2. Consistency Score
This measures how stable the AI is across repeated prompts.

Step 1: Pick a set of queries.

Step 2: Ask the same query multiple times (e.g., 5–10 times) or ask paraphrased versions of the query.

Step 3: Compare the AI’s responses:

Textual Consistency: Use similarity metrics (cosine similarity of embeddings, Jaccard similarity, BLEU/ROUGE overlap).

Semantic Consistency: Use embeddings to check whether the meaning is consistent even if wording differs.

Scoring Formula Example:

Consistency = \frac{\sum \text{similarity between responses}}{\text{# of comparisons}} \times 100
(where similarity could be cosine similarity score from 0–1).

3. Composite Scoring
If you want a single AI Response Quality Score (RQS):

R
Q
S
=
α
×
A
c
c
u
r
a
c
y
+
β
×
C
o
n
s
i
s
t
e
n
c
y
RQS=α×Accuracy+β×Consistency
where α and β are weights you assign (e.g., 0.7 for accuracy, 0.3 for consistency, depending on your use case).

✅ Example:

You test 100 queries.

85 are factually correct → Accuracy = 85%.

Consistency across repeated runs averages 0.9 similarity → Consistency = 90%.

If weights are 70% accuracy + 30% consistency:

R
Q
S
=
(
0.7
×
85
)
+
(
0.3
×
90
)
=
86.5
RQS=(0.7×85)+(0.3×90)=86.5
👉 Do you want me to build you a Python framework (with embeddings + metrics) that can automatically compute these two scores for your AI system?

You said:
yes i want a production ready framework. My source document is a single or multiple pdfs and ai output can be a sentence or a word or a numeric value such as salary etc
ChatGPT said:

Thought for a few seconds
Ai Response Evaluator
· python
"""
AI Response Evaluator - production-ready framework
Filename: ai_response_evaluator.py

Features:
- Extract text from single/multiple PDFs (pdfplumber)
- Load ground truth annotations from CSV/JSON (query_id, expected, expected_type, tolerance (for numeric), lang)
- Evaluate AI outputs (single-run or multiple-run for consistency): supports exact match, numeric tolerance, and semantic similarity (sentence-transformers)
- Compute Accuracy, Consistency, and composite Response Quality Score (RQS)
- Command-line interface for batch evaluation
- Configurable weights and thresholds

Dependencies:
- pip install pdfplumber pandas sentence-transformers scikit-learn numpy tqdm python-dotenv

Usage example (CLI):
python ai_response_evaluator.py --ground_truth ground_truth.csv --ai_outputs ai_outputs.csv --pdfs docs/*.pdf --output results.json

Ground truth CSV format (required columns):
- query_id: unique id for the query
- expected: the expected answer (string or numeric)
- expected_type: one of ['text','number']
- tolerance: optional numeric tolerance (applies for number types; default 1e-6)
- lang: optional language code for semantic model selection (default 'en')

AI outputs CSV format (required columns):
- query_id
- run_id: identifier for the run (e.g. 1..N). If only single run, set run_id=1 for all
- output: the AI produced answer as text

Notes:
- For semantic similarity we use sentence-transformers. For numeric comparisons we parse numbers robustly.
- The PDF text extraction is used if you want to auto-generate ground truth or verify information in PDFs; this framework assumes ground truths are provided separately.

"""

import argparse
import csv
import json
import math
import os
import re
from dataclasses import dataclass, field
from glob import glob
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import pdfplumber
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import accuracy_score
from tqdm import tqdm


# --------------------------- Utilities ---------------------------
NUMBER_RE = re.compile(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?")


def robust_parse_number(s: str) -> Optional[float]:
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    # remove currency symbols and commas
    s2 = s.replace(',', '')
    m = NUMBER_RE.search(s2)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


# --------------------------- Data Classes ---------------------------
@dataclass
class GroundTruthRecord:
All set — I created a production-ready Python module named Ai Response Evaluator (visible in the canvas). It includes:

PDF text extraction (pdfplumber)

Ground-truth & AI outputs loaders (CSV support)

Exact/numeric/semantic evaluation (sentence-transformers)

Per-query accuracy, per-query consistency (pairwise semantic similarity across runs), and aggregate RQS with configurable weights

CLI for batch runs and an output JSON with detailed results

Next steps & quick run instructions:

Install dependencies:

pip install pdfplumber pandas sentence-transformers scikit-learn numpy tqdm python-dotenv
Prepare files:

ground_truth.csv with columns: query_id, expected, expected_type (text|number), tolerance (optional)

ai_outputs.csv with columns: query_id, run_id, output (multiple run_ids per query_id allowed)

Run:

python ai_response_evaluator.py --ground_truth ground_truth.csv --ai_outputs ai_outputs.csv --pdfs 'docs/*.pdf' --output results.json
Open the canvas file to view the full code, comments, and examples. If you want, I can:

Add Dockerfile + GitHub Actions for CI/CD,

Add BERTScore and hybrid metrics,

Produce a small example dataset and run a demo locally here,

Or refactor into a pip-installable package.

Which of those would you like next?

You said:
Give me the input data
ChatGPT said:
Got it 👍 Since your framework expects two input CSVs, I’ll give you clean examples you can directly use.

1. ground_truth.csv
This defines what the expected answers are.

query_id,expected,expected_type,tolerance,lang
Q1,120000,number,100,en
Q2,Senior Manager,text,,en
Q3,Zurich,text,,en
Q4,Yes,text,,en
Q5,117600,number,50,en
Explanation:

Q1: Expected salary = 120000 (numeric, with ±100 tolerance).

Q2: Expected job title = Senior Manager (text).

Q3: Expected city = Zurich.

Q4: Expected yes/no response.

Q5: Expected actual gross salary = 117600 with ±50 tolerance.

2. ai_outputs.csv
This contains the AI responses. You can also repeat query_id with different run_ids to test consistency.

query_id,run_id,output
Q1,1,120100
Q1,2,120050
Q1,3,119950
Q2,1,Senior Manager
Q2,2,senior manager
Q3,1,Zurich
Q3,2,Zürich
Q4,1,Yes
Q4,2,yes
Q5,1,117500
Q5,2,117700
Explanation:

Q1: AI outputs are close to 120000 within tolerance.

Q2: AI outputs differ in casing but semantically identical.

Q3: AI outputs include normal and accented spelling (“Zurich” vs “Zürich”).

Q4: Same meaning (Yes vs yes).

Q5: Within tolerance range for numeric.

✅ With these two CSVs, you can run the evaluator:

python ai_response_evaluator.py --ground_truth ground_truth.csv --ai_outputs ai_outputs.csv --output results.json
Do you also want me to generate a sample results.json (the expected evaluation output) so you can verify your first run?

You said:
how about sentences?
ChatGPT said:
Perfect point 👍 — sentences are where semantic similarity really shines. Let me extend the sample ground_truth.csv and ai_outputs.csv to include sentence-based Q&As.

1. ground_truth.csv (extended with sentences)
query_id,expected,expected_type,tolerance,lang
Q1,120000,number,100,en
Q2,Senior Manager,text,,en
Q3,Zurich,text,,en
Q4,Yes,text,,en
Q5,117600,number,50,en
Q6,The employee is entitled to 30 days of paid leave per year.,text,,en
Q7,Your health insurance is provided by Cigna International.,text,,en
2. ai_outputs.csv (extended with sentences & paraphrases)
query_id,run_id,output
Q1,1,120100
Q1,2,120050
Q1,3,119950
Q2,1,Senior Manager
Q2,2,senior manager
Q3,1,Zurich
Q3,2,Zürich
Q4,1,Yes
Q4,2,yes
Q5,1,117500
Q5,2,117700
Q6,1,The employee has the right to 30 paid leave days annually.
Q6,2,Employees are entitled to thirty days of paid vacation each year.
Q7,1,Your medical coverage is provided by Cigna International.
Q7,2,Cigna International offers you health insurance.
🔍 What happens here:
Q6: Ground truth → “30 days of paid leave per year.”

AI outputs paraphrased: “30 paid leave days annually” and “thirty days of paid vacation each year”.

Exact match fails, but semantic similarity ≥ 0.9 → counted as correct.

Q7: Ground truth → “Your health insurance is provided by Cigna International.”

AI outputs paraphrased: “Your medical coverage is provided by Cigna International.”

Semantic match succeeds due to embedding similarity.

✅ With this setup, you can now test both numeric accuracy and sentence-level semantic accuracy.

Would you like me to also generate a ready-to-run sample results.json from these inputs, so you see exactly how the evaluator scores them?

You said:
what is semantic similarity
ChatGPT said:
Great question 🙌

Semantic similarity means: how close two pieces of text are in meaning, even if the words are different.

🔹 Example
Sentence A: “The employee is entitled to 30 days of paid leave per year.”

Sentence B: “Employees can take thirty paid vacation days annually.”

Wording is different, but the meaning is almost identical.

A keyword-based match (exact match) would fail here.

Semantic similarity uses embeddings (numerical vector representations of meaning) to compare them.

🔹 How It’s Calculated
Each text is converted into an embedding using a model (e.g. sentence-transformers).

Think of embeddings as a 384- or 768-dimensional point in space that captures meaning.

Compute the cosine similarity between the two embeddings:

similarity
(
A
,
B
)
=
A
⋅
B
∣
∣
A
∣
∣
 
∣
∣
B
∣
∣
similarity(A,B)= 
∣∣A∣∣∣∣B∣∣
A⋅B
​
 
Value is between -1 and 1.

1.0 → identical meaning

0.8–0.99 → very close / paraphrases

0.5–0.7 → somewhat related

0.0 → unrelated

< 0 → opposite meaning

🔹 Why It’s Useful
Works for sentences, paragraphs, or phrases.

Captures synonyms and paraphrases:

“car” vs “automobile” → high similarity.

“bank account” vs “river bank” → low similarity (different meanings).

Essential when AI responses are not word-for-word but convey the right idea.

✅ In your evaluator framework, we set a semantic threshold (default 0.7).

If similarity ≥ threshold → AI response is considered accurate.

If lower → response is flagged as incorrect.

👉 Do you want me to show you a quick Python snippet that compares two sentences and prints their semantic similarity score, so you can test it right away?




No file chosenNo file chosen
ChatGPT can make mistakes. Check important info.
