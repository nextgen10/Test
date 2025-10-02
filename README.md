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



-------------------------------

```
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
    query_id: str
    expected: str
    expected_type: str = 'text'  # 'text' or 'number'
    tolerance: float = 1e-6
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AIOutputRecord:
    query_id: str
    run_id: str
    output: str


# --------------------------- PDF Extraction ---------------------------
class PDFExtractor:
    @staticmethod
    def extract_text_from_pdf(path: str) -> str:
        all_text = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    all_text.append(text)
        return "\n".join(all_text)

    @staticmethod
    def extract_text_from_pdfs(paths: List[str]) -> Dict[str, str]:
        result = {}
        for p in paths:
            result[p] = PDFExtractor.extract_text_from_pdf(p)
        return result


# --------------------------- Loader Functions ---------------------------
class GroundTruthLoader:
    @staticmethod
    def from_csv(path: str) -> Dict[str, GroundTruthRecord]:
        df = pd.read_csv(path, dtype=str).fillna('')
        records = {}
        for _, row in df.iterrows():
            qid = str(row['query_id'])
            expected = str(row['expected'])
            expected_type = str(row.get('expected_type', 'text'))
            tol = row.get('tolerance', '')
            try:
                tolerance = float(tol) if tol != '' else 1e-6
            except Exception:
                tolerance = 1e-6
            meta = {}
            # include other columns as metadata
            for c in df.columns:
                if c not in ['query_id', 'expected', 'expected_type', 'tolerance']:
                    meta[c] = row[c]
            records[qid] = GroundTruthRecord(qid, expected, expected_type, tolerance, meta)
        return records


class AIOutputsLoader:
    @staticmethod
    def from_csv(path: str) -> List[AIOutputRecord]:
        df = pd.read_csv(path, dtype=str).fillna('')
        records = []
        for _, row in df.iterrows():
            records.append(AIOutputRecord(str(row['query_id']), str(row.get('run_id', '1')), str(row['output'])))
        return records


# --------------------------- Evaluator ---------------------------
class Evaluator:
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', device: str = None):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)

    # Exact match (case-insensitive, trimmed)
    @staticmethod
    def exact_match(a: str, b: str) -> bool:
        if a is None or b is None:
            return False
        return a.strip().lower() == b.strip().lower()

    # Numeric comparison with tolerance
    @staticmethod
    def numeric_match(a: str, b: str, tol: float = 1e-6) -> bool:
        na = robust_parse_number(a)
        nb = robust_parse_number(b)
        if na is None or nb is None:
            return False
        return math.isclose(na, nb, rel_tol=tol, abs_tol=tol)

    # Semantic similarity using embeddings
    def semantic_similarity(self, a: str, b: str) -> float:
        if (not a) or (not b):
            return 0.0
        emb = self.model.encode([a, b], convert_to_tensor=True)
        score = util.pytorch_cos_sim(emb[0], emb[1]).item()
        return float(score)  # in [-1,1]

    # Evaluate single-run: returns dict of scores for each query
    def evaluate_once(self, ground_truths: Dict[str, GroundTruthRecord], ai_outputs: List[AIOutputRecord],
                      semantic_threshold: float = 0.7) -> Dict[str, Dict[str, Any]]:
        # Group outputs by query_id, pick run_id '1' if multiple runs present
        chosen = {}
        for out in ai_outputs:
            if out.query_id not in chosen:
                chosen[out.query_id] = out
        results = {}
        for qid, gt in ground_truths.items():
            out = chosen.get(qid)
            if out is None:
                results[qid] = {
                    'found': False,
                    'match_type': None,
                    'accuracy': 0,
                    'semantic_score': 0.0,
                    'expected': gt.expected,
                    'output': None
                }
                continue
            output = out.output
            if gt.expected_type == 'number':
                nm = self.numeric_match(output, gt.expected, gt.tolerance)
                results[qid] = {
                    'found': True,
                    'match_type': 'numeric',
                    'accuracy': 1 if nm else 0,
                    'semantic_score': None,
                    'expected': gt.expected,
                    'output': output
                }
            else:
                em = self.exact_match(output, gt.expected)
                sem = self.semantic_similarity(output, gt.expected)
                acc = 1 if (em or sem >= semantic_threshold) else 0
                results[qid] = {
                    'found': True,
                    'match_type': 'text',
                    'accuracy': acc,
                    'exact_match': bool(em),
                    'semantic_score': sem,
                    'expected': gt.expected,
                    'output': output
                }
        return results

    # Consistency: given multiple runs per query_id, compute average pairwise similarity of outputs
    def evaluate_consistency(self, ai_outputs: List[AIOutputRecord]) -> Dict[str, float]:
        # group by query_id
        from collections import defaultdict
        groups = defaultdict(list)
        for out in ai_outputs:
            groups[out.query_id].append(out.output)
        consistency_scores = {}
        for qid, outputs in groups.items():
            n = len(outputs)
            if n <= 1:
                consistency_scores[qid] = 1.0  # single output considered fully consistent
                continue
            # compute pairwise semantic similarities
            emb = self.model.encode(outputs, convert_to_tensor=True)
            sims = util.pytorch_cos_sim(emb, emb).cpu().numpy()
            # only consider upper triangle without diagonal
            pairs = []
            for i in range(n):
                for j in range(i + 1, n):
                    pairs.append(sims[i, j])
            if not pairs:
                consistency = 1.0
            else:
                consistency = float(np.mean(pairs))  # in [-1,1]
            # normalize from [-1,1] to [0,1]
            consistency_scores[qid] = (consistency + 1) / 2
        return consistency_scores

    # Compute overall metrics
    def aggregate_scores(self, single_eval: Dict[str, Dict[str, Any]], consistency_scores: Optional[Dict[str, float]] = None,
                         alpha: float = 0.7, beta: float = 0.3) -> Dict[str, Any]:
        qids = list(single_eval.keys())
        accs = [single_eval[q]['accuracy'] for q in qids]
        accuracy = float(np.mean(accs)) if accs else 0.0
        if consistency_scores is None:
            consistency = 1.0
        else:
            cs = [consistency_scores.get(q, 1.0) for q in qids]
            consistency = float(np.mean(cs))
        rqs = alpha * accuracy + beta * consistency
        return {
            'accuracy': accuracy,
            'consistency': consistency,
            'rqs': rqs,
            'alpha': alpha,
            'beta': beta,
            'n_queries': len(qids)
        }


# --------------------------- CLI ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description='AI Response Evaluation Framework')
    p.add_argument('--ground_truth', required=True, help='CSV file with ground truth')
    p.add_argument('--ai_outputs', required=True, help='CSV file with AI outputs')
    p.add_argument('--pdfs', nargs='*', help='Optional PDF(s) to extract text from')
    p.add_argument('--model', default='sentence-transformers/all-MiniLM-L6-v2', help='SentenceTransformer model')
    p.add_argument('--semantic_threshold', type=float, default=0.72, help='Threshold for semantic match')
    p.add_argument('--alpha', type=float, default=0.7, help='Weight for accuracy in RQS')
    p.add_argument('--beta', type=float, default=0.3, help='Weight for consistency in RQS')
    p.add_argument('--output', default='results.json', help='Path to write results JSON')
    return p.parse_args()


def main():
    args = parse_args()
    gt = GroundTruthLoader.from_csv(args.ground_truth)
    ai_outs = AIOutputsLoader.from_csv(args.ai_outputs)

    if args.pdfs:
        pdf_paths = []
        for p in args.pdfs:
            pdf_paths.extend(glob(p))
        pdf_texts = PDFExtractor.extract_text_from_pdfs(pdf_paths)
        # optionally write extracted pdf text for human verification
        for k, v in pdf_texts.items():
            outp = os.path.splitext(os.path.basename(k))[0] + '.extracted.txt'
            with open(outp, 'w', encoding='utf-8') as f:
                f.write(v)
            print(f'Extracted text written to {outp}')

    evaluator = Evaluator(model_name=args.model)
    single_eval = evaluator.evaluate_once(gt, ai_outs, semantic_threshold=args.semantic_threshold)
    consistency = evaluator.evaluate_consistency(ai_outs)
    agg = evaluator.aggregate_scores(single_eval, consistency_scores=consistency, alpha=args.alpha, beta=args.beta)

    results = {
        'per_query': single_eval,
        'consistency_per_query': consistency,
        'aggregate': agg
    }

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'Results written to {args.output}')


if __name__ == '__main__':
    main()

```



#############UPDATED CODE###########

```
"""
AI Response Evaluator - production-ready framework
Filename: ai_response_evaluator.py

Features:
- Extract text from single/multiple PDFs (pdfplumber)
- Load ground truth annotations from CSV/JSON (query_id, expected, expected_type, tolerance (for numeric), lang)
- Evaluate AI outputs (single-run or multiple-run for consistency): supports exact match, numeric tolerance, semantic similarity (sentence-transformers), and PDF-grounded checks
- Compute Accuracy, Consistency, PDF Support, and composite Response Quality Score (RQS)
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
"""

import argparse
import json
import math
import os
import re
from dataclasses import dataclass, field
from glob import glob
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import pdfplumber
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# --------------------------- Utilities ---------------------------
NUMBER_RE = re.compile(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?")


def robust_parse_number(s: str) -> Optional[float]:
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
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
    query_id: str
    expected: str
    expected_type: str = 'text'
    tolerance: float = 1e-6
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AIOutputRecord:
    query_id: str
    run_id: str
    output: str


# --------------------------- PDF Extraction ---------------------------
class PDFExtractor:
    @staticmethod
    def extract_text_from_pdf(path: str) -> str:
        all_text = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    all_text.append(text)
        return "\n".join(all_text)

    @staticmethod
    def extract_text_from_pdfs(paths: List[str]) -> Dict[str, str]:
        result = {}
        for p in paths:
            result[p] = PDFExtractor.extract_text_from_pdf(p)
        return result


# --------------------------- Loader Functions ---------------------------
class GroundTruthLoader:
    @staticmethod
    def from_csv(path: str) -> Dict[str, GroundTruthRecord]:
        df = pd.read_csv(path, dtype=str).fillna('')
        records = {}
        for _, row in df.iterrows():
            qid = str(row['query_id'])
            expected = str(row['expected'])
            expected_type = str(row.get('expected_type', 'text'))
            tol = row.get('tolerance', '')
            try:
                tolerance = float(tol) if tol != '' else 1e-6
            except Exception:
                tolerance = 1e-6
            meta = {}
            for c in df.columns:
                if c not in ['query_id', 'expected', 'expected_type', 'tolerance']:
                    meta[c] = row[c]
            records[qid] = GroundTruthRecord(qid, expected, expected_type, tolerance, meta)
        return records


class AIOutputsLoader:
    @staticmethod
    def from_csv(path: str) -> List[AIOutputRecord]:
        df = pd.read_csv(path, dtype=str).fillna('')
        records = []
        for _, row in df.iterrows():
            records.append(AIOutputRecord(str(row['query_id']), str(row.get('run_id', '1')), str(row['output'])))
        return records


# --------------------------- Evaluator ---------------------------
class Evaluator:
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', device: str = None):
        self.model = SentenceTransformer(model_name, device=device)

    @staticmethod
    def exact_match(a: str, b: str) -> bool:
        if a is None or b is None:
            return False
        return a.strip().lower() == b.strip().lower()

    @staticmethod
    def numeric_match(a: str, b: str, tol: float = 1e-6) -> bool:
        na = robust_parse_number(a)
        nb = robust_parse_number(b)
        if na is None or nb is None:
            return False
        return math.isclose(na, nb, rel_tol=tol, abs_tol=tol)

    def semantic_similarity(self, a: str, b: str) -> float:
        if (not a) or (not b):
            return 0.0
        emb = self.model.encode([a, b], convert_to_tensor=True)
        score = util.pytorch_cos_sim(emb[0], emb[1]).item()
        return float(score)

    def check_in_pdf(self, output: str, pdf_texts: Dict[str, str], threshold: float = 0.75) -> Optional[bool]:
        if not pdf_texts:
            return None
        if not output:
            return False
        texts = list(pdf_texts.values())
        emb_out = self.model.encode(output, convert_to_tensor=True)
        emb_pdfs = self.model.encode(texts, convert_to_tensor=True)
        sims = util.pytorch_cos_sim(emb_out, emb_pdfs).cpu().numpy()
        return bool(np.max(sims) >= threshold)

    def evaluate_once(self, ground_truths: Dict[str, GroundTruthRecord], ai_outputs: List[AIOutputRecord],
                      semantic_threshold: float = 0.7, pdf_texts: Optional[Dict[str, str]] = None) -> Dict[str, Dict[str, Any]]:
        chosen = {}
        for out in ai_outputs:
            if out.query_id not in chosen:
                chosen[out.query_id] = out

        results = {}
        for qid, gt in ground_truths.items():
            out = chosen.get(qid)
            if out is None:
                results[qid] = {
                    'found': False,
                    'match_type': None,
                    'accuracy': 0,
                    'semantic_score': 0.0,
                    'expected': gt.expected,
                    'output': None,
                    'in_pdf': None
                }
                continue

            output = out.output
            in_pdf = self.check_in_pdf(output, pdf_texts) if pdf_texts else None

            if gt.expected_type == 'number':
                nm = self.numeric_match(output, gt.expected, gt.tolerance)
                results[qid] = {
                    'found': True,
                    'match_type': 'numeric',
                    'accuracy': 1 if nm else 0,
                    'semantic_score': None,
                    'expected': gt.expected,
                    'output': output,
                    'in_pdf': in_pdf
                }
            else:
                em = self.exact_match(output, gt.expected)
                sem = self.semantic_similarity(output, gt.expected)
                acc = 1 if (em or sem >= semantic_threshold) else 0
                results[qid] = {
                    'found': True,
                    'match_type': 'text',
                    'accuracy': acc,
                    'exact_match': bool(em),
                    'semantic_score': sem,
                    'expected': gt.expected,
                    'output': output,
                    'in_pdf': in_pdf
                }
        return results

    def evaluate_consistency(self, ai_outputs: List[AIOutputRecord]) -> Dict[str, float]:
        from collections import defaultdict
        groups = defaultdict(list)
        for out in ai_outputs:
            groups[out.query_id].append(out.output)
        consistency_scores = {}
        for qid, outputs in groups.items():
            n = len(outputs)
            if n <= 1:
                consistency_scores[qid] = 1.0
                continue
            emb = self.model.encode(outputs, convert_to_tensor=True)
            sims = util.pytorch_cos_sim(emb, emb).cpu().numpy()
            pairs = [sims[i, j] for i in range(n) for j in range(i + 1, n)]
            consistency = float(np.mean(pairs)) if pairs else 1.0
            consistency_scores[qid] = (consistency + 1) / 2
        return consistency_scores

    def aggregate_scores(self, single_eval: Dict[str, Dict[str, Any]], consistency_scores: Optional[Dict[str, float]] = None,
                         alpha: float = 0.6, beta: float = 0.2, gamma: float = 0.2) -> Dict[str, Any]:
        qids = list(single_eval.keys())

        accs, pdf_flags = [], []
        for q in qids:
            res = single_eval[q]
            acc = res['accuracy']
            if res.get('in_pdf') is False:
                acc = 0
            accs.append(acc)
            if res.get('in_pdf') is not None:
                pdf_flags.append(1 if res['in_pdf'] else 0)

        accuracy = float(np.mean(accs)) if accs else 0.0
        consistency = float(np.mean([consistency_scores.get(q, 1.0) for q in qids])) if consistency_scores else 1.0
        pdf_support = float(np.mean(pdf_flags)) if pdf_flags else 1.0

        rqs = alpha * accuracy + beta * consistency + gamma * pdf_support

        return {
            'accuracy': accuracy,
            'consistency': consistency,
            'pdf_support_rate': pdf_support,
            'rqs': rqs,
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma,
            'n_queries': len(qids)
        }


# --------------------------- CLI ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description='AI Response Evaluation Framework')
    p.add_argument('--ground_truth', required=True, help='CSV file with ground truth')
    p.add_argument('--ai_outputs', required=True, help='CSV file with AI outputs')
    p.add_argument('--pdfs', nargs='*', help='Optional PDF(s) to extract text from')
    p.add_argument('--model', default='sentence-transformers/all-MiniLM-L6-v2', help='SentenceTransformer model')
    p.add_argument('--semantic_threshold', type=float, default=0.72, help='Threshold for semantic match')
    p.add_argument('--alpha', type=float, default=0.6, help='Weight for accuracy in RQS')
    p.add_argument('--beta', type=float, default=0.2, help='Weight for consistency in RQS')
    p.add_argument('--gamma', type=float, default=0.2, help='Weight for PDF support in RQS')
    p.add_argument('--output', default='results.json', help='Path to write results JSON')
    return p.parse_args()


def main():
    args = parse_args()
    gt = GroundTruthLoader.from_csv(args.ground_truth)
    ai_outs = AIOutputsLoader.from_csv(args.ai_outputs)

    pdf_texts = None
    if args.pdfs:
        pdf_paths = []
        for p in args.pdfs:
            pdf_paths.extend(glob(p))
        pdf_texts = PDFExtractor.extract_text_from_pdfs(pdf_paths)
        for k, v in pdf_texts.items():
            outp = os.path.splitext(os.path.basename(k))[0] + '.extracted.txt'
            with open(outp, 'w', encoding='utf-8') as f:
                f.write(v)
            print(f'Extracted text written to {outp}')

    evaluator = Evaluator(model_name=args.model)
    single_eval = evaluator.evaluate_once(gt, ai_outs, semantic_threshold=args.semantic_threshold, pdf_texts=pdf_texts)
    consistency = evaluator.evaluate_consistency(ai_outs)
    agg = evaluator.aggregate_scores(single_eval, consistency_scores=consistency,
                                     alpha=args.alpha, beta=args.beta, gamma=args.gamma)

    results = {
        'per_query': single_eval,
        'consistency_per_query': consistency,
        'aggregate': agg
    }

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'Results written to {args.output}')


if __name__ == '__main__':
    main()

```





##NEw CODE

```
#!/usr/bin/env python3
"""
ai_assurance_app.py
Production-ready evaluator for agent outputs (JSON LUMA messages) vs PDF sources.

Features:
- Input: ai_outputs.json (LUMA messages), ground_truth.csv or ground_truth.json, PDFs
- Chroma vector store (robust client creation across versions)
- Metrics: exact, numeric, date normalization, semantic similarity (SBERT),
  BERTScore, ROUGE, BLEU, METEOR, consistency, RQS
- Optional: Azure OpenAI LLM judge and toxicity scoring
- Outputs: results.json, results.csv, results.html
- Sample data generator: --generate-sample

Author: ChatGPT (GPT-5 Thinking mini)
"""

import argparse
import base64
import csv
import html
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from glob import glob
from io import BytesIO
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdfplumber
import sacrebleu
import nltk
from dateutil import parser as dateparser
from nltk.translate import meteor_score
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
from bert_score import score as bert_score

# try to import chroma in a way compatible with various versions
try:
    from chromadb.utils import embedding_functions
    try:
        # modern preferred client
        from chromadb import PersistentClient as ChromaPersistentClient  # type: ignore
        CHROMA_MODE = "persistent"
    except Exception:
        try:
            from chromadb import Client as ChromaClient  # type: ignore
            CHROMA_MODE = "client"
        except Exception:
            ChromaClient = None
            CHROMA_MODE = None
except Exception:
    embedding_functions = None
    CHROMA_MODE = None

# optional OpenAI Azure client
try:
    from openai import OpenAI as AzureOpenAIClient  # openai-python 1.x style
    HAVE_AZURE_OPENAI = True
except Exception:
    AzureOpenAIClient = None
    HAVE_AZURE_OPENAI = False

# optional toxicity model
try:
    from transformers import pipeline as hf_pipeline
    HAVE_TOXICITY = True
except Exception:
    HAVE_TOXICITY = False

# Ensure nltk punkt available
try:
    nltk.data.find("tokenizers/punkt")
except Exception:
    nltk.download("punkt")

# ---------------------- Utilities ----------------------
NUMBER_RE = re.compile(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?")
DATE_CANDIDATE_RE = re.compile(r"\b\d{1,4}[-/\.]\d{1,2}[-/\.]\d{1,4}\b")

def robust_parse_number(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    s2 = str(s)
    # remove currency symbols & thousand separators but keep minus/decimal
    s2 = re.sub(r"[^\d\.\-eE]", "", s2)
    m = NUMBER_RE.search(s2)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None

def normalize_date_string(s: Optional[str]) -> Optional[str]:
    if not s or not isinstance(s, str):
        return None
    try:
        dt = dateparser.parse(s, dayfirst=False, fuzzy=True)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None

def normalize_email(s: Optional[str]) -> str:
    if not s:
        return ""
    s = s.strip().lower()
    s = s.replace(" at ", "@").replace(" dot ", ".")
    s = s.replace("(at)", "@").replace("[at]", "@")
    return s

def to_serializable(o):
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (set,)):
        return list(o)
    return str(o)

# ---------------------- Data Classes ----------------------
@dataclass
class GroundTruthRecord:
    query_id: str
    expected: str
    expected_type: str = "text"
    tolerance: float = 1e-6
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AIOutputRecord:
    query_id: str
    run_id: str
    output: str

# ---------------------- PDF Extraction ----------------------
class PDFExtractor:
    @staticmethod
    def extract_text_from_pdf(path: str) -> str:
        out = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    out.append(text)
        return "\n".join(out)

    @staticmethod
    def extract_texts(paths: List[str]) -> Dict[str, str]:
        res = {}
        for p in paths:
            res[p] = PDFExtractor.extract_text_from_pdf(p)
        return res

# ---------------------- Chroma Vector Store Wrapper ----------------------
class PDFVectorStore:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", collection_name: str = "pdfs", persist_dir: str = "./chroma_db"):
        if CHROMA_MODE is None or embedding_functions is None:
            raise RuntimeError("chromadb or required submodules not installed")
        os.makedirs(persist_dir, exist_ok=True)

        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)

        # create client robustly across versions
        self.collection = None
        if CHROMA_MODE == "persistent":
            # PersistentClient available
            try:
                from chromadb import PersistentClient  # type: ignore
                self.client = PersistentClient(path=persist_dir)
            except Exception as e:
                raise RuntimeError("Unable to create PersistentClient: " + str(e))
        else:
            # older Client API
            try:
                from chromadb import Client  # type: ignore
                self.client = Client(Settings=None) if 'Settings' in globals() else Client()
            except Exception:
                # fallback attempt
                from chromadb import Client  # type: ignore
                self.client = Client()

        # create or get collection with embedding function wrapper
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model)
        existing = [c.name for c in self.client.list_collections()]
        if collection_name in existing:
            self.collection = self.client.get_collection(collection_name)
        else:
            self.collection = self.client.create_collection(name=collection_name, embedding_function=ef)

    def add_pdf_texts(self, pdf_texts: Dict[str, str], chunk_sentence_max: int = 2):
        for path, text in pdf_texts.items():
            # split into sentences and group (small chunks) to provide more context per embedding
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
            # group sentences into small chunks
            docs = []
            ids = []
            idx = 0
            for i in range(0, len(sentences), chunk_sentence_max):
                chunk = " ".join(sentences[i:i + chunk_sentence_max])
                docs.append(chunk)
                ids.append(f"{os.path.basename(path)}_{idx}")
                idx += 1
            if docs:
                self.collection.add(documents=docs, ids=ids)

    def query(self, text: str, top_k: int = 1) -> float:
        if not text:
            return 0.0
        res = self.collection.query(query_texts=[text], n_results=top_k)
        # results format depends on chroma version; handle common structure
        if isinstance(res, dict):
            dists = res.get("distances") or []
        else:
            # if object with attribute
            dists = getattr(res, "distances", [])
        if dists and len(dists) > 0 and len(dists[0]) > 0:
            # chroma distances are cosine distances -> similarity = 1 - distance
            return 1.0 - dists[0][0]
        return 0.0

# ---------------------- Evaluator ----------------------
class Evaluator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None, pdf_vector_store: Optional[PDFVectorStore] = None, enable_toxicity: bool = False, enable_llm_judge: bool = False):
        # allow local path detection
        if os.path.exists(model_name):
            model_id = model_name
        else:
            model_id = model_name if model_name.startswith("sentence-transformers/") else f"sentence-transformers/{model_name}"
        self.model = SentenceTransformer(model_id, device=device)
        self.rouge = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
        self.pdf_vector_store = pdf_vector_store
        self.enable_toxicity = enable_toxicity and HAVE_TOXICITY
        self.enable_llm_judge = enable_llm_judge and HAVE_AZURE_OPENAI

        if self.enable_toxicity:
            # unitary/toxic-bert or similar; return_all_scores True gives all labels
            self.toxicity_pipe = hf_pipeline("text-classification", model="unitary/toxic-bert", return_all_scores=True)

        if self.enable_llm_judge:
            # init azure client from env vars
            AZ_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
            AZ_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
            AZ_DEPLOY = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")
            AZ_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2023-07-01-preview")
            if not AZ_KEY or not AZ_ENDPOINT or not AZ_DEPLOY:
                raise RuntimeError("Azure OpenAI env vars missing (AZURE_OPENAI_API_KEY/ENDPOINT/DEPLOYMENT_NAME)")
            self.azure_client = AzureOpenAIClient(api_key=AZ_KEY, api_base=AZ_ENDPOINT, api_type="azure", api_version=AZ_VERSION)
            self.azure_deployment = AZ_DEPLOY

    # simple matchers
    @staticmethod
    def exact_match(a: Optional[str], b: Optional[str]) -> bool:
        if a is None or b is None:
            return False
        return a.strip().lower() == b.strip().lower()

    @staticmethod
    def numeric_match(a: str, b: str, tol: float = 1e-6) -> bool:
        na = robust_parse_number(a)
        nb = robust_parse_number(b)
        if na is None or nb is None:
            return False
        return math.isclose(na, nb, abs_tol=tol)

    def semantic_similarity(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        emb = self.model.encode([a, b], convert_to_tensor=True)
        return float(util.pytorch_cos_sim(emb[0], emb[1]).item())

    @staticmethod
    def bertscore_eval(candidate: str, reference: str, lang: str = "en") -> Dict[str, float]:
        P, R, F1 = bert_score([candidate], [reference], lang=lang, verbose=False)
        return {'bertscore_precision': float(P[0].cpu().numpy()), 'bertscore_recall': float(R[0].cpu().numpy()), 'bertscore_f1': float(F1[0].cpu().numpy())}

    def rouge_eval(self, candidate: str, reference: str) -> Dict[str, float]:
        scored = self.rouge.score(reference, candidate)
        return {'rouge1': scored['rouge1'].fmeasure, 'rouge2': scored['rouge2'].fmeasure, 'rougeL': scored['rougeL'].fmeasure}

    @staticmethod
    def bleu_eval(candidate: str, reference: str) -> float:
        try:
            return float(sacrebleu.sentence_bleu(candidate, [reference]).score)
        except Exception:
            return 0.0

    @staticmethod
    def meteor_eval(candidate: str, reference: str) -> float:
        try:
            return float(meteor_score.single_meteor_score(reference, candidate))
        except Exception:
            return 0.0

    def evaluate_toxicity(self, text: str) -> float:
        if not self.enable_toxicity or not text:
            return 0.0
        outs = self.toxicity_pipe(text)
        # compute a toxic score = max(prob of toxic labels)
        toks = []
        for label_scores in outs:
            for rec in label_scores:
                lab = rec.get("label", "").lower()
                sc = rec.get("score", 0.0)
                if lab not in ("not_toxic", "cleaned", "non-toxic"):
                    toks.append(sc)
        return float(max(toks)) if toks else 0.0

    def call_llm_judge(self, candidate: str, reference: str) -> Optional[float]:
        if not self.enable_llm_judge:
            return None
        prompt = (
            "You are an automated evaluator. Given a ground truth and a candidate answer,\n"
            "return a JSON object with a single field 'score' with a float from 0.0 to 1.0\n"
            "representing how correct the candidate is (1.0 = perfect). Only return JSON.\n\n"
            f"GROUND_TRUTH: {reference}\n\nCANDIDATE: {candidate}\n\nReturn:"
        )
        # Azure OpenAI python sdk (openai.OpenAI client) usage
        resp = self.azure_client.chat.completions.create(
            engine=self.azure_deployment,
            messages=[{"role": "system", "content": "You are a concise evaluator."},
                      {"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=60
        )
        text = resp.choices[0].message.content.strip()
        try:
            obj = json.loads(text)
            return float(obj.get("score", 0.0))
        except Exception:
            # attempt to parse a float in text
            m = re.search(r"([0-1](?:\.\d+)?)", text)
            if m:
                return float(m.group(1))
        return None

    # ----------------- core evaluation for one run -----------------
    def evaluate_once(self, ground_truths: Dict[str, GroundTruthRecord], ai_outputs: List[AIOutputRecord],
                      semantic_threshold: float = 0.72, pdf_threshold: float = 0.75) -> Dict[str, Dict[str, Any]]:

        chosen: Dict[str, AIOutputRecord] = {out.query_id: out for out in ai_outputs}
        results: Dict[str, Dict[str, Any]] = {}

        for qid, gt in ground_truths.items():
            out = chosen.get(qid)
            if out is None:
                results[qid] = {'found': False, 'match_type': gt.expected_type, 'accuracy': 0,
                                'expected': gt.expected, 'output': None, 'in_pdf': None, 'error_type': 'missing'}
                continue

            output = out.output
            in_pdf = self.pdf_vector_store.query(output, top_k=1) >= pdf_threshold if self.pdf_vector_store else False
            rec: Dict[str, Any] = {'found': True, 'match_type': gt.expected_type, 'accuracy': 0,
                                   'expected': gt.expected, 'output': output, 'in_pdf': in_pdf}

            # optional toxicity
            if self.enable_toxicity:
                rec['toxicity'] = self.evaluate_toxicity(output)

            # LLM judge (optional override / supplemental)
            if self.enable_llm_judge:
                llmscore = self.call_llm_judge(output, gt.expected)
                if llmscore is not None:
                    rec['llm_score'] = llmscore

            # type-specific comparisons
            if gt.expected_type == "number":
                nm = self.numeric_match(output, gt.expected, tol=gt.tolerance)
                rec['accuracy'] = 1 if nm else 0
                rec['error_type'] = 'correct' if nm else ('hallucination' if not in_pdf else 'wrong_answer')

            elif gt.expected_type in ("email",):
                em = normalize_email(output) == normalize_email(gt.expected)
                rec['accuracy'] = 1 if em else 0
                rec['error_type'] = 'correct' if em else ('partial_match' if in_pdf else 'hallucination')

            elif gt.expected_type in ("date",):
                dm = normalize_date_string(output) == normalize_date_string(gt.expected)
                rec['accuracy'] = 1 if dm else 0
                rec['error_type'] = 'correct' if dm else ('partial_match' if in_pdf else 'hallucination')

            elif gt.expected_type == "text":
                em = self.exact_match(output, gt.expected)
                sem = self.semantic_similarity(output, gt.expected)
                rec['semantic_score'] = sem
                # if LLM judge exists use it as tie-breaker/supplement
                if 'llm_score' in rec:
                    rec['accuracy'] = 1 if rec['llm_score'] >= 0.75 else (1 if (em or sem >= semantic_threshold) else 0)
                else:
                    rec['accuracy'] = 1 if (em or sem >= semantic_threshold) else 0
                rec['error_type'] = 'correct' if rec['accuracy'] else ('partial_match' if in_pdf else 'hallucination')

            elif gt.expected_type == "paragraph":
                # paragraph-level metrics
                b = self.bertscore_eval(output, gt.expected)
                r = self.rouge_eval(output, gt.expected)
                bleu = self.bleu_eval(output, gt.expected)
                met = self.meteor_eval(output, gt.expected)
                rec.update(b)
                rec.update(r)
                rec['bleu'] = bleu
                rec['meteor'] = met
                # rely primarily on bertscore_f1/rougeL
                if (rec.get('bertscore_f1', 0.0) >= 0.85) or (rec.get('rougeL', 0.0) >= 0.7) or (rec.get('bleu', 0.0) >= 40.0):
                    rec['accuracy'] = 1
                    rec['error_type'] = 'correct'
                else:
                    rec['accuracy'] = 1 if ('llm_score' in rec and rec['llm_score'] >= 0.75) else 0
                    rec['error_type'] = 'partial_match' if in_pdf else ('hallucination' if rec['accuracy']==0 else 'correct')

            results[qid] = rec

        return results

    # ----------------- Consistency -----------------
    def evaluate_consistency(self, ai_outputs: List[AIOutputRecord]) -> Dict[str, float]:
        groups = defaultdict(list)
        for out in ai_outputs:
            groups[out.query_id].append(out.output)
        scores: Dict[str, float] = {}
        for qid, outputs in groups.items():
            n = len(outputs)
            if n <= 1:
                scores[qid] = 1.0
                continue
            emb = self.model.encode(outputs, convert_to_tensor=True)
            sims = util.pytorch_cos_sim(emb, emb).cpu().numpy()
            i_upper, j_upper = np.triu_indices(n, k=1)
            upper_vals = sims[i_upper, j_upper]
            mean_pair = float(np.mean(upper_vals)) if upper_vals.size > 0 else 1.0
            scores[qid] = (mean_pair + 1.0) / 2.0
        return scores

    # ----------------- Aggregation (RQS) -----------------
    def aggregate_scores(self, per_query: Dict[str, Dict[str, Any]], consistency_scores: Dict[str, float], alpha=0.6, beta=0.2, gamma=0.2):
        qids = list(per_query.keys())
        accs = [per_query[q].get('accuracy', 0) for q in qids]
        accuracy = float(np.mean(accs)) if accs else 0.0
        cs = [consistency_scores.get(q, 1.0) for q in qids]
        consistency = float(np.mean(cs)) if cs else 1.0
        pdf_supports = [1 if per_query[q].get('in_pdf') else 0 for q in qids]
        pdf_support_rate = float(np.mean(pdf_supports)) if pdf_supports else 0.0
        rqs = alpha * accuracy + beta * consistency + gamma * pdf_support_rate
        return {'accuracy': accuracy, 'consistency': consistency, 'pdf_support_rate': pdf_support_rate, 'rqs': rqs, 'alpha': alpha, 'beta': beta, 'gamma': gamma, 'n_queries': len(qids)}

# ---------------------- Reporting ----------------------
def save_csv(per_query: Dict[str, Dict[str, Any]], path: str = "results_table.csv"):
    rows = []
    for qid, v in per_query.items():
        row = {"query_id": qid}
        row.update({k: (json.dumps(v[k]) if isinstance(v[k], (dict, list)) else v[k]) for k in v})
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)
    print("Wrote", path)

def generate_error_chart(error_summary: Dict[str, int]) -> str:
    labels = list(error_summary.keys()); sizes = list(error_summary.values())
    if not labels:
        return ""
    fig, ax = plt.subplots(figsize=(4,4))
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140); ax.axis("equal")
    buf = BytesIO(); plt.savefig(buf, format="png"); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def save_html(per_query: Dict[str, Dict[str, Any]], agg: Dict[str, Any], error_summary: Dict[str, int], path: str = "results.html"):
    rows_html = []
    for qid, data in per_query.items():
        extras = []
        for m in ("semantic_score","bertscore_f1","rougeL","bleu","meteor","toxicity","llm_score"):
            if m in data:
                extras.append(f"{m}: {round(float(data[m]),3)}")
        extra_html = "<br/>".join(extras)
        row = f"<tr><td>{html.escape(qid)}</td><td>{html.escape(str(data.get('expected','')))}</td><td>{html.escape(str(data.get('output','')))}</td><td>{html.escape(str(data.get('match_type','')))}</td><td>{data.get('accuracy')}</td><td>{html.escape(str(data.get('error_type','')))}</td><td>{'✅' if data.get('in_pdf') else '❌'}</td><td>{extra_html}</td></tr>"
        rows_html.append(row)
    chart = generate_error_chart(error_summary)
    chart_html = f'<img src="data:image/png;base64,{chart}" width="300"/>' if chart else ""
    agg_html = "".join([f"<li>{k}: {v}</li>" for k,v in agg.items()])
    html_content = f"""<html><head><title>AI Eval</title></head><body><h1>Aggregate</h1><ul>{agg_html}</ul><h2>Error summary</h2>{chart_html}<h2>Per-query</h2><table border=1><tr><th>Query</th><th>Expected</th><th>Output</th><th>Type</th><th>Accuracy</th><th>Error</th><th>In PDF</th><th>Extras</th></tr>{''.join(rows_html)}</table></body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print("Wrote", path)

# ---------------------- Loaders & Helpers ----------------------
def load_ai_outputs_json(path: str) -> List[AIOutputRecord]:
    """
    Expects a JSON array where each item is a LUMA message or object:
      { "query_id": "q1", "run_id": "r1", "message": { ... } }
    For simplicity we expect AI output text under "output" field or "content".
    """
    raw = json.load(open(path, "r", encoding="utf-8"))
    recs: List[AIOutputRecord] = []
    for i, item in enumerate(raw):
        qid = item.get("query_id") or item.get("qid") or item.get("id") or f"q{i+1}"
        run_id = item.get("run_id") or item.get("run") or "1"
        # try common fields
        out = item.get("output") or item.get("answer") or None
        if out is None:
            # look into message structure (LUMA)
            msg = item.get("message") or item.get("messages") or {}
            if isinstance(msg, dict):
                out = msg.get("content") or msg.get("text") or None
            elif isinstance(msg, list) and len(msg) > 0:
                first = msg[-1]
                out = first.get("content") if isinstance(first, dict) else str(first)
        out = out if out is not None else ""
        recs.append(AIOutputRecord(query_id=str(qid), run_id=str(run_id), output=str(out)))
    return recs

def load_ground_truth_csv_or_json(path: str) -> Dict[str, GroundTruthRecord]:
    if path.lower().endswith(".csv"):
        # try robust CSV read
        try:
            df = pd.read_csv(path, dtype=str, on_bad_lines="skip").fillna('')
        except TypeError:
            # older pandas
            df = pd.read_csv(path, dtype=str, error_bad_lines=False).fillna('')
        out = {}
        for _, r in df.iterrows():
            qid = str(r.get("query_id") or r.get("id") or r.get("qid"))
            expected = str(r.get("expected") or "")
            typ = str(r.get("expected_type") or "text").lower()
            tol = r.get("tolerance", "")
            try:
                tolerance = float(tol) if tol not in (None, "") else 1e-6
            except Exception:
                tolerance = 1e-6
            meta = {c: r[c] for c in df.columns if c not in ("query_id", "expected", "expected_type", "tolerance")}
            out[qid] = GroundTruthRecord(qid, expected, typ, tolerance, meta)
        return out
    else:
        raw = json.load(open(path, "r", encoding="utf-8"))
        out = {}
        for item in raw:
            qid = str(item.get("query_id") or item.get("id"))
            expected = str(item.get("expected") or "")
            typ = str(item.get("expected_type") or "text").lower()
            tol = item.get("tolerance", 1e-6)
            out[qid] = GroundTruthRecord(qid, expected, typ, tol, item.get("metadata", {}))
        return out

# ---------------------- Sample Data Generator ----------------------
def generate_sample(output_dir: str = "."):
    os.makedirs(output_dir, exist_ok=True)
    # write a sample PDF
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        pdf_path = os.path.join(output_dir, "contract.pdf")
        c = canvas.Canvas(pdf_path, pagesize=letter)
        text = c.beginText(40, 700)
        lines = [
            "The contract states that the employee will be issued a Swiss Residence Permit L valid until August 29, 2025.",
            "The company will provide mobile subscription plan 'Lebara Next M' at CHF 45.95 monthly.",
            "Medical insurance is provided by Cigna International and the employee is entitled to 30 days paid leave per year.",
            "Section 5.2: Employee indemnification — the employee is responsible for returning company property within 7 days after termination.",
            "Section 7.1: Confidentiality obligations survive for 3 years.",
            "This agreement shall be governed by Swiss law and any disputes shall be submitted to the courts of Zurich.",
            "Contact: aniketkmarwadi@gmail.com"
        ]
        for L in lines:
            text.textLine(L)
        c.drawText(text); c.save()
        print("Sample PDF written:", pdf_path)
    except Exception as e:
        print("reportlab not installed, skipping pdf creation:", e)
        pdf_path = None

    # ground truth CSV
    gt = [
        {"query_id": "q1", "expected": "aniketkmarwadi@gmail.com", "expected_type": "email"},
        {"query_id": "q2", "expected": "CHF 45.95", "expected_type": "number"},
        {"query_id": "q3", "expected": "Swiss Residence Permit L", "expected_type": "text"},
        {"query_id": "q4", "expected": "2025-08-29", "expected_type": "date"},
        {"query_id": "q5", "expected": "Lebara Next M", "expected_type": "text"},
        {"query_id": "q6", "expected": "The contract states that the employee will be issued a Swiss Residence Permit L valid until August 29, 2025. The company will provide mobile subscription plan 'Lebara Next M' at CHF 45.95 monthly. Medical insurance is provided by Cigna International and the employee is entitled to 30 days paid leave per year.", "expected_type": "paragraph"}
    ]
    gt_path = os.path.join(output_dir, "ground_truth.csv")
    pd.DataFrame(gt).to_csv(gt_path, index=False)
    print("Sample ground truth written:", gt_path)

    # AI outputs JSON (LUMA-like)
    ai_outputs = [
        {"query_id": "q1", "run_id": "r1", "output": "aniketkmarwadi@gmail.com"},
        {"query_id": "q2", "run_id": "r1", "output": "CHF 45.95"},
        {"query_id": "q3", "run_id": "r1", "output": "Swiss Residence Permit L"},
        {"query_id": "q4", "run_id": "r1", "output": "August 29, 2025"},
        {"query_id": "q5", "run_id": "r1", "output": "Lebara Next M"},
        {"query_id": "q6", "run_id": "r1", "output": gt[5]["expected"]}
    ]
    ai_path = os.path.join(output_dir, "ai_outputs.json")
    json.dump(ai_outputs, open(ai_path, "w", encoding="utf-8"), indent=2)
    print("Sample AI outputs written:", ai_path)
    return {"pdf": pdf_path, "gt": gt_path, "ai": ai_path}

# ---------------------- CLI / Main ----------------------
def parse_args():
    p = argparse.ArgumentParser(description="AI Assurance Evaluator (JSON + PDF)")
    p.add_argument("--ai_outputs", required=False, help="Path to AI outputs JSON (LUMA messages)")
    p.add_argument("--ground_truth", required=False, help="Path to ground truth CSV or JSON")
    p.add_argument("--pdfs", nargs="*", help="Glob or list of PDF files")
    p.add_argument("--model", default="all-MiniLM-L6-v2", help="Sentence-transformers model name or local path")
    p.add_argument("--persist_chroma", default="./chroma_db", help="Chroma persist directory")
    p.add_argument("--semantic_threshold", type=float, default=0.72)
    p.add_argument("--pdf_threshold", type=float, default=0.75)
    p.add_argument("--alpha", type=float, default=0.6)
    p.add_argument("--beta", type=float, default=0.2)
    p.add_argument("--gamma", type=float, default=0.2)
    p.add_argument("--output", default="results.json")
    p.add_argument("--enable_toxicity", action="store_true", help="Enable toxicity scoring (requires transformers)")
    p.add_argument("--enable_llm_judge", action="store_true", help="Enable Azure OpenAI LLM judge (requires env vars)")
    p.add_argument("--generate-sample", action="store_true", help="Generate sample PDF, ground truth and AI outputs then exit")
    return p.parse_args()

def main():
    args = parse_args()

    if args.generate_sample:
        generate_sample()
        print("Sample generation complete; run again with --ai_outputs and --ground_truth pointing to generated files.")
        return

    if not args.ai_outputs or not args.ground_truth or not args.pdfs:
        print("Must provide --ai_outputs, --ground_truth and --pdfs unless using --generate-sample")
        sys.exit(1)

    # load data
    ai_outputs = load_ai_outputs_json(args.ai_outputs)
    ground_truths = load_ground_truth_csv_or_json(args.ground_truth)
    # collect pdf files from globs
    pdf_paths = []
    for p in args.pdfs:
        pdf_paths.extend(glob(p))
    pdf_texts = PDFExtractor.extract_texts(pdf_paths) if pdf_paths else {}

    # build chroma if available
    pdf_vector_store = None
    if CHROMA_MODE:
        try:
            print("Initializing Chroma vector store...")
            pdf_vector_store = PDFVectorStore(embedding_model=args.model, persist_dir=args.persist_chroma)
            print("Adding PDF texts to vector store...")
            pdf_vector_store.add_pdf_texts(pdf_texts)
        except Exception as e:
            print("Warning: could not initialize Chroma vector store:", e)
            pdf_vector_store = None
    else:
        print("chromadb not installed or incompatible version; PDF semantic search disabled.")

    # create evaluator
    evaluator = Evaluator(model_name=args.model, pdf_vector_store=pdf_vector_store, enable_toxicity=args.enable_toxicity, enable_llm_judge=args.enable_llm_judge)

    # run evaluation
    per_query = evaluator.evaluate_once(ground_truths, ai_outputs, semantic_threshold=args.semantic_threshold, pdf_threshold=args.pdf_threshold)
    consistency = evaluator.evaluate_consistency(ai_outputs)
    agg = evaluator.aggregate_scores(per_query, consistency, alpha=args.alpha, beta=args.beta, gamma=args.gamma)

    error_counts = Counter(v.get("error_type", "unknown") for v in per_query.values())
    n_queries = agg.get("n_queries", len(per_query))
    hallucination_rate = error_counts.get("hallucination", 0) / n_queries if n_queries else 0.0

    evaluation_status = "PASS"
    fail_reasons = []
    if agg["accuracy"] < 0.5:
        evaluation_status = "FAIL"; fail_reasons.append(f"Accuracy {agg['accuracy']:.2f} < 0.5")
    if agg["consistency"] < 0.5:
        evaluation_status = "FAIL"; fail_reasons.append(f"Consistency {agg['consistency']:.2f} < 0.5")
    if hallucination_rate > 0.5:
        evaluation_status = "FAIL"; fail_reasons.append(f"Hallucination rate {hallucination_rate:.2f} > 0.5")
    if agg["rqs"] < 0.5:
        evaluation_status = "FAIL"; fail_reasons.append(f"RQS {agg['rqs']:.2f} < 0.5")

    results = {
        "per_query": per_query,
        "consistency_per_query": consistency,
        "aggregate": agg,
        "error_summary": dict(error_counts),
        "evaluation_status": evaluation_status,
        "fail_reasons": fail_reasons
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=to_serializable)
    print("Wrote", args.output)

    save_csv(per_query, path="results_table.csv")
    save_html(per_query, agg, dict(error_counts), path="results.html")

    print("Evaluation Status:", evaluation_status)
    if fail_reasons:
        print("Fail reasons:")
        for r in fail_reasons:
            print(" -", r)

if __name__ == "__main__":
    main()

```

#Command
python ai_assurance_app.py --ai_outputs ai_outputs.json --ground_truth ground_truth.csv --pdfs "./pdfs/contract.pdf" --output results.json --enable_toxicity


#Data
ai_outputs.json:

[
  {
    "query_id": "q00001",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "aniketkmarwadi@gmail.com"
      }
    ],
    "output": "aniketkmarwadi@gmail.com"
  },
  {
    "query_id": "q00001",
    "run_id": "r2",
    "message": [
      {
        "role": "assistant",
        "content": "aniketkmarwadi at gmail.com"
      }
    ],
    "output": "aniketkmarwadi at gmail.com"
  },
  {
    "query_id": "q00001",
    "run_id": "r3",
    "message": [
      {
        "role": "assistant",
        "content": "ANIKETKMARWADI@GMAIL.COM"
      }
    ],
    "output": "ANIKETKMARWADI@GMAIL.COM"
  },
  {
    "query_id": "q00001",
    "run_id": "r4",
    "message": [
      {
        "role": "assistant",
        "content": "aniketkmarwadi(at)gmail.com"
      }
    ],
    "output": "aniketkmarwadi(at)gmail.com"
  },
  {
    "query_id": "q00001",
    "run_id": "r99",
    "message": [
      {
        "role": "assistant",
        "content": "nobody@madeup.com"
      }
    ],
    "output": "nobody@madeup.com"
  },
  {
    "query_id": "q00002",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "aniketkmarwadi@gmail.com"
      }
    ],
    "output": "aniketkmarwadi@gmail.com"
  },
  {
    "query_id": "q00002",
    "run_id": "r2",
    "message": [
      {
        "role": "assistant",
        "content": "aniketkmarwadi at gmail.com"
      }
    ],
    "output": "aniketkmarwadi at gmail.com"
  },
  {
    "query_id": "q00002",
    "run_id": "r3",
    "message": [
      {
        "role": "assistant",
        "content": "ANIKETKMARWADI@GMAIL.COM"
      }
    ],
    "output": "ANIKETKMARWADI@GMAIL.COM"
  },
  {
    "query_id": "q00002",
    "run_id": "r4",
    "message": [
      {
        "role": "assistant",
        "content": "aniketkmarwadi(at)gmail.com"
      }
    ],
    "output": "aniketkmarwadi(at)gmail.com"
  },
  {
    "query_id": "q00002",
    "run_id": "r99",
    "message": [
      {
        "role": "assistant",
        "content": "nobody@madeup.com"
      }
    ],
    "output": "nobody@madeup.com"
  },
  {
    "query_id": "q00003",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "aniketkmarwadi@gmail.com"
      }
    ],
    "output": "aniketkmarwadi@gmail.com"
  },
  {
    "query_id": "q00003",
    "run_id": "r2",
    "message": [
      {
        "role": "assistant",
        "content": "aniketkmarwadi at gmail.com"
      }
    ],
    "output": "aniketkmarwadi at gmail.com"
  },
  {
    "query_id": "q00003",
    "run_id": "r3",
    "message": [
      {
        "role": "assistant",
        "content": "ANIKETKMARWADI@GMAIL.COM"
      }
    ],
    "output": "ANIKETKMARWADI@GMAIL.COM"
  },
  {
    "query_id": "q00003",
    "run_id": "r4",
    "message": [
      {
        "role": "assistant",
        "content": "aniketkmarwadi(at)gmail.com"
      }
    ],
    "output": "aniketkmarwadi(at)gmail.com"
  },
  {
    "query_id": "q00003",
    "run_id": "r99",
    "message": [
      {
        "role": "assistant",
        "content": "nobody@madeup.com"
      }
    ],
    "output": "nobody@madeup.com"
  },
  {
    "query_id": "q00004",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "aniketkmarwadi@gmail.com"
      }
    ],
    "output": "aniketkmarwadi@gmail.com"
  },
  {
    "query_id": "q00004",
    "run_id": "r2",
    "message": [
      {
        "role": "assistant",
        "content": "aniketkmarwadi at gmail.com"
      }
    ],
    "output": "aniketkmarwadi at gmail.com"
  },
  {
    "query_id": "q00004",
    "run_id": "r3",
    "message": [
      {
        "role": "assistant",
        "content": "ANIKETKMARWADI@GMAIL.COM"
      }
    ],
    "output": "ANIKETKMARWADI@GMAIL.COM"
  },
  {
    "query_id": "q00004",
    "run_id": "r4",
    "message": [
      {
        "role": "assistant",
        "content": "aniketkmarwadi(at)gmail.com"
      }
    ],
    "output": "aniketkmarwadi(at)gmail.com"
  },
  {
    "query_id": "q00004",
    "run_id": "r99",
    "message": [
      {
        "role": "assistant",
        "content": "nobody@madeup.com"
      }
    ],
    "output": "nobody@madeup.com"
  },
  {
    "query_id": "q00005",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "aniketkmarwadi@gmail.com"
      }
    ],
    "output": "aniketkmarwadi@gmail.com"
  },
  {
    "query_id": "q00005",
    "run_id": "r2",
    "message": [
      {
        "role": "assistant",
        "content": "aniketkmarwadi at gmail.com"
      }
    ],
    "output": "aniketkmarwadi at gmail.com"
  },
  {
    "query_id": "q00005",
    "run_id": "r3",
    "message": [
      {
        "role": "assistant",
        "content": "ANIKETKMARWADI@GMAIL.COM"
      }
    ],
    "output": "ANIKETKMARWADI@GMAIL.COM"
  },
  {
    "query_id": "q00005",
    "run_id": "r4",
    "message": [
      {
        "role": "assistant",
        "content": "aniketkmarwadi(at)gmail.com"
      }
    ],
    "output": "aniketkmarwadi(at)gmail.com"
  },
  {
    "query_id": "q00005",
    "run_id": "r99",
    "message": [
      {
        "role": "assistant",
        "content": "nobody@madeup.com"
      }
    ],
    "output": "nobody@madeup.com"
  },
  {
    "query_id": "q00006",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "aniketkmarwadi@gmail.com"
      }
    ],
    "output": "aniketkmarwadi@gmail.com"
  },
  {
    "query_id": "q00006",
    "run_id": "r2",
    "message": [
      {
        "role": "assistant",
        "content": "aniketkmarwadi at gmail.com"
      }
    ],
    "output": "aniketkmarwadi at gmail.com"
  },
  {
    "query_id": "q00006",
    "run_id": "r3",
    "message": [
      {
        "role": "assistant",
        "content": "ANIKETKMARWADI@GMAIL.COM"
      }
    ],
    "output": "ANIKETKMARWADI@GMAIL.COM"
  },
  {
    "query_id": "q00006",
    "run_id": "r4",
    "message": [
      {
        "role": "assistant",
        "content": "aniketkmarwadi(at)gmail.com"
      }
    ],
    "output": "aniketkmarwadi(at)gmail.com"
  },
  {
    "query_id": "q00006",
    "run_id": "r99",
    "message": [
      {
        "role": "assistant",
        "content": "nobody@madeup.com"
      }
    ],
    "output": "nobody@madeup.com"
  },
  {
    "query_id": "q00007",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "aniketkmarwadi@gmail.com"
      }
    ],
    "output": "aniketkmarwadi@gmail.com"
  },
  {
    "query_id": "q00007",
    "run_id": "r2",
    "message": [
      {
        "role": "assistant",
        "content": "aniketkmarwadi at gmail.com"
      }
    ],
    "output": "aniketkmarwadi at gmail.com"
  },
  {
    "query_id": "q00007",
    "run_id": "r3",
    "message": [
      {
        "role": "assistant",
        "content": "ANIKETKMARWADI@GMAIL.COM"
      }
    ],
    "output": "ANIKETKMARWADI@GMAIL.COM"
  },
  {
    "query_id": "q00007",
    "run_id": "r4",
    "message": [
      {
        "role": "assistant",
        "content": "aniketkmarwadi(at)gmail.com"
      }
    ],
    "output": "aniketkmarwadi(at)gmail.com"
  },
  {
    "query_id": "q00007",
    "run_id": "r99",
    "message": [
      {
        "role": "assistant",
        "content": "nobody@madeup.com"
      }
    ],
    "output": "nobody@madeup.com"
  },
  {
    "query_id": "q00008",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "aniketkmarwadi@gmail.com"
      }
    ],
    "output": "aniketkmarwadi@gmail.com"
  },
  {
    "query_id": "q00008",
    "run_id": "r2",
    "message": [
      {
        "role": "assistant",
        "content": "aniketkmarwadi at gmail.com"
      }
    ],
    "output": "aniketkmarwadi at gmail.com"
  },
  {
    "query_id": "q00008",
    "run_id": "r3",
    "message": [
      {
        "role": "assistant",
        "content": "ANIKETKMARWADI@GMAIL.COM"
      }
    ],
    "output": "ANIKETKMARWADI@GMAIL.COM"
  },
  {
    "query_id": "q00008",
    "run_id": "r4",
    "message": [
      {
        "role": "assistant",
        "content": "aniketkmarwadi(at)gmail.com"
      }
    ],
    "output": "aniketkmarwadi(at)gmail.com"
  },
  {
    "query_id": "q00008",
    "run_id": "r99",
    "message": [
      {
        "role": "assistant",
        "content": "nobody@madeup.com"
      }
    ],
    "output": "nobody@madeup.com"
  },
  {
    "query_id": "q00009",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "CHF 45.95"
      }
    ],
    "output": "CHF 45.95"
  },
  {
    "query_id": "q00009",
    "run_id": "r2",
    "message": [
      {
        "role": "assistant",
        "content": "45.95 CHF"
      }
    ],
    "output": "45.95 CHF"
  },
  {
    "query_id": "q00009",
    "run_id": "r3",
    "message": [
      {
        "role": "assistant",
        "content": "CHF45.95"
      }
    ],
    "output": "CHF45.95"
  },
  {
    "query_id": "q00009",
    "run_id": "r4",
    "message": [
      {
        "role": "assistant",
        "content": "45.95"
      }
    ],
    "output": "45.95"
  },
  {
    "query_id": "q00009",
    "run_id": "r5",
    "message": [
      {
        "role": "assistant",
        "content": "45.95"
      }
    ],
    "output": "45.95"
  },
  {
    "query_id": "q00009",
    "run_id": "rOCR",
    "message": [
      {
        "role": "assistant",
        "content": "CHF 45‚95"
      }
    ],
    "output": "CHF 45‚95"
  },
  {
    "query_id": "q00010",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "CHF 45.95"
      }
    ],
    "output": "CHF 45.95"
  },
  {
    "query_id": "q00010",
    "run_id": "r2",
    "message": [
      {
        "role": "assistant",
        "content": "45.95 CHF"
      }
    ],
    "output": "45.95 CHF"
  },
  {
    "query_id": "q00010",
    "run_id": "r3",
    "message": [
      {
        "role": "assistant",
        "content": "CHF45.95"
      }
    ],
    "output": "CHF45.95"
  },
  {
    "query_id": "q00010",
    "run_id": "r4",
    "message": [
      {
        "role": "assistant",
        "content": "45.95"
      }
    ],
    "output": "45.95"
  },
  {
    "query_id": "q00010",
    "run_id": "r5",
    "message": [
      {
        "role": "assistant",
        "content": "45.95"
      }
    ],
    "output": "45.95"
  },
  {
    "query_id": "q00010",
    "run_id": "rOCR",
    "message": [
      {
        "role": "assistant",
        "content": "CHF 45‚95"
      }
    ],
    "output": "CHF 45‚95"
  },
  {
    "query_id": "q00011",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "CHF 45.95"
      }
    ],
    "output": "CHF 45.95"
  },
  {
    "query_id": "q00011",
    "run_id": "r2",
    "message": [
      {
        "role": "assistant",
        "content": "45.95 CHF"
      }
    ],
    "output": "45.95 CHF"
  },
  {
    "query_id": "q00011",
    "run_id": "r3",
    "message": [
      {
        "role": "assistant",
        "content": "CHF45.95"
      }
    ],
    "output": "CHF45.95"
  },
  {
    "query_id": "q00011",
    "run_id": "r4",
    "message": [
      {
        "role": "assistant",
        "content": "45.95"
      }
    ],
    "output": "45.95"
  },
  {
    "query_id": "q00011",
    "run_id": "r5",
    "message": [
      {
        "role": "assistant",
        "content": "45.95"
      }
    ],
    "output": "45.95"
  },
  {
    "query_id": "q00011",
    "run_id": "rOCR",
    "message": [
      {
        "role": "assistant",
        "content": "CHF 45‚95"
      }
    ],
    "output": "CHF 45‚95"
  },
  {
    "query_id": "q00012",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "CHF 45.95"
      }
    ],
    "output": "CHF 45.95"
  },
  {
    "query_id": "q00012",
    "run_id": "r2",
    "message": [
      {
        "role": "assistant",
        "content": "45.95 CHF"
      }
    ],
    "output": "45.95 CHF"
  },
  {
    "query_id": "q00012",
    "run_id": "r3",
    "message": [
      {
        "role": "assistant",
        "content": "CHF45.95"
      }
    ],
    "output": "CHF45.95"
  },
  {
    "query_id": "q00012",
    "run_id": "r4",
    "message": [
      {
        "role": "assistant",
        "content": "45.95"
      }
    ],
    "output": "45.95"
  },
  {
    "query_id": "q00012",
    "run_id": "r5",
    "message": [
      {
        "role": "assistant",
        "content": "45.95"
      }
    ],
    "output": "45.95"
  },
  {
    "query_id": "q00012",
    "run_id": "rOCR",
    "message": [
      {
        "role": "assistant",
        "content": "CHF 45‚95"
      }
    ],
    "output": "CHF 45‚95"
  },
  {
    "query_id": "q00013",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "CHF 45.95"
      }
    ],
    "output": "CHF 45.95"
  },
  {
    "query_id": "q00013",
    "run_id": "r2",
    "message": [
      {
        "role": "assistant",
        "content": "45.95 CHF"
      }
    ],
    "output": "45.95 CHF"
  },
  {
    "query_id": "q00013",
    "run_id": "r3",
    "message": [
      {
        "role": "assistant",
        "content": "CHF45.95"
      }
    ],
    "output": "CHF45.95"
  },
  {
    "query_id": "q00013",
    "run_id": "r4",
    "message": [
      {
        "role": "assistant",
        "content": "45.95"
      }
    ],
    "output": "45.95"
  },
  {
    "query_id": "q00013",
    "run_id": "r5",
    "message": [
      {
        "role": "assistant",
        "content": "45.95"
      }
    ],
    "output": "45.95"
  },
  {
    "query_id": "q00013",
    "run_id": "rOCR",
    "message": [
      {
        "role": "assistant",
        "content": "CHF 45‚95"
      }
    ],
    "output": "CHF 45‚95"
  },
  {
    "query_id": "q00014",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "CHF 45.95"
      }
    ],
    "output": "CHF 45.95"
  },
  {
    "query_id": "q00014",
    "run_id": "r2",
    "message": [
      {
        "role": "assistant",
        "content": "45.95 CHF"
      }
    ],
    "output": "45.95 CHF"
  },
  {
    "query_id": "q00014",
    "run_id": "r3",
    "message": [
      {
        "role": "assistant",
        "content": "CHF45.95"
      }
    ],
    "output": "CHF45.95"
  },
  {
    "query_id": "q00014",
    "run_id": "r4",
    "message": [
      {
        "role": "assistant",
        "content": "45.95"
      }
    ],
    "output": "45.95"
  },
  {
    "query_id": "q00014",
    "run_id": "r5",
    "message": [
      {
        "role": "assistant",
        "content": "45.95"
      }
    ],
    "output": "45.95"
  },
  {
    "query_id": "q00014",
    "run_id": "rOCR",
    "message": [
      {
        "role": "assistant",
        "content": "CHF 45‚95"
      }
    ],
    "output": "CHF 45‚95"
  },
  {
    "query_id": "q00015",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "CHF 145.95"
      }
    ],
    "output": "CHF 145.95"
  },
  {
    "query_id": "q00015",
    "run_id": "r2",
    "message": [
      {
        "role": "assistant",
        "content": "145.95 CHF"
      }
    ],
    "output": "145.95 CHF"
  },
  {
    "query_id": "q00015",
    "run_id": "r3",
    "message": [
      {
        "role": "assistant",
        "content": "CHF145.95"
      }
    ],
    "output": "CHF145.95"
  },
  {
    "query_id": "q00015",
    "run_id": "r4",
    "message": [
      {
        "role": "assistant",
        "content": "145.95"
      }
    ],
    "output": "145.95"
  },
  {
    "query_id": "q00015",
    "run_id": "r5",
    "message": [
      {
        "role": "assistant",
        "content": "145.95"
      }
    ],
    "output": "145.95"
  },
  {
    "query_id": "q00015",
    "run_id": "rOCR",
    "message": [
      {
        "role": "assistant",
        "content": "CHF 145‚95"
      }
    ],
    "output": "CHF 145‚95"
  },
  {
    "query_id": "q00016",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "CHF 145.95"
      }
    ],
    "output": "CHF 145.95"
  },
  {
    "query_id": "q00016",
    "run_id": "r2",
    "message": [
      {
        "role": "assistant",
        "content": "145.95 CHF"
      }
    ],
    "output": "145.95 CHF"
  },
  {
    "query_id": "q00016",
    "run_id": "r3",
    "message": [
      {
        "role": "assistant",
        "content": "CHF145.95"
      }
    ],
    "output": "CHF145.95"
  },
  {
    "query_id": "q00016",
    "run_id": "r4",
    "message": [
      {
        "role": "assistant",
        "content": "145.95"
      }
    ],
    "output": "145.95"
  },
  {
    "query_id": "q00016",
    "run_id": "r5",
    "message": [
      {
        "role": "assistant",
        "content": "145.95"
      }
    ],
    "output": "145.95"
  },
  {
    "query_id": "q00016",
    "run_id": "rOCR",
    "message": [
      {
        "role": "assistant",
        "content": "CHF 145‚95"
      }
    ],
    "output": "CHF 145‚95"
  },
  {
    "query_id": "q00017",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "CHF 145.95"
      }
    ],
    "output": "CHF 145.95"
  },
  {
    "query_id": "q00017",
    "run_id": "r2",
    "message": [
      {
        "role": "assistant",
        "content": "145.95 CHF"
      }
    ],
    "output": "145.95 CHF"
  },
  {
    "query_id": "q00017",
    "run_id": "r3",
    "message": [
      {
        "role": "assistant",
        "content": "CHF145.95"
      }
    ],
    "output": "CHF145.95"
  },
  {
    "query_id": "q00017",
    "run_id": "r4",
    "message": [
      {
        "role": "assistant",
        "content": "145.95"
      }
    ],
    "output": "145.95"
  },
  {
    "query_id": "q00017",
    "run_id": "r5",
    "message": [
      {
        "role": "assistant",
        "content": "145.95"
      }
    ],
    "output": "145.95"
  },
  {
    "query_id": "q00017",
    "run_id": "rOCR",
    "message": [
      {
        "role": "assistant",
        "content": "CHF 145‚95"
      }
    ],
    "output": "CHF 145‚95"
  },
  {
    "query_id": "q00018",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "CHF 145.95"
      }
    ],
    "output": "CHF 145.95"
  },
  {
    "query_id": "q00018",
    "run_id": "r2",
    "message": [
      {
        "role": "assistant",
        "content": "145.95 CHF"
      }
    ],
    "output": "145.95 CHF"
  },
  {
    "query_id": "q00018",
    "run_id": "r3",
    "message": [
      {
        "role": "assistant",
        "content": "CHF145.95"
      }
    ],
    "output": "CHF145.95"
  },
  {
    "query_id": "q00018",
    "run_id": "r4",
    "message": [
      {
        "role": "assistant",
        "content": "145.95"
      }
    ],
    "output": "145.95"
  },
  {
    "query_id": "q00018",
    "run_id": "r5",
    "message": [
      {
        "role": "assistant",
        "content": "145.95"
      }
    ],
    "output": "145.95"
  },
  {
    "query_id": "q00018",
    "run_id": "rOCR",
    "message": [
      {
        "role": "assistant",
        "content": "CHF 145‚95"
      }
    ],
    "output": "CHF 145‚95"
  },
  {
    "query_id": "q00019",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "CHF 145.95"
      }
    ],
    "output": "CHF 145.95"
  },
  {
    "query_id": "q00019",
    "run_id": "r2",
    "message": [
      {
        "role": "assistant",
        "content": "145.95 CHF"
      }
    ],
    "output": "145.95 CHF"
  },
  {
    "query_id": "q00019",
    "run_id": "r3",
    "message": [
      {
        "role": "assistant",
        "content": "CHF145.95"
      }
    ],
    "output": "CHF145.95"
  },
  {
    "query_id": "q00019",
    "run_id": "r4",
    "message": [
      {
        "role": "assistant",
        "content": "145.95"
      }
    ],
    "output": "145.95"
  },
  {
    "query_id": "q00019",
    "run_id": "r5",
    "message": [
      {
        "role": "assistant",
        "content": "145.95"
      }
    ],
    "output": "145.95"
  },
  {
    "query_id": "q00019",
    "run_id": "rOCR",
    "message": [
      {
        "role": "assistant",
        "content": "CHF 145‚95"
      }
    ],
    "output": "CHF 145‚95"
  },
  {
    "query_id": "q00020",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "CHF 145.95"
      }
    ],
    "output": "CHF 145.95"
  },
  {
    "query_id": "q00020",
    "run_id": "r2",
    "message": [
      {
        "role": "assistant",
        "content": "145.95 CHF"
      }
    ],
    "output": "145.95 CHF"
  },
  {
    "query_id": "q00020",
    "run_id": "r3",
    "message": [
      {
        "role": "assistant",
        "content": "CHF145.95"
      }
    ],
    "output": "CHF145.95"
  },
  {
    "query_id": "q00020",
    "run_id": "r4",
    "message": [
      {
        "role": "assistant",
        "content": "145.95"
      }
    ],
    "output": "145.95"
  },
  {
    "query_id": "q00020",
    "run_id": "r5",
    "message": [
      {
        "role": "assistant",
        "content": "145.95"
      }
    ],
    "output": "145.95"
  },
  {
    "query_id": "q00020",
    "run_id": "rOCR",
    "message": [
      {
        "role": "assistant",
        "content": "CHF 145‚95"
      }
    ],
    "output": "CHF 145‚95"
  },
  {
    "query_id": "q00021",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "CHF 1000.00"
      }
    ],
    "output": "CHF 1000.00"
  },
  {
    "query_id": "q00021",
    "run_id": "r2",
    "message": [
      {
        "role": "assistant",
        "content": "1000.00 CHF"
      }
    ],
    "output": "1000.00 CHF"
  },
  {
    "query_id": "q00021",
    "run_id": "r3",
    "message": [
      {
        "role": "assistant",
        "content": "CHF1000.00"
      }
    ],
    "output": "CHF1000.00"
  },
  {
    "query_id": "q00021",
    "run_id": "r4",
    "message": [
      {
        "role": "assistant",
        "content": "1,000.0"
      }
    ],
    "output": "1,000.0"
  },
  {
    "query_id": "q00021",
    "run_id": "r5",
    "message": [
      {
        "role": "assistant",
        "content": "1000.0"
      }
    ],
    "output": "1000.0"
  },
  {
    "query_id": "q00021",
    "run_id": "rOCR",
    "message": [
      {
        "role": "assistant",
        "content": "CHF 1000‚0"
      }
    ],
    "output": "CHF 1000‚0"
  },
  {
    "query_id": "q00022",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "CHF 1000.00"
      }
    ],
    "output": "CHF 1000.00"
  },
  {
    "query_id": "q00022",
    "run_id": "r2",
    "message": [
      {
        "role": "assistant",
        "content": "1000.00 CHF"
      }
    ],
    "output": "1000.00 CHF"
  },
  {
    "query_id": "q00022",
    "run_id": "r3",
    "message": [
      {
        "role": "assistant",
        "content": "CHF1000.00"
      }
    ],
    "output": "CHF1000.00"
  },
  {
    "query_id": "q00022",
    "run_id": "r4",
    "message": [
      {
        "role": "assistant",
        "content": "1,000.0"
      }
    ],
    "output": "1,000.0"
  },
  {
    "query_id": "q00022",
    "run_id": "r5",
    "message": [
      {
        "role": "assistant",
        "content": "1000.0"
      }
    ],
    "output": "1000.0"
  },
  {
    "query_id": "q00022",
    "run_id": "rOCR",
    "message": [
      {
        "role": "assistant",
        "content": "CHF 1000‚0"
      }
    ],
    "output": "CHF 1000‚0"
  },
  {
    "query_id": "q00023",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "CHF 1000.00"
      }
    ],
    "output": "CHF 1000.00"
  },
  {
    "query_id": "q00023",
    "run_id": "r2",
    "message": [
      {
        "role": "assistant",
        "content": "1000.00 CHF"
      }
    ],
    "output": "1000.00 CHF"
  },
  {
    "query_id": "q00023",
    "run_id": "r3",
    "message": [
      {
        "role": "assistant",
        "content": "CHF1000.00"
      }
    ],
    "output": "CHF1000.00"
  },
  {
    "query_id": "q00023",
    "run_id": "r4",
    "message": [
      {
        "role": "assistant",
        "content": "1,000.0"
      }
    ],
    "output": "1,000.0"
  },
  {
    "query_id": "q00023",
    "run_id": "r5",
    "message": [
      {
        "role": "assistant",
        "content": "1000.0"
      }
    ],
    "output": "1000.0"
  },
  {
    "query_id": "q00023",
    "run_id": "rOCR",
    "message": [
      {
        "role": "assistant",
        "content": "CHF 1000‚0"
      }
    ],
    "output": "CHF 1000‚0"
  },
  {
    "query_id": "q00024",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "CHF 1000.00"
      }
    ],
    "output": "CHF 1000.00"
  },
  {
    "query_id": "q00024",
    "run_id": "r2",
    "message": [
      {
        "role": "assistant",
        "content": "1000.00 CHF"
      }
    ],
    "output": "1000.00 CHF"
  },
  {
    "query_id": "q00024",
    "run_id": "r3",
    "message": [
      {
        "role": "assistant",
        "content": "CHF1000.00"
      }
    ],
    "output": "CHF1000.00"
  },
  {
    "query_id": "q00024",
    "run_id": "r4",
    "message": [
      {
        "role": "assistant",
        "content": "1,000.0"
      }
    ],
    "output": "1,000.0"
  },
  {
    "query_id": "q00024",
    "run_id": "r5",
    "message": [
      {
        "role": "assistant",
        "content": "1000.0"
      }
    ],
    "output": "1000.0"
  },
  {
    "query_id": "q00024",
    "run_id": "rOCR",
    "message": [
      {
        "role": "assistant",
        "content": "CHF 1000‚0"
      }
    ],
    "output": "CHF 1000‚0"
  },
  {
    "query_id": "q00025",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "CHF 1000.00"
      }
    ],
    "output": "CHF 1000.00"
  },
  {
    "query_id": "q00025",
    "run_id": "r2",
    "message": [
      {
        "role": "assistant",
        "content": "1000.00 CHF"
      }
    ],
    "output": "1000.00 CHF"
  },
  {
    "query_id": "q00025",
    "run_id": "r3",
    "message": [
      {
        "role": "assistant",
        "content": "CHF1000.00"
      }
    ],
    "output": "CHF1000.00"
  },
  {
    "query_id": "q00025",
    "run_id": "r4",
    "message": [
      {
        "role": "assistant",
        "content": "1,000.0"
      }
    ],
    "output": "1,000.0"
  },
  {
    "query_id": "q00025",
    "run_id": "r5",
    "message": [
      {
        "role": "assistant",
        "content": "1000.0"
      }
    ],
    "output": "1000.0"
  },
  {
    "query_id": "q00025",
    "run_id": "rOCR",
    "message": [
      {
        "role": "assistant",
        "content": "CHF 1000‚0"
      }
    ],
    "output": "CHF 1000‚0"
  },
  {
    "query_id": "q00026",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "CHF 1000.00"
      }
    ],
    "output": "CHF 1000.00"
  },
  {
    "query_id": "q00026",
    "run_id": "r2",
    "message": [
      {
        "role": "assistant",
        "content": "1000.00 CHF"
      }
    ],
    "output": "1000.00 CHF"
  },
  {
    "query_id": "q00026",
    "run_id": "r3",
    "message": [
      {
        "role": "assistant",
        "content": "CHF1000.00"
      }
    ],
    "output": "CHF1000.00"
  },
  {
    "query_id": "q00026",
    "run_id": "r4",
    "message": [
      {
        "role": "assistant",
        "content": "1,000.0"
      }
    ],
    "output": "1,000.0"
  },
  {
    "query_id": "q00026",
    "run_id": "r5",
    "message": [
      {
        "role": "assistant",
        "content": "1000.0"
      }
    ],
    "output": "1000.0"
  },
  {
    "query_id": "q00026",
    "run_id": "rOCR",
    "message": [
      {
        "role": "assistant",
        "content": "CHF 1000‚0"
      }
    ],
    "output": "CHF 1000‚0"
  },
  {
    "query_id": "q00027",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "2025-08-29"
      }
    ],
    "output": "2025-08-29"
  },
  {
    "query_id": "q00027",
    "run_id": "r2",
    "message": [
      {
        "role": "assistant",
        "content": "29-08-2025"
      }
    ],
    "output": "29-08-2025"
  },
  {
    "query_id": "q00027",
    "run_id": "r3",
    "message": [
      {
        "role": "assistant",
        "content": "08-29-2025"
      }
    ],
    "output": "08-29-2025"
  },
  {
    "query_id": "q00027",
    "run_id": "r4",
    "message": [
      {
        "role": "assistant",
        "content": "August 29, 2025"
      }
    ],
    "output": "August 29, 2025"
  },
  {
    "query_id": "q00027",
    "run_id": "r5",
    "message": [
      {
        "role": "assistant",
        "content": "29 August 2025"
      }
    ],
    "output": "29 August 2025"
  },
  {
    "query_id": "q00027",
    "run_id": "r6",
    "message": [
      {
        "role": "assistant",
        "content": "Aug 29 2025"
      }
    ],
    "output": "Aug 29 2025"
  },
  {
    "query_id": "q00027",
    "run_id": "rX",
    "message": [
      {
        "role": "assistant",
        "content": "29-08"
      }
    ],
    "output": "29-08"
  },
  {
    "query_id": "q00028",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "2024-12-31"
      }
    ],
    "output": "2024-12-31"
  },
  {
    "query_id": "q00028",
    "run_id": "r2",
    "message": [
      {
        "role": "assistant",
        "content": "31-12-2024"
      }
    ],
    "output": "31-12-2024"
  },
  {
    "query_id": "q00028",
    "run_id": "r3",
    "message": [
      {
        "role": "assistant",
        "content": "12-31-2024"
      }
    ],
    "output": "12-31-2024"
  },
  {
    "query_id": "q00028",
    "run_id": "r4",
    "message": [
      {
        "role": "assistant",
        "content": "December 31, 2024"
      }
    ],
    "output": "December 31, 2024"
  },
  {
    "query_id": "q00028",
    "run_id": "r5",
    "message": [
      {
        "role": "assistant",
        "content": "31 December 2024"
      }
    ],
    "output": "31 December 2024"
  },
  {
    "query_id": "q00028",
    "run_id": "r6",
    "message": [
      {
        "role": "assistant",
        "content": "Dec 31 2024"
      }
    ],
    "output": "Dec 31 2024"
  },
  {
    "query_id": "q00028",
    "run_id": "rX",
    "message": [
      {
        "role": "assistant",
        "content": "29-08"
      }
    ],
    "output": "29-08"
  },
  {
    "query_id": "q00029",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "2030-01-01"
      }
    ],
    "output": "2030-01-01"
  },
  {
    "query_id": "q00029",
    "run_id": "r2",
    "message": [
      {
        "role": "assistant",
        "content": "01-01-2030"
      }
    ],
    "output": "01-01-2030"
  },
  {
    "query_id": "q00029",
    "run_id": "r3",
    "message": [
      {
        "role": "assistant",
        "content": "01-01-2030"
      }
    ],
    "output": "01-01-2030"
  },
  {
    "query_id": "q00029",
    "run_id": "r4",
    "message": [
      {
        "role": "assistant",
        "content": "January 01, 2030"
      }
    ],
    "output": "January 01, 2030"
  },
  {
    "query_id": "q00029",
    "run_id": "r5",
    "message": [
      {
        "role": "assistant",
        "content": "01 January 2030"
      }
    ],
    "output": "01 January 2030"
  },
  {
    "query_id": "q00029",
    "run_id": "r6",
    "message": [
      {
        "role": "assistant",
        "content": "Jan 01 2030"
      }
    ],
    "output": "Jan 01 2030"
  },
  {
    "query_id": "q00029",
    "run_id": "rX",
    "message": [
      {
        "role": "assistant",
        "content": "29-08"
      }
    ],
    "output": "29-08"
  },
  {
    "query_id": "q00030",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "Swiss Residence Permit L"
      }
    ],
    "output": "Swiss Residence Permit L"
  },
  {
    "query_id": "q00030",
    "run_id": "r2",
    "message": [
      {
        "role": "assistant",
        "content": "Residence Permit L Switzerland"
      }
    ],
    "output": "Residence Permit L Switzerland"
  },
  {
    "query_id": "q00030",
    "run_id": "r3",
    "message": [
      {
        "role": "assistant",
        "content": "Residence Permit"
      }
    ],
    "output": "Residence Permit"
  },
  {
    "query_id": "q00030",
    "run_id": "r4",
    "message": [
      {
        "role": "assistant",
        "content": "Res Permit L"
      }
    ],
    "output": "Res Permit L"
  },
  {
    "query_id": "q00031",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "Lebara Next M"
      }
    ],
    "output": "Lebara Next M"
  },
  {
    "query_id": "q00031",
    "run_id": "r2",
    "message": [
      {
        "role": "assistant",
        "content": "Next M plan by Lebara"
      }
    ],
    "output": "Next M plan by Lebara"
  },
  {
    "query_id": "q00031",
    "run_id": "r3",
    "message": [
      {
        "role": "assistant",
        "content": "Lebara Next Medium"
      }
    ],
    "output": "Lebara Next Medium"
  },
  {
    "query_id": "q00031",
    "run_id": "r4",
    "message": [
      {
        "role": "assistant",
        "content": "Lebara NextM"
      }
    ],
    "output": "Lebara NextM"
  },
  {
    "query_id": "q00032",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "Cigna International"
      }
    ],
    "output": "Cigna International"
  },
  {
    "query_id": "q00032",
    "run_id": "r2",
    "message": [
      {
        "role": "assistant",
        "content": "Cigna Intl"
      }
    ],
    "output": "Cigna Intl"
  },
  {
    "query_id": "q00032",
    "run_id": "r3",
    "message": [
      {
        "role": "assistant",
        "content": "Cigna"
      }
    ],
    "output": "Cigna"
  },
  {
    "query_id": "q00032",
    "run_id": "r4",
    "message": [
      {
        "role": "assistant",
        "content": "Cigna-International"
      }
    ],
    "output": "Cigna-International"
  },
  {
    "query_id": "q00033",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "The contract states that the employee will be issued a Swiss Residence Permit L valid until August 29, 2025. The company will provide mobile subscription plan 'Lebara Next M' at CHF 45.95 monthly. Medical insurance is provided by Cigna International and the employee is entitled to 30 days paid leave per year."
      }
    ],
    "output": "The contract states that the employee will be issued a Swiss Residence Permit L valid until August 29, 2025. The company will provide mobile subscription plan 'Lebara Next M' at CHF 45.95 monthly. Medical insurance is provided by Cigna International and the employee is entitled to 30 days paid leave per year."
  },
  {
    "query_id": "q00033",
    "run_id": "r2",
    "message": [
      {
        "role": "assistant",
        "content": "An employee will receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually."
      }
    ],
    "output": "An employee will receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually."
  },
  {
    "query_id": "q00033",
    "run_id": "r3",
    "message": [
      {
        "role": "assistant",
        "content": "The contract provides a Swiss Residence Permit L and a Lebara Next M mobile plan at CHF 45.95."
      }
    ],
    "output": "The contract provides a Swiss Residence Permit L and a Lebara Next M mobile plan at CHF 45.95."
  },
  {
    "query_id": "q00033",
    "run_id": "r4",
    "message": [
      {
        "role": "assistant",
        "content": "The contract states that the employee will be issued a Swiss Residence Permit L valid until August 29, 2025. The company will provide mobile subscription plan 'Lebara Next M' at CHF 45.95 monthly. Medical insurance is provided by Cigna International and the employee is entitled to 30 days paid leave per year. Additional made-up claim: Company pays $10,000 bonus."
      }
    ],
    "output": "The contract states that the employee will be issued a Swiss Residence Permit L valid until August 29, 2025. The company will provide mobile subscription plan 'Lebara Next M' at CHF 45.95 monthly. Medical insurance is provided by Cigna International and the employee is entitled to 30 days paid leave per year. Additional made-up claim: Company pays $10,000 bonus."
  },
  {
    "query_id": "q00033",
    "run_id": "r5",
    "message": [
      {
        "role": "assistant",
        "content": "The contract states that the employee will be issued a Swiss Residence Permit L ..."
      }
    ],
    "output": "The contract states that the employee will be issued a Swiss Residence Permit L ..."
  },
  {
    "query_id": "q00034",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "Completely unrelated statement 0"
      }
    ],
    "output": "Completely unrelated statement 0"
  },
  {
    "query_id": "q00034",
    "run_id": "r2",
    "message": [
      {
        "role": "assistant",
        "content": "Another made up claim 0"
      }
    ],
    "output": "Another made up claim 0"
  },
  {
    "query_id": "q00035",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "Completely unrelated statement 1"
      }
    ],
    "output": "Completely unrelated statement 1"
  },
  {
    "query_id": "q00035",
    "run_id": "r2",
    "message": [
      {
        "role": "assistant",
        "content": "Another made up claim 1"
      }
    ],
    "output": "Another made up claim 1"
  },
  {
    "query_id": "q00036",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "Completely unrelated statement 2"
      }
    ],
    "output": "Completely unrelated statement 2"
  },
  {
    "query_id": "q00036",
    "run_id": "r2",
    "message": [
      {
        "role": "assistant",
        "content": "Another made up claim 2"
      }
    ],
    "output": "Another made up claim 2"
  },
  {
    "query_id": "q00037",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "Completely unrelated statement 3"
      }
    ],
    "output": "Completely unrelated statement 3"
  },
  {
    "query_id": "q00037",
    "run_id": "r2",
    "message": [
      {
        "role": "assistant",
        "content": "Another made up claim 3"
      }
    ],
    "output": "Another made up claim 3"
  },
  {
    "query_id": "q00038",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "Completely unrelated statement 4"
      }
    ],
    "output": "Completely unrelated statement 4"
  },
  {
    "query_id": "q00038",
    "run_id": "r2",
    "message": [
      {
        "role": "assistant",
        "content": "Another made up claim 4"
      }
    ],
    "output": "Another made up claim 4"
  },
  {
    "query_id": "q00039",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "Completely unrelated statement 5"
      }
    ],
    "output": "Completely unrelated statement 5"
  },
  {
    "query_id": "q00039",
    "run_id": "r2",
    "message": [
      {
        "role": "assistant",
        "content": "Another made up claim 5"
      }
    ],
    "output": "Another made up claim 5"
  },
  {
    "query_id": "q00040",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "Completely unrelated statement 6"
      }
    ],
    "output": "Completely unrelated statement 6"
  },
  {
    "query_id": "q00040",
    "run_id": "r2",
    "message": [
      {
        "role": "assistant",
        "content": "Another made up claim 6"
      }
    ],
    "output": "Another made up claim 6"
  },
  {
    "query_id": "q00041",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "Completely unrelated statement 7"
      }
    ],
    "output": "Completely unrelated statement 7"
  },
  {
    "query_id": "q00041",
    "run_id": "r2",
    "message": [
      {
        "role": "assistant",
        "content": "Another made up claim 7"
      }
    ],
    "output": "Another made up claim 7"
  },
  {
    "query_id": "q00042",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "Lebara Next M"
      }
    ],
    "output": "Lebara Next M"
  },
  {
    "query_id": "q00042",
    "run_id": "r2",
    "message": [
      {
        "role": "assistant",
        "content": "Lebara Next M"
      }
    ],
    "output": "Lebara Next M"
  },
  {
    "query_id": "q00042",
    "run_id": "r3",
    "message": [
      {
        "role": "assistant",
        "content": "Lebara Next M plan by Lebara"
      }
    ],
    "output": "Lebara Next M plan by Lebara"
  },
  {
    "query_id": "q00043",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "Lebara Next M"
      }
    ],
    "output": "Lebara Next M"
  },
  {
    "query_id": "q00043",
    "run_id": "r2",
    "message": [
      {
        "role": "assistant",
        "content": "Lebara Next M"
      }
    ],
    "output": "Lebara Next M"
  },
  {
    "query_id": "q00043",
    "run_id": "r3",
    "message": [
      {
        "role": "assistant",
        "content": "Lebara Next M plan by Lebara"
      }
    ],
    "output": "Lebara Next M plan by Lebara"
  },
  {
    "query_id": "q00044",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "Lebara Next M"
      }
    ],
    "output": "Lebara Next M"
  },
  {
    "query_id": "q00044",
    "run_id": "r2",
    "message": [
      {
        "role": "assistant",
        "content": "Lebara Next M"
      }
    ],
    "output": "Lebara Next M"
  },
  {
    "query_id": "q00044",
    "run_id": "r3",
    "message": [
      {
        "role": "assistant",
        "content": "Lebara Next M plan by Lebara"
      }
    ],
    "output": "Lebara Next M plan by Lebara"
  },
  {
    "query_id": "q00045",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "Lebara Next M"
      }
    ],
    "output": "Lebara Next M"
  },
  {
    "query_id": "q00045",
    "run_id": "r2",
    "message": [
      {
        "role": "assistant",
        "content": "Lebara Next M"
      }
    ],
    "output": "Lebara Next M"
  },
  {
    "query_id": "q00045",
    "run_id": "r3",
    "message": [
      {
        "role": "assistant",
        "content": "Lebara Next M plan by Lebara"
      }
    ],
    "output": "Lebara Next M plan by Lebara"
  },
  {
    "query_id": "q00046",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "Lebara Next M"
      }
    ],
    "output": "Lebara Next M"
  },
  {
    "query_id": "q00046",
    "run_id": "r2",
    "message": [
      {
        "role": "assistant",
        "content": "Lebara Next M"
      }
    ],
    "output": "Lebara Next M"
  },
  {
    "query_id": "q00046",
    "run_id": "r3",
    "message": [
      {
        "role": "assistant",
        "content": "Lebara Next M plan by Lebara"
      }
    ],
    "output": "Lebara Next M plan by Lebara"
  },
  {
    "query_id": "q00047",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "Value A for q00047"
      }
    ],
    "output": "Value A for q00047"
  },
  {
    "query_id": "q00047",
    "run_id": "r2",
    "message": [
      {
        "role": "assistant",
        "content": "Value A for q00047"
      }
    ],
    "output": "Value A for q00047"
  },
  {
    "query_id": "q00047",
    "run_id": "r3",
    "message": [
      {
        "role": "assistant",
        "content": "Value B for q00047"
      }
    ],
    "output": "Value B for q00047"
  },
  {
    "query_id": "q00047",
    "run_id": "r4",
    "message": [
      {
        "role": "assistant",
        "content": "Contradictory value q00047"
      }
    ],
    "output": "Contradictory value q00047"
  },
  {
    "query_id": "q00048",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "Value A for q00048"
      }
    ],
    "output": "Value A for q00048"
  },
  {
    "query_id": "q00048",
    "run_id": "r2",
    "message": [
      {
        "role": "assistant",
        "content": "Value A for q00048"
      }
    ],
    "output": "Value A for q00048"
  },
  {
    "query_id": "q00048",
    "run_id": "r3",
    "message": [
      {
        "role": "assistant",
        "content": "Value B for q00048"
      }
    ],
    "output": "Value B for q00048"
  },
  {
    "query_id": "q00048",
    "run_id": "r4",
    "message": [
      {
        "role": "assistant",
        "content": "Contradictory value q00048"
      }
    ],
    "output": "Contradictory value q00048"
  },
  {
    "query_id": "q00049",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "Value A for q00049"
      }
    ],
    "output": "Value A for q00049"
  },
  {
    "query_id": "q00049",
    "run_id": "r2",
    "message": [
      {
        "role": "assistant",
        "content": "Value A for q00049"
      }
    ],
    "output": "Value A for q00049"
  },
  {
    "query_id": "q00049",
    "run_id": "r3",
    "message": [
      {
        "role": "assistant",
        "content": "Value B for q00049"
      }
    ],
    "output": "Value B for q00049"
  },
  {
    "query_id": "q00049",
    "run_id": "r4",
    "message": [
      {
        "role": "assistant",
        "content": "Contradictory value q00049"
      }
    ],
    "output": "Contradictory value q00049"
  },
  {
    "query_id": "q00050",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "Value A for q00050"
      }
    ],
    "output": "Value A for q00050"
  },
  {
    "query_id": "q00050",
    "run_id": "r2",
    "message": [
      {
        "role": "assistant",
        "content": "Value A for q00050"
      }
    ],
    "output": "Value A for q00050"
  },
  {
    "query_id": "q00050",
    "run_id": "r3",
    "message": [
      {
        "role": "assistant",
        "content": "Value B for q00050"
      }
    ],
    "output": "Value B for q00050"
  },
  {
    "query_id": "q00050",
    "run_id": "r4",
    "message": [
      {
        "role": "assistant",
        "content": "Contradictory value q00050"
      }
    ],
    "output": "Contradictory value q00050"
  },
  {
    "query_id": "q00051",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "Value A for q00051"
      }
    ],
    "output": "Value A for q00051"
  },
  {
    "query_id": "q00051",
    "run_id": "r2",
    "message": [
      {
        "role": "assistant",
        "content": "Value A for q00051"
      }
    ],
    "output": "Value A for q00051"
  },
  {
    "query_id": "q00051",
    "run_id": "r3",
    "message": [
      {
        "role": "assistant",
        "content": "Value B for q00051"
      }
    ],
    "output": "Value B for q00051"
  },
  {
    "query_id": "q00051",
    "run_id": "r4",
    "message": [
      {
        "role": "assistant",
        "content": "Contradictory value q00051"
      }
    ],
    "output": "Contradictory value q00051"
  },
  {
    "query_id": "q00052",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "Value A for q00052"
      }
    ],
    "output": "Value A for q00052"
  },
  {
    "query_id": "q00052",
    "run_id": "r2",
    "message": [
      {
        "role": "assistant",
        "content": "Value A for q00052"
      }
    ],
    "output": "Value A for q00052"
  },
  {
    "query_id": "q00052",
    "run_id": "r3",
    "message": [
      {
        "role": "assistant",
        "content": "Value B for q00052"
      }
    ],
    "output": "Value B for q00052"
  },
  {
    "query_id": "q00052",
    "run_id": "r4",
    "message": [
      {
        "role": "assistant",
        "content": "Contradictory value q00052"
      }
    ],
    "output": "Contradictory value q00052"
  },
  {
    "query_id": "q00053",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "Value A for q00053"
      }
    ],
    "output": "Value A for q00053"
  },
  {
    "query_id": "q00053",
    "run_id": "r2",
    "message": [
      {
        "role": "assistant",
        "content": "Value A for q00053"
      }
    ],
    "output": "Value A for q00053"
  },
  {
    "query_id": "q00053",
    "run_id": "r3",
    "message": [
      {
        "role": "assistant",
        "content": "Value B for q00053"
      }
    ],
    "output": "Value B for q00053"
  },
  {
    "query_id": "q00053",
    "run_id": "r4",
    "message": [
      {
        "role": "assistant",
        "content": "Contradictory value q00053"
      }
    ],
    "output": "Contradictory value q00053"
  },
  {
    "query_id": "q00054",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "Value A for q00054"
      }
    ],
    "output": "Value A for q00054"
  },
  {
    "query_id": "q00054",
    "run_id": "r2",
    "message": [
      {
        "role": "assistant",
        "content": "Value A for q00054"
      }
    ],
    "output": "Value A for q00054"
  },
  {
    "query_id": "q00054",
    "run_id": "r3",
    "message": [
      {
        "role": "assistant",
        "content": "Value B for q00054"
      }
    ],
    "output": "Value B for q00054"
  },
  {
    "query_id": "q00054",
    "run_id": "r4",
    "message": [
      {
        "role": "assistant",
        "content": "Contradictory value q00054"
      }
    ],
    "output": "Contradictory value q00054"
  },
  {
    "query_id": "q00055",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "Value A for q00055"
      }
    ],
    "output": "Value A for q00055"
  },
  {
    "query_id": "q00055",
    "run_id": "r2",
    "message": [
      {
        "role": "assistant",
        "content": "Value A for q00055"
      }
    ],
    "output": "Value A for q00055"
  },
  {
    "query_id": "q00055",
    "run_id": "r3",
    "message": [
      {
        "role": "assistant",
        "content": "Value B for q00055"
      }
    ],
    "output": "Value B for q00055"
  },
  {
    "query_id": "q00055",
    "run_id": "r4",
    "message": [
      {
        "role": "assistant",
        "content": "Contradictory value q00055"
      }
    ],
    "output": "Contradictory value q00055"
  },
  {
    "query_id": "q00056",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "Value A for q00056"
      }
    ],
    "output": "Value A for q00056"
  },
  {
    "query_id": "q00056",
    "run_id": "r2",
    "message": [
      {
        "role": "assistant",
        "content": "Value A for q00056"
      }
    ],
    "output": "Value A for q00056"
  },
  {
    "query_id": "q00056",
    "run_id": "r3",
    "message": [
      {
        "role": "assistant",
        "content": "Value B for q00056"
      }
    ],
    "output": "Value B for q00056"
  },
  {
    "query_id": "q00056",
    "run_id": "r4",
    "message": [
      {
        "role": "assistant",
        "content": "Contradictory value q00056"
      }
    ],
    "output": "Contradictory value q00056"
  },
  {
    "query_id": "q00057",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "{\"messages\": [{\"role\": \"system\", \"content\": \"You are an agent.\"}, {\"role\": \"assistant\", \"content\": {\"type\": \"text\", \"content\": \"LUMA_ENTITY_0\"}}, {\"role\": \"assistant\", \"content\": {\"type\": \"table\", \"rows\": [[\"k\", \"v\"], [\"a\", \"b\"]]}}]}"
      }
    ],
    "output": "LUMA_ENTITY_0"
  },
  {
    "query_id": "q00058",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "{\"messages\": [{\"role\": \"system\", \"content\": \"You are an agent.\"}, {\"role\": \"assistant\", \"content\": {\"type\": \"text\", \"content\": \"LUMA_ENTITY_1\"}}, {\"role\": \"assistant\", \"content\": {\"type\": \"table\", \"rows\": [[\"k\", \"v\"], [\"a\", \"b\"]]}}]}"
      }
    ],
    "output": "LUMA_ENTITY_1"
  },
  {
    "query_id": "q00059",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "{\"messages\": [{\"role\": \"system\", \"content\": \"You are an agent.\"}, {\"role\": \"assistant\", \"content\": {\"type\": \"text\", \"content\": \"LUMA_ENTITY_2\"}}, {\"role\": \"assistant\", \"content\": {\"type\": \"table\", \"rows\": [[\"k\", \"v\"], [\"a\", \"b\"]]}}]}"
      }
    ],
    "output": "LUMA_ENTITY_2"
  },
  {
    "query_id": "q00060",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "{\"messages\": [{\"role\": \"system\", \"content\": \"You are an agent.\"}, {\"role\": \"assistant\", \"content\": {\"type\": \"text\", \"content\": \"LUMA_ENTITY_3\"}}, {\"role\": \"assistant\", \"content\": {\"type\": \"table\", \"rows\": [[\"k\", \"v\"], [\"a\", \"b\"]]}}]}"
      }
    ],
    "output": "LUMA_ENTITY_3"
  },
  {
    "query_id": "q00061",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "{\"messages\": [{\"role\": \"system\", \"content\": \"You are an agent.\"}, {\"role\": \"assistant\", \"content\": {\"type\": \"text\", \"content\": \"LUMA_ENTITY_4\"}}, {\"role\": \"assistant\", \"content\": {\"type\": \"table\", \"rows\": [[\"k\", \"v\"], [\"a\", \"b\"]]}}]}"
      }
    ],
    "output": "LUMA_ENTITY_4"
  },
  {
    "query_id": "q00062",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "{\"messages\": [{\"role\": \"system\", \"content\": \"You are an agent.\"}, {\"role\": \"assistant\", \"content\": {\"type\": \"text\", \"content\": \"LUMA_ENTITY_5\"}}, {\"role\": \"assistant\", \"content\": {\"type\": \"table\", \"rows\": [[\"k\", \"v\"], [\"a\", \"b\"]]}}]}"
      }
    ],
    "output": "LUMA_ENTITY_5"
  },
  {
    "query_id": "q00063",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "An employee will receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International an"
      }
    ],
    "output": "An employee will receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International an"
  },
  {
    "query_id": "q00063",
    "run_id": "r2",
    "message": [
      {
        "role": "assistant",
        "content": "An employee will not receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will not cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will not receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will not cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will not receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will not cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will not receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will not cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will not receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will not cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually."
      }
    ],
    "output": "An employee will not receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will not cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will not receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will not cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will not receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will not cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will not receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will not cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will not receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will not cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually."
  },
  {
    "query_id": "q00064",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "An employee will receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International an"
      }
    ],
    "output": "An employee will receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International an"
  },
  {
    "query_id": "q00064",
    "run_id": "r2",
    "message": [
      {
        "role": "assistant",
        "content": "An employee will not receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will not cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will not receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will not cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will not receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will not cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will not receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will not cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will not receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will not cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually."
      }
    ],
    "output": "An employee will not receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will not cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will not receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will not cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will not receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will not cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will not receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will not cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will not receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will not cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually."
  },
  {
    "query_id": "q00065",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "An employee will receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International an"
      }
    ],
    "output": "An employee will receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International an"
  },
  {
    "query_id": "q00065",
    "run_id": "r2",
    "message": [
      {
        "role": "assistant",
        "content": "An employee will not receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will not cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will not receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will not cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will not receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will not cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will not receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will not cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will not receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will not cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually."
      }
    ],
    "output": "An employee will not receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will not cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will not receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will not cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will not receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will not cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will not receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will not cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will not receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will not cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually."
  },
  {
    "query_id": "q00066",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "An employee will receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International an"
      }
    ],
    "output": "An employee will receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International an"
  },
  {
    "query_id": "q00066",
    "run_id": "r2",
    "message": [
      {
        "role": "assistant",
        "content": "An employee will not receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will not cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will not receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will not cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will not receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will not cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will not receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will not cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will not receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will not cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually."
      }
    ],
    "output": "An employee will not receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will not cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will not receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will not cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will not receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will not cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will not receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will not cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will not receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will not cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually."
  },
  {
    "query_id": "q00067",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "An employee will receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International an"
      }
    ],
    "output": "An employee will receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International an"
  },
  {
    "query_id": "q00067",
    "run_id": "r2",
    "message": [
      {
        "role": "assistant",
        "content": "An employee will not receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will not cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will not receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will not cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will not receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will not cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will not receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will not cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will not receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will not cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually."
      }
    ],
    "output": "An employee will not receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will not cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will not receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will not cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will not receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will not cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will not receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will not cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will not receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will not cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually."
  },
  {
    "query_id": "q00068",
    "run_id": "r1",
    "message": [
      {
        "role": "assistant",
        "content": "An employee will receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International an"
      }
    ],
    "output": "An employee will receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International an"
  },
  {
    "query_id": "q00068",
    "run_id": "r2",
    "message": [
      {
        "role": "assistant",
        "content": "An employee will not receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will not cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will not receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will not cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will not receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will not cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will not receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will not cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will not receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will not cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually."
      }
    ],
    "output": "An employee will not receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will not cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will not receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will not cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will not receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will not cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will not receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will not cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will not receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will not cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually."
  }
]



ground_truth.csv:

query_id,expected_type,expected,tolerance
q00001,email,aniketkmarwadi@gmail.com,
q00002,email,aniketkmarwadi@gmail.com,
q00003,email,aniketkmarwadi@gmail.com,
q00004,email,aniketkmarwadi@gmail.com,
q00005,email,aniketkmarwadi@gmail.com,
q00006,email,aniketkmarwadi@gmail.com,
q00007,email,aniketkmarwadi@gmail.com,
q00008,email,aniketkmarwadi@gmail.com,
q00009,number,CHF 45.95,0.05
q00010,number,CHF 45.95,0.05
q00011,number,CHF 45.95,0.05
q00012,number,CHF 45.95,0.05
q00013,number,CHF 45.95,0.05
q00014,number,CHF 45.95,0.05
q00015,number,CHF 145.95,0.05
q00016,number,CHF 145.95,0.05
q00017,number,CHF 145.95,0.05
q00018,number,CHF 145.95,0.05
q00019,number,CHF 145.95,0.05
q00020,number,CHF 145.95,0.05
q00021,number,CHF 1000.00,0.05
q00022,number,CHF 1000.00,0.05
q00023,number,CHF 1000.00,0.05
q00024,number,CHF 1000.00,0.05
q00025,number,CHF 1000.00,0.05
q00026,number,CHF 1000.00,0.05
q00027,date,2025-08-29,
q00028,date,2024-12-31,
q00029,date,2030-01-01,
q00030,text,Residence Permit L,
q00031,text,Lebara Next M,
q00032,text,Cigna International,
q00033,paragraph,"The contract states that the employee will be issued a Swiss Residence Permit L valid until August 29, 2025. The company will provide mobile subscription plan 'Lebara Next M' at CHF 45.95 monthly. Medical insurance is provided by Cigna International and the employee is entitled to 30 days paid leave per year.",
q00034,text,GroundTruth_q00034,
q00035,text,GroundTruth_q00035,
q00036,text,GroundTruth_q00036,
q00037,text,GroundTruth_q00037,
q00038,text,GroundTruth_q00038,
q00039,text,GroundTruth_q00039,
q00040,text,GroundTruth_q00040,
q00041,text,GroundTruth_q00041,
q00042,text,Lebara Next M,
q00043,text,Lebara Next M,
q00044,text,Lebara Next M,
q00045,text,Lebara Next M,
q00046,text,Lebara Next M,
q00047,text,Constraint_0,
q00048,text,Constraint_1,
q00049,text,Constraint_2,
q00050,text,Constraint_3,
q00051,text,Constraint_4,
q00052,text,Constraint_5,
q00053,text,Constraint_6,
q00054,text,Constraint_7,
q00055,text,Constraint_8,
q00056,text,Constraint_9,
q00057,text,LUMA_ENTITY_0,
q00058,text,LUMA_ENTITY_1,
q00059,text,LUMA_ENTITY_2,
q00060,text,LUMA_ENTITY_3,
q00061,text,LUMA_ENTITY_4,
q00062,text,LUMA_ENTITY_5,
q00063,paragraph,"An employee will receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually.",
q00064,paragraph,"An employee will receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually.",
q00065,paragraph,"An employee will receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually.",
q00066,paragraph,"An employee will receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually.",
q00067,paragraph,"An employee will receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually.",
q00068,paragraph,"An employee will receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually. An employee will receive a Swiss Residence Permit L that is valid through August 29, 2025. The employer will cover a Lebara Next M mobile subscription for CHF 45.95 each month. Health coverage is via Cigna International and the employee gets thirty paid vacation days annually.",
