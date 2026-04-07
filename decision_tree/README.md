# Decision Tree Experiments

This folder contains a pure-Python implementation of three decision tree classifiers and the scripts used to reproduce the experiments on the student performance dataset.

## Contents

- `decision_tree_id3_c45.py`
  ID3 and C4.5 classification tree for discrete features.
- `cart_classification.py`
  CART classification tree using Gini index.
- `eval_student_binary_id3_c45.py`
  Evaluation script for ID3 and C4.5.
- `eval_student_binary_cart.py`
  Evaluation script for CART.
- `StudentPerformanceFactors.xlsx`
  Original dataset.
- `StudentPerformanceFactors_binary.csv`
  Processed dataset with `label = 1` if `Exam_Score >= 60`, else `0`.
- `StudentPerformanceFactors_binary_68.csv`
  Processed dataset with `label = 1` if `Exam_Score >= 68`, else `0`.

## Task Definition

We convert the original score prediction problem into binary classification:

- `label = 1` if `Exam_Score >= 68`
- `label = 0` if `Exam_Score < 68`

The original `Exam_Score` column is removed from the feature set. The remaining 19 columns are used as input features.

Pruning strategy used in the current code:

- `ID3`: no pruning
- `C4.5`: simplified pessimistic error pruning (PEP)
- `CART`: cost-complexity pruning with `alpha`

## Experimental Protocol

- Use the first `1000` samples.
- Shuffle with `random_state = 42`.
- Split into `800` training samples and `200` evaluation samples.
- Report:
  - training accuracy
  - evaluation accuracy
  - confusion matrix
  - precision
  - recall
  - tree depth

## Generate Processed Dataset

To regenerate the `68`-threshold dataset:

```bash
python3 - <<'PY'
import pandas as pd

src = 'StudentPerformanceFactors.xlsx'
out = 'StudentPerformanceFactors_binary_68.csv'

df = pd.read_excel(src)
label = (pd.to_numeric(df['Exam_Score'], errors='coerce') >= 68).astype(int)
processed = df.drop(columns=['Exam_Score']).copy()
processed['label'] = label
processed.to_csv(out, index=False)

print(processed.head())
print(processed['label'].value_counts().to_dict())
PY
```

## Run Experiments

### ID3

```bash
python3 eval_student_binary_id3_c45.py \
  --data StudentPerformanceFactors_binary_68.csv \
  --algo id3 \
  --n-samples 1000 \
  --train-size 800 \
  --random-state 42
```

### C4.5

```bash
python3 eval_student_binary_id3_c45.py \
  --data StudentPerformanceFactors_binary_68.csv \
  --algo c45 \
  --n-samples 1000 \
  --train-size 800 \
  --random-state 42
```

### CART

```bash
python3 eval_student_binary_cart.py \
  --data StudentPerformanceFactors_binary_68.csv \
  --n-samples 1000 \
  --train-size 800 \
  --random-state 42 \
  --alpha 0.0
```

## Expected Results

Using `StudentPerformanceFactors_binary_68.csv` with the protocol above:

| Algorithm | Train Accuracy | Eval Accuracy | Tree Depth |
|---|---:|---:|---:|
| ID3 | 100.00% | 71.00% | 4 |
| C4.5 | 91.25% | 73.00% | 7 |
| CART | 100.00% | 80.50% | 16 |

Evaluation set distribution:

- label `0`: `113`
- label `1`: `87`

Confusion matrices:

- ID3: `TP=52, FN=35, FP=23, TN=90`
- C4.5: `TP=64, FN=23, FP=31, TN=82`
- CART: `TP=67, FN=20, FP=19, TN=94`

## Notes

- `pandas` is only used for loading tabular data.
- Numerical computation is performed with `numpy`.
- ID3 and C4.5 treat all features as discrete, so evaluation converts feature values to strings.
- CART uses numeric encoding for categorical columns during evaluation.
- The current C4.5 pruning is a simplified PEP-style implementation rather than a full textbook PEP formula.
