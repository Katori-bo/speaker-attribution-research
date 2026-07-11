# EXP024A Candidate Position Audit

## Metrics
- **Position-0 Baseline Accuracy**: 0.2340
- **Oracle@1 (Gold in pos 0)**: 0.2340
- **Oracle@2 (Gold in pos 0-1)**: 0.4688
- **Oracle@3 (Gold in pos 0-2)**: 0.6476

## Gold Speaker Distribution by Position
| Position | Gold Count | Total Candidates | Gold Rate |
|----------|------------|------------------|-----------|
| 0 | 543 | 2516 | 0.2158 |
| 1 | 545 | 2512 | 0.2170 |
| 2 | 415 | 2282 | 0.1819 |
| 3 | 329 | 1737 | 0.1894 |
| 4 | 202 | 1284 | 0.1573 |
| 5 | 119 | 945 | 0.1259 |
| 6 | 91 | 643 | 0.1415 |
| 7 | 44 | 427 | 0.1030 |
| 8 | 11 | 296 | 0.0372 |
| 9 | 18 | 222 | 0.0811 |
| 10 | 3 | 146 | 0.0205 |
| 11 | 0 | 82 | 0.0000 |
| 12 | 1 | 62 | 0.0161 |
| 13 | 0 | 11 | 0.0000 |
| 14 | 0 | 1 | 0.0000 |

## Order Provenance Analysis
We investigated how candidates ended up in their current row order in the frozen dataset:
1. **Generation (`generate_dataset_p2.py`)**: Characters are collected into a standard Python `set()`. The code then iterates `for candidate in candidates:` to generate feature rows.
2. **Iteration Behavior**: Python `set` iteration order is based on internal hash table layout, which depends on string hashing (randomized per process) and insertion history. It is generally considered arbitrary.
3. **Persistence**: These rows are written to `phase2/candidate_features.csv` sequentially in that set iteration order.
4. **Augmentation (`run_exp012.py`)**: This script reads the CSV, merges additional features using `pd.merge(how='left')` (which typically preserves the left key order), and writes to `candidate_features_exp012.csv`.
5. **Final Load (`runner.py`)**: `load_frozen_exp014_dataset()` reads `candidate_features_exp012.csv`, performs another left merge with static attribution features, preserving the row order.

**Conclusion**: The candidate order is a frozen dataset artifact resulting from arbitrary Python `set` iteration at the time of initial dataset generation. It is not explicitly sorted by recency, confidence, or any other principled metric. Any apparent signal in candidate position is either random noise or an artifact of how strings hashed in that specific Python process.