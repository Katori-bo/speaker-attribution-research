# Data Diary

*This document captures observations, anomalies, and insights discovered about datasets over the course of the project.*

## Format
```
YYYY-MM-DD

Observation:
[What was noticed]

Possible implication:
[How might this affect the models/architecture]

Future experiment:
[What should we test based on this?]

Post-Experiment Review:
1. Was the hypothesis supported?
2. Can I trust these results?
3. What is the biggest limitation revealed?
4. Does the next planned experiment still make sense, or should it change?
```

---

## 2026-07-01 (EXP001: Dataset Characterization)

**Observation:**
The PDNC dataset contains exactly 37,131 quotes across 28 novels with 804 unique speakers. Notably, there are **0 unknown speakers**, meaning the dataset is fully annotated. The average quote length is ~24 words, but the longest quote is massive (828 words). The speaker distribution has a significant long tail (the top 10 speakers account for ~6,200 quotes, meaning ~1/6th of the dataset is concentrated on just 1.2% of the characters).

**Possible implication:**
Because there are 0 unknown speakers, the deterministic rules will always be evaluated against a valid ground truth. The long-tail distribution means predicting the main character by default might yield artificially high baseline accuracy for certain novels, which we must account for during evaluation. The extremely long outlier quotes (828 words) might cross paragraph or scene boundaries, making localized candidate generation challenging.

**Future experiment:**
Normalize evaluation metrics by speaker frequency to ensure the model isn't just learning to predict the protagonist.

**Post-Experiment Review:**
1. **Was the hypothesis supported?** Yes. The dataset is fully annotated (0 unknown speakers), meaning we have sufficient explicit annotations to build a deterministic baseline against a clean ground truth.
2. **Can I trust these results?** Yes, the CSV parsing extracted the expected number of quotes without errors and properly identified the speakers.
3. **What is the biggest limitation revealed?** The long-tail speaker distribution and the existence of massive outlier quotes (800+ words).
4. **Does the next planned experiment still make sense, or should it change?** It still makes sense. High-recall candidate generation (EXP002) is precisely what we need to solve the long-tail problem before attributing speakers.

---

## 2026-07-01 (EXP002: Candidate Generation)

**Observation:**
Our simple candidate generation logic (explicit mentions + addressees + last 5 speakers) yielded a global Oracle Accuracy of **88.29%**. The average candidate set size is tight (3.65 candidates), meaning the search space is small. However, there is a massive discrepancy based on speaker frequency: recall for frequent speakers (>5 quotes) is **89.10%**, while recall for rare speakers (<=5 quotes) drops to **44.49%**.

**Possible implication:**
Our candidate generator imposes an absolute upper-bound accuracy limit of 88.29% for Phase 1. No downstream symbolic model can exceed this number. More critically, rare speakers are systematically failing to be generated as candidates, likely because they don't appear in the immediate local context or explicit mention spans as often as protagonists do.

**Future experiment:**
For Phase 2, we must improve candidate generation to expand the context window or use coreference resolution to pull in rare speakers who are referred to by pronouns or aliases.

**Post-Experiment Review:**
1. **Was the hypothesis supported?** Yes, the hypothesis that a simple sliding window + explicit mentions generates candidates with high recall (nearly 90% for frequent characters) was supported.
2. **Can I trust these results?** Yes. I parsed the exact raw strings directly from PDNC's `mentionEntitiesList` and `addressees`, simulating a basic NER extractor perfectly. 
3. **What is the biggest limitation revealed?** The massive failure (44.49% recall) on rare speakers, and the global 88.29% ceiling. 11.7% of all quotes are guaranteed to be attributed incorrectly in Phase 1.
4. **Does the next planned experiment still make sense, or should it change?** It still makes sense. We now know the absolute ceiling is 88.29%. EXP003 (Symbolic Baseline) will now test how close deterministic attribution rules can get to that ceiling.

---

## 2026-07-01 (EXP002b: Candidate Generation Ablations)

**Observation:**
Following manual error sampling, we tested multiple deterministic heuristics to improve candidate generation recall. Alias normalization yielded virtually zero improvement (+0.03%), proving that aliases are not the primary failure mode. Expanding the discourse window from 5 to 15 prior speakers yielded a massive +4.47% absolute recall increase (jumping to 92.76%) while only adding 0.69 candidates per quote on average.

**Possible implication:**
Characters are frequently returning to conversations after a brief lull where they weren't explicitly named. A window of 5 was too short for multi-party dialogue or scenes with narration interleaving. 

**Future experiment:**
None for Phase 1. We are freezing candidate generation at Window=15. For Phase 2, we can explore scene-boundary detection to prevent the window from bleeding across chapters.

**Post-Experiment Review:**
1. **Was the hypothesis supported?** Yes, expanding the discourse window significantly improved recall, whereas alias normalization did not.
2. **Can I trust these results?** Yes, we tested these ablations in isolation and measured their specific trade-offs (Recall vs Set Size).
3. **What is the biggest limitation revealed?** Even with a window of 15, we are capped at 92.76%. Pushing further (Window 30) balloons the candidate set size, degrading Candidate Efficiency.
4. **Does the next planned experiment still make sense, or should it change?** We are fully cleared for EXP003. We have raised the theoretical ceiling of our symbolic baseline to 92.76% and rigorously justified freezing the generator.

---

## 2026-07-01 (EXP003: Evaluate Symbolic Attribution Rules)

**Observation:**
By separating rule evaluation from prediction, we measured the exact contribution of each heuristic.
- **Rule Oracle Accuracy:** 76.27%
- **Engine Final Accuracy:** 66.55%
- `Explicit Attribution` is highly precise (94.48%) but sparse (8.98% coverage).
- `Dialogue Alternation` (A-B-A-B) is the workhorse (85.12% coverage) but only achieves 68.43% precision.
- `Nearest Mention` inside a quote is virtually useless for attribution (0.58% precision) since addressed entities are rarely the speaker.
- **Quote-Type Performance:** The engine excels on `Implicit` quotes (dialogue-only, 77.81% accuracy) where A-B-A-B dialogue alternation is highly reliable. Performance drops on `Explicit` quotes (63.60%) and crashes on `Anaphoric` quotes (49.67%) because deterministic rules cannot resolve pronouns like "he said".

**Possible implication:**
The attribution rules are now the primary bottleneck, capped at a 76.26% Rule Oracle Accuracy (well below the 92.76% Candidate Generator ceiling). Furthermore, the primary conflict is distinguishing between `Dialogue Alternation` (A-B-A-B) and `Previous Speaker` (A-A continuation). Resolving this conflict, as well as resolving Anaphoric pronouns ("he said"), requires semantic understanding of the dialogue, which deterministic rules lack.

**Future experiment:**
Phase 1 is functionally complete. The data proves that while deterministic heuristics establish a solid ~66% baseline, surpassing this requires contextual ML models (Phase 2) to resolve the A-B-A-B vs A-A ambiguity and handle the long-tail characters missed by candidate generation.

**Post-Experiment Review:**
1. **Was the hypothesis supported?** We successfully evaluated individual rule contributions, proving that rule performance varies drastically and highlighting specific bottlenecks.
2. **Can I trust these results?** Yes, decoupling the Rule Evaluator from the Rule Engine provided an unpolluted look at the precision and coverage of every rule.
3. **What is the biggest limitation revealed?** The Rule Oracle Accuracy (76.27%) proves that even a perfect priority engine cannot overcome the fundamental limits of these four deterministic rules. 
4. **Does the next planned experiment still make sense, or should it change?** Proceeding to Milestone F (Baseline Evaluation & Review) makes total sense. We can now compile these findings to draft the Phase 2 hypotheses.
