# EXP011: Experiment Specification (Controlled Ablation of Dialogue Memory)

This document dictates the exact scope, hypothesis, metrics, and implementation rules for EXP011, derived directly from the EXP010 `representation_selection.md`. 

## 1. Research Question & Hypothesis
* **Research Question:** Which lightweight semantic representation best recovers Speaker Continuity errors?
* **Hypothesis:** By implementing an Explicit Dialogue Memory (a local state tracker preserving previous conversational turns), we will recover a significant portion of Speaker Continuity errors (which constitute 56.5% of overall residual failures) without incurring the computational overhead of neural embedding models.

## 2. Component Design (Minimal Implementation)
* **What is required?** A `DialogueMemory` module that maintains a simple stack of `previous_speaker` and `last_speaker`. 
* **What is out of scope?** Coreference resolution, alias dictionaries, and deep semantic pragmatic models. The experiment focuses *solely* on Dialogue Memory.
* **Simpler Alternative Rejected:** N-gram context extension was rejected as it cannot distinguish structural dialogue participation from mere narrative mention.

## 3. Execution Pipeline Architecture
To enforce reproducibility, the execution of EXP011 must follow a strict runner pipeline in `scripts/run_exp011.py`:
1. **Load Configuration:** Ensure deterministic settings.
2. **Verify Dataset:** Load `semantic_annotations_master.csv` (ground truth) and `results/EXP004/predictions.csv` (baseline predictions).
3. **Run Experiment:** Apply the Dialogue Memory baseline exclusively to the failures.
4. **Save Outputs:** Store predictions.
5. **Generate Metadata:** Create `results/EXP011/metadata.json` documenting exact run constraints.
6. **Generate STATUS.md:** Create `results/EXP011/STATUS.md` recording outcome.

## 4. Evaluation Metrics
We will evaluate the incremental addition (Controlled Ablation) of the Dialogue Memory.
* **Global Accuracy:** The overall accuracy compared to the baseline.
* **Category Recovery Rate:** The specific percentage of `Discourse: Speaker Continuity` errors from EXP010 that are successfully flipped from incorrect to correct.
* **Degradation Check:** Ensure the new representation does not introduce false positives that degrade previously correct attributions.
* **Runtime / Memory:** Verify that the O(1) state tracker preserves the lightweight processing goals of the project.

## 5. Next Steps
Once this specification is approved, we will transition out of documentation and immediately begin the minimal codebase implementation and testing phase for EXP011.
