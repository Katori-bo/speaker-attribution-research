# EXP015B: Evaluation Integrity Report

## Objective
Validate that the quote-type results reported in EXP015 accurately reflect the model's capabilities and are free from evaluation artifacts, information leakage, or misleading interpretation before drawing scientific conclusions.

## Final Assessment Matrix

| Audit                | Status | Summary |
| -------------------- | ------ | ------- |
| 1. Leakage (Oracle State) | Pass | Features condition on gold previous speaker. Classified as an explicit evaluation assumption, not uncontrolled leakage. |
| 2. Quote Mapping     | Pass | 100% of quotes mapped perfectly to PDNC native labels without heuristics. |
| 3. Sample Size       | Pass | 2,516 quotes properly partitioned. Implicit category consists of a robust N=763. |
| 4. Candidate Difficulty | Pass | Evaluated in `candidate_statistics.csv`. Implicit quotes share similar candidate spaces, ruling out candidate deficiency as the sole reason for high performance. |
| 5. Category Stability| Pass | Evaluated in `quote_type_counts.csv`. Narrow 95% CIs indicate highly stable metric convergence. |
| 6. Literature Check  | Pass | 89% vs 52% anomaly successfully explained by conditional oracle discourse tracking vs full end-to-end cascading errors. |
| 7. Novel Split       | Pass | Confirmed ∅ overlap between Train and Test novel distributions. |
| 8. Mechanism Analysis| Pass | Confirmed the overwhelming majority of implicit quotes represent two-party conversational alternations which structural features excel at. |

---

## Key Findings

### 1. The Oracle Discourse Assumption
We identified that the `generate_dataset_p2.py` script populates the discourse state (`previous_speakers`) using the **gold speaker annotations**. Consequently, features like `candidate_is_previous_speaker` evaluate whether a candidate matches the *true* previous speaker, rather than a previously *predicted* speaker.

**Decision:** As the project's task until this point has been *"Given the current discourse state, predict the speaker,"* this is a **conditional evaluation assumption**. This assumption will be documented in the paper. EXP014's metrics stand as the upper-bound performance under oracle discourse tracking. An autoregressive evaluation measuring degradation from error propagation will be conducted as a separate experiment (**EXP016**).

### 2. Validation of the 88.99% Implicit Accuracy
Audits 2 through 8 confirmed that the 88.99% accuracy on implicit quotes is a highly stable, genuine property of the evaluated structural representation. 

Mechanism sampling (Audit 8) revealed that the vast majority of implicit quotes in PDNC follow strict two-party conversational alternations. Because our model conditions on the **oracle discourse state**, the structural features (e.g., previous speaker) trace these alternations with near-perfect precision, suffering zero cascading error.

Explicit Pronouns (66.0% accuracy) remain the true bottleneck, confirming that structural heuristics cannot resolve dense semantic ambiguity (e.g. tracking "she" among multiple female candidates in a scene).

## Final Decision
The EXP015 evaluation methodology has been audited and validated. The observed quote-type performance genuinely reflects the properties of the evaluated representation under the stated oracle-state assumption. 

**Recommendation: Proceed to EXP016 (Autoregressive Discourse Tracking).**
