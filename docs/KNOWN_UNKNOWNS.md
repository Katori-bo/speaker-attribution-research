# Known Unknowns

Track unanswered questions.

## Unknown 1: Anaphoric Reference Resolution
- **Question:** How reliably can a lightweight ML model (or feature set) resolve anaphoric pronouns ("he said") to the correct speaker candidate?
- **Why does it matter?** Anaphoric quotes account for a large portion of errors, scoring only 49.67% accuracy in Phase 1. 
- **Current understanding:** Deterministic substring matching fails entirely on pronouns. Coreference resolution is needed.
- **Planned experiment:** Evaluate Phase 2 ranking model explicitly on the Anaphoric quote slice (Hypothesis 1).
- **Status:** Open.

## Unknown 2: Distinguishing Dialogue Alternation vs Speaker Continuation
- **Question:** What semantic or discourse features most reliably distinguish an A-B-A-B turn from an A-A monologue continuation?
- **Why does it matter?** This conflict caused 31,605 rule collisions in Phase 1. Without resolving this, accuracy is capped at ~76%.
- **Current understanding:** The presence of a question mark often prompts a reply (A-B), while narrative paragraphs between quotes might imply A-A. 
- **Planned experiment:** Ablate discourse history features in Phase 2 ranking models (Hypothesis 2).
- **Status:** Open.

## Unknown 3: Candidate Ceiling Limitations
- **Question:** How do we recover the 7.24% of quotes where the true speaker is not in the preceding 15 characters?
- **Why does it matter?** No downstream model can attribute a quote if the generator misses the candidate.
- **Current understanding:** Pushing Window=30 balloons the candidate set and hurts precision.
- **Planned experiment:** None currently scheduled. We will first push ranking accuracy as high as possible on the 92.76% of solvable quotes.
- **Status:** Deferred (Post-Phase 2).
