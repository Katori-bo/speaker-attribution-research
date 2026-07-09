# Audit 6: Literature Comparison

## Objective
Identify unusually large improvements requiring additional explanation by comparing EXP014's performance against BookNLP and expected literature baselines.

## Comparison Table

| System | Explicit Named | Explicit Pronoun | Implicit |
| ------ | -------------: | ---------------: | -------: |
| BookNLP|          92.8% |            77.4% |    52.2% |
| EXP014 |          84.2% |            66.0% |    89.0% |

## Interpretation
**Anomaly Detected in Implicit Quotes:**
BookNLP's implicit performance is remarkably low (52.2%), whereas EXP014's is remarkably high (89.0%). 
This vast discrepancy exists because BookNLP is an open-domain pipeline that relies strictly on syntactically identifiable speaker cues and coreference models, whereas EXP014 utilizes strong, structural discourse heuristics (like `candidate_is_previous_speaker`). 

Since implicit quotes in PDNC predominantly consist of back-and-forth conversational alternations, structural heuristics excel. However, we must also note that EXP014's evaluation was conditioned on the **gold discourse state** (Oracle Previous Speaker), which perfectly traces these alternations without error propagation. End-to-end literature models (like BookNLP) suffer from cascading errors when tracking long conversational chains, which explains why the literature/baseline performance drops to ~52%, while our oracle-conditioned discourse features inflate it to ~89%.

**Explicit Pronouns:**
EXP014 (66%) significantly trails BookNLP (77%) in explicit pronouns. BookNLP uses a highly tuned neural coreference resolution system explicitly trained for pronoun resolution in literature, whereas EXP014 relies on lightweight structural approximations (recency, previous speaker, entity alias matching). This underperformance is expected and confirms that structural representation cannot fully substitute deep semantic coreference parsing.
