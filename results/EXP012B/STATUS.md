# EXP012B Status

## Decision: ACCEPT
Coreference representation is accepted as a permanent part of the pipeline architecture.

## Reason
- Positive global improvement (+0.52%) across the dataset.
- No architectural regressions.
- The majority of failures (87.6%) are no longer mapping failures (Type A), meaning the coreference capability adds robust semantic representation when entities align.

## Limitation
Performance depends heavily on entity alignment quality. When BookNLP mapping succeeds (e.g., Daisy Miller, Pride and Prejudice), performance gains are strong (+1-2%). When mapping fails (e.g., Alice in Wonderland), the fallback logic results in a negative impact on performance.
