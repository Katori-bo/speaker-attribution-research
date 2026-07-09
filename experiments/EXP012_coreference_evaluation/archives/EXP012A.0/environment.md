# EXP012A.0 Environment Documentation

## System
- **Operating System:** Linux
- **Python Version:** 3.14 (Virtual Environment)

## Package Versions
- **BookNLP:** 1.0.8
- **Torch:** 2.12.1
- **Transformers:** 5.12.1
- **spaCy:** 3.8.13 (en_core_web_sm model 3.8.0)

## Model Checkpoints (BookNLP 'small' architecture)
- `entities_google_bert_uncased_L-4_H-256_A-4-v1.0.model`
- `coref_google_bert_uncased_L-2_H-256_A-4-v1.0.model`
- `speaker_google_bert_uncased_L-8_H-256_A-4-v1.0.1.model`

## Compatibility Patches
- **`booknlp_pytorch_compat.patch`**: Due to updates in `transformers` and `torch` causing strict key mismatch exceptions during `load_state_dict()` (e.g., unexpected `bert.embeddings.position_ids`), the BookNLP tagging files (`entity_tagger.py`, `litbank_coref.py`, `bert_qa.py`) were patched to load model checkpoints with `strict=False`.
- *Note on Scientific Impact:* This is purely an engineering compatibility fix to allow older serialized models to load on newer PyTorch versions. It does not alter the underlying model weights, inference algorithms, or logical behavior.
