# EXP013A.1 Speaker-Addressee Representation

## Purpose
To explicitly represent the transient speaker-addressee state during dialogue sequences, allowing the speaker attribution model to predict next-turn speakers based on conversational flow rather than simple proximity.

The representation aims to answer: *Who is likely participating in the current conversation?* It does **not** attempt to build a global relationship graph or track complete character importance.

## Inputs
- **BookNLP Tokens**: Used to extract syntactical dependencies (e.g. `npadvmod` or `prep` -> `pobj`).
- **BookNLP Entities**: To identify character bounds.
- **BookNLP Quotes**: To identify dialogue bounds.
- **BookNLP Aliases**: To resolve mentioned characters to underlying character IDs.
- **Speech Constructions / Detected Vocatives**: For explicitly parsing attribution signals.

*(Note: PDNC gold addressee annotations are strictly forbidden for populating this representation.)*

## Outputs
A lightweight `DialogueInteraction` event and a running `InteractionState` that keeps track of recent transitions.

## State Variables
The representation consists of two main objects (defined in `src/addressee/schemas.py`):

1. **`DialogueInteraction`**
   - `quote_id`: Unique identifier for the quote.
   - `speaker_id`: Resolved ID of the speaker.
   - `addressee_id`: Resolved ID of the addressee (or `None`).
   - `confidence`: Confidence score of the extraction (0 for unknown).
   - `extraction_method`: Enum defining how it was derived (`VOCATIVE`, `SPEECH_TAG_OBJECT`, or `UNKNOWN`).

2. **`InteractionState`**
   - `last_interaction`: The most recent `DialogueInteraction` object.
   - `recent_addressees`: A list of character IDs recently addressed.
   - `speaker_transition_history`: A list recording the transitions (e.g., A → B).

## Update Rules
- When a new quote is processed, a `DialogueInteraction` object is generated.
- If an addressee is reliably extracted via `VOCATIVE` or `SPEECH_TAG_OBJECT`, the `addressee_id` is populated, and `confidence` is set appropriately.
- If no reliable extraction occurs, the `addressee_id` defaults to `None`, `confidence` to `0`, and `extraction_method` to `UNKNOWN`.
- The `InteractionState` is then updated:
  - `last_interaction` is updated to the new interaction.
  - If `addressee_id` is present, it is added/bumped in `recent_addressees`.
  - The transition from the previous speaker to the current speaker (if known) is logged in `speaker_transition_history`.

## Failure Cases / Unknown Handling
Because automatic extraction is sparse, most quotes will lack explicit addressee signals.
- **Handling Strategy:** Most extraction failures should **not** become wrong information. 
- **Default State:** Use `addressee_id = None` and `confidence = 0`.
- **Reasoning:** Do not guess the addressee if explicit signals are missing. Sparse but accurate signals are strictly better than dense, noisy signals. The downstream model will rely on `recent_addressees` or past transitions when the current quote's addressee is `UNKNOWN`.
