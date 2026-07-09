# ADR-005

## Title
Freeze Explicit Conversation-State Architecture

## Decision
No further expansion of explicit conversation-state representations will be pursued unless new experimental evidence demonstrates that they introduce novel contextual information capable of recovering the dominant residual failure categories.

## Reason
EXP011 demonstrated diminishing returns for explicit discourse-state enrichment. The performance plateau indicates that adding richer explicit conversation-state representations (Active Conversation ID, Participant Stack, and Interruption Distance) does not meaningfully resolve the dominant residual error categories (such as coreference and alias resolution). 

To prevent feature creep and maintain the project's focus on minimal, effective representations, the explicit discourse-state architecture is hereby frozen. Future work should investigate minimal semantic representations instead.
