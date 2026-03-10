# Product Specification

## Problem statement

Reviewers need triage help for text that may be machine-generated or heavily machine-edited, but a detector must communicate uncertainty and avoid overclaiming.

## MVP scope

- Paste text or upload `PDF`, `DOCX`, `TXT`
- Extract paragraphs and analyze segments
- Return risk score, confidence, rationale, and recommendation
- Show segment highlights and visible limitations

## Non-goals

- proving authorship
- plagiarism detection
- punitive workflow automation
- model-family attribution

## UX principles

- simple layout, low visual noise
- neutral language focused on risk and uncertainty
- clear warnings and human review guidance
