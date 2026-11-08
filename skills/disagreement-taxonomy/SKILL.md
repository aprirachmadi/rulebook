---
name: disagreement-taxonomy
description: The four reasons labelers disagree (OCR_ERROR, MODEL_ERROR, AMBIGUOUS, GUIDELINE_GAP) and how to tell them apart. Load before diagnosing.
---
# Disagreement taxonomy

Classify a cluster of disagreements into exactly one class.

## OCR_ERROR
The text itself is wrong, so the labelers could not agree. Signs:
- OCR confidence below 0.6 (get_ocr_conf).
- Digit/letter swaps in the text: O and 0, l and 1, S and 5 (e.g. "SUBT0TAL", "C1" for "2").
- The receipt sum check fails (sum_check) and the disputed words hold a number.

## MODEL_ERROR
Two of three labelers agree, the guideline clearly supports the majority, and the
dissenter's label or reason contradicts the guideline. The guideline is fine; one model slipped.

## AMBIGUOUS
The guideline covers the case, but the input itself is unclear, e.g. a truncated line or a
word that could belong to two fields even for a careful human.

## GUIDELINE_GAP
No guideline rule covers the case, or two rules conflict. This is the only class that leads
to a new rule. Typical signs: the labelers spread over several labels with low confidence,
or each label is defensible under the current definitions.

## How to decide
1. Look at the cluster (get_cluster) and a few receipts' full lines (get_votes).
2. Check the OCR evidence first (get_ocr_conf, sum_check). If the text is wrong: OCR_ERROR.
3. Read the guideline (cord-guideline skill). If it clearly settles the case and most
   labelers followed it: MODEL_ERROR. If it settles it but the input is unclear: AMBIGUOUS.
4. Otherwise: GUIDELINE_GAP.
Never calculate numbers yourself; sum_check does it.
