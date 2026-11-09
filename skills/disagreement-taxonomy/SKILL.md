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
The guideline has a rule for this case, but the text itself is unclear: a truncated or
garbled line, or a word that a careful human could not place even with the rule and the
neighboring lines. AMBIGUOUS is about the text, never about the guideline.
Not AMBIGUOUS: "it is unclear whether X should be label A or B" when no rule says which.
That is a GUIDELINE_GAP.

## GUIDELINE_GAP
No guideline rule covers the case, or two rules conflict. This is the only class that leads
to a new rule. Typical signs: the labelers spread over several labels with low confidence,
or each label is defensible under the current definitions, or the guideline is silent on a
situation the text shows clearly (e.g. a keyword like "TOTAL" or "CASH" on its own line
with its amount on the next line).

## How to decide
1. Look at the cluster (get_cluster) and a few receipts' full lines (get_votes). Each item
   shows the lines above and below it: OCR often puts a keyword and its amount on
   separate lines, so read them before calling a line "standalone".
2. Check the OCR evidence first (get_ocr_conf, sum_check). If the text is wrong: OCR_ERROR.
3. Read the guideline (cord-guideline skill). Ask: is there a rule that says which label
   this text gets? If yes and most labelers followed it: MODEL_ERROR. If yes but the text
   itself is unreadable or cut off: AMBIGUOUS.
4. If no rule says which label it gets: GUIDELINE_GAP, even when the case looks hard.
Never calculate numbers yourself; sum_check does it.
