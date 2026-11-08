# Run 20261108-153641-ccfab6: labeling vs v2 (gold text, guideline v1)

Receipts: 20. Words: 644. Matched to a CORD word: 644 (100.0%); OCR misses: 0.

**Word-level Krippendorff's alpha (3 labelers, nominal): 0.714**
Agreement pattern: {'all_agree': 429, 'two_agree': 178, 'all_differ': 37, 'incomplete': 0}

| labeler | model | complete receipts | words labeled (of all) | agreement with v2 (excl. other) | `other` used (matched / OCR-miss words) |
|---|---|---|---|---|---|
| labeler_a | qwen3.5:4b-8k | 20/20 | 644 / 644 | 87.2% (525 words) | 18.5% (119 / 0) |
| labeler_b | llama3.1:8b-8k | 20/20 | 644 / 644 | 85.7% (622 words) | 3.4% (22 / 0) |
| labeler_c | gemma4:12b-8k | 20/20 | 644 / 644 | 92.5% (626 words) | 2.8% (18 / 0) |

Majority vote (2+ agree, excl. other) vs v2: 95.2% of 583 words.
All three agree (excl. other): 425 words, of which 97.9% match v2.

Top 10 disagreements with v2 (v2 label -> labeler label, count over all labelers):

- menu.unitprice -> menu.price: 21
- menu.cnt -> menu.nm: 14
- menu.cnt -> menu.num: 13
- menu.nm -> menu.etc: 12
- total.total_price -> sub_total.subtotal_price: 11
- menu.sub.nm -> menu.etc: 9
- menu.sub.nm -> menu.nm: 8
- sub_total.service_price -> sub_total.tax_price: 8
- sub_total.subtotal_price -> total.total_price: 7
- total.menuqty_cnt -> total.menutype_cnt: 7

Labeler run stats:

- labeler_a: calls 33 (targeted repairs 5), words failed 0, 140s, 65008 tokens
- labeler_b: calls 46 (targeted repairs 18), words failed 0, 219s, 79396 tokens
- labeler_c: calls 29 (targeted repairs 1), words failed 0, 140s, 61954 tokens
