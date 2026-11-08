"""Receipt numbers: parsing and the sum check. Models never parse prices; this module does.

A number token can be read two ways:
- "dot": `.` groups thousands, `,` is the decimal mark (Indonesian style: 2.000,00)
- "comma": `,` groups thousands, `.` is the decimal mark (English style: 2,000.00)
Most shapes are valid in only one reading, or give the same value in both. The one shape
that differs is a single separator followed by exactly 3 digits (`2.000`, `75,000`):
2000 or 2.0, 75000 or 75. Those tokens are `ambiguous`.

Money rule (approved 2026-11-08): money is never printed with 3 decimals, so for amounts a lone
separator followed by exactly 3 digits is a thousands separator, whichever symbol it is.
This is decided per token, not per receipt: 88/800 train receipts mix both styles.
A receipt whose numbers then fail the sum check gets its numeric lines flagged, never guessed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from rulebook.config import load_config

Reading = Literal["dot", "comma"]
READINGS: tuple[Reading, ...] = ("dot", "comma")

_TOKEN = re.compile(r"(?P<neg>-\s?)?(?:(?:Rp|IDR)\.?\s*)?(?P<num>\d[\d.,]*)(?P<pct>\s?%)?")
_SHAPES: dict[Reading, re.Pattern[str]] = {
    "dot": re.compile(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:,\d+)?"),
    "comma": re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"),
}


def read_number(digits: str, reading: Reading) -> float | None:
    """Value of `digits` under one reading, or None if the shape is invalid for it."""
    if not _SHAPES[reading].fullmatch(digits):
        return None
    thousands, decimal = (".", ",") if reading == "dot" else (",", ".")
    return float(digits.replace(thousands, "").replace(decimal, "."))


@dataclass
class NumberToken:
    raw: str  # as written, e.g. "-9,545" or "Rp 194,726"
    digits: str  # the digit part, e.g. "9,545"
    negative: bool
    percent: bool
    readings: dict[Reading, float] = field(default_factory=dict)  # valid readings only

    @property
    def ambiguous(self) -> bool:
        return len(set(self.readings.values())) > 1

    def value(self, reading: Reading) -> float | None:
        """Signed value under `reading`. Falls back to the only valid reading."""
        v = self.readings.get(reading)
        if v is None and len(set(self.readings.values())) == 1:
            v = next(iter(self.readings.values()))
        if v is None:
            return None
        return -v if self.negative else v


def money_value(tok: NumberToken) -> float | None:
    """Signed amount under the money rule."""
    if tok.ambiguous:  # lone separator + exactly 3 digits -> thousands
        return tok.value("dot" if "." in tok.digits else "comma")
    return tok.value("dot")  # unambiguous: every valid reading agrees


def parse_amount(text: str) -> float | None:
    """The money amount on a line, as a number (None if the line holds no amount)."""
    tok = amount_token(text)
    return money_value(tok) if tok else None


def find_numbers(text: str) -> list[NumberToken]:
    tokens = []
    for m in _TOKEN.finditer(text):
        digits = m.group("num").rstrip(".,")
        readings = {r: v for r in READINGS if (v := read_number(digits, r)) is not None}
        if readings:
            tokens.append(
                NumberToken(
                    raw=m.group(0).strip(),
                    digits=digits,
                    negative=bool(m.group("neg")),
                    percent=bool(m.group("pct")),
                    readings=readings,
                )
            )
    return tokens


def amount_token(text: str) -> NumberToken | None:
    """The money amount on a line: the last number that is not a percentage."""
    candidates = [t for t in find_numbers(text) if not t.percent]
    return candidates[-1] if candidates else None


# --- sum check ---

# How each label takes part in: items - item discounts = subtotal;
# subtotal - bill discounts + tax + service = total.
ROLE_OF_LABEL = {
    "menu.price": "item",
    "menu.discountprice": "item_discount",
    "sub_total.subtotal_price": "subtotal",
    "sub_total.discount_price": "bill_discount",
    "sub_total.tax_price": "charge",
    "sub_total.service_price": "charge",
    "sub_total.othersvc_price": "charge",
    "total.total_price": "total",
}


@dataclass
class SumCheck:
    passed: bool | None  # None: not enough lines to check
    checks: list[str] = field(default_factory=list)  # human-readable equations
    suspicious: list[int] = field(default_factory=list)  # line indexes holding numbers


def sum_check(lines: list[tuple[str, str]], tolerance: float | None = None) -> SumCheck:
    """Check a receipt's arithmetic, with amounts parsed by the money rule.

    `lines` = (text, label) per line; labels come from labelers, not from the answer key.
    """
    tol = load_config()["sum_check"]["tolerance"] if tolerance is None else tolerance
    by_role: dict[str, list[tuple[int, float]]] = {}
    for idx, (text, label) in enumerate(lines):
        role = ROLE_OF_LABEL.get(label)
        v = parse_amount(text) if role else None
        if role and v is not None:
            by_role.setdefault(role, []).append((idx, abs(v)))

    def total(role: str) -> float:
        return sum(v for _, v in by_role.get(role, []))

    def close(a: float, b: float) -> bool:
        return abs(a - b) <= tol * max(abs(b), 1.0)

    result = SumCheck(passed=None)
    if not by_role.get("item"):
        return result
    goods = total("item") - total("item_discount")
    results = []
    if by_role.get("subtotal"):
        sub = by_role["subtotal"][-1][1]
        ok = close(goods, sub)
        results.append(ok)
        result.checks.append(f"items {goods:g} vs subtotal {sub:g}: {'ok' if ok else 'FAIL'}")
        base = sub
    else:
        base = goods
    if by_role.get("total"):
        tot = by_role["total"][-1][1]
        expect = base - total("bill_discount") + total("charge")
        ok = close(expect, tot)
        results.append(ok)
        verdict = "ok" if ok else "FAIL"
        result.checks.append(f"expected total {expect:g} vs total {tot:g}: {verdict}")
    if results:
        result.passed = all(results)
        if not result.passed:
            result.suspicious = sorted(i for role in by_role for i, _ in by_role[role])
    return result
