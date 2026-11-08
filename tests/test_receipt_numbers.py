"""Number parsing and sum check. Every input string is a real CORD line text."""

import pytest

from rulebook.tools.receipt import (
    amount_token,
    find_numbers,
    parse_amount,
    read_number,
    sum_check,
)


@pytest.mark.parametrize(
    ("text", "readings"),
    [
        ("10.500,00", {"dot": 10500.0}),  # train_0058, Indonesian with decimals
        ("8.000,00", {"dot": 8000.0}),
        ("44,000.00", {"comma": 44000.0}),  # train_0029, English with decimals
        ("CHANGE 23,500.00", {"comma": 23500.0}),
        ("Sub-Total 1,346,000", {"comma": 1346000.0}),  # train_0000, several groups
        ("Kembali: 0,00", {"dot": 0.0}),  # train_0141
        ("Rp 116000.00", {"comma": 116000.0}),  # train_0129, no thousands separator
        ("Total Qty 16", {"dot": 16.0, "comma": 16.0}),
        ("75,000", {"dot": 75.0, "comma": 75000.0}),  # the ambiguous shape
        ("TOTAL 43.636", {"dot": 43636.0, "comma": 43.636}),
    ],
)
def test_readings(text: str, readings: dict) -> None:
    (tok,) = find_numbers(text)
    assert tok.readings == readings


def test_only_lone_separator_plus_three_digits_is_ambiguous() -> None:
    assert find_numbers("75,000")[0].ambiguous
    assert find_numbers("KEMBALI 2.000")[0].ambiguous
    for clear in ("10.500,00", "44,000.00", "1,346,000", "0,00", "116000.00", "16"):
        assert not find_numbers(clear)[0].ambiguous, clear


def test_value_falls_back_to_the_only_valid_reading() -> None:
    (tok,) = find_numbers("10.500,00")
    assert tok.value("dot") == tok.value("comma") == 10500.0
    (amb,) = find_numbers("75,000")
    assert (amb.value("dot"), amb.value("comma")) == (75.0, 75000.0)


@pytest.mark.parametrize(
    ("text", "digits", "value_comma"),
    [
        ("TAX 10.00 % 4.364", "4.364", 4.364),  # train_0004: skip the percentage
        ("5,618 TAX 10.00 %", "5,618", 5618.0),  # train_0006: amount before the key
        ("DISC ITEM -30.000% -11,700", "11,700", -11700.0),  # train_0156: negative
        ("Tax (10.0%) Rp 1.500", "1.500", 1.5),  # train_0026: Rp prefix
        ("PB1 10% 10.000% AMOUNT 3,273", "3,273", 3273.0),  # train_0564
        ("Net Total: Rp 194,726", "194,726", 194726.0),  # dev
        ("Discount: -9,545", "9,545", -9545.0),  # dev
    ],
)
def test_amount_token(text: str, digits: str, value_comma: float) -> None:
    tok = amount_token(text)
    assert tok is not None and tok.digits == digits
    assert tok.value("comma") == value_comma


def test_percentages_and_text_without_numbers() -> None:
    assert [t.percent for t in find_numbers("TAX 10.00 % 4.364")] == [True, False]
    assert amount_token("Cheese Croquette") is None
    assert read_number("1,000,00", "dot") is None and read_number("1,000,00", "comma") is None


# Line texts from train_0616, labels assigned by hand for the test.
MIXED = [
    ("43.636", "menu.price"),
    ("8,000", "menu.price"),
    ("DPP 51,636", "sub_total.subtotal_price"),
    ("TAX 10.00 % 5,164", "sub_total.tax_price"),
    ("GRAND TOTAL 56.800", "total.total_price"),
]
CONSISTENT = [
    ("Nasi Goreng", "menu.nm"),
    ("43.636", "menu.price"),
    ("8.000", "menu.price"),
    ("SUBTOTAL 51.636", "sub_total.subtotal_price"),
    ("TAX 10.00 % 5.164", "sub_total.tax_price"),
    ("GRAND TOTAL 56.800", "total.total_price"),
]


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("TOTAL 43.636", 43636.0),  # lone separator + 3 digits = thousands
        ("75,000", 75000.0),
        ("KEMBALI 2.000", 2000.0),
        ("Total. 35.000,00", 35000.0),
        ("CHANGE 23,500.00", 23500.0),
        ("Sub-Total 1,346,000", 1346000.0),
        ("Rp 116000.00", 116000.0),
        ("Kembali: 0,00", 0.0),
        ("Discount: -9,545", -9545.0),
        ("TAX 10.00 % 4.364", 4364.0),
        ("Net Total: Rp 194,726", 194726.0),
        ("Cheese Croquette", None),
    ],
)
def test_parse_amount_money_rule(text: str, value: float | None) -> None:
    assert parse_amount(text) == value


def test_sum_check_passes_on_a_consistent_receipt() -> None:
    res = sum_check(CONSISTENT)
    assert res.passed and res.suspicious == []


def test_sum_check_passes_on_a_mixed_receipt() -> None:
    """train_0616 mixes `.` and `,` thousands; the per-token rule reads it correctly."""
    res = sum_check(MIXED)
    assert res.passed, res.checks


def test_sum_check_flags_numeric_lines_when_it_fails() -> None:
    wrong = [*CONSISTENT[:-1], ("GRAND TOTAL 66.800", "total.total_price")]  # off by 10,000
    res = sum_check(wrong)
    assert res.passed is False
    assert res.suspicious == [1, 2, 3, 4, 5]  # every line holding a number, not line 0


def test_sum_check_needs_item_lines() -> None:
    assert sum_check([("TOTAL 45,500", "total.total_price")]).passed is None
