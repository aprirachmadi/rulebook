from rulebook.data.cord import label_space, load_image, load_split, parse_receipt


def _quad(x1: int, y1: int, x2: int, y2: int) -> dict[str, int]:
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y1, "x3": x2, "y3": y2, "x4": x1, "y4": y2}


def _word(text: str, x: int, y: int, row: int) -> dict:
    return {"text": text, "quad": _quad(x, y, x + 10, y + 10), "row_id": row, "is_key": 0}


GT = {
    "meta": {"image_id": 7, "image_size": {"width": 100, "height": 200}},
    "valid_line": [
        {
            "category": "menu.nm",
            "group_id": 1,
            "sub_group_id": 0,
            # out of order: second row first, and words within the first row swapped
            "words": [_word("Trsi", 60, 50, 2), _word("Kgkung", 40, 10, 1), _word("Mie", 5, 12, 1)],
        }
    ],
}


def test_parse_receipt_reading_order_and_box() -> None:
    receipt = parse_receipt("train", GT)
    assert receipt.receipt_id == "train_0007"
    (line,) = receipt.lines
    assert line.text == "Mie Kgkung Trsi"
    assert line.box == (5, 10, 70, 60)
    assert line.line_idx == 0


def test_validation_split_loads() -> None:
    receipts = load_split("validation")
    assert len(receipts) == 100
    assert sum(len(r.lines) for r in receipts) == 1219
    # dataset index == image_id, which `load_image` relies on
    assert [r.image_id for r in receipts] == list(range(100))
    assert all(ln.text and ln.words for r in receipts for ln in r.lines)


def test_load_image_matches_meta() -> None:
    receipt = load_split("validation")[3]
    assert load_image("validation", 3).size == (receipt.width, receipt.height)


def test_label_space_derived_from_data() -> None:
    space = label_space()
    assert len(space) == 29
    assert "menu.nm" in space and "total.total_price" in space
