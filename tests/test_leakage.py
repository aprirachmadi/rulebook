"""Fail if anything outside `eval/` (and the data layer itself) can reach v2 labels."""

import ast
import copy
from pathlib import Path

import pytest

from rulebook.config import load_config
from rulebook.data import gold
from rulebook.data.cord import Line, Receipt, Word, load_split

ROOT = Path(__file__).resolve().parent.parent
LABEL_KEYS = {"category", "label", "labels", "group_id", "sub_group_id"}

# The only modules allowed to read raw ground truth.
RAW_GT_ALLOWED = {"rulebook/data/cord.py", "rulebook/data/gold.py"}
# The only modules allowed to call the gold-set accessor. The Verifier's tool joins this in M5.
GOLD_ALLOWED = {"rulebook/data/gold.py"}


def _modules(*dirs: str) -> list[Path]:
    return [p for d in dirs for p in (ROOT / d).rglob("*.py")]


def _names(path: Path) -> set[str]:
    """Every identifier, attribute, imported module and string literal in a file."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Name):
            found.add(node.id)
        elif isinstance(node, ast.Attribute):
            found.add(node.attr)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.split(".")[0])
            found.update(a.name for a in node.names)
        elif isinstance(node, ast.Import):
            found.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            found.add(node.value)
    return found


def test_agent_facing_models_have_no_label_fields() -> None:
    for model in (Word, Line, Receipt):
        assert not LABEL_KEYS & set(model.model_fields), model.__name__


def test_loaded_receipts_carry_no_labels() -> None:
    dumped = [r.model_dump() for r in load_split("validation")]

    def keys(obj: object) -> set[str]:
        if isinstance(obj, dict):
            return set(obj) | {k for v in obj.values() for k in keys(v)}
        if isinstance(obj, list):
            return {k for v in obj for k in keys(v)}
        return set()

    assert not LABEL_KEYS & keys(dumped)


def test_only_allowed_modules_touch_raw_ground_truth() -> None:
    for path in _modules("rulebook"):
        rel = path.relative_to(ROOT).as_posix()
        if rel in RAW_GT_ALLOWED:
            continue
        names = _names(path)
        assert "_raw_ground_truth" not in names, f"{rel} reads raw ground truth"
        assert "ground_truth" not in names, f"{rel} reads the ground_truth column"
        assert "category" not in names, f"{rel} reads line categories"


def test_rulebook_never_imports_eval() -> None:
    for path in _modules("rulebook"):
        names = _names(path)
        rel = path.relative_to(ROOT).as_posix()
        assert "eval" not in names, f"{rel} imports eval/"
        assert "load_answer_key" not in names, f"{rel} uses the answer key"


def test_detector_catches_a_leak(tmp_path: Path) -> None:
    leaky = tmp_path / "leaky_tool.py"
    leaky.write_text("from eval.answer_key import load_answer_key\nx = line['category']\n")
    names = _names(leaky)
    assert {"eval", "load_answer_key", "category"} <= names


def _config_with(**data_overrides: object) -> dict:
    cfg = copy.deepcopy(load_config())
    for key, value in data_overrides.items():
        if key == "gold_split":
            cfg["data"]["splits"]["gold"] = value
        else:
            cfg["data"][key] = value
    return cfg


def test_gold_labels_cover_only_the_gold_receipts() -> None:
    labels = gold.gold_labels()
    ids = set(load_config()["data"]["gold_receipt_ids"])
    assert len(ids) == load_config()["data"]["gold_receipt_count"]
    assert {rid for rid, _ in labels} == ids
    assert all(rid.startswith("validation_") for rid, _ in labels)


@pytest.mark.parametrize("split", ["train", "test"])
def test_gold_labels_refuse_other_splits_in_ids(monkeypatch, split: str) -> None:
    ids = [*load_config()["data"]["gold_receipt_ids"], f"{split}_0000"]
    monkeypatch.setattr(gold, "load_config", lambda: _config_with(gold_receipt_ids=ids))
    with pytest.raises(gold.LeakageError):
        gold.gold_labels()


@pytest.mark.parametrize("split", ["train", "test"])
def test_gold_labels_refuse_other_gold_split(monkeypatch, split: str) -> None:
    monkeypatch.setattr(gold, "load_config", lambda: _config_with(gold_split=split))
    with pytest.raises(gold.LeakageError):
        gold.gold_labels()


def test_gold_module_reads_only_the_gold_split() -> None:
    """Every raw-ground-truth call in gold.py must pass GOLD_SPLIT, never another split."""
    tree = ast.parse((ROOT / "rulebook/data/gold.py").read_text(encoding="utf-8"))
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "_raw_ground_truth"
    ]
    assert calls
    for call in calls:
        assert [getattr(a, "id", None) for a in call.args] == ["GOLD_SPLIT"]
    assert gold.GOLD_SPLIT == "validation"


def test_only_allowed_modules_use_gold_labels() -> None:
    for path in _modules("rulebook"):
        rel = path.relative_to(ROOT).as_posix()
        if rel not in GOLD_ALLOWED:
            assert "gold_labels" not in _names(path), f"{rel} calls gold_labels"
