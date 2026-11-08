"""Write `reports/cord_v2_summary.md`: receipts and lines per split, label space, label counts."""

from __future__ import annotations

from collections import Counter

from eval.answer_key import load_answer_key
from rulebook.config import ROOT, load_config
from rulebook.data.cord import label_space, load_split


def build_summary() -> str:
    splits = load_config()["data"]["splits"]
    counts: dict[str, Counter[str]] = {}
    out = ["# CORD v2 data summary", "", f"Dataset: `{load_config()['data']['dataset']}`", ""]
    out += ["| role | split | receipts | lines | words |", "|---|---|---|---|---|"]
    for role, split in splits.items():
        receipts = load_split(split)
        lines = [ln for r in receipts for ln in r.lines]
        out.append(
            f"| {role} | {split} | {len(receipts)} | {len(lines)} "
            f"| {sum(len(ln.words) for ln in lines)} |"
        )
        counts[split] = Counter(load_answer_key(split).values())

    space = label_space()
    out += ["", f"## Label space ({len(space)} categories, derived from the work split)", ""]
    header = " | ".join(splits.values())
    out += [f"| category | {header} |", "|---|" + "---|" * len(splits)]
    for cat in space:
        cells = " | ".join(str(counts[s][cat]) for s in splits.values())
        out.append(f"| `{cat}` | {cells} |")
    unseen = sorted(set().union(*counts.values()) - set(space))
    out += ["", f"Categories outside the label space: {unseen or 'none'}", ""]
    return "\n".join(out)


def main() -> None:
    path = ROOT / "reports" / "cord_v2_summary.md"
    path.write_text(build_summary(), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
