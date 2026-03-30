"""
Strip class names from wikipedia_text to reduce name-bias in zero-shot experiments.

Replaces each class's primary name variants (class_name + wikipedia_title) with
[SPECIES], producing *_anon.jsonl files alongside the originals.

Usage:
    python data/strip_class_names.py                        # process both datasets
    python data/strip_class_names.py --birds_only           # birds only
    python data/strip_class_names.py --flowers_only         # flowers only

Then pass the anonymised files to training:
    python scripts/train.py --dataset cub \\
        --wikipedia_jsonl data/wikipedia/birds_anon.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PLACEHOLDER = "[SPECIES]"

# Root of the project (one level up from this file's directory)
_ROOT = Path(__file__).resolve().parent


def _build_pattern(name: str) -> re.Pattern:
    """Match *name* plus common English inflections (plural, possessive).

    Examples matched for "black-footed albatross":
      black-footed albatross
      Black-footed Albatross
      black-footed albatrosses
      black-footed albatross's
    """
    escaped = re.escape(name.strip())
    return re.compile(r"\b" + escaped + r"(?:e?s|'s)?\b", re.IGNORECASE)


def _anonymize(text: str, patterns: list[re.Pattern]) -> str:
    for pat in patterns:
        text = pat.sub(PLACEHOLDER, text)
    return text


def process_jsonl(src: Path, dst: Path) -> None:
    entries: list[dict] = []
    with open(src, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                # Fix leading-zero integers (e.g. "idx": 001) — same fix as dataset.py
                fixed = re.sub(r':\s*0+(\d+)', r': \1', line)
                entries.append(json.loads(fixed))

    n_replaced = 0
    with open(dst, "w", encoding="utf-8") as f:
        for entry in entries:
            class_name  = entry.get("class_name", "") or ""
            wiki_title  = entry.get("wikipedia_title", "") or ""
            original    = entry.get("wikipedia_text", "") or ""

            # Deduplicate: e.g. birds have class_name ≈ wikipedia_title
            names = {n.strip() for n in (class_name, wiki_title) if n.strip()}
            patterns = [_build_pattern(n) for n in names]

            new_text = _anonymize(original, patterns)
            n_replaced += sum(len(p.findall(original)) for p in patterns)

            f.write(json.dumps({**entry, "wikipedia_text": new_text}, ensure_ascii=False) + "\n")

    print(f"  {src.name} → {dst.name}: {len(entries)} classes, {n_replaced} replacements")


def _pair(stem: str, wikipedia_dir: Path) -> tuple[Path, Path]:
    src = wikipedia_dir / f"{stem}.jsonl"
    dst = wikipedia_dir / f"{stem}_anon.jsonl"
    return src, dst


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wikipedia_dir", default=str(_ROOT / "wikipedia"),
                        help="Directory containing *.jsonl files (default: data/wikipedia)")
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--birds_only",   action="store_true")
    grp.add_argument("--flowers_only", action="store_true")
    args = parser.parse_args()

    wiki_dir = Path(args.wikipedia_dir)

    targets = []
    if not args.flowers_only:
        targets.append("birds")
    if not args.birds_only:
        targets.append("flowers")

    for stem in targets:
        src, dst = _pair(stem, wiki_dir)
        if not src.exists():
            print(f"  Skipping {src} (not found)")
            continue
        print(f"Processing {stem}…")
        process_jsonl(src, dst)

    print("Done.")


if __name__ == "__main__":
    main()
