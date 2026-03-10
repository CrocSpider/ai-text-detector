from __future__ import annotations

import json
import random
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

from trainer.text_utils import normalize_text


@dataclass(slots=True)
class SplitRatio:
    train: float = 0.8
    validation: float = 0.1
    test: float = 0.1


def prepare_hc3_dataset(
    *,
    output_dir: str | Path,
    seed: int = 42,
    subset: str = "all",
    max_per_class: int | None = None,
    split_ratio: SplitRatio | None = None,
) -> dict[str, int | str]:
    load_dataset = getattr(import_module("datasets"), "load_dataset")

    ratios = split_ratio or SplitRatio()
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("Hello-SimpleAI/HC3", subset, split="train")
    human_records: list[dict[str, object]] = []
    ai_records: list[dict[str, object]] = []

    for row in dataset:
        source = str(row.get("source", subset) or subset)
        question = normalize_text(str(row.get("question", "") or ""))
        base_metadata = {
            "language": "en",
            "domain": source,
            "writer_profile": "unknown",
            "attack_type": "none",
            "source_dataset": "HC3",
            "extraction_quality": "good",
            "question": question,
        }

        for index, answer in enumerate(row.get("human_answers", []) or []):
            text = normalize_text(str(answer or ""))
            if len(text) < 80:
                continue
            human_records.append(
                {
                    "document_id": f"hc3-{subset}-{row.get('id', 'row')}-human-{index}",
                    "text": text,
                    "label": 0,
                    **base_metadata,
                }
            )

        for index, answer in enumerate(row.get("chatgpt_answers", []) or []):
            text = normalize_text(str(answer or ""))
            if len(text) < 80:
                continue
            ai_records.append(
                {
                    "document_id": f"hc3-{subset}-{row.get('id', 'row')}-ai-{index}",
                    "text": text,
                    "label": 1,
                    **base_metadata,
                }
            )

    rng = random.Random(seed)
    rng.shuffle(human_records)
    rng.shuffle(ai_records)

    if max_per_class is not None:
        human_records = human_records[:max_per_class]
        ai_records = ai_records[:max_per_class]

    train_rows, validation_rows, test_rows = _balanced_split(
        human_records=human_records,
        ai_records=ai_records,
        ratios=ratios,
    )

    _write_jsonl(output_path / "train.jsonl", train_rows)
    _write_jsonl(output_path / "validation.jsonl", validation_rows)
    _write_jsonl(output_path / "test.jsonl", test_rows)

    return {
        "dataset": "HC3",
        "subset": subset,
        "output_dir": str(output_path),
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "test_rows": len(test_rows),
        "human_rows": len(human_records),
        "ai_rows": len(ai_records),
    }


def _balanced_split(
    *,
    human_records: list[dict[str, object]],
    ai_records: list[dict[str, object]],
    ratios: SplitRatio,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    train_h, val_h, test_h = _split_one_class(human_records, ratios)
    train_ai, val_ai, test_ai = _split_one_class(ai_records, ratios)

    train_rows = train_h + train_ai
    validation_rows = val_h + val_ai
    test_rows = test_h + test_ai

    random.Random(7).shuffle(train_rows)
    random.Random(8).shuffle(validation_rows)
    random.Random(9).shuffle(test_rows)
    return train_rows, validation_rows, test_rows


def _split_one_class(
    rows: list[dict[str, object]], ratios: SplitRatio
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    total = len(rows)
    train_end = int(total * ratios.train)
    validation_end = train_end + int(total * ratios.validation)
    return rows[:train_end], rows[train_end:validation_end], rows[validation_end:]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
