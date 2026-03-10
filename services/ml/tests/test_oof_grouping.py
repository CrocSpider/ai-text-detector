from __future__ import annotations

from types import SimpleNamespace

import pytest

import trainer.classifier as classifier_module
import trainer.stylometry as stylometry_module
from trainer.config import ModelConfig, RunConfig, StylometryConfig
from trainer.datasets import SegmentRecord


class CaptureGroups(Exception):
    pass


def build_segments() -> list[SegmentRecord]:
    return [
        SegmentRecord(
            segment_id="doc-1::seg::0",
            document_id="doc-1",
            text="Human paragraph one.",
            label=0,
            segment_index=0,
            metadata={},
        ),
        SegmentRecord(
            segment_id="doc-1::seg::1",
            document_id="doc-1",
            text="Human paragraph two.",
            label=0,
            segment_index=1,
            metadata={},
        ),
        SegmentRecord(
            segment_id="doc-2::seg::0",
            document_id="doc-2",
            text="AI paragraph one.",
            label=1,
            segment_index=0,
            metadata={},
        ),
        SegmentRecord(
            segment_id="doc-2::seg::1",
            document_id="doc-2",
            text="AI paragraph two.",
            label=1,
            segment_index=1,
            metadata={},
        ),
    ]


def test_classifier_oof_groups_by_document(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    captured: dict[str, list[str]] = {}

    class FakeTrainingArguments:
        def __init__(self, eval_strategy: str | None = None, **_kwargs) -> None:
            self.eval_strategy = eval_strategy

    class FakeGroupKFold:
        def __init__(self, n_splits: int, shuffle: bool, random_state: int) -> None:
            self.n_splits = n_splits
            self.shuffle = shuffle
            self.random_state = random_state

        def split(self, _x, _y, groups):
            captured["groups"] = list(groups)
            raise CaptureGroups

    def fake_import_module(name: str):
        if name == "datasets":
            return SimpleNamespace(Dataset=object)
        if name == "transformers":
            return SimpleNamespace(
                AutoModelForSequenceClassification=object,
                AutoTokenizer=object,
                DataCollatorWithPadding=object,
                Trainer=object,
                TrainingArguments=FakeTrainingArguments,
                set_seed=lambda *_args, **_kwargs: None,
            )
        if name == "sklearn.model_selection":
            return SimpleNamespace(StratifiedGroupKFold=FakeGroupKFold)
        raise AssertionError(f"Unexpected import: {name}")

    monkeypatch.setattr(classifier_module, "import_module", fake_import_module)

    with pytest.raises(CaptureGroups):
        classifier_module.generate_oof_probabilities(
            train_segments=build_segments(),
            model_config=ModelConfig(),
            run_config=RunConfig(output_dir=str(tmp_path)),
            n_folds=2,
            output_dir=tmp_path,
        )

    assert captured["groups"] == ["doc-1", "doc-1", "doc-2", "doc-2"]


def test_stylometry_oof_groups_by_document(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, list[str]] = {}

    class FakeGroupKFold:
        def __init__(self, n_splits: int, shuffle: bool, random_state: int) -> None:
            self.n_splits = n_splits
            self.shuffle = shuffle
            self.random_state = random_state

        def split(self, _x, _y, groups):
            captured["groups"] = list(groups)
            raise CaptureGroups

    def fake_import_module(name: str):
        if name == "numpy":
            return SimpleNamespace(asarray=lambda values, dtype=None: values, int32=int, float64=float)
        if name == "sklearn.model_selection":
            return SimpleNamespace(StratifiedGroupKFold=FakeGroupKFold)
        if name == "lightgbm":
            return SimpleNamespace(LGBMClassifier=object)
        raise AssertionError(f"Unexpected import: {name}")

    monkeypatch.setattr(stylometry_module, "import_module", fake_import_module)
    monkeypatch.setattr(stylometry_module, "_build_matrix", lambda *_args, **_kwargs: [[0.0], [0.1], [0.9], [1.0]])

    with pytest.raises(CaptureGroups):
        stylometry_module.generate_oof_probabilities(
            train_segments=build_segments(),
            feature_rows={},
            stylometry_config=StylometryConfig(),
            n_folds=2,
            seed=42,
        )

    assert captured["groups"] == ["doc-1", "doc-1", "doc-2", "doc-2"]
