from __future__ import annotations

import json
import logging
import math
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.services.features import SegmentFeatureSet, stylometric_feature_vector


logger = logging.getLogger(__name__)


class ArtifactBundle:
    def __init__(self, artifact_dir: Path, manifest: dict[str, Any], device_preference: str) -> None:
        self.artifact_dir = artifact_dir
        self.manifest = manifest
        self.device_preference = device_preference
        self.model_version = str(manifest.get("model_version", "artifact-unknown"))
        self.calibration_version = str(manifest.get("calibration_version", "artifact-unknown"))
        self.feature_version = str(manifest.get("feature_version", "stylometry-v1"))
        self._classifier_tokenizer: Any | None = None
        self._classifier_model: Any | None = None
        self._stylometry_model: Any | None = None
        self._stylometry_feature_names: list[str] | None = None
        self._meta_model: Any | None = None
        self._meta_feature_names: list[str] | None = None
        self._calibrator: Any | None = None

    def predict_classifier_probabilities(self, texts: list[str]) -> list[float] | None:
        spec = self.manifest.get("classifier")
        if not spec or not texts:
            return None

        tokenizer, model, torch = self._load_classifier_components(spec)
        if tokenizer is None or model is None or torch is None:
            return None

        positive_label = int(spec.get("positive_label", 1))
        max_length = int(spec.get("max_length", 512))
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        device = self._resolve_torch_device(torch)
        model.to(device)
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.inference_mode():
            logits = model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1)[:, positive_label].detach().cpu().tolist()
        return [float(value) for value in probabilities]

    def predict_stylometry_probabilities(self, feature_sets: list[SegmentFeatureSet]) -> list[float] | None:
        spec = self.manifest.get("stylometry")
        if not spec or not feature_sets:
            return None

        model = self._load_stylometry_model(spec)
        feature_names = self._load_json_list(spec.get("feature_names_path"))
        if model is None or feature_names is None:
            return None

        matrix = [stylometric_feature_vector(feature_set, feature_names) for feature_set in feature_sets]
        input_data = self._to_named_dataframe(matrix, feature_names)
        return self._predict_binary_probabilities(model, input_data)

    def predict_document_probability(self, feature_map: dict[str, float]) -> float | None:
        spec = self.manifest.get("meta")
        if not spec:
            return None

        model = self._load_meta_model(spec)
        feature_names = self._load_json_list(spec.get("feature_names_path"))
        if model is None or feature_names is None:
            return None

        # Clamp features that are likely out-of-distribution for the meta-model.
        # The meta-model was trained on typical documents (200–2000 tokens); very
        # long scientific papers produce extreme token_count / segment_token values
        # that cause the model to extrapolate incorrectly.
        _OOD_CLAMPS: dict[str, tuple[float, float]] = {
            "token_count": (0.0, 5000.0),
            "mean_segment_tokens": (0.0, 600.0),
            "max_segment_tokens": (0.0, 800.0),
            "segment_count": (0.0, 20.0),
        }
        clamped_map = dict(feature_map)
        for key, (lo, hi) in _OOD_CLAMPS.items():
            if key in clamped_map:
                clamped_map[key] = max(lo, min(hi, clamped_map[key]))

        row = [[clamped_map.get(name, 0.0) for name in feature_names]]
        input_data = self._to_named_dataframe(row, feature_names)
        raw_prediction = self._predict_binary_probabilities(model, input_data)
        if raw_prediction is None:
            return None

        probability = float(raw_prediction[0])
        calibrator = self._load_calibrator(spec)
        if calibrator is None:
            return probability
        return self._apply_calibrator(calibrator, probability)

    def _load_classifier_components(self, spec: dict[str, Any]) -> tuple[Any | None, Any | None, Any | None]:
        if self._classifier_model is not None and self._classifier_tokenizer is not None:
            try:
                torch = import_module("torch")
            except ImportError:
                return self._classifier_tokenizer, self._classifier_model, None
            return self._classifier_tokenizer, self._classifier_model, torch

        model_path = self._resolve_path(spec.get("path"))
        if model_path is None:
            return None, None, None

        try:
            torch = import_module("torch")
            transformers = import_module("transformers")
            auto_tokenizer = getattr(transformers, "AutoTokenizer")
            auto_model = getattr(transformers, "AutoModelForSequenceClassification")

            self._classifier_tokenizer = auto_tokenizer.from_pretrained(model_path)
            self._classifier_model = auto_model.from_pretrained(model_path)
            return self._classifier_tokenizer, self._classifier_model, torch
        except ImportError:
            logger.warning("Artifact-backed classifier requested but optional ML dependencies are not installed.")
            return None, None, None
        except Exception as exc:
            logger.warning("Failed to load classifier artifacts from %s: %s", model_path, exc)
            return None, None, None

    def _load_stylometry_model(self, spec: dict[str, Any]) -> Any | None:
        if self._stylometry_model is not None:
            return self._stylometry_model

        model_path = self._resolve_path(spec.get("path"))
        if model_path is None:
            return None

        try:
            load = getattr(import_module("joblib"), "load")

            self._stylometry_model = load(model_path)
            return self._stylometry_model
        except ImportError:
            logger.warning("Artifact-backed stylometry requested but joblib is not installed.")
            return None
        except Exception as exc:
            logger.warning("Failed to load stylometry model from %s: %s", model_path, exc)
            return None

    def _load_meta_model(self, spec: dict[str, Any]) -> Any | None:
        if self._meta_model is not None:
            return self._meta_model

        model_path = self._resolve_path(spec.get("path"))
        if model_path is None:
            return None

        try:
            load = getattr(import_module("joblib"), "load")

            self._meta_model = load(model_path)
            return self._meta_model
        except ImportError:
            logger.warning("Artifact-backed meta-model requested but joblib is not installed.")
            return None
        except Exception as exc:
            logger.warning("Failed to load meta-model from %s: %s", model_path, exc)
            return None

    def _load_calibrator(self, spec: dict[str, Any]) -> Any | None:
        if self._calibrator is not None:
            return self._calibrator

        calibrator_path = self._resolve_path(spec.get("calibrator_path"))
        if calibrator_path is None:
            return None

        try:
            load = getattr(import_module("joblib"), "load")

            self._calibrator = load(calibrator_path)
            return self._calibrator
        except ImportError:
            logger.warning("Artifact-backed calibration requested but joblib is not installed.")
            return None
        except Exception as exc:
            logger.warning("Failed to load calibrator from %s: %s", calibrator_path, exc)
            return None

    def _load_json_list(self, relative_path: str | None) -> list[str] | None:
        if relative_path is None:
            return None
        absolute_path = self._resolve_path(relative_path)
        if absolute_path is None:
            return None
        try:
            payload = json.loads(absolute_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to load artifact metadata from %s: %s", absolute_path, exc)
            return None

        if isinstance(payload, list):
            return [str(item) for item in payload]
        return None

    def _resolve_path(self, relative_path: str | None) -> Path | None:
        if not relative_path:
            return None
        candidate = (self.artifact_dir / relative_path).resolve()
        if not candidate.exists():
            logger.warning("Artifact path does not exist: %s", candidate)
            return None
        return candidate

    def _resolve_torch_device(self, torch: Any) -> Any:
        if self.device_preference == "cpu":
            return torch.device("cpu")
        if self.device_preference == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if self.device_preference == "cuda":
            logger.warning("CUDA was requested for artifact inference but is not available; falling back to CPU.")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _to_named_dataframe(self, matrix: list[list[float]], feature_names: list[str]) -> Any:
        """Wrap the matrix in a pandas DataFrame with named columns.

        Eliminates sklearn/LightGBM warnings about missing feature names.
        Falls back to the raw matrix if pandas is not installed.
        """
        try:
            pd = import_module("pandas")
            return pd.DataFrame(matrix, columns=feature_names)
        except Exception:
            return matrix

    def _predict_binary_probabilities(self, model: Any, matrix: Any) -> list[float] | None:
        try:
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(matrix)
                return [float(row[1]) for row in probabilities]
            if hasattr(model, "decision_function"):
                scores = model.decision_function(matrix)
                return [float(1.0 / (1.0 + math.exp(-score))) for score in scores]
            predictions = model.predict(matrix)
            return [float(value) for value in predictions]
        except Exception as exc:
            logger.warning("Failed to generate probabilities from artifact-backed model: %s", exc)
            return None

    def _apply_calibrator(self, calibrator: Any, probability: float) -> float:
        try:
            if hasattr(calibrator, "predict_proba"):
                calibrated = calibrator.predict_proba([[probability]])
                return float(calibrated[0][1])
            calibrated = calibrator.predict([probability])
            return float(calibrated[0])
        except Exception as exc:
            logger.warning("Failed to calibrate artifact-backed prediction: %s", exc)
            return probability


@lru_cache
def get_artifact_bundle() -> ArtifactBundle | None:
    settings = get_settings()
    if not settings.enable_artifact_models or not settings.ml_artifact_dir:
        return None

    artifact_dir = Path(settings.ml_artifact_dir).expanduser().resolve()
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.exists():
        logger.warning("Artifact-backed inference is enabled, but no manifest was found at %s", manifest_path)
        return None

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to parse artifact manifest from %s: %s", manifest_path, exc)
        return None

    return ArtifactBundle(artifact_dir=artifact_dir, manifest=manifest, device_preference=settings.artifact_device)
