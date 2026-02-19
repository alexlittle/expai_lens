from typing import Optional, Sequence, List
import numpy as np
import pandas as pd

from xaicompare.registry.model_registry import register_model
from xaicompare.adapters.models.model_base import ModelAdapter, ArrayLike


@register_model("sklearn")
class SklearnPipelineAdapter(ModelAdapter):
    def __init__(self, pipeline, class_names: Optional[Sequence[str]] = None):
        """
        Normalizes attributes so runner and utilities can rely on:
          - self.model  (canonical handle to the model/pipeline)
          - self.pipeline (kept for backward-compat)
          - self._class_names  (list[str] or None)
        Also caches common pipeline steps if they exist.
        """
        # Canonical attribute expected by runner/utilities
        self.model = pipeline

        # Backward-compat: keep your original name too
        self.pipeline = pipeline

        # Optional cached steps (if using sklearn Pipeline with named_steps)
        named_steps = getattr(pipeline, "named_steps", {}) or {}
        self._tfidf = named_steps.get("tfidf", None)
        self._clf   = named_steps.get("xgb", None)

        # Normalize class names to a list of str (or None)
        self._class_names = list(class_names) if class_names is not None else None

    # ---- Required API ----
    def predict(self, X: ArrayLike) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: ArrayLike):
        """
        Return probabilities when available; otherwise None.
        Keeps the runner logic simple and robust.
        """
        if hasattr(self.model, "predict_proba"):
            try:
                return self.model.predict_proba(X)
            except Exception:
                return None
        return None

    def class_names(self) -> Optional[List[str]]:
        return self._class_names

    def feature_names(self) -> List[str]:
        """
        Best-effort recovery of feature names.
        Tries vectorizers (get_feature_names_out) then estimator.feature_names_in_.
        """
        # Try any step exposing get_feature_names_out (e.g., TfidfVectorizer)
        for step in self._iterate_pipeline_steps():
            if hasattr(step, "get_feature_names_out"):
                try:
                    return list(step.get_feature_names_out())
                except Exception:
                    pass

        # Try final estimator's feature_names_in_
        est = self._final_estimator_or_self()
        if hasattr(est, "feature_names_in_"):
            try:
                return list(est.feature_names_in_)
            except Exception:
                pass

        return []

    # ---- Optional, used by runner to build text_index.parquet ----
    def build_text_index(
        self,
        X_test,
        y_test: Optional[Sequence] = None,
        raw_text: Optional[Sequence] = None,
        class_names: Optional[Sequence[str]] = None,
        **_
    ) -> pd.DataFrame:
        """
        Create a rich text index with:
          - sample_id, text
          - y_true (if provided), y_pred (if predict works)
          - proba_{class} columns when probabilities + class names are available
        """
        # 1) Determine text column
        if raw_text is None:
            # If X_test already looks like text, use it; else string-coerce
            try:
                if len(X_test) > 0 and isinstance(X_test[0], str):
                    raw_text = X_test
                else:
                    raw_text = [str(x) for x in X_test]
            except Exception:
                raw_text = [str(x) for x in X_test]
        n = len(raw_text)

        # 2) Predictions / probabilities (best-effort)
        y_pred = None
        try:
            y_pred = self.predict(X_test)
        except Exception:
            pass

        probas = None
        try:
            probas = self.predict_proba(X_test)
        except Exception:
            pass

        # 3) Assemble frame
        df = pd.DataFrame({
            "sample_id": np.arange(n, dtype=int),
            "text": list(raw_text),
        })

        if y_test is not None:
            df["y_true"] = list(y_test)

        if y_pred is not None:
            df["y_pred"] = list(y_pred)

        # 4) Per-class probability columns if consistent
        cnames = list(class_names) if class_names is not None else (self._class_names or None)
        if probas is not None and cnames is not None:
            if getattr(probas, "ndim", 1) == 2 and len(cnames) == probas.shape[1]:
                for j, cname in enumerate(cnames):
                    df[f"proba_{cname}"] = probas[:, j]

        return df

    # ---- Helpers ----
    def _iterate_pipeline_steps(self):
        """Yield steps from sklearn Pipeline if available, else yield the model itself."""
        named_steps = getattr(self.model, "named_steps", None)
        if isinstance(named_steps, dict) and named_steps:
            for _, step in named_steps.items():
                yield step
        else:
            yield self.model

    def _final_estimator_or_self(self):
        """Try to get the final estimator from a Pipeline; otherwise return the model itself."""
        steps = getattr(self.model, "steps", None)
        if steps and len(steps) > 0:
            return steps[-1][1]
        return self.model