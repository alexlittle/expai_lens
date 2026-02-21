# tests/conftest.py
import os
import json
import time
import types
import pandas as pd
import pytest

class _DummyExpander:
    def __init__(self, *a, **k):
        pass
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False

class _DummySidebar:
    def __init__(self, st):
        self._st = st

    def header(self, *args, **kwargs): pass

    def text_input(self, label, value="", help=None):
        # Default behavior: return current default value (i.e., no user override)
        return value

    def slider(self, *args, **kwargs):
        # Return provided default if present; otherwise a simple fallback
        if "value" in kwargs:
            return kwargs["value"]
        if len(args) >= 3:
            return args[2]  # default positional
        return 20

    def number_input(self, *args, **kwargs):
        # Always return 0 for predictability
        return 0

    def expander(self, *args, **kwargs):
        return _DummyExpander()

class DummyStreamlit:
    """
    Minimal stub for streamlit functions used in the app. All methods are no-ops
    except .stop() which raises SystemExit to emulate Streamlit's flow control.
    """
    def __init__(self):
        self.sidebar = _DummySidebar(self)
        self._query_params = {}
        self._writes = []  # optional: collect certain calls if you want

    # Page/controls
    def set_page_config(self, **kwargs): pass

    # Query params
    def experimental_get_query_params(self):
        return self._query_params

    def set_query_params_for_test(self, params: dict):
        """Helper for tests to set simulated URL query params."""
        self._query_params = params

    # Text / UI
    def write(self, *args, **kwargs): self._writes.append(("write", args, kwargs))
    def warning(self, *args, **kwargs): self._writes.append(("warning", args, kwargs))
    def error(self, *args, **kwargs): self._writes.append(("error", args, kwargs))
    def info(self, *args, **kwargs): self._writes.append(("info", args, kwargs))
    def title(self, *args, **kwargs): self._writes.append(("title", args, kwargs))
    def caption(self, *args, **kwargs): self._writes.append(("caption", args, kwargs))
    def json(self, *args, **kwargs): self._writes.append(("json", args, kwargs))
    def header(self, *args, **kwargs): self._writes.append(("header", args, kwargs))
    def dataframe(self, *args, **kwargs): self._writes.append(("dataframe", args, kwargs))
    def bar_chart(self, *args, **kwargs): self._writes.append(("bar_chart", args, kwargs))
    def selectbox(self, label, options, index=0, help=None): return options[index]

    # Flow control
    def stop(self):
        raise SystemExit()

@pytest.fixture
def patch_meta_name(monkeypatch):
    """
    Force META_INFO_FILENAME in the target module to be 'meta.json' for tests,
    regardless of the real constant in xaicompare.consts.
    """
    import xaicompare.dashboard.app as app
    monkeypatch.setattr(app, "META_INFO_FILENAME", "meta.json", raising=False)
    return "meta.json"

@pytest.fixture
def patch_streamlit(monkeypatch):
    """
    Replace 'st' in the target module with a dummy stub to run 'main' without
    real Streamlit.
    """
    import xaicompare.dashboard.app as app
    stub = DummyStreamlit()
    monkeypatch.setattr(app, "st", stub, raising=False)
    return stub

@pytest.fixture
def temp_run_dir(tmp_path, patch_meta_name):
    """
    Create a minimal valid run directory with a meta.json in it. Returns the path.
    """
    run_dir = tmp_path / "runs" / "myrun"
    run_dir.mkdir(parents=True)
    (run_dir / "meta.json").write_text(json.dumps({"id": "myrun", "ts": "2026-01-01T00:00:00Z"}))
    return run_dir

@pytest.fixture
def fake_load_artifacts(monkeypatch):
    """
    Mock pd.read_parquet globally to avoid needing pyarrow/fastparquet in tests.
    Returns a factory to configure what DataFrames the mock will yield.
    """
    def _setup(preds=None, global_imp=None, local=None, text=None):
        if preds is None:
            preds = pd.DataFrame({"pred": [0.1, 0.9]})
        if global_imp is None:
            global_imp = pd.DataFrame({"feature": ["a", "b"], "mean_abs_importance": [0.5, 0.3]})
        if local is None:
            local = pd.DataFrame({
                "sample_id": [0, 0, 1],
                "feature": ["a", "b", "a"],
                "abs_value": [0.4, 0.2, 0.1],
                "value": [0.4, -0.2, 0.1],
            })
        if text is None:
            text = pd.DataFrame({"sample_id": [0, 1], "text": ["hello", "world"]})

        frames = [preds, global_imp, local, text]
        seq = {"i": 0}

        def fake_read_parquet(path):
            # Return frames in the order load_run expects
            i = seq["i"]
            df = frames[i]
            seq["i"] = (i + 1) % len(frames)
            return df

        monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)
        return preds, global_imp, local, text

    return _setup