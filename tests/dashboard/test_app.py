# tests/test_app.py
import json
import os
import time
from pathlib import Path

import pandas as pd
import pytest

import xaicompare.dashboard.app as app


# ---------- Helpers: filesystem-based ----------

def test_find_latest_run_and_list_valid_runs_ordering(tmp_path, patch_meta_name):
    runs = tmp_path / "runs"
    runs.mkdir()

    a = runs / "2026-01-01"
    b = runs / "2026-01-02_no_meta"
    c = runs / "2026-01-03"
    a.mkdir()
    b.mkdir()
    c.mkdir()

    # Add meta.json to a and c (b is invalid)
    (a / "meta.json").write_text(json.dumps({"run": "a"}))
    (c / "meta.json").write_text(json.dumps({"run": "c"}))

    # Set mtimes: a older, c newer
    t0 = time.time()
    os.utime(a, (t0 - 100, t0 - 100))
    os.utime(c, (t0 - 10, t0 - 10))

    latest = app.find_latest_run(runs)
    assert latest == c

    ordered = app.list_valid_runs(runs)
    assert ordered == [c, a]


# ---------- Helpers: argv/query parsing ----------

def test_parse_cli_run_arg_variants(monkeypatch):
    # Case 1: '--' separator with '--run value'
    monkeypatch.setattr(app, "sys", type("S", (), {"argv": ["streamlit", "run", "app.py", "--", "--run", "X"]}))
    assert app.parse_cli_run_arg() == "X"

    # Case 2: '--' separator with '--run=value'
    monkeypatch.setattr(app, "sys", type("S", (), {"argv": ["streamlit", "run", "app.py", "--", "--run=Y"]}))
    assert app.parse_cli_run_arg() == "Y"

    # Case 3: no '--', but '--run' present
    monkeypatch.setattr(app, "sys", type("S", (), {"argv": ["xaicompare-dash", "--run", "Z"]}))
    assert app.parse_cli_run_arg() == "Z"

    # Case 4: absent -> default
    monkeypatch.setattr(app, "sys", type("S", (), {"argv": ["xaicompare-dash"]}))
    assert app.parse_cli_run_arg(default="DEFAULT") == "DEFAULT"


def test_get_run_from_query_params_ok_and_exception(monkeypatch, patch_streamlit):
    # Normal path
    app.st.set_query_params_for_test({"run": ["abc"]})
    assert app.get_run_from_query_params() == "abc"

    # Exception path
    def boom():
        raise RuntimeError("bad")
    monkeypatch.setattr(app.st, "experimental_get_query_params", boom)
    assert app.get_run_from_query_params() is None


# ---------- Helpers: load_run ----------

def test_load_run_reads_meta_and_parquet(tmp_path, monkeypatch, patch_meta_name, fake_load_artifacts):
    # Arrange
    run_dir = tmp_path / "r"
    run_dir.mkdir()
    (run_dir / "meta.json").write_text(json.dumps({"k": 1}))

    # Mock pd.read_parquet to avoid parquet engine deps
    p, g, l, t = fake_load_artifacts()

    # Act
    meta, preds, global_imp, local, text = app.load_run(str(run_dir))

    # Assert
    assert meta == {"k": 1}
    # The dfs we mocked should come back as-is
    assert list(preds.columns) == list(p.columns)
    assert list(global_imp.columns) == list(g.columns)
    assert list(local.columns) == list(l.columns)
    assert list(text.columns) == list(t.columns)


# ---------- main(): resolution & flows ----------

def _mock_load_run_capture(monkeypatch):
    """
    Returns (calls, set_mock) to capture the arg main() passes into load_run.
    """
    calls = []
    def fake_load_run(run_dir):
        calls.append(run_dir)
        # Return minimally compatible frames for the UI branch
        preds = pd.DataFrame({"pred": [0.1, 0.9]})
        global_imp = pd.DataFrame({"feature": ["a", "b"], "mean_abs_importance": [0.5, 0.3]})
        local = pd.DataFrame({
            "sample_id": [0, 0, 1],
            "feature": ["a", "b", "a"],
            "abs_value": [0.4, 0.2, 0.1],
            "value": [0.4, -0.2, 0.1],
        })
        text = pd.DataFrame({"sample_id": [0, 1], "text": ["hello", "world"]})
        meta = {"id": "dummy"}
        return meta, preds, global_imp, local, text
    monkeypatch.setattr(app, "load_run", fake_load_run)
    return calls

def test_main_prefers_cli_over_others(tmp_path, patch_streamlit, patch_meta_name, monkeypatch):
    run_dir = tmp_path / "runs" / "cli_run"
    run_dir.mkdir(parents=True)
    (run_dir / "meta.json").write_text("{}")

    # Run inside tmp_path for relative paths
    monkeypatch.chdir(tmp_path)

    # Precedence: CLI > URL > latest > fallback
    monkeypatch.setattr(app, "parse_cli_run_arg", lambda default=None: str(run_dir))
    monkeypatch.setattr(app, "get_run_from_query_params", lambda: None)
    monkeypatch.setattr(app, "find_latest_run", lambda base=Path("runs"): None)

    calls = _mock_load_run_capture(monkeypatch)

    # Default text_input returns the provided default value, so no override
    app.main()
    assert calls and Path(calls[0]) == run_dir

def test_main_uses_url_when_no_cli(tmp_path, patch_streamlit, patch_meta_name, monkeypatch):
    run_dir = tmp_path / "runs" / "url_run"
    run_dir.mkdir(parents=True)
    (run_dir / "meta.json").write_text("{}")

    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(app, "parse_cli_run_arg", lambda default=None: None)
    monkeypatch.setattr(app, "get_run_from_query_params", lambda: str(run_dir))
    monkeypatch.setattr(app, "find_latest_run", lambda base=Path("runs"): None)

    calls = _mock_load_run_capture(monkeypatch)
    app.main()
    assert Path(calls[0]) == run_dir

def test_main_uses_latest_when_no_cli_or_url(tmp_path, patch_streamlit, patch_meta_name, monkeypatch):
    run_dir = tmp_path / "runs" / "latest_run"
    run_dir.mkdir(parents=True)
    (run_dir / "meta.json").write_text("{}")

    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(app, "parse_cli_run_arg", lambda default=None: None)
    monkeypatch.setattr(app, "get_run_from_query_params", lambda: None)
    monkeypatch.setattr(app, "find_latest_run", lambda base=Path("runs"): run_dir)

    calls = _mock_load_run_capture(monkeypatch)
    app.main()
    assert Path(calls[0]) == run_dir

def test_main_uses_fallback_runs_latest(tmp_path, patch_streamlit, patch_meta_name, monkeypatch):
    # Create fallback runs/_latest with meta.json
    monkeypatch.chdir(tmp_path)
    fallback = Path("runs") / "_latest"
    fallback.mkdir(parents=True)
    (fallback / "meta.json").write_text("{}")

    monkeypatch.setattr(app, "parse_cli_run_arg", lambda default=None: None)
    monkeypatch.setattr(app, "get_run_from_query_params", lambda: None)
    monkeypatch.setattr(app, "find_latest_run", lambda base=Path("runs"): None)

    calls = _mock_load_run_capture(monkeypatch)
    app.main()
    # main passes str(path), so compare to string
    assert calls[0] == str(fallback)

def test_main_no_runs_shows_warning_and_stops(monkeypatch, patch_streamlit, patch_meta_name):
    # Force sidebar text_input to return empty, and no valid runs
    monkeypatch.setattr(app, "parse_cli_run_arg", lambda default=None: None)
    monkeypatch.setattr(app, "get_run_from_query_params", lambda: None)
    monkeypatch.setattr(app, "find_latest_run", lambda base=Path("runs"): None)
    monkeypatch.setattr(app, "list_valid_runs", lambda base=Path("runs"): [])

    # Make the text_input return "" explicitly
    def empty_text_input(label, value="", help=None):
        return ""
    app.st.sidebar.text_input = empty_text_input

    with pytest.raises(SystemExit):
        app.main()

def test_main_invalid_run_path_errors_and_stops(monkeypatch, patch_streamlit, patch_meta_name, tmp_path):
    # Make user type an invalid run path (no meta.json)
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(app, "parse_cli_run_arg", lambda default=None: None)
    monkeypatch.setattr(app, "get_run_from_query_params", lambda: None)
    monkeypatch.setattr(app, "find_latest_run", lambda base=Path("runs"): None)

    # Force sidebar text_input to return a non-existent path string
    bad = "not/a/real/run"
    app.st.sidebar.text_input = lambda *a, **k: bad

    with pytest.raises(SystemExit):
        app.main()
