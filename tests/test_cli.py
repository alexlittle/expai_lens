# tests/test_cli.py
import sys
import types
from pathlib import Path
import builtins
import importlib
import importlib.util
import pytest


def make_fake_streamlit_modules(monkeypatch, *, modern=True, record=None):
    """
    Install fake Streamlit modules into sys.modules so that:
      - modern=True:   from streamlit.web import cli as stcli  -> works
      - modern=False:  import streamlit.cli as stcli           -> works
    A callable `record(name)` can be passed to track which main() was used.
    """
    record = record or (lambda name: None)

    # Create 'streamlit' package
    streamlit = types.ModuleType("streamlit")
    streamlit.__path__ = []  # mark as package
    monkeypatch.setitem(sys.modules, "streamlit", streamlit)

    if modern:
        # Create 'streamlit.web' package with attribute 'cli' that has main()
        st_web = types.ModuleType("streamlit.web")
        st_web.__path__ = []
        monkeypatch.setitem(sys.modules, "streamlit.web", st_web)

        st_web_cli = types.ModuleType("streamlit.web.cli")
        def modern_main():
            record("modern")
            return 0
        st_web_cli.main = modern_main
        # Either expose as submodule or as attribute; 'from streamlit.web import cli' needs attribute 'cli'
        st_web.cli = st_web_cli
        monkeypatch.setitem(sys.modules, "streamlit.web.cli", st_web_cli)
    else:
        # Ensure modern path is absent to force fallback import
        sys.modules.pop("streamlit.web", None)
        sys.modules.pop("streamlit.web.cli", None)

        # Create 'streamlit.cli' module with main()
        st_cli = types.ModuleType("streamlit.cli")
        def legacy_main():
            # Flag for assertions
            setattr(st_cli, "USED_LEGACY", True)
            record("legacy")
            return 0
        st_cli.main = legacy_main
        monkeypatch.setitem(sys.modules, "streamlit.cli", st_cli)


def import_cli_module(tmp_path, monkeypatch):
    """
    Dynamically import xaicompare.cli from the real package if present;
    otherwise create a temporary module that mirrors the user's code,
    and return that module. We also allow overriding cli.__file__ in tests.
    """
    # Try normal import first (works if user runs tests inside their package)
    try:
        import xaicompare.cli as cli
        return cli
    except Exception:
        pass

    # If not importable, create a temp module with the given content
    pkg_dir = tmp_path / "xaicompare"
    pkg_dir.mkdir(parents=True, exist_ok=True)
    (pkg_dir / "__init__.py").write_text("")
    source = """\
import argparse
import sys
from pathlib import Path

def main():
    # Try Streamlit CLI import across versions
    try:
        from streamlit.web import cli as stcli  # modern
    except Exception:  # pragma: no cover
        import streamlit.cli as stcli  # older

    # Resolve path to your app.py inside the installed package
    app_py = Path(__file__).resolve().parent / "dashboard" / "app.py"
    if not app_py.exists():
        print(f"Cannot find Streamlit app at: {app_py}", file=sys.stderr)
        sys.exit(1)

    # Simple CLI: optional positional RUN_DIR and pass-through for other args
    parser = argparse.ArgumentParser(
        prog="xaicompare-dash",
        description="Launch XAICompare Dashboard",
        add_help=True,
    )
    parser.add_argument(
        "run",
        nargs="?",
        help="Optional path to a run folder (containing meta.json). If omitted, the app will let you pick.",
    )
    # Parse known args; forward the rest (e.g., --server.port 8502) to Streamlit
    args, remainder = parser.parse_known_args()

    # Build args for streamlit
    st_args = ["run", str(app_py), "--"]
    if args.run:
        st_args += ["--run", args.run]

    # Forward any extra Streamlit flags the user provides
    st_args += remainder

    # Emulate: streamlit run xaicompare/dashboard/app.py -- [--run ...] [extra]
    sys.argv = ["streamlit"] + st_args
    sys.exit(stcli.main())
"""
    cli_path = pkg_dir / "cli.py"
    cli_path.write_text(source)

    spec = importlib.util.spec_from_file_location("xaicompare.cli", cli_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["xaicompare.cli"] = module
    spec.loader.exec_module(module)  # type: ignore
    return module


def put_app_py_under(module, tmp_path, monkeypatch):
    """
    Create xaicompare/dashboard/app.py under the same folder where module.__file__ sits.
    Then monkeypatch module.__file__ to that path so the code resolves app_py correctly.
    Returns the app_py path.
    """
    base_dir = tmp_path / "xaicompare"
    base_dir.mkdir(parents=True, exist_ok=True)
    # Fake the module file path to point to tmp package dir
    fake_cli_file = base_dir / "cli.py"
    fake_cli_file.write_text("# dummy __file__ for cli module")
    monkeypatch.setattr(module, "__file__", str(fake_cli_file), raising=False)

    dashboard = base_dir / "dashboard"
    dashboard.mkdir(exist_ok=True)
    app_py = dashboard / "app.py"
    app_py.write_text("# fake streamlit app")
    return app_py


def set_argv(argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", argv.copy(), raising=False)


# -------------------------
#           TESTS
# -------------------------

def test_builds_sysargv_no_run_no_extra(tmp_path, monkeypatch, capsys):
    # Arrange
    cli = import_cli_module(tmp_path, monkeypatch)
    app_py = put_app_py_under(cli, tmp_path, monkeypatch)

    called = []
    make_fake_streamlit_modules(monkeypatch, modern=True, record=lambda v: called.append(v))

    set_argv(["xaicompare-dash"], monkeypatch)

    # Replace sys.exit to turn exit into exception we can catch
    with pytest.raises(SystemExit) as excinfo:
        cli.main()

    # Assert exit code is 0 (streamlit main returned 0)
    assert excinfo.value.code == 0

    # sys.argv should be rewritten for Streamlit
    assert sys.argv == ["streamlit", "run", str(app_py), "--"]

    # Ensure modern path was used
    assert called == ["modern"]


def test_with_run_and_extra_flags_order_preserved(tmp_path, monkeypatch):
    # Arrange
    cli = import_cli_module(tmp_path, monkeypatch)
    app_py = put_app_py_under(cli, tmp_path, monkeypatch)

    called = []
    make_fake_streamlit_modules(monkeypatch, modern=True, record=lambda v: called.append(v))

    # Simulate user invocation with positional run and extra streamlit flags
    set_argv(["xaicompare-dash", "runs/session-42", "--server.port", "8502", "--theme.base", "dark"], monkeypatch)

    with pytest.raises(SystemExit) as excinfo:
        cli.main()

    assert excinfo.value.code == 0

    # Expect "--run <path>" injected right after "--"
    assert sys.argv == [
        "streamlit",
        "run",
        str(app_py),
        "--",
        "--run",
        "runs/session-42",
        "--server.port",
        "8502",
        "--theme.base",
        "dark",
    ]
    assert called == ["modern"]


def test_missing_app_exits_with_error_and_stderr_message(tmp_path, monkeypatch, capsys):
    # Arrange
    cli = import_cli_module(tmp_path, monkeypatch)

    # Point __file__ to a temp dir with NO dashboard/app.py
    base_dir = tmp_path / "xaicompare"
    base_dir.mkdir(parents=True, exist_ok=True)
    fake_cli_file = base_dir / "cli.py"
    fake_cli_file.write_text("# dummy")
    monkeypatch.setattr(cli, "__file__", str(fake_cli_file), raising=False)

    # Ensure streamlit import succeeds so we reach the app.py existence check
    make_fake_streamlit_modules(monkeypatch, modern=True)

    set_argv(["xaicompare-dash"], monkeypatch)

    # Act
    with pytest.raises(SystemExit) as excinfo:
        cli.main()

    # Assert exit code 1
    assert excinfo.value.code == 1

    # Error message goes to stderr and contains the resolved app path
    stderr = capsys.readouterr().err
    expected_app_path = (base_dir / "dashboard" / "app.py")
    assert "Cannot find Streamlit app at:" in stderr
    assert str(expected_app_path) in stderr


def test_fallback_to_legacy_streamlit_cli_when_modern_import_fails(tmp_path, monkeypatch):
    # Arrange
    cli = import_cli_module(tmp_path, monkeypatch)
    app_py = put_app_py_under(cli, tmp_path, monkeypatch)

    # Force modern import to be unavailable so code takes the except path
    make_fake_streamlit_modules(monkeypatch, modern=False)

    set_argv(["xaicompare-dash"], monkeypatch)

    # Act
    with pytest.raises(SystemExit) as excinfo:
        cli.main()

    # Assert legacy was used – we set a flag on streamlit.cli.main
    assert excinfo.value.code == 0
    st_cli = sys.modules.get("streamlit.cli")
    assert st_cli is not None
    assert getattr(st_cli, "USED_LEGACY", False) is True

    # Also confirm sys.argv was constructed
    assert sys.argv == ["streamlit", "run", str(app_py), "--"]