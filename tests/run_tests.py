"""Minimal pytest-compatible runner for environments without pytest.
Discovers test_* functions, provides tmp_path, honors pytest.raises.
The real project uses pytest (see Makefile / CI); this exists so the
suite can be verified anywhere."""
import importlib.util
import inspect
import sys
import tempfile
import traceback
from pathlib import Path


class _Raises:
    def __init__(self, exc):
        self.exc = exc
    def __enter__(self):
        return self
    def __exit__(self, etype, e, tb):
        if etype is None:
            raise AssertionError(f"expected {self.exc.__name__}, none raised")
        return issubclass(etype, self.exc)


class _FakePytest:
    raises = _Raises

sys.modules.setdefault("pytest", _FakePytest())  # noqa: E402

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))

passed, failed = 0, []
for path in sorted((root / "tests").glob("test_*.py")):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name, fn in inspect.getmembers(mod, inspect.isfunction):
        if not name.startswith("test_"):
            continue
        try:
            kwargs = {}
            if "tmp_path" in inspect.signature(fn).parameters:
                kwargs["tmp_path"] = Path(tempfile.mkdtemp())
            fn(**kwargs)
            passed += 1
            print(f"PASS {path.stem}::{name}")
        except Exception:
            failed.append(f"{path.stem}::{name}")
            print(f"FAIL {path.stem}::{name}")
            traceback.print_exc()

print(f"\n{passed} passed, {len(failed)} failed")
sys.exit(1 if failed else 0)
