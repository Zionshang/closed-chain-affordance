"""Shared loader that imports both implementations under distinct names.

* ``cca_cpp`` -> the compiled pybind11 binding (``tests/_cpp_ref/closed_chain_affordance.so``)
* ``cca_py``  -> the pure-Python package (``python/closed_chain_affordance``)

The compiled extension only exposes ``PyInit_closed_chain_affordance``, so it is
loaded under a dotted name (``_cca_cpp.closed_chain_affordance``); CPython's
dynamic loader matches the init function against the last path component, which
keeps the binding's internal name intact while avoiding a clash with the
pure-Python package imported as ``closed_chain_affordance``.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CPP_SO = REPO_ROOT / "tests" / "_cpp_ref" / "closed_chain_affordance.so"
PYTHON_DIR = REPO_ROOT / "python"


def load_cpp():
    """Load the compiled C++ binding under the alias ``cca_cpp``."""
    if not CPP_SO.exists():
        raise FileNotFoundError(
            f"C++ binding not found at {CPP_SO}. Build it with "
            "`cmake -S . -B build -DPython3_EXECUTABLE=$(which python) && cmake --build build -j4` "
            "then move python/closed_chain_affordance.so into tests/_cpp_ref/."
        )

    parent_name = "_cca_cpp"
    if parent_name not in sys.modules:
        parent = types.ModuleType(parent_name)
        parent.__path__ = []
        sys.modules[parent_name] = parent

    full_name = f"{parent_name}.closed_chain_affordance"
    if full_name in sys.modules:
        return sys.modules[full_name]

    spec = importlib.util.spec_from_file_location(full_name, str(CPP_SO))
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


def load_py():
    """Load the pure-Python package under the alias ``cca_py``."""
    if str(PYTHON_DIR) not in sys.path:
        sys.path.insert(0, str(PYTHON_DIR))
    import closed_chain_affordance as mod

    return mod


def load_both():
    """Return ``(cca_cpp, cca_py)``."""
    return load_cpp(), load_py()
