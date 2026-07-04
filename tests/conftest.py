"""Import ggswarm submodules under pytest without executing the package
__init__ (which pulls in Isaac Sim). A stub package module with __path__
pointing at the real package dir lets `import ggswarm.<submodule>` resolve
normally while skipping ggswarm/__init__.py entirely."""
import sys
import types
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parents[1] / "source" / "ggswarm" / "ggswarm"

if "ggswarm" not in sys.modules:
    pkg = types.ModuleType("ggswarm")
    pkg.__path__ = [str(PKG_DIR)]
    sys.modules["ggswarm"] = pkg
