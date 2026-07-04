"""Make the ggswarm package importable without an editable install."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source" / "ggswarm"))
