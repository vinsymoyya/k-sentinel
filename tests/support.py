"""Keep test scratch files in the checkout, including restricted Windows hosts."""
import tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SCRATCH=ROOT/"tmp"/"validation-tests"


def temporary_directory():
    SCRATCH.mkdir(parents=True,exist_ok=True)
    if not SCRATCH.resolve().is_relative_to(ROOT.resolve()):
        raise RuntimeError("Test scratch path must remain inside the checkout")
    return tempfile.TemporaryDirectory(dir=SCRATCH)
