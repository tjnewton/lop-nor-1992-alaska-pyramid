"""Single source of data paths for AKP.

All data lives under ``$AKP_DATA`` (the external data drive). Code calls
``data_root()`` / ``data_dir(name)``; nothing else in ``akp/`` spells a path.
"""
from __future__ import annotations

import os
from pathlib import Path

ENV_VAR = "AKP_DATA"


def data_root() -> Path:
    """Return ``$AKP_DATA`` as a Path; raise if unset or missing (drive not mounted)."""
    raw = os.environ.get(ENV_VAR)
    if not raw:
        raise RuntimeError(
            f"{ENV_VAR} is not set. Activate the swe-akp env (it sets {ENV_VAR}) "
            f"or export {ENV_VAR}=<data root on the external drive>."
        )
    root = Path(raw)
    if not root.is_dir():
        raise RuntimeError(f"{ENV_VAR}={raw!r} does not exist. Is the external drive mounted?")
    return root


def data_dir(name: str) -> Path:
    """Return (creating if needed) the ``$AKP_DATA/<name>`` subdirectory."""
    d = data_root() / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def data_glob(directory: Path, pattern: str) -> list[Path]:
    """Sorted ``directory.glob(pattern)`` minus macOS AppleDouble ``._*`` files.

    The data drive is exFAT, where macOS writes a binary ``._name`` sidecar next to files;
    a bare glob such as ``*.semd`` or ``*.mseed`` picks those up. Glob $AKP_DATA through here.
    """
    return sorted(p for p in Path(directory).glob(pattern) if not p.name.startswith("._"))


if __name__ == "__main__":
    import tempfile

    saved = os.environ.pop(ENV_VAR, None)
    try:
        try:
            data_root()
            raise AssertionError("data_root() must raise when AKP_DATA is unset")
        except RuntimeError:
            pass
        os.environ[ENV_VAR] = "/nonexistent/akp"
        try:
            data_root()
            raise AssertionError("data_root() must raise when AKP_DATA is missing")
        except RuntimeError:
            pass
        with tempfile.TemporaryDirectory() as tmp:
            os.environ[ENV_VAR] = tmp
            assert data_root() == Path(tmp)
            assert data_dir("x").is_dir()
            (Path(tmp) / "a.semd").write_text("1")
            (Path(tmp) / "._a.semd").write_bytes(b"\xb0")
            assert [q.name for q in data_glob(Path(tmp), "*.semd")] == ["a.semd"]
    finally:
        os.environ.pop(ENV_VAR, None)
        if saved is not None:
            os.environ[ENV_VAR] = saved
    print("akp.paths self-check OK")
