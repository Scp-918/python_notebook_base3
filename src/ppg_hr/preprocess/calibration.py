"""Load subject calibration coefficients from the shared MATLAB file."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

__all__ = ["CalibrationCoefficients", "load_subject_calibration"]


@dataclass(frozen=True)
class CalibrationCoefficients:
    """Validated coefficients used to derive HF2 and UD2."""

    source_path: Path
    source_sha256: str
    c2: float
    c3: float
    k2: float
    k3: float

    def to_metadata(self) -> dict[str, object]:
        return {
            "calibration_path": str(self.source_path),
            "calibration_sha256": self.source_sha256,
            "calibration_c2": self.c2,
            "calibration_c3": self.c3,
            "calibration_k2": self.k2,
            "calibration_k3": self.k3,
        }


def load_subject_calibration(
    subject_dir: str | Path,
    *,
    on_log: Callable[[str], None] | None = None,
) -> CalibrationCoefficients:
    """Load and validate ``subject_dir.parent / 'ck.mat'``."""

    subject = Path(subject_dir).resolve()
    path = (subject.parent / "ck.mat").resolve()
    message = f"标定文件解析路径: {path}"
    if on_log is not None:
        on_log(message)
    if not path.is_file():
        raise FileNotFoundError(f"Calibration file not found: {path}")
    try:
        payload = loadmat(path, simplify_cells=True)
    except Exception as exc:
        raise ValueError(f"Cannot parse calibration MAT file {path}: {exc}") from exc
    c_result = _cell_sequence(payload, "cResult", minimum=3)
    k_result = _cell_sequence(payload, "kResult", minimum=2)

    c2_entry = _struct(c_result[1], "cResult{2}")
    c3_entry = _struct(c_result[2], "cResult{3}")
    k2_entry = _struct(k_result[0], "kResult{1}")
    k3_entry = _struct(k_result[1], "kResult{2}")
    _expect_integer(c2_entry, "channel", 2, "cResult{2}.channel")
    _expect_integer(c3_entry, "channel", 3, "cResult{3}.channel")
    _expect_integer(k2_entry, "mainChannel", 2, "kResult{1}.mainChannel")
    _expect_integer(k3_entry, "mainChannel", 3, "kResult{2}.mainChannel")
    return CalibrationCoefficients(
        source_path=path,
        source_sha256=_sha256_file(path),
        c2=_finite_scalar(c2_entry, "c", "cResult{2}.c"),
        c3=_finite_scalar(c3_entry, "c", "cResult{3}.c"),
        k2=_finite_scalar(k2_entry, "k", "kResult{1}.k"),
        k3=_finite_scalar(k3_entry, "k", "kResult{2}.k"),
    )


def _cell_sequence(payload: dict[str, Any], name: str, *, minimum: int) -> list[Any]:
    if name not in payload:
        raise ValueError(f"Calibration MAT missing field {name}")
    value = payload[name]
    if isinstance(value, np.ndarray):
        items = list(value.ravel())
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        items = [value]
    if len(items) < minimum:
        raise ValueError(f"Calibration field {name} must contain at least {minimum} cells")
    return items


def _struct(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a MATLAB struct")
    return value


def _finite_scalar(entry: dict[str, Any], field: str, label: str) -> float:
    if field not in entry:
        raise ValueError(f"Calibration field missing: {label}")
    value = np.asarray(entry[field])
    if value.size != 1:
        raise ValueError(f"{label} must be scalar")
    try:
        scalar = float(value.reshape(-1)[0])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not np.isfinite(scalar):
        raise ValueError(f"{label} must be finite")
    return scalar


def _expect_integer(entry: dict[str, Any], field: str, expected: int, label: str) -> None:
    value = _finite_scalar(entry, field, label)
    if value != expected:
        raise ValueError(f"{label} must equal {expected}, got {value:g}")


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
