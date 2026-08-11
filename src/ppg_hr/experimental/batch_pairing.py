"""Strict discovery of paired files inside one subject directory."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ..params import MotionType

__all__ = [
    "LEGAL_MOTION_TYPES",
    "PairDiscovery",
    "SamplePair",
    "UnpairedSample",
    "discover_sample_pairs",
    "discover_sample_pairs_with_unpaired",
    "parse_motion_id",
]

LEGAL_MOTION_TYPES = tuple(item.value for item in MotionType)
_FILE_RE = re.compile(
    r"^(?P<subject>.+)_(?P<motion>write|gripper|run|rope)_"
    r"(?P<index>\d+)_(?P<kind>sensor|HRdata)\.csv$"
)
_MOTION_RE = re.compile(
    r"^(?:(?P<subject>.+)_)?(?P<motion>write|gripper|run|rope)_"
    r"(?P<index>\d+)(?:_(?:sensor|HRdata)\.csv)?$"
)


@dataclass(frozen=True)
class SamplePair:
    """A paired sensor/reference CSV sample.

    中文说明：``motion_id`` 是运动类型和编号的组合，例如 ``kaihe1``；
    ``motion_type`` 用于按运动类型分组训练，``motion_index`` 用于固定拆分。
    """

    motion_id: str
    motion_type: str
    motion_index: int
    stem: str
    sensor_csv: Path
    ref_csv: Path
    subject: str = ""

    @property
    def sample_id(self) -> str:
        """Return the canonical ``subject_motion_index`` identifier."""

        return self.stem


@dataclass(frozen=True)
class UnpairedSample:
    """A CSV that cannot enter training because its counterpart is absent."""

    file_name: str
    file_path: Path
    reason: str
    category: str = "unpaired"


@dataclass(frozen=True)
class PairDiscovery:
    """Full discovery result including rejected unpaired files."""

    pairs: list[SamplePair]
    unpaired: list[UnpairedSample]


def parse_motion_id(stem_or_motion_id: str) -> tuple[str, int, str] | None:
    """Parse the canonical motion/index suffix from the right-hand side."""

    match = _MOTION_RE.fullmatch(Path(str(stem_or_motion_id)).name)
    if match is None:
        return None
    motion = match.group("motion")
    index = int(match.group("index"))
    subject = match.group("subject")
    sample_id = f"{subject}_{motion}_{index}" if subject else f"{motion}_{index}"
    return motion, index, sample_id


def discover_sample_pairs(input_dir: Path) -> list[SamplePair]:
    """Return valid pairs from one flat subject directory."""

    return discover_sample_pairs_with_unpaired(input_dir).pairs


def discover_sample_pairs_with_unpaired(input_dir: Path) -> PairDiscovery:
    """Discover strict sensor/HR pairs and return every rejected CSV."""

    root = Path(input_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"Subject directory not found: {root}")
    subject = root.name
    csv_files = sorted(
        (path for path in root.iterdir() if path.is_file() and path.suffix.lower() == ".csv"),
        key=lambda path: path.name,
    )
    accepted: dict[tuple[str, int, str], list[tuple[Path, re.Match[str]]]] = {}
    unpaired: list[UnpairedSample] = []

    for path in csv_files:
        match = _FILE_RE.fullmatch(path.name)
        if match is None:
            unpaired.append(_issue(path, "invalid_name", "filename does not match the subject data contract"))
            continue
        if match.group("subject") != subject:
            unpaired.append(
                _issue(path, "subject_mismatch", f"filename subject must equal directory name {subject!r}")
            )
            continue
        if not _has_data_row(path):
            unpaired.append(_issue(path, "empty_data", "CSV contains no data rows"))
            continue
        key = (match.group("motion"), int(match.group("index")), match.group("kind"))
        accepted.setdefault(key, []).append((path, match))

    duplicate_keys = {key for key, items in accepted.items() if len(items) > 1}
    for key in sorted(duplicate_keys):
        for path, _ in accepted[key]:
            unpaired.append(_issue(path, "duplicate", "duplicate file for normalized motion/index/kind"))

    pairs: list[SamplePair] = []
    motion_order = {motion: idx for idx, motion in enumerate(LEGAL_MOTION_TYPES)}
    sample_keys = sorted(
        {(motion, index) for motion, index, _ in accepted},
        key=lambda item: (motion_order[item[0]], item[1]),
    )
    for motion, index in sample_keys:
        sensor_key = (motion, index, "sensor")
        ref_key = (motion, index, "HRdata")
        sensor_items = [] if sensor_key in duplicate_keys else accepted.get(sensor_key, [])
        ref_items = [] if ref_key in duplicate_keys else accepted.get(ref_key, [])
        if len(sensor_items) == 1 and len(ref_items) == 1:
            sensor = sensor_items[0][0]
            ref = ref_items[0][0]
            sample_id = f"{subject}_{motion}_{index}"
            pairs.append(
                SamplePair(
                    motion_id=sample_id,
                    motion_type=motion,
                    motion_index=index,
                    stem=sample_id,
                    sensor_csv=sensor,
                    ref_csv=ref,
                    subject=subject,
                )
            )
            continue
        if len(sensor_items) == 1:
            unpaired.append(_issue(sensor_items[0][0], "missing_pair", "missing matching HRdata CSV"))
        if len(ref_items) == 1:
            unpaired.append(_issue(ref_items[0][0], "missing_pair", "missing matching sensor CSV"))

    return PairDiscovery(pairs=pairs, unpaired=sorted(unpaired, key=lambda item: item.file_name))


def _has_data_row(path: Path) -> bool:
    """Return whether a CSV has at least one non-blank row after its header."""

    try:
        with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
            next(handle, None)
            return any(line.strip() for line in handle)
    except OSError:
        return False


def _issue(path: Path, category: str, reason: str) -> UnpairedSample:
    return UnpairedSample(path.name, path, reason, category)
