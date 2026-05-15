"""Discovery of ``multi_<motion_type><index>.csv`` / ``*_ref.csv`` sample pairs.

中文说明：本模块只做文件名层面的样本发现。传感器文件必须形如
``multi_kaihe1.csv``，参考心率文件支持两种命名：同 stem 加 ``_ref``
（旧 Polar 格式）或 ``_HR_ref``（新格式 CSV）。没有配对的 CSV 会写入
unpaired，不进入后续 QC、预处理或训练。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "LEGAL_MOTION_TYPES",
    "PairDiscovery",
    "SamplePair",
    "UnpairedSample",
    "discover_sample_pairs",
    "discover_sample_pairs_with_unpaired",
    "parse_motion_id",
]

LEGAL_MOTION_TYPES = ("tiaosheng", "wanju", "fuwo", "kaihe", "bobi")
_SENSOR_RE = re.compile(r"^multi_(?P<motion_id>[A-Za-z]+(?P<motion_index>\d+))$")


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


@dataclass(frozen=True)
class UnpairedSample:
    """A CSV that cannot enter training because its counterpart is absent."""

    file_name: str
    file_path: Path
    reason: str


@dataclass(frozen=True)
class PairDiscovery:
    """Full discovery result including rejected unpaired files."""

    pairs: list[SamplePair]
    unpaired: list[UnpairedSample]


def parse_motion_id(stem_or_motion_id: str) -> tuple[str, int, str] | None:
    """Parse motion type/index from ``kaihe1`` or ``multi_kaihe1``.

    中文说明：只接受当前实验定义的 5 类运动；非法类型返回 ``None``，调用方把
    文件记入 unpaired 表，避免错误文件混入训练。
    """

    text = str(stem_or_motion_id).removeprefix("multi_").removesuffix("_ref")
    for motion_type in LEGAL_MOTION_TYPES:
        prefix = motion_type
        suffix = text[len(prefix) :]
        if text.startswith(prefix) and suffix.isdigit():
            return motion_type, int(suffix), f"{motion_type}{int(suffix)}"
    return None


def discover_sample_pairs(input_dir: Path) -> list[SamplePair]:
    """Return paired ``multi_<motion_id>.csv`` samples from ``input_dir``."""

    return discover_sample_pairs_with_unpaired(input_dir).pairs


def discover_sample_pairs_with_unpaired(input_dir: Path) -> PairDiscovery:
    """Discover valid sample pairs and all lone sensor/reference CSVs.

    中文说明：先正向扫描传感器 CSV，再反向扫描孤立参考 CSV；所有 rejected 文件
    都有明确 reason，Notebook 和 ``unpaired_samples.csv`` 可直接展示。
    """

    root = Path(input_dir)
    csv_files = sorted(p for p in root.glob("*.csv") if p.is_file())
    by_stem = {p.stem: p for p in csv_files}

    pairs: list[SamplePair] = []
    unpaired: list[UnpairedSample] = []
    paired_ref_stems: set[str] = set()

    for sensor in csv_files:
        if sensor.stem.endswith("_ref"):
            continue
        match = _SENSOR_RE.match(sensor.stem)
        parsed = parse_motion_id(sensor.stem)
        if match is None or parsed is None:
            unpaired.append(
                UnpairedSample(
                    file_name=sensor.name,
                    file_path=sensor,
                    reason=(
                        "sensor name must match multi_<motion_type><index>.csv "
                        f"with motion_type in {', '.join(LEGAL_MOTION_TYPES)}"
                    ),
                )
            )
            continue
        motion_type, motion_index, motion_id = parsed
        ref = None
        ref_stem = ""
        for suffix in ("_ref", "_HR_ref"):
            candidate = f"{sensor.stem}{suffix}"
            found = by_stem.get(candidate)
            if found is not None:
                ref = found
                ref_stem = candidate
                break
        if ref is None:
            unpaired.append(
                UnpairedSample(
                    file_name=sensor.name,
                    file_path=sensor,
                    reason=f"missing reference file (tried {sensor.stem}_ref.csv, {sensor.stem}_HR_ref.csv)",
                )
            )
            continue
        paired_ref_stems.add(ref_stem)
        pairs.append(
            SamplePair(
                motion_id=motion_id,
                motion_type=motion_type,
                motion_index=motion_index,
                stem=sensor.stem,
                sensor_csv=sensor,
                ref_csv=ref,
            )
        )

    sensor_stems = {p.sensor_csv.stem for p in pairs}
    for ref in csv_files:
        if not (ref.stem.endswith("_ref") or ref.stem.endswith("_HR_ref")):
            continue
        sensor_stem = ref.stem
        for suffix in ("_HR_ref", "_ref"):
            if sensor_stem.endswith(suffix):
                sensor_stem = sensor_stem.removesuffix(suffix)
                break
        if ref.stem in paired_ref_stems or sensor_stem in sensor_stems:
            continue
        unpaired.append(
            UnpairedSample(
                file_name=ref.name,
                file_path=ref,
                reason=f"missing sensor file {sensor_stem}.csv",
            )
        )

    return PairDiscovery(
        pairs=sorted(pairs, key=lambda x: (x.motion_type, x.motion_index, x.stem)),
        unpaired=sorted(unpaired, key=lambda x: x.file_name),
    )
