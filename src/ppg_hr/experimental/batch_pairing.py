"""Discovery of ``multi_<motion_id>.csv`` / ``*_ref.csv`` sample pairs.

中文说明：
本模块只做文件名层面的样本发现。普通运动文件必须是
``multi_<运动拼音数字>.csv``，参考心率文件必须是同 stem 加 ``_ref``。
没有成对出现的 CSV 会被记录为 unpaired，不进入后续 QC/优化。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "PairDiscovery",
    "SamplePair",
    "UnpairedSample",
    "discover_sample_pairs",
    "discover_sample_pairs_with_unpaired",
]

_SENSOR_RE = re.compile(r"^multi_(?P<motion_id>[A-Za-z0-9_]+)$")


@dataclass(frozen=True)
class SamplePair:
    """A paired sensor/reference CSV sample."""

    motion_id: str
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


def discover_sample_pairs(input_dir: Path) -> list[SamplePair]:
    """Return paired ``multi_<motion_id>.csv`` samples from ``input_dir``.

    Reference files are never treated as sensor inputs. Unpaired files can be
    obtained with :func:`discover_sample_pairs_with_unpaired`.
    """

    return discover_sample_pairs_with_unpaired(input_dir).pairs


def discover_sample_pairs_with_unpaired(input_dir: Path) -> PairDiscovery:
    """Discover valid sample pairs and all lone sensor/reference CSVs."""

    root = Path(input_dir)
    csv_files = sorted(p for p in root.glob("*.csv") if p.is_file())
    by_stem = {p.stem: p for p in csv_files}

    # 中文注释：先遍历运动文件；任何 ``*_ref.csv`` 都不能被误认为运动数据。
    pairs: list[SamplePair] = []
    unpaired: list[UnpairedSample] = []
    paired_ref_stems: set[str] = set()

    for sensor in csv_files:
        if sensor.stem.endswith("_ref"):
            continue
        match = _SENSOR_RE.match(sensor.stem)
        if match is None:
            unpaired.append(
                UnpairedSample(
                    file_name=sensor.name,
                    file_path=sensor,
                    reason="sensor name does not match multi_<motion_id>.csv",
                )
            )
            continue
        ref_stem = f"{sensor.stem}_ref"
        ref = by_stem.get(ref_stem)
        if ref is None:
            unpaired.append(
                UnpairedSample(
                    file_name=sensor.name,
                    file_path=sensor,
                    reason=f"missing reference file {ref_stem}.csv",
                )
            )
            continue
        paired_ref_stems.add(ref_stem)
        pairs.append(
            SamplePair(
                motion_id=match.group("motion_id"),
                stem=sensor.stem,
                sensor_csv=sensor,
                ref_csv=ref,
            )
        )

    sensor_stems = {p.sensor_csv.stem for p in pairs}
    for ref in csv_files:
        # 中文注释：再反向检查孤立参考文件，便于 QC summary 说明丢弃原因。
        if not ref.stem.endswith("_ref"):
            continue
        sensor_stem = ref.stem.removesuffix("_ref")
        if ref.stem in paired_ref_stems or sensor_stem in sensor_stems:
            continue
        unpaired.append(
            UnpairedSample(
                file_name=ref.name,
                file_path=ref,
                reason=f"missing sensor file {sensor_stem}.csv",
            )
        )

    return PairDiscovery(pairs=pairs, unpaired=sorted(unpaired, key=lambda x: x.file_name))
