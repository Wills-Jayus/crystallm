from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SettingConvention:
    setting_id: str = "crystalformer"
    origin_shift: tuple[float, float, float] | None = None
    basis_transform: list[list[float]] | None = None
    note: str = "CrystalFormer/Wyckoff lookup operations are treated as the OrbitEngine source of truth."


DEFAULT_SETTING = SettingConvention()

