from __future__ import annotations

from typing import TypedDict


class RegionDict(TypedDict, total=False):
    region_id: int
    x: int
    y: int
    w: int
    h: int
    tamper_types: list[str]


class SampleDict(TypedDict, total=False):
    id: str
    image_w: int
    image_h: int
    is_authentic: bool
    regions: list[RegionDict]
