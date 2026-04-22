from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SpatialIntegrationVisualSpec:
    kind: str
    scene_shape: str
    scene_fill_rgb: tuple[int, int, int]
    scene_accent_rgb: tuple[int, int, int]
    scene_outline_rgb: tuple[int, int, int]
    topdown_icon: str
    topdown_rgb: tuple[int, int, int]
    asset_ids: tuple[str, ...]
    scale_class: str
    heading_class: str
    scene_scale_bias: float
    allow_answer_map_text: bool = True


def _visual_seed(*tokens: object) -> int:
    joined = "|".join(str(token) for token in tokens)
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(joined))


_VISUAL_SPECS: dict[str, SpatialIntegrationVisualSpec] = {
    "building": SpatialIntegrationVisualSpec(
        kind="building",
        scene_shape="hangar",
        scene_fill_rgb=(172, 180, 172),
        scene_accent_rgb=(134, 110, 96),
        scene_outline_rgb=(88, 92, 86),
        topdown_icon="hangar",
        topdown_rgb=(144, 104, 66),
        asset_ids=("building_hangar",),
        scale_class="building",
        heading_class="orthogonal",
        scene_scale_bias=1.15,
    ),
    "tower": SpatialIntegrationVisualSpec(
        kind="tower",
        scene_shape="tower",
        scene_fill_rgb=(204, 202, 172),
        scene_accent_rgb=(238, 228, 132),
        scene_outline_rgb=(90, 88, 76),
        topdown_icon="tower",
        topdown_rgb=(166, 168, 156),
        asset_ids=("building_tower",),
        scale_class="tower",
        heading_class="free",
        scene_scale_bias=1.15,
    ),
    "truck": SpatialIntegrationVisualSpec(
        kind="truck",
        scene_shape="truck",
        scene_fill_rgb=(112, 136, 92),
        scene_accent_rgb=(140, 158, 112),
        scene_outline_rgb=(62, 78, 52),
        topdown_icon="truck",
        topdown_rgb=(124, 110, 62),
        asset_ids=("truck_olive",),
        scale_class="truck",
        heading_class="vehicle",
        scene_scale_bias=1.0,
    ),
    "foot_soldiers": SpatialIntegrationVisualSpec(
        kind="foot_soldiers",
        scene_shape="soldiers",
        scene_fill_rgb=(26, 30, 34),
        scene_accent_rgb=(122, 124, 88),
        scene_outline_rgb=(24, 28, 20),
        topdown_icon="soldiers",
        topdown_rgb=(102, 104, 66),
        asset_ids=("soldiers_patrol",),
        scale_class="soldiers",
        heading_class="patrol",
        scene_scale_bias=1.0,
    ),
    "forest": SpatialIntegrationVisualSpec(
        kind="forest",
        scene_shape="trees",
        scene_fill_rgb=(68, 128, 66),
        scene_accent_rgb=(90, 70, 40),
        scene_outline_rgb=(34, 66, 32),
        topdown_icon="trees",
        topdown_rgb=(72, 124, 66),
        asset_ids=("trees_field_cluster", "forest_canopy_patch"),
        scale_class="forest",
        heading_class="free",
        scene_scale_bias=1.0,
    ),
    "tent": SpatialIntegrationVisualSpec(
        kind="tent",
        scene_shape="tent",
        scene_fill_rgb=(198, 186, 112),
        scene_accent_rgb=(168, 156, 88),
        scene_outline_rgb=(110, 98, 62),
        topdown_icon="tent",
        topdown_rgb=(192, 170, 98),
        asset_ids=("spatial_tent_canvas",),
        scale_class="tent",
        heading_class="orthogonal",
        scene_scale_bias=1.0,
    ),
    "sheep": SpatialIntegrationVisualSpec(
        kind="sheep",
        scene_shape="sheep",
        scene_fill_rgb=(232, 234, 228),
        scene_accent_rgb=(84, 88, 84),
        scene_outline_rgb=(126, 132, 128),
        topdown_icon="sheep",
        topdown_rgb=(214, 218, 220),
        asset_ids=("spatial_sheep_flock",),
        scale_class="sheep",
        heading_class="free",
        scene_scale_bias=1.0,
    ),
}


def spatial_integration_visual_spec(kind: str) -> SpatialIntegrationVisualSpec | None:
    return _VISUAL_SPECS.get(str(kind).strip().lower())


def supported_spatial_integration_landmark_kinds() -> tuple[str, ...]:
    return tuple(sorted(_VISUAL_SPECS))


def spatial_integration_landmark_asset_id(*, label: str, kind: str) -> str:
    spec = spatial_integration_visual_spec(kind)
    if spec is None:
        raise KeyError(f"Unsupported Spatial Integration landmark kind: {kind}")
    if len(spec.asset_ids) == 1:
        return spec.asset_ids[0]
    return spec.asset_ids[_visual_seed(label, kind) % len(spec.asset_ids)]


def spatial_integration_landmark_heading_deg(
    *,
    label: str,
    kind: str,
    asset_id: str | None = None,
) -> tuple[float, float, float]:
    spec = spatial_integration_visual_spec(kind)
    if spec is None:
        raise KeyError(f"Unsupported Spatial Integration landmark kind: {kind}")
    resolved_asset_id = asset_id or spatial_integration_landmark_asset_id(label=label, kind=kind)
    seed = _visual_seed(label, kind, resolved_asset_id)
    if spec.heading_class == "orthogonal":
        heading = float((seed % 4) * 90)
    elif spec.heading_class == "vehicle":
        heading = float((seed % 8) * 45)
    elif spec.heading_class == "patrol":
        heading = float((seed % 12) * 30)
    else:
        heading = float(seed % 360)
    return (heading, 0.0, 0.0)


def spatial_integration_landmark_gl_scale(
    *,
    label: str,
    kind: str,
    asset_id: str | None = None,
) -> tuple[float, float, float]:
    spec = spatial_integration_visual_spec(kind)
    if spec is None:
        raise KeyError(f"Unsupported Spatial Integration landmark kind: {kind}")
    resolved_asset_id = asset_id or spatial_integration_landmark_asset_id(label=label, kind=kind)
    base = {
        "building_hangar": 2.4,
        "building_tower": 3.2,
        "truck_olive": 1.9,
        "soldiers_patrol": 1.8,
        "trees_field_cluster": 2.8,
        "forest_canopy_patch": 3.1,
        "spatial_tent_canvas": 1.7,
        "spatial_sheep_flock": 1.5,
    }.get(resolved_asset_id)
    if base is None:
        base = {
            "building": 2.4,
            "tower": 3.2,
            "truck": 1.9,
            "soldiers": 1.8,
            "forest": 2.9,
            "tent": 1.7,
            "sheep": 1.5,
        }[spec.scale_class]
    variation = 0.94 + ((_visual_seed(label, kind, resolved_asset_id) % 11) * 0.012)
    scale = base * variation
    return (scale, scale, scale)

