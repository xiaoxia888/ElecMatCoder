from __future__ import annotations


def looks_like_od_wall_thickness(
    outer_diameter: object,
    wall_thickness: object,
    *,
    minimum_ratio: float = 3.0,
    maximum_thickness: float = 80.0,
) -> bool:
    """Judge whether a numeric pair is physically plausible as OD x wall."""

    try:
        od = float(str(outer_diameter or "").strip())
        thickness = float(str(wall_thickness or "").strip())
    except (TypeError, ValueError):
        return False
    if od <= 0 or thickness <= 0 or thickness >= od or thickness > maximum_thickness:
        return False
    return od / thickness >= minimum_ratio
