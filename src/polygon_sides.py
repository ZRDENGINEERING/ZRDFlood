"""
polygon_sides.py

Split a skinny/elongated polygon's boundary into its two long sides
(e.g. left bank / right bank of a channel polygon), each as its own
LineString, for saving as separate layers.

Two ways to locate the two "tip" cut points where the sides meet:
  1. Default: from the polygon's minimum rotated rectangle — the midpoints
     of its two short edges, projected onto the polygon boundary. Robust,
     no dependencies beyond shapely, works well when width << length.
  2. If you pass a `centerline` (e.g. from polygon_centerline.py, called
     with trim_ends=0 so its endpoints sit at the true tips), its first and
     last vertices are used instead — better for a strongly curved polygon
     where the MRR's long axis doesn't track the actual bend.

Left/right labeling is relative to the direction from the start cut point
to the end cut point (or the supplied centerline's direction, if given):
standing at the start looking toward the end, "left" is on your left.

Dependencies: shapely (>=2.0), geopandas (I/O only)
"""

from __future__ import annotations

import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import substring


def _mrr_short_edge_midpoints(polygon: Polygon) -> tuple[Point, Point]:
    """Return the midpoints of the two short edges of the polygon's
    minimum rotated rectangle (the likely 'tip' ends of an elongated shape)."""
    mrr = polygon.minimum_rotated_rectangle
    coords = list(mrr.exterior.coords)[:-1]  # drop closing duplicate
    if len(coords) != 4:
        raise ValueError(
            "Polygon's minimum rotated rectangle didn't return 4 corners "
            "(degenerate/near-zero-area polygon?)."
        )
    edges = [(coords[i], coords[(i + 1) % 4]) for i in range(4)]
    lengths = [Point(a).distance(Point(b)) for a, b in edges]
    # the two short edges are opposite each other: indices of the 2 smallest lengths
    short_idx = np.argsort(lengths)[:2]
    midpoints = []
    for i in short_idx:
        a, b = edges[i]
        midpoints.append(Point((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0))
    return midpoints[0], midpoints[1]


def _split_ring(ring_line: LineString, p1: Point, p2: Point) -> tuple[LineString, LineString]:
    """Split a closed ring LineString into two open LineStrings at the
    boundary points nearest p1 and p2."""
    d1 = ring_line.project(p1)
    d2 = ring_line.project(p2)
    if d1 == d2:
        raise ValueError("Both cut points project to the same location on the boundary; ends may be too close together or the polygon too short for this operation.")
    lo, hi = sorted([d1, d2])

    side_a = substring(ring_line, lo, hi)

    tail = substring(ring_line, hi, ring_line.length)
    head = substring(ring_line, 0, lo)
    tail_coords = list(tail.coords)
    head_coords = list(head.coords)
    # tail ends at ring start/end vertex, head starts there too -> merge, dropping the duplicate
    side_b_coords = tail_coords + head_coords[1:]
    side_b = LineString(side_b_coords)

    return side_a, side_b


def extract_polygon_sides(
    polygon: Polygon,
    *,
    centerline: LineString | None = None,
    trim_ends: float = 0.0,
) -> dict:
    """
    Split a skinny polygon's boundary into its two long sides.

    Parameters
    ----------
    polygon : Polygon
        Must have no interior holes (raises otherwise) and be elongated
        enough that "two long sides" is a meaningful concept.
    centerline : LineString, optional
        If given, its first/last vertices locate the tip cut points instead
        of the minimum-rotated-rectangle method. Recommended for strongly
        curved polygons; pass one built with trim_ends=0 so the endpoints
        sit at the actual tips, not pulled inward.
    trim_ends : float, default 0.0
        Distance to shorten each resulting side line at both ends, in CRS
        units. Useful to drop the flat end-cap edge from each side.

    Returns
    -------
    dict with keys "left", "right" (LineString), "cut_points" (the two
    Points used to split the boundary), and "length" values for each side.
    """
    if not isinstance(polygon, Polygon):
        raise TypeError(f"Expected Polygon, got {type(polygon)}")
    if len(polygon.interiors) > 0:
        raise ValueError("Polygon has interior holes; extract_polygon_sides expects a simple ring boundary.")

    if centerline is not None:
        p_start = Point(centerline.coords[0])
        p_end = Point(centerline.coords[-1])
    else:
        p_start, p_end = _mrr_short_edge_midpoints(polygon)

    ring_line = LineString(polygon.exterior.coords)
    side_a, side_b = _split_ring(ring_line, p_start, p_end)

    # label left/right using the direction from p_start to p_end (or centerline direction)
    if centerline is not None and len(centerline.coords) >= 2:
        dx = centerline.coords[-1][0] - centerline.coords[0][0]
        dy = centerline.coords[-1][1] - centerline.coords[0][1]
    else:
        dx = p_end.x - p_start.x
        dy = p_end.y - p_start.y

    def _side_of(line: LineString) -> float:
        # sample the side's midpoint and test which side of the start->end
        # direction vector it falls on via 2D cross product sign
        mid = line.interpolate(0.5, normalized=True)
        vx, vy = mid.x - p_start.x, mid.y - p_start.y
        return dx * vy - dy * vx  # >0 = left, <0 = right

    cross_a = _side_of(side_a)
    if cross_a > 0:
        left, right = side_a, side_b
    else:
        left, right = side_b, side_a

    if trim_ends > 0:
        for name, line in [("left", left), ("right", right)]:
            if trim_ends * 2 >= line.length:
                raise ValueError(f"trim_ends={trim_ends} would remove the entire '{name}' side (length={line.length:.1f}); reduce trim_ends.")
        left = substring(left, trim_ends, left.length - trim_ends)
        right = substring(right, trim_ends, right.length - trim_ends)

    return {
        "left": left,
        "right": right,
        "cut_points": (p_start, p_end),
        "length": {"left": left.length, "right": right.length},
    }


def polygon_sides_to_gdf(sides: dict, crs=None) -> gpd.GeoDataFrame:
    """Package extract_polygon_sides() output as a 2-row GeoDataFrame."""
    if isinstance(sides, gpd.GeoDataFrame):
        raise TypeError(
            "polygon_sides_to_gdf() expects the dict returned by extract_polygon_sides() "
            "(keys: 'left', 'right', 'cut_points', 'length'), but got a GeoDataFrame. "
            "This usually means you already have the finished output from sides_from_file() "
            "and don't need to call polygon_sides_to_gdf() again — use that result directly."
        )
    if not isinstance(sides, dict) or "length" not in sides or not isinstance(sides.get("length"), dict):
        raise TypeError(
            "polygon_sides_to_gdf() expects the dict returned by extract_polygon_sides() "
            "(keys: 'left', 'right', 'cut_points', 'length', where 'length' is itself a "
            f"dict with 'left'/'right' keys); got {type(sides)} instead."
        )
    return gpd.GeoDataFrame(
        {"side": ["left", "right"], "length": [sides["length"]["left"], sides["length"]["right"]]},
        geometry=[sides["left"], sides["right"]],
        crs=crs,
    )


def sides_from_file(polygon_path: str, crs=None, use_centerline: bool = True, **kwargs) -> gpd.GeoDataFrame:
    """
    Load a polygon from file, optionally derive its centerline (for more
    accurate tip cut points on curved shapes), split into sides, and return
    them as a GeoDataFrame.
    """
    gdf = gpd.read_file(polygon_path)
    if crs is not None:
        gdf = gdf.to_crs(crs)
    if gdf.crs is None:
        print(
            f"[warning] '{polygon_path}' has no CRS defined (missing .prj / SRID "
            "metadata) — pass crs=<EPSG code> explicitly to avoid downstream mislabeling."
        )

    from shapely.ops import unary_union
    geom = unary_union(gdf.geometry.values)
    if geom.geom_type != "Polygon":
        raise ValueError(f"Expected a single Polygon after union, got {geom.geom_type}; process multi-part inputs per-part.")

    centerline = None
    if use_centerline:
        from polygon_centerline import polygon_centerline
        centerline = polygon_centerline(geom, trim_ends=0.0)

    sides = extract_polygon_sides(geom, centerline=centerline, **kwargs)
    return polygon_sides_to_gdf(sides, crs=gdf.crs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Split a skinny polygon into its two long sides (e.g. left/right bank).")
    parser.add_argument("polygon", help="Path to polygon vector file (shp/gpkg/geojson)")
    parser.add_argument("--out", required=True, help="Output vector file path for the two sides")
    parser.add_argument("--crs", default=None, help="Optional EPSG code to reproject to before processing")
    parser.add_argument("--trim-ends", type=float, default=0.0, help="Distance to trim off each end of each side (CRS units)")
    parser.add_argument("--no-centerline", action="store_true", help="Use the minimum-rotated-rectangle method instead of deriving a centerline (faster, less accurate on curved polygons)")
    args = parser.parse_args()

    out_gdf = sides_from_file(
        args.polygon, crs=args.crs, use_centerline=not args.no_centerline, trim_ends=args.trim_ends
    )
    out_gdf.to_file(args.out)
    print(f"Wrote left/right sides to {args.out}")
    print(out_gdf[["side", "length"]].to_string(index=False))