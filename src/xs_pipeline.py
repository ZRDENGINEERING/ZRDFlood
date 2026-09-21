"""
xs_pipeline.py

Chains polygon_centerline.py -> generate_cross_sections.py into one step:
skinny bank/channel polygon in, cross sections (with self-checks) out.

Dependencies: geopandas, shapely (>=2.0), numpy, scipy, networkx
"""

from __future__ import annotations

import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

from polygon_centerline import polygon_centerline, polygon_centerline_multi
from generate_cross_sections  import generate_cross_sections, check_cross_sections



def cross_sections_from_polygon(
    polygon,
    spacing: float,
    half_width: float,
    *,
    # polygon_centerline params
    segment_length: float | None = None,
    simplify_tolerance: float | None = None,
    trim_ends: float = 0.0,
    # generate_cross_sections params
    tangent_step: float | None = None,
    # check_cross_sections params
    run_checks: bool = True,
    bearing_jump_threshold: float = 25.0,
) -> gpd.GeoDataFrame:
    """
    Full pipeline: derive a centerline from a skinny polygon, then generate
    (and self-check) cross sections along it.

    Parameters mirror polygon_centerline() and generate_cross_sections()/
    check_cross_sections() — see those modules' docstrings for details on
    trim_ends (important for square-cut polygon ends), tangent_step, and
    bearing_jump_threshold.

    For a MultiPolygon, each part is processed independently and the results
    are concatenated, with an added `part_id` column so you can tell which
    polygon part each cross section came from.

    Returns
    -------
    GeoDataFrame of cross sections (columns: xs_id, station, bearing, plus
    check_cross_sections columns if run_checks, plus part_id if the input
    was a MultiPolygon), with .crs set from the input if it has one.
    """
    if isinstance(polygon, gpd.GeoDataFrame) or isinstance(polygon, gpd.GeoSeries):
        crs = polygon.crs
        geom = unary_union(polygon.geometry.values if hasattr(polygon, "geometry") else polygon.values)
    elif isinstance(polygon, (Polygon, MultiPolygon)):
        crs = None
        geom = polygon
    else:
        raise TypeError(f"Unsupported polygon input type: {type(polygon)}")

    cl_kwargs = dict(segment_length=segment_length, simplify_tolerance=simplify_tolerance, trim_ends=trim_ends)

    if isinstance(geom, Polygon):
        parts = [geom]
    else:
        parts = list(geom.geoms)

    all_xs = []
    for part_id, part in enumerate(parts):
        centerline = polygon_centerline(part, **cl_kwargs)
        xs = generate_cross_sections(centerline, spacing, half_width, tangent_step=tangent_step)
        if run_checks:
            xs = check_cross_sections(xs, centerline, bearing_jump_threshold=bearing_jump_threshold)
        if len(parts) > 1:
            xs.insert(0, "part_id", part_id)
        all_xs.append(xs)

    result = gpd.GeoDataFrame(
        gpd.pd.concat(all_xs, ignore_index=True), geometry="geometry"
    )
    if crs is not None:
        result = result.set_crs(crs)
    return result


def cross_sections_from_polygon_file(
    polygon_path: str,
    spacing: float,
    half_width: float,
    crs=None,
    **kwargs,
) -> gpd.GeoDataFrame:
    """Load a polygon from any file GeoPandas can read and run the full pipeline."""
    gdf = gpd.read_file(polygon_path)
    if crs is not None:
        gdf = gdf.to_crs(crs)
    if gdf.crs is None:
        print(
            f"[warning] '{polygon_path}' has no CRS defined (missing .prj / SRID "
            "metadata). Output cross sections will also have crs=None — pass "
            "crs=<EPSG code> explicitly to avoid downstream SRID mislabeling."
        )
    return cross_sections_from_polygon(gdf, spacing, half_width, **kwargs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate cross sections directly from a skinny bank/channel polygon.")
    parser.add_argument("polygon", help="Path to polygon vector file (shp/gpkg/geojson)")
    parser.add_argument("--spacing", type=float, required=True, help="Along-stream spacing between XS, in CRS units")
    parser.add_argument("--half-width", type=float, required=True, help="Half-width of each XS, in CRS units")
    parser.add_argument("--out", required=True, help="Output vector file path (e.g. cross_sections.gpkg)")
    parser.add_argument("--crs", default=None, help="Optional EPSG code to reproject polygon to before processing")
    parser.add_argument("--segment-length", type=float, default=None, help="Centerline: boundary densify step (CRS units)")
    parser.add_argument("--simplify-tolerance", type=float, default=None, help="Centerline: simplify tolerance (CRS units)")
    parser.add_argument("--trim-ends", type=float, default=0.0, help="Centerline: distance to trim off each end (CRS units); use for square-cut polygon ends")
    parser.add_argument("--bearing-jump-threshold", type=float, default=25.0, help="XS check: degrees of bearing change that triggers a flag")
    parser.add_argument("--no-checks", action="store_true", help="Skip cross-section self-checks")
    args = parser.parse_args()

    xs_gdf = cross_sections_from_polygon_file(
        args.polygon,
        args.spacing,
        args.half_width,
        crs=args.crs,
        segment_length=args.segment_length,
        simplify_tolerance=args.simplify_tolerance,
        trim_ends=args.trim_ends,
        run_checks=not args.no_checks,
        bearing_jump_threshold=args.bearing_jump_threshold,
    )

    if args.out.lower().endswith(".shp") and not args.no_checks:
        print(
            "[warning] writing to .shp: the 'flags' text field may be truncated "
            "at 254 chars by the DBF format; 'flagged' (bool) is safe to symbolize "
            "on in QGIS either way. Use .gpkg to keep the full 'flags' text intact."
        )

    xs_gdf.to_file(args.out)
    print(f"Wrote {len(xs_gdf)} cross sections to {args.out}")
    if not args.no_checks and "flagged" in xs_gdf.columns:
        flagged = xs_gdf[xs_gdf["flagged"]]
        if len(flagged):
            cols = [c for c in ["part_id", "xs_id", "station", "flags"] if c in flagged.columns]
            print(flagged[cols].to_string(index=False))