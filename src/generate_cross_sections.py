"""
generate_cross_sections.py

Automatically generate perpendicular cross-section lines along a stream
centerline, at regular intervals, for GIS/hydraulic modeling use
(e.g. HEC-RAS input, bank delineation, Manning's validation).

Input:  a single-LineString (or multi-part, auto-merged) stream centerline
Output: a GeoDataFrame / shapefile / PostGIS table of cross-section lines,
        each centered on the centerline, oriented perpendicular to local
        flow direction, tagged with station (distance along centerline).

Dependencies: geopandas, shapely (>=2.0), numpy
"""

from __future__ import annotations

import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, Point, MultiLineString
from shapely.ops import linemerge


def _as_single_linestring(geom) -> LineString:
    """Merge a MultiLineString centerline into one LineString if possible."""
    if isinstance(geom, LineString):
        return geom
    if isinstance(geom, MultiLineString):
        merged = linemerge(geom)
        if isinstance(merged, LineString):
            return merged
        raise ValueError(
            "Centerline MultiLineString could not be merged into a single "
            "LineString (gaps or branches present). Dissolve/clean it first."
        )
    raise TypeError(f"Unsupported centerline geometry type: {geom.geom_type}")


def _local_direction(line: LineString, station: float, tangent_step: float) -> tuple[float, float]:
    """
    Estimate the local tangent direction (unit vector) at a given station
    (distance along line) using a small forward/backward difference,
    clamped to the line's domain.
    """
    length = line.length
    s0 = max(station - tangent_step, 0.0)
    s1 = min(station + tangent_step, length)
    if s1 == s0:
        # degenerate (station at an endpoint with zero-length window)
        s0, s1 = max(0.0, length - tangent_step), length
    p0 = line.interpolate(s0)
    p1 = line.interpolate(s1)
    dx, dy = p1.x - p0.x, p1.y - p0.y
    norm = (dx**2 + dy**2) ** 0.5
    if norm == 0:
        return 1.0, 0.0
    return dx / norm, dy / norm


def generate_cross_sections(
    centerline,
    spacing: float,
    half_width: float,
    *,
    tangent_step: float | None = None,
    start_station: float = 0.0,
    end_station: float | None = None,
    include_endpoints: bool = True,
) -> gpd.GeoDataFrame:
    """
    Generate perpendicular cross-section lines along a stream centerline.

    Parameters
    ----------
    centerline : LineString or MultiLineString
        The stream centerline, in a projected CRS (feet/meters), not lat/lon.
    spacing : float
        Distance between cross sections along the centerline, in CRS units.
    half_width : float
        Half-length of each cross section (extends half_width to each side
        of the centerline). Total XS length = 2 * half_width.
    tangent_step : float, optional
        Distance used to estimate local flow direction (central difference).
        Defaults to spacing / 4, with a floor of 1.0.
    start_station, end_station : float
        Station range (distance along line) to generate sections over.
        end_station defaults to the full line length.
    include_endpoints : bool
        Whether to force a cross section exactly at start/end station
        even if it falls off the regular spacing grid.

    Returns
    -------
    GeoDataFrame with columns:
        xs_id     - sequential integer id
        station   - distance along centerline (in CRS units)
        geometry  - LineString cross section, perpendicular to centerline
    """
    line = _as_single_linestring(centerline)
    length = line.length
    if end_station is None:
        end_station = length
    if tangent_step is None:
        tangent_step = max(spacing / 4.0, 1.0)

    stations = list(np.arange(start_station, end_station, spacing))
    if include_endpoints:
        if not stations or stations[0] > start_station:
            stations.insert(0, start_station)
        if stations[-1] < end_station:
            stations.append(end_station)

    records = []
    for i, s in enumerate(stations):
        center = line.interpolate(s)
        dx, dy = _local_direction(line, s, tangent_step)
        # perpendicular = rotate tangent 90 degrees
        px, py = -dy, dx
        p1 = Point(center.x - px * half_width, center.y - py * half_width)
        p2 = Point(center.x + px * half_width, center.y + py * half_width)
        bearing = np.degrees(np.arctan2(px, py)) % 360.0  # bearing of the XS line itself
        records.append(
            {
                "xs_id": i,
                "station": round(s, 3),
                "bearing": round(bearing, 3),
                "geometry": LineString([p1, p2]),
            }
        )

    gdf = gpd.GeoDataFrame(records, geometry="geometry")
    return gdf


def check_cross_sections(
    xs_gdf: gpd.GeoDataFrame,
    centerline,
    *,
    bearing_jump_threshold: float = 25.0,
    require_crs: bool = True,
) -> gpd.GeoDataFrame:
    """
    Run self-checks on generated cross sections and return an annotated copy
    flagging rows that likely need a look in QGIS before use.

    Checks:
      - bearing_delta: change in XS bearing vs. the previous station. A large
        jump usually means either a genuinely sharp bend or a noisy/jagged
        centerline destabilizing the tangent estimate.
      - self_intersects: whether the XS line crosses itself (shouldn't
        normally happen for a straight 2-point line, but flagged defensively
        in case of future edits to the geometry builder).
      - crosses_centerline_twice: an XS should cross the centerline exactly
        once (at its own station); crossing it 0 or 2+ times means the XS is
        long enough, and the bend sharp enough, that it wraps back over the
        stream — a strong sign to shorten half_width or increase spacing
        density at that station.
      - overlaps_neighbor: whether this XS geometrically intersects the
        previous or next XS line, which usually indicates crossed/bowtied
        sections through a tight meander.
      - crs_missing: flagged once (on row 0) if require_crs and the
        GeoDataFrame has no CRS set.

    Attribute columns added (kept to <=10 chars so they survive a .shp write
    without silent truncation/collision):
        brng_dlt  (float) - bearing_delta, degrees vs. previous station
        self_int  (bool)  - self_intersects
        bad_cross (bool)  - crosses_centerline_twice
        ovlp_nbr  (bool)  - overlaps_neighbor
        brng_flg  (bool)  - bearing_delta exceeded bearing_jump_threshold
        flagged   (bool)  - True if any check triggered (for quick QGIS symbology)
        flags     (str)   - human-readable list of triggered checks (may get
                             truncated if written to .shp; prefer .gpkg for
                             the full text, `flagged` is safe on either)
    """
    gdf = xs_gdf.copy().reset_index(drop=True)
    line = _as_single_linestring(centerline)

    n = len(gdf)
    bearing_delta = np.zeros(n)
    self_intersects = np.zeros(n, dtype=bool)
    crosses_centerline_twice = np.zeros(n, dtype=bool)
    overlaps_neighbor = np.zeros(n, dtype=bool)

    bearings = gdf["bearing"].to_numpy() if "bearing" in gdf.columns else None

    for i, geom in enumerate(gdf.geometry):
        # bearing jump vs previous station
        if bearings is not None and i > 0:
            raw_delta = abs(bearings[i] - bearings[i - 1]) % 360.0
            bearing_delta[i] = min(raw_delta, 360.0 - raw_delta)

        # self-intersection (defensive; a simple 2-point line can't normally do this)
        self_intersects[i] = not geom.is_simple

        # how many times does this XS cross the centerline?
        n_crossings = geom.intersection(line)
        if n_crossings.is_empty:
            crosses_centerline_twice[i] = True  # 0 crossings is also wrong
        elif n_crossings.geom_type == "MultiPoint":
            crosses_centerline_twice[i] = len(n_crossings.geoms) != 1
        elif n_crossings.geom_type != "Point":
            # a LineString/overlap result means the XS ran along the centerline
            crosses_centerline_twice[i] = True

        # overlap with immediate neighbors
        neighbors = []
        if i > 0:
            neighbors.append(gdf.geometry.iloc[i - 1])
        if i < n - 1:
            neighbors.append(gdf.geometry.iloc[i + 1])
        overlaps_neighbor[i] = any(geom.crosses(nb) or geom.overlaps(nb) for nb in neighbors)

    gdf["brng_dlt"] = np.round(bearing_delta, 2)
    gdf["self_int"] = self_intersects
    gdf["bad_cross"] = crosses_centerline_twice
    gdf["ovlp_nbr"] = overlaps_neighbor
    gdf["brng_flg"] = gdf["brng_dlt"] > bearing_jump_threshold

    crs_missing = require_crs and gdf.crs is None

    def _row_flags(row, is_first):
        flags = []
        if row["brng_flg"]:
            flags.append(f"bearing_jump>{bearing_jump_threshold}deg")
        if row["self_int"]:
            flags.append("self_intersects")
        if row["bad_cross"]:
            flags.append("bad_centerline_crossing_count")
        if row["ovlp_nbr"]:
            flags.append("overlaps_neighbor")
        if is_first and crs_missing:
            flags.append("crs_missing")
        return ",".join(flags)

    gdf["flags"] = [
        _row_flags(row, is_first=(i == 0)) for i, row in gdf.iterrows()
    ]
    gdf["flagged"] = gdf["flags"] != ""

    n_flagged = gdf["flagged"].sum()
    if n_flagged:
        print(f"[check_cross_sections] {n_flagged} of {n} cross sections flagged for review.")
    else:
        print(f"[check_cross_sections] all {n} cross sections passed self-checks.")

    return gdf


def cross_sections_from_file(
    centerline_path: str,
    spacing: float,
    half_width: float,
    crs=None,
    run_checks: bool = True,
    bearing_jump_threshold: float = 25.0,
    **kwargs,
) -> gpd.GeoDataFrame:
    """
    Load a centerline from any file GeoPandas can read, build cross sections,
    and (by default) run self-checks, returning the annotated GeoDataFrame.
    """
    cl = gpd.read_file(centerline_path)
    if crs is not None:
        cl = cl.to_crs(crs)
    if cl.crs is None:
        print(
            f"[warning] '{centerline_path}' has no CRS defined (missing .prj / SRID "
            "metadata). Output cross sections will also have crs=None. Downstream "
            "consumers that default an unset CRS to something else (e.g. NAD83 "
            "geographic, EPSG:4269) will silently mislabel these projected "
            "coordinates. Pass crs=<EPSG code> explicitly to avoid this."
        )
    geom = linemerge(cl.geometry.values) if len(cl) > 1 else cl.geometry.iloc[0]
    xs = generate_cross_sections(geom, spacing, half_width, **kwargs)
    xs = xs.set_crs(cl.crs)
    if run_checks:
        xs = check_cross_sections(xs, geom, bearing_jump_threshold=bearing_jump_threshold)
    return xs


def write_to_postgis(gdf: gpd.GeoDataFrame, table_name: str, schema: str, conn_str: str, if_exists: str = "replace"):
    """
    Write cross sections to PostGIS. conn_str is a SQLAlchemy-style URL, e.g.
    'postgresql://user:pass@localhost:5432/zrdproj'
    """
    from sqlalchemy import create_engine

    engine = create_engine(conn_str)
    gdf.to_postgis(table_name, engine, schema=schema, if_exists=if_exists, index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate stream cross sections from a centerline.")
    parser.add_argument("centerline", help="Path to centerline vector file (shp/gpkg/geojson)")
    parser.add_argument("--spacing", type=float, required=True, help="Along-stream spacing between XS, in CRS units")
    parser.add_argument("--half-width", type=float, required=True, help="Half-width of each XS, in CRS units")
    parser.add_argument("--out", required=True, help="Output vector file path (e.g. cross_sections.gpkg)")
    parser.add_argument("--crs", default=None, help="Optional EPSG code to reproject centerline to before processing")
    parser.add_argument("--bearing-jump-threshold", type=float, default=25.0, help="Degrees of XS-to-XS bearing change that triggers a flag")
    parser.add_argument("--no-checks", action="store_true", help="Skip self-checks")
    args = parser.parse_args()

    xs_gdf = cross_sections_from_file(
        args.centerline,
        args.spacing,
        args.half_width,
        crs=args.crs,
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
    if not args.no_checks:
        flagged = xs_gdf[xs_gdf["flagged"]]
        if len(flagged):
            print(flagged[["xs_id", "station", "flags"]].to_string(index=False))