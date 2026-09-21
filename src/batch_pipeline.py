"""
batch_pipeline.py

Batch driver for polygon_centerline.py / polygon_sides.py / generate_cross_sections.py
/ xs_pipeline.py, for running the full workflow (centerline -> sides ->
cross sections) over many polygons (thousands of stream reaches) at once,
without one bad feature aborting the whole run.

Each feature is processed independently and wrapped in a try/except; failures
are logged to a status DataFrame (feature id, stage, error message) rather
than raised, so you get partial results plus a clear list of what to fix.

Output is three consolidated GeoDataFrames (all features stacked, each row
tagged with the source feature's id column) plus the status log:
    centerlines : one row per successfully-processed feature
    sides       : two rows per successfully-processed feature (left/right)
    cross_sections : many rows per feature (one per station), including
                      the check_cross_sections flag columns

Dependencies: same as the other modules (geopandas, shapely>=2.0, numpy,
scipy, networkx)
"""

from __future__ import annotations

import time
import traceback
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon

from shapely.ops import substring
from polygon_centerline import polygon_centerline
from polygon_sides import extract_polygon_sides, polygon_sides_to_gdf
from generate_cross_sections import generate_cross_sections, check_cross_sections


def process_batch(
    polygons_gdf: gpd.GeoDataFrame,
    id_col: str,
    *,
    spacing: float,
    half_width: float,
    # centerline params
    segment_length: float | None = None,
    simplify_tolerance: float | None = None,
    trim_ends: float = 0.0,
    # xs params
    tangent_step: float | None = None,
    run_checks: bool = True,
    bearing_jump_threshold: float = 25.0,
    # sides params
    sides_cut_method: str = "mrr",
    # batch behavior
    progress_every: int = 100,
    skip_multipolygons: bool = False,
) -> dict:
    """
    Run centerline -> sides -> cross sections over every feature in
    polygons_gdf, isolating failures per feature.

    Parameters
    ----------
    polygons_gdf : GeoDataFrame
        One row per polygon to process (e.g. thousands of bank/channel
        polygons). Must have a projected CRS.
    id_col : str
        Column identifying each feature (e.g. a reach id). Must be unique.
        Carried onto every output row so you can join back to the source.
    spacing, half_width : float
        Passed to generate_cross_sections (see that module's docstring).
    segment_length, simplify_tolerance, trim_ends : see polygon_centerline
    tangent_step, run_checks, bearing_jump_threshold : see generate_cross_sections
    sides_cut_method : "mrr" (default) or "centerline"
        How extract_polygon_sides() finds where to cut the polygon boundary
        into left/right sides. "mrr" uses the polygon's minimum rotated
        rectangle (robust, independent of the derived centerline) — the
        right default for wide/floodplain-scale polygons, where a
        polygon_centerline()-derived skeleton's own endpoints can land
        nowhere near the true tips (its endpoint-selection heuristic can
        latch onto a spurious wide-branch pair instead of the real ends when
        the polygon is wide relative to its length — this doesn't reliably
        improve with a finer segment_length). "centerline" reuses the
        already-derived centerline's endpoints instead, which is more
        accurate for a narrow, strongly curved channel polygon where the MRR
        long axis won't track the bend well — but don't use it on a wide
        polygon without first confirming the centerline's own endpoints
        actually land near the true tips.
    progress_every : int
        Print a progress line every N features (0 to disable).
    skip_multipolygons : bool
        If True, log-and-skip any MultiPolygon feature instead of processing
        each part (parts would otherwise share the same id_col value with
        no way to distinguish them here — handle multipart features
        upstream, e.g. explode() them first with a distinguishing column,
        if you need per-part results).

    Returns
    -------
    dict with keys:
        "centerlines"    : GeoDataFrame [id_col, geometry]
        "sides"          : GeoDataFrame [id_col, side, length, geometry]
        "cross_sections" : GeoDataFrame [id_col, xs_id, station, ..., geometry]
        "status"         : DataFrame [id_col, stage, ok, error] — one row
                            per feature; stage is the last stage attempted
                            ("centerline", "sides", "cross_sections", or
                            "done"); check status[~status.ok] for failures.
    """
    if id_col not in polygons_gdf.columns:
        raise ValueError(f"id_col '{id_col}' not found in polygons_gdf columns: {list(polygons_gdf.columns)}")
    if polygons_gdf[id_col].duplicated().any():
        dupes = polygons_gdf.loc[polygons_gdf[id_col].duplicated(), id_col].tolist()
        raise ValueError(f"id_col '{id_col}' has duplicate values, e.g. {dupes[:5]}. Must be unique per feature.")
    if polygons_gdf.crs is None:
        print(
            "[warning] polygons_gdf has no CRS defined — outputs will also have "
            "crs=None. Set it explicitly (polygons_gdf = polygons_gdf.set_crs(...) "
            "or .to_crs(...)) before batch processing to avoid downstream SRID issues."
        )

    n = len(polygons_gdf)
    t0 = time.time()

    centerline_rows = []
    side_rows = []
    xs_rows = []
    status_rows = []

    for i, (_, row) in enumerate(polygons_gdf.iterrows()):
        fid = row[id_col]
        geom = row.geometry
        stage = "start"
        try:
            if geom is None or (not hasattr(geom, "is_empty")) or geom.is_empty:
                raise ValueError("geometry is missing or empty")

            if isinstance(geom, MultiPolygon):
                if skip_multipolygons:
                    status_rows.append({id_col: fid, "stage": "geometry_check", "ok": False, "error": "MultiPolygon skipped (skip_multipolygons=True)"})
                    continue
                # merge parts isn't safe in general (disjoint parts don't
                # have one centerline) -- take the largest part as a
                # reasonable default and warn via the status log
                parts = sorted(geom.geoms, key=lambda g: g.area, reverse=True)
                geom = parts[0]
                largest_note = f" (MultiPolygon: used largest of {len(parts)} parts)"
            else:
                largest_note = ""

            if not isinstance(geom, Polygon):
                raise TypeError(f"unsupported geometry type: {geom.geom_type}")

            stage = "centerline"
            # Derive the centerline UNTRIMMED first. extract_polygon_sides()
            # needs the true tip locations to find where to cut the polygon
            # boundary -- if it's handed an already-trimmed centerline
            # instead, that centerline's endpoint is just some interior
            # point, and on a wide polygon (floodplain-scale, not just the
            # channel) the nearest boundary point to an interior point is
            # often on a SIDE/bank rather than the actual end cap, which
            # produces a badly wrong side split (cutting through the middle
            # of the channel instead of at the reach's actual ends).
            cl_full = polygon_centerline(
                geom, segment_length=segment_length,
                simplify_tolerance=simplify_tolerance, trim_ends=0.0,
            )
            if trim_ends > 0:
                cl = substring(cl_full, trim_ends, cl_full.length - trim_ends)
            else:
                cl = cl_full
            centerline_rows.append({id_col: fid, "geometry": cl})

            stage = "sides"
            sides_centerline = cl_full if sides_cut_method == "centerline" else None
            sides = extract_polygon_sides(geom, centerline=sides_centerline, trim_ends=trim_ends)
            sides_gdf = polygon_sides_to_gdf(sides)
            sides_gdf.insert(0, id_col, fid)
            side_rows.append(sides_gdf)

            stage = "cross_sections"
            xs = generate_cross_sections(cl, spacing, half_width, tangent_step=tangent_step)
            if run_checks:
                xs = check_cross_sections(xs, cl, bearing_jump_threshold=bearing_jump_threshold)
            xs.insert(0, id_col, fid)
            xs_rows.append(xs)

            status_rows.append({id_col: fid, "stage": "done", "ok": True, "error": largest_note or None})

        except Exception as e:
            status_rows.append({
                id_col: fid, "stage": stage, "ok": False,
                "error": f"{type(e).__name__}: {e}",
            })

        if progress_every and (i + 1) % progress_every == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (n - (i + 1)) / rate if rate > 0 else float("nan")
            print(f"[batch] {i + 1}/{n} processed ({rate:.1f}/s, ~{eta:.0f}s remaining)")

    crs = polygons_gdf.crs

    def _empty_gdf(columns):
        return gpd.GeoDataFrame({c: [] for c in columns}, geometry="geometry", crs=crs)

    centerlines_gdf = (
        gpd.GeoDataFrame(centerline_rows, geometry="geometry", crs=crs)
        if centerline_rows else _empty_gdf([id_col, "geometry"])
    )
    sides_gdf_all = (
        gpd.GeoDataFrame(pd.concat(side_rows, ignore_index=True), geometry="geometry", crs=crs)
        if side_rows else _empty_gdf([id_col, "side", "length", "geometry"])
    )
    xs_gdf_all = (
        gpd.GeoDataFrame(pd.concat(xs_rows, ignore_index=True), geometry="geometry", crs=crs)
        if xs_rows else _empty_gdf([id_col, "xs_id", "station", "bearing", "geometry"])
    )
    status_df = pd.DataFrame(status_rows)

    n_ok = status_df["ok"].sum() if len(status_df) else 0
    n_fail = len(status_df) - n_ok
    elapsed = time.time() - t0
    print(f"[batch] done: {n_ok}/{n} succeeded, {n_fail} failed, in {elapsed:.1f}s")
    if n_fail:
        print("[batch] failure breakdown by stage:")
        print(status_df[~status_df["ok"]]["stage"].value_counts().to_string())

    return {
        "centerlines": centerlines_gdf,
        "sides": sides_gdf_all,
        "cross_sections": xs_gdf_all,
        "status": status_df,
    }


def write_batch_results(results: dict, insert_table_fn, schema: str, table_prefix: str = "", srid=None):
    """
    Convenience wrapper to push all three output GeoDataFrames through an
    insert_table-style function (signature: insert_table(gdf, table_name,
    schema, if_exists='replace', srid=None)), plus the status log via a
    plain-DataFrame write if your insert_table supports non-spatial tables
    (otherwise write status separately, e.g. status.to_csv()).

    table_prefix lets you namespace tables per run, e.g. table_prefix='xs_'
    -> 'xs_centerlines', 'xs_sides', 'xs_cross_sections'.
    """
    insert_table_fn(results["centerlines"], f"{table_prefix}centerlines", schema, if_exists="replace", srid=srid)
    insert_table_fn(results["sides"], f"{table_prefix}sides", schema, if_exists="replace", srid=srid)
    insert_table_fn(results["cross_sections"], f"{table_prefix}cross_sections", schema, if_exists="replace", srid=srid)