"""
bank_detection.py

Given cross sections (from generate_cross_sections.py / xs_pipeline.py) and a
DEM, detect the bank edge on each side of each cross section using several
independent geomorphic indicators, combine them into a consensus bank point
with an agreement/confidence flag, connect the consensus points across
cross sections into 3D bank lines, and rasterize an inundation extent from
the resulting along-reach water-surface profile.

Side convention
----------------
generate_cross_sections.py builds each XS as [p1, p2] where p2 is offset
+px,+py = rotate the local tangent +90 degrees (counterclockwise), which is
the standard math convention for "left" relative to the direction of travel
along the centerline. So: the p2 half of the line (offset > half_width) is
labeled "left", the p1 half (offset < half_width) is "right". This matches
polygon_sides.py's convention when the two centerlines agree in direction,
but isn't guaranteed identical station-by-station on a noisy centerline —
treat "left"/"right" here as approximate and verify against your own
convention before relying on it for reporting.

Indicators (per side, run on the elevation profile from the centerline
outward to the XS end)
-----------------------------------------------------------------------
  local_maxima     : first prominent peak outward from the channel
                      (scipy.signal.find_peaks) — a berm/bank crest.
  curvature         : point of maximum concave-down curvature (steepest
                      transition from rising bank slope to flatter ground)
                      — the classic "top of bank" knickpoint.
  terrace_step      : inner edge of the first flat bench (low local slope)
                      following a steeper rise — a bank terrace/step.
  hydraulic_geom    : stage at which top width vs. stage shows the sharpest
                      jump (sudden overbank widening = bankfull break),
                      found from the WHOLE cross section, then the station
                      on each side where the profile crosses that stage.
  regional_bankfull : optional. If you pass regional_depth_fn(discharge) ->
                      depth, the bankfull stage = channel invert + that
                      depth, station = crossing point on each side. Wire
                      this to your own regional hydraulic-geometry
                      regression (e.g. the USGS SIR 2020-5086 check already
                      used in bank_validation.py) — left unimplemented here
                      since the regression itself is project-specific.

Consensus per side = median of whichever indicators returned a value;
agreement = std of those values (in CRS/elevation units); flagged if fewer
than min_indicators agreed or spread exceeds max_spread.

Dependencies: rasterio, numpy, scipy, shapely (>=2.0), geopandas
"""

from __future__ import annotations

import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree
import shapely


def _station_along_line_indexed(line: LineString, points):
    """
    Vectorized, spatially-indexed equivalent of shapely.line_locate_point()
    for many points at once against one (possibly long/complex) line.

    shapely.line_locate_point() scans every vertex of `line` for each input
    point (no spatial index), which is fine for a few points but becomes the
    dominant cost when evaluating a whole DEM corridor (potentially millions
    of points) against a many-vertex centerline. This builds an STRtree over
    the line's individual segments (segment lookup is then O(log n) per
    point, vectorized via STRtree.nearest's array support) and computes the
    projection onto each point's nearest segment with plain numpy instead of
    a per-point GEOS call.

    Returns station (distance along `line` from its start) for each point.
    """
    coords = np.array(line.coords)
    seg_starts = coords[:-1]
    seg_ends = coords[1:]
    seg_vecs = seg_ends - seg_starts
    seg_lengths = np.hypot(seg_vecs[:, 0], seg_vecs[:, 1])
    cum_dist = np.concatenate([[0.0], np.cumsum(seg_lengths)])[:-1]  # station at start of each segment

    segments = [LineString([tuple(seg_starts[i]), tuple(seg_ends[i])]) for i in range(len(seg_starts))]
    tree = STRtree(segments)
    nearest_idx = tree.nearest(points)  # vectorized: array of segment indices, one per point

    pt_coords = shapely.get_coordinates(points)
    p0 = seg_starts[nearest_idx]
    v = seg_vecs[nearest_idx]
    vlen2 = np.where(seg_lengths[nearest_idx] > 0, seg_lengths[nearest_idx] ** 2, 1.0)
    t = np.einsum("ij,ij->i", pt_coords - p0, v) / vlen2
    t = np.clip(t, 0.0, 1.0)
    return cum_dist[nearest_idx] + t * seg_lengths[nearest_idx]


# ---------------------------------------------------------------- sampling

def sample_dem_along_line(dem, line: LineString, n_samples: int):
    """
    Sample elevation at n_samples evenly spaced points along `line`.

    Parameters
    ----------
    dem : rasterio dataset (already open) or a path to open one.
    line : LineString to sample along.
    n_samples : number of sample points (evenly spaced by distance).

    Returns
    -------
    dist : np.ndarray, distance along the line from its start (line CRS units)
    elev : np.ndarray, sampled elevation (nan where DEM nodata/out of bounds)
    """
    close_after = False
    if isinstance(dem, str):
        dem = rasterio.open(dem)
        close_after = True
    try:
        dist = np.linspace(0, line.length, n_samples)
        pts = [line.interpolate(d) for d in dist]
        coords = [(p.x, p.y) for p in pts]
        nodata = dem.nodata
        vals = np.array([v[0] for v in dem.sample(coords)], dtype=float)
        if nodata is not None:
            vals[vals == nodata] = np.nan
        return dist, vals
    finally:
        if close_after:
            dem.close()


# --------------------------------------------------------------- indicators

def _smooth(elev, sigma=1.0):
    valid = ~np.isnan(elev)
    if valid.sum() < 3:
        return elev
    out = elev.copy()
    out[~valid] = np.interp(np.flatnonzero(~valid), np.flatnonzero(valid), elev[valid])
    return gaussian_filter1d(out, sigma=sigma)


def indicator_local_maxima(dist, elev, prominence=0.5, skip_dist=0.0):
    """First prominent peak outward from the channel. Returns station or None."""
    mask = dist >= skip_dist
    if mask.sum() < 3:
        return None
    e = _smooth(elev[mask])
    peaks, props = find_peaks(e, prominence=prominence)
    if len(peaks) == 0:
        return None
    return float(dist[mask][peaks[0]])


def indicator_curvature(dist, elev, skip_dist=0.0):
    """Station of maximum concave-down curvature (steepest slope-to-flat transition)."""
    mask = dist >= skip_dist
    if mask.sum() < 5:
        return None
    d, e = dist[mask], _smooth(elev[mask])
    slope = np.gradient(e, d)
    curvature = np.gradient(slope, d)
    idx = np.argmin(curvature)  # most concave-down
    return float(d[idx])


def indicator_terrace_step(dist, elev, flat_slope_thresh=0.02, min_bench_width=3.0, skip_dist=0.0):
    """
    Inner edge of the first flat bench following a steeper rise.
    flat_slope_thresh: |slope| below this counts as 'flat' (elev units / dist units).
    min_bench_width: minimum contiguous flat run length to count as a bench.
    """
    mask = dist >= skip_dist
    if mask.sum() < 5:
        return None
    d, e = dist[mask], _smooth(elev[mask])
    slope = np.gradient(e, d)
    flat = np.abs(slope) < flat_slope_thresh
    steep_before = np.abs(slope) >= flat_slope_thresh

    i = 1
    n = len(d)
    while i < n:
        if flat[i] and steep_before[i - 1]:
            j = i
            while j < n and flat[j]:
                j += 1
            if d[j - 1] - d[i] >= min_bench_width:
                return float(d[i])
            i = j
        else:
            i += 1
    return None


def _xs_polygons(d, e):
    """
    Build the two geometries the hydraulic-depth method needs from a
    cross-section profile (station d, elevation e, both sorted, no NaNs):
      - polygon_xs: the profile closed into a "valley" polygon by adding two
        corners above the max elevation (one above each end) and connecting
        them across the top. Intersecting this with a below-stage box gives
        the correct wetted area and top width even for a multi-channel
        profile (secondary channels show up as separate MultiPolygon parts,
        whose areas/widths get summed).
      - border_xs: the raw (open) ground-profile LineString. Intersecting
        THIS with a below-stage box gives the wetted perimeter (length of
        actual ground surface underwater), as opposed to the water surface.
    """
    top = float(np.max(e)) + max(1.0, (np.max(e) - np.min(e)) * 0.1)
    closed_coords = [(d[0], top)] + list(zip(d, e)) + [(d[-1], top)]
    polygon_xs = shapely.geometry.Polygon(closed_coords)
    border_xs = LineString(list(zip(d, e)))
    return polygon_xs, border_xs


def indicator_hydraulic_geometry(dist_full, elev_full, n_stages=100, min_hyd_depth=None, smooth_sigma=2.0):
    """
    Whole-cross-section (both sides) bankfull break, via the hydraulic-depth
    local-maximum method (Nixon 1959 and others; ported/adapted from
    github.com/pierluigiderosa/BankFullDetection's BankElevationDetection.py,
    with its R-spline smoothing replaced by scipy and its geometry
    intersections kept as-is).

    Method: sweep a trial stage from the channel invert up to the profile's
    max elevation. At each stage, compute hydraulic depth = wetted area /
    top width (via proper polygon intersection against a closed "valley"
    polygon built from the profile, so a multi-channel cross section is
    handled correctly). As stage crosses the true bankfull elevation, top
    width jumps suddenly (flow spills onto the floodplain), so area/width
    actually DIPS right at that crossing even though area keeps rising --
    producing a local maximum in the hydraulic-depth-vs-stage curve exactly
    at bankfull. Returns the stage (elevation) of the first such local
    maximum whose hydraulic depth is >= min_hyd_depth, or None if no local
    maximum clears that bar (falls back to the top of the swept range, i.e.
    the profile's own max elevation, matching the original tool's fallback).

    n_stages : number of trial stages to sweep (100 in the original tool).
    min_hyd_depth : minimum hydraulic depth (elevation units) a candidate
        peak must reach to count, filtering out noise-scale local maxima
        near the channel invert. Defaults to 5% of the profile's total
        relief if not given.
    smooth_sigma : Gaussian smoothing applied to the hydraulic-depth curve
        before peak-finding (the original tool used an R smoothing spline;
        this is the same role, just a simpler smoother).
    """
    valid = ~np.isnan(elev_full)
    if valid.sum() < 5:
        return None
    d, e = dist_full[valid], elev_full[valid]
    order = np.argsort(d)
    d, e = d[order], e[order]

    lo, hi = float(np.min(e)), float(np.max(e))
    if hi <= lo:
        return None
    if min_hyd_depth is None:
        min_hyd_depth = 0.05 * (hi - lo)

    polygon_xs, border_xs = _xs_polygons(d, e)
    minx, _, maxx, _ = polygon_xs.bounds

    # sweep from just above invert to just below max elevation, as in the
    # original tool (avoids degenerate zero-width/zero-area edge stages)
    eps = (hi - lo) * 1e-3
    stages = np.linspace(lo + eps, hi - eps, n_stages)

    hyd_depth = np.full(n_stages, np.nan)
    for i, s in enumerate(stages):
        below_box = shapely.geometry.box(minx, lo - 1.0, maxx, s)
        wet_area_geom = polygon_xs.intersection(below_box)
        wt_line = LineString([(minx, s), (maxx, s)])
        wet_wt_geom = wt_line.intersection(polygon_xs)

        area = wet_area_geom.area
        top_width = wet_wt_geom.length
        if top_width > 0:
            hyd_depth[i] = area / top_width

    valid_hd = ~np.isnan(hyd_depth)
    if valid_hd.sum() < 5:
        return None

    hd_smooth = hyd_depth.copy()
    hd_smooth[valid_hd] = gaussian_filter1d(hyd_depth[valid_hd], sigma=smooth_sigma)

    peaks, _ = find_peaks(hd_smooth)
    candidates = [p for p in peaks if hd_smooth[p] >= min_hyd_depth]
    if candidates:
        return float(stages[candidates[0]])

    # no qualifying local max found -- fall back to the top of the swept
    # range, matching the original tool's behavior in this case
    return float(stages[-1])


def indicator_regional_bankfull(dist, elev, channel_invert_elev, discharge, regional_depth_fn):
    """Bankfull stage = invert + regional_depth_fn(discharge); station = first
    crossing outward from the channel. Returns station, or None."""
    if regional_depth_fn is None or discharge is None:
        return None
    depth = regional_depth_fn(discharge)
    if depth is None:
        return None
    target_stage = channel_invert_elev + depth
    return _first_crossing_station(dist, elev, target_stage)


def _first_crossing_station(dist, elev, target_elev):
    valid = ~np.isnan(elev)
    d, e = dist[valid], elev[valid]
    if len(d) < 2:
        return None
    order = np.argsort(d)
    d, e = d[order], e[order]
    for i in range(1, len(d)):
        if (e[i - 1] - target_elev) * (e[i] - target_elev) <= 0 and e[i] != e[i - 1]:
            frac = (target_elev - e[i - 1]) / (e[i] - e[i - 1])
            return float(d[i - 1] + frac * (d[i] - d[i - 1]))
    return None


# ------------------------------------------------------------- consensus

def _elev_at_station(dist, elev, station):
    valid = ~np.isnan(elev)
    d, e = dist[valid], elev[valid]
    if len(d) < 2 or station is None:
        return None
    order = np.argsort(d)
    return float(np.interp(station, d[order], e[order]))


def analyze_cross_section(
    xs_line: LineString,
    dem,
    *,
    n_samples: int = 200,
    skip_dist: float = 5.0,
    prominence: float = 0.5,
    flat_slope_thresh: float = 0.02,
    min_bench_width: float = 3.0,
    discharge: float | None = None,
    regional_depth_fn=None,
    min_indicators: int = 2,
    max_spread: float = 5.0,
    hyd_n_stages: int = 100,
    hyd_min_depth: float | None = None,
    hyd_smooth_sigma: float = 2.0,
) -> dict:
    """
    Run all indicators on one cross section and return a consensus bank
    point for each side.

    Returns
    -------
    dict with keys "left", "right", each a dict:
        {
          "station": consensus distance-from-center (None if no indicators fired),
          "elevation": elevation at that station,
          "n_indicators": how many indicators contributed,
          "spread": std of contributing indicator stations,
          "flagged": True if n_indicators < min_indicators or spread > max_spread,
          "methods": {method_name: station_or_None, ...},
        }
    plus "channel_invert_elev" and "bankfull_stage" (from hydraulic_geom,
    shared across both sides).

    hyd_n_stages, hyd_min_depth, hyd_smooth_sigma : passed straight through to
    indicator_hydraulic_geometry() (see that function's docstring) — tune
    these, not the old top-width-slope params, since that method has been
    replaced by the hydraulic-depth-local-maximum method.
    """
    half_width = xs_line.length / 2.0
    dist_full, elev_full = sample_dem_along_line(dem, xs_line, n_samples)

    channel_invert_elev = float(np.nanmin(elev_full)) if np.any(~np.isnan(elev_full)) else None
    bankfull_stage = indicator_hydraulic_geometry(
        dist_full, elev_full,
        n_stages=hyd_n_stages, min_hyd_depth=hyd_min_depth, smooth_sigma=hyd_smooth_sigma,
    )

    results = {}
    for side_name, side_mask in [
        ("right", dist_full <= half_width),
        ("left", dist_full >= half_width),
    ]:
        if side_name == "right":
            d = half_width - dist_full[side_mask]           # 0 at center, increasing outward
            order = np.argsort(d)
            d = d[order]
            e = elev_full[side_mask][order]
        else:
            d = dist_full[side_mask] - half_width
            order = np.argsort(d)
            d = d[order]
            e = elev_full[side_mask][order]

        methods = {}
        methods["local_maxima"] = indicator_local_maxima(d, e, prominence=prominence, skip_dist=skip_dist)
        methods["curvature"] = indicator_curvature(d, e, skip_dist=skip_dist)
        methods["terrace_step"] = indicator_terrace_step(d, e, flat_slope_thresh=flat_slope_thresh, min_bench_width=min_bench_width, skip_dist=skip_dist)
        methods["hydraulic_geom"] = _first_crossing_station(d, e, bankfull_stage) if bankfull_stage is not None else None
        methods["regional_bankfull"] = (
            indicator_regional_bankfull(d, e, channel_invert_elev, discharge, regional_depth_fn)
            if channel_invert_elev is not None else None
        )

        valid_vals = [v for v in methods.values() if v is not None]
        if valid_vals:
            consensus_station = float(np.median(valid_vals))
            spread = float(np.std(valid_vals)) if len(valid_vals) > 1 else 0.0
            consensus_elev = _elev_at_station(d, e, consensus_station)
            flagged = (len(valid_vals) < min_indicators) or (spread > max_spread)
        else:
            consensus_station = None
            consensus_elev = None
            spread = None
            flagged = True

        results[side_name] = {
            "station": consensus_station,
            "elevation": consensus_elev,
            "n_indicators": len(valid_vals),
            "spread": spread,
            "flagged": flagged,
            "methods": methods,
        }

    results["channel_invert_elev"] = channel_invert_elev
    results["bankfull_stage"] = bankfull_stage
    return results


def analyze_cross_sections(
    xs_gdf: gpd.GeoDataFrame,
    dem_path: str,
    *,
    id_col: str = "xs_id",
    station_col: str = "station",
    **kwargs,
) -> gpd.GeoDataFrame:
    """
    Run analyze_cross_section over every row of xs_gdf.

    Returns a GeoDataFrame, two rows per input XS (side='left'/'right'), with
    the consensus result PLUS every individual indicator's own offset and
    elevation logged as its own attribute column (so you can review/QA each
    method's result in GIS, not just the consensus):

        [id_col, station_col, side,
         bank_stn, elevation,                 -- consensus offset + elevation
         n_indctr, spread, flagged,           -- consensus confidence info
         off_lmax, el_lmax,                   -- local_maxima offset + elev
         off_curv, el_curv,                   -- curvature offset + elev
         off_terr, el_terr,                   -- terrace_step offset + elev
         off_hydr, el_hydr,                   -- hydraulic_geom offset + elev
         off_reg,  el_reg,                    -- regional_bankfull offset + elev
         invert_el, bkfl_stg,                 -- channel invert elev, bankfull stage (whole-XS, shared both sides)
         geometry (Point on the XS at the consensus bank station)]

    Column names are kept to <=10 chars so they survive a .shp write intact.
    A None/NaN offset or elevation means that indicator didn't fire for that
    side of that cross section.
    """
    method_cols = {
        "local_maxima": ("off_lmax", "el_lmax"),
        "curvature": ("off_curv", "el_curv"),
        "terrace_step": ("off_terr", "el_terr"),
        "hydraulic_geom": ("off_hydr", "el_hydr"),
        "regional_bankfull": ("off_reg", "el_reg"),
    }

    rows = []
    with rasterio.open(dem_path) as dem:
        for _, row in xs_gdf.iterrows():
            xs_line = row.geometry
            res = analyze_cross_section(xs_line, dem, **kwargs)
            half_width = xs_line.length / 2.0
            for side in ("left", "right"):
                s = res[side]
                pt = xs_line.interpolate(s["station"]) if s["station"] is not None else None
                # note: xs_line.interpolate uses distance-from-line-start, but our
                # per-side "station" is distance-from-center -- convert:
                if s["station"] is not None:
                    line_dist = (half_width - s["station"]) if side == "right" else (half_width + s["station"])
                    pt = xs_line.interpolate(line_dist)

                out_row = {
                    id_col: row[id_col],
                    station_col: row[station_col] if station_col in row else None,
                    "side": side,
                    "bank_stn": s["station"],
                    "elevation": s["elevation"],
                    "n_indctr": s["n_indicators"],
                    "spread": s["spread"],
                    "flagged": s["flagged"],
                }
                # per-method offset + elevation, so each indicator's result
                # is independently inspectable/QA-able in the GIS attribute table
                d_full, e_full = sample_dem_along_line(dem, xs_line, kwargs.get("n_samples", 200))
                if side == "right":
                    d_side = half_width - d_full[d_full <= half_width]
                    order = np.argsort(d_side)
                    d_side = d_side[order]
                    e_side = e_full[d_full <= half_width][order]
                else:
                    d_side = d_full[d_full >= half_width] - half_width
                    order = np.argsort(d_side)
                    d_side = d_side[order]
                    e_side = e_full[d_full >= half_width][order]
                for method_name, (off_col, el_col) in method_cols.items():
                    off_val = s["methods"].get(method_name)
                    out_row[off_col] = off_val
                    out_row[el_col] = _elev_at_station(d_side, e_side, off_val) if off_val is not None else None

                out_row["invert_el"] = res["channel_invert_elev"]
                out_row["bkfl_stg"] = res["bankfull_stage"]
                out_row["geometry"] = pt
                rows.append(out_row)
    out = gpd.GeoDataFrame(rows, geometry="geometry", crs=xs_gdf.crs)
    n_flagged = out["flagged"].sum()
    print(f"[analyze_cross_sections] {len(out)} bank points ({len(xs_gdf)} XS x 2 sides), {n_flagged} flagged (low agreement).")
    return out


# ------------------------------------------------------------- bank lines

def build_bank_lines(bank_points_gdf: gpd.GeoDataFrame, station_col: str = "station") -> dict:
    """
    Connect consensus bank points into 3D LineStrings (one per side), ordered
    by along-centerline station. Points with no consensus (geometry is None)
    are dropped with a printed warning.

    Returns {"left": LineString (z=elevation) or None, "right": ... }
    """
    out = {}
    for side in ("left", "right"):
        sub = bank_points_gdf[bank_points_gdf["side"] == side].copy()
        sub = sub[sub.geometry.notna() & sub["elevation"].notna()]
        sub = sub.sort_values(station_col)
        n_dropped = (bank_points_gdf["side"] == side).sum() - len(sub)
        if n_dropped:
            print(f"[build_bank_lines] {side}: dropped {n_dropped} XS with no consensus bank point.")
        if len(sub) < 2:
            print(f"[build_bank_lines] {side}: fewer than 2 valid points, cannot build a line.")
            out[side] = None
            continue
        coords = [(p.x, p.y, z) for p, z in zip(sub.geometry, sub["elevation"])]
        out[side] = LineString(coords)
    return out


# -------------------------------------------------------- inundation raster

def build_inundation_raster(
    xs_gdf: gpd.GeoDataFrame,
    bank_points_gdf: gpd.GeoDataFrame,
    centerline: LineString,
    dem_path: str,
    out_path: str,
    *,
    station_col: str = "station",
    corridor_buffer: float | None = None,
    wse_mode: str = "average",  # "average", "min" (conservative dry), "max" (conservative wet)
):
    """
    Build a boolean inundation raster (1=inundated, 0=dry) by comparing the
    DEM to a water-surface elevation interpolated along the centerline from
    each cross section's bank elevations.

    wse_mode:
        "average" : WSE at each XS = mean of left/right consensus elevations
        "min"     : lower of the two (conservative — smaller flood extent)
        "max"     : higher of the two (conservative — larger flood extent)

    Cells are only evaluated within `corridor_buffer` of the centerline
    (default: the cross sections' own half-width, from xs_gdf geometry
    lengths) to avoid extrapolating the WSE profile far past the surveyed
    reach. WSE beyond the first/last cross-section station is held constant
    (no extrapolation past the ends).
    """
    # per-XS WSE from left/right consensus elevations
    piv = bank_points_gdf.pivot_table(index=[bank_points_gdf.columns[0]], columns="side", values="elevation")
    stations = bank_points_gdf.drop_duplicates(bank_points_gdf.columns[0]).set_index(bank_points_gdf.columns[0])[station_col]
    piv = piv.join(stations)
    piv = piv.dropna(subset=["left", "right"], how="all")

    if wse_mode == "average":
        piv["wse"] = piv[["left", "right"]].mean(axis=1)
    elif wse_mode == "min":
        piv["wse"] = piv[["left", "right"]].min(axis=1)
    elif wse_mode == "max":
        piv["wse"] = piv[["left", "right"]].max(axis=1)
    else:
        raise ValueError(f"Unknown wse_mode: {wse_mode}")

    piv = piv.dropna(subset=["wse", station_col]).sort_values(station_col)
    if len(piv) < 2:
        raise ValueError("Need at least 2 cross sections with a valid WSE to build an inundation raster.")

    xs_stations = piv[station_col].to_numpy()
    xs_wse = piv["wse"].to_numpy()

    if corridor_buffer is None:
        corridor_buffer = float(np.median(xs_gdf.geometry.length) / 2.0)

    corridor = centerline.buffer(corridor_buffer)
    minx, miny, maxx, maxy = corridor.bounds

    with rasterio.open(dem_path) as dem:
        window = rasterio.windows.from_bounds(minx, miny, maxx, maxy, transform=dem.transform)
        window = window.round_lengths().round_offsets()
        dem_arr = dem.read(1, window=window)
        win_transform = dem.window_transform(window)
        nodata = dem.nodata
        crs = dem.crs

    rows, cols = dem_arr.shape

    # Rasterize the corridor polygon first (fast vectorized scanline fill) to
    # get the in-corridor mask, INSTEAD OF computing shapely.distance() for
    # every cell in the bounding box. For a sinuous reach, the bbox of the
    # buffered corridor is usually far larger than the corridor itself, so
    # doing the expensive per-cell line-distance/line-locate math over the
    # full bbox (as opposed to just the masked-in cells) can be extremely
    # slow -- e.g. tens of millions of cells for a several-mile reach at a
    # 1-2 ft DEM resolution, almost all of which get thrown away by the
    # corridor-distance check anyway.
    from rasterio.features import geometry_mask
    in_corridor = ~geometry_mask([corridor], out_shape=(rows, cols), transform=win_transform, invert=False)

    dem_flat = dem_arr.ravel().astype(float)
    mask_flat = in_corridor.ravel()
    valid = mask_flat.copy()
    if nodata is not None:
        valid &= dem_flat != nodata

    # only compute station (for WSE interpolation) on the masked-in subset
    valid_idx = np.flatnonzero(valid)
    row_idx_v, col_idx_v = np.unravel_index(valid_idx, (rows, cols))
    xs_coords, ys_coords = rasterio.transform.xy(win_transform, row_idx_v, col_idx_v)
    pts = shapely.points(np.array(xs_coords), np.array(ys_coords))
    cell_station = _station_along_line_indexed(centerline, pts)
    cell_wse = np.interp(cell_station, xs_stations, xs_wse)  # holds constant past ends

    inundated_flat = np.zeros(dem_flat.shape, dtype=bool)
    inundated_flat[valid_idx] = dem_flat[valid_idx] <= cell_wse

    out_arr = np.full(dem_flat.shape, 255, dtype=np.uint8)  # 255 = outside corridor / nodata
    out_arr[valid_idx] = np.where(inundated_flat[valid_idx], 1, 0).astype(np.uint8)
    out_arr = out_arr.reshape(rows, cols)
    inundated = inundated_flat  # keep downstream summary line unchanged

    with rasterio.open(
        out_path, "w", driver="GTiff", height=rows, width=cols, count=1,
        dtype=np.uint8, crs=crs, transform=win_transform, nodata=255,
        compress="deflate",
    ) as dst:
        dst.write(out_arr, 1)

    n_valid = valid.sum()
    n_inund = inundated.sum()
    print(f"[build_inundation_raster] wrote {out_path}: {n_inund}/{n_valid} valid cells inundated ({100*n_inund/max(n_valid,1):.1f}%)")
    return out_path