"""
polygon_centerline.py

Derive the centerline of a skinny/elongated polygon (a river/channel bank
polygon, a road or corridor footprint, etc.) using a vector Voronoi-skeleton
approach, rather than raster skeletonization. This avoids pixel-resolution
artifacts and works directly on vector geometry.

Algorithm
---------
1. Densify ("segmentize") the polygon boundary so Voronoi vertices land
   close together along the edge.
2. Compute the Voronoi diagram of the boundary points.
3. Keep only Voronoi edges whose midpoint falls inside the polygon — these
   approximate the medial axis / skeleton.
4. Build a graph from the kept edges; find the two skeleton endpoints
   (degree-1 nodes) with the greatest shortest-path distance between them.
   That path is the centerline.
5. Simplify (Douglas-Peucker) to smooth Voronoi zigzag.

Dependencies: shapely (>=2.0), scipy, networkx, numpy, geopandas (I/O only)
"""

from __future__ import annotations

import numpy as np
import networkx as nx
import shapely
from scipy.spatial import Voronoi
from shapely.geometry import LineString, Polygon, MultiPolygon, Point
from shapely.ops import unary_union


def _densify_boundary(polygon: Polygon, segment_length: float) -> np.ndarray:
    """Return an array of (x, y) points sampled along exterior + interior rings."""
    points = []

    def _sample_ring(coords):
        ring = LineString(coords)
        length = ring.length
        if length == 0:
            return
        n = max(int(np.ceil(length / segment_length)), 1)
        for d in np.linspace(0, length, n, endpoint=False):
            p = ring.interpolate(d)
            points.append((p.x, p.y))

    _sample_ring(polygon.exterior.coords)
    for interior in polygon.interiors:
        _sample_ring(interior.coords)

    return np.array(points)


def _default_segment_length(polygon: Polygon) -> float:
    """Pick a sensible default densify step: a small fraction of the
    polygon's shorter dimension (via its minimum rotated rectangle)."""
    minx, miny, maxx, maxy = polygon.bounds
    mrr = polygon.minimum_rotated_rectangle
    mrr_coords = np.array(mrr.exterior.coords)
    side_lengths = np.linalg.norm(np.diff(mrr_coords, axis=0), axis=1)
    short_side = np.sort(side_lengths)[0] if len(side_lengths) else min(maxx - minx, maxy - miny)
    return max(short_side / 8.0, 1e-6)


def polygon_centerline(
    polygon: Polygon,
    *,
    segment_length: float | None = None,
    simplify_tolerance: float | None = None,
    trim_ends: float = 0.0,
    max_length_ratio: float = 4.0,
    strict: bool = False,
) -> LineString:
    """
    Derive a single-thread centerline LineString through a polygon.

    Parameters
    ----------
    polygon : Polygon
        Must be a simple (single-part) polygon in a projected CRS.
        For a MultiPolygon, run this per-part (see `polygon_centerline_multi`).
    segment_length : float, optional
        Boundary densify step. Defaults to ~1/8 of the polygon's short-axis
        width (via its minimum rotated rectangle). Smaller = smoother/slower.
    simplify_tolerance : float, optional
        Douglas-Peucker tolerance applied to the final centerline to remove
        Voronoi zigzag. Defaults to segment_length / 2.
    trim_ends : float, default 0.0
        Distance to trim off each end of the derived centerline. Square-cut
        polygon ends (e.g. a channel polygon clipped at a straight cross
        section) produce an unreliable skeleton right at the end face — the
        medial axis kinks or bulges there since a flat end has its own
        "center". Trimming a short length (e.g. ~1-2x the half-width you'll
        use for cross sections) from each end avoids building cross sections
        off a spurious tail. Leave at 0 if your polygon ends taper naturally
        (e.g. following real bank convergence) rather than being clipped.
    max_length_ratio : float, default 4.0
        Sanity check: if the derived centerline's length exceeds this many
        times the straight-line distance between its two endpoints, it's a
        strong sign the skeleton is corrupted rather than just curvy — too
        coarse a segment_length starves the Voronoi diagram of boundary
        points, which lets spurious "bridge" edges cut straight across the
        polygon's interior; the longest-path search then routes through
        those, producing a long, jagged, wandering line instead of a smooth
        one that tracks the polygon's actual shape. A real, sharply
        meandering channel can legitimately have a high ratio, so this is a
        loose default (4x) — tighten it if you know your reaches are gentler,
        or raise it if you're deliberately processing very sinuous reaches.
    strict : bool, default False
        If True, raise ValueError when max_length_ratio is exceeded instead
        of just printing a warning.

    Returns
    -------
    LineString centerline. Raises ValueError if no valid skeleton could be
    built (e.g. polygon too small/degenerate relative to segment_length),
    or if strict=True and max_length_ratio is exceeded.
    """
    if not isinstance(polygon, Polygon):
        raise TypeError(f"Expected Polygon, got {type(polygon)}")

    if segment_length is None:
        segment_length = _default_segment_length(polygon)
    if simplify_tolerance is None:
        simplify_tolerance = segment_length / 2.0

    boundary_pts = _densify_boundary(polygon, segment_length)
    if len(boundary_pts) < 4:
        raise ValueError("Not enough boundary points to build a Voronoi diagram; polygon may be too small for segment_length.")

    vor = Voronoi(boundary_pts)

    # keep only finite ridges whose midpoint is inside the polygon.
    # Filtering is vectorized (one batched shapely.contains call over all
    # ridge midpoints) rather than looping polygon.contains() per ridge in
    # Python -- that loop is the dominant cost at small segment_length,
    # since ridge count grows with boundary point count.
    ridge_vertices = np.asarray(vor.ridge_vertices)
    finite_mask = ~np.any(ridge_vertices == -1, axis=1)
    finite_ridges = ridge_vertices[finite_mask]

    G = nx.Graph()
    if len(finite_ridges) > 0:
        p1s = vor.vertices[finite_ridges[:, 0]]
        p2s = vor.vertices[finite_ridges[:, 1]]
        mids = (p1s + p2s) / 2.0
        mid_points = shapely.points(mids[:, 0], mids[:, 1])
        inside_mask = shapely.contains(polygon, mid_points)

        kept_p1 = p1s[inside_mask]
        kept_p2 = p2s[inside_mask]
        lengths = np.hypot(kept_p1[:, 0] - kept_p2[:, 0], kept_p1[:, 1] - kept_p2[:, 1])

        for p1, p2, length in zip(kept_p1, kept_p2, lengths):
            G.add_edge(tuple(p1), tuple(p2), weight=float(length))

    if G.number_of_edges() == 0:
        raise ValueError(
            "No interior Voronoi edges found — try a smaller segment_length, "
            "or check the polygon isn't self-intersecting/degenerate."
        )

    # keep the largest connected component (skeleton noise can fragment small islands)
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()

    endpoints = [n for n, deg in G.degree() if deg == 1]
    if len(endpoints) < 2:
        # fall back: use the two nodes that are geometrically farthest apart
        nodes = list(G.nodes())
        endpoints = nodes

    # find the pair of endpoints with the longest shortest-path (the "diameter" path)
    best_path = None
    best_length = -1.0
    # limit combinations for large endpoint sets by sampling extremes along
    # the polygon's long axis first, then refine with a full search if small
    candidates = endpoints if len(endpoints) <= 40 else _prune_endpoints(endpoints, polygon)

    # Only run Dijkstra from the (few) candidate endpoints, not from every
    # node in the graph -- nx.all_pairs_dijkstra_path_length does the latter
    # and dominates runtime at small segment_length, where the skeleton graph
    # has thousands of nodes but usually only a handful of real endpoints.
    lengths = {a: nx.single_source_dijkstra_path_length(G, a, weight="weight") for a in candidates}
    for i, a in enumerate(candidates):
        for b in candidates[i + 1:]:
            if b not in lengths.get(a, {}):
                continue
            d = lengths[a][b]
            if d > best_length:
                best_length = d
                best_path = (a, b)

    if best_path is None:
        raise ValueError("Could not find a connecting path between skeleton endpoints.")

    path_nodes = nx.dijkstra_path(G, best_path[0], best_path[1], weight="weight")
    line = LineString(path_nodes)
    if simplify_tolerance > 0:
        line = line.simplify(simplify_tolerance, preserve_topology=False)

    straight_dist = Point(best_path[0]).distance(Point(best_path[1]))
    if straight_dist > 0:
        ratio = line.length / straight_dist
        if ratio > max_length_ratio:
            msg = (
                f"derived centerline length ({line.length:.1f}) is {ratio:.1f}x the "
                f"straight-line distance between its endpoints ({straight_dist:.1f}); "
                f"exceeds max_length_ratio={max_length_ratio}. This usually means "
                f"segment_length is too coarse for this polygon, producing a "
                f"corrupted/wandering skeleton rather than a real curvy centerline "
                f"— try a smaller segment_length. (If this reach is genuinely very "
                f"sinuous, raise max_length_ratio instead.)"
            )
            if strict:
                raise ValueError(msg)
            else:
                print(f"[warning] {msg}")

    if trim_ends > 0:
        length = line.length
        if trim_ends * 2 >= length:
            raise ValueError(
                f"trim_ends={trim_ends} would remove the entire centerline "
                f"(length={length:.1f}); reduce trim_ends."
            )
        # substring between trim_ends and length - trim_ends
        from shapely.ops import substring
        line = substring(line, trim_ends, length - trim_ends)

    return line


def _prune_endpoints(endpoints, polygon: Polygon, keep: int = 40):
    """For polygons with many skeleton endpoints (noisy boundary), keep the
    ones farthest from the polygon centroid to bound the pairwise search."""
    c = polygon.centroid
    endpoints = sorted(endpoints, key=lambda p: -Point(p).distance(c))
    return endpoints[:keep]


def polygon_centerline_multi(geom, **kwargs) -> LineString | list[LineString]:
    """Handle MultiPolygon input by running per-part; returns a list of
    LineStrings (one per polygon part) if given a MultiPolygon, else a
    single LineString."""
    if isinstance(geom, Polygon):
        return polygon_centerline(geom, **kwargs)
    if isinstance(geom, MultiPolygon):
        return [polygon_centerline(p, **kwargs) for p in geom.geoms]
    raise TypeError(f"Unsupported geometry type: {geom.geom_type}")


def centerline_from_file(path: str, crs=None, **kwargs):
    """Load a polygon (or multipolygon) from a file and derive its centerline(s)."""
    import geopandas as gpd

    gdf = gpd.read_file(path)
    if crs is not None:
        gdf = gdf.to_crs(crs)
    geom = unary_union(gdf.geometry.values)
    return polygon_centerline_multi(geom, **kwargs), gdf.crs


if __name__ == "__main__":
    import argparse
    import geopandas as gpd

    parser = argparse.ArgumentParser(description="Derive centerline(s) from a skinny polygon.")
    parser.add_argument("polygon", help="Path to polygon vector file (shp/gpkg/geojson)")
    parser.add_argument("--out", required=True, help="Output vector file path for the centerline(s)")
    parser.add_argument("--segment-length", type=float, default=None, help="Boundary densify step (CRS units)")
    parser.add_argument("--simplify-tolerance", type=float, default=None, help="Simplify tolerance (CRS units)")
    parser.add_argument("--trim-ends", type=float, default=0.0, help="Distance to trim off each end (CRS units); use for square-cut polygon ends")
    parser.add_argument("--crs", default=None, help="Optional EPSG code to reproject to before processing")
    args = parser.parse_args()

    result, crs = centerline_from_file(
        args.polygon, crs=args.crs,
        segment_length=args.segment_length,
        simplify_tolerance=args.simplify_tolerance,
        trim_ends=args.trim_ends,
    )
    lines = result if isinstance(result, list) else [result]
    out_gdf = gpd.GeoDataFrame({"part_id": range(len(lines))}, geometry=lines, crs=crs)
    out_gdf.to_file(args.out)
    print(f"Wrote {len(lines)} centerline part(s) to {args.out}")