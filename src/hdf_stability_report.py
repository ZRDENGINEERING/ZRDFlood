"""
hdf_stability_report.py

Scan a HEC-RAS plan HDF (.p*.hdf) results file for numerical-stability
problems:

  1. Courant number violations   -> Cell Courant > courant_threshold
  2. Water-surface error spikes  -> any dataset matching an "error"
     keyword (e.g. "Water Surface Error", "Cell WSE Error",
     "Cell Volume Error") whose magnitude exceeds ws_error_threshold
  3. High solver-iteration cells -> any dataset matching an "iteration"
     keyword (e.g. "Cell Last Iteration") whose value exceeds
     iteration_threshold, plus a domain-wide percentile view (many cells
     near the iteration cap at once is a better runtime predictor than a
     handful of isolated cells)
  4. Small cells                 -> analyze_hdf_stability(small_cell_area_
     threshold=...), from geometry (Geometry/2D Flow Areas/.../Cells
     Surface Area)
  5. A physically-grounded Courant cross-check -> analyze_velocity_courant():
     Cr = V*dt/sqrt(cell area) computed directly from velocity, timestep,
     and geometry, since a flat Courant threshold means something different
     on a tiny cell than a huge one
  6. Wet/dry cell oscillation    -> analyze_wet_dry_transitions(): cells that
     repeatedly flip wet/dry force extra iterations each flip
  7. Compute-time ground truth   -> read_compute_time_summary(): dumps
     whatever run/compute-time bookkeeping RAS wrote to Results/.../Summary,
     to correlate against the flags above
  8. Domain-wide computation summary -> analyze_computation_summary(): reads
     the scalar-per-timestep "Computations" group (Time Step, Total/Inner/
     Outer Iteration Number, Volume Error, Percent Active Cells, plus which
     cell/face was each timestep's worst offender). Unlike #1-3/5/6, THIS
     WORKS EVEN WITHOUT "Detailed" 2D output variables turned on in the
     plan — #1, #3, #5, and #6 all depend on per-cell datasets (Cell
     Courant, Cell Velocity - Velocity X/Y, Cell Last Iteration, a per-cell
     wet fraction) that only exist if specific boxes were checked under
     Unsteady Flow Analysis -> Options -> Output Options before the run. If
     those scans come back with zero datasets found, that's very likely why
     — check what's actually in your file (e.g. list the 2D Flow Area's
     "Unsteady Time Series" group) before assuming the keyword is wrong.

HEC-RAS HDF layouts vary a bit between versions and between steady/
unsteady 2D output, so instead of hard-coding one dataset path this
walks the whole file and pattern-matches dataset names. That makes it
resilient to different flow-area names / RAS versions.

Reporting is grouped by cell rather than emitting one line per
(timestep, cell) hit: a cell whose Courant number sits above threshold
for 80 consecutive timesteps produces ONE summary line (count + peak
value + first/last occurrence), not 80 near-identical lines. Raw,
ungrouped violations are still available on the returned report if you
need per-timestep detail.

A single physical event (e.g. a cell's peak Courant number) can also
show up in more than one HDF dataset — RAS often stores both a
per-timestep "Unsteady Time Series" block and a "Summary Output"
(max-over-run) block for the same quantity in the same 2D flow area.
Both match the same keyword and get scanned, so the same cell can be
flagged from two different datasets. Grouping is done per-dataset (see
`grouped_by_dataset` below); the report also lists which datasets were
scanned so you can see when this overlap is happening, and you can
narrow `error_keywords`/`iteration_keywords` if a summary block is
just noise.

Requires: h5py, numpy  (pip install h5py numpy)

Usage (CLI):
    python hdf_stability_report.py path/to/plan.p01.hdf --courant 2.0 --ws-error 0.05 --iteration 20

Usage (import):
    from hdf_stability_report import analyze_hdf_stability
    report = analyze_hdf_stability("plan.p01.hdf", courant_threshold=2.0,
                                    ws_error_threshold=0.05,
                                    iteration_threshold=20,
                                    log_path="stability_report.log")
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class Violation:
    dataset: str            # full HDF path of the dataset
    timestep: int           # row index in the dataset (time axis)
    cell: int                # column index (cell / element index)
    value: float
    area_name: str | None = None   # 2D flow area name, so you don't have to re-parse `dataset`
    time_label: str | None = None  # human-readable time, if available


@dataclass
class CellSummary:
    """One cell's violations within a single dataset, collapsed into a
    single record instead of one row per timestep."""
    dataset: str
    cell: int
    count: int
    max_value: float
    max_time_label: str | None
    first_timestep: int
    last_timestep: int
    first_time_label: str | None
    last_time_label: str | None
    area_name: str | None = None   # 2D flow area name, so you don't have to re-parse `dataset`


@dataclass
class SmallCell:
    """A single cell flagged for small surface area, from geometry (no time
    axis — this is a static mesh property, not a per-timestep result)."""
    dataset: str      # geometry dataset the area came from
    area_name: str    # 2D flow area name parsed out of the dataset path
    cell: int
    area: float


@dataclass
class StabilityReport:
    hdf_path: str
    courant_threshold: float
    ws_error_threshold: float
    iteration_threshold: float | None
    small_cell_area_threshold: float | None
    courant_datasets_scanned: list[str] = field(default_factory=list)
    error_datasets_scanned: list[str] = field(default_factory=list)
    iteration_datasets_scanned: list[str] = field(default_factory=list)
    geometry_area_datasets_scanned: list[str] = field(default_factory=list)
    courant_violations: list[Violation] = field(default_factory=list)
    ws_error_violations: list[Violation] = field(default_factory=list)
    iteration_violations: list[Violation] = field(default_factory=list)
    courant_by_cell: list[CellSummary] = field(default_factory=list)
    ws_error_by_cell: list[CellSummary] = field(default_factory=list)
    iteration_by_cell: list[CellSummary] = field(default_factory=list)
    small_cells: list[SmallCell] = field(default_factory=list)
    iteration_domain_percentile_series: dict = field(default_factory=dict)  # dataset -> np.ndarray, per-timestep percentile across cells

    def summary(self) -> str:
        lines = [
            f"HDF file: {self.hdf_path}",
            f"Courant threshold: {self.courant_threshold}",
            f"Water-surface error threshold: {self.ws_error_threshold}",
            f"Iteration threshold: {self.iteration_threshold}",
            f"Small-cell area threshold: {self.small_cell_area_threshold}",
            "",
            f"Courant datasets scanned ({len(self.courant_datasets_scanned)}):",
            *[f"  - {d}" for d in self.courant_datasets_scanned],
            f"Error datasets scanned ({len(self.error_datasets_scanned)}):",
            *[f"  - {d}" for d in self.error_datasets_scanned],
            f"Iteration datasets scanned ({len(self.iteration_datasets_scanned)}):",
            *[f"  - {d}" for d in self.iteration_datasets_scanned],
            f"Geometry area datasets scanned ({len(self.geometry_area_datasets_scanned)}):",
            *[f"  - {d}" for d in self.geometry_area_datasets_scanned],
            "",
            f"Courant violations: {len(self.courant_violations)} rows -> {len(self.courant_by_cell)} distinct cell/dataset hits",
            f"Water-surface error violations: {len(self.ws_error_violations)} rows -> {len(self.ws_error_by_cell)} distinct cell/dataset hits",
            f"Iteration violations: {len(self.iteration_violations)} rows -> {len(self.iteration_by_cell)} distinct cell/dataset hits",
            f"Small cells (by area): {len(self.small_cells)}",
            f"Iteration domain-wide percentile series computed for: {list(self.iteration_domain_percentile_series.keys())}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_datasets(
    h5file: h5py.File,
    keywords: list[str],
    path_contains: str | None = None,
) -> list[str]:
    """Return full paths of every dataset whose name contains any keyword
    (case-insensitive). If `path_contains` is given, only datasets whose
    full path also contains that substring (case-insensitive) are kept —
    used to scope the geometry area search to Geometry/2D Flow Areas/...
    and avoid matching unrelated "area"-named datasets elsewhere."""
    matches: list[str] = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            leaf = name.rsplit("/", 1)[-1].lower()
            if path_contains is not None and path_contains.lower() not in name.lower():
                return
            if any(kw in leaf for kw in keywords):
                matches.append(name)

    h5file.visititems(visitor)
    return matches


def _parse_flow_area_name(dataset_path: str) -> str:
    """Pull the 2D flow area name out of a dataset path like
    '.../2D Flow Areas/<AreaName>/Cells Surface Area' or
    '.../2D Flow Areas/<AreaName>/Cell Courant'. Falls back to the parent
    folder name if the path doesn't match that shape."""
    parts = dataset_path.split("/")
    for i, part in enumerate(parts):
        if part == "2D Flow Areas" and i + 1 < len(parts):
            return parts[i + 1]
    return parts[-2] if len(parts) >= 2 else dataset_path


def _find_time_labels(h5file: h5py.File, dataset_path: str, n_rows: int) -> list[str] | None:
    """Best-effort lookup of a matching 'Time' / 'Time Date Stamp' dataset
    living alongside `dataset_path`, to make violation timesteps readable.
    Returns None if nothing suitable is found."""
    parent = dataset_path.rsplit("/", 1)[0]
    candidates = [
        f"{parent}/Time Date Stamp",
        f"{parent}/Time",
        "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/Time Date Stamp",
        "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/Time",
    ]
    for path in candidates:
        if path in h5file:
            try:
                raw = h5file[path][()]
                if len(raw) != n_rows:
                    continue
                if raw.dtype.kind in ("S", "O", "U"):
                    return [v.decode() if isinstance(v, bytes) else str(v) for v in raw]
                # numeric (e.g. hours from start) — format as a plain number
                return [f"t={v:g}" for v in raw]
            except Exception:
                continue
    return None


def _scan_dataset(
    h5file: h5py.File,
    dataset_path: str,
    threshold: float,
    comparison: str = "gt",
) -> list[Violation]:
    """Load a dataset and return every (timestep, cell) entry that violates
    the threshold. `comparison` is 'gt' (value > threshold) or 'abs_gt'
    (abs(value) > threshold, used for water-surface *error*, which can be
    negative)."""
    data = h5file[dataset_path][()]
    data = np.asarray(data)

    if data.ndim == 1:
        # Some summary datasets are per-cell (max over run), no time axis.
        data = data[np.newaxis, :]

    if comparison == "abs_gt":
        mask = np.abs(data) > threshold
    else:
        mask = data > threshold

    if not mask.any():
        return []

    time_labels = _find_time_labels(h5file, dataset_path, data.shape[0])
    area_name = _parse_flow_area_name(dataset_path) if "2D Flow Areas" in dataset_path else None

    violations = []
    rows, cols = np.nonzero(mask)
    for t, c in zip(rows.tolist(), cols.tolist()):
        violations.append(
            Violation(
                dataset=dataset_path,
                timestep=t,
                cell=c,
                value=float(data[t, c]),
                area_name=area_name,
                time_label=time_labels[t] if time_labels else None,
            )
        )
    return violations


def _find_area_names(h5file: h5py.File, geometry_area_keywords: list[str]) -> list[str]:
    """Enumerate 2D flow area names by finding the geometry cell-area
    dataset in each one."""
    area_datasets = _find_datasets(h5file, geometry_area_keywords, path_contains="Geometry/2D Flow Areas")
    return [_parse_flow_area_name(d) for d in area_datasets]


def _get_cell_areas(h5file: h5py.File, area_name: str, geometry_area_keywords: list[str]) -> tuple[str, np.ndarray] | None:
    """Return (dataset_path, areas array) for a flow area's per-cell surface
    area, or None if no matching geometry dataset is found."""
    candidates = _find_datasets(h5file, geometry_area_keywords, path_contains=f"Geometry/2D Flow Areas/{area_name}")
    if not candidates:
        return None
    ds = candidates[0]
    return ds, np.asarray(h5file[ds][()]).reshape(-1)


def _get_cell_velocity_magnitude(h5file: h5py.File, area_name: str) -> tuple[str, np.ndarray] | None:
    """Return (label, velocity magnitude array shape [n_time, n_cells]) for a
    flow area's cell velocity, or None if nothing usable is found. Handles
    either a single magnitude dataset or separate X/Y component datasets."""
    prefix = f"2D Flow Areas/{area_name}"
    candidates = _find_datasets(h5file, ["velocity"], path_contains=prefix)
    # Prefer explicit X/Y component pairs so we can compute a true magnitude.
    x_ds = next((d for d in candidates if d.lower().endswith("velocity x")), None)
    y_ds = next((d for d in candidates if d.lower().endswith("velocity y")), None)
    if x_ds and y_ds:
        vx = np.asarray(h5file[x_ds][()])
        vy = np.asarray(h5file[y_ds][()])
        return f"{x_ds} & {y_ds} (magnitude)", np.sqrt(vx ** 2 + vy ** 2)

    # Otherwise fall back to a single "Cell Velocity"-type dataset, taking it
    # to already be a magnitude. Skip anything on cell faces, not cells.
    fallback = [d for d in candidates if "face" not in d.lower() and "cell" in d.lower()]
    if fallback:
        ds = fallback[0]
        return ds, np.asarray(h5file[ds][()])

    return None


def _get_time_step_series(h5file: h5py.File) -> tuple[str, np.ndarray] | None:
    """Return (dataset_path, dt array in seconds) for the run's per-output
    computation time step, if the HDF has one. This reflects the timestep at
    (or leading up to) each output write, not every internal computational
    substep — see module docstring caveats."""
    candidates = _find_datasets(h5file, ["time step"])
    # Prefer one that's NOT under a specific 2D Flow Area (i.e. the
    # run-global unsteady time series time step), but fall back to whatever
    # is found.
    global_candidates = [d for d in candidates if "2D Flow Areas" not in d]
    ds = global_candidates[0] if global_candidates else (candidates[0] if candidates else None)
    if ds is None:
        return None
    return ds, np.asarray(h5file[ds][()]).reshape(-1)


def _group_by_cell(violations: list[Violation]) -> list[CellSummary]:
    """Collapse a flat violation list into one CellSummary per
    (dataset, cell) pair, so a cell that violates for many consecutive
    timesteps produces a single summarized record instead of one row
    per timestep."""
    buckets: dict[tuple[str, int], list[Violation]] = defaultdict(list)
    for v in violations:
        buckets[(v.dataset, v.cell)].append(v)

    summaries: list[CellSummary] = []
    for (dataset, cell), vs in buckets.items():
        vs_sorted = sorted(vs, key=lambda v: v.timestep)
        peak = max(vs_sorted, key=lambda v: abs(v.value))
        first, last = vs_sorted[0], vs_sorted[-1]
        summaries.append(
            CellSummary(
                dataset=dataset,
                cell=cell,
                count=len(vs_sorted),
                max_value=peak.value,
                max_time_label=peak.time_label,
                first_timestep=first.timestep,
                last_timestep=last.timestep,
                first_time_label=first.time_label,
                last_time_label=last.time_label,
                area_name=first.area_name,
            )
        )
    # Worst offenders first.
    summaries.sort(key=lambda s: (-abs(s.max_value), -s.count))
    return summaries


@dataclass
class VelocityCourantReport:
    """Courant number computed directly from cell velocity, the run's
    timestep, and cell size (Cr = V * dt / sqrt(cell area)) — an independent
    cross-check against RAS's own reported Cell Courant, which uses an
    internal length-scale definition you can't see from the HDF. See the
    module docstring for caveats (dt is only sampled at the output interval;
    dx is approximated as sqrt(area), not RAS's exact face-normal spacing)."""
    hdf_path: str
    target_courant: float
    time_step_dataset: str | None
    areas_scanned: list[str] = field(default_factory=list)
    velocity_datasets_scanned: dict[str, str] = field(default_factory=dict)  # area -> dataset label
    cell_area_datasets_scanned: dict[str, str] = field(default_factory=dict)  # area -> dataset path
    violations: list[Violation] = field(default_factory=list)
    violations_by_cell: list[CellSummary] = field(default_factory=list)
    skipped_areas: dict[str, str] = field(default_factory=dict)  # area -> reason skipped

    def summary(self) -> str:
        lines = [
            f"HDF file: {self.hdf_path}",
            f"Target Courant (computed): {self.target_courant}",
            f"Time step dataset: {self.time_step_dataset}",
            f"Areas scanned: {self.areas_scanned}",
            f"Skipped areas: {self.skipped_areas}",
            f"Computed-Courant violations: {len(self.violations)} rows -> {len(self.violations_by_cell)} distinct cells",
        ]
        return "\n".join(lines)


def analyze_velocity_courant(
    hdf_path: str,
    target_courant: float = 1.0,
    geometry_area_keywords: list[str] | None = None,
    log_path: str | None = None,
    max_logged_per_dataset: int = 200,
) -> VelocityCourantReport:
    """
    Compute Cr = V * dt / sqrt(cell_area) per cell per output timestep, from
    raw cell velocity, the run's time step, and cell surface area — instead
    of trusting RAS's own reported "Cell Courant" dataset, which uses an
    internal cell length-scale you can't inspect from the HDF. Flags any
    cell/timestep where this computed value exceeds `target_courant`.

    A flat threshold like 2.0 doesn't mean the same thing on a 5 sq-ft cell
    as it does on a 5,000 sq-ft cell for the same velocity, since Courant
    scales with 1/sqrt(area); this gives you the actual physically grounded
    number instead.

    Caveats (see module docstring for the fuller version):
      - dt here is only available at the output-write interval, not every
        internal computational substep, same limitation as the reported
        Cell Courant dataset.
      - dx is approximated as sqrt(cell surface area). RAS's own internal
        Courant number likely uses something closer to a minimum
        center-to-face distance, so don't expect an exact match to the
        "Cell Courant" dataset — use this as an independent sanity check,
        and compare the two if you want to see how conservative RAS's own
        number is for your mesh.

    Returns
    -------
    VelocityCourantReport with the flagged cells, plus which areas were
    skipped and why (e.g. no velocity dataset found, no time-step dataset
    found) so a silent zero-violations result isn't mistaken for "all clear."
    """
    hdf_path = str(hdf_path)
    if geometry_area_keywords is None:
        geometry_area_keywords = ["cells surface area", "cell surface area", "surface area"]

    report = VelocityCourantReport(
        hdf_path=hdf_path,
        target_courant=target_courant,
        time_step_dataset=None,
    )

    if log_path is None:
        log_path = str(Path(hdf_path).with_suffix("")) + "_velocity_courant.log"

    logger = logging.getLogger(f"velocity_courant_{Path(hdf_path).stem}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)

    logger.info("HEC-RAS computed-Courant scan (from velocity, timestep, and cell size)")
    logger.info(f"Run at: {_dt.datetime.now().isoformat(timespec='seconds')}")
    logger.info(f"File:   {hdf_path}")
    logger.info(f"Target Courant: {target_courant}")
    logger.info("-" * 70)

    all_violations: list[Violation] = []

    with h5py.File(hdf_path, "r") as f:
        dt_result = _get_time_step_series(f)
        if dt_result is None:
            logger.info("No 'Time Step' dataset found anywhere in the file — cannot compute Courant. Aborting.")
            fh.close()
            return report
        dt_ds, dt_values = dt_result
        report.time_step_dataset = dt_ds
        logger.info(f"Time step dataset: {dt_ds} ({len(dt_values)} rows)")
        logger.info("")

        area_names = _find_area_names(f, geometry_area_keywords)
        report.areas_scanned = area_names

        for area_name in area_names:
            area_result = _get_cell_areas(f, area_name, geometry_area_keywords)
            if area_result is None:
                report.skipped_areas[area_name] = "no cell-area geometry dataset found"
                logger.info(f"[{area_name}] SKIPPED — no cell-area geometry dataset found")
                continue
            area_ds, areas = area_result
            report.cell_area_datasets_scanned[area_name] = area_ds

            vel_result = _get_cell_velocity_magnitude(f, area_name)
            if vel_result is None:
                report.skipped_areas[area_name] = "no cell velocity dataset found"
                logger.info(f"[{area_name}] SKIPPED — no cell velocity dataset found")
                continue
            vel_label, vel = vel_result
            report.velocity_datasets_scanned[area_name] = vel_label

            n_time = vel.shape[0]
            n_cells = vel.shape[1] if vel.ndim > 1 else vel.shape[0]
            if vel.ndim == 1:
                vel = vel[np.newaxis, :]

            if len(areas) != vel.shape[1]:
                report.skipped_areas[area_name] = (
                    f"cell count mismatch: {len(areas)} areas vs {vel.shape[1]} velocity columns"
                )
                logger.info(f"[{area_name}] SKIPPED — {report.skipped_areas[area_name]}")
                continue

            dt_use = dt_values[:n_time] if len(dt_values) >= n_time else None
            if dt_use is None:
                report.skipped_areas[area_name] = (
                    f"time step series too short ({len(dt_values)} rows) for velocity series ({n_time} rows)"
                )
                logger.info(f"[{area_name}] SKIPPED — {report.skipped_areas[area_name]}")
                continue

            dx = np.sqrt(areas)
            dx_safe = np.where(dx > 0, dx, np.nan)  # avoid divide-by-zero on degenerate cells
            computed = vel * dt_use[:, np.newaxis] / dx_safe[np.newaxis, :]

            mask = computed > target_courant
            time_labels = _find_time_labels(f, dt_ds, n_time)

            rows, cols = np.nonzero(np.nan_to_num(mask, nan=False))
            area_violations = [
                Violation(
                    dataset=f"computed_courant/{area_name}",
                    timestep=int(t),
                    cell=int(c),
                    value=float(computed[t, c]),
                    area_name=area_name,
                    time_label=time_labels[t] if time_labels else None,
                )
                for t, c in zip(rows.tolist(), cols.tolist())
            ]
            all_violations.extend(area_violations)

            grouped = _group_by_cell(area_violations)
            logger.info(
                f"[{area_name}] velocity={vel_label} | cell-area={area_ds} | "
                f"{len(area_violations)} row(s) -> {len(grouped)} cell(s) flagged"
            )
            for g in grouped[:max_logged_per_dataset]:
                when_first = g.first_time_label if g.first_time_label else f"timestep {g.first_timestep}"
                when_last = g.last_time_label if g.last_time_label else f"timestep {g.last_timestep}"
                span = when_first if g.first_timestep == g.last_timestep else f"{when_first} .. {when_last}"
                logger.info(
                    f"    cell {g.cell} | area={areas[g.cell]:.2f} | {g.count} timestep(s) | "
                    f"peak computed Cr = {g.max_value:.3f} | {span}"
                )
            if len(grouped) > max_logged_per_dataset:
                logger.info(f"    ... {len(grouped) - max_logged_per_dataset} more cell(s) not shown")
            logger.info("")

    report.violations = all_violations
    report.violations_by_cell = _group_by_cell(all_violations)

    logger.info("-" * 70)
    logger.info(f"TOTAL computed-Courant violations: {len(report.violations)} rows / {len(report.violations_by_cell)} cells")
    fh.close()

    return report


# ---------------------------------------------------------------------------
# Compute-time summary (ground truth to correlate the above against)
# ---------------------------------------------------------------------------

def read_compute_time_summary(hdf_path: str, log_path: str | None = None) -> dict:
    """
    Pull whatever run-time / compute-time bookkeeping RAS wrote into
    Results/.../Summary for this plan — things like total computation time,
    broken out by phase. Field names vary by RAS version, so rather than
    hard-coding one path, this dumps every attribute (and any small,
    "time"-named dataset) found under any group whose path contains
    "Summary" beneath "Results". Treat the returned dict as ground truth to
    correlate the other scans in this module against: if a run took an
    unusually long time, cross-reference which of the stability/courant/
    iteration/wet-dry flags were present in that same run.

    Returns
    -------
    dict mapping "<group path>::<attr or dataset name>" -> value.
    """
    hdf_path = str(hdf_path)
    findings: dict = {}

    logger = logging.getLogger(f"compute_time_{Path(hdf_path).stem}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    if log_path is None:
        log_path = str(Path(hdf_path).with_suffix("")) + "_compute_time.log"
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)

    logger.info("HEC-RAS compute-time summary")
    logger.info(f"File: {hdf_path}")
    logger.info("-" * 70)

    with h5py.File(hdf_path, "r") as f:
        summary_groups: list[str] = []

        def visitor(name, obj):
            if isinstance(obj, h5py.Group) and "summary" in name.lower() and name.lower().startswith("results"):
                summary_groups.append(name)

        f.visititems(visitor)

        if not summary_groups:
            logger.info("No group under Results/ with 'Summary' in its name was found.")
            logger.info("RAS version/output-block naming may differ — try inspecting the file's")
            logger.info("structure directly (e.g. list(f['Results/Unsteady'].keys())).")

        for group_path in summary_groups:
            grp = f[group_path]
            logger.info(f"[{group_path}]")

            for attr_name, attr_val in grp.attrs.items():
                key = f"{group_path}::{attr_name}"
                findings[key] = attr_val
                logger.info(f"  attr  {attr_name} = {attr_val}")

            for child_name, child in grp.items():
                if isinstance(child, h5py.Dataset) and child.size <= 64:
                    try:
                        val = child[()]
                    except Exception:
                        continue
                    key = f"{group_path}/{child_name}"
                    findings[key] = val
                    tag = "time" if "time" in child_name.lower() else "data"
                    logger.info(f"  {tag}  {child_name} = {val}")
            logger.info("")

    fh.close()
    return findings


# ---------------------------------------------------------------------------
# Wet/dry cell transitions
# ---------------------------------------------------------------------------

@dataclass
class WetDryCell:
    dataset: str
    area_name: str
    cell: int
    transitions: int


@dataclass
class WetDryReport:
    hdf_path: str
    min_transitions: int
    datasets_scanned: list[str] = field(default_factory=list)
    cells: list[WetDryCell] = field(default_factory=list)  # sorted worst-first

    def summary(self) -> str:
        lines = [
            f"HDF file: {self.hdf_path}",
            f"Min transitions to flag: {self.min_transitions}",
            f"Wet/dry datasets scanned ({len(self.datasets_scanned)}):",
            *[f"  - {d}" for d in self.datasets_scanned],
            f"Cells flagged for repeated wet/dry transitions: {len(self.cells)}",
        ]
        return "\n".join(lines)


def analyze_wet_dry_transitions(
    hdf_path: str,
    wet_keywords: list[str] | None = None,
    transition_threshold: float = 0.5,
    min_transitions: int = 5,
    log_path: str | None = None,
    max_logged_per_dataset: int = 200,
) -> WetDryReport:
    """
    Flag cells that repeatedly flip between wet and dry over the run. Each
    flip forces extra nonlinear iterations that timestep/iteration/Courant
    counts alone don't directly show, and a cell oscillating wet/dry many
    times is a different (and often worse) runtime cost than one that's
    simply wet with a high Courant number throughout.

    Looks for a per-cell "wet fraction"-style dataset (RAS naming varies by
    version — tries "percent wet", "cell wet", "wetted" by default) under
    each 2D Flow Area. Values are auto-scaled (handles both 0-1 fractions
    and 0-100 percentages) and a "transition" is counted every time the
    series crosses `transition_threshold` (as a fraction, i.e. 0.5 = 50%
    wet) between consecutive output rows.

    Parameters
    ----------
    hdf_path : path to the plan HDF
    wet_keywords : dataset-name substrings identifying wet-fraction data.
        Defaults to ["percent wet", "cell wet", "wetted"].
    transition_threshold : fraction (0-1) crossing point that counts as a
        wet<->dry flip. Default 0.5.
    min_transitions : only cells with at least this many transitions are
        included in the report (keeps a normally-wet cell with one settle-in
        transition off the list).
    log_path : defaults to "<hdf stem>_wet_dry.log".
    max_logged_per_dataset : cap on how many flagged cells get individually
        logged per dataset.

    Returns
    -------
    WetDryReport. If `datasets_scanned` is empty, no matching dataset was
    found for your RAS version/output settings — check the file's actual
    dataset names and pass a custom `wet_keywords` list.
    """
    hdf_path = str(hdf_path)
    if wet_keywords is None:
        wet_keywords = ["percent wet", "cell wet", "wetted"]

    report = WetDryReport(hdf_path=hdf_path, min_transitions=min_transitions)

    if log_path is None:
        log_path = str(Path(hdf_path).with_suffix("")) + "_wet_dry.log"

    logger = logging.getLogger(f"wet_dry_{Path(hdf_path).stem}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)

    logger.info("HEC-RAS wet/dry cell transition scan")
    logger.info(f"File: {hdf_path}")
    logger.info(f"Transition threshold (fraction wet): {transition_threshold}")
    logger.info(f"Minimum transitions to flag: {min_transitions}")
    logger.info("-" * 70)

    with h5py.File(hdf_path, "r") as f:
        wet_datasets = _find_datasets(f, wet_keywords, path_contains="2D Flow Areas")
        report.datasets_scanned = wet_datasets
        logger.info(f"Wet/dry datasets found ({len(wet_datasets)}):")
        for d in wet_datasets:
            logger.info(f"  {d}")
        if not wet_datasets:
            logger.info("  (none — try passing custom wet_keywords based on this file's actual dataset names)")
        logger.info("")

        all_cells: list[WetDryCell] = []

        for ds in wet_datasets:
            area_name = _parse_flow_area_name(ds)
            data = np.asarray(f[ds][()])
            if data.ndim == 1:
                data = data[np.newaxis, :]

            # Normalize to a 0-1 fraction if this looks like a percentage.
            threshold = transition_threshold * 100.0 if np.nanmax(data) > 1.5 else transition_threshold

            above = data > threshold
            # Count sign changes along the time axis for each cell.
            flips = np.diff(above.astype(np.int8), axis=0) != 0
            transition_counts = flips.sum(axis=0)

            flagged_cells = np.nonzero(transition_counts >= min_transitions)[0]
            ds_cells = [
                WetDryCell(dataset=ds, area_name=area_name, cell=int(c), transitions=int(transition_counts[c]))
                for c in flagged_cells
            ]
            ds_cells.sort(key=lambda w: -w.transitions)
            all_cells.extend(ds_cells)

            logger.info(f"[{area_name}] {ds}: {len(ds_cells)} cell(s) with >= {min_transitions} transitions")
            for w in ds_cells[:max_logged_per_dataset]:
                logger.info(f"    cell {w.cell} | {w.transitions} transitions")
            if len(ds_cells) > max_logged_per_dataset:
                logger.info(f"    ... {len(ds_cells) - max_logged_per_dataset} more not shown")
            logger.info("")

        all_cells.sort(key=lambda w: -w.transitions)
        report.cells = all_cells

    logger.info("-" * 70)
    logger.info(f"TOTAL cells flagged: {len(report.cells)}")
    fh.close()

    return report


# ---------------------------------------------------------------------------
# Domain-level computation summary (works even without "Detailed" 2D output)
# ---------------------------------------------------------------------------
#
# HEC-RAS only writes the dense per-cell diagnostic datasets used above
# (Cell Courant, Cell Velocity - Velocity X/Y, Cell Last Iteration, a
# per-cell wet fraction) when specific 2D Output Variables are checked in
# the plan's Unsteady Flow Analysis -> Options -> Output Options before the
# run. Without that, a 2D flow area's "Unsteady Time Series" only has:
#   Water Surface        (per cell, [n_time, n_cells])
#   Face Velocity         (per face, [n_time, n_faces])
#   Computations/         domain-wide SCALAR time series (one value per
#                         timestep, not per cell):
#       Time Step, Total/Inner/Outer Iteration Number, Volume, Volume Error,
#       Percent Active Cells, Max/Min Water Surface (+ which Cell),
#       Inner Max Volume Residual (+ which Cell), Outer Max Water Surface
#       Correction (+ which Cell), Max Face Velocity (+ which Face)
#
# The "Computations" scalars ARE still a strong runtime-cost signal —
# Total Iteration Number in particular is close to a direct proxy for
# solver work per timestep — and the paired "* Cell"/"* Face" index
# datasets tell you which cell/face was the domain's worst offender at
# each timestep, which lets you find recurring problem cells without a
# dense per-cell array at all.

_COMPUTATION_SCALAR_FIELDS = [
    "Time Step",
    "Total Iteration Number",
    "Inner Iteration Number",
    "Outer Iteration Number",
    "Volume",
    "Volume Error",
    "Percent Active Cells",
    "Max Water Surface",
    "Min Water Surface",
    "Inner Max Volume Residual",
    "Outer Max Water Surface Correction",
    "Max Face Velocity",
]

# Maps a value field to the companion dataset naming the cell/face it
# occurred at, if RAS wrote one alongside it.
_COMPUTATION_LOCATION_FIELDS = {
    "Max Water Surface": "Max Water Surface Cell",
    "Min Water Surface": "Min Water Surface Cell",
    "Inner Max Volume Residual": "Inner Max Volume Residual Cell",
    "Outer Max Water Surface Correction": "Outer Max Water Surface Correction Cell",
    "Max Face Velocity": "Max Face Velocity Face",
}


@dataclass
class WorstOffender:
    field: str          # value field this location came from, e.g. "Inner Max Volume Residual"
    location_field: str  # the "* Cell"/"* Face" dataset name
    index: int           # cell or face index
    count: int           # how many timesteps this index was the worst offender
    max_value: float     # peak value of `field` recorded at this index
    area_name: str | None = None  # which 2D flow area this came from


@dataclass
class AreaComputationSummary:
    """Computation-summary results for a single 2D flow area."""
    area_name: str
    computations_path: str
    series: dict = field(default_factory=dict)             # field name -> np.ndarray
    flagged_timesteps: dict = field(default_factory=dict)   # field name -> list[(timestep, value, time_label)]
    worst_offenders: list[WorstOffender] = field(default_factory=list)  # sorted by count desc


@dataclass
class ComputationSummaryReport:
    hdf_path: str
    areas: dict[str, AreaComputationSummary] = field(default_factory=dict)  # area name -> its summary
    skipped_areas: dict[str, str] = field(default_factory=dict)  # area name -> reason skipped

    def summary(self) -> str:
        lines = [
            f"HDF file: {self.hdf_path}",
            f"Areas analyzed: {list(self.areas.keys())}",
            f"Areas skipped: {self.skipped_areas}",
        ]
        for name, a in self.areas.items():
            lines.append(f"-- {name} --")
            lines.append(f"  Series read: {list(a.series.keys())}")
            lines.append(f"  Flagged fields: {[(k, len(v)) for k, v in a.flagged_timesteps.items()]}")
            lines.append(f"  Worst offenders tracked: {len(a.worst_offenders)}")
        return "\n".join(lines)


def _analyze_computation_area(
    f: h5py.File,
    base: str,
    area_name: str,
    iteration_threshold: float | None,
    timestep_drop_fraction: float | None,
    volume_error_threshold: float | None,
    logger: logging.Logger,
    max_logged: int,
) -> AreaComputationSummary | None:
    """Run the computation-summary analysis for one 2D flow area. Returns
    None if this area has no Computations group under `base` (e.g. not a
    2D area, or this output block doesn't carry it for this file)."""
    comp_path = f"{base}/{area_name}/Computations"
    if comp_path not in f:
        return None

    result = AreaComputationSummary(area_name=area_name, computations_path=comp_path)
    logger.info(f"Area: {area_name}")
    logger.info(f"Computations group: {comp_path}")
    logger.info("")

    comp_grp = f[comp_path]
    series: dict[str, np.ndarray] = {}
    for field_name in _COMPUTATION_SCALAR_FIELDS:
        if field_name in comp_grp:
            series[field_name] = np.asarray(comp_grp[field_name][()]).reshape(-1)
    result.series = series

    n_time = len(next(iter(series.values()))) if series else 0
    time_labels = _find_time_labels(f, comp_path, n_time) if n_time else None

    logger.info(f"Fields found: {list(series.keys())}")
    logger.info(f"Timesteps: {n_time}")
    logger.info("")

    # --- Total Iteration Number ---------------------------------------
    if iteration_threshold is not None and "Total Iteration Number" in series:
        vals = series["Total Iteration Number"]
        idx = np.nonzero(vals > iteration_threshold)[0]
        rows = [(int(t), float(vals[t]), time_labels[t] if time_labels else None) for t in idx.tolist()]
        result.flagged_timesteps["Total Iteration Number"] = rows
        logger.info(f"[ITERATION] Total Iteration Number > {iteration_threshold}: {len(rows)} timestep(s)")
        for t, v, label in sorted(rows, key=lambda r: -r[1])[:max_logged]:
            logger.info(f"    {label or f'timestep {t}'} | {v:.0f} iterations")
        logger.info("")

    # --- Time Step drops -----------------------------------------------
    if timestep_drop_fraction is not None and "Time Step" in series:
        dt = series["Time Step"]
        nominal = float(np.nanmax(dt)) if len(dt) else 0.0
        cutoff = nominal * timestep_drop_fraction
        idx = np.nonzero(dt < cutoff)[0]
        rows = [(int(t), float(dt[t]), time_labels[t] if time_labels else None) for t in idx.tolist()]
        result.flagged_timesteps["Time Step"] = rows
        logger.info(
            f"[TIME STEP] below {timestep_drop_fraction:.0%} of run max ({nominal:.3g}s, cutoff {cutoff:.3g}s): "
            f"{len(rows)} timestep(s)"
        )
        for t, v, label in sorted(rows, key=lambda r: r[1])[:max_logged]:
            logger.info(f"    {label or f'timestep {t}'} | dt = {v:.3g}s")
        logger.info("")

    # --- Volume Error ----------------------------------------------------
    if volume_error_threshold is not None and "Volume Error" in series:
        ve = series["Volume Error"]
        idx = np.nonzero(np.abs(ve) > volume_error_threshold)[0]
        rows = [(int(t), float(ve[t]), time_labels[t] if time_labels else None) for t in idx.tolist()]
        result.flagged_timesteps["Volume Error"] = rows
        logger.info(f"[VOLUME ERROR] |value| > {volume_error_threshold}: {len(rows)} timestep(s)")
        for t, v, label in sorted(rows, key=lambda r: -abs(r[1]))[:max_logged]:
            logger.info(f"    {label or f'timestep {t}'} | volume error = {v:.4g}")
        logger.info("")

    # --- Worst-offender tally (which cell/face keeps showing up) -------
    offenders: list[WorstOffender] = []
    for value_field, location_field in _COMPUTATION_LOCATION_FIELDS.items():
        if value_field not in comp_grp or location_field not in comp_grp:
            continue
        values = np.asarray(comp_grp[value_field][()]).reshape(-1)
        locations = np.asarray(comp_grp[location_field][()]).reshape(-1)
        tally: dict[int, list[float]] = defaultdict(list)
        for loc, val in zip(locations.tolist(), values.tolist()):
            tally[int(loc)].append(float(val))
        for idx_val, vals_list in tally.items():
            offenders.append(
                WorstOffender(
                    field=value_field,
                    location_field=location_field,
                    index=idx_val,
                    count=len(vals_list),
                    max_value=max(vals_list, key=abs),
                    area_name=area_name,
                )
            )
    offenders.sort(key=lambda o: -o.count)
    result.worst_offenders = offenders

    logger.info("[WORST OFFENDERS] cells/faces most frequently named as the domain's extreme value")
    for o in offenders[:max_logged]:
        logger.info(
            f"    {o.location_field} = {o.index} | named worst {o.count} time(s) | "
            f"peak {o.field} = {o.max_value:.4g}"
        )
    logger.info("")

    return result


def analyze_computation_summary(
    hdf_path: str,
    area_names: list[str] | None = None,
    iteration_threshold: float | None = 40,
    timestep_drop_fraction: float | None = 0.5,
    volume_error_threshold: float | None = None,
    log_path: str | None = None,
    max_logged: int = 200,
) -> ComputationSummaryReport:
    """
    Read the domain-wide "Computations" scalar time series RAS always
    writes for each 2D flow area (even without "Detailed" output enabled)
    and flag runtime-relevant behavior:

      - Total Iteration Number above `iteration_threshold` — direct solver-
        effort spikes, a close proxy for where wall-clock time went.
      - Time Step dropping below `timestep_drop_fraction` of the run's own
        max Time Step — RAS's adaptive timestep cutting back, meaning more
        computational steps were needed to cover the same simulated time.
      - Volume Error whose absolute value exceeds `volume_error_threshold`
        (pass None, the default, to skip — units are whatever your geometry
        uses, so there's no universal default).

    It also tallies the "* Cell"/"* Face" companion datasets (e.g. "Inner
    Max Volume Residual Cell") to find which cell/face recurs most often as
    the domain's worst offender — the closest thing to "problem cells" you
    can get without per-cell arrays.

    Parameters
    ----------
    hdf_path : path to the plan HDF
    area_names : which 2D flow area(s) to read. If None (the default),
        every 2D flow area found in the file is scanned — don't assume a
        model has just one.
    iteration_threshold : flag timesteps with Total Iteration Number above
        this. Pass None to skip.
    timestep_drop_fraction : flag timesteps where Time Step is below this
        fraction of the run's own max Time Step. Pass None to skip.
    volume_error_threshold : flag timesteps where |Volume Error| exceeds
        this. Pass None (default) to skip — there's no universal value,
        since units depend on your geometry.
    log_path : defaults to "<hdf stem>_computation_summary.log".
    max_logged : cap on how many flagged rows / worst offenders get logged
        per field, per area.

    Returns
    -------
    ComputationSummaryReport with one AreaComputationSummary per area found
    (raw series for your own plotting/correlation against actual run time,
    flagged timesteps, and the worst-offender tally), plus which areas (if
    any) were skipped and why.
    """
    hdf_path = str(hdf_path)
    report = ComputationSummaryReport(hdf_path=hdf_path)

    if log_path is None:
        log_path = str(Path(hdf_path).with_suffix("")) + "_computation_summary.log"

    logger = logging.getLogger(f"computation_summary_{Path(hdf_path).stem}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)

    logger.info("HEC-RAS domain-level computation summary")
    logger.info(f"File: {hdf_path}")
    logger.info("-" * 70)

    # "Computations" scalar series have shown up under more than one output
    # block depending on what was checked in Output Options before the run —
    # confirmed so far under "Base Output" (e.g. the USM file). Some files
    # also carry a "Computation Block", which uses a slightly different path
    # (no "Unsteady Time Series" segment) and was only confirmed so far to
    # hold the sparse "2D Iterations" event log rather than "Computations" —
    # but rather than assume that always holds, check it here too so a file
    # that *does* put "Computations" there isn't silently skipped.
    candidate_bases = [
        ("Base Output", "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas"),
        ("Computation Block", "Results/Unsteady/Output/Output Blocks/Computation Block/2D Flow Areas"),
    ]

    with h5py.File(hdf_path, "r") as f:
        available_bases = [(block_name, path) for block_name, path in candidate_bases if path in f]
        if not available_bases:
            logger.info(
                "Neither 'Base Output' nor 'Computation Block' found under Output Blocks — "
                "is this a plan results HDF?"
            )
            fh.close()
            return report

        # Collect the union of area names across whichever blocks exist, so
        # an area only listed under one block still gets checked against both.
        if area_names is not None:
            names = list(area_names)
        else:
            names = []
            for _, path in available_bases:
                for n in f[path].keys():
                    if n not in names:
                        names.append(n)
        if not names:
            logger.info("No 2D flow areas found under any output block.")
            fh.close()
            return report

        for area_name in names:
            result = None
            tried = []
            for block_name, path in available_bases:
                if f"{path}/{area_name}" not in f:
                    tried.append(f"not present under '{block_name}'")
                    continue
                result = _analyze_computation_area(
                    f, path, area_name, iteration_threshold, timestep_drop_fraction,
                    volume_error_threshold, logger, max_logged,
                )
                if result is not None:
                    logger.info(f"[{area_name}] Computations group found under '{block_name}'")
                    break
                tried.append(f"no Computations group under '{block_name}'")

            if result is None:
                reason = "; ".join(tried) if tried else "area not found under any output block"
                report.skipped_areas[area_name] = reason
                logger.info(f"[{area_name}] SKIPPED — {reason}")
                logger.info("")
                continue

            report.areas[area_name] = result
            logger.info("-" * 70)

    logger.info(f"TOTAL areas analyzed: {len(report.areas)} / skipped: {len(report.skipped_areas)}")
    fh.close()

    return report


# ---------------------------------------------------------------------------
# Sparse iteration-trouble event log ("2D Iterations" / "2D Iteration Error")
# ---------------------------------------------------------------------------
#
# Some RAS versions/plans write a per-2D-flow-area event log under
# Results/.../Output Blocks/Computation Block/2D Flow Areas/<Area>/, named
# "2D Iterations" (shape [n_events, 2]) and "2D Iteration Error" (shape
# [n_events]). This was reverse-engineered against one real project file —
# HEC-RAS's HDF layout isn't documented in enough detail to know this for
# certain — and verified in two stages:
#
#   1. First pass: assumed column 0 was a per-event inner-iteration depth
#      (0..19, matching this project's plan-file UNET D2 Max Iterations=20)
#      for EVERY row, and column 1 == -1 was just a sentinel/padding row to
#      discard.
#   2. That pass was wrong in an important way. Splitting the depth
#      histogram by whether column 1 is -1 or a real cell index showed:
#        - rows WITH a real cell index sit almost entirely at depth 0 (a
#          small few at depth 1, essentially never deeper) — depth doesn't
#          discriminate severity for a real cell in this format.
#        - rows with column 1 == -1 carry the full 0..19 decay curve. These
#          aren't padding at all — they're a separate, domain-wide
#          iteration-effort log (closer in spirit to an Outer/Total
#          Iteration Number series) that happens to share the same array,
#          not tied to any specific cell.
#
# Conclusion this module now works from: for a REAL cell, the meaningful
# signal in this log is FREQUENCY (how many times it shows up at all) — a
# cell logged 1,000+ times across the run needed rework almost every
# timestep, a real and otherwise-invisible runtime cost, even though each
# individual occurrence is "shallow." Depth is kept on the record for
# completeness but should not be assumed informative on a new file until
# you've checked the same split there too — a different RAS version/plan
# configuration may behave differently, and this interpretation should be
# re-verified rather than trusted blindly.
#
# There is also no reliable per-event timestep recoverable from this log —
# see git history / prior analysis for why a naive "depth reset -> new
# timestep" heuristic didn't hold up either.

@dataclass
class IterationEventCell:
    area_name: str
    cell: int
    count: int                 # how many logged events for this cell, across the whole run
    max_iteration_depth: int   # informational only — see module note: often doesn't vary for real cells
    max_abs_error: float       # largest |residual| seen for this cell in the log


@dataclass
class IterationEventsReport:
    hdf_path: str
    areas_scanned: list[str] = field(default_factory=list)
    skipped_areas: dict = field(default_factory=dict)  # area -> reason
    cell_depth_histogram: dict = field(default_factory=dict)     # area -> np.ndarray, depths for rows WITH a real cell
    global_depth_histogram: dict = field(default_factory=dict)   # area -> np.ndarray, depths for cell==-1 rows (domain-wide iteration effort, NOT per-cell)
    cells: list[IterationEventCell] = field(default_factory=list)  # sorted by frequency, worst-first, all areas combined

    def summary(self) -> str:
        lines = [
            f"HDF file: {self.hdf_path}",
            f"Areas scanned: {self.areas_scanned}",
            f"Areas skipped: {self.skipped_areas}",
            f"Flagged cells (all areas): {len(self.cells)}",
        ]
        return "\n".join(lines)


def analyze_iteration_events(
    hdf_path: str,
    area_names: list[str] | None = None,
    min_occurrences: int = 1,
    top_n: int | None = 200,
    log_path: str | None = None,
    max_logged: int = 200,
) -> IterationEventsReport:
    """
    Read the sparse "2D Iterations" / "2D Iteration Error" event log (see
    module notes above — this is reverse-engineered, and the key finding is
    that FREQUENCY, not iteration depth, is the meaningful per-cell signal)
    and tally, per real cell (excluding the -1 "no cell" rows, which are a
    separate domain-wide series, not a per-cell one), how often it shows up.

    Parameters
    ----------
    hdf_path : path to the plan HDF
    area_names : which 2D flow area(s) to scan; None (default) scans every
        area that has a "Computation Block/2D Flow Areas/<Area>" group with
        this event log.
    min_occurrences : only include cells logged at least this many times.
        Default 1 (no floor) — combine with `top_n` to keep the result
        usable, since a real model can have thousands of distinct cells
        logged at least once.
    top_n : keep only the N most-frequent cells per area (after
        `min_occurrences` filtering). None keeps everything filtered by
        `min_occurrences` alone — can be a very long list on a real model.
    log_path : defaults to "<hdf stem>_iteration_events.log".
    max_logged : cap on how many flagged cells get logged per area
        (separate from `top_n`, which caps what's returned).

    Returns
    -------
    IterationEventsReport. `cell_depth_histogram` (rows with a real cell)
    and `global_depth_histogram` (rows with no cell, cell index == -1) are
    kept SEPARATE — conflating them was the bug in an earlier version of
    this function. Check both on a new file: if `cell_depth_histogram`
    isn't concentrated near depth 0 like it was on the project this was
    built against, depth may actually be informative for your file and
    worth factoring into `cells` — this function doesn't currently do that
    automatically since it wasn't a reliable signal on the one file this
    was verified against.
    """
    hdf_path = str(hdf_path)
    report = IterationEventsReport(hdf_path=hdf_path)

    if log_path is None:
        log_path = str(Path(hdf_path).with_suffix("")) + "_iteration_events.log"

    logger = logging.getLogger(f"iteration_events_{Path(hdf_path).stem}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)

    logger.info("HEC-RAS sparse iteration-event log scan")
    logger.info(f"File: {hdf_path}")
    logger.info("NOTE: column semantics reverse-engineered, not from documented schema — see module comments.")
    logger.info("Ranking is by FREQUENCY, not iteration depth (depth didn't discriminate for real cells on the project this was verified against).")
    logger.info("-" * 70)

    with h5py.File(hdf_path, "r") as f:
        comp_base = "Results/Unsteady/Output/Output Blocks/Computation Block/2D Flow Areas"
        if comp_base not in f:
            logger.info(f"'{comp_base}' not found in this file — no Computation Block output.")
            fh.close()
            return report

        names = area_names if area_names is not None else list(f[comp_base].keys())
        all_cells: list[IterationEventCell] = []

        for area_name in names:
            area_path = f"{comp_base}/{area_name}"
            iters_path = f"{area_path}/2D Iterations"
            err_path = f"{area_path}/2D Iteration Error"

            if area_path not in f:
                report.skipped_areas[area_name] = "area not found under Computation Block"
                logger.info(f"[{area_name}] SKIPPED — area not found under Computation Block")
                continue
            if iters_path not in f or err_path not in f:
                report.skipped_areas[area_name] = "no '2D Iterations'/'2D Iteration Error' datasets"
                logger.info(f"[{area_name}] SKIPPED — no '2D Iterations'/'2D Iteration Error' datasets")
                continue

            report.areas_scanned.append(area_name)
            iters = np.asarray(f[iters_path][()])
            errors = np.asarray(f[err_path][()]).reshape(-1)

            is_global = iters[:, 1] == -1
            cell_depths = iters[~is_global, 0]
            global_depths = iters[is_global, 0]
            cell_idx = iters[~is_global, 1]
            cell_errs = errors[~is_global]

            cell_hist = np.bincount(cell_depths[cell_depths >= 0], minlength=20) if len(cell_depths) else np.zeros(20, dtype=int)
            global_hist = np.bincount(global_depths[global_depths >= 0], minlength=20) if len(global_depths) else np.zeros(20, dtype=int)
            report.cell_depth_histogram[area_name] = cell_hist
            report.global_depth_histogram[area_name] = global_hist

            logger.info(f"[{area_name}] {len(cell_idx)} cell-tied event(s), {is_global.sum()} domain-wide (no-cell) event(s)")
            logger.info(f"    Cell-tied depth histogram:   {cell_hist.tolist()}")
            logger.info(f"    Domain-wide depth histogram: {global_hist.tolist()} (NOT per-cell — see module note)")

            tally: dict[int, dict] = defaultdict(lambda: {"count": 0, "max_depth": 0, "max_err": 0.0})
            for d, c, e in zip(cell_depths.tolist(), cell_idx.tolist(), cell_errs.tolist()):
                t = tally[int(c)]
                t["count"] += 1
                t["max_depth"] = max(t["max_depth"], int(d))
                t["max_err"] = max(t["max_err"], abs(float(e)))

            area_cells = [
                IterationEventCell(area_name=area_name, cell=cell, count=v["count"],
                                    max_iteration_depth=v["max_depth"], max_abs_error=v["max_err"])
                for cell, v in tally.items()
                if v["count"] >= min_occurrences
            ]
            area_cells.sort(key=lambda c: (-c.count, -c.max_iteration_depth))
            if top_n is not None:
                area_cells = area_cells[:top_n]
            all_cells.extend(area_cells)

            logger.info(f"    {len(area_cells)} cell(s) kept (min_occurrences={min_occurrences}, top_n={top_n})")
            for c in area_cells[:max_logged]:
                logger.info(
                    f"        cell {c.cell} | seen {c.count}x | max depth {c.max_iteration_depth} | "
                    f"max |error| = {c.max_abs_error:.4g}"
                )
            if len(area_cells) > max_logged:
                logger.info(f"        ... {len(area_cells) - max_logged} more not shown")
            logger.info("")

        report.cells = all_cells

    logger.info("-" * 70)
    logger.info(f"TOTAL cells kept (all areas): {len(report.cells)}")
    fh.close()

    return report


# ---------------------------------------------------------------------------
# Export flagged cells to a point vector file (GeoPackage by default)
# ---------------------------------------------------------------------------

def _get_cell_centers(
    h5file: h5py.File,
    area_name: str,
    keywords: list[str] | None = None,
) -> tuple[str, np.ndarray] | None:
    """Return (dataset_path, [n_cells, 2] x/y array) for a flow area's cell
    center coordinates, or None if no matching geometry dataset is found."""
    if keywords is None:
        keywords = ["cells center coordinate", "cell center coordinate", "center coordinate"]
    candidates = _find_datasets(h5file, keywords, path_contains=f"Geometry/2D Flow Areas/{area_name}")
    if not candidates:
        return None
    ds = candidates[0]
    return ds, np.asarray(h5file[ds][()])


def export_cells_to_points(
    hdf_path: str,
    rows: list,
    out_path: str,
    layer_name: str = "flagged_cells",
    driver_name: str = "GPKG",
) -> str | None:
    """
    Export flagged-cell records — from any of this module's functions
    (IterationEventCell, CellSummary, SmallCell, WorstOffender, WetDryCell,
    or anything else with an `area_name` and a `cell`/`index` attribute) —
    to a point vector file, one point per record at that cell's geometry
    center, with every other dataclass field carried over as an attribute
    column. Defaults to GeoPackage; pass driver_name="ESRI Shapefile" for a
    .shp instead.

    Uses GDAL's Python bindings (osgeo.ogr/osr) rather than geopandas/
    fiona, since GDAL already ships with QGIS/OSGeo4W — no extra install,
    and no risk of it fighting OSGeo4W's own GDAL/HDF5 DLLs the way a
    separately pip-installed geo package could.

    Coordinates come from Geometry/2D Flow Areas/<Area>/Cells Center
    Coordinate (or a similarly named dataset — matched the same
    keyword-search way as the rest of this module). The output CRS is
    read from the HDF root's "Projection" attribute if present (this is
    the common convention for where RAS stores the geometry's WKT
    projection, but isn't guaranteed across every version — if the output
    file comes through with no CRS, check `list(f.attrs.keys())` at the
    HDF root yourself and tell me the actual attribute name so this can be
    fixed to match).

    Parameters
    ----------
    hdf_path : path to the plan HDF (source of cell-center geometry + CRS)
    rows : list of flagged-cell records to export (mixing types from
        different functions in one call is fine — they're grouped by
        area_name internally, and non-overlapping fields just come out
        blank on rows that don't have them)
    out_path : output file path, e.g. "flagged_cells.gpkg"
    layer_name : layer name inside the output file
    driver_name : OGR driver name; "GPKG" (default) or "ESRI Shapefile",
        for example

    Returns
    -------
    out_path if a file was written, or None if `rows` was empty (see note
    below) — always check the return value in a batch/multi-file script
    rather than assuming a file was created.

    An empty `rows` is a real, expected outcome in a pipeline that runs
    the same analysis across many models: some plans simply don't have
    the dataset a given analysis needs (e.g. no "Computation Block"
    iteration log in this RAS version/output config), so
    `analyze_iteration_events` etc. correctly return zero flagged cells
    rather than erroring. This function does NOT raise for that case —
    it prints a one-line notice and returns None — so a batch script
    calling this unconditionally after each analysis doesn't crash the
    whole run over one model having nothing to flag. If you want a hard
    failure on empty input instead, check `if not rows:` yourself before
    calling.

    Raises
    ------
    ImportError if GDAL's Python bindings aren't importable in the
    interpreter this is run from.
    RuntimeError if the requested OGR driver isn't available.
    """
    try:
        from osgeo import ogr, osr
    except ImportError as e:
        raise ImportError(
            "osgeo (GDAL Python bindings) not importable. Run this from your "
            "OSGeo4W/QGIS Python, not a plain venv without GDAL installed."
        ) from e

    if not rows:
        print(f"NOTE: no rows to export -- skipping {out_path} (nothing was flagged, or this analysis found no matching dataset for this file)")
        return None

    # Group by area, since cell-center geometry is looked up per area.
    by_area: dict[str, list] = defaultdict(list)
    for r in rows:
        area = getattr(r, "area_name", None)
        if area is None:
            raise ValueError(f"row {r!r} has no area_name attribute -- can't look up its geometry")
        by_area[area].append(r)

    # Union of extra attribute fields across all row types, skipping the
    # ones that become dedicated columns (area_name, and cell/index which
    # both map to "cell_id").
    skip_fields = {"area_name", "cell", "index"}
    field_names: list[str] = []
    for r in rows:
        if dataclasses.is_dataclass(r):
            for f in dataclasses.fields(r):
                if f.name not in skip_fields and f.name not in field_names:
                    field_names.append(f.name)

    driver = ogr.GetDriverByName(driver_name)
    if driver is None:
        raise RuntimeError(f"OGR driver '{driver_name}' not available in this GDAL build")

    ds_out = None
    file_exists = Path(out_path).exists()

    if file_exists and driver_name == "GPKG":
        # A GeoPackage is a multi-layer SQLite file -- if it exists, open it
        # in UPDATE mode and just replace the one layer, rather than
        # deleting the whole file. SQLite normally tolerates a writer
        # alongside another process (e.g. QGIS) that has the file open for
        # read, so this sidesteps the Windows "can't delete an open file"
        # problem that a full-file delete-and-recreate runs into.
        ds_out = ogr.Open(out_path, update=1)
        if ds_out is not None:
            # OGR/gdal.Dataset has no GetLayerIndex() -- find the index by
            # scanning layer names ourselves.
            existing_idx = None
            for i in range(ds_out.GetLayerCount()):
                lyr = ds_out.GetLayerByIndex(i)
                if lyr is not None and lyr.GetName() == layer_name:
                    existing_idx = i
                    break
            if existing_idx is not None:
                ds_out.DeleteLayer(existing_idx)
        # If ogr.Open returned None (e.g. the file is corrupt, or exclusively
        # locked rather than just open), fall through to the delete-and-
        # recreate path below.

    if ds_out is None:
        if file_exists:
            driver.DeleteDataSource(out_path)  # can fail silently (GDAL logs "ERROR 1" but doesn't raise)
            if Path(out_path).exists():
                # Most common cause: the file is still open elsewhere (e.g. loaded in
                # QGIS). Try a plain filesystem delete as a fallback.
                try:
                    Path(out_path).unlink()
                except OSError as e:
                    raise RuntimeError(
                        f"Could not remove or update existing file '{out_path}'. For a GeoPackage, having it "
                        f"open in QGIS should be fine (SQLite allows a writer alongside a reader); this "
                        f"failure suggests something stronger has it locked (e.g. an exclusive lock, or a "
                        f"different process). Underlying error: {e}. Close whatever has it open and try "
                        f"again, or export to a different path."
                    ) from e
        ds_out = driver.CreateDataSource(out_path)

    if ds_out is None:
        raise RuntimeError(
            f"GDAL's {driver_name} driver failed to create '{out_path}' for an unspecified reason "
            f"(CreateDataSource returned None). Check the path is writable and the parent directory exists."
        )

    written = 0
    skipped = 0

    with h5py.File(hdf_path, "r") as f:
        srs = None
        wkt = f.attrs.get("Projection")
        if wkt is not None:
            if isinstance(wkt, bytes):
                wkt = wkt.decode()
            srs = osr.SpatialReference()
            srs.ImportFromWkt(wkt)

        layer = ds_out.CreateLayer(layer_name, srs=srs, geom_type=ogr.wkbPoint)
        layer.CreateField(ogr.FieldDefn("area_name", ogr.OFTString))
        cell_id_field = ogr.FieldDefn("cell_id", ogr.OFTInteger64)
        layer.CreateField(cell_id_field)
        for name in field_names:
            layer.CreateField(ogr.FieldDefn(name, ogr.OFTReal))
        layer_defn = layer.GetLayerDefn()

        for area_name, area_rows in by_area.items():
            center_result = _get_cell_centers(f, area_name)
            if center_result is None:
                skipped += len(area_rows)
                continue
            _, centers = center_result

            for r in area_rows:
                cell_id = getattr(r, "cell", None)
                if cell_id is None:
                    cell_id = getattr(r, "index", None)
                if cell_id is None or cell_id < 0 or cell_id >= len(centers):
                    skipped += 1
                    continue
                x, y = float(centers[cell_id][0]), float(centers[cell_id][1])

                feat = ogr.Feature(layer_defn)
                feat.SetField("area_name", area_name)
                feat.SetField("cell_id", int(cell_id))
                for name in field_names:
                    val = getattr(r, name, None)
                    if val is not None:
                        try:
                            feat.SetField(name, float(val))
                        except (TypeError, ValueError):
                            pass
                geom = ogr.Geometry(ogr.wkbPoint)
                geom.AddPoint_2D(x, y)
                feat.SetGeometry(geom)
                layer.CreateFeature(feat)
                feat = None
                written += 1

    ds_out = None  # closes/flushes the file
    if srs is None:
        print(f"NOTE: no 'Projection' attribute found on the HDF root — {out_path} has no CRS assigned. "
              f"Set one manually in QGIS (Layer Properties -> Source -> Assigned CRS) to match your geometry's projection.")
    print(f"Wrote {written} point(s) to {out_path} ({skipped} row(s) skipped — no matching geometry or out-of-range cell id)")

    return out_path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyze_hdf_stability(
    hdf_path: str,
    courant_threshold: float = 2.0,
    ws_error_threshold: float = 0.05,
    iteration_threshold: float | None = 20,
    small_cell_area_threshold: float | None = None,
    iteration_domain_percentile: float = 90,
    iteration_domain_threshold: float | None = None,
    error_keywords: list[str] | None = None,
    iteration_keywords: list[str] | None = None,
    geometry_area_keywords: list[str] | None = None,
    log_path: str | None = None,
    max_logged_per_dataset: int = 200,
) -> StabilityReport:
    """
    Scan `hdf_path` for Courant-number, water-surface-error, and
    high-iteration-count violations.

    Parameters
    ----------
    hdf_path : path to a HEC-RAS plan .hdf results file
    courant_threshold : flag any Cell Courant value strictly greater than this
        (default 2.0)
    ws_error_threshold : flag any water-surface-error value whose absolute
        value exceeds this (units match whatever the RAS output uses, usually
        feet or meters)
    iteration_threshold : flag any solver-iteration-count value strictly
        greater than this. Pass None to skip the iteration scan entirely.
    small_cell_area_threshold : flag any 2D mesh cell whose surface area
        (read from the geometry portion of the HDF, e.g.
        "Geometry/2D Flow Areas/<Area>/Cells Surface Area") is strictly less
        than this, in whatever horizontal units the geometry was built in
        (typically ft^2 for Texas projects in state-plane feet). Pass None
        (the default) to skip this scan — pick a value based on your mesh's
        typical cell size; there's no universally "correct" small-cell size.
    error_keywords : dataset-name substrings (case-insensitive) that identify
        "error" datasets to scan. Defaults to a broad set covering common
        HEC-RAS 2D output naming across versions.
    iteration_keywords : dataset-name substrings (case-insensitive) that
        identify iteration-count datasets to scan (e.g. "Cell Last
        Iteration"). Defaults to ["iteration"].
    geometry_area_keywords : dataset-name substrings (case-insensitive) that
        identify per-cell surface-area datasets under Geometry/2D Flow
        Areas/. Defaults to ["cells surface area", "cell surface area",
        "surface area"].
    iteration_domain_percentile : for each iteration dataset, also reports
        the Nth percentile iteration count across ALL cells at each
        timestep (default 90th) — a handful of cells pinned at the
        iteration cap is often local geometry, but many cells simultaneously
        near the cap is a broader convergence struggle and a better runtime
        predictor. Stored per-dataset on
        `report.iteration_domain_percentile_series`.
    iteration_domain_threshold : if given, only log domain-wide percentile
        rows at or above this value (still computes/stores the full series
        either way). None (default) logs the worst `max_logged_per_dataset`
        timesteps regardless of value.
    log_path : if given, a plain-text log is written here in addition to the
        StabilityReport being returned. Defaults to "<hdf stem>_stability.log"
        next to the input file if not provided.
    max_logged_per_dataset : cap on how many grouped cell-summary lines are
        written per dataset (counts still reflect the true total).

    Returns
    -------
    StabilityReport. `*_violations` holds the raw per-timestep rows;
    `*_by_cell` holds the same data collapsed to one summary per
    (dataset, cell) — this is what the log file prints, and normally what
    you want when scanning results.
    """
    hdf_path = str(hdf_path)
    if error_keywords is None:
        error_keywords = ["water surface error", "wse error", "error"]
    if iteration_keywords is None:
        iteration_keywords = ["iteration"]
    if geometry_area_keywords is None:
        geometry_area_keywords = ["cells surface area", "cell surface area", "surface area"]

    report = StabilityReport(
        hdf_path=hdf_path,
        courant_threshold=courant_threshold,
        ws_error_threshold=ws_error_threshold,
        iteration_threshold=iteration_threshold,
        small_cell_area_threshold=small_cell_area_threshold,
    )

    if log_path is None:
        log_path = str(Path(hdf_path).with_suffix("")) + "_stability.log"

    logger = logging.getLogger(f"hdf_stability_{Path(hdf_path).stem}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)

    logger.info("HEC-RAS HDF stability scan")
    logger.info(f"Run at: {_dt.datetime.now().isoformat(timespec='seconds')}")
    logger.info(f"File:   {hdf_path}")
    logger.info(f"Courant threshold:             > {courant_threshold}")
    logger.info(f"Water-surface error threshold: |value| > {ws_error_threshold}")
    logger.info(f"Iteration threshold:           > {iteration_threshold}" if iteration_threshold is not None else "Iteration threshold:           (skipped)")
    logger.info(f"Small-cell area threshold:     < {small_cell_area_threshold}" if small_cell_area_threshold is not None else "Small-cell area threshold:     (skipped)")
    logger.info("-" * 70)

    def _run_section(tag: str, datasets: list[str], threshold: float, comparison: str,
                      value_fmt: str) -> tuple[list[Violation], list[CellSummary]]:
        all_violations: list[Violation] = []
        for ds in datasets:
            violations = _scan_dataset(f, ds, threshold, comparison=comparison)
            all_violations.extend(violations)
            grouped = _group_by_cell(violations)
            logger.info(f"[{tag}] {ds}: {len(violations)} row(s) -> {len(grouped)} cell(s) flagged")
            for g in grouped[:max_logged_per_dataset]:
                when_first = g.first_time_label if g.first_time_label else f"timestep {g.first_timestep}"
                when_last = g.last_time_label if g.last_time_label else f"timestep {g.last_timestep}"
                span = when_first if g.first_timestep == g.last_timestep else f"{when_first} .. {when_last}"
                logger.info(
                    f"    cell {g.cell} | {g.count} timestep(s) | peak = {g.max_value:{value_fmt}} | {span}"
                )
            if len(grouped) > max_logged_per_dataset:
                logger.info(f"    ... {len(grouped) - max_logged_per_dataset} more cell(s) not shown")
            logger.info("")
        return all_violations, _group_by_cell(all_violations)

    with h5py.File(hdf_path, "r") as f:
        # --- Courant ---------------------------------------------------
        courant_datasets = _find_datasets(f, ["courant"])
        report.courant_datasets_scanned = courant_datasets
        logger.info(f"Courant datasets found ({len(courant_datasets)}):")
        for d in courant_datasets:
            logger.info(f"  {d}")
        logger.info("")

        report.courant_violations, report.courant_by_cell = _run_section(
            "COURANT", courant_datasets, courant_threshold, "gt", ".4f"
        )

        # --- Water surface error -----------------------------------------
        error_datasets = _find_datasets(f, error_keywords)
        report.error_datasets_scanned = error_datasets
        logger.info(f"Error datasets found ({len(error_datasets)}):")
        for d in error_datasets:
            logger.info(f"  {d}")
        logger.info("")

        report.ws_error_violations, report.ws_error_by_cell = _run_section(
            "WS ERROR", error_datasets, ws_error_threshold, "abs_gt", ".5f"
        )

        # --- High iteration count -----------------------------------------
        if iteration_threshold is not None:
            iteration_datasets = _find_datasets(f, iteration_keywords)
            report.iteration_datasets_scanned = iteration_datasets
            logger.info(f"Iteration datasets found ({len(iteration_datasets)}):")
            for d in iteration_datasets:
                logger.info(f"  {d}")
            logger.info("")

            report.iteration_violations, report.iteration_by_cell = _run_section(
                "ITERATION", iteration_datasets, iteration_threshold, "gt", ".0f"
            )

            # Domain-wide struggle, not just isolated cells: a handful of
            # cells pinned at the iteration cap is often local geometry: many
            # cells simultaneously near the cap, at the same timestep, is a
            # broader convergence struggle and a better runtime predictor.
            for ds in iteration_datasets:
                data = np.asarray(f[ds][()])
                if data.ndim == 1:
                    continue  # summary/max dataset, no per-timestep spread to report
                pct = np.nanpercentile(data, iteration_domain_percentile, axis=1)
                time_labels = _find_time_labels(f, ds, data.shape[0])
                worst_idx = np.argsort(-pct)[:max_logged_per_dataset]
                logger.info(
                    f"[ITERATION domain-wide] {ds}: p{iteration_domain_percentile} iteration count across all "
                    f"cells, worst timesteps first"
                )
                for t in worst_idx.tolist():
                    if iteration_domain_threshold is not None and pct[t] < iteration_domain_threshold:
                        continue
                    when = time_labels[t] if time_labels else f"timestep {t}"
                    logger.info(f"    {when} | p{iteration_domain_percentile} = {pct[t]:.1f}")
                report.iteration_domain_percentile_series[ds] = pct
                logger.info("")

        # --- Small cells (geometry, static — no time axis) ------------------
        if small_cell_area_threshold is not None:
            # Restrict to the Geometry/2D Flow Areas/ tree so this can't pick
            # up an unrelated "area" dataset from results output elsewhere.
            area_datasets = _find_datasets(
                f, geometry_area_keywords, path_contains="Geometry/2D Flow Areas"
            )
            report.geometry_area_datasets_scanned = area_datasets
            logger.info(f"Geometry area datasets found ({len(area_datasets)}):")
            for d in area_datasets:
                logger.info(f"  {d}")
            logger.info("")

            for ds in area_datasets:
                area_name = _parse_flow_area_name(ds)
                data = np.asarray(f[ds][()]).reshape(-1)
                small_idx = np.nonzero(data < small_cell_area_threshold)[0]
                logger.info(f"[SMALL CELL] {ds} (area '{area_name}'): {len(small_idx)} cell(s) below {small_cell_area_threshold}")
                for cell in small_idx.tolist()[:max_logged_per_dataset]:
                    area_val = float(data[cell])
                    report.small_cells.append(
                        SmallCell(dataset=ds, area_name=area_name, cell=cell, area=area_val)
                    )
                    logger.info(f"    cell {cell} | area = {area_val:.3f}")
                if len(small_idx) > max_logged_per_dataset:
                    for cell in small_idx.tolist()[max_logged_per_dataset:]:
                        report.small_cells.append(
                            SmallCell(dataset=ds, area_name=area_name, cell=int(cell), area=float(data[cell]))
                        )
                    logger.info(f"    ... {len(small_idx) - max_logged_per_dataset} more not shown")
                logger.info("")

            report.small_cells.sort(key=lambda s: s.area)

    logger.info("-" * 70)
    logger.info(f"TOTAL Courant:   {len(report.courant_violations)} rows / {len(report.courant_by_cell)} cells")
    logger.info(f"TOTAL WS error:  {len(report.ws_error_violations)} rows / {len(report.ws_error_by_cell)} cells")
    logger.info(f"TOTAL Iteration: {len(report.iteration_violations)} rows / {len(report.iteration_by_cell)} cells")
    logger.info(f"TOTAL Small cells: {len(report.small_cells)}")
    fh.close()

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main():
    parser = argparse.ArgumentParser(
        description="Scan a HEC-RAS HDF results file for Courant, water-surface-error, and high-iteration violations."
    )
    parser.add_argument("hdf_path", help="Path to the HEC-RAS plan .hdf file")
    parser.add_argument("--courant", type=float, default=2.0, help="Courant number threshold (default 2.0)")
    parser.add_argument("--ws-error", type=float, default=0.05, help="Water-surface error threshold, abs value (default 0.05)")
    parser.add_argument("--iteration", type=float, default=20, help="Iteration-count threshold; use a negative number to skip (default 20)")
    parser.add_argument("--min-cell-area", type=float, default=None, help="Flag 2D mesh cells with surface area below this (units match your geometry, e.g. ft^2); omit to skip")
    parser.add_argument("--computed-courant", type=float, default=None, help="Also compute Courant = V*dt/sqrt(area) from raw velocity/timestep/geometry and flag cells above this value; omit to skip")
    parser.add_argument("--wet-dry-transitions", type=int, default=None, metavar="MIN_TRANSITIONS", help="Also flag cells with at least this many wet/dry flips over the run; omit to skip")
    parser.add_argument("--compute-time-summary", action="store_true", help="Also dump whatever run/compute-time bookkeeping RAS wrote to Results/.../Summary")
    parser.add_argument("--computation-summary", action="store_true", help="Also read the domain-wide 'Computations' scalar series (Time Step, Total Iteration Number, Volume Error, etc.) -- works even without Detailed 2D output enabled")
    parser.add_argument("--iteration-events", action="store_true", help="Also read the sparse 'Computation Block' iteration-trouble event log, if present, and tally worst-offender cells")
    parser.add_argument("--log", default=None, help="Path to write the log file (default: <hdf name>_stability.log)")
    args = parser.parse_args()

    report = analyze_hdf_stability(
        args.hdf_path,
        courant_threshold=args.courant,
        ws_error_threshold=args.ws_error,
        iteration_threshold=None if args.iteration < 0 else args.iteration,
        small_cell_area_threshold=args.min_cell_area,
        log_path=args.log,
    )
    print(report.summary())

    if args.computed_courant is not None:
        vc_report = analyze_velocity_courant(args.hdf_path, target_courant=args.computed_courant)
        print()
        print(vc_report.summary())

    if args.wet_dry_transitions is not None:
        wd_report = analyze_wet_dry_transitions(args.hdf_path, min_transitions=args.wet_dry_transitions)
        print()
        print(wd_report.summary())

    if args.compute_time_summary:
        findings = read_compute_time_summary(args.hdf_path)
        print()
        print(f"Compute-time summary fields found: {len(findings)} (see log for details)")

    if args.computation_summary:
        cs_report = analyze_computation_summary(args.hdf_path)
        print()
        print(cs_report.summary())

    if args.iteration_events:
        ie_report = analyze_iteration_events(args.hdf_path)
        print()
        print(ie_report.summary())


if __name__ == "__main__":
    _main()