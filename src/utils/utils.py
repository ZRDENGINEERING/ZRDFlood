import os
import psutil
import time

from shapely.ops import substring

from pathlib import Path
from osgeo import ogr
import rasterio as rio
from rasterio.warp import calculate_default_transform

import geopandas as gpd


from qgis.core import (QgsVectorLayer, QgsApplication, QgsVectorFileWriter, QgsField, QgsLayerTreeLayer)
                       

QgsApplication.setPrefixPath('C:\\OSGeo4W\\apps\\qgis-ltr\\', True)
qgs = QgsApplication([], False)
qgs.initQgis()





# def _add_qgis_plugins_path():
#     """The core `processing` plugin lives under QGIS's plugins dir,
#     which QGIS Desktop adds to sys.path at startup — but a standalone
#     script via python-qgis.bat doesn't get this for free."""
#     candidates = [
#         os.environ.get("QGIS_PLUGINPATH"),
#         r"C:\OSGeo4W\apps\qgis\python\plugins",
#         r"C:\OSGeo4W\apps\qgis-ltr\python\plugins",
#     ]
#     for path in candidates:
#         if path and os.path.isdir(path) and path not in sys.path:
#             sys.path.append(path)
#             return path
#     return None

# _add_qgis_plugins_path()



def area_units(gdf):
    crs = gdf.crs
    if crs is None:
        raise ValueError("GeoDataFrame has no CRS set")
    if crs.is_geographic:
        raise ValueError(f"CRS {crs.to_epsg()} is geographic (degrees) — reproject first")

    unit = crs.axis_info[0].unit_name
    if unit not in ("US survey foot", "foot"):
        print(f"WARNING: unexpected unit '{unit}' — verify before trusting .area")
    return unit






def linestring_cut(line, start_distance, length):
    end_distance = start_distance + length
    cut_segment = substring(line, start_distance, end_distance)
    return cut_segment


# def cut(line, dist_beg):
#     # Cuts a line in two at a distance from its beging point
#     if dist_beg <= 0.0 or dist_beg >= line.length:
#         return [LineString(line)]
#     coords = list(line.coords)

#     for i, p in enumerate(coords):
#         pd = line.project(Point(p))

#         if pd == dist_beg:
#             return [
#                 LineString(coords[:i+1]),
#                 LineString(coords[i:])]
#         if pd > dist_beg:
#             cp = line.interpolate(dist_beg)
#             return [
#                 LineString(coords[:i] + [(cp.x, cp.y)]),
#                 LineString([(cp.x, cp.y)] + coords[i:])]









def gpkg_add_vec(path_src, path_tgt, lname = None):
        if not lname:
            lname = os.path.basename(path_tgt).split(".")[0]
                
        if os.path.exists(path_tgt):
            try:
                gdf = gpd.read_file(path_src, layer=lname)

                if 'fid' in gdf.columns:
                    gdf['fid'] = gdf['fid'].astype(int)
                if 'FID' in gdf.columns:
                    gdf['FID'] = gdf['FID'].astype(int)

                gdf.to_file(path_tgt, layer=lname, mode='w', overwrite='yes')
            except Exception as e:
                print(f'LAYER {lname} DOES NOT EXIST....{e}')















def get_locking_process(path: str):
    target = os.path.abspath(path).lower()
    for proc in psutil.process_iter(['name', 'open_files']):
        try:
            for f in proc.info['open_files'] or []:
                if os.path.abspath(f.path).lower() == target:
                    return proc.info['name'], proc.pid
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return None, None


def is_file_locked(path: str) -> bool:
    target = os.path.abspath(path).lower()
    for proc in psutil.process_iter(['open_files']):
        try:
            for f in proc.info['open_files'] or []:
                if os.path.abspath(f.path).lower() == target:
                    return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return False


def wait_for_file(path: str, poll: float = 1.0, timeout: float = 30.0):
    if not os.path.exists(path):
        return
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if os.path.exists(path) and not is_file_locked(path):
            return
        remaining = deadline - time.monotonic()
        size_mb = os.path.getsize(path) / 1_048_576 if os.path.exists(path) else 0
        name, pid = get_locking_process(path)
        who = f"{name} (pid {pid})" if name else "another program"
        print(f"  [{remaining:4.0f}s remaining] {who} has this file open — close it to continue — {size_mb:.1f} MB — {path}")
        time.sleep(poll)

    name, pid = get_locking_process(path)
    who = f"{name} (pid {pid})" if name else "another program"
    print(f"  ERROR: file still locked by {who} after {timeout:.0f}s — {path}")
    raise TimeoutError(f"File still locked after {timeout:.0f}s: {path}")


def safe_remove(path: str, retries: int = 3, delay: float = 1.0):
    for i in range(retries):
        try:
            os.remove(path)
            return
        except PermissionError:
            if i == retries - 1:
                raise
            time.sleep(delay)




def ensure_gpkg(path, layer=None, overwrite_layer=True):
    """Make sure the gpkg's directory (and file) exist before running an OGR/GDAL algorithm against it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        # Create an empty, valid GeoPackage container — no layers yet.
        driver = ogr.GetDriverByName("GPKG")
        ds = driver.CreateDataSource(str(path))
        ds = None
        return

    # File already exists (e.g. PROJ.gpkg) — drop the target layer if it's
    # already there, so gdal:polygonize doesn't fail on "layer already exists".
    if layer and overwrite_layer:
        ds = ogr.Open(str(path), update=1)
        if ds and ds.GetLayerByName(layer) is not None:
            ds.DeleteLayer(layer)
        ds = None



def get_cell_size(path_rast, crs):
    with rio.open(path_rast) as src:
        if src.crs.is_geographic:
            transform, width, height = calculate_default_transform(
                src.crs, crs, src.width, src.height, *src.bounds
            )
            cell_size = abs(transform[0] * transform[4])  # now in proj_crs units (ft²)
        else:
            cell_size = abs(src.transform[0] * src.transform[4])
    return round(cell_size, 1)




if __name__ == "__main__":
    print(__name__)
    pass
    