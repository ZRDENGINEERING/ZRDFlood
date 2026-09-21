import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="osgeo")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pyogrio")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pyproj")
warnings.filterwarnings("ignore", message="Conversion of an array with ndim > 0")

import dotenv
from pathlib import Path
from shapely.geometry import box
import geopandas as gpd
from pathlib import Path
import pandas as pd

import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

import os
print(os.environ['PATH'])


import os, sys

_h5py_dir = r"C:\OSGeo4W\apps\Python312\Lib\site-packages\h5py"
if os.path.isdir(_h5py_dir):
    os.add_dll_directory(_h5py_dir)

from hdf_stability_report import (
    analyze_computation_summary,
    analyze_iteration_events,
    export_cells_to_points,
)

from hdf_stability_report import (
    analyze_hdf_stability,
    analyze_velocity_courant,
    analyze_wet_dry_transitions,
)

from cfg import CFG, get_current_qgis_project

from db.db import (
    table_exists, has_features, get_table, insert_table,
    make_schema, insert_gpkg_layer, gpkg_layer_exists, get_table_mtime, get_srid
)




class ZRDFLOOD:
    def __init__(self, proj_name=None, proj_num=None):
        self.current_file = Path(__file__).resolve()
        self.env_file = Path("Z:/10 DEV/zrdhnh/cfg/proj.env")
        dotenv.load_dotenv(self.env_file)

        proj_q = get_current_qgis_project()
        if not proj_q == None and len(proj_q) > 0:
            self.proj_name = proj_q.split('/')[3]
            self.proj_num = self.proj_name.split(' ')[0]
        else:
            self.proj_name = proj_name
            self.proj_num = proj_num

        if self.proj_num is None or self.proj_name is None:
            dotenv.load_dotenv(self.env_file)

        self.zhnh = CFG(self.proj_name, self.proj_num)

        self.zhnh_proj = self.zhnh.PROJ_DICT
        self.proj_crs = self.zhnh.proj_crs

        self.path_proj = self.zhnh.path_proj
        self.path_prj = self.zhnh.path_prj

        self.schema = make_schema(self.zhnh.proj_num)

        # self.ws.prep_dem()
        # self.qc.run_depth_in_sink()

        # path_cline = r"Z:\01 ZRD\03 PROJECTS\199805002 Test Riv\03 GIS\99 WKNG\03 VECTOR\CL_SHP.shp"
        # gdf_out = cross_sections_from_file(path_cline, 500, 500, crs=self.proj_crs)

        # insert_table(gdf_out, 'xs_rw', self.schema, if_exists='replace', srid=self.proj_crs)

        # hdf_path = r"Z:\zNO_BAK\GLO\1201000511_1201000510-Adams & Cow Bayou\Adams_Cow_Bayou_AA_RASv641\Pinehurst\Pinehurst.p04-08,60-64,66-70\Pinehurst.p04.hdf"
        # hdf_path = r"Z:\zNO_BAK\GLO\1201000511_1201000510-Adams & Cow Bayou\Adams_Cow_Bayou_AA_RASv641\ColeCreek\ColeCreek\ColeCreek.p01.hdf"
        # hdf_path = r"Z:\zNO_BAK\GLO\1201000511_1201000510-Adams & Cow Bayou\Adams_Cow_Bayou_AA_RASv641\ColeCreek\ColeCreek\ColeCreek.p02.hdf"
        hdf_path = r"C:\Temp\HECRAS\LUCAS_SG_v66\LUCAS_SG_v66.p01.hdf"
        # hdf_path = r"C:\Users\desro\Downloads\11130206_Models\Wichita_Models\Input\Wichita.p01.hdf"
        # hdf_path = r"Z:\zNO_BAK\BLE\11140302  Lower Sulpher\11140302_Models Lower Sulpher\RAS Submittal\EastLowerSulphur_Texas\Input\LowerSulphurEast_TX.p01.hdf"
        hdf_path = r"Z:\zNO_BAK\BLE\11140302  Lower Sulpher\11140302_Models Lower Sulpher\RAS Submittal\Hydraulic_Models_2\RAS Submittal\EastLowerSulphur_Texas\Input\LowerSulphurEast_TX.p01.hdf"
        # hdf_path = r"Z:\zNO_BAK\GLO\1210020303 - Upper San Marcos Resubmittal\USM_RASv641\Exist1.p23.hdf"


        # Domain-wide Time Step / Iteration / Volume Error / Percent Active Cells,
        # scans every 2D flow area automatically
        cs = analyze_computation_summary(hdf_path)
        print(cs.summary())

        # Sparse per-cell iteration-trouble log, ranked by frequency
        ie = analyze_iteration_events(hdf_path, min_occurrences=50, top_n=200)
        print(ie.summary())

        # Export the flagged cells to a GeoPackage for QGIS
        export_cells_to_points(hdf_path, ie.cells, r"C:\Temp\del\flagged_cells.gpkg")




        # Only works if the plan's 2D Output Variables include the detailed per-cell datasets 
        # (Cell Courant, Cell Velocity, Cell Last Iteration, per-cell wet fraction) — check Base Output's contents first if unsure:

        # Courant (threshold 2.0), water-surface error, iteration count, small cells
        report = analyze_hdf_stability(hdf_path, courant_threshold=2.0, small_cell_area_threshold=None)
        print(report.summary())

        # Cr = V*dt/sqrt(area), computed independently of RAS's own reported Courant
        vc = analyze_velocity_courant(hdf_path, target_courant=1.0)

        # Cells flipping wet/dry repeatedly
        wd = analyze_wet_dry_transitions(hdf_path, min_transitions=5)




if __name__ == "__main__":
    z = ZRDFLOOD()
