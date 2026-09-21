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

import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

from cfg import CFG, get_current_qgis_project

from db.db import (
    table_exists, has_features, get_table, insert_table,
    make_schema, insert_gpkg_layer, gpkg_layer_exists, get_table_mtime, get_srid
)


from paramsws import PARAMSWS
from paramsqc import PARAMSQC

from batch_pipeline import process_batch, write_batch_results
from bank_detection import analyze_cross_sections, build_bank_lines, build_inundation_raster

from xs_pipeline import cross_sections_from_polygon_file

from polygon_sides import extract_polygon_sides, polygon_sides_to_gdf
from polygon_sides import sides_from_file
from polygon_centerline import polygon_centerline


 git remote set-url origin https://github.com/ZRDENGINEERING/ZRDFlood.git

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

        self.ws = PARAMSWS(self.zhnh)
        self.qc = PARAMSQC(self.zhnh)


        # self.ws.prep_dem()
        # self.qc.run_depth_in_sink()


        # path_cline = r"Z:\01 ZRD\03 PROJECTS\199805002 Test Riv\03 GIS\99 WKNG\03 VECTOR\CL_SHP.shp"
        # gdf_out = cross_sections_from_file(path_cline, 500, 500, crs=self.proj_crs)

        # insert_table(gdf_out, 'xs_rw', self.schema, if_exists='replace', srid=self.proj_crs)

        path_poly = r"Z:\01 ZRD\03 PROJECTS\199805002 Test Riv\03 GIS\99 WKNG\03 VECTOR\largevec.shp"
        gdf_poly = gpd.read_file(path_poly)


        results = process_batch(
            gdf_poly,
            id_col='reach_id',
            spacing=2000, half_width=5000, 
            segment_length=500, simplify_tolerance=0,
            trim_ends=500,
            progress_every=200,
        )
        print(results)
        print(len(results['centerlines'].geometry.iloc[0].coords), results['centerlines'].length)

        write_batch_results(results, insert_table, self.schema, table_prefix='xs_', srid=self.proj_crs)

        dem_path = r"Z:\01 ZRD\03 PROJECTS\199805002 Test Riv\03 GIS\99 WKNG\04 RASTER\EG_CLIP.tif"
        inun_path = r"Z:\01 ZRD\03 PROJECTS\199805002 Test Riv\03 GIS\99 WKNG\04 RASTER\inundation.tif"

        pts = analyze_cross_sections(results['cross_sections'], dem_path, id_col='xs_id', station_col='station')
        insert_table(pts, 'xs_bank_pts', self.schema, if_exists='replace', srid=self.proj_crs)

        centerline_geom = results['centerlines'].geometry.iloc[0]
        build_inundation_raster(results['cross_sections'], pts, centerline_geom, dem_path, inun_path, wse_mode='average')



if __name__ == "__main__":
    z = ZRDFLOOD()
