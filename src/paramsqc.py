import os
import sys
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon, LineString

from cfg import CFG
from db.db import get_table, make_schema, insert_table

from paramsws import PARAMSWS

from whitebox_workflows import WbEnvironment

wbe = WbEnvironment()
wbe.verbose = False
wbe.max_procs = -1




class PARAMSQC:
    def __init__(self, zhnh):
        self.current_file = Path(__file__).resolve()

        self.proj_name = zhnh.proj_name
        self.proj_num = zhnh.proj_num

        self.zhnh = zhnh
        

        self.schema = make_schema(self.proj_num)

        self.output_folder = self.zhnh.path_proj +  '\\03 GIS\\99 WKNG\\04 RASTER'





    def run_depth_in_sink(self):
        wbe.working_directory = os.path.normpath(self.zhnh.path_proj +  '\\03 GIS\\99 WKNG\\04 RASTER')

        # wbe.hydrology.depth_in_sink(
        #     dem=self.zhnh.path_dem,
        #     output="depression_depth.tif",
        #     zero_background=False   # True gives 0 instead of NoData outside sinks
        # )

        # help(wbe.zonal_statistics)

        dem_ft = wbe.read_raster(self.zhnh.path_eg)

        depth_ft = wbe.hydrology.depressions_storage.depth_in_sink(dem=dem_ft)

        meta = dem_ft.metadata()
        cell_x = meta.resolution_x
        cell_y = meta.resolution_y
        cell_area_sqft = cell_x * cell_y

        volume_per_cell = depth_ft * cell_area_sqft

        d8_pntr = wbe.hydrology.flow_routing.d8_pointer(dem=dem_ft)
        basins  = wbe.hydrology.watersheds_basins.basins(d8_pntr=d8_pntr)

        basin_volume_total = wbe.raster.general.zonal_statistics(
            input=volume_per_cell,
            features=basins,
            stat_type='total',
            zero_is_background=True,
            output="basin_volume_total_cuft.tif"
        )

        wbe.write_raster(depth_ft, "depression_depth_ft.tif")
        wbe.write_raster(basins, "depression_basins.tif")

        print(f'\n run_depth_in_sink COMPLETE....')




if __name__ == "__main__":

    proj_num = sys.argv[1] if len(sys.argv) > 1 else ''

    if proj_num:
        cfg = CFG(proj_num=proj_num)
    else:
        cfg = CFG(proj_num=proj_num, proj_name=f'{proj_num} Test Project')


    qc = PARAMSQC(cfg)