import os, sys
import json

import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio as rio

from rasterio.features import shapes
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import shape

from osgeo import gdal
gdal.SetConfigOption('SHAPE_RESTORE_SHX', 'YES')

from utils.utils import safe_remove, area_units, get_cell_size
from db.db import make_schema, insert_table

path = os.path.normpath(r"C:/Users/desro/AppData/Roaming/QGIS/QGIS4/profiles/default/python/plugins/whitebox_workflows_for_qgis")
if path not in sys.path and os.path.exists(path):
    sys.path.append(path)

from qgis.core import QgsApplication
QgsApplication.setPrefixPath(os.environ.get("QGIS_PREFIX_PATH", "C:/OSGeo4W/apps/qgis"), True)
qgs = QgsApplication([], False)
qgs.initQgis()

plugins_path = os.path.join(QgsApplication.pkgDataPath(), "python", "plugins")
if plugins_path not in sys.path:
    sys.path.append(plugins_path)

from qgis import processing
from processing.core.Processing import Processing
Processing.initialize()

from whitebox_workflows import WbEnvironment

wbe = WbEnvironment()
wbe.verbose = False
wbe.max_procs = -1
wbe.working_directory = os.path.normpath(r"Z:\01 ZRD\05 GIS\03 RASTER\03County\Bexar\stratmap21-28cm-50cm-bexar-travis_2998461_dem")



class PARAMSWS:
    def __init__(self, zhnh):
        self.proj_name = zhnh.proj_name
        self.proj_num = zhnh.proj_num

        self.in_rast_d8point = None
        self.in_rast_filled = None
        self.out_vect_streams = None
        self.in_rast_acc = None

        self.zhnh = zhnh
        self.zcncfg = self.zhnh.PROJ_DICT

        self.schema = make_schema(self.proj_num)

        self.path_filled_dem = self.zcncfg['path_proj'] + '\\03 GIS\\99 WKNG\\04 RASTER\\DEM_filled.tif'
        self.path_d8_pointer = self.zcncfg['path_proj'] + '\\03 GIS\\99 WKNG\\04 RASTER\\d8_Pointer.tif'
        self.path_d8_flow_accum = self.zcncfg['path_proj'] + '\\03 GIS\\99 WKNG\\04 RASTER\\d8_Flow_Accum.tif'
        self.path_extr_streams = self.zcncfg['path_proj'] + '\\03 GIS\\99 WKNG\\04 RASTER\\Extr_Streams.tif'
        self.path_ws_from_of_rast = self.zcncfg['path_proj'] + '\\03 GIS\\99 WKNG\\04 RASTER\\ws_from_of_rast.tif'
        self.path_ws_from_of_vec = self.zcncfg['path_proj'] + '\\03 GIS\\99 WKNG\\03 VECTOR\\ws_from_of_vec.shp'

        self.path_sb_rast = self.zcncfg['path_proj'] + '\\03 GIS\\99 WKNG\\04 RASTER\\sb_rast.tif'
        self.path_sb_vec = self.zcncfg['path_proj'] + '\\03 GIS\\99 WKNG\\03 VECTOR\\sb_vec.shp'
        self.path_sb_vec_gpkg = self.zcncfg['path_proj'] + '\\03 GIS\\99 WKNG\\03 VECTOR\\sb_vec.gpkg'

        self.path_da_flowlines = self.zcncfg['path_proj'] + '\\03 GIS\\99 WKNG\\03 VECTOR\\da_flowlines.shp' 
        self.path_da_flowl_gpkg = self.zcncfg['path_proj'] + '\\03 GIS\\99 WKNG\\03 VECTOR\\da_flowlines.gpkg' 
        self.path_da_flowpaths = self.zcncfg['path_proj'] + '\\03 GIS\\99 WKNG\\03 VECTOR\\da_flowpaths.shp' 
        self.pour_point = self.zcncfg['path_proj'] + '\\03 GIS\\99 WKNG\\03 VECTOR\\pnt_pour_snapd.shp'

        # atexit.register(self.cleanup_exit)



    def update_ws(self):
        self.prep_dem()
        self.d8_point_accum()
        self.extract_streams()

        # self.chk_pour_pnt()

        self.run_watershed()
        # self.chk_vec_cleanup()

        self.run_subbasin()
        self.subbasin_cleanup()

        self.run_longest_path()
        # self.flowp_bld_gpkg()


    def cleanup_exit(self):
        self.flowp_bld_gpkg()
        return 1

    def chk_time(self, path_chk, path_tgt):
        '''
            RETURN TRUE IF UP TO DATE
        '''
        time_chk = os.path.getctime(path_chk)
        if os.path.exists(path_tgt):
            time_tgt = os.path.getmtime(path_tgt)
            if time_chk > time_tgt:
                return True
            else:
                return False
        else:
            return False


    def chk_centroid(self, path_chk, path_tgt):
        '''
            RETURN TRUE IF SIZES MATCH
        '''
        if os.path.basename(path_chk).split('.')[1] == 'tif':
            with rio.open(path_chk) as src:
                left, bottom, right, top = src.bounds
                cent_x_chk = (left + right) / 2
                cent_y_chk = (bottom + top) / 2
        else:
            cent_chk = gpd.read_file(path_chk)
        

        if os.path.basename(path_tgt).split('.')[1] == 'tif':
            with rio.open(path_tgt) as src:
                left, bottom, right, top = src.bounds
                cent_x_tgt = (left + right) / 2
                cent_y_tgt = (bottom + top) / 2
        else:
            cent_tgt = gpd.read_file(path_tgt)

        delt = cent_x_chk - cent_x_tgt
        if delt > 1000 or delt < -1000:
            return False
        else:
            return True


    def prep_dem(self):
        print('\n RUNNING prep_dem....')

        if os.path.exists(self.zcncfg['path_dem']):
            in_rast_dem = wbe.read_raster(self.zcncfg['path_dem'])
        
            out_rast_filled = wbe.hydrology.fill_depressions_wang_and_liu(
                # dem=in_rast_dem, fix_flats=True, flat_increment=0.001
                dem=in_rast_dem, fix_flats=True, flat_increment=0.001,
            )
            
            if os.path.exists(self.path_filled_dem):
                safe_remove(self.path_filled_dem)

            wbe.write_raster(out_rast_filled, self.path_filled_dem)

            print('\n prep_dem COMPLETE')
        else: 
            print(f'\n {self.zcncfg['path_dem']} NOT FOUND....')
            sys.exit()


    def d8_point_accum(self):
        self.in_rast_filled = wbe.read_raster(self.path_filled_dem)

        out_rast_d8point = wbe.hydrology.d8_pointer(dem=self.in_rast_filled, esri_pntr=False)

        if os.path.exists(self.path_d8_pointer):
            safe_remove(self.path_d8_pointer)

        wbe.write_raster(out_rast_d8point, self.path_d8_pointer)
        self.in_rast_d8point = out_rast_d8point

        out_rast_acc = wbe.hydrology.d8_flow_accum(
            input=self.in_rast_filled,
            out_type="cells",
            log_transform=False,
            clip=False,
            input_is_pointer=False,
            esri_pntr=False
        )

        
        # out_rast_acc = wbe.hydrology.d8_flow_accum(
        #     input=self.in_rast_filled,
        #     out_type="sca",
        #     log_transform=False,
        #     clip=False,
        #     input_is_pointer=False,
        #     esri_pntr=False
        # )

        if os.path.exists(self.path_d8_flow_accum):
            safe_remove(self.path_d8_flow_accum)

        wbe.write_raster(out_rast_acc, self.path_d8_flow_accum)
        print('\n d8_pointer COMPLETE')


    def extract_streams(self):
        self.in_rast_acc = wbe.read_raster(self.path_d8_flow_accum)

        cell_size = get_cell_size(self.path_d8_pointer, self.zhnh.proj_crs)

        threshold_acres = 5.0
        threshold_cells_equiv = int((threshold_acres * 43560) / cell_size)

        # streams = flow_accum > threshold  # or 
        out_rast_streams = wbe.streams.extract_streams(flow_accumulation=self.in_rast_acc, threshold=threshold_cells_equiv)

        # out_rast_streams = wbe.streams.extract_streams(
        #     flow_accumulation=self.in_rast_acc,
        #     threshold=10000,
        #     zero_background=False
        # )

        if os.path.exists(self.path_extr_streams):
            safe_remove(self.path_extr_streams)

        wbe.write_raster(out_rast_streams, self.path_extr_streams)

        in_rast_streams = wbe.read_raster(self.path_extr_streams)
        self.out_vect_streams = wbe.streams.raster_streams_to_vector(
            d8_pntr=self.in_rast_d8point,
            streams_raster=in_rast_streams,
            esri_pntr=False
        )
        wbe.write_vector(self.out_vect_streams, self.path_da_flowlines)

        gdf_fl = gpd.read_file(self.path_da_flowlines)
        if gdf_fl.crs == None:
            gdf_fl.set_crs(self.zhnh.proj_crs, inplace=True)
        else:
            gdf_fl.to_crs(self.zhnh.proj_crs, inplace=True)

        gdf_fl.to_file(self.path_da_flowl_gpkg, layer='da_flowlines', driver="GPKG", mode='w')
        insert_table(gdf_fl, 'da_flowlines', self.schema, if_exists='replace')

        print(f'\n extract_streams COMPLETE with ', len(gdf_fl), ' FEATURES')



    def chk_pour_pnt(self):
        outf = gpd.read_file(self.zhnh.path_prj, layer='pnt_pour')

        in_vecpnt_pour = wbe.read_vector(self.zhnh.path_pnt_pour)
        pp = wbe.hydrology.jenson_snap_pour_points(
            pour_pts=in_vecpnt_pour,
            streams=self.in_rast_acc,
            snap_dist=1,
            output=self.zhnh.path_pnt_pour
        )

        if os.path.exists(self.path_d8_pointer):
                    safe_remove(self.path_d8_pointer)
        
        wbe.write_vector(pp, self.zhnh.path_pnt_pour)

        gdf_pp = gpd.read_file(self.zhnh.path_pnt_pour)
        gdf_pp.to_file(self.zhnh.path_prj, layer='pnt_pour', driver="GPKG", mode='w', overwrite=True)
        print('\n chk_pour_pnt COMPLETE WITH UPDATES')



    def run_watershed(self):
        #5. Create a point shapefile with one or more points at the outlet locations using QGIS. The outlet locations
        #  are the points of analysis. Ideally the points should be located on the stream network defined at Step 4
        #  (blue pixels on Figure 6), however it is difficult or sometimes impossible to put this point on stream network
        #  at first attempt. Whitebox Tools (WBT) Plugin has a function to move any initial outlet location points to the 
        #  extracted stream network – JensonSnapPourPoints (similar to TauDEM Plugin’s Move Outlets to Streams). 
        if os.path.exists(self.path_ws_from_of_rast):
            try:
                os.remove(self.path_ws_from_of_rast)
            except:
                print("ws_from_of_rast IS OPEN!!!.")
                sys.exit()

        if os.path.exists(self.path_d8_pointer):
            in_rast_d8point = wbe.read_raster(self.path_d8_pointer)
        else:
            print('172.................')

        self.cpy_pour_to_shp()
        if os.path.exists(self.zhnh.path_pnt_pour):
            in_vecpnt_pour = wbe.read_vector(self.zhnh.path_pnt_pour)
        else:
            print('178.................')

        try:
            #6 Run Watershed delineation (Figure 9): After the final outlet point shapefile is created at Step 5, run Watershed 
            # of Whitebox Tools (WBT) to delineate the watershed (upstream area) of the outlets. The input files are the
            # D8 Pointer file and the relocated outlet point shapefile; while the output file is a watershed raster file
            
            if self.chk_within(self.path_d8_pointer, self.zhnh.path_pnt_pour):
                out_rast_ws = wbe.hydrology.watershed(d8_pntr=in_rast_d8point, pour_pts=in_vecpnt_pour, esri_pntr=False)

                stats_path = os.path.join(wbe.working_directory, "ws_stats.json")
                wbe.raster.raster_summary_stats(input=out_rast_ws, output=stats_path)
                stats = wbe.raster.raster_summary_stats(input=out_rast_ws)


                with open(stats_path) as f:
                    stats = json.load(f)
                n_cells = stats['count']
                
                if n_cells > 0:

                    if os.path.exists(self.path_ws_from_of_rast):
                        safe_remove(self.path_ws_from_of_rast)

                    wbe.write_raster(out_rast_ws, self.path_ws_from_of_rast)
                    self.conv_rast_vec(self.path_ws_from_of_rast, self.path_ws_from_of_vec)
                    print(f'\n ws_from_of CREATED with {n_cells} CELLS')

                    self.flowp_bld_gpkg()
                else:
                    print("!!!!!!!!!!!!! ERROR out_rast_ws FAIL VIA raster_summary_stats -- CHECK pnt_pour IS ON STREAM")
                    sys.exit()
            else:
                self.chk_pour_pnt()
                print("!!!! ERROR out_rast_ws FAIL VIA chk_within....POUR POINT REBUILT, RUN AGAIN !!!!")
                sys.exit()

        except Exception as e:
            print("ERROR: ", e)



    def run_subbasin(self):
        in_rast_d8point = wbe.read_raster(self.path_d8_pointer)
        in_streams = wbe.read_raster(self.path_extr_streams)

        out_rast_sb = wbe.hydrology.subbasins(d8_pntr=in_rast_d8point, 
                                            streams=in_streams, 
                                            esri_pntr=False)

        if os.path.exists(self.path_sb_rast):
            safe_remove(self.path_sb_rast)
        
        wbe.write_raster(out_rast_sb, self.path_sb_rast)

        # --- diagnostic: check actual unique subbasin IDs in the raster ---
        # with rio.open(self.path_sb_rast) as src:
        #     sb_array = src.read(1)
        #     nodata = src.nodata

        # if nodata is not None:
        #     valid = sb_array[sb_array != nodata]
        # else:
        #     valid = sb_array.ravel()

        # vals, counts = np.unique(valid, return_counts=True)
        # print(f"\n[run_subbasin] unique raster values: {len(vals)}  (nodata={nodata})")
        # for v, c in sorted(zip(vals, counts), key=lambda x: -x[1])[:50]:
        #     print(f"  id={v}  pixel_count={c}")
        # ---------------------------------------------------------------

        if os.path.exists(self.path_sb_rast):
            params = {
                'INPUT': self.path_sb_rast,
                'BAND': 1,
                'FIELD': 'Band 1',
                'EIGHT_CONNECTEDNESS': False,
                'OUTPUT': self.path_sb_vec_gpkg
            }
            processing.run("gdal:polygonize", params)


    def subbasin_cleanup(self):
        ws_tmps = gpd.read_file(self.path_sb_vec_gpkg)

        if len(ws_tmps) > 0:
            ws_tmps.to_crs(self.zhnh.proj_crs, inplace=True)

            ws_tmps = ws_tmps[ws_tmps.geometry.notna() & ws_tmps.geometry.is_valid]
            ws_tmps.rename(columns={'Band 1': 'name'}, inplace=True)
            ws_tmps = ws_tmps[ws_tmps['name'] != 0]

            ws_tmps = ws_tmps.dissolve(by='name', as_index=False)

            ws_tmps.crs.axis_info[0].unit_name

            if 'foot' in area_units(ws_tmps):
                ws_tmps['area_ft'] = ws_tmps.geometry.area.astype(int)
                ws_tmps['area_ac'] = round(ws_tmps.geometry.area / 43560, 2)
                ws_tmps['area_sqmi'] = round(ws_tmps.geometry.area / 43560 / 640, 4)

            ws_tmps = ws_tmps.drop_duplicates(subset=['geometry'])
            ws_tmps.to_file(self.zhnh.path_prj, layer='sb_vec', driver='GPKG', mode='w', overwrite='yes')

            insert_table(ws_tmps, 'sb_vec', self.schema, if_exists='replace')

            os.remove(self.path_sb_vec_gpkg)

            print("\n subbasin_cleanup COMPLETE with ", len(ws_tmps), " FEATURES")
        else:
            print("\n subbasin_cleanup FAIL....NO FEATURES CREATED")




    def conv_rast_vec(self, path_rast, path_vec):
        if os.path.exists(path_rast):
            with rio.open(path_rast) as src:            
                data = src.read(1, masked=True)
                shape_gen = ((shape(s), v) for s, v in shapes(data, transform=src.transform))

                df = pd.DataFrame(shape_gen, columns=['geometry', 'class'])
                gdf = gpd.GeoDataFrame(df["class"], geometry=df.geometry, crs=src.crs)

            # gdf.set_crs(self.zhnh.proj_crs, inplace=True)
            gdf = gdf.dissolve()
            gdf.to_file(path_vec, mode='w')

            insert_table(gdf, 'ws_from_of_vec', self.schema, if_exists='replace')
            # gdf.to_file(self.zhnh.path_prj, layer='ws_from_of_vec', driver="GPKG", mode='w', overwrite=True)

        else:
            print("\n !!!!!!!!!!!!!!!!!!!!!! ERROR conv_rast_vec FAIL !!!!!!!!!!!!!!!!!!!!!! \n")
            # sys.exit()





    def run_longest_path(self):
        if os.path.exists(self.path_ws_from_of_rast):
            self.in_rast_ws = wbe.read_raster(self.path_ws_from_of_rast)
            out_vect_flowpaths = wbe.hydrology.longest_flowpath(
                dem=self.in_rast_filled,
                basins=self.in_rast_ws,
                output=self.path_da_flowpaths
            )
            self.val_vector_crs(self.path_da_flowpaths)

            gdf_fl = gpd.read_file(self.path_da_flowpaths)
            insert_table(gdf_fl, 'tc_path', self.schema ,if_exists='replace')

            if not os.path.exists(self.path_da_flowpaths):
                print("!!!!!!!!!!!!!!!!!!!!!! ERROR create_flowpaths FAIL !!!!!!!!!!!!!!!!!!!!!!")
            else:
                print("\n run_longest_path COMPLETE")
        else: 
                print("\n ws_from_of_rast FILE DOES NOT EXIST....")


    def flowp_bld_gpkg(self):
        lst_vLyrs = [
            self.zhnh.path_pnt_pour,
            self.path_da_flowpaths, 
            self.path_da_flowlines, 
            self.zcncfg['path_tc_pnts'], 
            self.zcncfg['path_tc_path'], 
            self.path_ws_from_of_vec
        ]
        for vlyr in lst_vLyrs:
            if os.path.exists(vlyr):
                self.gpkg_add_vec(vlyr)
        
        self.gpkg_add_vec(self.zhnh.path_prj, lname='BNDY')
        # lst_rLyrs = [
        #     self.path_d8_pointer
        # ]
        # for rlyr in lst_rLyrs:
        #     self.gpkg_add_rast(rlyr)
        print('\n flowp_bld_gpkg COMPLETE')


    def gpkg_add_vec(self, path, lname = None):
        if not lname:
            lname = os.path.basename(path).split(".")[0]

        if os.path.exists(path):
            gdf = gpd.read_file(path, layer=lname)

            if 'fid' in gdf.columns:
                gdf['fid'] = gdf['fid'].astype(int)
            if 'FID' in gdf.columns:
                gdf['FID'] = gdf['FID'].astype(int)

            gdf.to_file(self.zhnh.path_prj, layer=lname, mode='w', overwrite='yes')


    def gpkg_add_rast(self, path):
        with rio.open(path) as src:
            height, width = src.height, src.width
            # height, width = 100, 100

            params = {
                'driver': 'GTiff',
                'height': height,
                'width': width,
                'count': 1,
                'dtype': 'float32',
                'crs': 'EPSG:2278',
                'APPEND_SUBDATASET': 'YES'
            }


    def val_vector_crs(self, path_src, crs_src=None):
        gdf_tmp = gpd.read_file(path_src)

        if crs_src == None:
            crs_src = self.zhnh.proj_crs

        if gdf_tmp.crs == None:
            gdf_tmp.set_crs(crs_src, inplace=True)
        else:
            gdf_tmp.to_crs(crs_src, inplace=True)
            
        gdf_tmp.to_file(path_src, mode='w')
        


    def val_raster_crs(self, ras_path, out_path):
        with rio.open(ras_path) as src:
            src_crs = {"init": "EPSG:" + str(src.crs.to_epsg())}
            dst_crs = {"init": "EPSG:" + str(self.zhnh.proj_crs)}

            transform, width, height = calculate_default_transform(
                src_crs, 
                dst_crs, 
                src.width, 
                src.height, 
                *src.bounds
            )
            kwargs = src.meta.copy()

            kwargs.update({
                'crs': dst_crs,
                'transform': transform,
                'width': width,
                'height': height
            })

            with rio.open(out_path, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rio.band(src, i),
                        destination=rio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.nearest)


    def cpy_pour_to_shp(self):
        gdf = gpd.read_file(self.zhnh.path_prj, layer='pnt_pour')

        with rio.open(self.path_filled_dem) as src:
            # src_crs = {"init": "EPSG:" + str(src.crs.to_epsg())}
            src_crs = src.crs.to_epsg()

        gdf.to_crs(src_crs, inplace=True)
        try:
            gdf.to_file(self.zhnh.path_pnt_pour, mode='w')
        except Exception as e:
            print('EXCEPTION AT cpy_to_shp', e)
            sys.exit()


    def chk_within(self, path_out, path_in, lyr_name=None):
        if lyr_name:
            src_pnt = gpd.read_file(path_in, layer=lyr_name)
        else:
            src_pnt = gpd.read_file(path_in)


        ext = path_out.rsplit(".", 1)[-1].lower()

        if ext == 'gpkg':
            bndy = gpd.read_file(path_out, layer='BNDY')
            return bool(src_pnt.geometry.within(bndy).all())

        elif ext == 'shp':
            bndy = gpd.read_file(path_out)
            return bool(src_pnt.geometry.within(bndy).all())

        elif ext == 'tif':
            if not os.path.exists(path_out):
                return False

            with rio.open(path_out) as src_out:
                bnd = src_out.bounds
                nodata = src_out.nodata

                coords = [
                    (geom.x, geom.y)
                    for geom in src_pnt.geometry
                ]

                valid_coords = [
                    (x, y) for x, y in coords
                    if bnd.left <= x <= bnd.right and bnd.bottom <= y <= bnd.top
                ]

                if len(valid_coords) < len(coords):
                    return False

                samples = list(src_out.sample(valid_coords, masked=True))

            return all(
                not val[0] is np.ma.masked
                and (nodata is None or val[0] != nodata)
                and val[0] > 0
                for val in samples
            )

        return False
                


    def chk_within_ras(self, path_out, path_in):
        # with rio.open(path_out) as src_out:
            # bnd_out = src_out.bounds

        src_out = gpd.read_file(self.zhnh.path_prj, layer='BNDY')
        bnd_out = src_out.bounds
        area_out = src_out.area

        with rio.open(path_in) as src_in:
            bnd_in = src_in.bounds
            area_in = src_in.BoundingBox







# INSPECT
# Check Raster CRS and Extent
# with rio.open("d8_pntr.tif") as src:
#     print("Raster CRS:", src.crs)
#     print("Raster Bounds:", src.bounds)

# Check Shapefile CRS and Extent
# with fiona.open("snapped_pts.shp") as src:
#     print("Vector CRS:", src.crs)
#     print("Vector Bounds:", src.bounds)