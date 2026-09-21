import os
import dotenv
from pathlib import Path
import configparser
import re

from pathlib import Path
import geopandas as gpd


os.environ['GDAL_DATA'] = r'Z:\10 DEV\zrdhnh\.venv\Lib\site-packages\pyogrio\gdal_data'

# C:\OSGeo4W\apps\Python312\Lib\site-packages

# os.environ['GDAL_DATA'] = os.environ['CONDA_PREFIX'] + r'\Library\share\gdal'
# os.environ['GDAL_DATA'] = os.path.join(f'{os.sep}'.join(sys.executable.split(os.sep)[:-1]), 'share', 'proj')

# import pyproj
# print(pyproj.datadir.get_data_dir())
# projsync --target-dir "C:\OSGeo4W\share\proj" --bbox -106.7,25.8,-93.5,36.5


class CFG:
    def __init__(self, proj_name = None, proj_num = None):
        self.current_file = Path(__file__).resolve()
        
        self.env_file = Path("Z:/10 DEV/zrdhnh/cfg/proj.env")

        self.ptype = None;
        self.ppath = None;

        if not proj_name:
            self.cfg_get_env()
        else:
            self.proj_name = proj_name
            self.proj_num = proj_num

        proj_q = get_current_qgis_project()
        if not proj_q is None and len(proj_q) > 0:
            self.proj_name = next(p for p in Path(proj_q).parts if p[:6].isdigit())
            self.proj_num = int(self.proj_name.split(' ')[0])
        else:
            self.proj_name = proj_name
            self.proj_num = proj_num

        self.path_proj = ''
        self.proj_co = ''

        self.proj_crs = ''
        self.PROJ_DICT = ''

        self.path_zrd = "Z://01 ZRD//05 GIS//03 VECTOR//ZRD//ZRD.shp"
        

        self.path_pnt_pour = ''
        self.path_da_flowpaths = ''
        self.path_da_flowlines = ''
        self.path_nhd = ''

        # path_prj = "Z://01 ZRD//05 GIS//05 PRJ//NAD_1983_TCMS_Albers_FtUS.prj"
        # with open(path_prj) as file:
        #     wkt_crs = file.readline()
        # self.proj_crs = CRS.from_wkt(wkt_crs)

        self.ws_locked = True
        self.gdf_proj = None

        self.cfg_get_proj_names()
        self.cfg_upd_env()


    def cfg_get_env(self):
        dotenv.load_dotenv(self.env_file)
        self.proj_name = os.getenv("PROJ_NAME")
        self.proj_num = os.getenv("PROJ_NUM")

    def cfg_upd_env(self):
        dotenv.load_dotenv(self.env_file)
        dotenv.set_key(self.env_file, "PROJ_NAME", self.proj_name)
        dotenv.set_key(self.env_file, "PROJ_NUM", str(self.proj_num))
        dotenv.set_key(self.env_file, "PATH_PRJ", str(self.path_prj))


    def cfg_get_proj_names(self):

        if isinstance(self.proj_num, int):
            self.proj_num = self.proj_num
        else:
            self.cfg_get_env()

        proj_feat = self.cfg_get_projshape()
        try:
            self.proj_crs = int(proj_feat['proj_crs'].values[0]) if proj_feat is not None else None

        except ValueError:
            print(f'SET PROJECT CRS IN ZRD.shp for {self.proj_name}')


        if proj_feat is not None:
            if len(proj_feat['proj_pname'].values[0]) > 0:
                self.proj_name = proj_feat['proj_pname'].values[0]
                self.proj_co = proj_feat['coname'].values[0]
                self.gdf_proj = self.gdf_proj.to_crs(self.proj_crs)

                try:
                    self.proj_crs = int(proj_feat['proj_crs'].values[0])
                except ValueError:
                    self.proj_crs = int(2278)
                

        else:
            print(f'PROJECT NAME NOT FOUND IN ZRD.shp.......{self.proj_name}')
            self.cfg_get_env()
        
        if self.gdf_proj is not None and len(self.gdf_proj) > 0:
            self.ptype = self.gdf_proj['notes'].values[0]
        else:
            self.ptype = None

        if self.ptype == None or self.ptype == 'proposal':
            self.ppath = '02 PROPOSALS'
        else: 
            self.ppath = '03 PROJECTS'

        if self.proj_co == 'ZRD':
            self.path_proj = F'Z:\\01 ZRD\\' + self.ppath + '\\' + self.proj_name
            # self.path_proj = 'Z:\\01 ZRD\\02 PROPOSALS\\' + self.proj_name

        elif self.proj_co == 'SG':
            self.path_proj = 'Z:\\04 SG\\' + self.ppath + '\\' + self.proj_name

        else:
            self.proj_co == 'ZRD'
            print(f'Project company not found in ZRD.shp for {self.proj_name}')


        # print(f'self.ptype.......{self.ptype}')
        # print(f'self.ppath.......{self.ppath}')
        # print(f'self.path_proj.......{self.path_proj}')

        self.cfg_bld_proj_dict()
        print('\n============================================',
              '\n==========', self.proj_name, '==========',
              '\n============================================')
              



    def cfg_bld_proj_dict(self):
        self.path_proj_map = self.path_proj + '\\03 GIS\\01 MAPS\\QMAP_SETUP.qgz'

        self.path_prj = self.path_proj + '\\03 GIS\\03 VECTOR\\PROJ.gpkg'
        self.path_bndy = self.path_proj + '\\03 GIS\\03 VECTOR\\PROJ.gpkg|layername=BNDY'
        
        self.path_nhd = self.path_proj + '\\03 GIS\\03 VECTOR\\PROJ_NHD.gpkg'
        self.path_model = self.path_proj + '\\03 GIS\\03 VECTOR\\PROJ_MODEL.gpkg'

        self.path_dem = self.path_proj + '\\03 GIS\\99 WKNG\\04 RASTER\\Output\\dem_clip.tif'
        self.path_cog = self.path_proj + '\\03 GIS\\99 WKNG\\04 RASTER\\dem_cog.cog'
        self.path_eg = self.path_proj + '\\03 GIS\\04 RASTER\\EGLIDAR.tif'
        self.path_fg = self.path_proj + '\\03 GIS\\04 RASTER\\FG.tif'

        self.path_tc_pnts = self.path_proj + '\\03 GIS\\99 WKNG\\03 VECTOR\\tc_pnts.shp'
        self.path_tc_path = self.path_proj + '\\03 GIS\\99 WKNG\\03 VECTOR\\tc_path.shp'

        self.path_da_flowpaths = self.path_proj + '\\03 GIS\\99 WKNG\\03 VECTOR\\da_flowpaths.shp'
        self.path_da_flowlines = self.path_proj + '\\03 GIS\\99 WKNG\\03 VECTOR\\da_flowlines.shp'

        self.path_pnt_pour = self.path_proj + '\\03 GIS\\99 WKNG\\03 VECTOR\\pnt_pour.shp'
        self.path_of_ini = self.path_proj + '\\03 GIS\\99 WKNG\\03 VECTOR\\_of_ini.shp'


        self.PROJ_DICT = {
            'path_zrd': self.path_zrd,
            'proj_name': self.proj_name,
            'proj_num': self.proj_num,
            'path_proj': self.path_proj,
            'proj_co': self.proj_co,
            'proj_crs': self.proj_crs,
            'path_proj_map': self.path_proj_map,
            'path_prj': self.path_prj,
            'path_bndy': self.path_bndy,
            'path_nhd': self.path_nhd,
            'path_dem': self.path_dem,
            'path_cog': self.path_cog,
            'path_dem': self.path_dem,
            'path_eg': self.path_eg,
            'path_fg': self.path_fg,
            'path_tc_pnts': self.path_tc_pnts,
            'path_tc_path': self.path_tc_path,
            'path_da_flowpaths': self.path_da_flowpaths,
        }

        if not os.path.exists(self.path_prj):
            Path(self.path_prj).parent.mkdir(parents=True, exist_ok=True)
            if self.gdf_proj is not None:
                self.gdf_proj.to_file(self.path_prj, layer='pnt_pour', driver="GPKG")
        
        if self.PROJ_DICT:
            return True
        else:
            print(f'FILE DOES NOT EXIST!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            return False



    def cfg_get_projshape(self):
        gdf = gpd.read_file(self.path_zrd)
        if gdf.geometry.name != 'geom':
            gdf = gdf.rename_geometry('geom')

        gdf_attr_only = gdf.drop(columns='geometry', errors='ignore')

        if self.proj_num is not None:
            self.gdf_proj = gdf[gdf['projno'] == str(self.proj_num)]
            proj_found = gdf_attr_only[gdf_attr_only['projno'] == str(self.proj_num)]
            if len(proj_found) > 1:
                self.gdf_proj = gdf[gdf['proj_pname'] == str(self.proj_name)]
                proj_found = gdf_attr_only[gdf_attr_only['proj_pname'] == str(self.proj_name)]
        else:
            print('\nPROJSHAPE NOT FOUND.....OPEN PROJECT IN QGIS.shp \n')
            print(f'{self.proj_num}')
            print(f'{self.proj_name} \n')
            raise ValueError('proj_num is None — open project in QGIS')

        if len(proj_found) > 0:
            return proj_found
        else:
            print('\nPROJSHAPE NOT FOUND.....ADD PROJECT TO ZRD.shp')
            print(f'{self.proj_num}')
            print(f'{self.proj_name} \n')
            return None


    def cfg_get_projpoint(self):
        path_object = Path(self.path_zrd)
        gdf = gpd.read_file(path_object)
        
        gdf_attr_only = gdf.drop(columns='geometry')

        # rslt = gdf_attr_only[gdf_attr_only['projname'] == self.proj_name]

        if self.proj_num is not None:
            self.gdf_proj = gdf[gdf['projno'] == str(self.proj_num)]
            proj_found = gdf_attr_only[gdf_attr_only['projno'] == str(self.proj_num)]

        else:
            print('PROJSHAPE NOT FOUND @ cfg_get_projpoint .....ADD PROJECT TO ZRD.shp')
            raise ValueError('proj_num is None in cfg_get_projpoint')

        if len(proj_found) > 0:
            return proj_found
        else:
            return ''



def get_current_qgis_project(profile: str = None) -> str | None:
    import psutil

    # Check if QGIS is running
    qgis_running = any(
        'qgis' in p.name().lower()
        for p in psutil.process_iter(['name'])
    )
    if not qgis_running:
        return None

    if profile:
        ini = Path.home() / f"AppData\\Roaming\\QGIS\\QGIS4\\profiles\\{profile}\\QGIS\\QGIS4.ini"
    else:
        profiles_dir = Path.home() / "AppData\\Roaming\\QGIS\\QGIS4\\profiles"
        if not profiles_dir.exists():
            return None
        profiles = [p for p in profiles_dir.iterdir() if p.is_dir() and p.name != "default"]
        ini = (profiles[0] / "QGIS" / "QGIS4.ini") if profiles else (profiles_dir / "default" / "QGIS" / "QGIS4.ini")

    if not ini.exists():
        return None

    config = configparser.RawConfigParser()
    config.read(ini, encoding="utf-8")

    if "UI" not in config:
        return None

    projects = {}
    for key, val in config["UI"].items():
        m = re.match(r"recentprojects\\(\d+)\\path", key)
        if m:
            if "project_default.qgs" not in val:
                projects[int(m.group(1))] = val

    return projects.get(min(projects)) if projects else None