
from qgis.PyQt.QtCore import QVariant
# from qgis.PyQt.QtCore import QMetaType

from qgis.core import (
    QgsProject,
    QgsFields,
    QgsExpressionContext,
    QgsExpressionContextUtils,
    QgsExpression,
    QgsCoordinateReferenceSystem,
    QgsFeature,
    edit
)

from qgis.utils import *

import geopandas as gpd
from pathlib import Path

from utils.utils import *

from osgeo import gdal
gdal.SetConfigOption('SHAPE_RESTORE_SHX', 'YES')

QgsApplication.setPrefixPath('C:\\OSGeo4W\\apps\\qgis-ltr\\', True)
qgs = QgsApplication([], False)
qgs.initQgis()

from qgis import processing
from processing.core.Processing import Processing
Processing.initialize()

# gisPath = "Z:\\10 DEV\\zrdhnh"

# if gisPath not in sys.path:
#     sys.path.append(gisPath)



class ZUTILSLYR:
    def __init__(self, proj_dict):
        self.zproj = proj_dict

        # self.zproj = ZCNCFG(proj_name).PROJ_DICT


    def lyr_mkr_proj(self):
        layerName = 'PROJ'
        for layer in QgsProject.instance().mapLayers().values():
                if layer.name()==layerName:
                    QgsProject.instance().removeMapLayers( [layer.id()] )

        layer_type = 'Polygon' 

        layerFields = QgsFields()
        layerFields.append(QgsField('ID', QVariant.Int))
        layerFields.append(QgsField('NAME', QVariant.String))
        layerFields.append(QgsField('NO', QVariant.String))
        layerFields.append(QgsField('PRJ', QVariant.Double))
        layerFields.append(QgsField('AREA', QVariant.Double))

        # crs = QgsCoordinateReferenceSystem("EPSG:2278") 
        crs = QgsCoordinateReferenceSystem(self.zproj['proj_crs']) 

        layer = QgsVectorLayer(f"{layer_type}?crs={crs.authid()}", layerName, "memory")

        # Add fields to the layer
        data_provider = layer.dataProvider()
        data_provider.addAttributes(layerFields)
        layer.updateFields()

        expression = '$area'
        context = QgsExpressionContext()
        context.appendScopes(QgsExpressionContextUtils.globalProjectLayerScopes(layer))

        with edit(layer):
            for feature in layer.getFeatures():
                context.setFeature(feature)
                area_value = QgsExpression(expression).evaluate(context)
                feature["AREA"] = area_value
                layer.updateFeature(feature)

        #Adding a polygon feature
        feature = QgsFeature()
        feature.setAttributes([1, layerName])

        # x1 = 2369176
        # y1 = 13826720
        # x2 = 2377110
        # y2 = 13833548

        # polygon = QgsGeometry.fromPolygonXY([[QgsPointXY(x1, y1), QgsPointXY(x2, y1), QgsPointXY(x2, y2), QgsPointXY(x1,y2)]])
        # # polygon = QgsGeometry.fromPolygonXY([[QgsPointXY(1, 1), QgsPointXY(2, 2), QgsPointXY(2, 1), QgsPointXY(1,1)]])

        # feature.setGeometry(polygon)
        # data_provider.addFeatures([feature])

        # Save the layer to a shapefile
        path_proj =  self.zproj['path_proj'] +  '\\03 GIS\\99 WKNG\\03 VECTOR\\proj.gpkg'
        QgsVectorFileWriter.writeAsVectorFormatV3(layer, path_proj, "UTF-8", crs, "ESRI Shapefile")

        # Add the layer to the map
        QgsProject.instance().addMapLayer(layer)

        layer.setDataSource(self.zproj['path_bndy'], layer.name(), 'ogr')
        layer.reload()

        print('\n lyr_mkr_proj COMPLETE ')




    def attr_add(self, path_lyr):
        lName = path_lyr.split("\\")[-1].split(".")[0]
        vLayer = QgsVectorLayer(path_lyr, lName, 'ogr')

        # vLayer = iface.activeLayer()
        vLayer.selectAll()

        exp = QgsExpression('$area')
        context = QgsExpressionContext()
        context.appendScopes(QgsExpressionContextUtils.globalProjectLayerScopes(vLayer))

        with edit(vLayer):
            for f in vLayer.getFeatures():
                context.setFeature(f)
                f['AREA_AC'] = exp.evaluate(context)
                f['LATITUDE'] = 22.22222222222
                vLayer.updateFeature(f)

        print(f['AREA'])


    def lyr_dissolve(self, path):
        lyr_in = QgsVectorLayer(path)

        if lyr_in.featureCount() > 0:
            lyr_out = self.zproj['path_proj'] + '\\03 GIS\\99 WKNG\\03 VECTOR\\lyr_out.gpkg'

            processing.run('native:dissolve',
                {'INPUT':lyr_in,
                 'DISSOLVE': True,
                    'OUTPUT': lyr_out
                })
        return lyr_out
    
    
