gdaldem slope /home/pf/pfstaff/projects/Daudt_DeepSnow/dataset/DEM/swissalti3d_2017_LV95.tif /home/pf/pfstaff/projects/Daudt_DeepSnow/dataset/DEM/swissalti3d_2017_LV95_slope.tif -of GTiff -co TILED=YES -alg Horn

gdaldem aspect /home/pf/pfstaff/projects/Daudt_DeepSnow/dataset/DEM/swissalti3d_2017_LV95.tif /home/pf/pfstaff/projects/Daudt_DeepSnow/dataset/DEM/swissalti3d_2017_LV95_aspect.tif -of GTiff -co TILED=YES -alg Horn -zero_for_flat

gdaldem TRI /home/pf/pfstaff/projects/Daudt_DeepSnow/dataset/DEM/swissalti3d_2017_LV95.tif /home/pf/pfstaff/projects/Daudt_DeepSnow/dataset/DEM/swissalti3d_2017_LV95_TRI.tif -of GTiff -co TILED=YES

gdaldem TPI /home/pf/pfstaff/projects/Daudt_DeepSnow/dataset/DEM/swissalti3d_2017_LV95.tif /home/pf/pfstaff/projects/Daudt_DeepSnow/dataset/DEM/swissalti3d_2017_LV95_TPI.tif -of GTiff -co TILED=YES

# IMPORTANT: Aspect needs to be converted into sine and cosine, e.g. using QGIS raster calculator. Be careful with degrees and radians. 

