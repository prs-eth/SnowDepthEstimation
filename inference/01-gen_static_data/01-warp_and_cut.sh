cp /home/pf/pfstaff/projects/Daudt_DeepSnow/dem_ch/swissalti3d_2017.tif /scratch/swissalti3d_2017.tif

gdalwarp -wo NUM_THREADS=ALL_CPUS -multi -of GTiff -t_srs EPSG:2056 -r bilinear -cutline ../assets/switzerland_shp/swissBOUNDARIES3D_1_3_TLM_LANDESGEBIET.shp -cl swissBOUNDARIES3D_1_3_TLM_LANDESGEBIET -crop_to_cutline /scratch/swissalti3d_2017.tif /scratch/temp.tif

rm /scratch/swissalti3d_2017.tif

# projwin values and outsize values were computed using QGIS to obtain ~10m GSD. These values need to be the same everywhere.
gdal_translate -projwin 2485410.215 1295933.6975 2833857.7237 1075268.1363 -of GTiff -co TILED=YES -r bilinear -outsize 34858 22075 /scratch/temp.tif ../data/DEM/swissalti3d_2017_LV95.tif

rm /scratch/temp.tif
