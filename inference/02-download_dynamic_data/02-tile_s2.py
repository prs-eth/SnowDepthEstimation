from datetime import date, timedelta
from glob import glob
from tqdm import tqdm
import os
import subprocess
import zipfile
import argparse


# Note: This code processes everything sequentially (dates and bands), but it can be parallelized


SOURCE_PATH = './downloaded'
OUTPUT_PATH = './merged'
TEMP_FOLDER = '/scratch/S2_scratch'

PROTOTYPE = False



parser = argparse.ArgumentParser(description='Merge ')
parser.add_argument('--start', type=str, default='2023-01-17', help='Start date (format: YYYY-MM-DD)')
parser.add_argument('--end', type=str, default='2023-01-22', help='End date (format: YYYY-MM-DD)')
args = parser.parse_args()


def get_date_list(start=None, end=None):
    if start == None:
        # start = date(2022,10,15)
        start = date.fromisoformat(args.start)
    if end == None:
        # end = date(2022,11,22)
        end = date.fromisoformat(args.end)
    length = (end - start).days + 1
    date_list = [start + timedelta(days=x) for x in range(length)]
    
    return date_list





if not os.path.exists(TEMP_FOLDER):
    os.mkdir(TEMP_FOLDER)
    
    

for day in get_date_list():
    
    print('\nProcessing date {}.\n\n'.format(str(day)))
    
    # Create output folder if it doesn't exist
    date_folder = os.path.join(OUTPUT_PATH, str(day))    
    if not os.path.exists(date_folder):
        os.mkdir(date_folder)
        
    # Get file list
    search_string = os.path.join(SOURCE_PATH, 'S2*_MSIL2A_{}T*.zip'.format(day.strftime('%Y%m%d')))
    file_list = sorted(glob(search_string))
    if len(file_list) == 0:
        print('No images from date {}. Moving on...'.format(str(day)))
        continue
        
    # Copy files to scratch
    print('Copying files to scratch...')
    file_list_scratch = []
    for f in tqdm(file_list):
        new_path = f.replace(SOURCE_PATH, TEMP_FOLDER)
        cmd = ['cp', f, new_path]
        subprocess.call(cmd)
        file_list_scratch.append(new_path)
        
    file_list = file_list_scratch
    
    # List of layers to be processed
    layers = []
    layers.extend(['_TCI_10m.', '_B02_10m.', '_B03_10m.', '_B04_10m.', '_B08_10m.']) # 10m
    layers.extend(['_B05_20m.', '_B06_20m.', '_B07_20m.', '_B11_20m.', '_B12_20m.', '_B8A_20m.', 'MSK_CLDPRB_20m.', 'MSK_SNWPRB_20m.']) # 20m
#     layers.extend(['MSK_CLDPRB_20m.', 'MSK_SNWPRB_20m.']) # 20m
    layers.extend(['_B01_60m.', '_B09_60m.']) # 60m, B10 is ommited at level 2A
    

    for layer in tqdm(layers):
        print('\n\n\nProcessing {}_{}...'.format(str(day), layer))
        
        # Warp to LV95 projection and cut
        cmd = ['gdalwarp']
        cmd.append('-of')
        cmd.append('GTiff')
#         cmd.append('-s_srs')
#         cmd.append('EPSG:32632')
        cmd.append('-t_srs')
        cmd.append('EPSG:2056')
        cmd.append('-r')
        cmd.append('lanczos')
        cmd.append('-cutline')
        cmd.append('/home/pf/pfstaff/projects/Daudt_DeepSnow/switzerland_shp/swissBOUNDARIES3D_1_3_TLM_LANDESGEBIET.shp')
        cmd.append('-cl')
        cmd.append('swissBOUNDARIES3D_1_3_TLM_LANDESGEBIET')
        cmd.append('-crop_to_cutline')
#         cmd.append('-dstalpha')

        # Source files
        for zip_path in file_list:
            zip_file = zipfile.ZipFile(zip_path)
            namelist = zip_file.namelist()
            for n in namelist:
                if layer in n:
                    zip_inception_path = '/vsizip/' + zip_path + '/' + n
                    cmd.append(zip_inception_path)
        
        
        # Output path for merged file
#         save_to_merged = os.path.join(date_folder, 'S2{}_merged.tif'.format(layer[:-1]))
        save_to_merged = os.path.join(TEMP_FOLDER, 'S2{}_merged.tif'.format(layer[:-1]))
        cmd.append(save_to_merged)
        
        # Run command
#         if PROTOTYPE:
#             print(cmd)
#         else:
#             subprocess.call(cmd)
        subprocess.call(cmd)
            
            
        # Resample image with standard output size (10m) (calculated elsewhere)
        cmd = ['gdal_translate']
        cmd.append('-projwin')
        cmd.append('2485410.215')
        cmd.append('1295933.6975')
        cmd.append('2833857.7237')
        cmd.append('1075268.1363')
        cmd.append('-of')
        cmd.append('GTiff')
        cmd.append('-co')
        cmd.append('TILED=YES')
        cmd.append('-co')
        cmd.append('COMPRESS=LZW')
        cmd.append('-r')
        cmd.append('lanczos')
        cmd.append('-outsize')
        cmd.append('34858')
        cmd.append('22075')
        cmd.append(save_to_merged)
        
        save_to = save_to_merged.replace('_merged', '').replace(TEMP_FOLDER, date_folder).replace('S2MSK', 'S2_MSK')
        cmd.append(save_to)
        
        # Run command
#         if PROTOTYPE:
#             print(cmd)
#         else:
#             subprocess.call(cmd)
        subprocess.call(cmd)
    
    
        
        if PROTOTYPE:
            break
            
        
        # Delete intermediary file
        cmd = ['rm']
        cmd.append(save_to_merged)
        
#         # Run command
#         if PROTOTYPE:
#             print(cmd)
#         else:
#             subprocess.call(cmd)
        subprocess.call(cmd)
            
    
        
    # Clear temporary files
    print('Cleaning up temporary files...')
    # Note: subprocess is weird with wildcards, better do it in a loop
    for f in file_list:
        cmd = ['rm', f]
        subprocess.call(cmd)
        




        
print('\n\n\nAll processes completed.\n\n')
