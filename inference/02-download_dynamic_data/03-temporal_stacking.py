import rasterio
import os
from tqdm import tqdm
import numpy as np
from datetime import date , timedelta
import argparse


parser = argparse.ArgumentParser(description='Merge ')
parser.add_argument('--start', type=str, default='2023-01-17', help='Start date (format: YYYY-MM-DD)')
parser.add_argument('--end', type=str, default='2023-01-22', help='End date (format: YYYY-MM-DD)')
parser.add_argument('--init_from_prev', action='store_true', help='Continue from the day before --start')
args = parser.parse_args()


# Note: this code currently uses the sentinel 2 cloud masks, but other masks can be downloaded from GEE

# Note: the code processes the entire time series, one band at a time. This is the best way to do things
# offline, but not online. The code can be adapted to load the most recent stacked image and update it 
# with data from a new acquisition.


S2_PATH = '/scratch2/deepsnow_deliverables/02-download_dynamic_data/merged'
# CLOUDS_PATH = '/home/pf/pfstaff/projects/Daudt_DeepSnow/dataset/v1/clouds'
# DEM_PATH = '/home/pf/pfstaff/projects/Daudt_DeepSnow/dataset/v1/DEM/swissalti3d_2017_LV95.tif'
OUT_PATH = '/scratch2/deepsnow_deliverables/data/sentinel_2'

CLOUD_THRESHOLD = 49

def get_date_list(start_date, end_date):

    length = (end_date - start_date).days + 1
    date_list = [start_date + timedelta(days=x) for x in range(length)]
    return date_list

date_list = get_date_list(
    date.fromisoformat(args.start),
    date.fromisoformat(args.end),
)

file_names = [
    'S2_B03_10m.tif',
    'S2_B02_10m.tif',
    'S2_B04_10m.tif',
    'S2_B01_60m.tif',
    'S2_B05_20m.tif',
    'S2_B06_20m.tif',
    'S2_B07_20m.tif',
    'S2_B08_10m.tif',
    'S2_B8A_20m.tif',
    'S2_B09_60m.tif',
    'S2_B11_20m.tif',
    'S2_B12_20m.tif',
    'S2_MSK_CLDPRB_20m.tif',
    'S2_MSK_SNWPRB_20m.tif',
    # 'S2_TCI_10m.tif',
]

# Generate target folders
for d in date_list:
    path = os.path.join(OUT_PATH, str(d))
    if not os.path.exists(path):
        os.makedirs(path)

for fn in file_names:
    model_file_path = os.path.join(S2_PATH, '{}', fn)
    target_file_path = os.path.join(OUT_PATH, '{}', fn)
    # cloud_path = os.path.join(CLOUDS_PATH, '{}', 'cloud_probability.tif')
    cloud_path = os.path.join(S2_PATH, '{}', 'S2_MSK_CLDPRB_20m.tif')

    for idx, d in enumerate(date_list):
        # print(model_file_path.format(date_list[idx]))
        if os.path.exists(model_file_path.format(date_list[idx])):
            d_idx = idx
    profile = rasterio.open(model_file_path.format(date_list[d_idx])).profile.copy()

    if args.init_from_prev:
        prev_date = date.fromisoformat(args.start) + timedelta(days=-1)
        img = rasterio.open(target_file_path.format(str(prev_date))).read()
    else:
        img = np.zeros_like(rasterio.open(model_file_path.format(date_list[d_idx])).read())


    for d in tqdm(date_list):
        print(f'Processing {fn} for {str(d)}...')

        img_path = model_file_path.format(str(d))
        if os.path.exists(img_path):
            new_img = rasterio.open(model_file_path.format(str(d))).read()
            cloud_probs = rasterio.open(cloud_path.format(str(d))).read()

            mask = (cloud_probs <= CLOUD_THRESHOLD) & (new_img > 0)
            img[mask] = new_img[mask]

        path = target_file_path.format(str(d))
        writer = rasterio.open(path, 'w', **profile)
        writer.write(img)
        writer.close()
        




print('\nEnd\n')