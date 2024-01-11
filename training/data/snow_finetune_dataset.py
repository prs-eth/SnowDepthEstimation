"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
from data.base_dataset import BaseDataset 
import torch
from datetime import date , timedelta
import os
import numpy as np
import time
import rasterio



sentinel_2_files = [
    'S2_B01_60m.tif',
    'S2_B02_10m.tif',
    'S2_B03_10m.tif',
    'S2_B04_10m.tif',
    'S2_B05_20m.tif',
    'S2_B06_20m.tif',
    'S2_B07_20m.tif',
    'S2_B08_10m.tif',
    'S2_B09_60m.tif',
    'S2_B11_20m.tif',
    'S2_B12_20m.tif',
    'S2_B8A_20m.tif',
]

dem_files = [
    'swissalti3d_2017_LV95.tif',
    'swissalti3d_2017_LV95_slope.tif',
    'swissalti3d_2017_LV95_aspect_sin.tif',
    'swissalti3d_2017_LV95_aspect_cos.tif',
    'swissalti3d_2017_LV95_TPI.tif',
    'swissalti3d_2017_LV95_TRI.tif',
]

static_file_structure = {
    'DEM': dem_files
}

mask_file_structure = {
    'mask': ['mask_LV95.tif']
}

dynamic_input_structure = {
    'sentinel_1': ['ascending.tif', 'descending.tif'],
    'sentinel_2': sentinel_2_files
}



dynamic_output_structure = {
    'snow_depth': ['snow_depth.tif']
}


train_dates = {
    '2019': [
    '2020-02-07',
    '2020-02-17',
    '2020-04-06',
    '2020-04-22',
    '2020-12-11',
    '2020-12-18',
    '2021-02-24',
    '2021-02-25',
    '2021-03-26',
    '2021-04-16',
],
    '2020': [
    '2018-12-12',
    '2019-02-18',
    '2019-03-12',
    '2019-03-16',
    '2020-12-11',
    '2020-12-18',
    '2021-02-24',
    '2021-02-25',
    '2021-03-26',
    '2021-04-16',
],
    '2021': [
    '2018-12-12',
    '2019-02-18',
    '2019-03-12',
    '2019-03-16',
    '2020-02-07',
    '2020-02-17',
    '2020-04-06',
    '2020-04-22',
],
}
val_dates = {
    '2019': [
    '2018-12-12',
    '2019-02-18',
    '2019-03-12',
    '2019-03-16',
],
    '2020': [
    '2020-02-07',
    '2020-02-17',
    '2020-04-06',
    '2020-04-22',
],
    '2021': [
    '2020-12-11',
    '2020-12-18',
    '2021-02-24',
    '2021-02-25',
    '2021-03-26',
    '2021-04-16',
],
}

# All dates
time_series_start = {
    '2018-12-12': '2018-11-07',
    '2019-02-18': '2018-11-05',
    '2019-03-12': '2018-11-06',
    '2019-03-16': '2018-11-03',
    '2020-02-07': '2019-11-01',
    '2020-02-17': '2019-11-04',
    '2020-04-06': '2019-11-04',
    '2020-04-22': '2019-11-06',
    '2020-12-11': '2020-11-06',
    '2020-12-18': '2020-11-06',
    '2021-02-24': '2020-11-04',
    '2021-02-25': '2020-11-05',
    '2021-03-26': '2020-11-06',
    '2021-04-16': '2020-11-06',
}






MEAN_AND_STD = {
    'ascending_ch1.tif': [-4.935377397788123, 8.196131354631488],
    'ascending_ch2.tif': [-3.1383077790080556, 5.7571205106891385],
    'descending_ch1.tif': [-4.924559624020597, 8.177076673477872],
    'descending_ch2.tif': [-3.093390274467086, 5.698294003521802],
    'S2_B01_60m.tif': [1678.0207671939827, 3352.354053499641],
    'S2_B02_10m.tif': [1654.9863773107784, 3336.978820889729],
    'S2_B03_10m.tif': [1623.4478750096187, 3256.0881664722733],
    'S2_B04_10m.tif': [1594.9361073193509, 3211.796362007882],
    'S2_B05_20m.tif': [1646.0285326000965, 3245.225442975153],
    'S2_B06_20m.tif': [1667.5762044094665, 3182.1681041206352],
    'S2_B07_20m.tif': [1650.8895401220138, 3113.618743692261],
    'S2_B08_10m.tif': [1706.138438655782, 3211.198150658011],
    'S2_B09_60m.tif': [1930.8753387658357, 3678.9240955351133],
    'S2_B11_20m.tif': [710.955555457068, 1462.9848790242393],
    'S2_B12_20m.tif': [628.9722860854072, 1306.7587629900165],
    'S2_B8A_20m.tif': [1633.6656907177257, 3053.9363273780104],
    'S2_MSK_CLDPRB_20m.tif': [9.285597481985699, 28.293297048627224],
    'S2_MSK_SNWPRB_20m.tif': [7.969524793498736, 24.589720723155875],
    'swissalti3d_2017_LV95.tif': [1309.5357471185685, 759.1525437538234],
    'swissalti3d_2017_LV95_slope.tif': [20.897933494750877, 15.585990538333803],
    'swissalti3d_2017_LV95_aspect_sin.tif': [0.0, 1.0], # sin could be normalized but given its properties I choose not to for now
    'swissalti3d_2017_LV95_aspect_cos.tif': [0.0, 1.0], # cos could be normalized but given its properties I choose not to for now
    'swissalti3d_2017_LV95_TPI.tif': [-0.00010149758509592295, 1.1530011058255116],
    'swissalti3d_2017_LV95_TRI.tif': [3.3874958529668624, 3.222045574791955],
    'scd_blur_lzw.tif': [2844.025022525569, 2636.7904107654895],
    'Tmax': [6.3546671867370605, 6.807162284851074], # calculated using 2020 data (including Jan-Apr)
    'Tmin': [-1.6042462587356567, 5.4717254638671875], # calculated using 2020 data (including Jan-Apr)
    'Tabs': [2.2031376361846924, 5.920366287231445], # calculated using 2020 data (including Jan-Apr)
    'Rhires': [2.269566059112549, 5.583887577056885], # calculated using 2020 data (including Jan-Apr)
}



def get_transform(img_path):
    return rasterio.open(img_path).transform


def get_random_window(img_size = [34858, 22075], offset=[0, 0], patch_size = 256):
    """Get random window within certain boundaries.

    No split: img_size = [34858, 22075], offset=[0, 0]
    Train: img_size = [27886, 22075], offset=[0, 0] # Western 80%, doesn't include test zone (Davos)
    Validation: img_size = [34858, 22075], offset=[27886, 0] # Eastern 20%, includes test zone (Davos)

    Args:
        img_size (list, optional): Max x and y coordinates. Defaults to [34858, 22075].
        offset (list, optional): Minimum values for x and y coordinates. Defaults to [0, 0].
        patch_size (int, optional): Defaults to 256.

    Returns:
        [type]: [description]
    """
    x1 = torch.randint(offset[0], img_size[0] - patch_size, (1,)) # left boundary
    x2 = x1 + patch_size # right boundary + 1
    y1 = torch.randint(offset[1], img_size[1] - patch_size, (1,)) # top boundary
    y2 = y1 + patch_size # bottom boundary + 1
    
    return y1, y2, x1, x2



def window_check(h1, h2, w1, w2, mask):
    mask_max = mask[0, h1:h2, w1:w2].max()
    if mask_max == 0:
        return False
    
    return True



def get_random_date_and_location(valid_locations):
    date_idx = torch.randint(0, len(valid_locations), (1,))
    date = list(valid_locations.keys())[date_idx]
    x, y = valid_locations[date]
    loc_idx = torch.randint(0, len(x), (1,))
    x, y = x[loc_idx], y[loc_idx]

    return date, x, y


static_inputs_names = []
for f in dem_files:
    static_inputs_names.append('DEM' + '/' + f)
static_inputs_names = [np.string_(x) for x in static_inputs_names]

dynamic_inputs_names = []
for k in dynamic_input_structure.keys():
    for f in dynamic_input_structure[k]:
        if 'sentinel_1' in k:
            dynamic_inputs_names.append(k + '/' + f + ' (ch01)')
            dynamic_inputs_names.append(k + '/' + f + ' (ch02)')
        else:
            dynamic_inputs_names.append(k + '/' + f)
dynamic_inputs_names = [np.string_(x) for x in dynamic_inputs_names]

def get_dynamic_inputs(date_list, window, base_path):

    i = window[0][0]
    ii = window[0][1]
    j = window[1][0]
    jj = window[1][1]
    chunk = np.zeros((len(date_list), len(dynamic_inputs_names), ii - i, jj - j), dtype=np.float32)

    if len(dynamic_inputs_names) == 0:
        return

    for day_index, day in enumerate(date_list):

        ch_pointer = 0
        # Load dynamic inputs
        for k in dynamic_input_structure.keys():
            for f in dynamic_input_structure[k]:
                path = os.path.join(base_path, k, str(day), f)
                if os.path.exists(path):

                    #read image
                    img = rasterio.open(path).read(window=window)
                    
                    # normalize
                    if 'sentinel_1' in k:
                        mean, std = MEAN_AND_STD[f.replace('.tif', '_ch1.tif')]
                        img[0,:,:] = (img[0,:,:] - mean) / std
                        chunk[day_index, ch_pointer, :, :] = img[0,:,:]
                        # ch_pointer += 1

                        mean, std = MEAN_AND_STD[f.replace('.tif', '_ch2.tif')]
                        img[1,:,:] = (img[1,:,:] - mean) / std
                        chunk[day_index, ch_pointer+1, :, :] = img[1,:,:]
                        # ch_pointer += 1
                    else:
                        mean, std = MEAN_AND_STD[f]
                        img = (img - mean) / std
                        chunk[day_index, ch_pointer, :, :] = img[0,:,:]
                        # ch_pointer += 1

                else:
                    # layer is "normalized zeroes" if data is missing
                    if 'sentinel_1' in k:
                        mean, std = MEAN_AND_STD[f.replace('.tif', '_ch1.tif')]
                        img = (chunk[day_index, ch_pointer, :, :] - mean) / std
                        chunk[day_index, ch_pointer, :, :] =  img
                        # ch_pointer += 1

                        mean, std = MEAN_AND_STD[f.replace('.tif', '_ch2.tif')]
                        img = (chunk[day_index, ch_pointer, :, :] - mean) / std
                        chunk[day_index, ch_pointer+1, :, :] =  img
                        # ch_pointer += 1

                    else:
                        mean, std = MEAN_AND_STD[f]
                        img = (chunk[day_index, ch_pointer, :, :] - mean) / std
                        chunk[day_index, ch_pointer, :, :] =  img
                        # ch_pointer += 1

                ch_pointer += 1
                if 'sentinel_1' in k:
                    ch_pointer += 1

    return chunk


def get_date_list(start_date_str, length, step = 7):
    start_date = date.fromisoformat(start_date_str)
    date_list = [start_date + timedelta(days=step*x) for x in range(length)]
    return date_list


def get_date_list_str(start_date_str, length, step = 7):
    start_date = date.fromisoformat(start_date_str)
    date_list = [np.string_(str(start_date + timedelta(days=step*x))) for x in range(length)]
    return date_list


def get_static_inputs(window, base_path):

    i = window[0][0]
    ii = window[0][1]
    j = window[1][0]
    jj = window[1][1]
    
    chunk = np.zeros((len(static_inputs_names), ii - i, jj - j), dtype=np.float32)

    for ch_number, fname in enumerate(dem_files):
        mean, std = MEAN_AND_STD[fname]
        fpath = os.path.join(base_path, 'DEM', fname)

        img = rasterio.open(fpath).read(window=window)
        img[img < -9998] = 0
        chunk[ch_number:ch_number+1, :, :] = (img - mean) / std

    return chunk



def get_snow_depth(date_list, window, base_path):

    i = window[0][0]
    ii = window[0][1]
    j = window[1][0]
    jj = window[1][1]
    
    chunk = np.zeros((len(date_list), ii - i, jj - j), dtype=np.float32)

    for day_index, day in enumerate(date_list):

        path = os.path.join(base_path, 'snow_depth', str(day), 'snow_depth.tif')
        if os.path.exists(path):
            img = rasterio.open(path).read(window=window)
            chunk[day_index, :, :] =  img[0,:,:]

    chunk[chunk < 0] = 0

    return chunk



class SnowFineTuneDataset(BaseDataset):
    """Dataset class for loading data for training and validation finetuning files.
    
    Reads data relative to dates with a given regular interval starting at a fixed initial date.
    """

    @staticmethod
    def modify_commandline_options(parser):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:

            ``parser``: original option parser

        Returns:
            
            The modified ``parser``.
        """
        parser.add_argument('--num_train_samples', type=int, default=8, help='Number of samples to be used for validation per epoch.')
        parser.add_argument('--num_val_samples', type=int, default=2, help='Number of samples to be used for validation per epoch.')
        parser.add_argument('--no_dem', action='store_true', help='if specified, DEM data is not loaded')
        
        parser.add_argument('--val_year', type=str, default='2021', help='One of: 2019 | 2020 | 2021')

        return parser

    def __init__(self, opt, is_train=False, benchmark=False):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        BaseDataset.__init__(self, opt)
        
        self.opt = opt
        self.d_ch = opt.in_ch_dynamic # number of dynamic channels
        self.s_ch = opt.in_ch_static # number of static channels
        self.is_train = is_train # set to False for validation dataset
        self.base_path = opt.dataroot
        self.crop_size = opt.crop_size
        self.n_days = opt.n_days
        self.benchmark = benchmark
        self.no_dem = opt.no_dem

        if self.is_train:
            self.valid_dates = train_dates[opt.val_year]
        else:
            self.valid_dates = val_dates[opt.val_year]
        
        self.valid_locations = {}
        self.img_size = None
        for d in self.valid_dates:
            with rasterio.open(os.path.join(self.base_path, 'snow_depth', d, 'mask.tif')) as f:
                mask = f.read()
                if not self.img_size:
                    self.img_size = mask.shape[1:3]
                _, y, x = np.where(mask > 0)
                self.valid_locations[d] = (x, y)


        self.rio_transform = get_transform(os.path.join(self.base_path, 'mask', 'mask_LV95.tif'))


    def __getitem__(self, _):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """

        if self.benchmark:
            t0 = time.time()

        date, x, y = get_random_date_and_location(self.valid_locations)
        x0 = x - self.crop_size // 2
        x1 = x0 + self.crop_size
        y0 = y - self.crop_size // 2
        y1 = y0 + self.crop_size
        window = ((y0,y1),(x0,x1))

        date_list = get_date_list(time_series_start[date], self.n_days)

        if self.d_ch == 0:
            i = window[0][0]
            ii = window[0][1]
            j = window[1][0]
            jj = window[1][1]
            dynamic_inputs = torch.zeros((len(date_list), len(dynamic_inputs_names), ii - i, jj - j)).float()
        else:
            dynamic_inputs = torch.from_numpy(get_dynamic_inputs(date_list, window, self.base_path))


        if self.no_dem:
            static_inputs = torch.zeros((0, x1-x0, y1-y0))
        else:
            static_inputs = torch.from_numpy(get_static_inputs(window, self.base_path))

        snow_depth = 100 * torch.from_numpy(get_snow_depth(date_list, window, self.base_path))
        snow_depth = torch.clip(snow_depth, 0, 700)

        snow_depth_valid_flags = torch.from_numpy(np.array([1 if str(d) == date else 0 for d in date_list]))
        assert(snow_depth_valid_flags.max() > 0)

        mask_path = os.path.join(self.base_path, 'snow_depth', date, 'mask.tif')
        with rasterio.open(mask_path) as f:
            mask = torch.from_numpy(f.read(window=window))
        mask[mask < 0] = 0
        mask[mask > 3.0] = 3.0

        if self.benchmark:
            t5 = time.time()


        sample = {
            'dynamic_inputs': dynamic_inputs,
            'static_inputs': static_inputs, 
            'snow_depth': snow_depth, 
            'mask': mask, 
            'snow_depth_valid_flags': snow_depth_valid_flags,
            }

        return sample

    def __len__(self):
        """Return the total number samples per epoch."""
        if self.is_train:
            return self.opt.num_train_samples
        else:
            return self.opt.num_val_samples
