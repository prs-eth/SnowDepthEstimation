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
import h5py
from data.base_dataset import BaseDataset
import torch
import rasterio
from datetime import date
import os
import numpy as np
import time
from tqdm import tqdm



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
    # 'scd_blur_lzw.tif',
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

class YetiV4Dataset(BaseDataset):
    """Dataset class for loading data for training and validation from HDF5 files.
    
    Reads data relative to consecutive dates starting at a random valid date.
    """

    @staticmethod
    def modify_commandline_options(parser):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:

            ``parser``: original option parser

        Returns:
            
            The modified ``parser``.
        """
        parser.add_argument('--num_train_samples', type=int, default=800, help='Number of samples to be used for validation per epoch.')
        parser.add_argument('--num_val_samples', type=int, default=200, help='Number of samples to be used for validation per epoch.')
        parser.add_argument('--no_weather', action='store_true', help='if specified, weather data is not loaded')
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
        self.no_weather = opt.no_weather

        self.hdf5_file_1 = h5py.File(os.path.join(self.base_path, 'Yeti_dataset-01-misc.hdf5'), 'r')
        if self.is_train:
            self.hdf5_file_2 = h5py.File(os.path.join(self.base_path, 'Yeti_dataset-02-mask-train.hdf5'), 'r')
            self.hdf5_file_3 = h5py.File(os.path.join(self.base_path, 'Yeti_dataset-03-static_inputs-train.hdf5'), 'r')
            self.hdf5_file_4 = h5py.File(os.path.join(self.base_path, 'Yeti_dataset-04-dynamic_inputs-train.hdf5'), 'r')
            self.hdf5_file_5 = h5py.File(os.path.join(self.base_path, 'Yeti_dataset-05-snow_depth-train.hdf5'), 'r')
            self.hdf5_file_6 = h5py.File(os.path.join(self.base_path, 'Yeti_dataset-06-weather-train.hdf5'), 'r')
            self.hdf5_file_7 = h5py.File(os.path.join(self.base_path, 'Yeti_dataset-07-valid_squares-train.hdf5'), 'r')
        else:
            self.hdf5_file_2 = h5py.File(os.path.join(self.base_path, 'Yeti_dataset-02-mask-val.hdf5'), 'r')
            self.hdf5_file_3 = h5py.File(os.path.join(self.base_path, 'Yeti_dataset-03-static_inputs-val.hdf5'), 'r')
            self.hdf5_file_4 = h5py.File(os.path.join(self.base_path, 'Yeti_dataset-04-dynamic_inputs-val.hdf5'), 'r')
            self.hdf5_file_5 = h5py.File(os.path.join(self.base_path, 'Yeti_dataset-05-snow_depth-val.hdf5'), 'r')
            self.hdf5_file_6 = h5py.File(os.path.join(self.base_path, 'Yeti_dataset-06-weather-val.hdf5'), 'r')
            self.hdf5_file_7 = h5py.File(os.path.join(self.base_path, 'Yeti_dataset-07-valid_squares-val.hdf5'), 'r')


        self.date_list = self.hdf5_file_1['date_list'].asstr()
        self.static_inputs_names = self.hdf5_file_1['static_inputs_names'].asstr()
        self.dynamic_inputs_names = self.hdf5_file_1['dynamic_inputs_names'].asstr()
        self.valid_dates = self.hdf5_file_1['valid_dates']

        self.mask = self.hdf5_file_2['mask'][:]

        self.static_inputs = self.hdf5_file_3['static_inputs']

        self.dynamic_inputs = self.hdf5_file_4['dynamic_inputs']

        self.snow_depth = self.hdf5_file_5['snow_depth']

        self.weather = self.hdf5_file_6['weather']

        self.num_squares = self.mask.shape[0]
        self.img_size = self.mask.shape[-2:]
        self.offset = [0, 0]

        self.rio_transform = get_transform(os.path.join(self.base_path, 'mask', 'mask_LV95.tif'))
        
        self.valid_end_dates = np.nonzero(self.valid_dates)[0]

        self.squares_with_snow_per_valid_end_date = self.hdf5_file_7 
        assert(self.valid_end_dates.sum() > 0)
        assert(len(self.valid_dates) >= self.n_days)
        

    def __getitem__(self, index):
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
        
        c = 1

        end_date_idx = torch.randint(len(self.valid_end_dates), (1,))
        end_date = self.valid_end_dates[end_date_idx]

        # square_idx = torch.randint(0, self.num_squares, (1,))
        squares_with_snow = self.squares_with_snow_per_valid_end_date[str(int(end_date_idx))]
        square_idx = squares_with_snow[torch.randint(len(squares_with_snow), (1,))]

        days_str = list(self.date_list[end_date-self.n_days+1:end_date+1])

        h1, h2, w1, w2 = get_random_window(img_size=self.img_size, offset=self.offset, patch_size=self.crop_size)
        found = False


        while found == False:
            if window_check(h1, h2, w1, w2, self.mask[square_idx]):
                found = True
            else:
                h1, h2, w1, w2 = get_random_window(img_size=self.img_size, offset=self.offset, patch_size=self.crop_size)
                c += 1
                if c > 50:
                    print('Warning: data loader struggling to find an appropriate window')

        if self.benchmark:
            t1 = time.time()
        
        mask = torch.from_numpy(self.mask[square_idx, :, h1:h2, w1:w2])


        if self.benchmark:
            t2 = time.time()


        dynamic_inputs = torch.from_numpy(self.dynamic_inputs[square_idx, end_date-self.n_days+1:end_date+1, :, h1:h2, w1:w2])
        


        if not self.no_weather:
            weather_tensor = torch.from_numpy(self.weather[square_idx, end_date-self.n_days+1:end_date+1, :, h1:h2, w1:w2])

            dynamic_inputs = torch.cat((dynamic_inputs, weather_tensor), 1)


        if self.benchmark:
            t3 = time.time()


        static_inputs = torch.from_numpy(self.static_inputs[square_idx, :, h1:h2, w1:w2])


        if self.benchmark:
            t4 = time.time()

        
        snow_depth_valid_flags = torch.from_numpy(self.valid_dates[end_date-self.n_days+1:end_date+1])

        snow_depth = torch.from_numpy(self.snow_depth[square_idx, end_date-self.n_days+1:end_date+1, h1:h2, w1:w2])


        if self.benchmark:
            t5 = time.time()                
        
        

        sample = {
            'dynamic_inputs': dynamic_inputs,
            'static_inputs': static_inputs, 
            'snow_depth': snow_depth, 
            'mask': mask, 
            'date_list': days_str,
            'snow_depth_valid_flags': snow_depth_valid_flags,
            }

        
        if self.benchmark:
            t6 = time.time()

            print('\nc = {} attempts'.format(c))
            print('t1 - t0 = {} s (get valid window)'.format(t1 - t0))
            print('t2 - t1 = {} s (load mask)'.format(t2 - t1))
            print('t3 - t2 = {} s (dynamic inputs)'.format(t3 - t2))
            print('t4 - t3 = {} s (static inputs)'.format(t4 - t3))
            print('t5 - t4 = {} s (snow depth)'.format(t5 - t4))
            print('t6 - t5 = {} s (assemble dictionary)\n'.format(t6 - t5))




        return sample

    def __len__(self):
        """Return the total number samples per epoch."""
        if self.is_train:
            return self.opt.num_train_samples
        else:
            return self.opt.num_val_samples
