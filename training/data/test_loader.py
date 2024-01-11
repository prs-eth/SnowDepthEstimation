import data.yeti_v4_dataset as ydt
import rasterio
import os
import torch
from datetime import date, timedelta





def get_date_list(start_date=None, end_date=None, step=1):
    """Generate a list of dates between ``start_date`` and ``end_date`` with a stride of ``step``.

    ``start_date`` and ``end_date`` should be strings with the format ``YYYY-MM-DD``
    """
    if start_date == None:
        start = date(2020,11,1)
    else:
        start = date.fromisoformat(start_date)
    if end_date == None:
        end = date(2021,1,23)
    else:
        end = date.fromisoformat(end_date)
        
    length = (end - start).days + 1
    date_list = [start + timedelta(days=x) for x in range(length)]
    
    return date_list[::step]

def get_mask(base_path, cuda=True):
    """Read mask of valid pixel locations from the given ``base_path`` dataset path.

    ``cuda`` controls whether the returned tensor should be in GPU or CPU.
    """
    mask_path = os.path.join(base_path, 'mask', 'mask_LV95.tif')
    if cuda:
        dataset = rasterio.open(mask_path)
        img = torch.from_numpy(dataset.read().astype('uint8')).cuda()
        dataset.close()
        return img
    else:
        dataset = rasterio.open(mask_path)
        img = torch.from_numpy(dataset.read().astype('uint8'))
        dataset.close()
        return img


def get_static_inputs(base_path, s_ch, size, cuda=True, window=None):
    """Read static inputs from dataset in ``base_path``
    
    Parameters:

        ``base_path``: Dataset path

        ``s_ch``: Number of feature channels

        ``size``: Spatial dimensions

        ``cuda``: CPU or GPU flag
    """

    static_inputs = torch.zeros((1, s_ch, size[1], size[0]))
    # Load static inputs
    ch_pointer = 0
    if s_ch != 0:
        for k in ydt.static_file_structure.keys():
            for f in ydt.static_file_structure[k]:
                path = os.path.join(base_path, k, f)
                if os.path.exists(path):
                    #read image
                    with rasterio.open(path) as file:
                        if window:
                            img = torch.from_numpy(file.read(window=window).astype('float')).float()
                        else:
                            img = torch.from_numpy(file.read().astype('float')).float()

                    # Deal with nodata values
                    img[img < -9998] = 0

                    mean, std = ydt.MEAN_AND_STD[f]
                    img = (img - mean) / std
                    
                    static_inputs[0, ch_pointer, :, :] = img[0,:,:]
                    ch_pointer += 1
                else:
                    #append zeroes
                    print('ERROR: STATIC LAYERS SHOULD ALWAYS BE AVAILABLE. MISSING FILE: {}'.format(path))
                    raise Exception

    if cuda:
        return static_inputs.cuda()
    else:
        return static_inputs


def get_dynamic_inputs(base_path, size, day, d_ch = 16, cuda=True, window=None):
    """Read dynamic inputs (excluding weather variables) from dataset in ``base_path``
    
    Parameters:

        ``base_path``: Dataset path

        ``size``: Spatial dimensions

        ``day``: Date to be loaded

        ``d_ch``: Number of feature channels

        ``cuda``: CPU or GPU flag
    """

    dynamic_inputs = torch.zeros((1, d_ch, size[1], size[0]))
    if d_ch == 0:
        if cuda:
            return  dynamic_inputs.cuda()
        else:
            return  dynamic_inputs
    ch_pointer = 0
    for k in ydt.dynamic_input_structure.keys():
        if d_ch == 12 and k == 'sentinel_1':
            continue
        if d_ch == 4 and k == 'sentinel_2':
            continue
        for f in ydt.dynamic_input_structure[k]:
            path = os.path.join(base_path, k, str(day), f)
            if os.path.exists(path):
                #read image
                with rasterio.open(path) as file:
                    if window:
                        img = torch.from_numpy(file.read(window=window).astype('float')).float()
                    else:
                        img = torch.from_numpy(file.read().astype('float')).float()
                
                # normalize
                if 'sentinel_1' in k:
                    mean, std = ydt.MEAN_AND_STD[f.replace('.tif', '_ch1.tif')]
                    img[0,:,:] = (img[0,:,:] - mean) / std
                    dynamic_inputs[0, ch_pointer, :, :] = img[0,:,:]
                    ch_pointer += 1

                    mean, std = ydt.MEAN_AND_STD[f.replace('.tif', '_ch2.tif')]
                    img[1,:,:] = (img[1,:,:] - mean) / std
                    dynamic_inputs[0, ch_pointer, :, :] = img[1,:,:]
                    ch_pointer += 1
                else:
                    mean, std = ydt.MEAN_AND_STD[f]
                    img = (img - mean) / std
                    dynamic_inputs[0, ch_pointer, :, :] = img[0,:,:]
                    ch_pointer += 1

            else:
                # layer is "normalized zeroes" if data is missing
                if 'sentinel_1' in k:
                    mean, std = ydt.MEAN_AND_STD[f.replace('.tif', '_ch1.tif')]
                    dynamic_inputs[0, ch_pointer, :, :] -=  mean / std
                    ch_pointer += 1

                    mean, std = ydt.MEAN_AND_STD[f.replace('.tif', '_ch2.tif')]
                    dynamic_inputs[0, ch_pointer, :, :] -=  mean / std
                    ch_pointer += 1

                else:
                    mean, std = ydt.MEAN_AND_STD[f]
                    dynamic_inputs[0, ch_pointer, :, :] -=  mean / std
                    ch_pointer += 1
    
    if cuda:
        return  dynamic_inputs.cuda()
    else:
        return  dynamic_inputs


weather_vars = ['Tmax', 'Tmin', 'Tabs', 'Rhires']
def get_weather_inputs(base_path, size, day, cuda=True):
    """Read weather inputs from dataset in ``base_path``
    
    Parameters:

        ``base_path``: Dataset path

        ``size``: Spatial dimensions

        ``day``: Date to be loaded

        ``cuda``: CPU or GPU flag
    """
    weather_inputs = torch.zeros((1, len(weather_vars), size[1], size[0]))
    for c, var_name in enumerate(weather_vars):
        path = os.path.join(base_path, 'weather', var_name, f'{str(day)}.tif')
        img = torch.from_numpy(rasterio.open(path).read().astype('float')).float()
        mean, std = ydt.MEAN_AND_STD[var_name]
        weather_inputs[0, c] = (img[0,:,:] - mean) / std
    
    if cuda:
        return weather_inputs.cuda()
    else:
        return weather_inputs