
import os
from re import A
import torch
from options.test_options import TestOptions
from models import create_model
# from util.visualizer import save_images
# from util.argfarce import argfarce
from tqdm import tqdm
import data.test_loader as tl
import rasterio
from util.windowing import hanning_2d
import time

arg_list = []


USE_WEATHER = False

USE_CUDA = True

# # Coordinates for Ultracam tests
# y0 = 800 #900
# y1 = 3800 #3370
# x0 = 5000 #5860
# x1 = 8000 #7620

# # 32TNS
# y0 = 0
# y1 = 11197
# x0 = 0
# x1 = 11197

# 
y0 = 0
y1 = 22075
x0 = 0
x1 = 34858


# val_size = [11197, 11197]
val_size = [34858, 22075]
step_size = 7
# scale = 50.0
L = 1536
L_overlap = 32
# y1 = y0 + L
# x1 = x0 + L



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

def main():
    """Main validation function. Parameters are given as command line arguments.

    Main arguments:

        ``--dataroot``: Path to dataset

        ``--epoch``: Epoch index of checkpoint that is to be loaded.

        ``--rnn_type``: Name of neural network architecture that is to be used.

        ``--name``: Name of the experiment.

        ``--model``: Name of the base model.

    Further arguments can be found in ``options/base_options.py`` and ``options/test_options.py``.

    Note that additional options may be available depending on the chosen model and dataset classes.
    """

    t0 = time.time()

    opt = TestOptions().parse()

    scale = opt.output_scale

    if opt.variable == 'mean':
        variables = [
            'mean',
        ]
    elif opt.variable == 'variance':
        variables = [
            'variance',
        ]
    elif opt.variable == 'both':
        variables = [
            'mean',
            'variance',
        ]
    else:
        raise Exception

    # opt.dataroot = f'/home/pf/pfstaff/projects/Daudt_DeepSnow/dataset/validation/{version}'

    # Create model

    model = create_model(opt)
    model.setup(opt)
    net = model.netModel
    net.eval()

    # out_path_prototype = os.path.join(opt.dataroot, 'results', opt.name, opt.test_date, '{}')
    # n = opt.name.replace('_2019', '').replace('_2020', '').replace('_2021', '')
    out_path_prototype = os.path.join(opt.dataroot.replace('data', 'outputs'), opt.test_date, '{}')

    for variable in variables:
        out_path = out_path_prototype.format(variable)
        if not os.path.exists(out_path):
            os.makedirs(out_path)

    outputs = {}
    path_template = os.path.join(opt.dataroot, 'mask', 'mask_LV95.tif').replace('/v3/', '/v1/')
    with rasterio.open(path_template) as gt: 
        for variable in variables:
            print(f'Initializing {variable}...')
            outputs[variable] = 0 * gt.read().astype('float64')
        outputs['weights'] = 0 * gt.read().astype('float64') + 1e-6

    path_mask = os.path.join(opt.dataroot, 'mask', 'mask_LV95.tif')
    with rasterio.open(path_mask) as file:
        mask = file.read().astype('float32')

    window_weights = hanning_2d(L)


    # Date list
    if step_size == 7:
        dates = tl.get_date_list(start_date=opt.start_timeseries_date, end_date=opt.test_date, step=step_size)
    else:
        raise Exception

    
    x_corner_list = [x for x in range(x0, x1-L, L - L_overlap)]
    x_corner_list.append(x1-L)
    y_corner_list = [y for y in range(y0, y1-L, L - L_overlap)]
    y_corner_list.append(y1-L)

    N_patches = len(x_corner_list) * len(y_corner_list)
    counter = 0
    for x_corner in x_corner_list:
        for y_corner in y_corner_list:
            counter += 1
            print(f'Processing patch {counter}/{N_patches}...')

            if mask[0, y_corner:y_corner+L, x_corner:x_corner+L].sum() <= 0:
                continue

            window = ((y_corner,y_corner+L),(x_corner,x_corner+L))

            # Load static data
            static_inputs = tl.get_static_inputs(opt.dataroot, opt.in_ch_static, [L, L], cuda=USE_CUDA, window=window)

            # Initialize
            hidden = None

            with torch.no_grad():

                for day in tqdm(dates):
                    dynamic_inputs = tl.get_dynamic_inputs(opt.dataroot, [L, L], day, d_ch = opt.in_ch_dynamic, cuda=USE_CUDA, window=window)



                    inputs = torch.cat((dynamic_inputs, static_inputs), 1)
                    hidden = net(inputs, hidden=hidden)

                outputs['weights'][0, y_corner:y_corner+L, x_corner:x_corner+L] += window_weights
                if opt.model == 'yeti_v3' or opt.model == 'yeti_v6':
                    for variable in variables:
                        if variable == 'mean':
                            prediction = net.module.forward_tail(hidden).cpu().numpy()
                            prediction = prediction[0] * window_weights
                            outputs[variable][0, y_corner:y_corner+L, x_corner:x_corner+L] += prediction[0]
                        elif variable == 'variance':
                            prediction = torch.exp(net.module.forward_tail_var(hidden)).cpu().numpy()
                            prediction = prediction[0] * window_weights
                            outputs[variable][0, y_corner:y_corner+L, x_corner:x_corner+L] += prediction[0]
                        else:
                            raise Exception
                else:
                    raise Exception


    with rasterio.open(path_template) as gt:
        for variable in variables:
            outputs[variable] = outputs[variable] / outputs['weights']
            print(f'Saving {variable}...')
            file_path = os.path.join(out_path_prototype.format(variable), 'pred.tif')

            if opt.model == 'yeti_v3' or opt.model == 'yeti_v6':
                if variable == 'mean':
                    outputs[variable] *= scale / 100
                elif variable == 'variance':
                    outputs[variable] *= (scale ** 2) / (100 ** 2)
                else:
                    raise Exception

            else:
                raise Exception

            new_dataset = rasterio.open(
                file_path,
                'w',
                driver='GTiff',
                height=val_size[1],
                width=val_size[0],
                count=1,
                dtype=outputs[variable].dtype,
                crs=gt.crs,
                transform=gt.transform,
            )
            new_dataset.write(outputs[variable][0] * mask[0], 1)
            new_dataset.close()


    

    
    print('Finished.')
    t1 = time.time()
    print('Elapsed time: {} s'.format(t1 - t0))


if __name__ == '__main__':
    main()
