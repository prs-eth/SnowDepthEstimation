

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
    # 'S2_MSK_CLDPRB_20m.tif',
    # 'S2_MSK_SNWPRB_20m.tif',
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
    # 'sentinel_1': ['ascending.tif', 'descending.tif'],
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




