# Snow-Depth-Estimation

This repository contains code for 

[Daudt, R.C., Wulf, H., Hafner, E.D., Bühler, Y., Schindler, K. and Wegner, J.D., 2023. Snow depth estimation at country-scale with high spatial and temporal resolution. ISPRS Journal of Photogrammetry and Remote Sensing, 197, pp.105-121.](https://www.sciencedirect.com/science/article/pii/S0924271623000230)


## Disclaimer

For various reasons, data to reproduce the experiments presented in the paper can not be shared. Nevertheless, the code in this repository aims to help others understand the details of how exactly the neural networks were trained and applied. A dummy dataloader is provided such that the code should be able to be run for "training" the network.

## Data

Sentinel-1 and Sentinel-2 data can be downloaded directly using the sentinelsat api.

High quality snow depth maps are available here: [https://www.envidat.ch/dataset/snow-depth-mapping-by-airplane-photogrammetry-2017-ongoing](https://www.envidat.ch/dataset/snow-depth-mapping-by-airplane-photogrammetry-2017-ongoing)


## Sources

Code in this repository contains adapted code from [CycleGAN](https://github.com/junyanz/CycleGAN), [MEDUSA](https://github.com/aboulch/medusa_tb), and [ConvSTAR](https://github.com/0zgur0/multi-stage-convSTAR-network).
