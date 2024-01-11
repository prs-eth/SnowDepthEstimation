# Inference

Code to download and run inference using Sentinel-2 images.

Files should be run from the directories they are located (not from root directory)

Ideally, the following files should be executed to produce outputs:
 - 01-gen_static_data/01-warp_and_cut.sh
 - 01-gen_static_data/02-gen_derivates.sh
 - 01-gen_static_data/03-generate_mask.sh
 - 02-download_dynamic_data/01-download_s2.sh (possibly several times if data has been archived, Sentinel-2 LTA can be tricky)
 - 02-download_dynamic_data/02-tile_s2.py
 - 02-download_dynamic_data/03-temporal_stacking.py
 - 03-generate_results/example.sh

Additional comments can be found inside different files.

