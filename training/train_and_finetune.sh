# Commands to train and finetune a network with a dummy dataloader

EXPNAME="ExperimentName"
YEAR="2021"

python train.py \
    --dataset_mode dummy \
    --dataroot /path/to/dataset \
    --in_ch_static 6 \
    --in_ch_dynamic 16 \
    --name ${EXPNAME} \
    --n_epochs 5 \
    --n_epochs_decay 5

mkdir "checkpoints/${EXPNAME}_finetune_${YEAR}"

cp checkpoints/${EXPNAME}/latest* checkpoints/${EXPNAME}_finetune_${YEAR}/

python finetune.py \
    --dataset_mode dummy \
    --dataroot /path/to/dataset \
    --in_ch_static 6 \
    --in_ch_dynamic 16 \
    --name ${EXPNAME}_finetune_${YEAR}\
    --n_epochs 5 \
    --n_epochs_decay 5 \
    --continue_train \
    --epoch latest


