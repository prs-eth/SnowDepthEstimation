import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
import torch
from tqdm import tqdm




def main():
    """Main training loop. Parameters are given as command line arguments.

    Main arguments:

        ``--dataroot``: Path to dataset

        ``--crop_size``: Width (in pixels) of patches that are used for training.

        ``--rnn_type``: Name of neural network architecture that is to be used.

        ``--lr``: Learning rate.

        ``--n_days``: Number of days in the time series.

        ``--n_epochs``: Number of epochs that the network is to be trained.

        ``--name``: Name of the current experiment.

        ``--model``: Name of the base model.

    Further arguments can be found in ``options/base_options.py`` and ``options/train_options.py``.

    Note that additional options may be available depending on the chosen model and dataset classes.
    """

    # torch.manual_seed(0)
    opt = TrainOptions().parse() # get training options
    torch.manual_seed(opt.seed)
    dataset = create_dataset(opt) # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset) # get the number of images in the dataset.
    print('The number of training samples per epoch = %d' % dataset_size)
    dataset_val = create_dataset(opt, validation=True)  # create a dataset given opt.dataset_mode and other options
    dataset_size_val = len(dataset_val)    # get the number of images in the dataset.
    print('The number of validation samples per epoch = %d' % dataset_size_val)

    model = create_model(opt) # create a model given opt.model and other options
    model.setup(opt) # regular setup: load and print networks; create schedulers
    model.add_graph(next(iter(dataset)))
    total_iters = 0 # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1): # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time() # timer for entire epoch
        iter_data_time = time.time() # timer for data loading per iteration
        epoch_iter = 0 # the number of training iterations in current epoch, reset to 0 every epoch

        print(f'Epoch {epoch}: training...')
        model.train()
        for i, data in enumerate(tqdm(dataset)):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing

            if epoch <= 50:
                N = 10
            elif epoch <= 100:
                N = 3
            else:
                N = 1

            for _ in range(N):
                model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        model.compute_visuals()

        print(f'Epoch {epoch}: validation...')
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(tqdm(dataset_val)):

                total_iters += opt.batch_size
                epoch_iter += opt.batch_size
                model.set_input(data) # unpack data from dataset and apply preprocessing
                model.optimize_parameters(validation=True) # calculate loss functions, get gradients, update network weights

            model.compute_visuals(validation=True)

        model.send_to_tensorboard(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))

        model.update_learning_rate() # update learning rates at the end of every epoch.

        print('\n')




if __name__ == '__main__':
    main()
