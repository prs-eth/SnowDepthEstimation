from data.base_dataset import BaseDataset 
import torch

date_list = ['2020-11-01', '2020-11-08', '2020-11-15', '2020-11-22',
       '2020-11-29', '2020-12-06', '2020-12-13', '2020-12-20',
       '2020-12-27', '2021-01-03', '2021-01-10', '2021-01-17',
       '2021-01-24', '2021-01-31', '2021-02-07', '2021-02-14',
       '2021-02-21', '2021-02-28', '2021-03-07', '2021-03-14',
       '2021-03-21', '2021-03-28', '2021-04-04', '2021-04-11',
       '2021-04-18', '2021-04-25']


class DummyDataset(BaseDataset):
    """Dataset class for loading data for training and validation from HDF5 files.
    
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
        return parser

    def __init__(self, opt, is_train=False):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        BaseDataset.__init__(self, opt)
        
        self.is_train = is_train
        self.opt = opt
        self.d_ch = opt.in_ch_dynamic # number of dynamic channels
        self.s_ch = opt.in_ch_static # number of static channels
        self.crop_size = opt.crop_size
        self.n_days = opt.n_days

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

        end_date = self.n_days - 1
        days_str = list(date_list[end_date-self.n_days+1:end_date+1])
        
        dynamic_inputs = torch.randn((self.n_days, self.d_ch, self.crop_size, self.crop_size))
        static_inputs = torch.randn((self.s_ch, self.crop_size, self.crop_size))
        snow_depth = torch.rand((self.n_days, self.crop_size, self.crop_size))
        mask = torch.ones((1, self.crop_size, self.crop_size))
        snow_depth_valid_flags = torch.ones((self.n_days,))

        sample = {
            'dynamic_inputs': dynamic_inputs,
            'static_inputs': static_inputs, 
            'snow_depth': snow_depth, 
            'mask': mask, 
            'date_list': days_str,
            'snow_depth_valid_flags': snow_depth_valid_flags,
            }

        return sample

    def __len__(self):
        """Return the total number samples per epoch."""
        if self.is_train:
            return self.opt.num_train_samples
        else:
            return self.opt.num_val_samples
