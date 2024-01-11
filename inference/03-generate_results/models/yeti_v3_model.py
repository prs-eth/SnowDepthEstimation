import torch
# import itertools
from .base_model import BaseModel
from . import networks
# from .EMANet import EMANet
# import settings
from torch.utils.tensorboard import SummaryWriter
import os
import datetime
import copy
# from torchsummary import summary
from util.probabilistic_losses import GaussianNLLLoss, LaplacianNLLLoss





class YetiV3Model(BaseModel):
    """
    Wrapper class for snow depth estimation networks.

    V3 uses model averaging and Gaussian Negative Log Likelihood loss.

        Parameters:

            ``opt``: stores all the experiment flags and parameters
    """
    @staticmethod
    def modify_commandline_options(parser):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:

            ``parser``: original option parser

        Returns:
            
            The modified ``parser``.
        """
        
        # TODO: make input_nc adaptive
        parser.set_defaults(input_nc=18, output_nc=1)
        
#         parser.add_argument('--N_classes', type=int, default=2, help='Number of output classes.')
        parser.add_argument('--start_date', type=str, default='2020-11-01', help='Start date of dataset. Format: YYYY-MM-DD')
        parser.add_argument('--end_date', type=str, default='2021-04-30', help='End date of dataset. Format: YYYY-MM-DD')

        parser.add_argument('--n_days', type=int, default=10, help='Number of days considered in each sequence.')

        parser.add_argument('--depth', type=int, default=5, help='Number of ConvRNN blocks to be used.')
        parser.add_argument('--hidden_size', type=int, default=64, help='Number feature channels in each block.')

        parser.add_argument('--in_ch_dynamic', type=int, default=12, help='Number of days considered in each sequence. (S2=12, S2_PRB=2, S1=4)')
        parser.add_argument('--in_ch_static', type=int, default=6, help='Number of days considered in each sequence.')
        parser.add_argument('--rnn_type', type=str, default='ConvGRU', help='One of: ConvGRU | ConvLSTM | ConvSTAR | (more models were added later, check code)')

        parser.add_argument('--ignore_first_N', type=int, default=5, help='Number of days ignored for loss in the beginning of sequence.')

        parser.add_argument('--validate_only_last', action='store_true', help='if specified, only the last output is used for validation')

        parser.add_argument('--mse', action='store_true', help='if specified, replace L1 loss by MSE loss')
        parser.add_argument('--rl1', action='store_true', help='if specified, replace L1 loss by RL1 loss')
        parser.add_argument('--adam', action='store_true', help='if specified, use ADAM optimizer')
        parser.add_argument('--partial', action='store_true', help='if specified, use partial convolutions for TCNs')

        parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay.')
        parser.add_argument('--efficientnet_b', type=int, default=7, help='EfficientNet size: 0 to 7')

        parser.add_argument('--parameter_momentum', type=float, default=0.9975, help='Parameter momentum.')
        parser.add_argument('--gradient_clipping', type=float, default=1.0, help='Parameter momentum.')
        parser.add_argument('--output_scale', type=float, default=100.0, help='Parameter momentum.')




        return parser

    def __init__(self, opt):
        """Initialize the model wrapper class.

        Parameters:

            ``opt``: stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)

        assert(opt.ignore_first_N < opt.n_days)

        if opt.phase != 'test':
            log_dir = os.path.join('logs', opt.model, opt.rnn_type, opt.name + '_' + str(datetime.datetime.now())[:19].replace(' ', '_').replace(':', '-'))
            print(f'log_dir is: "{log_dir}"')
            self.writer = SummaryWriter(log_dir=log_dir)

        self.masks_available = opt.dataset_mode in ['yeti_v2']
        self.partial = opt.partial
        if opt.partial:
            assert(self.masks_available)

        self.rnn_type = opt.rnn_type

        self.momentum = opt.parameter_momentum
        self.gradient_clipping = opt.gradient_clipping
        self.output_scale = opt.output_scale


        self.batch_size = opt.batch_size
        self.n_days = opt.n_days

        self.vol = opt.validate_only_last
        self.ifn = opt.ignore_first_N
        self.in_ch_dynamic = opt.in_ch_dynamic
        
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        # self.loss_names = ['L1']
        # self.loss_L1 = 0.0

        self.L1_acc = 0.0
        self.L1_train = 0.0
        self.L1_val = 0.0
        self.Loss_acc = 0.0
        self.Loss_train = 0.0
        self.Loss_val = 0.0
        self.niter_since_flush = 0
        

        self.visual_names = ['I_in', 'depth_gt', 'depth_pred']  # combine visualizations for A and B
        
        
        self.model_names = ['Model', 'Model_NoMom']

        if opt.efficientnet_b < 0 or opt.efficientnet_b > 7:
            print('EfficientNet only exists from B0 to B7.')
            raise Exception

        self.netModel = networks.get_model(
            opt.rnn_type + '_v', 
            opt.in_ch_dynamic + opt.in_ch_static, 
            hidden_sizes=opt.hidden_size, 
            kernel_sizes=3, 
            n_layers=opt.depth,
            init_type=opt.init_type,
            init_gain=opt.init_gain,
            gpu_ids=opt.gpu_ids,
            patch_size=opt.crop_size,
            sequence_length=self.n_days,
            partial=self.partial,
            efficientnet_level=opt.efficientnet_b,
            )

        self.netModel_NoMom = copy.deepcopy(self.netModel)
        for p, p_m in zip(self.netModel_NoMom.parameters(), self.netModel.parameters()):
            # p_m.data.copy_(p.data)
            p_m.requires_grad = False

        # summary(self.netModel_NoMom, (opt.in_ch_dynamic + opt.in_ch_static, opt.crop_size, opt.crop_size))
        # raise Exception

        

        self.L1 = torch.nn.L1Loss(reduction = 'none')


        self.criterionLoss = GaussianNLLLoss(reduction = 'none')
        # self.criterionLoss = LaplacianNLLLoss(reduction = 'none')

        # ADAM seems to be significantly worse than SGD, at least for ConvGRU
        # self.optimizer_regression = torch.optim.Adam(self.netModel.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        # self.optimizer_regression = torch.optim.Adam(self.netModel.parameters(), lr=opt.lr)

        if opt.phase != 'test':
            if opt.adam:
                self.optimizer_regression = torch.optim.Adam(self.netModel_NoMom.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
            else:
                self.optimizer_regression = torch.optim.SGD(self.netModel_NoMom.parameters(), lr=opt.lr, momentum=0.8, weight_decay=opt.weight_decay)
            self.optimizers.append(self.optimizer_regression)


    def reshape_for_TCN(self, dynamic_inputs, static_inputs):
        """Reshape variables for using TCN architecture and its variants.

        Parameters:

            ``dynamic_inputs``: Inputs with a time component, e.g. satellite images, weather, etc. 

            ``static_inputs``: Inputs without a time component, e.g. DEM and derived features.

        Returns:
            
            ``inputs``: Properly formatted variables for processing with TCN.
        """
        static_inputs = static_inputs.unsqueeze(1).repeat(1, dynamic_inputs.size(1), 1, 1, 1)
        inputs = torch.cat((dynamic_inputs, static_inputs), 2)
        inputs = inputs.transpose(1, 2)

        return inputs
    
    def add_graph(self, input_dict):
        """Add network graph to tensorboard for visualization of network architecture.

        Parameters:

            ``input_dict``: Example batch of data to be used for generating graph.
        """
        if self.rnn_type in ['STCNN', 'TempCNN']:
            self.inputs = self.reshape_for_TCN(
                input_dict['dynamic_inputs'].to(self.device),
                input_dict['static_inputs'].to(self.device),
            )

            if self.partial:
                self.masks = self.reshape_for_TCN(
                    input_dict['dynamic_masks'].to(self.device),
                    input_dict['static_masks'].to(self.device),
                )

        else:
            self.dynamic_inputs = input_dict['dynamic_inputs'][:,0,:,:,:].to(self.device)
            self.static_inputs = input_dict['static_inputs'].to(self.device)
            self.inputs = torch.cat((self.dynamic_inputs, self.static_inputs), 1)
        
        if self.partial:
            # self.writer.add_graph(self.netModel, *[self.inputs, self.masks])
            pass
        else:
            self.writer.add_graph(self.netModel_NoMom, self.inputs)

            
    def set_input(self, input_dict):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            
            ``input_dict``: dictionary containing batch data

        """
                
        self.dynamic_inputs = input_dict['dynamic_inputs'].to(self.device) # [B, T, C, M, N]
        # print(f'self.dynamic_inputs shape: {self.dynamic_inputs.shape}')
        self.static_inputs = input_dict['static_inputs'].to(self.device) # [B, C, M, N]
        # print(f'self.static_inputs shape: {self.static_inputs.shape}')
        self.snow_depth = input_dict['snow_depth'].to(self.device) # [B, T, M, N]
        self.snow_depth /= self.output_scale
        # print(f'self.snow_depth shape: {self.snow_depth.shape}')
        self.snow_depth_valid_flags = input_dict['snow_depth_valid_flags'].to(self.device) # [B, T]
        # print(f'self.snow_depth_valid_flags shape: {self.snow_depth_valid_flags.shape}')
        self.mask = input_dict['mask'].to(self.device) # [B, 1, M, N]
        # print(f'self.mask shape: {self.mask.shape}')
        # self.date_list = input_dict['date_list']

        if self.partial:
            self.dynamic_masks = input_dict['dynamic_masks'].to(self.device) # [B, T, C, M, N]
            self.static_masks = input_dict['static_masks'].to(self.device) # [B, C, M, N]
        
#         assert(False)
        

    def forward(self, validation=False):
        """Run forward pass using latest data received by ``set_input``.

        Should be called by ``optimize_parameters``.

        Outputs are stored in ``self.outputs``.
        """
        # self.outputs = self.netModel(self.inputs).squeeze(1)
        if self.rnn_type in ['STCNN', 'TempCNN']:
            print('Gaussian NLLL not implemented yet for TCNs')
            raise NotImplementedError
            self.inputs = self.reshape_for_TCN(self.dynamic_inputs, self.static_inputs)
            if self.partial:
                self.masks = self.reshape_for_TCN(self.dynamic_masks, self.static_masks)
                if validation:
                    self.outputs = self.netModel(self.inputs, self.masks)
                else:
                    self.outputs = self.netModel_NoMom(self.inputs, self.masks)
            else:
                if validation:
                    self.outputs = self.netModel(self.inputs)
                else:
                    self.outputs = self.netModel_NoMom(self.inputs)
        else:
            self.outputs = []
            self.outputs_vars = []
            self.hidden = None
            self.hidden_mid = None
            self.hidden_head = None
            self.cell_state = None # For ConvLSTM
            for i in range(self.dynamic_inputs.shape[1]):
                self.inputs = torch.cat((self.dynamic_inputs[:,i,:,:,:], self.static_inputs), 1)
                if self.rnn_type == 'ConvLSTM':
                    print('Gaussian NLLL not implemented yet for ConvLSTM')
                    raise NotImplementedError
                    if validation:
                        self.hidden, self.cell_state = self.netModel(self.inputs, hidden_state=self.hidden, cell_state=self.cell_state)
                        self.outputs.append(self.netModel.module.forward_tail(self.hidden))
                    else:
                        self.hidden, self.cell_state = self.netModel_NoMom(self.inputs, hidden_state=self.hidden, cell_state=self.cell_state)
                        self.outputs.append(self.netModel_NoMom.module.forward_tail(self.hidden))
                elif self.rnn_type == 'RUN':
                    print('Gaussian NLLL not implemented yet for RUN')
                    raise NotImplementedError
                    if validation:
                        output, self.hidden_mid, self.hidden_head = self.netModel(self.inputs, hidden_mid=self.hidden_mid, hidden_head=self.hidden_head)
                    else:
                        output, self.hidden_mid, self.hidden_head = self.netModel_NoMom(self.inputs, hidden_mid=self.hidden_mid, hidden_head=self.hidden_head)
                    self.outputs.append(output)
                else:
                    if validation:
                        self.hidden = self.netModel(self.inputs, hidden=self.hidden)
                        self.outputs.append(self.netModel.module.forward_tail(self.hidden))
                        self.outputs_vars.append(self.netModel.module.forward_tail_var(self.hidden))
                    else:
                        self.hidden = self.netModel_NoMom(self.inputs, hidden=self.hidden)
                        self.outputs.append(self.netModel_NoMom.module.forward_tail(self.hidden))
                        self.outputs_vars.append(self.netModel_NoMom.module.forward_tail_var(self.hidden))


    def backward(self, validation=False):
        """Calculate the loss and gradients through backpropagation.

        Should be called by ``optimize_parameters``.
        
        Parameters:
        
            ``validation``: Boolean that controls whether to run in training or validation mode.
        """
        
        # Masked loss at every time step
        self.loss_sample = 0.0
        self.l1_sample = 0.0

        if self.rnn_type in ['STCNN', 'TempCNN']:
            c = 0
            for b in range(self.batch_size):
                for d in [self.n_days - 1]:
                    if self.snow_depth_valid_flags[b, d] == 1: # should always occur
                        loss_sample_day = self.criterionLoss(self.outputs[b,0,:,:], self.snow_depth[b,d,:,:], self.outputs_vars[b,0,:,:]) * self.mask[b,0,:,:]
                        self.loss_sample += loss_sample_day.sum() / self.mask[b,0,:,:].sum()
                        with torch.no_grad():
                            l1_sample_day = self.L1(self.outputs[b,0,:,:], self.snow_depth[b,d,:,:]) * self.mask[b,0,:,:]
                            self.l1_sample += l1_sample_day.sum() / self.mask[b,0,:,:].sum()

                        c += 1
                    else:
                        print('Why is the last day not valid!?')
                        raise Exception
        else:        
            c = 0
            for b in range(self.batch_size):
                if validation: # validation only on last sample
                    for d in range(self.ifn, self.n_days):
                        if self.snow_depth_valid_flags[b, d] == 1: # should always occur
                            loss_sample_day = self.criterionLoss(self.outputs[d][b,0,:,:], self.snow_depth[b,d,:,:], self.outputs_vars[d][b,0,:,:]) * self.mask[b,0,:,:]
                            self.loss_sample += loss_sample_day.sum() / self.mask[b,0,:,:].sum()
                            with torch.no_grad():
                                l1_sample_day = self.L1(self.outputs[d][b,0,:,:], self.snow_depth[b,d,:,:]) * self.mask[b,0,:,:]
                                self.l1_sample += l1_sample_day.sum() / self.mask[b,0,:,:].sum()
                            c += 1
                        # else:
                        #     print('Why is the last day not valid!?')
                        #     raise Exception
                else:
                    for d in range(self.ifn, self.n_days):
                        if self.snow_depth_valid_flags[b, d] == 1:
                            loss_sample_day = self.criterionLoss(self.outputs[d][b,0,:,:], self.snow_depth[b,d,:,:], self.outputs_vars[d][b,0,:,:]) * self.mask[b,0,:,:]
                            if torch.isnan(loss_sample_day.sum()):
                                print('oh no')
                            self.loss_sample += loss_sample_day.sum() / self.mask[b,0,:,:].sum()
                            with torch.no_grad():
                                l1_sample_day = self.L1(self.outputs[d][b,0,:,:], self.snow_depth[b,d,:,:]) * self.mask[b,0,:,:]
                                self.l1_sample += l1_sample_day.sum() / self.mask[b,0,:,:].sum()
                            c += 1
            self.loss_sample /= c
            self.l1_sample /= c
        

        if not torch.isnan(self.loss_sample):
            if not validation:
                self.loss_sample.backward()

            with torch.no_grad():
                self.Loss_acc += self.loss_sample
                self.L1_acc += self.l1_sample
                self.niter_since_flush += 1
        else:
            # NaNs seem to come from exploding gradients if inputs are not normalized
            print('========== Warning: Found NaNs when computing loss function. ==========')
        

    def optimize_parameters(self, validation=False):
        """Wrapper function for performing optimization step.
        
        Parameters:

            ``validation``: Boolean that controls whether to run in training or validation mode.
        """
        # forward
        self.forward(validation=validation)    
        self.optimizer_regression.zero_grad()  
        self.backward(validation=validation)
        
        # TODO: find a better fix, find origin of NaNs
        if not torch.isnan(self.loss_sample) and not validation:
            # torch.nn.utils.clip_grad_value_(self.netModel_NoMom.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.netModel_NoMom.parameters(), self.gradient_clipping)
            self.optimizer_regression.step()

            for p, p_m in zip(self.netModel_NoMom.parameters(), self.netModel.parameters()):
                p_m.data = self.momentum * p_m.data + (1 - self.momentum) * p.data


    def compute_visuals(self, validation=False):
        """Compute visuals (scalars and images) to be sent to tensorboard

        Parameters:

            ``validation``: Boolean that controls whether to run in training or validation mode.
        """

        if validation:
            # self.writer.add_scalar('Loss/Loss_val', self.Loss_acc / self.niter_since_flush, x)
            self.Loss_val = self.Loss_acc / self.niter_since_flush
            self.metric = self.Loss_val # for controlling 'plateau' scheduler if it is used

            self.L1_val = self.L1_acc / self.niter_since_flush

            # find a usable image
            idx = -1
            if self.in_ch_dynamic != 0:
                for i in range(self.dynamic_inputs.shape[1] - 1, -1, -1):
                    if self.dynamic_inputs[0,i,-1,:,:].std() > 0.001:
                        idx = i
                        break

            # Align colour channels in RGB and undo normalization
            if self.in_ch_dynamic == 12:
                self.vis_in = torch.flip(self.dynamic_inputs[0,idx,1:4,:,:], [0])
                self.vis_in[0,:,:] = 3211.796362007882 * self.vis_in[0,:,:] + 1594.9361073193509
                self.vis_in[1,:,:] = 3256.0881664722733 * self.vis_in[1,:,:] + 1623.4478750096187
                self.vis_in[2,:,:] = 3336.978820889729 * self.vis_in[2,:,:] + 1654.9863773107784
            elif self.in_ch_dynamic == 4:
                self.vis_in = self.dynamic_inputs[0,idx,0:3,:,:]
                self.vis_in[0,:,:] = 8.196131354631488 * self.vis_in[0,:,:] - 4.935377397788123
                self.vis_in[1,:,:] = 5.7571205106891385 * self.vis_in[1,:,:] - 3.1383077790080556
                self.vis_in[2,:,:] = 8.177076673477872 * self.vis_in[2,:,:] - 4.924559624020597
                self.vis_in -= self.vis_in.min()
            elif self.in_ch_dynamic == 0:
                self.vis_in = self.static_inputs[0,0:3,:,:]
                self.vis_in -= self.vis_in.min()
            else:
                # self.vis_in = torch.flip(self.dynamic_inputs[0,idx,6:9,:,:], [0])
                self.vis_in = torch.flip(self.dynamic_inputs[0,idx,5:8,:,:], [0])
                self.vis_in[0,:,:] = 3211.796362007882 * self.vis_in[0,:,:] + 1594.9361073193509
                self.vis_in[1,:,:] = 3256.0881664722733 * self.vis_in[1,:,:] + 1623.4478750096187
                self.vis_in[2,:,:] = 3336.978820889729 * self.vis_in[2,:,:] + 1654.9863773107784
            # Ensure proper visualization

            self.vis_in /= self.vis_in.max()


            # Extract and co-normalize the snow depth maps
            date_idx = torch.where(self.snow_depth_valid_flags[0])[0].max().cpu().numpy()
            if self.rnn_type in ['STCNN', 'TempCNN']:
                self.vis_out = self.outputs[0,:,:,:] * self.output_scale
            else:
                self.vis_out = self.outputs[date_idx][0,:,:,:] * self.output_scale
                self.vis_out_vars = torch.exp(self.outputs_vars[date_idx][0,:,:,:]) * self.output_scale
                self.pred_mean = self.vis_out.mean()
                self.var_mean = self.outputs_vars[date_idx][0,:,:,:].mean()
            self.vis_gt = self.snow_depth[0,date_idx,:,:] * self.output_scale
            self.gt_mean = self.vis_gt.mean()
            m = max(self.vis_out.max(), self.vis_gt.max())
            self.vis_out /= m
            self.vis_gt /= m
            self.vis_out_vars /= self.vis_out_vars.max()



        else:
            # self.writer.add_scalar('Loss/Loss_train', self.Loss_acc / self.niter_since_flush, x)
            self.Loss_train = self.Loss_acc / self.niter_since_flush
            self.L1_train = self.L1_acc / self.niter_since_flush


        self.flush()

    def flush(self):
        """Clear class metric variables for next epoch
        """

        self.Loss_acc = 0.0
        self.L1_acc = 0.0
        self.niter_since_flush = 0

        return

    def send_to_tensorboard(self, x):
        """Send visuals to tensorboard

        Parameters:

            ``x``: X axis value (epoch number)
        """

        if x > 1:
            tag_scalar_dict = {
                'Train': self.Loss_train,
                'Val': self.Loss_val,
            }
            self.writer.add_scalars('Loss', tag_scalar_dict, x)

            tag_scalar_dict = {
                'Train': self.L1_train,
                'Val': self.L1_val,
            }
            self.writer.add_scalars('MAE', tag_scalar_dict, x)

            tag_scalar_dict = {
                'pred_mean': self.pred_mean,
                'gt_mean': self.gt_mean,
            }
            self.writer.add_scalars('Mean of predictions', tag_scalar_dict, x)

            tag_scalar_dict = {
                'var_mean': self.var_mean,
            }
            self.writer.add_scalars('Mean of log variances', tag_scalar_dict, x)

        # print(self.vis_in.shape)
        # print(self.vis_out.shape)
        # print(self.vis_gt.shape)

        self.writer.add_image('Images/01-Input', self.vis_in, x, dataformats='CHW')
        self.writer.add_image('Images/02-Ground_truth', self.vis_gt, x, dataformats='HW')
        self.writer.add_image('Images/03-Output', self.vis_out, x, dataformats='CHW')
        self.writer.add_image('Images/04-Estimated_variance', self.vis_out_vars, x, dataformats='CHW')

        self.writer.flush()
        a = 0


        
        
    # def compute_visuals(self):
    #     """Calculate additional output images for visdom and HTML visualization"""
        
    #     self.I_in = self.inputs[:1,:3,:,:]
    #     self.depth_gt = self.snow_depth.unsqueeze(1)
    #     self.depth_pred = self.outputs.unsqueeze(1)
