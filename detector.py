import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2
from .utils import parse

# from utils import L2Normalization

# from utils import MaxPool2dSamePadding


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()
    
    def forward(self, x):
        return x


class L2Normalization(nn.Module):
    '''
    Performs L2 normalization on the input tensor with a learnable scaling parameter
    as described in the paper "Parsenet: Looking Wider to See Better" (see references)
    and as used in the original SSD model.

    Arguments:
        gamma_init (int): The initial scaling parameter. Defaults to 20 following the
            SSD paper.

    Input shape:
        4D tensor of shape `(batch, channels, height, width)` 

    Returns:
        The scaled tensor. Same shape as the input tensor.

    References:
        http://cs.unc.edu/~wliu/papers/parsenet.pdf
    '''

    def __init__(self, gamma_init=20, axis=1, norm_shape=1): 
        super(L2Normalization, self).__init__()
        self.axis = axis
        self.gamma_init = gamma_init
        gamma = self.gamma_init * np.ones(norm_shape)
        self.gamma = nn.Parameter(torch.tensor(gamma))
    
    
    def forward(self, x):
        output = F.normalize(x, p=2, dim=self.axis)
        output *= self.gamma.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(x.shape[0], -1, x.shape[2], x.shape[3])
        return output



class CatLayer(nn.Module):
    # def __init__(self, catinfo, index, output_filters):
    #     super(CatLayer, self).__init__()
    #     layers = [int(i) for i in catinfo['layers'].split(',')]
    #     deconv_filters = [int(i) for i in catinfo['filters'].split(',')]
    #     deconv_size = [int(i) for i in catinfo['size'].split(',')]
    #     deconv_stride = [int(i) for i in catinfo['stride'].split(',')]
    #     deconv_padding = [int(i) for i in catinfo['pad'].split(',')]
    #     gamma_init = [int(i) for i in catinfo['gamma_init'].split(',')]
    #     self.deconv_list = []
    #     self.norm_list = [] 
            
    #     for i, l_ind in enumerate(layers):
    #         in_filters = output_filters[l_ind + index]
    #         deconv = nn.ConvTranspose2d(in_filters, deconv_filters[i], deconv_size[i], stride=deconv_stride[i], padding=deconv_padding[i])
    #         L2_norm = L2Normalization(gamma_init[i], 1, deconv_filters[i])
    #         self.deconv_list.append(deconv)
    #         self.norm_list.append(L2_norm)

    def __init__(self, catinfo, index, output_filters):
        super(CatLayer, self).__init__()
        layers = [int(i) for i in catinfo['layers'].split(',')]
        deconv_filters = [int(i) for i in catinfo['filters'].split(',')]
        deconv_size = [int(i) for i in catinfo['size'].split(',')]
        deconv_stride = [int(i) for i in catinfo['stride'].split(',')]
        deconv_padding = [int(i) for i in catinfo['pad'].split(',')]
        gamma_init = [int(i) for i in catinfo['gamma_init'].split(',')]
       
        for i, l_ind in enumerate(layers):
            in_filters = output_filters[l_ind + index]
            deconv = nn.ConvTranspose2d(in_filters, deconv_filters[i], deconv_size[i], stride=deconv_stride[i], padding=deconv_padding[i])
            self.add_module("deconv_{}".format(i), deconv)
            L2_norm = L2Normalization(gamma_init[i], 1, deconv_filters[i])
            self.add_module("L2norm_{}".format(i), L2_norm)
        
    
    def forward(self, xs):
        t = None
        child_iter = self.children()
        for ind, x in enumerate(xs):
            x = next(child_iter)(x)
            x = next(child_iter)(x)
            if t == None:
                t = x
            else:
                t = torch.cat((t, x), 1)
        return t
            


class Conv2dSamePadding(nn.Module):
    # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', padding_mode_tf='valid'):
        super(Conv2dSamePadding, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding_mode_tf = padding_mode_tf
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        if self.padding_mode_tf == 'same':
            x = same_padding(x, x.shape[2:], self.kernel_size, self.stride, self.dilation)
        x = self.conv(x)
        return x

 

class MaxPool2dSamePadding(nn.Module):
    def __init__(self, kernel_size, stride, pool_type, dilation=1, padding_mode_tf='valid'):
        super(MaxPool2dSamePadding, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding_mode_tf = padding_mode_tf
        self.pool_type = pool_type
        if self.pool_type == 'maxpool2d':
            self.pool = nn.MaxPool2d(self.kernel_size, self.stride)
        else:
            pass
    
    def forward(self, x):
        if self.padding_mode_tf == 'same':
            x = same_padding(x, x.shape[2:], self.kernel_size, self.stride, self.dilation)
        x = self.pool(x)
        return x 


def same_padding(input, input_size, kernel_size, stride, dilation=1):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    input_rows = input_size[0]
    filter_rows = kernel_size[0]
    effective_filter_size_rows = (filter_rows - 1) * dilation[0] + 1
    out_rows = (input_rows + stride[0] - 1) // stride[0]
    padding_rows = max(0, (out_rows - 1) * stride[0] +
                        (filter_rows - 1) * dilation[0] + 1 - input_rows)
    rows_odd = (padding_rows % 2 != 0)


    input_cols = input_size[1]
    filter_cols = kernel_size[1]
    effective_filter_size_cols = (filter_cols - 1) * dilation[1] + 1
    out_cols = (input_cols + stride[1] - 1) // stride[1]
    padding_cols = max(0, (out_cols - 1) * stride[1] +
                        (filter_cols - 1) * dilation[1] + 1 - input_cols)
    cols_odd = (padding_cols % 2 != 0)

    if rows_odd or cols_odd:
        input = F.pad(input, [0, int(cols_odd), 0, int(rows_odd)])
    input = F.pad(input, (padding_cols // 2, padding_cols // 2, padding_rows // 2, padding_rows // 2), "constant", 0)
    
    return input

# def same_padding(x, input_size, kernel_size, stride, dilation=1):
#     if isinstance(kernel_size, int):
#         kernel_size = (kernel_size, kernel_size)
#     if isinstance(stride, int):
#         stride = (stride, stride)
#     if isinstance(dilation, int):
#         dilation = (dilation, dilation)
#     filter_h = (kernel_size[0] - 1) * dilation[0] + 1
#     filter_w = (kernel_size[1] - 1) * dilation[1] + 1
#     leftover_bottom = (input_size[0] - filter_h) % stride[0]
#     pad_bottom = 0 if leftover_bottom == 0 else stride[0] - leftover_bottom
#     leftover_right = (input_size[1] - filter_w) % stride[1]
#     pad_right = 0 if leftover_right == 0 else stride[1] - leftover_right
#     return F.pad(x, (0, leftover_right, 0, leftover_bottom), "constant", 0)



def parse_cfg(cfgfile):
    '''
    takes a configuration file
    returns a list of blocks. Each block describes a block in the neural network to be built.
    block is represented as a dictionary in the list
    '''
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if len(x) > 0] # get rid of empty lines
    lines = [x for x in lines if x[0] != '#'] # get rid of comment lines
    lines = [x.lstrip().rstrip() for x in lines] # get rid of frange whitespaces
    
    blocks = []
    block = {}

    for line in lines:
        if line[0] == '[': # start of the new block
            if len(block) != 0: # content of the last block 
                blocks.append(block)
                block = {} # re-init dict for the next block
                block['type'] = line[1:-1].rstrip()
        else:
            key, value = line.split('=')
            block[key.rstrip()] = value.lstrip()
    blocks.append(block) # store the last one

    return blocks
    


def create_modules(blocks):
    module_list = nn.ModuleList()
    prev_filters = 3 # RGB 3 channels
    output_filters = [] # the filters number of the output of each module before, note that is the output of each module, but not of a single layer

    input_info = {
        'height': blocks[0]['height'],
        'width': blocks[0]['width'],
        'filters': blocks[0]['filters']
    }
    filters = int(input_info['filters'])

    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()
        # check the type of block
        # create a new module for the block
        # append it to module_list
        if x['type'] == 'convolutional':
            # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
            activation = x['activation']
            filters = int(x['filters'])
            kernel_size = int(x['size'])
            stride = int(x['stride'])
            padding = int(x['pad'])
            try:
                batch_normalize = int(x['batch_normalize'])
            except:
                batch_normalize = 0

            try:
                padding_mode_tf = x['padding']
            except:
                padding_mode_tf = 'valid' 
            try:
                dilation = int(x['dilation_rate'])
            except:
                dilation = 1
            
            conv_same_padding = Conv2dSamePadding(prev_filters, filters, kernel_size, stride=stride, dilation=dilation, padding_mode_tf=padding_mode_tf)
            
            module.add_module("conv_same_padding_{}".format(index), conv_same_padding)

            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{}".format(index), bn)
            
            if activation == 'relu':
                activ = nn.ReLU()
                module.add_module("relu_{}".format(index), activ)
            else:
                pass
        elif x['type'] == 'pooling':
            # pooling layer
            # torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
            pool_type = x['pool_type']
            filters = prev_filters
            size = int(x['size'])
            stride = int(x['stride'])
            padding = int(x['pad'])
            pool = MaxPool2dSamePadding(size, stride, pool_type=pool_type, padding_mode_tf='same')
            module.add_module("maxpool2d_same_padding_{}".format(index), pool)

        elif x['type'] == 'padding':
            pad_type = x['pad_type']
            padding = int(x['pad'])
            if pad_type == 'zero':
                pad = nn.ZeroPad2d(padding)
                module.add_module("padding_{}".format(index), pad)
            else:
                pass
        elif x['type'] == 'route':
            print('*' * 10)
            print(index)
            print('*' * 10)
            # torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros')
            filters = [int(i) for i in x['filters'].split(',')]
            filters = sum(filters)
            catlayer = CatLayer(x, index, output_filters)
            module.add_module('route_{}'.format(index), catlayer)
        elif x['type'] == 'shortcut':
            convolutional = int(x['convolutional'])
            try:
                batch_normalize = x['batch_normalize']
            except:
                batch_normalize = 0
            from_ind = int(x['from'])
            if convolutional:
                filters = int(x['filters'])
                size = int(x['size'])
                stride = int(x['stride'])
                padding = int(x['pad'])
                conv = nn.Conv2d(output_filters[index + from_ind], filters, size, stride=stride, padding=padding)
                bn = nn.BatchNorm2d(filters)
                module.add_module('shortcut_conv_{}'.format(index), conv)
                module.add_module('shortcut_bn_{}'.format(index), bn)
            else:
                filters = prev_filters
                shortcut = EmptyLayer()
                module.add_module("shortcut_{}".format(index), shortcut)
        elif x['type'] == 'output':
            filters = int(x['filters'])
            size = int(x['size'])
            stride = int(x['stride'])
            padding = int(x['pad'])
            # TODO: specify the initialization method of convolution kernel and bias 
            try:
                kernel_init = x['kernel_initializer']
                bias_init = x['bias_initializer']
            except:
                pass
            from_ind = int(x['from'])
            in_filters = output_filters[index + from_ind]
            out = nn.Conv2d(in_filters, filters, size, stride=stride, padding=padding)
            module.add_module("output_conv_{}".format(index), out)
        else:
            pass

        prev_filters = filters
        output_filters.append(filters)
        module_list.append(module)
    return (input_info, module_list)


# # test
# blocks = parse_cfg('configs\\network_arch.cfg') 
# print(create_modules(blocks))       
            
class CSPNet(nn.Module):
    def __init__(self, cfgfile):
        super(CSPNet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.input_info, self.module_list = create_modules(self.blocks)
        # specific to the 
        self.same_padding = (0, 1, 0, 1)
    
    def forward(self, x, CUDA):
        modules = self.blocks[1:] # start from zeropadding -> 0
        outputs = {}
        write = 0
        output_ind = []
        for i, module in enumerate(modules):
            module_type = module['type']
            print(str(i) + ': ' + module_type)
            if module_type == 'convolutional' or module_type == 'padding' or module_type == 'pooling':
                x = self.module_list[i](x)
            elif module_type == 'route':
                # pytorch -> (batch, channel, height, width) catenate along the channel dimension
                layers = [int(a) for a in module['layers'].split(',')]
                xs = [outputs[i + la_ind] for la_ind in layers]
                x = self.module_list[i](xs)
            elif module_type == 'shortcut':
                from_ind = int(module['from'])
                activation = module['activation']
                pre_x = self.module_list[i](outputs[i + from_ind])
                x += pre_x
                if activation == 'relu':
                    x = F.relu(x)
                else:
                    pass
            elif module_type == 'output':
                from_ind = int(module['from'])
                activation = module['activation']
                prev_x = outputs[i + from_ind]
                x = self.module_list[i](prev_x)
                if activation == 'sigmoid':
                    x = torch.sigmoid(x)
                elif activation == 'linear':
                    # keep unchanged
                    pass
                else:
                    pass
                output_ind.append(i)
            outputs[i] = x
            print(x.shape)

        return [outputs[i] for i in output_ind]     

def get_test_input():
    img = cv2.imread('C:\\Users\\Elsa\\Pictures\\LPFOX.jfif')
    img = cv2.resize(img, (336, 448)) # width, height, channel
    img_ = img[:, :, ::-1].transpose(2, 0, 1)
    img_ = img_[np.newaxis, :, :, :] / 255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)
    return img_


# test forward
model = CSPNet("configs\\network_arch.cfg")
input = get_test_input()
pred = model(input, torch.cuda.is_available())
# torch.save(model.state_dict(), 'CSP_Pytorch_params1.pkl')

weights_lst = parse("net_e382_l0.hdf5")
weights_dict = collections.OrderedDict()
torch_params_arch = torch.load('CSP_Pytorch_params1.pkl')
for i, key in enumerate(torch_params_arch.keys()):
    weights_dict[key] = weights_lst[i] if not isinstance(weights_lst[i], str) else torch_params_arch[key]

torch.save(weights_dict, "CSP_Pytorch_e382_l0,.pkl")


# print('*' * 20)
# print('load model....')
# input = get_test_input()
# model = torch.load('csp_weights.pkl')
# print('done')
# print('*' * 20)
# pred = model(input, torch.cuda.is_available())
# print(pred)
# # model.load_state_dict(torch.load(PATH))
