import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2
import collections
from py_utils import L2Normalization, MaxPool2dSamePadding, Conv2dSamePadding, CatLayer, EmptyLayer
from .utils import parse_cfg



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

