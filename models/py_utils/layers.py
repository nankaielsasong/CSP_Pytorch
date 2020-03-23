import torch.nn as nn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
            if t is None:
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
