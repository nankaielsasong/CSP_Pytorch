# 处理一下 keras 参数文件 => 转换成单级的 key, value 的形式存成一个 .pkl 文件 
# 按照命名顺序读取 keras 参数文件中的数值
# 按顺序读取 pytorch 参数文件中的数值
# 并按照命名顺序将 pytorch 读取结果中的 key，和 keras 读取结果中的 value, 存到 collections.OrderedDict 中 
# torch.save(dict, newfile.pkl) => 即得到了新的参数文件

import collections
import torch
# import h5py

# f = open('CSP_Pytorch_params.pkl', 'rb')
# data = torch.load('CSP_Pytorch_params1.pkl')
# print(type(data))
# count = 0

# for key, value in data.items():
#     print(key + ": ", end="")
#     print(value.shape)
#     count += 1
# print('total count is {} '.format(count)) 




def tf_to_torch(arr):
    t = torch.tensor(arr)
    return t.permute(3, 2, 1, 0)


def np_to_torch(arr):
    return torch.tensor(arr)




def parse_res_params(f, alp_lst_out, stage_num):
    count = 0
    weights_lst = []
    alp_lst_in = ['a', 'b', 'c']
    for i in alp_lst_out:
        for j in alp_lst_in:
            level_1 = 'res' + str(stage_num) + i + '_branch2' + j
            level_2 = level_1 + '_1'
            level_3_1 = 'kernel:0'
            level_3_2 = 'bias:0'
            
            bn_level_1 = 'bn' + str(stage_num) + i + '_branch2' + j
            bn_level_2 =  bn_level_1 + '_1'
            bn_level_3_1 = 'beta:0'
            bn_level_3_2 = 'gamma:0'
            bn_level_3_3 = 'moving_mean:0'
            bn_level_3_4 = 'moving_variance:0'

            # kernel weights
            weights_lst.append(tf_to_torch(f[level_1][level_2][level_3_1][()]))
            # kernel bias
            weights_lst.append(np_to_torch(f[level_1][level_2][level_3_2][()]))

            # BN beta, gamma, mean, variance
            weights_lst.append(np_to_torch(f[bn_level_1][bn_level_2][bn_level_3_1][()]))
            weights_lst.append(np_to_torch(f[bn_level_1][bn_level_2][bn_level_3_2][()]))
            weights_lst.append(np_to_torch(f[bn_level_1][bn_level_2][bn_level_3_3][()]))
            weights_lst.append(np_to_torch(f[bn_level_1][bn_level_2][bn_level_3_4][()]))
            weights_lst.append('skip')

            count += 7
        
        if i == 'a':
            # res2a_branch1 weights
            # kernel weights
            level_1 = 'res' + str(stage_num) + 'a_branch1'
            level_2 = level_1 + '_1'
            level_3_1 = 'kernel:0'
            level_3_2 = 'bias:0'

            bn_level_1 = 'bn' + str(stage_num) + 'a_branch1'
            bn_level_2 =  bn_level_1 + '_1'
            bn_level_3_1 = 'beta:0'
            bn_level_3_2 = 'gamma:0'
            bn_level_3_3 = 'moving_mean:0'
            bn_level_3_4 = 'moving_variance:0'

            weights_lst.append(tf_to_torch(f[level_1][level_2][level_3_1][()]))
            # kernel bias
            weights_lst.append(np_to_torch(f[level_1][level_2][level_3_2][()]))

            # BN beta, gamma, mean, variance
            weights_lst.append(np_to_torch(f[bn_level_1][bn_level_2][bn_level_3_1][()]))
            weights_lst.append(np_to_torch(f[bn_level_1][bn_level_2][bn_level_3_2][()]))
            weights_lst.append(np_to_torch(f[bn_level_1][bn_level_2][bn_level_3_3][()]))
            weights_lst.append(np_to_torch(f[bn_level_1][bn_level_2][bn_level_3_4][()]))
            weights_lst.append('skip')

            count += 7
    
    return weights_lst, count


def parse(hdf5_file_path):

    f = h5py.File(hdf5_file_path)
    weights_lst = []
    count = 0
    # pre ConvBN layer weight
    # conv1  conv1_1  bias:0 kernel
    weights_lst.append(tf_to_torch(f['conv1']['conv1_1']['kernel:0'][()]))
    weights_lst.append(np_to_torch(f['conv1']['conv1_1']['bias:0'][()]))
    weights_lst.append(np_to_torch(f['bn_conv1']['bn_conv1_1']['beta:0'][()]))
    weights_lst.append(np_to_torch(f['bn_conv1']['bn_conv1_1']['gamma:0'][()]))
    weights_lst.append(np_to_torch(f['bn_conv1']['bn_conv1_1']['moving_mean:0'][()]))
    weights_lst.append(np_to_torch(f['bn_conv1']['bn_conv1_1']['moving_variance:0'][()]))
    weights_lst.append('skip')

    count += 7

    alp_lst_out_1 = ['a', 'b', 'c']
    alp_lst_out_2 = ['a', 'b', 'c', 'd']
    alp_lst_out_3 = ['a', 'b', 'c', 'd', 'e', 'f']

    alp_lst_in = ['a', 'b', 'c']

    # resnet stage2 weight 
    # res2a_branch2a -> bn2a_branch2a -> res2a_branch2b -> bn2a_branch2b -> res2a_branch2c -> bn2a_banch2c
    # [res2a_branch2a_1][bias:0] / [res2a_branch2a_1][kernel:0] =>  [kernel, kernel, input, output] -----> [output, input, kernel, kernel]
    # [bn2a_branch2a_1][beta:0] / [gamma:0] / [moving_mean:0] / [moving_variance:0] [output, ]
    # res2a_branch1
    # bn2a_branch1
                
    # res2b_branch2a -> bn2b_branch2a -> res2b_branch2b -> bn2b_branch2b -> res2b_branch2c -> bn2b_branch2c
    # res2c_branch2a -> bn2c_branch2a -> res2c_branch2b -> bn2c_branch2b -> res2c_branch2c -> bn2c_branch2c
    w, c = parse_res_params(f, alp_lst_out_1, 2)
    weights_lst.extend(w)
    count += c

    # resnet stage3 weight 
    # res3a_branch2a -> bn3a_branch2a -> res3a_branch2b -> bn3a_branch2b -> res3a_branch2c -> bn3a_banch2c
    # [res3a_branch2a_1][bias:0] / [res3a_branch2a_1][kernel] =>  [kernel, kernel, input, output] -----> [output, input, kernel, kernel]
    # [bn3a_branch2a_1][beta:0] / [gamma:0] / [moving_mean:0] / [moving_variance:0] [output, ]
    # res3a_branch1
    # bn3a_branch1


    # res3b_branch2a -> bn3b_branch2a -> res3b_branch2b -> bn3b_branch2b -> res3b_branch2c -> bn2b_branch2c
    # res3c_branch2a -> bn3c_branch2a -> res3c_branch2b -> bn3c_branch2b -> res3c_branch2c -> bn2c_branch2c
    # res3d_branch2a -> bn3d_branch2a -> res3d_branch2b -> bn3d_branch2b -> res3d_branch2c -> bn2d_branch2c
    w, c = parse_res_params(f, alp_lst_out_2, 3)
    weights_lst.extend(w)
    count += c


    # resnet stage4 weight 
    # res4a_branch2a -> bn4a_branch2a -> res4a_branch2b -> bn4a_branch2b -> res4a_branch2c -> bn4a_banch2c
    # [res4a_branch2a_1][bias:0] / [res4a_branch2a_1][kernel] =>  [kernel, kernel, input, output] -----> [output, input, kernel, kernel]
    # [bn4a_branch2a_1][beta:0] / [gamma:0] / [moving_mean:0] / [moving_variance:0] [output, ]
    # res4a_branch1
    # bn4a_branch1


    # res4b_branch2a -> bn4b_branch2a -> res4b_branch2b -> bn4b_branch2b -> res4b_branch2c -> bn4b_branch2c
    # res4c_branch2a -> bn4c_branch2a -> res4c_branch2b -> bn4c_branch2b -> res4c_branch2c -> bn4c_branch2c
    # res4d_branch2a -> bn4d_branch2a -> res4d_branch2b -> bn4d_branch2b -> res4d_branch2c -> bn4d_branch2c
    # res4e_branch2a -> bn4e_branch2a -> res4e_branch2b -> bn4e_branch2b -> res4e_branch2c -> bn4e_branch2c
    # res4f_branch2a -> bn4f_branch2a -> res4f_branch2b -> bn4f_branch2b -> res4f_branch2c -> bn4f_branch2c
    w, c = parse_res_params(f, alp_lst_out_3, 4)
    weights_lst.extend(w)
    count += c

    # resnet stage5 weight 
    # res5a_branch2a -> bn5a_branch2a -> res5a_branch2b -> bn5a_branch2b -> res5a_branch2c -> bn5a_banch2c
    # [res5a_branch2a_1][bias:0] / [res5a_branch2a_1][kernel:0] =>  [kernel, kernel, input, output] -----> [output, input, kernel, kernel]
    # [bn5a_branch2a_1][beta:0] / [gamma:0] / [moving_mean:0] / [moving_variance:0] [output, ]
    # res5a_branch1
    # bn5a_branch1


    # res5b_branch2a -> bn5b_branch2a -> res5b_branch2b -> bn5b_branch2b -> res5b_branch2c -> bn5b_branch2c
    # res5c_branch2a -> bn5c_branch2a -> res5c_branch2b -> bn5c_branch2b -> res5c_branch2c -> bn5c_branch2c
    # res5d_branch2a -> bn5d_branch2a -> res5d_branch2b -> bn5d_branch2b -> res5d_branch2c -> bn5d_branch2c
    w, c = parse_res_params(f, alp_lst_out_1, 5)
    weights_lst.extend(w)
    count += c

    # route layer weights
    # P5up 
    # [P5up1][bias:0] / [kernel:0]
    # [P5norm][P5norm1][P5norm_gamma:0]
    weights_lst.append(tf_to_torch(f['P5up']['P5up_1']['kernel:0'][()]))
    weights_lst.append(np_to_torch(f['P5up']['P5up_1']['bias:0'][()]))
    weights_lst.append(np_to_torch(f['P5norm']['P5norm_1']['P5norm_gamma:0'][()]))
    count += 3


    # P4up
    # [P4up1][bias:0] / [kernel:0]
    # [P4norm][P4norm1][P3norm_gamma:0]
    weights_lst.append(tf_to_torch(f['P4up']['P4up_1']['kernel:0'][()]))
    weights_lst.append(np_to_torch(f['P4up']['P4up_1']['bias:0'][()]))
    weights_lst.append(np_to_torch(f['P4norm']['P4norm_1']['P4norm_gamma:0'][()]))
    count += 3


    # P3up
    # [P3up_1][bias:0] / [kernel:0]
    # [P3norm][P3norm_1][P3norm_gamma:0]
    weights_lst.append(tf_to_torch(f['P3up']['P3up_1']['kernel:0'][()]))
    weights_lst.append(np_to_torch(f['P3up']['P3up_1']['bias:0'][()]))
    weights_lst.append(np_to_torch(f['P3norm']['P3norm_1']['P3norm_gamma:0'][()]))    
    count += 3


    # feat
    # bn_feat
    weights_lst.append(tf_to_torch(f['feat']['feat_1']['kernel:0'][()]))
    weights_lst.append(np_to_torch(f['feat']['feat_1']['bias:0'][()]))
    weights_lst.append(np_to_torch(f['bn_feat']['bn_feat_1']['beta:0'][()]))
    weights_lst.append(np_to_torch(f['bn_feat']['bn_feat_1']['gamma:0'][()]))
    weights_lst.append(np_to_torch(f['bn_feat']['bn_feat_1']['moving_mean:0'][()]))
    weights_lst.append(np_to_torch(f['bn_feat']['bn_feat_1']['moving_variance:0'][()]))
    weights_lst.append('skip')
    count += 7
    
    # center_cls
    weights_lst.append(tf_to_torch(f['center_cls']['center_cls_1']['kernel:0'][()]))
    weights_lst.append(np_to_torch(f['center_cls']['center_cls_1']['bias:0'][()]))
    count += 2

    # height_regr
    weights_lst.append(tf_to_torch(f['height_regr']['height_regr_1']['kernel:0'][()]))
    weights_lst.append(np_to_torch(f['height_regr']['height_regr_1']['bias:0'][()]))
    count += 2

    # offset_regr
    weights_lst.append(tf_to_torch(f['offset_regr']['offset_regr_1']['kernel:0'][()]))
    weights_lst.append(np_to_torch(f['offset_regr']['offset_regr_1']['bias:0'][()]))
    count += 2

    print("total count is: {}".format(count))
    print("total count is: {}".format(len(weights_lst)))

    return weights_lst

    # for i in weights_lst:
    #     if isinstance(i, str):
    #         continue

    #     print(i.shape)
