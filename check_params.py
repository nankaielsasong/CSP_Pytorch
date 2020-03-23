from models import CSPNet
import torch
import h5py

model=CSPNet('configs/network_arch.cfg')
model.load_state_dict(torch.load('CSP_Pytorch_e382_l0.pkl'))
d = model.state_dict()

f = h5py.File('net_e382_l0.hdf5')
k_conv1 = f['conv1']['conv1_1']['kernel:0']
k_conv1 = torch.tensor(k_conv1).permute(3, 2, 1, 0)
print(k_conv1.shape)
p_conv1 = d['module_list.1.conv_same_padding_1.conv.weight']
print(p_conv1.shape)

print(k_conv1 == p_conv1)

k_feat = f['feat']['feat_1']['kernel:0']
k_feat = torch.tensor(k_feat).permute(3, 2, 1, 0)
p_feat = d['module_list.68.conv_same_padding_68.conv.weight']
print('*' * 30)
print(k_feat == p_feat)
