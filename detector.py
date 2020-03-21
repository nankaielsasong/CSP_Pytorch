import collections
import torch
from models import CSPNet


def get_test_input():
    img = cv2.imread('/home/zk/Desktop/lena.jpeg')
    img = cv2.resize(img, (336, 448)) # width, height, channel
    img_ = img[:, :, ::-1].transpose(2, 0, 1)
    img_ = img_[np.newaxis, :, :, :] / 255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)
    return img_


# test forward
model = CSPNet("configs/network_arch.cfg")
input = get_test_input()
pred = model(input, torch.cuda.is_available())
torch.save(model.state_dict(), 'CSP_Pytorch_params.pkl')

weights_lst = parse("net_e382_l0.hdf5")
weights_dict = collections.OrderedDict()
torch_params_arch = torch.load('CSP_Pytorch_params.pkl')
for i, key in enumerate(torch_params_arch.keys()):
    weights_dict[key] = weights_lst[i] if not isinstance(weights_lst[i], str) else torch_params_arch[key]

torch.save(weights_dict, "CSP_Pytorch_e382_l0.pkl")

print('load model....')
model.load_state_dict(torch.load('CSP_Pytorch_e382_l0.pkl'))
pred = model(input, torch.cuda.is_available())
print(pred)
