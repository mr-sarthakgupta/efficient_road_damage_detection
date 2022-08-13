import torch
from torchsummary import summary
from get_segformer import get_segformer
from fast_seg import get_colorized_seg_map, get_fastseg_model
device = torch.device('cpu')
# segformer = get_segformer().eval().to('cuda:0')
fastseg = get_fastseg_model().eval()
x = torch.zeros(1, 3, 2462, 2048).to(device)
x = x.to('cuda:0')
# print(segformer(x))
# torch.set_printoptions(edgeitems=10000)
y = fastseg(x)[:, 7, :, :]
# print(y.shape)
# summary(segformer, (3, 2462, 2048))