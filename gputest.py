import torch
#1.查看gpu信息
if_cuda = torch.cuda.is_available()
print('if_cuda=', if_cuda)

#GPU的数量
gpu_count = torch.cuda.device_count()
print('gpu_count=', gpu_count)
print(torch.__version__)