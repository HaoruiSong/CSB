import torch
import torch.nn as nn
from data import Data
from opt import  opt
from network import ResnetConditionHR

data = Data()
train_loader = data.train_loader_woEr
test_loader = data.test_loader
query_loader = data.query_loader
testset = data.testset
queryset = data.queryset


def get_positive_pairs():
    idx = []
    for i in range(opt.batchimage * opt.batchid):
        r = i
        while r == i:
            r = int(torch.randint(
                low=opt.batchimage * (i // opt.batchimage), high=opt.batchimage * (i // opt.batchimage + 1),
                size=(1,)).item())
        idx.append(r)
    return idx

netM = ResnetConditionHR(input_nc=(3,3,3,1),output_nc=4,n_blocks1=7,n_blocks2=3)
netM = nn.DataParallel(netM)
netM.cuda()

def train():
    # 仅：输入四张图-->CSB输出特征 每张图输出256*96*32，后面接分类网络

    for batch, (rgb, sketch, gray, labels) in enumerate(train_loader):
        positive_idx = get_positive_pairs()
        positive = rgb[positive_idx] #到这里得到rgb、positive、sketch、gray分别为四种输入图

        feature = netM(rgb, positive, sketch, gray) #这里是直接把四种图输入到CSB，得到输出为feature。已测试过可以运行。features将接到后面的分类网络里面。


        break

if __name__ == '__main__':

    train()