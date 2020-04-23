'''Pre-activation ResNet in PyTorch.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class PreActResNet(nn.Module):

    def __init__(self, num_channels=5, num_classes=225):
        super(PreActResNet, self).__init__()
        self.in_planes = 1

        self.other_layers = nn.ModuleList()

        self.conv1 = nn.Conv2d(self.in_planes, num_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.layer_one = self.conv1

        self.blocks = nn.Sequential(
            PreActBlock(num_channels),
            nn.AvgPool2d(3, 2, 1),
            PreActBlock(num_channels),
            nn.AvgPool2d(3, 2, 1),

            nn.BatchNorm2d(num_channels),
            nn.Conv2d(num_channels, 2*num_channels, kernel_size=3, stride=1, padding=1, bias=False),
        )


        self.linear = GlobalpoolFC(2*num_channels, num_classes)


    def forward(self, x):

        x = self.layer_one(x)

        x = self.blocks(x)

        pred = self.linear(x)

        return pred

class GlobalpoolFC(nn.Module):

    def __init__(self, num_in, num_class):
        super(GlobalpoolFC, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(num_in, num_class)

    def forward(self, x):
        y = self.pool(x)
        y = y.reshape(y.shape[0], -1)
        y = self.fc(y)
        return y



class PreActBlock(nn.Module):
    '''Pre-activation version of the original basic module.'''

    def __init__(self, in_planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, bias=False)

    def forward(self, x):
        shortcut = x
        out = F.relu(self.bn1(x))

        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))

        out += shortcut
        #print(out.shape)
        return out

def create_network(num_channels=5):
    return PreActResNet(num_channels=num_channels)


def test():
    net = create_network()
    y = net((torch.randn(1, 1, 15, 15)))
    print(y.size())


if __name__ == '__main__':
    test()
