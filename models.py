import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, num_inputs, num_outputs, kernel_size=3, stride=1):
        super(ResBlock, self).__init__()
        assert num_outputs % 2 == 0
        assert kernel_size % 2 == 1
        self.conv1 = nn.Conv2d(num_inputs, num_outputs/2,
                               kernel_size=1,
                               stride=1,
                               padding=0)
        self.bn1 = nn.BatchNorm2d(num_outputs/2)
        self.conv2 = nn.Conv2d(num_outputs/2, num_outputs/2,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=(kernel_size-1)//2)
        self.bn2 = nn.BatchNorm2d(num_outputs/2)
        self.conv3 = nn.Conv2d(num_outputs/2, num_outputs,
                               kernel_size=1,
                               stride=1,
                               padding=0)
        self.bn3 = nn.BatchNorm2d(num_outputs)


        if stride != 1 or num_inputs != num_outputs:
            self.conv_shortcut = nn.Conv2d(num_inputs, num_outputs,
                                           kernel_size=1,
                                           stride=stride,
                                           padding=0)
        else:
            self.conv_shortcut = None

    def forward(self, x):
        residual = x
        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = F.relu(out1, True)

        out2 = self.conv2(out1)
        out2 = self.bn2(out2)
        out2 = F.relu(out2, True)

        out3 = self.conv3(out2)

        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)

        out3 += residual
        out3 = self.bn3(out3)
        out3 = F.relu(out3, True)

        return out3

class ResFCN256(nn.Module):
    def __init__(self, input_res, output_res, channel=3, base_size=16):
        super(ResFCN256, self).__init__()
        self.input_res = input_res
        self.output_res = output_res
        self.channel = channel
        self.base_size = base_size
        self.conv1 = nn.Conv2d(self.channel, self.base_size,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.bn1 = nn.BatchNorm2d(self.base_size)

        self.block1 = ResBlock(self.base_size, self.base_size * 2,
                               kernel_size=3,
                               stride=2)
        self.block2 = ResBlock(self.base_size * 2, self.base_size * 2,
                               kernel_size=3,
                               stride=1)

        self.block3 = ResBlock(self.base_size * 2, self.base_size * 4,
                               kernel_size=3,
                               stride=2)
        self.block4 = ResBlock(self.base_size * 4, self.base_size * 4,
                               kernel_size=3,
                               stride=1)

        self.block5 = ResBlock(self.base_size * 4, self.base_size * 8,
                               kernel_size=3,
                               stride=2)
        self.block6 = ResBlock(self.base_size * 8, self.base_size * 8,
                               kernel_size=3,
                               stride=1)

        self.block7 = ResBlock(self.base_size * 8, self.base_size * 16,
                               kernel_size=3,
                               stride=2)
        self.block8 = ResBlock(self.base_size * 16, self.base_size * 16,
                               kernel_size=3,
                               stride=1)

        self.block9 = ResBlock(self.base_size * 16, self.base_size * 32,
                               kernel_size=3,
                               stride=2)
        self.block10 = ResBlock(self.base_size * 32, self.base_size * 32,
                               kernel_size=3,
                               stride=1)


        self.deconv16 = nn.ConvTranspose2d(self.base_size * 32,
                                           self.base_size * 32,
                                           kernel_size=3,
                                           stride=1,
                                           padding=1)

        self.deconv15 = nn.ConvTranspose2d(self.base_size * 32,
                                           self.base_size * 16,
                                           kernel_size=4,
                                           stride=2,
                                           padding=1)
        self.deconv14 = nn.ConvTranspose2d(self.base_size * 16,
                                           self.base_size * 16,
                                           kernel_size=3,
                                           stride=1,
                                           padding=1)
        self.deconv13 = nn.ConvTranspose2d(self.base_size * 16,
                                           self.base_size * 16,
                                           kernel_size=3,
                                           stride=1,
                                           padding=1)

        self.deconv15 = nn.ConvTranspose2d(self.base_size * 16,
                                           self.base_size * 8,
                                           kernel_size=4,
                                           stride=2,
                                           padding=1)
        self.deconv14 = nn.ConvTranspose2d(self.base_size * 8,
                                           self.base_size * 8,
                                           kernel_size=3,
                                           stride=1,
                                           padding=1)
        self.deconv13 = nn.ConvTranspose2d(self.base_size * 8,
                                           self.base_size * 8,
                                           kernel_size=3,
                                           stride=1,
                                           padding=1)

        self.deconv15 = nn.ConvTranspose2d(self.base_size * 8,
                                           self.base_size * 4,
                                           kernel_size=4,
                                           stride=2,
                                           padding=1)
        self.deconv14 = nn.ConvTranspose2d(self.base_size * 4,
                                           self.base_size * 4,
                                           kernel_size=3,
                                           stride=1,
                                           padding=1)
        self.deconv13 = nn.ConvTranspose2d(self.base_size * 4,
                                           self.base_size * 4,
                                           kernel_size=3,
                                           stride=1,
                                           padding=1)

        self.deconv15 = nn.ConvTranspose2d(self.base_size * 8,
                                           self.base_size * 4,
                                           kernel_size=4,
                                           stride=2,
                                           padding=1)
        self.deconv14 = nn.ConvTranspose2d(self.base_size * 4,
                                           self.base_size * 4,
                                           kernel_size=3,
                                           stride=1,
                                           padding=1)
        self.deconv13 = nn.ConvTranspose2d(self.base_size * 4,
                                           self.base_size * 4,
                                           kernel_size=3,
                                           stride=1,
                                           padding=1)
