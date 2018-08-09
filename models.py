import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, num_inputs, num_outputs, kernel_size=4, stride=1):
        assert num_outputs % 2 == 0
        self.conv1 = nn.Conv2d(num_inputs, num_outputs/2,
                               kernel_size=1,
                               stride=1)


        if stride != 1 or num_inputs != num_outputs:
            self.downsample = nn.Sequential(
                nn.Conv2d(num_inputs, num_outputs,
                          kernel_size=1, stride=1, bias=False),
            )
        else:
            self.downsample = None
