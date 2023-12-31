'''LeNet in PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1)
        # self.fc1   = nn.Linear(13024, 120)
        self.fc2   = nn.Linear(71280, 84)
        # self.fc2   = nn.Linear(120, 168)
        # self.fc3.  = nn.Linear(168, 30)
        self.fc3   = nn.Linear(84, 30)

    def forward(self, x):
        outs = {}
        outs[0] = x
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv3(out))

        #out = out.view(out.size(0), -1)
        out = torch.flatten(out,1)
        outs[1] = out
        #out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        outs[2] = out
        out = self.fc3(out)
        outs['out'] = out
        return outs