import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Flownet(nn.Module):
    def __init__(self, classes, weight_path=None):
        super(Flownet, self).__init__()
        self.classes = classes

        conv_layers = []
        conv_layers.append(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=(3,3)))
        conv_layers.append(nn.ReLU())
        conv_layers.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=(2,2)))
        conv_layers.append(nn.ReLU())
        conv_layers.append(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=(2,2)))
        conv_layers.append(nn.ReLU())
        conv_layers.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=(1,1)))
        conv_layers.append(nn.ReLU())
        conv_layers.append(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=(1,1)))
        conv_layers.append(nn.ReLU())
        conv_layers.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=(1,1)))
        conv_layers.append(nn.ReLU())
        conv_layers.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=(1,1)))
        conv_layers.append(nn.ReLU())
        conv_layers.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=(1,1)))
        conv_layers.append(nn.ReLU())
        conv_layers.append(nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=(1,1)))
        conv_layers.append(nn.ReLU())
        self.conv_layers = conv_layers
        self.conv = nn.Sequential(*conv_layers)

        fc_layers = []
        fc_layers.append(nn.Linear(1024*6*8, 4096))
        fc_layers.append(nn.Dropout(0.5))
        fc_layers.append(nn.BatchNorm1d(4096))
        fc_layers.append(nn.ReLU())
        fc_layers.append(nn.Linear(4096, 4096))
        fc_layers.append(nn.Dropout(0.5))
        fc_layers.append(nn.BatchNorm1d(4096))
        fc_layers.append(nn.ReLU())
        fc_layers.append(nn.Linear(4096, len(classes)))
        fc_layers.append(nn.Dropout(0.5))
        fc_layers.append(nn.BatchNorm1d(len(classes)))
        fc_layers.append(nn.ReLU())
        fc_layers.append(nn.Softmax())
        self.fc_layers = fc_layers
        self.fc = nn.Sequential(*fc_layers)

        if weight_path == None:
            self._init_weights()
        else:
            self._load_weights(weight_path)


    def forward(self, x, t):
        batch_size = x.size(0)
        conv_y = self.conv(x)
        flatten_y = conv_y.view(batch_size, -1)
        y = self.fc(flatten_y)
        return y

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            if(type(m) == nn.ReLU): return
            if(type(m) == nn.Dropout): return
            if(type(m) == nn.Softmax): return
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()
        for layer in self.conv_layers:
            normal_init(layer, 0, 0.1)
            layer.requires_grad_(True)
        for layer in self.fc_layers:
            normal_init(layer, 0, 0.1)
            layer.requires_grad_(True)
