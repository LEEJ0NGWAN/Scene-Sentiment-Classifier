import torch
import torch.nn as nn
import torch.nn.functional as F

class Flownet(nn.Module):
    def __init__(self, classes, weight_path=None):
        super(Flownet, self).__init__()
        self.classes = classes

        conv_layers = []
        conv_layers.append(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=(3,3)))
        conv_layers.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=(2,2)))
        conv_layers.append(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=(2,2)))
        conv_layers.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=(1,1)))
        conv_layers.append(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=(1,1)))
        conv_layers.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=(1,1)))
        conv_layers.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=(1,1)))
        conv_layers.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=(1,1)))
        conv_layers.append(nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=(1,1)))

        self.conv = nn.Sequential(*conv_layers)
        self.fc = nn.Sequential(
            nn.Linear(1024*6*8, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1000),
            nn.ReLU(),
            nn.Linear(1000, len(classes)),
            nn.ReLU(),
            nn.Softmax(),
        )

        if weight_path == None:
            self.init_weight()
        else:
            self.load_weight(weight_path)


    def forward(self, x, t):
        batch_size = x.size(0)
        conv_y = self.conv(x)
        flatten_y = conv_y.view(batch_size, -1)
        y = self.fc(flatten_y)
        return y

    def init_weight(self):
        pass

    def load_weight(self):
        pass
