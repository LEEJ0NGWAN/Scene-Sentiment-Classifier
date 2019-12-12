import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import cv2
import numpy as np

class MotionNet(nn.Module):
    def __init__(self, classes, weight_path=None):
        super(MotionNet, self).__init__()
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
        fc_layers.append(nn.BatchNorm1d(4096))
        fc_layers.append(nn.ReLU())
        fc_layers.append(nn.Dropout(0.5))
        fc_layers.append(nn.Linear(4096, 4096))
        fc_layers.append(nn.BatchNorm1d(4096))
        fc_layers.append(nn.ReLU())
        fc_layers.append(nn.Dropout(0.5))
        fc_layers.append(nn.Linear(4096, len(classes)))
        fc_layers.append(nn.BatchNorm1d(len(classes)))
        fc_layers.append(nn.ReLU())
        fc_layers.append(nn.Dropout(0.5))
        fc_layers.append(nn.Softmax())
        self.fc_layers = fc_layers
        self.fc = nn.Sequential(*fc_layers)

        if weight_path == None:
            self._init_weights()
        else:
            self._load_weights(weight_path)


    def forward(self, x, t):
        batch_size = x.size(0)
        width = x.size(3)
        height = x.size(2)

        prev_tensor = x[:,0:3]
        next_tensor = x[:,3:6]
        prev_np = (prev_tensor.cpu().numpy() * 255).astype(np.uint8)
        next_np = (next_tensor.cpu().numpy() * 255).astype(np.uint8)
        for i in range(batch_size):
            prev_color = prev_np[i].transpose((1,2,0))
            next_color = next_np[i].transpose((1,2,0))
            print(prev_color)
            prev = cv2.cvtColor(prev_color, cv2.COLOR_RGB2GRAY)
            next = cv2.cvtColor(next_color, cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            channel = flow.reshape(height,width, 2)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            hsv = np.zeros_like(prev_color)
            hsv[...,0] = ang*180/np.pi/2
            hsv[...,1] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
            hsv[...,2] = 255
            rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
            cv2.imshow('prev_color',prev_color)
            cv2.imshow('next_color',next_color)
            cv2.imshow('visualized',rgb)
            cv2.waitKey(1000)

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
            normal_init(layer, 0, 0.05)
            layer.requires_grad_(True)
        for layer in self.fc_layers:
            normal_init(layer, 0, 0.05)
            layer.requires_grad_(True)
