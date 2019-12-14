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

        scene_layers = []
        scene_layers.append(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=(3,3)))
        scene_layers.append(nn.ReLU())
        scene_layers.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=(2,2)))
        scene_layers.append(nn.ReLU())
        scene_layers.append(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=(2,2)))
        scene_layers.append(nn.ReLU())
        scene_layers.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=(1,1)))
        scene_layers.append(nn.ReLU())
        scene_layers.append(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=(1,1)))
        scene_layers.append(nn.ReLU())
        scene_layers.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=(1,1)))
        scene_layers.append(nn.ReLU())
        scene_layers.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=(1,1)))
        scene_layers.append(nn.ReLU())
        scene_layers.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=(1,1)))
        scene_layers.append(nn.ReLU())
        scene_layers.append(nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=(1,1)))
        scene_layers.append(nn.ReLU())
        self.scene_layers = scene_layers
        self.scene = nn.Sequential(*scene_layers)

        fc1_layers = []
        fc1_layers.append(nn.Linear(1024*6*8, 4096))
        fc1_layers.append(nn.BatchNorm1d(4096))
        fc1_layers.append(nn.ReLU())
        fc1_layers.append(nn.Dropout(0.5))
        fc1_layers.append(nn.Linear(4096, 4096))
        fc1_layers.append(nn.BatchNorm1d(4096))
        fc1_layers.append(nn.ReLU())
        fc1_layers.append(nn.Dropout(0.5))
        fc1_layers.append(nn.Linear(4096, 1024))
        self.fc1_layers = fc1_layers
        self.fc1 = nn.Sequential(*fc1_layers)

        motion_layers = []
        motion_layers.append(nn.Conv2d(in_channels=2, out_channels=64, kernel_size=7, stride=2, padding=(3,3)))
        motion_layers.append(nn.ReLU())
        motion_layers.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=(2,2)))
        motion_layers.append(nn.ReLU())
        motion_layers.append(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=(1,1)))
        motion_layers.append(nn.ReLU())
        motion_layers.append(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=(1,1)))
        motion_layers.append(nn.ReLU())
        motion_layers.append(nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=(1,1)))
        motion_layers.append(nn.ReLU())
        motion_layers.append(nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=(1,1)))
        motion_layers.append(nn.ReLU())
        self.motion_layers = motion_layers
        self.motion = nn.Sequential(*motion_layers)

        fc2_layers = []
        fc2_layers.append(nn.Linear(1024*6*8, 1024))
        fc2_layers.append(nn.BatchNorm1d(1024))
        fc2_layers.append(nn.ReLU())
        fc2_layers.append(nn.Dropout(0.5))
        fc2_layers.append(nn.Linear(1024, 32))
        fc2_layers.append(nn.BatchNorm1d(32))
        fc2_layers.append(nn.ReLU())
        fc2_layers.append(nn.Dropout(0.5))
        self.fc2_layers = fc2_layers
        self.fc2 = nn.Sequential(*fc2_layers)

        fc_layers=[]
        fc_layers.append(nn.Linear(1056, 1056))
        fc_layers.append(nn.BatchNorm1d(1056))
        fc_layers.append(nn.ReLU())
        fc_layers.append(nn.Linear(1056, 1056))
        fc_layers.append(nn.BatchNorm1d(1056))
        fc_layers.append(nn.ReLU())
        fc_layers.append(nn.Dropout(0.5))
        fc_layers.append(nn.Linear(1056, len(classes)))
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


    def forward(self, x1, x2, motion_batch, t):
        batch_size = x1.size(0)
        width = x1.size(3)
        height = x1.size(2)

        prev_np = (x1.cpu().numpy() * 255).astype(np.uint8)
        next_np = (x2.cpu().numpy() * 255).astype(np.uint8)

        flow_arr = np.zeros((batch_size, 2, height, width))
        for i in range(batch_size):
            prev_color = prev_np[i].transpose((1,2,0))
            next_color = next_np[i].transpose((1,2,0))
            prev = cv2.cvtColor(prev_color, cv2.COLOR_RGB2GRAY)
            next = cv2.cvtColor(next_color, cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            channel = flow.reshape((height, width, 2))
            channel = channel.transpose(2,0,1)
            flow_arr[i] = channel
            '''
            # Visualizing Training
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            hsv = np.zeros_like(prev_color)
            hsv[...,0] = ang*180/np.pi/2
            hsv[...,1] = mag * 10  #cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
            hsv[...,2] = 255
            rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
            cv2.imshow('prev_color',prev_color)
            cv2.imshow('next_color',next_color)
            cv2.imshow('visualized',rgb)
            cv2.waitKey()
            '''

        conv_y1 = self.scene(x1)
        flatten_y1 = conv_y1.view(batch_size, -1)
        y1 = self.fc1(flatten_y1)

        motion_batch.copy_(torch.from_numpy(flow_arr))
        conv_y2 = self.motion(motion_batch)
        flatten_y2 = conv_y2.view(batch_size, -1)
        y2 = self.fc2(flatten_y2)

        y = self.fc(torch.cat([y1, y2], 1))
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
        for layer in self.scene_layers:
            normal_init(layer, 0, 0.05)
            layer.requires_grad_(True)
        for layer in self.motion_layers:
            normal_init(layer, 0, 0.05)
            layer.requires_grad_(True)
        for layer in self.fc1_layers:
            normal_init(layer, 0, 0.05)
            layer.requires_grad_(True)
        for layer in self.fc2_layers:
            normal_init(layer, 0, 0.05)
            layer.requires_grad_(True)
        for layer in self.fc_layers:
            normal_init(layer, 0, 0.05)
            layer.requires_grad_(True)
