import os
import sys, argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd import Variable
import cv2

from lib.data import MovieDataset, CATEGORY
from lib.vgg11 import VGG11

if __name__ == '__main__':
    print('PyTorch 버전 : ' + torch.__version__)
    print('GPU 사용가능 여부 : ' + str(torch.cuda.is_available()))
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    test_dataset = MovieDataset('test', (512,384))
    test_size = test_dataset.__len__()
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=4, shuffle=True)

    network = VGG11(CATEGORY)
    network = torch.load('model_19.pkl')

    image_batch = torch.FloatTensor(1)
    label_batch = torch.LongTensor(1)

    network.cuda()
    image_batch = image_batch.cuda()
    label_batch = label_batch.cuda()
    image_batch = Variable(image_batch)
    label_batch = Variable(label_batch)

    cnt = 0
    acc = np.zeros(len(CATEGORY))
    cnt = np.zeros(len(CATEGORY))
    pop = np.zeros(len(CATEGORY))
    for idx, item in enumerate(test_dataloader):
        image_batch.resize_(item[0].size()).copy_(item[0])
        label_batch.resize_(item[1].size()).copy_(item[1])
        y = network(x=image_batch, t=label_batch)
        predict = torch.argmax(y, dim=1).detach().cpu().numpy()[0]
        truth = label_batch.detach().cpu().numpy()[0]
        is_correct = predict == truth
        cnt[truth] = cnt[truth] + 1
        pop[predict] = pop[predict] + 1
        if is_correct:
            acc[truth] = acc[truth] + 1
        visual = image_batch.detach().cpu().numpy()[0]
        print(acc)
        print(cnt)
        print(pop)
        print()

        cv2.imshow('v', visual.transpose(1,2,0))
        print(CATEGORY[predict])
        cv2.waitKey()
