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
from lib.flownet import Flownet

if __name__ == '__main__':
    print('PyTorch 버전 : ' + torch.__version__)
    print('GPU 사용가능 여부 : ' + str(torch.cuda.is_available()))
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    test_dataset = MovieDataset('test', (512,384))
    test_size = test_dataset.__len__()
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=8, shuffle=True)

    network = Flownet(CATEGORY)
    #network = torch.load('model_79.pkl') #50.7
    network = torch.load('model_59.pkl') #51.8
    network = network.eval()

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
        print(y)
        print(y.max())
        predict = torch.argmax(y, dim=1).detach().cpu().numpy()[0]
        truth = label_batch.detach().cpu().numpy()[0]
        is_correct = predict == truth
        cnt[truth] = cnt[truth] + 1
        pop[predict] = pop[predict] + 1
        if is_correct:
            acc[truth] = acc[truth] + 1
        visual = image_batch.detach().cpu().numpy()[0]
        print(CATEGORY)
        print(acc)
        print(cnt)
        print(pop)
        print(np.sum(acc) / np.sum(cnt))
        print()

        visual = visual.transpose(1,2,0)
        color = (255, 255, 255)
        if predict != truth:
            color = (0, 0, 255)
        print(CATEGORY[predict])
        print('{0} ({1})'.format(CATEGORY[predict], CATEGORY[truth]))
        visual = cv2.putText(visual, CATEGORY[predict], (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, color,4, cv2.LINE_AA)
        cv2.imshow('output', visual)
        if cv2.waitKey() == ord('q'): break
