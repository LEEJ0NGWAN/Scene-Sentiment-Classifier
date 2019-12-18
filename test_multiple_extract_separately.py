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
from lib.moception3 import Moception

if __name__ == '__main__':
    print('PyTorch 버전 : ' + torch.__version__)
    print('GPU 사용가능 여부 : ' + str(torch.cuda.is_available()))
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    test_dataset = MovieDataset('test', (256,192))
    test_size = test_dataset.__len__()
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=8, shuffle=True)

    network = Moception(len(CATEGORY))
    network = torch.load('model_sep_39.pkl')
    network = network.eval()

    image_batch1 = torch.FloatTensor(1)
    image_batch2 = torch.FloatTensor(1)
    label_batch = torch.LongTensor(1)
    motion_batch = torch.FloatTensor(1)

    network.cuda()
    image_batch1 = image_batch1.cuda()
    image_batch2 = image_batch2.cuda()
    motion_batch = motion_batch.cuda()
    label_batch = label_batch.cuda()

    image_batch1 = Variable(image_batch1)
    image_batch2 = Variable(image_batch2)
    motion_batch = Variable(motion_batch)
    label_batch = Variable(label_batch)

    cnt = 0
    acc = np.zeros(len(CATEGORY))
    cnt = np.zeros(len(CATEGORY))
    pop = np.zeros(len(CATEGORY))
    for idx, item in enumerate(test_dataloader):
        item_size = item[0].size() # 16, 6, h, w
        _bs = item_size[0]
        cs = int(item_size[1]/2)
        h = item_size[2]
        w = item_size[3]
        image_batch1.resize_((_bs,cs,h,w)).copy_(item[0][:,0:3])
        image_batch2.resize_((_bs,cs,h,w)).copy_(item[0][:,3:6])
        motion_batch.resize_((_bs,1))
        label_batch.resize_(item[1].size()).copy_(item[1])
        y = network(x1=image_batch1, x2=image_batch2, motion_batch=motion_batch)
        print(y)
        print(y.max())
        predict = torch.argmax(y, dim=1).detach().cpu().numpy()[0]
        truth = label_batch.detach().cpu().numpy()[0]
        is_correct = predict == truth
        cnt[truth] = cnt[truth] + 1
        pop[predict] = pop[predict] + 1
        if is_correct:
            acc[truth] = acc[truth] + 1
        visual = image_batch1.detach().cpu().numpy()[0]
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
        if cv2.waitKey(1) == ord('q'): break
