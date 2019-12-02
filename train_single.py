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

from lib.data import MovieDataset, CATEGORY
from lib.flownet import Flownet

if __name__ == '__main__':
    print('PyTorch 버전 : ' + torch.__version__)
    print('GPU 사용가능 여부 : ' + str(torch.cuda.is_available()))
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', dest='epoch', type=int, default=50, help='epoch')
    parser.add_argument('--lr', dest='lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--bs', dest='bs', type=int, default=16, help='batch size')
    args = parser.parse_args()
    epoch = args.epoch
    lr = args.lr
    bs = args.bs
    train_dataset = MovieDataset('train', (512,384))
    test_dataset = MovieDataset('test', (512,384))
    train_size = train_dataset.__len__()
    test_size = test_dataset.__len__()
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, num_workers=4, shuffle=True)
    iters_per_epoch = int(train_size / bs)

    print('\n***** 학습정보 *****')
    print('EPOCH : ', (epoch))
    print('LEARNING_RATE : ', (lr))
    print('********************\n')

    network = Flownet(CATEGORY)

    image_batch = torch.FloatTensor(1)
    label_batch = torch.LongTensor(1)

    network.cuda()
    image_batch = image_batch.cuda()
    label_batch = label_batch.cuda()
    image_batch = Variable(image_batch)
    label_batch = Variable(label_batch)

    optimizer = optim.SGD(network.parameters(), lr=lr)
    for i in range(epoch):
        print('epoch : ',i)
        network.train()
        for idx, item in enumerate(train_dataloader):
            image_batch.resize_(item[0].size()).copy_(item[0])
            label_batch.resize_(item[1].size()).copy_(item[1])
            optimizer.zero_grad()
            y = network(x=image_batch, t=label_batch)
            #print(y.argmax(1))
            criterion = nn.CrossEntropyLoss()
            loss = criterion(y, label_batch)
            loss.backward()
            optimizer.step()
            print('epoch {0} : {1}/{2} \t loss : {3}'. format(i, bs * idx, train_size, loss))
        if((i+1)%5 == 0): torch.save(network, 'model_{0}.pkl'.format(i))
