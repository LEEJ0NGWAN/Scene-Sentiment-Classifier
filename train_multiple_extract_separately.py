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
from lib.motioncnn import MotionNet
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    print('PyTorch 버전 : ' + torch.__version__)
    print('GPU 사용가능 여부 : ' + str(torch.cuda.is_available()))
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', dest='epoch', type=int, default=60, help='epoch')
    parser.add_argument('--lr', dest='lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--bs', dest='bs', type=int, default=16, help='batch size')
    args = parser.parse_args()
    epoch = args.epoch
    lr = args.lr
    bs = args.bs
    ckpt = 0
    train_dataset = MovieDataset('train', (512,384))
    test_dataset = MovieDataset('test', (512,384))
    train_size = train_dataset.__len__()
    test_size = test_dataset.__len__()
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, num_workers=8, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, num_workers=4, shuffle=True)
    writer = SummaryWriter()

    print('\n***** 학습정보 *****')
    print('EPOCH : ', (epoch))
    print('LEARNING_RATE : ', (lr))
    print('********************\n')

    network = MotionNet(CATEGORY)
    if ckpt > 0:
        network = torch.load('model_{0}.pkl'.format(ckpt))

    image_batch = torch.FloatTensor(1)
    label_batch = torch.LongTensor(1)

    network.cuda()
    image_batch = image_batch.cuda()
    label_batch = label_batch.cuda()
    image_batch = Variable(image_batch)
    label_batch = Variable(label_batch)

    optimizer = optim.SGD(network.parameters(), lr=lr)
    iter = 0
    for i in range(ckpt, epoch):
        print('epoch : ',i)
        network.train()
        for idx, item in enumerate(train_dataloader):
            print(item[0].size()) # 16, 6, 384, 512
            image_batch.resize_(item[0].size()).copy_(item[0])
            label_batch.resize_(item[1].size()).copy_(item[1])
            optimizer.zero_grad()
            y = network(x=image_batch, t=label_batch)
            print(y.argmax(1))
            criterion = nn.CrossEntropyLoss()
            loss = criterion(y, label_batch)
            writer.add_scalar('Loss/Train', loss, iter)
            loss.backward()
            optimizer.step()
            print('epoch {0} : {1}/{2} \t loss : {3}'. format(i, bs * idx, train_size, loss))
            iter = iter + 1

            if(iter % 100 == 0):
                for test_idx, test_item in enumerate(test_dataloader):
                    with torch.no_grad():
                        image_batch.resize_(test_item[0].size()).copy_(test_item[0])
                        label_batch.resize_(test_item[1].size()).copy_(test_item[1])
                        y = network(x=image_batch, t=label_batch)
                        loss = criterion(y, label_batch)
                        writer.add_scalar('Loss/Test', loss, iter)
                    break
        #lr = lr * 0.95
        if((i+1)%4 == 0): torch.save(network, 'model_{0}.pkl'.format(i))
