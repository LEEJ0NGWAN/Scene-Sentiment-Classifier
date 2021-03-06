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

from lib.data_single import MovieDataset, CATEGORY
from lib.inception3 import Inception3
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    print('PyTorch 버전 : ' + torch.__version__)
    print('GPU 사용가능 여부 : ' + str(torch.cuda.is_available()))
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', dest='epoch', type=int, default=50, help='epoch')
    parser.add_argument('--lr', dest='lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--bs', dest='bs', type=int, default=32, help='batch size')
    args = parser.parse_args()
    epoch = args.epoch
    lr = args.lr
    bs = args.bs
    train_dataset = MovieDataset('train', (299,299))
    test_dataset = MovieDataset('test', (299,299))
    train_size = train_dataset.__len__()
    test_size = test_dataset.__len__()
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, num_workers=8, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, num_workers=4, shuffle=True)
    writer = SummaryWriter()

    print('\n***** 학습정보 *****')
    print('EPOCH : ', (epoch))
    print('LEARNING_RATE : ', (lr))
    print('********************\n')

    network = Inception3(len(CATEGORY))

    image_batch = torch.FloatTensor(1)
    label_batch = torch.LongTensor(1)

    network.cuda()
    image_batch = image_batch.cuda()
    label_batch = label_batch.cuda()
    image_batch = Variable(image_batch)
    label_batch = Variable(label_batch)

    optimizer = optim.Adam(network.parameters(), lr=lr)
    iter = 0
    for i in range(epoch):
        print('epoch : ',i)
        network.train()
        for idx, item in enumerate(train_dataloader):
            image_batch.resize_(item[0].size()).copy_(item[0])
            label_batch.resize_(item[1].size()).copy_(item[1])
            optimizer.zero_grad()
            y, aux = network(image_batch)
            #print(y.argmax(1))
            criterion = nn.CrossEntropyLoss()
            loss = criterion(y, label_batch)
            writer.add_scalar('Loss/Train', loss, iter)
            loss.backward()
            optimizer.step()

            iter = iter + 1

            if(iter % 100 == 0):
                for test_idx, test_item in enumerate(test_dataloader):
                    with torch.no_grad():
                        image_batch.resize_(test_item[0].size()).copy_(test_item[0])
                        label_batch.resize_(test_item[1].size()).copy_(test_item[1])
                        y, aux = network(image_batch)
                        loss = criterion(y, label_batch)
                        writer.add_scalar('Loss/Test', loss, iter)
                    break
        print('epoch {0} : {1}/{2} \t loss : {3}'. format(i, bs * idx, train_size, loss))

        if(i%5 == 4): torch.save(network, 'model_{0}.pkl'.format(i))
