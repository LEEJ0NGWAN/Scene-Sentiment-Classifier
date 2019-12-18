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
from lib.moception3 import Moception
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    print('PyTorch 버전 : ' + torch.__version__)
    print('GPU 사용가능 여부 : ' + str(torch.cuda.is_available()))
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', dest='epoch', type=int, default=60, help='epoch')
    parser.add_argument('--lr', dest='lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--bs', dest='bs', type=int, default=32, help='batch size')
    args = parser.parse_args()
    epoch = args.epoch
    lr = args.lr
    bs = args.bs
    ckpt = 0
    train_dataset = MovieDataset('train', (299,299))
    test_dataset = MovieDataset('test', (299,299))
    train_size = train_dataset.__len__()
    test_size = test_dataset.__len__()

    #num_workers 확인 안하면 broken pipe error
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, num_workers=8, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, num_workers=8, shuffle=True)
    writer = SummaryWriter()

    print('\n***** 학습정보 *****')
    print('EPOCH : ', (epoch))
    print('LEARNING_RATE : ', (lr))
    print('********************\n')

    network = Moception(len(CATEGORY))
    if ckpt > 0:
        network = torch.load('model_{0}.pkl'.format(ckpt))

    image_batch1 = torch.FloatTensor(1)
    image_batch2 = torch.FloatTensor(1)
    motion_batch = torch.FloatTensor(1)
    label_batch = torch.LongTensor(1)

    network = network.cuda()
    image_batch1 = image_batch1.cuda()
    image_batch2 = image_batch2.cuda()
    motion_batch = motion_batch.cuda()
    label_batch = label_batch.cuda()

    image_batch1 = Variable(image_batch1)
    image_batch2 = Variable(image_batch2)
    motion_batch = Variable(motion_batch)
    label_batch = Variable(label_batch)

    optimizer = optim.Adam(network.parameters(), lr=lr)
    iter = 0
    for i in range(ckpt, epoch):
        print('epoch : ',i)
        network.train()
        for idx, item in enumerate(train_dataloader):
            item_size = item[0].size() # 16, 6, h, w
            _bs = item_size[0]
            cs = int(item_size[1]/2)
            h = item_size[2]
            w = item_size[3]
            image_batch1.resize_((_bs,cs,h,w)).copy_(item[0][:,0:3])
            image_batch2.resize_((_bs,cs,h,w)).copy_(item[0][:,3:6])
            #motion_batch.resize_((_bs,2,h,w))
            motion_batch.resize_((_bs, 1))
            label_batch.resize_(item[1].size()).copy_(item[1])
            optimizer.zero_grad()
            y, aux = network(x1=image_batch1, x2=image_batch2, motion_batch=motion_batch)

            criterion = nn.CrossEntropyLoss()
            loss = criterion(y, label_batch)
            writer.add_scalar('Loss/Train', loss, iter)
            loss.backward()
            optimizer.step()
            print('epoch {0} : {1}/{2} \t loss : {3}'. format(i, bs * idx, train_size, loss))
            iter = iter + 1
        #lr = lr * 0.97
        correct = 0
        num = 0
        print('Get ACC of Test ...')
        for test_idx, test_item in enumerate(test_dataloader):
            with torch.no_grad():
                _bs = test_item[0].size()[0]
                image_batch1.resize_((_bs,cs,h,w)).copy_(test_item[0][:,0:3])
                image_batch2.resize_((_bs,cs,h,w)).copy_(test_item[0][:,3:6])
                motion_batch.resize_((_bs,1))
                label_batch.resize_(test_item[1].size()).copy_(test_item[1])
                y, aux = network(x1=image_batch1, x2=image_batch2, motion_batch=motion_batch)
                y = y.argmax(axis=1)
                correct = correct + (y == label_batch).detach().cpu().numpy().sum()
                num = num + y.size()[0]
        print(correct/num)
        writer.add_scalar('Accuarcy/Test', correct / num, i)
        torch.save(network, 'model_sep_{0}.pkl'.format(i))
