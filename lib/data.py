import torch
import torchvision
import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader

DATA_PATH = './data'
CATEGORY = ['action', 'horror', 'romance', 'ani2D', 'ani3D', 'sf']

class MovieDataset(Dataset):
    def __init__(self, type, size=None): # type : 'train' or 'test'
        self.img, self.label = load_data(type)
        self.size = size

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_file = self.img[idx]
        if(idx%2 == 1):
            img_file2 = self.img[idx-1]
        else:
            img_file2 = self.img[idx+1]

        print(img_file)
        print(img_file2)
        print()
        
        try:
            srcimg = cv2.imread(img_file)
            srcimg2 = cv2.imread(img_file2)
            if self.size != None:
                srcimg = cv2.resize(srcimg, self.size)
                srcimg2 = cv2.resize(srcimg2, self.size)
            assert(srcimg.shape[0] > 10)
            assert(srcimg2.shape[0] > 10)
            image = torch.from_numpy(np.vstack([srcimg.astype('float32').transpose(2,0,1) / 255,srcimg2.astype('float32').transpose(2,0,1) / 255]))
            label = torch.from_numpy(np.array(self.label[idx]))
            #item = {image: image, label: label}
        except:
            print('Error On : {0}'.format(img_file))
        return image, label


def load_data(type, d=1):
    train = []
    truth = []

    for i in range(len(CATEGORY)):
        label = CATEGORY[i]
        dirname = label + '_' + type
        path = os.path.join(DATA_PATH, dirname)
        file_list = os.listdir(path)
        for file in file_list:
            filename = os.path.join(path, file)
            train.append(filename)
            truth.append(i)
        print('{0} in {1} data'.format(len(file_list), label))

    print('*** {0} {1} Path are loaded'.format(len(train), type))
    assert(len(train) == len(truth))
    return train, truth

if __name__ == '__main__':
    load_data('train')
    load_data('test')
