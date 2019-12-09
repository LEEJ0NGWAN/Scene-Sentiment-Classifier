#-*- coding:utf-8 -*-
import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import PIL.Image as pilimg
import pickle

def sigmoid(x):
    return 1/(1+np.exp(-x))
def relu(x):
    return np.maximum(0,x)
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))
def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)
def relu_grad(x):
    grad = np.zeros(x.shape)
    grad[x>=0] = 1
    return grad
def cross_entropy_error(y,t):
    if y.ndim == 1:
        t = t.reshape(1,t.size)
        y = y.reshape(1,y.size)
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size
def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val
        it.iternext()

    return grad


class LayerNet:
    def __init__(self, input_size, hidden_size_1,hidden_size_2,hidden_size_3,isSigmoid, output_size, weight_init_std=0.01):
        self.isSigmoid = isSigmoid
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size_1)
        self.params['b1'] = np.zeros(hidden_size_1)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size_1, hidden_size_2)
        self.params['b2'] = np.zeros(hidden_size_2)
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size_2, hidden_size_3)
        self.params['b3'] = np.zeros(hidden_size_3)
        self.params['W4'] = weight_init_std * np.random.randn(hidden_size_3, output_size)
        self.params['b4'] = np.zeros(output_size)
    def predict(self,x):
        W1, W2, W3, W4 = self.params['W1'], self.params['W2'], self.params['W3'], self.params['W4']
        b1, b2, b3, b4 = self.params['b1'], self.params['b2'], self.params['b3'], self.params['b4']
        a1 = np.dot(x,W1)+b1
        if self.isSigmoid:
            z1 = sigmoid(a1)
            a2 = np.dot(z1, W2) + b2
            z2 = sigmoid(a2)
            a3 = np.dot(z2, W3) + b3
            z3 = sigmoid(a3)
        else:
            z1 = relu(a1)
            a2 = np.dot(z1,W2)+b2
            z2 = relu(a2)
            a3 = np.dot(z2,W3)+b3
            z3 = relu(a3)
        a4 = np.dot(z3,W4)+b4
        y = softmax(a4)
        return y
    def loss(self,x,t):
        y = self.predict(x)
        return cross_entropy_error(y,t)
    def accuracy(self,x,t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        return np.sum(y == t) / float(x.shape[0])
    def numerical_gradient(self,x,t):
        loss_W = lambda W: self.loss(x,t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        grads['W3'] = numerical_gradient(loss_W, self.params['W3'])
        grads['b3'] = numerical_gradient(loss_W, self.params['b3'])
        grads['W4'] = numerical_gradient(loss_W, self.params['W4'])
        grads['b4'] = numerical_gradient(loss_W, self.params['b4'])
        return grads

    def gradient(self, x, t):
        W1, W2, W3, W4 = self.params['W1'], self.params['W2'], self.params['W3'], self.params['W4']
        b1, b2, b3, b4 = self.params['b1'], self.params['b2'], self.params['b3'], self.params['b4']

        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x,W1)+b1
        if self.isSigmoid:
            z1 = sigmoid(a1)
            a2 = np.dot(z1, W2) + b2
            z2 = sigmoid(a2)
            a3 = np.dot(z2, W3) + b3
            z3 = sigmoid(a3)
        else:
            z1 = relu(a1)
            a2 = np.dot(z1,W2)+b2
            z2 = relu(a2)
            a3 = np.dot(z2,W3)+b3
            z3 = relu(a3)
        a4 = np.dot(z3,W4)+b4
        y = softmax(a4)

        # backward
        dy = (y - t) / batch_num
        grads['W4'] = np.dot(z3.T, dy)
        grads['b4'] = np.sum(dy, axis=0)
        dz3 = np.dot(dy, W4.T)
        if self.isSigmoid:
            da3 = sigmoid_grad(a3) * dz3
        else:
            da3 = relu_grad(a3) * dz3
        grads['W3'] = np.dot(z2.T, da3)
        grads['b3'] = np.sum(da3, axis=0)
        dz2 = np.dot(dz3, W3.T)
        if self.isSigmoid:
            da2 = sigmoid_grad(a2) * dz2
        else:
            da2 = relu_grad(a2) * dz2
        grads['W2'] = np.dot(z1.T, da2)
        grads['b2'] = np.sum(da2, axis=0)
        dz1 = np.dot(dz2, W2.T)
        if self.isSigmoid:
            da1 = sigmoid_grad(a1) * dz1
        else:
            da1 = relu_grad(a1) * dz1
        grads['W1'] = np.dot(x.T, da1)
        grads['b1'] = np.sum(da1, axis=0)

        return grads
# ==================================dataset ==============================

originPath = os.path.abspath('..')
print('originpath = ' + originPath)
DATA_PATH = '../data'
CATEGORY = ['action', 'horror', 'romance', 'ani2D', 'ani3D', 'sf']

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
            fimg = pilimg.open(filename).resize((120,68))
            img = np.ravel(np.array(fimg),order='C')
            train.append(img)
            label = [0,0,0,0,0,0]
            label[i] = 1
            truth.append(label)
        print('{0} in {1} data'.format(len(file_list), label))

    print('*** {0} {1} Path are loaded'.format(len(train), type))
    assert(len(train) == len(truth))
    return np.array(train), np.array(truth)

if __name__ == '__main__':

    dataset = {}
    save_file = os.path.join(DATA_PATH, 'dataset.pkl')
    if not os.path.exists(save_file):
        (dataset['x_train'], dataset['t_train']) = load_data('train')
        (dataset['x_test'], dataset['t_test']) = load_data('test')
        # with open(save_file, 'wb') as f:
        #     pickle.dump(dataset, f, -1)
        print("Done!")
    else:
        with open(save_file, 'rb') as f:
            dataset = pickle.load(f)


    x_train = dataset['x_train']
    t_train = dataset['t_train']
    x_test = dataset['x_test']
    t_test = dataset['t_test']

    iters_end_loss = 0.01
    train_size = x_train.shape[0]
    test_size = x_test.shape[0]
    batch_size = 32
    learning_rate = 0.001
    network = LayerNet(input_size=24480, hidden_size_1=50,hidden_size_2=10,hidden_size_3=40,isSigmoid=False, output_size=6)
    train_loss_list = []
    test_loss_list = []
    train_acc_list = []
    test_acc_list = []
    iter_per_epoch = max(int(train_size / batch_size), 1)
    train_loss = 1000000
    i=0
    iters_num = 10000
    for i in range(iters_num):
        #미니배치
        batch_mask = np.random.choice(train_size, batch_size)
        tbatch_mask = np.random.choice(test_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        xt_batch = x_test[tbatch_mask]
        tt_batch = t_test[tbatch_mask]
        #기울기 계산
        #grad = network.numerical_gradient(x_batch, t_batch) 너무 오래 걸림
        grad = network.gradient(x_batch, t_batch)

        #갱신
        for key in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3', 'W4', 'b4'):
            network.params[key] -= learning_rate * grad[key]
        train_loss = network.loss(x_batch, t_batch)
        test_loss = network.loss(xt_batch, tt_batch)
        print("ep " + str(i) + " - train loss, test loss : " + str(train_loss) + " , " + str(test_loss))
        #학습 경과 기록
        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            train_loss = network.loss(x_batch, t_batch)
            train_loss_list.append(train_loss)
            test_loss = network.loss(xt_batch, tt_batch)
            test_loss_list.append(test_loss)

            print("ep "+str(i)+" - train loss, test loss : "+str(train_loss)+" , "+str(test_loss))
            print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

    acc = train_acc_list
    val_acc = test_acc_list
    loss = train_loss_list
    val_loss = test_loss_list

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.clf()

    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()
    print(max(val_acc))