#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('..')  # 親ディレクトリのファイルをインポートするための設定
from dataset import spiral
from common.functions import softmax, sigmoid, cross_entropy_error

# Data preparation
x, t = spiral.load_data()
print(list(map(lambda x: np.array(x).shape, [x, t])))
plt.scatter(x[:,0], x[:,1], c=np.argmax(t, axis=1))


# Build up NN

# Hyper parameters
batch_size = 30
hidden_size = 10
learning_rate = 1.0

# Initialize weights and bias
W1 = 0.01 * np.random.randn(len(x[0]), hidden_size)
b1 = np.zeros(1)
W2 = 0.01 * np.random.randn(hidden_size, len(t[0]))
b2 = np.zeros(1)

def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 値を元に戻す
        it.iternext()   
        
    return grad

def predict(x):
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    return a2

def loss(x, t):
    z = predict(x)
    y = softmax(z)
    return cross_entropy_error(y, t)

def update(x, t):
    params = [W1, b1, W2, b2]
    f = lambda p: loss(x, t)
    grads = [numerical_gradient(f, param) for param in params]
    for p, g in zip(params, grads):
        p -= learning_rate * g


#%%time
# Train NN

loss_list = []
for epoch in range(300):
    # データのシャッフル
    idx = np.random.permutation(len(x))
    x, t = x[idx], t[idx]

    max = len(x) // batch_size
    for i in range(max):
        idx = np.arange(i*batch_size, (i+1)*batch_size) 
        xx, tt = x[idx], t[idx]

        _loss = loss(xx, tt)
        loss_list.append(_loss)
        
        update(xx, tt)

        
# 学習結果のプロット
print(f"loss={_loss}")          
plt.plot(np.arange(len(loss_list)), loss_list, label='train')
plt.xlabel('iterations')
plt.ylabel('loss')
plt.show()

# 境界領域のプロット
h = 0.001
x_min, x_max = x[:, 0].min() - .1, x[:, 0].max() + .1
y_min, y_max = x[:, 1].min() - .1, x[:, 1].max() + .1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
X = np.c_[xx.ravel(), yy.ravel()]
score = predict(X)
predict_cls = np.argmax(score, axis=1)
Z = predict_cls.reshape(xx.shape)
plt.contourf(xx, yy, Z)
plt.axis('off')

# データ点のプロット
x, t = spiral.load_data()
N = 100
CLS_NUM = 3
markers = ['o', 'x', '^']
for i in range(CLS_NUM):
    plt.scatter(x[i*N:(i+1)*N, 0], x[i*N:(i+1)*N, 1], s=40, marker=markers[i])
plt.show()

