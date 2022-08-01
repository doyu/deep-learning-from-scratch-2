# coding: utf-8
import sys
sys.path.append('..')  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from common.layers import Affine, Sigmoid, SoftmaxWithLoss
from common.functions import softmax, sigmoid, cross_entropy_error

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

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size

        # 重みとバイアスの初期化
        W1 = 0.01 * np.random.randn(I, H)
        b1 = np.zeros(H)
        W2 = 0.01 * np.random.randn(H, O)
        b2 = np.zeros(O)

        # レイヤの生成
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]
        self.loss_layer = SoftmaxWithLoss()

        # すべての重みと勾配をリストにまとめる
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads
          
    def predict(self, x):
#        for layer in self.layers:
#            x = layer.forward(x)
        W1, b1 = self.layers[0].params
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        W2, b2 = self.layers[2].params
        a2 = np.dot(z1, W2) + b2
        return a2

    def forward(self, x, t):
        y = self.predict(x)
        y = softmax(y)
        if t.size == y.size:
            t = t.argmax(axis=1)
        return cross_entropy_error(y, t)

    def backward(self, x, t, dout=1):
#        dout = self.loss_layer.backward(dout)
#        for layer in reversed(self.layers):
#            dout = layer.backward(dout)
  
        for i, val in enumerate(self.params):
            self.grads[i] = numerical_gradient(lambda _Wb: self.forward(x, t), val)
        
        return dout
