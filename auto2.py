import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data
import torch.nn as nn
#import snntorch
import pandas as pd#
#import tqdm
import argparse
from torch import nn, optim
from snn_model import network
from snn_model import p_snu_layer

##実データを自動でスパイク列に変換するモジュールを作成##(eval用はまだやってない)
df = pd.read_csv('iris.data', header=None)
y = df.iloc[0:100,4].values 
y = np.where(y=='Iris-setosa',0, 1)
X = df.iloc[0:100,[0,1,2,3]].values 
X_train = np.empty((80, 4)) 
X_test = np.empty((20, 4))
y_train = np.empty(80)
y_test = np.empty(20)
X_train[:40],X_train[40:] = X[:40],X[50:90]
X_test[:10],X_test[10:] = X[40:50],X[90:100]
y_train[:40],y_train[40:] = y[:40],y[50:90]
y_test[:10],y_test[10:] = y[40:50],y[90:100]

#学習用と教師データをまとめる(とりあえず学習させたいのでevalは無視)

#rate-encoding
def Spike_train_converter(inputs, dt, time, time_step, fr = 30, norm = 10):
    freq_temp = fr*norm/np.sum(inputs[time])
    freq = freq_temp*np.repeat(np.expand_dims(inputs[time],axis=0), time_step, axis= 0)
    spike_train = np.where(np.random.rand(time_step, 4) < freq*dt ,1 ,0)
    return spike_train

dt = 1e-3
time_step = 100
data = 100
#全部のデータ見る
state_converted = np.empty((80,100,4))
label = np.empty((80))
#今は学習用しかやってないけど、評価用データもスパイク信号に置き換えて(edited 6/18)
for time in range(len(X_train)):
    state_converted[time] = Spike_train_converter(inputs = X_train, dt = dt, time = time, time_step = time_step)
    label[time] = y_train[time]

#tupleでlabelと合成した方が汎用性が高いが、torch.utils.data.Tensorの中身変えてもだめだったので個別で管理する。
state_converted = torch.tensor(state_converted)
label = torch.tensor(label)
print(state_converted.shape) #torch.Size([80, 100, 4])
print(label[2])  #tensor(0., dtype=torch.float64)
print('###success making spike dataset and label!###')

'''
how to make Spiking converted data
(1)make 3 dimention empty ndarray
(2)decide time_step parameter(this will be the total time of poisson encoding for each data), give more requirments to Spike_train_converter
(3)when finished converting for train_data with size of (4, 100)(but this should be torch), concatenate label for each data by tuple
(4)repeat(1)-(3) for len(train) cycle (example shows 80)  
'''



