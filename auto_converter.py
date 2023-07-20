import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import snntorch
import pandas as pd
import tqdm
import argparse
from torch import nn, optim
from snn_model import network
from snn_model import p_snu_layer
#Iris have three output classes, but two are given


##実験(失敗作)##
df = pd.read_csv('iris.data', header=None)
x = df.iloc[0:100,[0,1,2,3]].values
y = df.iloc[0:100,4].values
y = y.reshape(100,1)
y = np.where(y == 'Iris-setosa', 0, 1) 
train_x_data = np.empty((80,4))
train_y_data = np.empty((80,1))
#とりあえず訓練データだけ
train_x_data[:40],train_x_data[40:] = x[:40],x[50:90]
train_y_data[:40],train_y_data[40:] = y[:40],y[50:90]
train_x_data = train_x_data.T
train_y_data = train_y_data.T
Data = (train_x_data,train_y_data)
print(train_x_data.shape)
print(Data[0][:,0])#IRIS取り出し
print(Data[1])

# def Spike_train_converter(inputs, dt, time_step, fr = 30, norm = 10):
#     freq_temp = fr*norm/np.sum(inputs[time])
#     freq = freq_temp*np.repeat(np.expand_dims(inputs[time],axis=0), time_step, axis= 0)
#     spike_train = np.where(np.random.rand(time_step, 4) < freq*dt ,1 ,0)
#     return spike_train

# time_step = 100
# spike = Spike_train_converter(inputs=train_x_data, dt = 1e-3, time_step = time_step)