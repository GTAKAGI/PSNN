# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from snn_model import p_snu_layer
from chainer import Variable

img_save_dir = "./imgs/"
os.makedirs(img_save_dir, exist_ok=True)
    
""" Build Spiking Neural Unit """
num_time = 50 # simulation time step
V_th = 2.5
tau = 25e-3 # sec
dt = 1e-3 # sec

snu_l = p_snu_layer.P_SNU(input_neuron=3, output_neuron=1, l_tau=(1-dt/tau),
                      soft=False, initial_bias=-V_th)
# snu_l.Wx.W = Variable(np.array([[1.0]], dtype=np.float32))


""" Generate Poisson Spike Trains(1入力1出力ver) """
# fr = 100 # Hz
# x = np.where(np.random.rand(1, num_time) < fr*dt, 1, 0)
# x = np.expand_dims(x, 0).astype(np.float32)
# x = torch.from_numpy(x.astype(np.float32)).clone()

""" Generate Poisson Spike Trains(3入力1出力ver) """
fr = 100 # Hz
x = np.where(np.random.rand(3, num_time) < fr*dt, 1, 0)
x = np.expand_dims(x, 0).astype(np.float32)
x = torch.from_numpy(x.astype(np.float32)).clone()
# print(x[0,0])
# exit()
""" Simulation """
s_arr = np.zeros(num_time) # array to save membrane potential
y_arr = np.zeros(num_time) # array to save output
# k = snu_l(x[:,:,:])
# print(k.array)
# print(snu_l.s.array)
# y,s,p= snu_l(torch.tensor([[0.],[0.],[0.]]))
# print(y,s,p)
# exit()
for i in range(num_time):    
    # print(x[:, :, i])
    # exit()
    y,s ,p= snu_l(x[:, :, i])
    print(p)
    # exit()
    # print(g)
    # exit()
    s_arr[i] = s
    y_arr[i] = y
    print(y,s_arr[i])

""" Plot results """    
plt.figure(figsize=(6,6))

plt.subplot(3,2,1)
plt.title("Spiking Neural Unit")
plt.plot(x[0,0])
plt.ylabel("Input")

plt.subplot(3,2,2)
plt.plot(s_arr)
plt.ylabel('Membrane\n Potential')

plt.subplot(3,2,3)
plt.plot(x[0,1])
plt.ylabel("Input-2")

plt.subplot(3,2,4)
plt.plot(y_arr)
plt.ylabel("Output")
plt.xlabel("Time (ms)")

plt.subplot(3,2,5)
plt.plot(x[0,2])
plt.ylabel("Input-3")

plt.tight_layout()
plt.savefig(img_save_dir+"Check_SNU_result.png")