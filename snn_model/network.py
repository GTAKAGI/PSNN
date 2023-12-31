import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
# import snntorch
import pandas as pd
import tqdm
import argparse
from . import p_snu_layer

class SNN_Net(torch.nn.Module):
    def __init__(self, inputs_num = 4, hidden_num = 4, outputs_num = 3 ,l_tau = 0.8,num_time = 100, batch_size = 80 ,soft = False, rec = False, power = False, gpu = True):
        super().__init__()
        
        self.num_time = num_time
        self.batch_size = batch_size
        self.rec = rec
        self.power = power
        
        #parametr
        # self.neuron_0 = 4
        # self.neuron_1 = 24
        # self.neuron_2 = 24
        # self.neuron_3 = 4
        
        #my #hidden num = 24
        # self.l1 = p_snu_layer.P_SNU(inputs_num, hidden_num, l_tau = l_tau, soft = soft, gpu = gpu)
        # self.l2 = p_snu_layer.P_SNU(hidden_num, hidden_num, l_tau = l_tau, soft = soft, gpu = gpu)
        # self.l3 = p_snu_layer.P_SNU(hidden_num, hidden_num, l_tau = l_tau, soft = soft, gpu = gpu)
        # self.l4 = p_snu_layer.P_SNU(hidden_num, outputs_num, l_tau = l_tau, soft = soft, gpu = gpu)
        
        #my2 hidden num = 4
        self.l1 = p_snu_layer.P_SNU(inputs_num, hidden_num, l_tau = l_tau, soft = soft, gpu = gpu)
        self.l2 = p_snu_layer.P_SNU(hidden_num, hidden_num, l_tau = l_tau, soft = soft, gpu = gpu)
        self.l3 = p_snu_layer.P_SNU(hidden_num, outputs_num, l_tau = l_tau, soft = soft, gpu = gpu)
        
        # for 1 layer test
        # self.l4 = p_snu_layer.P_SNU(inputs_num, outputs_num, l_tau = l_tau, soft = soft, gpu = gpu)
        
    def reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()
        self.l3.reset_state()
        # self.l4.reset_state()
    
    def forward(self,x,y):
        # y = torch.tensor(y)
        losse = None
        accuracy = None
        sum_out = None #タイムステップ毎のスパイク数の累積をとる
        out_list = [] #各データ(120×4)のタイムステップ(100ms)における出力スパイクを時系列で挿入
        out_total_list = []
        membrane_out = torch.empty(100,3)
        mem1_out = torch.empty(100,4)
        mem2_out = torch.empty(100,4)
        spikes_ = torch.empty(100,4)
        
        self.reset_state()
        
        for time in range(self.num_time): #num_timeはデフォルトで100になる。
            # spike_encoded_neuron = x[time]
            # target_ = torch.reshape(y[time],(1,3))
            # spike_encoded_neuron = torch.reshape(x[time],(4,1))
            
            #4→4→3(network)
            spike_encoded_neuron = x[time]
            h1,mem1,u1 = self.l1(spike_encoded_neuron)
            h2,mem2,u2 = self.l2(h1)
            # out,mem = self.l3(h2)
            
            # 1 layer test 4→3(network)
            #膜電位と入出力スパイクの確認
            # out,thresh,spike = self.l4(spike_encoded_neuron)
            
            #normal
            # out = self.l4(spike_encoded_neuron)
            
            # sum_out = out if sum_out is None else sum_out + out
            mem1_out[time] = mem1
            mem2_out[time] = mem2
            # membrane_out[time] = out
            spikes_[time] = h1
        
        #出力を確認する
        # return sum_out,y
        
        #バッチ学習の場合
        # criterion = nn.CrossEntropyLoss()
        # losse = criterion(sum_out,y)

        #正解率
        # predicted_label = torch.argmax(sum_out)
        # accuracy = 1 if predicted_label == y else 0
        return spikes_,mem1_out,mem2_out
        return mem_out
        return sum_out,losse,accuracy
