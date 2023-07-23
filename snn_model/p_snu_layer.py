import numpy as np
import matplotlib.pyplot as plt
import torch
torch.manual_seed(0)
import torch.nn.functional as F
import torch.nn as nn
# import snntorch
import pandas as pd
import tqdm
import argparse
from . import step_func

class P_SNU(nn.Module):
    def __init__(self,input_neuron,output_neuron,l_tau,soft = False,gpu = False,nobias = False, initial_bias = -2):
        super(P_SNU,self).__init__()
        
        self.input_neuron = input_neuron
        self.output_neuron = output_neuron
        self.l_tau = l_tau
        self.soft = soft
        self.gpu = gpu
        self.initial_bias = initial_bias
        self.s = None
        self.y = None
        
        if self.gpu:
            #xp = cuda.cupy
            dtype = torch.float
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            dtype = torch.float
            device=torch.device("cpu")
            
        self.input_current = nn.Linear(input_neuron,output_neuron,bias=False).to(device)
        # torch.manual_seed(1)
        torch.nn.init.xavier_uniform_(self.input_current.weight)
        #torch.nn.init.constant_(self.input_current.weight, 0.5)
        
        if nobias:
            self.b = None
            
        else:
            device = torch.device(device)
            self.b = nn.Parameter(torch.Tensor([initial_bias]).to(device))
    
    def reset_state(self,s = None, y = None):
        self.s = s #membrane potential
        self.y = y #emitted spikes
    
    def initialize_state(self, shape):
        if self.gpu:
            #xp = cuda.cupy
            dtype = torch.float
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            dtype = torch.float
            device=torch.device("cpu")
        
        self.s = torch.zeros((1, 3),device=device,dtype=dtype)
        self.y = torch.zeros((1, 3),device=device,dtype=dtype)
        # self.y = torch.zeros((shape[0], self.output_neuron),device=device,dtype=dtype)
    
    def forward(self,x):
        if self.s == None:
            # self.s = torch.tensor[0,0,0]
            self.initialize_state(x) #膜電位の初期化, membrane_potential = 0
            # a = self.initialize_state(x)
        
        if type(self.s) == np.ndarray:
            self.s = torch.from_numpy(self.s.astype(np.float32)).clone()
        #snu absで学習進める
        s = abs(self.input_current(x)) + self.s*(1 - self.y)
        # s = F.elu(self.input_current(x) + self.s*(1 - self.y))
        #s = F.elu(abs(self.input_current(x)) + self.l_tau * self.s * (1 - self.y))
        bias = self.b + s
        y = step_func.spike_fn(bias)
        
        self.s = s
        self.y = y
        #P-snu decayを排除
        # s = F.elu(self.input_current(x) + self.s*(1 - self.y))
    
        return s,bias,y#spike emission
        # return y,s,x