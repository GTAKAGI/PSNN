# %%
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data
import torch.nn as nn
# import snntorch
import pandas as pd
from tqdm import tqdm
import argparse
from torch import nn, optim
from sklearn import datasets
from snn_model import network
from snn_model import p_snu_layer

#教師ラベル
df = pd.read_csv('iris.data', header=None)
y = df.iloc[0:150, 4].values
y_train = []
y_eval = []
# 教師ラベル
for i in range(120):
    if 0 <= i <= 39:
        y_train.append([0])
    if 40 <= i <= 79:
        y_train.append([1])
    if 80 <= i:
        y_train.append([2])
y_train = torch.tensor(y_train)
#正解ラベル
for i in range(30):
    if 0 <= i <= 9:
        y_eval.append([0])
    if 10 <= i <= 19:
        y_eval.append([1])
    if 20 <= i:
        y_eval.append([2])
y_eval = torch.tensor(y_eval)

#学習用実データ
X = df.iloc[0:150,[0,1,2,3]].values 
X_train = np.empty((120, 4)) 
X_test = np.empty((30, 4))
#学習,評価実データ
X_train[:40],X_train[40:80],X_train[80:120] = X[:40],X[50:90],X[100:140]
X_test[:10],X_test[10:20],X_test[20:] = X[40:50],X[90:100], X[140:]

# encoding


# time = data_num, time_step = rate spike time
def Spike_train_converter(inputs, dt, time, time_step, fr=30, norm=10):
    freq_temp = fr*norm/np.sum(inputs[time])
    freq = freq_temp * \
        np.repeat(np.expand_dims(inputs[time], axis=0), time_step, axis=0)
    spike_train = np.where(np.random.rand(time_step, 4) < freq*dt, 1, 0)
    return spike_train


def Spike_eval_converter(inputs, dt, time_, time_step, fr=30, norm=10):
    freq_temp = fr*norm/np.sum(inputs[time_])
    freq = freq_temp * \
        np.repeat(np.expand_dims(inputs[time_], axis=0), time_step, axis=0)
    spike_train = np.where(np.random.rand(time_step, 4) < freq*dt, 1, 0)
    return spike_train


dt = 1e-3  # 微小時間にスパイクが発生するか
time_step = 100  # スパイクを観測する時間
data = 150  # 全データ長さ

# spike_datasets(edited 7/9 one hot vector to simple number)
state_converted_train = np.empty((120, 100, 4))
state_converted_eval = np.empty((30, 100, 4))
label_train = torch.empty((120))
label_eval = np.empty(30)

# 今は学習用しかやってないけど、評価用データもスパイク信号に置き換える(edited 6/18)
for time in range(len(X_train)):
    state_converted_train[time] = Spike_train_converter(
        inputs=X_train, dt=dt, time=time, time_step=time_step)

for time_ in range(len(X_test)):
    state_converted_eval[time_] = Spike_eval_converter(
        inputs=X_test, dt=dt, time_=time_, time_step=time_step)

# spike_datasets
state_converted_train = torch.from_numpy(
    state_converted_train).float()  # 時系列入力スパイク (120,100,4)
state_converted_eval = torch.from_numpy(
    state_converted_eval).float()  # 時系列スパイク (30,100,4)

#訓練データの表示###################################################################################################################################

# answers
# nyl = torch.sum(state_converted_train,dim=1) #(120,4) 時間軸方向にスパイクの和をとる。各データの、各クラスの合計スパイクが見れる。
# print(nyl)

# see spike data in order
# for k in range(120):
#     fig = plt.figure()
#     x = [i for i in range(0,100)]
#     y1 = state_converted_train[k,:,0]
#     y2 = state_converted_train[k,:,1]
#     y3 = state_converted_train[k,:,2]
#     y4 = state_converted_train[k,:,3]
#     ax1 = fig.add_subplot(4,1,1)
#     ax1.set_ylabel('Sepal-Length')
#     ax1.set_ylim(0,1)
#     plt.plot(x,y1)
#     ax2 = fig.add_subplot(4,1,2)
#     ax2.set_ylabel('Sepal-Width')
#     ax2.set_ylim(0,1)
#     plt.plot(x,y2)
#     ax3 = fig.add_subplot(4,1,3)
#     ax3.set_ylabel('Petal-Length')
#     ax3.set_ylim(0,1)
#     plt.plot(x,y3)
#     ax4 = fig.add_subplot(4,1,4)
#     ax4.set_ylabel('Petal-Width')
#     ax4.set_ylim(0,1)
#     plt.plot(x,y4)
#     ax4.set_xlabel('time')
#     ax1.set_title('data' + str(k+1))
#     plt.show()

# 指定したデータを見る
# print('what time_step of spike data you want to see?')
# idx = int(input())
# fig = plt.figure()
# x = [i for i in range(0,100)]
# y1 = state_converted_train[idx,:,1]
# y2 = state_converted_train[idx,:,1]
# y3 = state_converted_train[idx,:,2]
# y4 = state_converted_train[idx,:,3]
# ax1 = fig.add_subplot(4,1,1)
# ax1.set_ylabel('Sepal-Length')
# ax1.set_ylim(0,1)
# plt.plot(x,y1)
# ax2 = fig.add_subplot(4,1,2)
# ax2.set_ylabel('Sepal-Width')
# ax2.set_ylim(0,1)
# plt.plot(x,y2)
# ax3 = fig.add_subplot(4,1,3)
# ax3.set_ylabel('Petal-Length')
# ax3.set_ylim(0,1)
# plt.plot(x,y3)
# ax4 = fig.add_subplot(4,1,4)
# ax4.set_ylabel('Petal-Width')
# ax4.set_ylim(0,1)
# plt.plot(x,y4)
# ax4.set_xlabel('time')
# plt.show()

# print(nyl)
##############################################################################################################################################


class SNN(nn.Module):
    def __init__(self):
        super().__init__()


parser = argparse.ArgumentParser()
parser.add_argument('--batch','-b',type=int, default=120)# DON'T CHANGE!
parser.add_argument('--out_shape','-o',type=int, default=False)
parser.add_argument('--epoch','-e',type=int, default=100) #Epoch 
parser.add_argument('--power','-p',type=int, default = False) #power consumption
parser.add_argument('--num_time', '-t',type=int,default=100) #一つのデータを何ステップに分けてエンコーディングするか=スパイクの観測時間
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available()
                      else 'cpu')  # No need to use GPU

SNN_model = network.SNN_Net(num_time=args.num_time, l_tau=0.8, power=args.power,
                            batch_size=args.batch, gpu=False)  # l_tau : more complicated spikign neuron
SNN_model = SNN_model.to(device)

print("学習スタート")
# print(SNN_model)
# learning rate why Adam?, else SGD
optimizer = optim.Adam(SNN_model.parameters(), lr=1e-3)
epochs = args.epoch

for epoch in tqdm(range(epochs)):
    running_loss = 0
    local_loss = []
    local_accuracy = []
    print('EPOCH',epoch)
        
    with tqdm(total=len(state_converted_train),desc=f'Epoch{epoch+1}/{epochs}',unit='img') as pbar:
    
        # for i, (inputs_data,labels) in enumerate(X_train,0): #cannnot use enumrate function because of concatenation of inputs and labels were impossible
        for i in range(120):

            total_spike_per_data = []

            inputs_data = state_converted_train[i]
            labels = y_train[i]
            inputs_data = inputs_data.to(device)
            labels = labels.to(device)
            # print(inputs_data)
            # print(inputs_data[:,0])
            # exit()

            if args.power:  # add this function later
                pass

            else:
                # outputs
                pi1,pi2 = SNN_model(inputs_data,labels) 
                # spike_num,loss,accu = SNN_model(inputs_data,labels)
            # print(accu)
            #膜電位の状態を確認する 
            # membrane= membrane.detach().numpy()
            pi1 = pi1.detach().numpy()
            pi2 = pi2.detach().numpy()
        
            # print(pi)
            # print(ll)
            # exit()
            
            # print(sp)
            # print(membrane[:,0])
            # exit()
            if i == 0:
                plt.figure(figsize=(6,6))
                plt.subplot(4,2,1)
                plt.title("lay1-mem")
                plt.plot(pi1[:,0])
                plt.ylabel("mem1")
                
                plt.subplot(4,2,3)
                plt.plot(pi1[:,1])
                plt.ylabel("mem2")
                
                plt.subplot(4,2,5)
                plt.plot(pi1[:,2])
                plt.ylabel("mem3")
                
                plt.subplot(4,2,7)
                plt.plot(pi1[:,3])
                plt.ylabel("mem4")
                
                plt.subplot(4,2,2)
                plt.title('lay2-mem')
                plt.plot(pi2[:,0])
                plt.ylabel("mem1")
                
                plt.subplot(4,2,4)
                plt.plot(pi2[:,1])
                plt.ylabel("mem2")
                
                plt.subplot(4,2,6)
                plt.plot(pi2[:,2])
                plt.ylabel("mem3")
                
                plt.subplot(4,2,8)
                plt.plot(pi2[:,3])
                plt.ylabel("mem4")
                
                plt.show()
            #########################SNUの入出力確認+膜電位###############################
            # if i == 0:
            #     plt.figure(figsize=(6,6))
            #     plt.subplot(3,3,1)
            #     plt.title("input-neurons")
            #     plt.plot(inputs_data[:,0])
            #     plt.ylabel("Input-1")

            #     plt.subplot(3,3,4)
            #     plt.plot(inputs_data[:,1])
            #     plt.ylabel("Input-2")

            #     plt.subplot(3,3,7)
            #     plt.plot(inputs_data[:,2])
            #     plt.ylabel("Input-3")
            #     plt.xlabel("Time (ms)")

            #     # plt.subplot(4,2,7)
            #     # plt.plot(inputs_data[:,3])
            #     # plt.ylabel("Input-4")

            #     plt.subplot(3,3,2)
            #     plt.title("Membrane potential")
            #     plt.plot(membrane[:,0])
            #     plt.ylabel('Mem1')

            #     plt.subplot(3,3,5)
            #     plt.plot(membrane[:,1])
            #     plt.ylabel("Mem2")

            #     plt.subplot(3,3,8)
            #     plt.plot(membrane[:,2])
            #     plt.ylabel("Mem3")
            #     plt.xlabel("Time (ms)")

            #     plt.subplot(3,3,3)
            #     plt.title("output spikes")
            #     plt.plot(sp[:,0])
            #     plt.ylabel("spike-1")

            #     plt.subplot(3,3,6)
            #     plt.plot(sp[:,1])
            #     plt.ylabel("spike-2")

            #     plt.subplot(3,3,9)
            #     plt.plot(sp[:,2])
            #     plt.ylabel("spike-3")
            #     plt.xlabel("Time (ms)")
            #     plt.show()
            #################################################################
                
            # print(spike_num)
            # print(losse)
                
            # loss.backward()
            # running_loss += loss.item()
            # local_loss.append(loss.item())
            # optimizer.step()
            # optimizer.zero_grad()
            # local_accuracy.append(accu)
            # print(loss)

            if i == 119:
                # print(local_accuracy)
                # print('[{:d}, {:5d}] loss: {:.3f}'
                #             .format(epoch + 1, i + 1, running_loss / 120))
                print('[{:d}, {:5d}] loss: {:.3f} accuracy: {:.2f}%'
                            .format(epoch + 1, i + 1, running_loss / 120, 100*sum(local_accuracy)/len(local_accuracy)))
                print('This is Epoch' + str(epoch+1))
                running_loss = 0.0
                
            
    print('ok')

    # evaluation
    # Needs softmax function in order to evaluate

    # with torch.no_grad():
    #     for times in range(len(X_test)):
    #         inputs_data = state_converted_eval[times].to(device)
    #         labels = label_eval[times].to(device)
    #         loss, accuracy = SNN_model(inputs_data,labels)


torch.save(SNN_model.state_dict(), 'snn_model/spike_iris_model_snn.pth')
print('success model saving')
