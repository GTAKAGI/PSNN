# %%
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data
import torch.nn as nn
import snntorch
import pandas as pd
from tqdm import tqdm
import argparse
from torch import nn, optim
from sklearn import datasets
from snn_model import network
from snn_model import p_snu_layer
# (6/20)classで自動変換モジュールを作ってみたが、データセットを分割できないのでやっぱやめる。
# class LoadDataset(torch.utils.data.Dataset):
#     def __init__(self, num_data = 150, dt = 1e-3, num_time = 100, freq = 30,norm = 30):
#         super().__init__()

#         Data = datasets.load_iris()
#         inputs = Data.data
#         label = Data.target
#         input_data = np.zeros((num_data, num_time, 4))
#         output_data = np.zeros(num_data)
#         for k in tqdm(range(num_data)):
#             freq_temp = freq*norm/np.sum(inputs[k])
#             fr = freq_temp * np.repeat(np.expand_dims((inputs[k]),axis= 0), num_time, axis = 0)
#             input_data[k] = np.where(np.random.rand(num_time , 4) < fr*dt, 1, 0)
#             output_data[k] = label[k]

#         self.input_data = input_data.astype(np.float32)
#         self.output_data = output_data.astype(np.int8)
#         self.num_data = num_data

#     def __len__(self):
#         return self.num_data

#     def __getitem__(self, index):
#         return self.input_data[index],self.output_data[index]

# dataset = LoadDataset(num_data=150,dt = 1e-3,num_time=100,freq = 30)
# # print(dataset[:5][:].shape)
# train = dataset[:40][:],dataset[50:90][:],dataset[100:140][:]
# print(train)

# 実データセットに分ける(教師ラベルも同時にやる)
df = pd.read_csv('iris.data', header=None)
y = df.iloc[0:150, 4].values
y_label = []
# y_label = np.empty((150,3))#one-hot
for i in range(len(y)):
    if y[i] == 'Iris-setosa':
        y_label.append([0])
        # y_label[i] = np.array([1,0,0])
    elif y[i] == 'Iris-versicolor':
        y_label.append([1])
        # y_label[i] = np.array([0,1,0])
    else:
        y_label.append([2])
        # y_label[i] = np.array([0,0,1])
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
# 正解ラベル
for i in range(30):
    if 0 <= i <= 9:
        y_eval.append([0])
    if 10 <= i <= 19:
        y_eval.append([1])
    if 20 <= i:
        y_eval.append([2])
y_eval = torch.tensor(y_eval)

X = df.iloc[0:150, [0, 1, 2, 3]].values
X_train = np.empty((120, 4))
X_test = np.empty((30, 4))
# y_train = torch.empty(120)
# y_test = np.empty(30)

# y_train = np.empty((120,3))
# y_test = np.empty((30,3))
X_train[:40], X_train[40:80], X_train[80:120] = X[:40], X[50:90], X[100:140]
X_test[:10], X_test[10:20], X_test[20:] = X[40:50], X[90:100], X[140:]
# 7/20(ラベルがなぜかダウンロードから作れないので自作)
# y_train[:40],y_train[40:80],y_train[80:120] = y_label[:40],y_label[50:90],y_label[100:140]
# y_test[:10],y_test[10:20],y_test[20:] = y_label[40:50],y_label[90:100],y_label[140:]

# print(y_test) #ok
# print(X_train.shape) #120×4

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
# label_train = np.empty((120,3))
# label_eval = np.empty((30,3))

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
# label_train = torch.from_numpy(label_train) #教師ラベル (120,3)
# label_eval = torch.from_numpy(label_eval)
# print(torch.sum(state_converted_eval,dim=1)) #ok
# print(label_train)

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
# batch_learning DON'T CHANGE!
parser.add_argument('--batch', '-b', type=int, default=120)
parser.add_argument('--out_shape', '-o', type=int, default=False)
parser.add_argument('--epoch', '-e', type=int, default=100)  # epoch
# power consumption #NO FUNCTION
parser.add_argument('--power', '-p', type=int, default=False)
# 一つのデータを何ステップに分けてエンコーディングするか=スパイクの観測時間 DON'T CHANGE!
parser.add_argument('--num_time', '-t', type=int, default=100)
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available()
                      else 'cpu')  # No need to use GPU

SNN_model = network.SNN_Net(num_time=args.num_time, l_tau=0.8, power=args.power,
                            batch_size=args.batch, gpu=False)  # l_tau : more complicated spikign neuron
SNN_model = SNN_model.to(device)

print("学習スタート")

# learning rate why Adam?, else SGD
optimizer = optim.Adam(SNN_model.parameters(), lr=1e-4)
epochs = args.epoch

for epoch in tqdm(range(epochs)):
    running_loss = 0
    local_loss = []
    local_accuracy = []
    print('EPOCH', epoch)
    # modelを保存する
    if epoch == 0:
        torch.save(SNN_model.state_dict(),
                   'iris_models_state_dict_'+str(epoch)+'epochs.pth')
        print('success model saving')

    with tqdm(total=len(state_converted_train), desc=f'Epoch{epoch+1}/{epochs}', unit='img')as pbar:

        total_spike_per_epoch = torch.empty((3, 3))
        total_spike_per_epoch_all = torch.empty((120, 3))
        sum_spikes_per_epoch = 0

        # for i, (inputs_data,labels) in enumerate(X_train,0): #cannnot use enumrate function because of concatenation of inputs and labels were impossible
        for i in range(120):

            total_spike_per_data = []

            inputs_data = state_converted_train[i]
            labels = y_train[i]
            optimizer.zero_grad()
            inputs_data = inputs_data.to(device)
            # labels = labels.to(device)
            # print(labels,i)
            # print('#####3')

            if args.power:  # add this function later
                pass

            else:
                # requirements
                # spike_num,s_spike,loss = SNN_model(inputs_data,labels) #loss cant be None
                spike_num, label = SNN_model(inputs_data, labels)
            print(spike_num,label)
            continue
            # print(losse)

            # spike_num = [y1,y2,y3] < 100
            total_spike_per_epoch_all[i] = spike_num
            if i == 39:  # input last for each data
                total_spike_per_epoch[0] = spike_num
            elif i == 79:
                total_spike_per_epoch[1] = spike_num
            else:
                total_spike_per_epoch[2] = spike_num

            loss.backward()
            running_loss += loss.item()
            local_loss.append(loss.item())
            del loss
            optimizer.step()
            # local_accuracy.append(accuracy)
            # print(loss)

            # if i % 100 == 99:
            if i == 119:
                print('[{:d}, {:5d}] loss: {:.3f}'
                      .format(epoch + 1, i + 1, running_loss / 120))
                # print('[{:d}, {:5d}] loss: {:.3f} accuracy: {:.2f}%'
                #       .format(epoch + 1, i + 1, running_loss / 120, 100*sum(local_accuracy)/len(local_accuracy)))
                print('The total spikes are')
                # print(total_spike_per_epoch_all)
                print('Epoch' + str(epoch+1) + "だよ")
                running_loss = 0.0

            # print('One epoch done')
    # print('epoch' + str(epoch) + 'done')

    print('ok')  # learning is over

    # evaluation
    # Needs softmax function in order to evaluate

    # with torch.no_grad():
    #     for times in range(len(X_test)):
    #         inputs_data = state_converted_eval[times].to(device)
    #         labels = label_eval[times].to(device)
    #         loss, accuracy = SNN_model(inputs_data,labels)


torch.save(SNN_model.state_dict(), 'snn_model/spike_iris_model_key.pth')
print('success model saving')
