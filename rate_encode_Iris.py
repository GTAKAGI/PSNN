import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd
import tqdm
import argparse
from torch import nn, optim
from snn_model import network
from snn_model import p_snu_layer
#Iris have three output classes, but two are given

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
X_train = X_train.T
print(X_train.shape)

y_train = y_train.T
# print(len(X_train.T))
#学習用と教師データをまとめる

#rate-encoding
def Spike_train_converter(inputs, dt, time, time_step, fr = 30, norm = 10):
    freq_temp = fr*norm/np.sum([retu[0] for retu in inputs])
    freq = freq_temp*np.repeat(np.expand_dims(inputs[:][time], axis = 0), time_step, axis= 2)
    spike_train = np.where(np.random.rand(4, time_step) < freq*dt ,1 ,0)
    return spike_train

dt = 1e-3
time_step = 100
data = 100
converted = np.empty((80,4,100))
#今は学習用しかやってないけど、評価用データもスパイク信号に置き換えて(edited 6/18)
for time in range(len(X_train.T)):
    converted[time] = Spike_train_converter(inputs = X_train, dt = dt, time = time, time_step = time_step)
    #スパイク変換後の各データを結合する
    
    # if time == 0:
    #     print(converted[time])
    #     print('success input converted first data')
    # else:
    #     converted[time] = np.concatenate([converted[time],converted[time-1]],0)
print(converted.shape)
converted = torch.tensor(converted)
print(type(converted))
print(converted.shape)
    # print(type(inputs_data))


#     #スパイク列の描画
#     fig = plt.figure()
#     x = [i for i in range(0,100)]
#     y1 = inputs_data[:,0]
#     y2 = inputs_data[:,1]
#     y3 = inputs_data[:,2]
#     y4 = inputs_data[:,3]
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
#     ax4.set_ylabel('Petal-Wifth')
#     ax4.set_ylim(0,1)
#     plt.plot(x,y4)
#     ax4.set_xlabel('time')
#     plt.show()
#     # exit()
    
# inputs_data = Spike_train_converter(inputs = X_train, dt = dt, time = 0, time_step = time_step)
# print(len(inputs_data)) # 100

# #スパイク列の描画
# fig = plt.figure()
# x = [i for i in range(0,100)]
# y1 = inputs_data[:,0]
# y2 = inputs_data[:,1]
# y3 = inputs_data[:,2]
# y4 = inputs_data[:,3]
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
# ax4.set_ylabel('Petal-Wifth')
# ax4.set_ylim(0,1)
# plt.plot(x,y4)
# ax4.set_xlabel('time')
# plt.show()


# for i in range(time_step):
#     plt.imshow(np.reshape(inputs_data[i], (2,2)),cmap = 'gray')
#     plt.show()

#ネットワークの構築
class SNN(nn.Module):
    def __init__(self):
        super().__init__()
        
parser = argparse.ArgumentParser()
parser.add_argument('--batch','-b',type=int, default=80) #バッチ学習
parser.add_argument('--epoch','-e',type=int, default=100) #epoch
parser.add_argument('--power','-p',type=int, default = False) #消費電力の計算
parser.add_argument('--num_time', '-n',type=int,default=300) #一つのデータを何ステップに分けてエンコーディングするか
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

SNN_model = network.SNN_Net(num_time=args.time, l_tau=0.8,power=args.power,batch_size=args.batch,gpu=False)
SNN_model = SNN_model.to(device)

print("学習スタート")

optimizer = optim.Adam(SNN_model.parameters(),eta = 1e-3)
epochs = args.epoch

for epoch in tqdm(range(epochs)):
    running_loss = 0
    local_loss = []
    print('EPOCH',epoch)
    #modelを保存する
    if epoch == 0:
        torch.save(SNN_model.state_dict(), 'iris_models_state_dict_'+str(epoch)+'epochs.pth')
        print('success model saving')
        
    with tqdm(total=len(X_train),desc=f'Epoch{epoch+1}/{epochs}',unit='img')as pbar:
        
        #inputs_dataはirisの4分類データ、labelsは正解値(setosaであれば0とかそういうこと)
        for i, (inputs_data,labels) in enumerate(X_train,0): #X_trainはラベルと一緒にする 80×4 → 80×5
            optimizer.zero_grad()
            
            inputs_data = inputs_data.to(device)
            labels = labels.to(device)
            
            if args.power:
                pass
                # loss, pred
            
            else:
                #値として返すもの→正解率、損失関数、
                loss, prediction = SNN_model(inputs_data,labels)
            
            loss.backward()
            running_loss += loss.item()
            local_loss.append(loss.item())
            del loss
            optimizer.step()
            
            #バッチ学習終わり
            
            if i % 100 == 99:
                print('[{:d}, {:5d}] loss: {:.3f}'
                            .format(epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
            
print('ok')
            
