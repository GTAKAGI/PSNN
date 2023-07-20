import numpy as np
import matplotlib.pyplot as plt
import chainer

def online_load_and_encoding_dataset(dataset, i, dt, n_time, max_fr=32,norm=140):
    #np.sum(dataset[i][0])は画素の値の合計を返す
    fr_tmp = max_fr*norm/np.sum(dataset[i][0])
    fr = fr_tmp*np.repeat(np.expand_dims(dataset[i][0],axis=0), n_time, axis=0)
    input_spikes = np.where(np.random.rand(n_time, 784) < fr*dt, 1, 0)
    # input_spikes = input_spikes.astype(np.uint8)
    return input_spikes,dataset[0],fr


dt = 1e-3; t_inj = 0.350; nt_inj = round(t_inj/dt)
train, _ = chainer.datasets.get_mnist() # Chainer による MNIST データの読み込み,引数省略すると1次元ベクトルに自動で変換してくれる

input_spikes ,dataset, frequency = online_load_and_encoding_dataset(dataset=train, i=0,dt=dt, n_time=nt_inj)
# 描画
print(type(input_spikes))
print(input_spikes[0].reshape([28,28]))
print(input_spikes.shape)

#時系列データとして入力する際のコード
for i in range(0,5):
    # input_spikes[i] += input_spikes[i-1]
    print(input_spikes[i])
    plt.imshow(np.reshape(input_spikes[i],(28,28)), cmap= "gray")
    # plt.imshow(np.reshape(np.sum(input_spikes, axis=0), (28, 28)), cmap="gray")
    plt.show()      
    
# plt.imshow(np.reshape(np.sum(input_spikes, axis=0), (28, 28)), cmap="gray")
# plt.show()