import torch
import torch.nn as nn
import snntorch as snn

alpha = 0.9
beta = 0.85

num_steps = 100
num_inputs = 100
num_hidden = 1000
num_outputs = 10

# Define Network
class Net(nn.Module):
   def __init__(self):
      super().__init__()

      # initialize layers
      self.fc1 = nn.Linear(num_inputs, num_hidden)
      self.lif1 = snn.Leaky(beta=beta)
      self.fc2 = nn.Linear(num_hidden, num_outputs)
      self.lif2 = snn.Leaky(beta=beta)

   def forward(self, x):
      mem1 = self.lif1.init_leaky()
      mem2 = self.lif2.init_leaky()

      spk2_rec = []  # Record the output trace of spikes
      mem2_rec = []  # Record the output trace of membrane potential

      for step in range(num_steps):
            cur1 = self.fc1(x.flatten(1))
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

      return torch.stack(spk2_rec), torch.stack(mem2_rec)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = Net().to(device)

output, mem_rec = net(data)