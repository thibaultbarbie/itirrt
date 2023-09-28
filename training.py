"""
Copyright (c) 2023 OMRON Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the “Software”), to deal in the
Software without restriction, including without limitation the rights to use, copy,
modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


Author: Thibault Barbie
"""


import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

class TunnelDataset(Dataset):
    def __init__(self, n_data=100):
        self.dataset = np.zeros((n_data,9))
        for i in range(n_data):
            tunnel_y = np.random.uniform(0.1, 0.9) 
            self.dataset[i][0] = tunnel_y
            self.dataset[i][1:3] = np.array([start[0], start[1]]) # start  
            self.dataset[i][3:5] = np.array([goal[0], goal[1]]) # goal 
            self.dataset[i][5:7] = np.array([start[0], tunnel_y]) # left point 
            self.dataset[i][7:9] = np.array([goal[0], tunnel_y]) # right point 

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        batch_input = self.dataset[idx][:5]
        batch_target = self.dataset[idx][5:]

        batch_input_tensor  = torch.from_numpy(batch_input).float()
        batch_target_tensor = torch.from_numpy(batch_target).float()
        return batch_input_tensor, batch_target_tensor

class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        n_neurons = 2
        self.fc1 = nn.Linear(input_size, n_neurons)
        self.fc2 = nn.Linear(n_neurons, output_size)
        self.relu = nn.ReLU()

        self.optimizer = optim.Adam(self.parameters())
        self.loss_function = nn.MSELoss()

    def forward(self, x):
        h = self.relu(self.fc1(x))
        h = self.fc2(h)
        return h

    def update(self, batch_input, batch_target):
        self.optimizer.zero_grad()
        output = self.forward(batch_input)
        
        loss = self.loss_function(output, batch_target)
        
        loss.backward()
        self.optimizer.step()
        return loss.item()

    
start = np.array([0.05, 0.05])
goal  = np.array([0.95, 0.05])
dataset = TunnelDataset(10000)

model = MLP(5, 4)

data_loader = torch.utils.data.DataLoader(dataset, batch_size=15, shuffle=True, drop_last=True)
n_epochs = 10

pbar = tqdm(total=n_epochs, desc="Epoch", leave=False)
for _ in range(n_epochs):
    epoch_loss = 0

    for batch_idx, data in enumerate(data_loader):
        batch_input, batch_target = data[0], data[1]
        epoch_loss += model.update(batch_input, batch_target)             

    pbar.set_postfix(OrderedDict(loss=epoch_loss/(batch_idx+1)))
    pbar.update(1)
pbar.close()

# Record the model architecture
scripted_module = torch.jit.script(model)
torch.jit.save(scripted_module, f"tunnel_model.pt")

