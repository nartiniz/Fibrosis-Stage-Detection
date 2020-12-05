from scipy.io import loadmat
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torch.utils.data
from torch.utils import data
from torch.utils.data import DataLoader, TensorDataset


df = loadmat('data2.mat')
df = df['data2']
df = np.array(df, dtype=np.float)
print(df)
print(df.shape)


nb_training_patients = 95
nb_features, batch_size, nb_epochs = 19, 20, 350
df[df[:,nb_features]<1.5,nb_features] = 0
df[df[:,nb_features]>1.5,nb_features] = 1
inp_training = df[:nb_training_patients, :nb_features]
inp_test = df[nb_training_patients:, :nb_features]
out_training = df[:nb_training_patients, nb_features]
out_test = df[nb_training_patients:, nb_features]
inp_training = torch.tensor(inp_training, dtype=torch.float)
inp_test = torch.tensor(inp_test, dtype=torch.float)
out_training = torch.tensor(out_training, dtype=torch.long)
out_test = torch.tensor(out_test, dtype=torch.long)


class NeuralNetwork(nn.Module):
    def __init__(self, inp_dim, l1, l2):
        super(NeuralNetwork, self).__init__()
        self.b0 = nn.BatchNorm1d(num_features=inp_dim)
        self.layer1 = nn.Linear(in_features=inp_dim, out_features=l1)
        self.b1 = nn.BatchNorm1d(num_features=l1)
        self.rel1 = nn.ReLU()
        self.layer2 = nn.Linear(in_features=l1, out_features=l2)
        self.b2 = nn.BatchNorm1d(num_features=l2)
        self.rel2 = nn.ReLU()
        self.layer3 = nn.Linear(in_features=l2, out_features=5, bias=True)
        # self.rel4 = nn.ReLU(inplace=True)
        self.layer4 = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.b0(x)
        x = self.layer1(x)
        x = self.b1(x)
        x = self.rel1(x)
        x = self.layer2(x)
        x = self.b2(x)
        x = self.rel2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

NNetwork = NeuralNetwork(nb_features, 150,70)
optimizer = optim.Adam(params=NNetwork.parameters(),lr=0.0005)

dataset = TensorDataset(inp_training, out_training)
training_generator = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
criterion = nn.CrossEntropyLoss()

for _ in range(nb_epochs):
    epoch_loss = 0
    for batch_data, batch_labels in training_generator:
        optimizer.zero_grad()
        # print(batch_data.device)
        output = NNetwork(batch_data)
        loss = criterion(output, batch_labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(_, end =" : ")
    print(epoch_loss)


NNetwork.eval()

# print(inp_test.shape)
print(out_test)
ModelOutput = NNetwork(inp_test)
_,prediction = torch.max(ModelOutput.data,1)
print(prediction)
correctAnswers = (prediction == out_test).sum().item()
print(correctAnswers)
print("Accuracy : %",end='')
print(100*correctAnswers/(105 - nb_training_patients))

ModelOutput = NNetwork(inp_training)
_,prediction = torch.max(ModelOutput.data,1)

correctAnswers = (prediction==out_training).sum().item()
accuracy = correctAnswers/nb_training_patients
print(accuracy)