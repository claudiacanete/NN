# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 00:00:29 2023

@author: claud
"""

#generate training and testing data - classification example

import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import sys
import yfinance as yf
import pandas as pd
import statistics
from sklearn.metrics import mean_squared_error
from pandas.plotting import lag_plot
from statsmodels.tsa.ar_model import AutoReg
from math import sqrt
import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg as AR
import datetime as dt

amzn = yf.download(tickers=['NCLH'],start = "2019-01-01",end = "2021-01-01",interval="1d")
c=[0,0,0,0,0]
c[0]=np.corrcoef(amzn.Volume,amzn.Open)[0][1]
c[1]=np.corrcoef(amzn.Volume,amzn.High)[0][1]
c[3]=np.corrcoef(amzn.Volume,amzn.Close)[0][1]
c[2]=np.corrcoef(amzn.Volume,amzn.Low)[0][1]
c[4]=np.corrcoef(amzn.Volume,amzn["Adj Close"])[0][1]
# Create a dataset with 10,000 samples.
X=np.array([np.array(amzn.Close),np.array(amzn.Open)])
X=X.transpose()
y=np.array(amzn.Volume/max(amzn.Volume))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=26)

# Visualize the data.
fig, (train_ax, test_ax) = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(10, 5))
train_ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Spectral)
train_ax.set_title("Training Data")
train_ax.set_xlabel("Feature #0")
train_ax.set_ylabel("Feature #1")

test_ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
test_ax.set_xlabel("Feature #0")
test_ax.set_title("Testing data")
plt.show()

#convert training and testing data to torch tensors

import warnings
warnings.filterwarnings("ignore")
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Convert data to torch tensors
class Data(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.len = self.X.shape[0]
       
    def __getitem__(self, index):
        return self.X[index], self.y[index]
   
    def __len__(self):
        return self.len
   
batch_size = 5
# Instantiate training and test data
train_data = Data(X_train, y_train)
train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

test_data = Data(X_test, y_test)
test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

# Check it's working
for batch, (X, y) in enumerate(train_dataloader):
    print(f"Batch: {batch+1}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    break
#creating the NN model - pay attention at the activation function
import torch
from torch import nn
from torch import optim

input_dim = 2
hidden_dim = 10
output_dim = 1

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="relu")
        self.layer_2 = nn.Linear(hidden_dim, output_dim)
       
    def forward(self, x):
        x = torch.nn.functional.tanh(self.layer_1(x))
        x = torch.nn.functional.sigmoid(self.layer_2(x))

        return x
       
model = NeuralNetwork(input_dim, hidden_dim, output_dim)
print(model)
#choosing the optimizer and the learning rate
learning_rate = 0.001
loss_fn = nn.L1Loss()
#stochastic gradient descent
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#train the model
num_epochs = 700
loss_values = []
for epoch in range(num_epochs):
    for X, y in train_dataloader:
        # zero the parameter gradients
        optimizer.zero_grad()
       
        # forward + backward + optimize
        pred = model(X)
        loss = loss_fn(pred, y.unsqueeze(-1))
        loss_values.append(loss.item())
        loss.backward()
        optimizer.step()

print("Training Complete")
#visualize the training loss
step = np.linspace(0, 700, 47600)
fig, ax = plt.subplots(figsize=(8,5))
plt.plot(step, np.array(loss_values))
plt.title("Step-wise Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()
#use the model for classification
import itertools
"""
We're not training so we don't need to calculate the gradients for our outputs
"""
y_pred = []
y_test = []
correct = total = 0
with torch.no_grad():
    for X, y in test_dataloader:
        outputs = model(X)
        predicted = outputs
        predicted = list(itertools.chain(*predicted))
        y_pred.append(predicted)
        y_test.append(y)
        total += y.size(0)

i=0
j=0
k=0
p=[]
t=[]
while i<len(y_pred):
    j=0
    while j<len(y_pred[i]):
        p.append(y_pred[i][j].item()*max(amzn.Volume))
        t.append(y_test[i][j].item()*max(amzn.Volume))
        j=j+1
    i=i+1

plt.plot(p, label="prediction")
plt.plot(t, label="real data")
plt.legend()
plt.title("NN Model Volume Forecasting")
plt.show()
plt.plot([(a_i - b_i)**2 for a_i, b_i in zip(p, t)])
plt.title("MSE AMZN NN Model Volume Forecasting")
plt.show()
mse_nn_amzn=statistics.mean([(a_i - b_i)**2 for a_i, b_i in zip(p, t)])
print(mse_nn_amzn) #1614909303159822.5 for AMZN and 552510528553287.5 for NCLH
