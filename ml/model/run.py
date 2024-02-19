import numpy as np
import torch
import torch.nn as nn
from torch.optim import NAdam
from model import MLP_binary
from datahandler import train_test_split, minmax_scaler,TabularDataset
from torch.utils.data import Dataset, DataLoader
from train import training
from torch.optim.lr_scheduler import StepLR

print("LOADING DATA...")
cols = ["io_intensity","wall_time","diskio","memory_leak","IObytesWriteRate", "IObytesReadRate","IObytesRead","IObytesWritten","outputfilebytes","actualcorecount","inputfilebytes","cpu_eff", "cpuconsumptiontime","new_weights", "jobstatus"]
features = ["io_intensity","wall_time","diskio","memory_leak","IObytesWriteRate", "IObytesReadRate","IObytesRead","IObytesWritten","outputfilebytes","actualcorecount","inputfilebytes","cpu_eff", "cpuconsumptiontime"]

train_data, test_data = train_test_split([0.8,0.2],cols,seed=0,cond='("2023_09" in file) | ("2023_08" in file) | ("2023_10" in file)')

train_data.loc[:,features] = minmax_scaler(train_data.loc[:,features], "/home/kyang/master_grid/ml/model/model")

test_data.loc[:,features] = minmax_scaler(test_data.loc[:,features], "/home/kyang/master_grid/ml/model/model",True)

train_dataset = TabularDataset(train_data,features=features,index=False)
test_dataset= TabularDataset(test_data,features=features,index=False)

train_dataloader = DataLoader(train_dataset, batch_size = 128, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size = 128, shuffle=False)

input_size = len(features)
hidden_sizes = [32,64,32]
hidden_act_fns = [nn.ReLU(),nn.ReLU(),nn.ReLU()]
output_act_fn = nn.Sigmoid()
output_size = 1

print("INITIALIZING MODEL....")
model = MLP_binary(input_size = input_size,
                   hidden_sizes=hidden_sizes,
                   hidden_act_fns = hidden_act_fns,
                   output_act_fn = output_act_fn,
                   output_size = output_size)
model = model.double()
loss_fn = nn.BCELoss(reduction="none")
num_epochs = 60
early_stopping = True
optimizer = NAdam(model.parameters(),lr=1e-3, weight_decay=1e-5)
scheduler = StepLR(optimizer, step_size = 15, gamma=0.5)

print("LOAD MODEL CHECKPOINT")
checkpoint = torch.load("/home/kyang/master_grid/ml/model/model/model_checkpoint.tar")

print("START TRAINING...")
train_model = training(train_dataloader,test_dataloader,optimizer,model,loss_fn,num_epochs,scheduler=scheduler,checkpoint=checkpoint)
train_model(path="/home/kyang/master_grid/ml/model/model")
