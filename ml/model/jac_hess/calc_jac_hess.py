import sys

sys.path.append("/home/kyang/master_grid/ml/model")
import numpy as np
import torch.nn as nn
import pandas as pd 
from model import MLP_binary
import torch
from datahandler import minmax_scaler,TabularDataset
from torch.utils.data import Dataset, DataLoader
import argparse
import datetime
import copy

def main(end_date, start_date,features, feat,checkpoint_path,batch_size,name):
        
    scaled_data = []
    print("LOAD DATA")

    while start_date <= end_date:
        start_date_str = str(start_date).replace("-","_")
        scaled_data.append(pd.read_csv(f"/share/scratch1/kiajueng_yang/data/{start_date_str}.csv",usecols=feat))
        start_date += datetime.timedelta(days=1)
        
    scaled_data = pd.concat(scaled_data)

    scaled_data.loc[:,features] = minmax_scaler(scaled_data.loc[:,features], "/home/kyang/master_grid/ml/model/model",True)
    
    dataset = TabularDataset(scaled_data,features=features,glob_weight=1)
    dataloader = DataLoader(dataset,batch_size=batch_size, shuffle=False)

    len_dataset = len(dataset)
    print(len_dataset)

    print("INITIALIZE MODEL")
    input_size = len(features)
    hidden_sizes = [32,64,32]
    hidden_act_fns = [nn.ReLU(),nn.ReLU(),nn.ReLU()]
    output_act_fn = nn.Sigmoid()
    output_size = 1

    model = MLP_binary(input_size = input_size,
                       hidden_sizes=hidden_sizes,
                       hidden_act_fns = hidden_act_fns,
                       output_act_fn = output_act_fn,
                       output_size = output_size,
    )

    model = model.double()

    print("LOAD MODEL STATE DICT")
    model_checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(model_checkpoint["model_state_dict"])
    
    model.eval()

    num_feat = len(features)
    hess = torch.zeros(num_feat,num_feat)
    jac = torch.zeros(1,num_feat)

    print("RUN")
    for x,y,w in dataloader:
        x.requires_grad = True
        with torch.enable_grad():
            output = model(x).sum().reshape(1)

        jac_nosum = torch.autograd.grad(output,x,torch.ones(1),create_graph=True)[0]
        jac += abs(jac_nosum).sum(axis=0)/len_dataset

        for i in range(num_feat):
            hess[i,:] += abs(torch.autograd.grad(jac_nosum[:,i],x,torch.ones(len(y)),retain_graph=True)[0]).sum(axis=0)/len_dataset

    with open(f"/home/kyang/master_grid/ml/model/jac_hess/hessian_{name}.npy", "wb") as f:
        np.save(f,hess.detach().numpy())
            
    with open(f"/home/kyang/master_grid/ml/model/jac_hess/jacobian_{name}.npy", "wb") as f:
        np.save(f, jac.detach().numpy())
        
    with open(f"/home/kyang/master_grid/ml/model/jac_hess/features_{name}.txt", "w+") as f:
        for i in features:
            f.write(i+"\n")

if __name__=="__main__":

#    features = ["io_intensity","wall_time","diskio","memory_leak","IObytesWriteRate", "IObytesReadRate","IObytesRead","IObytesWritten","actualcorecount","inputfilebytes","cpu_eff"]
    features = ["io_intensity","wall_time","diskio","memory_leak","IObytesWriteRate", "IObytesReadRate","IObytesRead","IObytesWritten","outputfilebytes","actualcorecount","inputfilebytes","cpu_eff", "cpuconsumptiontime"]
    
    feat = copy.deepcopy(features)
    feat.append("jobstatus")
    name = "Test"
    start_date = datetime.date(2023,11,1)
    end_date = datetime.date(2023,11,30)

    checkpoint_path = "/home/kyang/master_grid/ml/model/model/model_checkpoint.tar"

    main(start_date=start_date,
         end_date=end_date,
         feat = feat,
         features=features,
         checkpoint_path=checkpoint_path,
         batch_size = 256,
         name=name,
    )
