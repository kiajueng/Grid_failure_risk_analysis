import sys

sys.path.append("/home/kyang/master_grid/ml/model")
import glob
import pandas as pd 
import numpy as np
from model import MLP_binary
import torch
import torch.nn as nn
from datahandler import minmax_scaler,TabularDataset
from torch.utils.data import Dataset, DataLoader
import argparse
import datetime

def main(date,features,checkpoint_path):
    
    date_str = str(date).replace("-","_")
    
    print("LOAD DATA")
    data = pd.read_csv(f"/share/scratch1/kiajueng_yang/data/{date_str}.csv")
    data["prediction"] = np.nan
    scaled_data = data.copy()
    scaled_data.loc[:,features] = minmax_scaler(scaled_data.loc[:,features], "/home/kyang/master_grid/ml/model/model",True)
    
    dataset = TabularDataset(scaled_data,features=features,index=True)
    dataloader = DataLoader(dataset,batch_size=256, shuffle=False)
    
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

    print("RUN")
    with torch.no_grad():
        for i,(index,x,y,w) in enumerate(dataloader):
            y_pred = model(x)
            data.loc[index.detach().numpy().flatten(),["prediction"]] = y_pred.detach().numpy().flatten()

            if i < 2:
                print(data.loc[index.detach().numpy().flatten(),"prediction"])

    #Save back to csv file
    if glob.glob(f"/share/scratch1/kiajueng_yang/data_pred/{date_str}.csv"):
        return 
    else:
        data.to_csv(f"/share/scratch1/kiajueng_yang/data_pred/{date_str}.csv", index=False)
        return 

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--day", type=int, required=True)
    parser.add_argument("--month", type=int, required=True)
    parser.add_argument("--year", type=int, required=True)
    args = parser.parse_args()

    
    features = ["io_intensity","wall_time","diskio","memory_leak","IObytesWriteRate", "IObytesReadRate","IObytesRead","IObytesWritten","outputfilebytes","actualcorecount","inputfilebytes","cpu_eff", "cpuconsumptiontime"]

    date = datetime.date(args.year,args.month,args.day)

    checkpoint_path = "/home/kyang/master_grid/ml/model/model/model_checkpoint.tar"

    main(date=date,
         features=features,
         checkpoint_path=checkpoint_path,
    )
