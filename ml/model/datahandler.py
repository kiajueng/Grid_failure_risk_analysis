import glob
import dask.dataframe as dd
import datetime
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import pickle

def load_data(cols, jobtype,cond):
    ddf = []
    
    for file in glob.glob("/share/scratch1/es-atlas/atlas_jobs_enr_skimmed/*2023*"):
        if eval(cond):
            data = dd.read_csv(file, usecols=cols)
            data = data.dropna()
            data =  data[(data["new_weights"] != 0) & (data["jobstatus"] == jobtype) & (data["cpu_eff"] > 0.05)]
            if jobtype == "failed":
                data["jobstatus"] = data["jobstatus"].map({jobtype:0})
            else:
                data["jobstatus"] = data["jobstatus"].map({jobtype:1})
            data = data.astype("float64")
            ddf.append(data)
    ddf = dd.concat(ddf)
    return ddf

def train_test_split(split,cols,seed,cond):

    train_fail, test_fail= load_data(cols,jobtype="failed",cond=cond).random_split(split,shuffle=True, random_state = seed)
    train_fin, test_fin = load_data(cols, jobtype="finished",cond=cond).random_split(split,shuffle=True, random_state = seed)

    train = dd.concat([train_fail,train_fin]).compute()
    test = dd.concat([test_fail,test_fin]).compute()
    
    return train,test

def minmax_scaler(data_x, path,read_file = False):
    if not read_file:
        minima = dict(data_x.min())
        maxima = dict(data_x.max())
    
    else:
        with open(f"{path}/scaler.pkl","rb") as fp:
            scaler_dict = pickle.load(fp)
            minima = scaler_dict["minimum"]
            maxima = scaler_dict["maximum"]
    

    for col,minimum,maximum in zip(minima.keys(),minima.values(),maxima.values()):
        data_x.loc[:,col] = (data_x.loc[:,col] - minimum) / maximum
    
    if not read_file:
        with open(f"{path}/scaler.pkl","wb") as fp:
            pickle.dump({"minimum":minima, "maximum":maxima},fp)

    return data_x
    
class TabularDataset(Dataset):

    def __init__(self, data):
        feat_cols = list(data.columns)
        feat_cols.remove("new_weights")
        feat_cols.remove("jobstatus")
        
        self.w = data["new_weights"].values[:,None]
        self.y = data["jobstatus"].values[:,None]
        self.x = data[feat_cols].values

    def __len__(self):
        return len(self.w)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.w[idx]
