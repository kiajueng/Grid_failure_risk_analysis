import glob
import dask.dataframe as dd
import datetime
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import pickle

def load_data(cols,start_date, end_date):

    ddf = []

    while start_date <= end_date:
        
        date_str = str(start_date).replace("-","_")
        
        data = dd.read_csv(f"/share/scratch1/kiajueng_yang/data/{date_str}.csv", usecols=cols)
        data =  data[(data["cpu_eff"] > 0.05)]
        ddf.append(data)

        start_date += datetime.timedelta(days=1)
        
    ddf = dd.concat(ddf)
    return ddf

def train_test_split(split,cols,seed,start_date,end_date):

    data = load_data(cols,start_date,end_date)
    data_fail = data[data.jobstatus == 0]
    data_fin = data[data.jobstatus == 1]
    
    train_fail, test_fail= data_fail.random_split(split,shuffle=True, random_state = seed)
    train_fin, test_fin = data_fin.random_split(split,shuffle=True, random_state = seed)

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

    def __init__(self, data, features, index=False, glob_weight=None):

        self.return_index = index
        
        if (self.return_index & ("index" in data.columns)):
            self.index = data["index"].values[:,None]
        elif (self.return_index & ("index" not in data.columns)):
            self.index = data.reset_index()["index"].values[:,None]
    
        self.glob_weight = glob_weight

        self.w = data["new_weights"].values[:,None]
        self.y = data["jobstatus"].values[:,None]
        self.x = data[features].values

    def __len__(self):
        return len(self.w)

    def __getitem__(self, idx):

        if self.return_index:
            return self.index[idx], self.x[idx], self.y[idx], self.w[idx]
            
        if self.glob_weight != None:
            return self.x[idx], self.y[idx], self.glob_weight


        return self.x[idx], self.y[idx], self.w[idx]
