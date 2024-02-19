import pandas as pd
import datetime
import argparse

parser= argparse.ArgumentParser()
parser.add_argument('-f', '--file', type=str, required=True)
args = parser.parse_args()


#Data to be read
data = pd.read_csv(args.file)

#Remove all columns with Unnamed in the name
columns = [column for column in data.columns if "Unnamed" in column]
data.drop(columns,axis=1,inplace=True)

#Create a new column bintime
data["bintime"] = pd.to_datetime(data["modificationtime"]).dt.floor("1h")

#Get the counts, grouped by bintime, computingsite and jobstatus
counts = data.groupby(["bintime","computingsite","jobstatus"]).count()

def set_weight(row, counts):
    
    if ((row["jobstatus"] != "failed") & (row["jobstatus"] != "finished")):
        return 0.
    
    try:
        N_failed = counts["modificationtime"][row["bintime"],row["computingsite"],"failed"]
    except KeyError:
        N_failed = 0
        
    try:
        N_finished = counts["modificationtime"][row["bintime"],row["computingsite"],"finished"]
    except KeyError:
        N_finished = 0
        
    N_total = N_failed + N_finished
    
    return N_total / counts["modificationtime"][row["bintime"],row["computingsite"],row["jobstatus"]] * 0.5
    
#Create new column with the weights
data["new_weights"] = data.apply(set_weight, counts=counts, axis=1)

#Drop the bintime as it is not needed anymore
data.drop(["bintime"],axis=1,inplace=True)

#Save the file and overwrite the old one
data.to_csv(args.file, index=False)

#Write in the weights_added.txt file if the weights have been added successfully
f = open("/home/kyang/master_grid/ml/weights/weights_added.txt", "a")
f.write(f"{args.file}\n")
f.close()
