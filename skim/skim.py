import pandas as pd
import datetime
import glob
import os
import argparse

#Take Commando line arguments: Day, Month, Year 
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--day', type = str, required=True)
parser.add_argument('-m', '--month', type = str, required=True)
parser.add_argument('-y', '--year', type = str, required=True)
args = parser.parse_args()

#Read the variables to keep from csv file
data_to_keep = pd.read_csv('/home/kyang/master_grid/skim/var_to_keep.csv')['variables'].values

#CSV files to load
files = glob.glob(f'user.swozniew/*{args.year}_{args.month}_{args.day}*') 

#Load skimmed dataframes to list
agg_df = [pd.read_csv(file, usecols=data_to_keep) for file in files]

if len(agg_df) == 0:
    print(f"No files for the {args.day}.{args.month}.{args.year}")
    exit(0)

#Concat the dfs of that certain day into one
skimmed_day_data = pd.concat(agg_df)

#Safe skimmed data files
skimmed_day_data.to_csv(f'/share/scratch1/kiajueng_yang/test/atlas_jobs_enr-{args.year}_{args.month}_{args.day}.csv', index=False)
