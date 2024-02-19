import dask.dataframe as dd
import pandas as pd
import datetime 
import glob 
import sys 

def load_data(cols: list[str],
              date: datetime.date,
) -> dd.DataFrame:
    """Load data for given day
    
    Parameters
    ---
    cols: list[str]
        List of strings, where each string represent one column name, which is loaded
    
    date: datetime.date
        Datetime.date object, initiating the Date of the data, which is loaded.

    Returns
    ---
    dd.DataFrame
        Loaded data from date with the coloumns cols.
    """
    
    #Turn date into string object, with form YYYY_MM_DD
    date_str = str(date).replace("-","_")
    
    #Initialize empty dataframe 
    data = None
    
    if (len(glob.glob(f"/share/scratch1/es-atlas/atlas_jobs_enr_skimmed/*{date_str}*")) == 1):
          f = glob.glob(f"/share/scratch1/es-atlas/atlas_jobs_enr_skimmed/*{date_str}*")[0]
    else:
          raise Exception("The number of files is unequal to 1")
    
    data = dd.read_csv(f, usecols=cols)

    return data 

def clean_data(data: dd.DataFrame,
) -> dd.DataFrame:
    """Clean the data from nans, delete all rows which is not failed or finished and
       turn "jobstatus" to 0 or 1 as float64 so it can be directly used during the training

    Parameters
    ---
    data: dd.DataFrame
       Dask dataframe, from which the nan columns are removes and jobstatus casted to float64

    Returns
    ---
    dd.DataFrame
        Dask dataframe, where nans are removed and jobstatus column casted to 0/1
    """

    #Drop all rows where the jobstatus is not failed or finished
    data = data[(data.jobstatus == "failed") | (data.jobstatus == "finished")]

    #Drop rows with nans
    data = data.dropna()
    
    #Cast failed and finished to 0/1 of type float64
    data["jobstatus"] = data["jobstatus"].map({"failed":0, "finished":1})
    data = data.astype("float64")

    return data

def run_preprocessing(cols:list[str],
                      date: datetime.date,
) -> None:
    """Load data from the atlas_jobs_enr_skimmed directory and bring into a form, such that
       it can be directly used for training and save it into 
       /share/scratch1/kiajueng.yang/data directory

    Parameters
    ---
    cols: list[str]
        List of strings, where each string represent one column name, which is loaded
    
    date: datetime.date
        Datetime.date object, initiating the Date of the data, which is loaded.

    Returns
    ---
    None
       Return nothing, since the task is to save it into new csv file
    """
    
    #Load Data
    data = load_data(cols = cols, date = date)
    
    #Clean Data
    data = clean_data(data)

    #Save data with name being the date
    date_str = str(date).replace("-","_")
    
    #Check if file already exist
    if len(glob.glob(f"/share/scratch1/kiajueng_yang/data/{date_str}.csv")) > 0:
        print(f"{date_str}.csv already exist.")
        sys.exit(0)

    data.to_csv(f"/share/scratch1/kiajueng_yang/data/{date_str}.csv", index=False, single_file=True)
    
    return

if __name__=="__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--day", type=int, required=True)
    parser.add_argument("--month", type=int, required=True)
    parser.add_argument("--year", type=int, required=True)
    args = parser.parse_args()

    date = datetime.date(args.year,args.month,args.day)
    cols = ["io_intensity","wall_time","diskio","memory_leak","IObytesWriteRate", "IObytesReadRate","IObytesRead","IObytesWritten","outputfilebytes","actualcorecount","inputfilebytes","cpu_eff", "cpuconsumptiontime","new_weights", "jobstatus"]
    
    run_preprocessing(cols=cols, date=date)

    with open("/home/kyang/master_grid/ml/model/preprocessing/done.txt", "a") as f:
        f.write(f"{str(date).replace('-','_')}\n")
