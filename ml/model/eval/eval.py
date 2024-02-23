import pandas as pd
import seaborn as sns
import glob
import matplotlib.pyplot as plt
import torch
import datetime 
import numpy as np

def confusion(data: pd.DataFrame,
              pred_col: str,
              cut: float = 0.5,
) -> None:
    """Create confusion matrix and save as pdf

    Parameters
    ---
    data: pd.DataFrame
        Data, where the prediction for each data point is included as a extra column

    pred_col: str
        String, determening in which column the prediction of the model is saved
    
    cut: float
        Value, which sets the threshold, when a value is rounded up to 1 (Have to be between 0 and 1)

    Returns
    ---
    None
    """
    
    #Creates a deep copy of the data
    data_conf = data.copy()
    
    #Round the prediction column to 0 or 1
    data_conf.loc[data_conf[pred_col] < cut, pred_col] = 0
    data_conf.loc[data_conf[pred_col] >= cut, pred_col] = 1
    
    #Calculates TP,FP,TN,FN
    true_pos = len(data_conf[(data_conf["jobstatus"]==1) & (data_conf["jobstatus"] == data_conf[pred_col])])
    false_neg = len(data_conf[(data_conf["jobstatus"]==1) & (data_conf["jobstatus"] != data_conf[pred_col])])

    true_neg = len(data_conf[(data_conf["jobstatus"]==0) & (data_conf["jobstatus"] == data_conf[pred_col])])
    false_pos = len(data_conf[(data_conf["jobstatus"]==0) & (data_conf["jobstatus"] != data_conf[pred_col])])
    
    #Create confusion matrix as pandas datarfame
    confusion = pd.DataFrame({"Pred. Finished":np.nan, "Pred. Failed":np.nan}, index = ["True Finished, True Failed"])
    confusion.loc["True Finished","Pred. Finished"] = true_pos / (true_pos + false_neg)
    confusion.loc["True Finished","Pred. Failed"] = false_neg / (true_pos + false_neg)
    confusion.loc["True Failed","Pred. Finished"] = false_pos / (true_neg + false_pos)
    confusion.loc["True Failed","Pred. Failed"] = true_neg / (true_neg + false_pos)
    
    #Create heatmap with seaborn
    sns.heatmap(confusion, annot=True, cmap="viridis")
    plt.savefig("confusion.pdf",bbox_inches="tight")

    return 

def score_hist(data: pd.DataFrame,
               pred_col: str = "prediction"
) -> None:
    """Create the histogram for the output score of the model

    Parameters
    ---
    data: pd.DataFrame
        Data, where the prediction for each data point is included as a extra column

    pred_col: str
        String, determening in which column the prediction of the model is saved
    
    Returns
    ---
    None
    """

    #Create deep copy of the data
    data_copy = data.copy()

    #Turn jobstatus column to string
    data_copy["jobstatus"] = data_copy["jobstatus"].astype("int16").map({0:"Failed",1:"Finished"})
    data_copy = data_copy[data_copy.cpu_eff > 0.05]
    data_copy = data_copy[(data_copy.jobstatus=="Finished") | ((data_copy.new_weights > 10/3 * 0.5) & (data_copy.jobstatus == "Failed"))]

    f,ax = plt.subplots(figsize=(10,5))
    sns.histplot(data=data_copy,x="prediction",hue="jobstatus",bins=20,alpha=0,element="step",stat="probability", ax=ax, common_norm = False)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1.04, 1))
    plt.savefig("score_hist.pdf",bbox_inches="tight")
    
    return

def loss_acc_plot(train: dict[int,float],
                  test: dict[int,float],
                  metric: str,
) -> None:
    """Create line plot for the metrics loss or accuracy for the test and train dataset

    Parameters
    ---
    train: dict{int:float}
        Loss/Accuracy of the train dataset for each epoch
        dict{int:float} -> int describes the epoch and float the loss/accuracy value

    test: dict{int:float}
        Loss/Accuracy of the test dataset for each epoch
        dict{int:float} -> int describes the epoch and float the loss/accuracy value
    
    metric: str
        Describes the metric which is plotted.

    Returns
    ---
    None
    """

    fig = plt.figure()
    plt.plot(train.keys(), train.values(), label = "train",color="blue")
    plt.plot(test.keys(), test.values(), label = "test",color="black")
    plt.xlabel("Epoch")
    plt.ylabel(f"{metric}".capitalize())
    plt.legend()
    plt.savefig(f"{metric}.pdf",bbox_inches='tight')

    return

def main(start_day: int,
         start_month: int,
         start_year: int,
         end_day: int,
         end_month: int,
         end_year: int,
         checkpoint_path: str,
         pred_col: str,
         weight_cut: float = 10/3*0.5,
         round_cut: float = 0.5,
) -> None:

    """Evaluate the performance of the model, by creating loss/accuracy plots, the confusion matrix
       and the score histogram of the model

    Parameters
    ---
    start_day: int
        Day from which to start read the data

    start_month: int
        month from which to start read the data

    start_year: int
        year from which to start read the data

    end_day: int
        Day, to which the data is read

    end_month: int
        month, to which the data is read

    end_year: int
        year, to which the data is read

    checkpoint_path: str
        String, determining the path, where the model checkpoint is stored, with the loss
        and accuracy information for each epoch

    pred_col: str
        string, determining the column name, which is used for the models output prediction 

    Returns
    ---
    None
    """
    
    #Initialize start and end date from which the data is read
    start_date = datetime.date(start_year,start_month,start_day)
    end_date = datetime.date(end_year,end_month,end_day)
    
    #Read the data and concatenate to oen dataframe

    data = []

    print("LOADING DATA...")

    while (start_date <= end_date):

        date_str = str(start_date).replace("-","_")
        f = f"/share/scratch1/kiajueng_yang/data_pred/{date_str}.csv"
        
        #Show warning if file is not there
        if not glob.glob(f):
            print(f"WARNING: FILE {date_str}.csv NOT EXISTENT. FILE WILL BE SKIPPED")
            start_date += datetime.timedelta(days=1)
            continue

        data.append(pd.read_csv(f,usecols=["cpu_eff","jobstatus","prediction","new_weights"]))
        start_date += datetime.timedelta(days=1)
 
    data = pd.concat(data)
    data = data[(data.jobstatus == 1) | ((data.new_weights >= weight_cut) & (data.jobstatus == 0))]

    print("LOADING CHECKPOINT...")

    #Load model checkpoint
    checkpoint = torch.load(checkpoint_path)
    train_loss_acc = checkpoint["train_loss_acc"]
    test_loss_acc = checkpoint["test_loss_acc"]

    #Create plots
    
    print("CREATING PLOTS...")
    
    confusion(data=data,pred_col=pred_col,cut=round_cut)
    score_hist(data=data,pred_col = pred_col)
    loss_acc_plot(train_loss_acc["loss"], test_loss_acc["loss"], "loss")
    loss_acc_plot(train_loss_acc["accuracy"], test_loss_acc["accuracy"], "accuracy")


if __name__=="__main__":
    
    start_day = 1
    start_month = 11
    start_year = 2023

    end_day = 30
    end_month = 11
    end_year = 2023

    checkpoint_path = "/home/kyang/master_grid/ml/model/model/model_checkpoint.tar"
    pred_col = "prediction"
    
    main(start_day = start_day,
         start_month = start_month,
         start_year = start_year,
         end_day = end_day, 
         end_month = end_month, 
         end_year = end_year, 
         checkpoint_path = checkpoint_path,
         pred_col = pred_col,
    )
