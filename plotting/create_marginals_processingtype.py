import plot 
import copy

cfg_cfg = {"io_intensity": {"xlabel":"IO Intensity", "overflow":1e7, "underflow":0},
           "wall_time": {"xlabel":"Walltime", "overflow": 1e5, "underflow":0},
           "diskio": {"xlabel":"Disk IO", "overflow": 10000, "underflow":0},
           "memory_leak": {"xlabel":"Memory Leak", "overflow": 5000, "underflow":-2500},
           "IObytesWriteRate": {"xlabel":"IO Bytes Write Rate", "overflow": 3.5e8, "underflow":0},
           "IObytesReadRate": {"xlabel":"IO Bytes Read Rate", "overflow": 2e8, "underflow":0},
           "IObytesRead": {"xlabel":"IO Bytes Read", "overflow": 1e7, "underflow":0},
           "IObytesWritten": {"xlabel":"IO Bytes Written", "overflow": 5e7, "underflow":0},
           "actualcorecount": {"xlabel":"Core Count", "overflow": 100, "underflow":0},
           "inputfilebytes": {"xlabel":"Input File Bytes", "overflow": 1e11, "underflow":0},
           "cpu_eff": {"xlabel":"CPU Efficiency", "overflow": 1.25, "underflow":0},
}

cfg = { "start_date": "2023_11_01",
        "end_date": "2023_11_30",
        "variables": ["modificationtime","jobstatus","prediction_weights","processingtype"],
        "path": "/share/scratch1/kiajueng_yang/data_pred",
        "bins": 50,
        "ylabel": "Probability",
        "stat": "probability",
        "hue": "processingtype",
        "type": "step",
        "datetime_var": ["modificationtime"],
        "ratio_plot": False,
       }

cfgs = []

for var in cfg_cfg:
    cfg_copy = copy.deepcopy(cfg)
    cfg_copy["hist_name"] = f"{var}_processingtype"
    cfg_copy["xlabel"] = cfg_cfg[var]["xlabel"]
    cfg_copy["x"] = var
    cfg_copy["condition"] = "(data['jobstatus'] == 0) & (data['processingtype'] != 'Test')"
    cfg_copy["underflow"] = cfg_cfg[var]["underflow"]
    cfg_copy["overflow"] = cfg_cfg[var]["overflow"]
    cfg_copy["variables"].append(var)
    cfgs.append(cfg_copy)


histo_plotting = plot.hist_plot(cfgs)
histo_plotting.create_plots()
