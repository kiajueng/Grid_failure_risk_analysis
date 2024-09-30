import datetime
import plot
import copy

cfg_4 = {"start_date" :"2023_08_01",
         "end_date" : "2023_10_31",
         "variables" : ["modificationtime","jobstatus","io_intensity"],
         "title" : "",
         "hist_name" : "io_intensity",
         "bins" :50,
         "ylabel" : "Probability",
         "xlabel": "IO Intensity in [B/s]",
         "xticks" : [],
         "stat": "probability",
         "overflow":1e7,
         "underflow":0,
         "hue" : "jobstatus",
#         "fig_size" : (10,5),
         "x" : "io_intensity",
         "y" : None,
         "x_rotate" : 0,
         "datetime_var" : ["modificationtime"],
         "condition" :"((data['jobstatus'] == 'finished') | (data['jobstatus'] == 'failed'))",
         "type":"step",
         "y_log_scale" : False,
         "x_log_scale" : False,
         "class_color": {"failed":"red","finished":"green"},
         "hue_order":["finished","failed"],
}

cfg_5 = {"start_date" :"2023_08_01",
         "end_date" : "2023_10_31",
         "variables" : ["modificationtime","jobstatus","wall_time"],
         "title" : "",
         "hist_name" : "wall_time",
         "bins" :100,
         "ylabel" : "Probability",
         "xlabel": "Walltime in [s]",
         "xticks" : [],
         "stat": "probability",
         "overflow":1e5,
         "underflow":0,
         "hue" : "jobstatus",
#         "fig_size" : (10,5),
         "x" : "wall_time",
         "y" : None,
         "x_rotate" : 0,
         "datetime_var" : ["modificationtime"],
         "condition" :"((data['jobstatus'] == 'finished') | (data['jobstatus'] == 'failed'))",
         "type":"step",
         "y_log_scale" : False,
         "x_log_scale" : False,
         "class_color": {"failed":"red","finished":"green"},
         "hue_order":["finished","failed"],
}

cfg_6 = {"start_date" :"2023_08_01",
         "end_date" : "2023_10_31",
         "variables" : ["modificationtime","jobstatus","diskio"],
         "title" : "",
         "hist_name" : "diskio",
         "bins" :100,
         "ylabel" : "Probability",
         "xlabel": "Disk IO in [B]",
         "xticks" : [],
         "stat": "probability",
         "overflow":10000,
         "underflow":0,
         "hue" : "jobstatus",
#         "fig_size" : (10,5),
         "x" : "diskio",
         "y" : None,
         "x_rotate" : 0,
         "datetime_var" : ["modificationtime"],
         "condition" :"((data['jobstatus'] == 'finished') | (data['jobstatus'] == 'failed'))",
         "type":"step",
         "y_log_scale" : False,
         "x_log_scale" : False,
         "class_color": {"failed":"red","finished":"green"},
         "hue_order":["finished","failed"],
}

cfg_7 = {"start_date" :"2023_08_01",
         "end_date" : "2023_10_31",
         "variables" : ["modificationtime","jobstatus","memory_leak"],
         "title" : "",
         "hist_name" : "memory_leak",
         "bins" :100,
         "ylabel" : "Probability",
         "xlabel": "Memory Leak in [B]",
         "xticks" : [],
         "stat": "probability",
         "overflow":5000,
         "underflow":-2500,
         "hue" : "jobstatus",
#         "fig_size" : (10,5),
         "x" : "memory_leak",
         "y" : None,
         "x_rotate" : 0,
         "datetime_var" : ["modificationtime"],
         "condition" :"((data['jobstatus'] == 'finished') | (data['jobstatus'] == 'failed'))",
         "type":"step",
         "y_log_scale" : False,
         "x_log_scale" : False,
         "class_color": {"failed":"red","finished":"green"},
         "hue_order":["finished","failed"],
}

cfg_8 = {"start_date" :"2023_08_01",
         "end_date" : "2023_10_31",
         "variables" : ["modificationtime","jobstatus","IObytesWriteRate"],
         "title" : "",
         "hist_name" : "IObytesWriteRate",
         "bins" :100,
         "ylabel" : "Probability",
         "xlabel": "IO Bytes Write Rate in [B/s]",
         "xticks" : [],
         "stat": "probability",
         "overflow":3.5e8,
         "underflow":0,
         "hue" : "jobstatus",
#         "fig_size" : (10,5),
         "x" : "IObytesWriteRate",
         "y" : None,
         "x_rotate" : 0,
         "datetime_var" : ["modificationtime"],
         "condition" :"((data['jobstatus'] == 'finished') | (data['jobstatus'] == 'failed'))",
         "type":"step",
         "y_log_scale" : False,
         "x_log_scale" : False,
         "class_color": {"failed":"red","finished":"green"},
         "hue_order":["finished","failed"],
}

cfg_9 = {"start_date" :"2023_08_01",
         "end_date" : "2023_10_31",
         "variables" : ["modificationtime","jobstatus","IObytesReadRate"],
         "title" : "",
         "hist_name" : "IObytesReadRate",
         "bins" :100,
         "ylabel" : "Probability",
         "xlabel": "IO Bytes Read Rate in [B/s]",
         "xticks" : [],
         "stat": "probability",
         "overflow":2e8,
         "underflow":0,
         "hue" : "jobstatus",
#         "fig_size" : (10,5),
         "x" : "IObytesReadRate",
         "y" : None,
         "x_rotate" : 0,
         "datetime_var" : ["modificationtime"],
         "condition" :"((data['jobstatus'] == 'finished') | (data['jobstatus'] == 'failed'))",
         "type":"step",
         "y_log_scale" : False,
         "x_log_scale" : False,
         "class_color": {"failed":"red","finished":"green"},
         "hue_order":["finished","failed"],
}

cfg_10 = {"start_date" :"2023_08_01",
         "end_date" : "2023_10_31",
         "variables" : ["modificationtime","jobstatus","IObytesRead"],
         "title" : "",
         "hist_name" : "IObytesRead",
         "bins" :100,
         "ylabel" : "Probability",
         "xlabel": "IO Bytes Read in [B]",
         "xticks" : [],
         "stat": "probability",
         "overflow":1e7,
         "underflow":0,
         "hue" : "jobstatus",
#         "fig_size" : (10,5),
         "x" : "IObytesRead",
         "y" : None,
         "x_rotate" : 0,
         "datetime_var" : ["modificationtime"],
         "condition" :"((data['jobstatus'] == 'finished') | (data['jobstatus'] == 'failed'))",
         "type":"step",
         "y_log_scale" : False,
         "x_log_scale" : False,
         "class_color": {"failed":"red","finished":"green"},
         "hue_order":["finished","failed"],
}

cfg_11 = {"start_date" :"2023_08_01",
         "end_date" : "2023_10_31",
         "variables" : ["modificationtime","jobstatus","IObytesWritten"],
         "title" : "",
         "hist_name" : "IObytesWritten",
         "bins" :100,
         "ylabel" : "Probability",
         "xlabel": "IO Bytes Written in [B]",
         "xticks" : [],
         "stat": "probability",
         "overflow":5e7,
         "underflow":0,
         "hue" : "jobstatus",
#         "fig_size" : (10,5),
         "x" : "IObytesWritten",
         "y" : None,
         "x_rotate" : 0,
         "datetime_var" : ["modificationtime"],
         "condition" :"((data['jobstatus'] == 'finished') | (data['jobstatus'] == 'failed'))",
         "type":"step",
         "y_log_scale" : False,
         "x_log_scale" : False,
         "class_color": {"failed":"red","finished":"green"},
         "hue_order":["finished","failed"],
}

cfg_12 = {"start_date" :"2023_08_01",
         "end_date" : "2023_10_31",
         "variables" : ["modificationtime","jobstatus","outputfilebytes"],
         "title" : "",
         "hist_name" : "outputfilebytes",
         "bins" :100,
         "ylabel" : "Probability",
         "xlabel": "Output File Bytes in [B]",
         "xticks" : [],
         "stat": "probability",
         "overflow":1e9,
         "underflow":0,
         "hue" : "jobstatus",
#         "fig_size" : (10,5),
         "x" : "outputfilebytes",
         "y" : None,
         "x_rotate" : 0,
         "datetime_var" : ["modificationtime"],
          "condition" :"((data['jobstatus'] == 'finished') | (data['jobstatus'] == 'failed'))",
         "type":"step",
         "y_log_scale" : False,
         "x_log_scale" : False,
         "class_color": {"failed":"red","finished":"green"},
         "hue_order":["finished","failed"],
}

cfg_13 = {"start_date" :"2023_08_01",
         "end_date" : "2023_10_31",
         "variables" : ["modificationtime","jobstatus","actualcorecount"],
         "title" : "",
         "hist_name" : "actualcorecount",
         "bins" :100,
         "ylabel" : "Probability",
         "xlabel": "Core Count",
         "xticks" : [],
         "stat": "probability",
         "overflow":100,
         "underflow":0,
         "hue" : "jobstatus",
#         "fig_size" : (10,5),
         "x" : "actualcorecount",
         "y" : None,
         "x_rotate" : 0,
         "datetime_var" : ["modificationtime"],
         "condition" :"((data['jobstatus'] == 'finished') | (data['jobstatus'] == 'failed'))",
         "type":"step",
         "y_log_scale" : False,
         "x_log_scale" : False,
         "class_color": {"failed":"red","finished":"green"},
         "hue_order":["finished","failed"],
}

cfg_14 = {"start_date" :"2023_08_01",
         "end_date" : "2023_10_31",
         "variables" : ["modificationtime","jobstatus","inputfilebytes"],
         "title" : "",
         "hist_name" : "inputfilebytes",
         "bins" :100,
         "ylabel" : "Probability",
         "xlabel": "Input File Bytes in [B]",
         "xticks" : [],
         "stat": "probability",
         "overflow":1e11,
         "underflow":0,
         "hue" : "jobstatus",
#         "fig_size" : (10,5),
         "x" : "inputfilebytes",
         "y" : None,
         "x_rotate" : 0,
         "datetime_var" : ["modificationtime"],
          "condition" :"((data['jobstatus'] == 'finished') | (data['jobstatus'] == 'failed'))",
         "type":"step",
         "y_log_scale" : False,
         "x_log_scale" : False,
         "class_color": {"failed":"red","finished":"green"},
         "hue_order":["finished","failed"],
}

cfg_15 = {"start_date" :"2023_08_01",
         "end_date" : "2023_10_31",
         "variables" : ["modificationtime","jobstatus","cpu_eff"],
         "title" : "",
         "hist_name" : "cpu_eff",
         "bins" :100,
         "ylabel" : "Probability",
         "xlabel": "CPU Efficiency",
         "xticks" : [],
         "stat": "probability",
         "overflow":1.25,
         "underflow":0,
         "hue" : "jobstatus",
 #        "fig_size" : (10,5),
         "x" : "cpu_eff",
         "y" : None,
         "x_rotate" : 0,
         "datetime_var" : ["modificationtime"],
         "condition" :"((data['jobstatus'] == 'finished') | (data['jobstatus'] == 'failed'))",
         "type":"step",
         "y_log_scale" : False,
         "x_log_scale" : False,
         "class_color": {"failed":"red","finished":"green"},
         "hue_order":["finished","failed"],
}

cfg_16 = {"start_date" :"2023_08_01",
         "end_date" : "2023_10_31",
         "variables" : ["modificationtime","jobstatus","cpuconsumptiontime"],
         "title" : "",
         "hist_name" : "cpuconsumptiontime",
         "bins" :100,
         "ylabel" : "Probability",
         "xlabel": "CPU Consumption Time in [s]",
         "xticks" : [],
         "stat": "probability",
         "overflow":15000,
         "underflow":0,
         "hue" : "jobstatus",
#         "fig_size" : (10,5),
         "x" : "cpuconsumptiontime",
         "y" : None,
         "x_rotate" : 0,
         "datetime_var" : ["modificationtime"],
         "condition" :"((data['jobstatus'] == 'finished') | (data['jobstatus'] == 'failed'))",
         "type":"step",
         "y_log_scale" : False,
         "x_log_scale" : False,
         "class_color": {"failed":"red","finished":"green"},
         "hue_order":["finished","failed"],
}



cfgs = [cfg_4,cfg_5,cfg_6,cfg_7,cfg_8,cfg_9,cfg_10,cfg_11,cfg_12,cfg_13,cfg_14,cfg_15,cfg_16]
histo_plotting = plot.hist_plot(cfgs)
histo_plotting.create_plots()
