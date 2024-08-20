import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import datetime
from typing import Union
import copy 
import numpy as np 
"""
example_conifg = {start_date: Start date to read the data from (Last modification time), in format YYYY_MM_DD, (string) *
                  end_date: End date to read the data from (Last modification time), in format YYYY_MM_DD, (string) *
                  variables: Specifies which columns to keep (For condition and for plotting) --> Saves memory (List of strings) *
                  path: Specifies from which path to read the csv files. Must contain the date in its file names. (string)
                  files: Custom files to load the data from. Also needs the paths to the files, not impacted by path option (List of Strings)
                  title: Title of the histogram (String),
                  hist_name: Name of output pdf (string), *
                  fig_size: Size of the output figure in (width,height) format (tuple of floats),
                  bins: Number of bins (int) or location of bin edges (list) or "auto" -> bins set automatically by seaborn (string),
                  ylabel: Label of y axis (string), *
                  xlabel: Label of x axis (string), *
                  xticks: xticks for the histogram (List of floats), have to be set to empty list if no xticks want to be specified,
                  stat: Aggregate statistic to compute in each bin (probability, count, percent, density) (string),
                  hue: Column name, whose unique values are used as class color for stacked histo (string) or None if no classes (None-type),
                  x: Column name, used to plot on the x-axis (string), *
                  y: Column name, used to plot on the y-axis for 2D histograms (string). Set to None as default if only 1D histograms desired (None-type)
                  x_rotate: Degree, to define how much the x-labels are rotated (float),
                  datetime_var: Variables which have to turned into datetime (list), *
                  condition: String of conditions, transformed later to expression (USE data[variable] instead of variable) (string),
                  type: Type of plot, choose either 'stack', '2D', or 'step' (string) 
                  y_log_scale: Boolean, to decide whether we want to scale the y-axis logarithmic or not (bool)
                  x_log_scale: Boolean, to decide whether we want to scale the x-axis logarithmic or not (bool)
                  class_color: Dictionary, giving each class an color {class:color} (Dictionary). If no color chosen, set the default value to None.
                  hue_order: List of strings, giving the order of the classes, used to create the legend (List of strings). Default: None
                  overflow: Float, threshold where every value above is projected down to overflow, to create overflow bins of x variable (Default: np.inf) (Float)
                  underflow: Float, threshold where every value below is projected down to overflow, to create underflow bins of x variable (Default: -np.inf) (Float)
                  hue_mapping: Tuple, with the type as string and the mapping as dictionary, to map the column used for hue to other values and potentially other type (str,dict)
                  hue_ratio_class: String, defining which is the class of the hue, for which the ratio is computed (Default: None)
                  ratio_plot: Boolean, deciding if along the histogram a ratio plot is also created (Default: False)
                  }
"""

            
    
class hist_plot():

    def __init__(self, cfgs):
        """
        Initialize hist_plot, which creates histograms based on a dictionary config

        :param cfgs: The config file from which to read
        :return:
        """
        self.cfgs = cfgs

    def cfg_setter(self,cfg: dict[str,Union[datetime.date,list[str], float, bool, None, tuple[float,float]]]
    ) -> dict[str,Union[datetime.date,list[str], float, bool, None, tuple[float,float]]]:
        """ Set default values and check if mandatory options are set
    
        Parameters
        ---
        cfg: dict
            Dictionary, with all the options for the plot, which needs to be created

        Returns
        ---
        cfg: dict
            Dictionary, where every value is checked.
        """

        #Mandatory options
        mand_opt = ["start_date",
                    "end_date",
                    "variables",
                    "hist_name",
                    "ylabel",
                    "xlabel",
                    "x",
                    "datetime_var",
        ]
        
        #Defaultet options, if not specified
        default_opt = {"title":None,
                       "path":"/share/scratch1/es-atlas/atlas_jobs_enr_skimmed/",
                       "files":[],
                       "fig_size":(8,6),
                       "bins":"auto",
                       "xticks":[],
                       "stat":"count",
                       "hue":None,
                       "y":None,
                       "x_rotate":0,
                       "condition":"",
                       "type":"stack",
                       "y_log_scale":False,
                       "x_log_scale":False,
                       "class_color":None,
                       "hue_order":None,
                       "hue_mapping":(),
                       "overflow": np.inf,
                       "underflow": -np.inf,
                       "hue_ratio_class":None,
                       "ratio_plot":False,
        }

        #Options set in config file
        cfg_opt = list(cfg.keys())
    
        for opt in mand_opt:
            if opt not in cfg_opt:
                raise KeyError(
                    f"The Option {opt} is not set in the config"
                )

        for opt in default_opt:
            if opt not in cfg_opt:
                cfg[opt] = default_opt[opt]

        if cfg["path"][-1] != "/":
            cfg["path"] += "/"


        return cfg

    def group_cfg_path(self, cfgs):
        """
        Sort config files according to their path, leaving out the configs with custom files

        :param cfgs: The config file from which to read
        :return cfgs_dict: Dictionary with key being the path of the configs and as value a list of configs with the same path
        """

        cfgs_dict = {}
        
        for cfg in cfgs:
            
            if cfg["files"]:
                continue
            
            if cfg["path"] not in cfgs_dict.keys():
                cfgs_dict[cfg["path"]] = [cfg]
            else:
                cfgs_dict[cfg["path"]].append(cfg)
            
        return cfgs_dict
                
    def check_overlap(self, date_range_1, date_range_2):
        """
        Checks if two date ranges have overlap

        :param date_range_1: First date range as tuple (start,end)
        :param date_range_2: Second date range as tuple (start,end)
        :return check: Returns Boolean, indicating if there is an overlap
        """

        latest_start = max(date_range_1[0], date_range_2[0])
        earliest_end = min(date_range_1[1], date_range_2[1])
        delta = (earliest_end - latest_start).days + 1
        overlap = max(0, delta)

        if overlap > 0:
            return True
        else:
            return False

    def list_to_date_range(self, list_date_range):
        """
        Find minimal starting and maximal ending date from a list of date ranges

        :param list_date_range: List of date ranges [(start_date,end_date),...]
        :return (start_date,end_date): Return tuple with start date being the earliest start date of all in the list and respectively for enddate (latest)
        """

        start_date = min([dates[0] for dates in list_date_range])
        end_date = max([dates[1] for dates in list_date_range])
        return (start_date, end_date)

    def group_cfg_date(self, configs):
        """
        Create config groups, whose dates do not overlap to not load data twice

        :param configs: List of configs from which plots is to be created
        :return: Dictionary, with tuple as key (start_date,end_date) and list as value [configs belonging to one group]
        """

        # Initialize dictionary to put the grouped configs in
        cfg_groups = {}

        for i, config in enumerate(configs):
            
            if config["files"]:
                continue
            
            # Convert start and end date of config to date object
            start = datetime.datetime.strptime(
                config["start_date"], "%Y_%m_%d").date()
            end = datetime.datetime.strptime(
                config["end_date"], "%Y_%m_%d").date()

            # Add condition that modificationtime is larger than start and
            # smaller than end time, since the files are ordered after their
            # modification time
            if config["condition"] == "":
                config["condition"] += f'(data["modificationtime"] > pd.to_datetime(datetime.datetime.strptime("{config["start_date"]}","%Y_%m_%d").date())) & (data["modificationtime"] < pd.to_datetime(datetime.datetime.strptime("{config["end_date"]}","%Y_%m_%d").date()))'

            else:
                config["condition"] += f' & (data["modificationtime"] > pd.to_datetime(datetime.datetime.strptime("{config["start_date"]}","%Y_%m_%d").date())) & (data["modificationtime"] < pd.to_datetime(datetime.datetime.strptime("{config["end_date"]}","%Y_%m_%d").date()))'

            date_range_config = (start, end)

            # If i=0, then add this as first daterange interval to cfg_groups
            if i == 0:
                cfg_groups[date_range_config] = [config]
                continue

            # Create list to input all keys (date_ranges), that has to be
            # converted into one, due to overlap
            to_be_combined_dates = []
            to_be_combined_configs = []
            
            # Check if date_ranges in cfg groups overlaps with the config one
            for date_range in cfg_groups:
                if self.check_overlap(date_range_config, date_range):
                    # If date range of config overlaps with multiple date
                    # ranges of the cfg groups, then put all of them in one
                    # list
                    to_be_combined_dates.append(date_range)
            
            for date_range in to_be_combined_dates:
                to_be_combined_configs += cfg_groups.pop(date_range)
                
            
            to_be_combined_dates.append(date_range_config)
            to_be_combined_configs.append(config)
                
            new_key = self.list_to_date_range(to_be_combined_dates)
            cfg_groups[new_key] = to_be_combined_configs

        return cfg_groups

    def cfgs_to_variables(self, cfgs, option="variables"):
        """
        Takes list of configs as input and return all variables needed for the list of configs
        
        :param option: String, defining to either return a unique list of variables or the unique list of date time variables which needs to be transformed
        :param cfgs: List of config files
        :return variables: List of (date time)-variables needed for the plotting
        """

        variables = []

        for cfg in cfgs:
            variables += cfg[option]

        return list(set(variables))

    def load_data(self, variables, dt_var, path=None, date_range=None, cfg=None):
        """
        Load the data from the files specified in config and concatenate into one pandas dataframe

        :param date_range: Date range from which days to load the data from
        :param variables: List with variables, defining which columns will be loaded
        :return pd.concat(data): Returns all the csv files concatenated into one pandas dataframe.
        """
        data = []
        
        #Check if custom files were given
        if cfg != None:
            for files in cfg["files"]:
                for f in glob.glob(files):
                    data.append(pd.read_csv(f),usecols=variables)
            
            if len(data) == 0:
                raise FileNotFoundError(
                    "No files found to load into pandas dataframe!")

            
            concat_data = pd.concat(data) if len(data) > 1 else data[0]
            
            for var in dt_var:
                concat_data[var] = pd.to_datetime(concat_data[var], errors="coerce")

            return concat_data

        start = date_range[0]
        end = date_range[1]
        
        while start <= end:
            # Load file into pandas dataframe and append to data
            date = str(start).replace("-", "_")
            for f in glob.glob(f"{path}*{date}*"):
                data.append(pd.read_csv(f,usecols=variables))
            start += datetime.timedelta(days=1)

        if len(data) == 0:
            raise FileNotFoundError(
                "No files found to load into pandas dataframe!")
        
        #Concatenate all days and tranform datetime columns to datetime objects
        concat_data = pd.concat(data) if len(data) > 1 else data[0]

        for var in dt_var:
            concat_data[var] = pd.to_datetime(concat_data[var], errors="coerce")

        return concat_data

    def ratio_plot(self, cfg, data):
        """
        Create and save the ratio histogram specified by the config for a specific hue

        :param cfg: Dictionary with the config for the plot
        :param data: Data, used for the plotting
        :return:
        """
        # Filter data with condition
        f_data = data.copy()
        f_data["ratio_weights"] = 1.0
        f_data = f_data.loc[eval(cfg["condition"])].reset_index(drop=True)
        f_data[cfg["x"]] = f_data[cfg["x"]].clip(cfg["underflow"],cfg["overflow"])
        
        #Map classes of hue to other value with potentially other type
        if cfg["hue_mapping"]:
            f_data[cfg["hue"]] = f_data[cfg["hue"]].map(cfg["hue_mapping"][1])
            f_data[cfg["hue"]] = f_data[cfg["hue"]].astype(cfg["hue_mapping"][0])

        histo_data = f_data.loc[f_data[cfg["hue"]] == cfg["hue_ratio_class"]]
        
        bins,bin_edges = np.histogram(histo_data[cfg["x"]].values, bins = cfg["bins"], density=False)

        for i in range(len(bin_edges) - 1):
            
            bin_left = bin_edges[i]
            bin_right = bin_edges[i + 1]

            if i < len(bin_edges)-2:
                f_data.loc[((f_data[cfg["x"]] >= bin_left) & (f_data[cfg["x"]] < bin_right)),["ratio_weights"]] /= bins[i]

            else:
                f_data.loc[((f_data[cfg["x"]] >= bin_left) & (f_data[cfg["x"]] <= bin_right)),["ratio_weights"]] /= bins[i]

        f, ax = plt.subplots(figsize=cfg["fig_size"])
        ax = sns.histplot(data=f_data,
                          x=cfg["x"],
                          element="step",
                          hue=cfg["hue"],
                          bins=cfg["bins"],
                          alpha=0,
                          ax=ax,
                          weights="ratio_weights",
                          palette=cfg["class_color"],
                          hue_order=cfg["hue_order"],
        )
        # Set xticks for the histogram
        if cfg["xticks"]:
            ax.set_xticks(cfg["xticks"])

        if cfg["y_log_scale"]:
            ax.set_yscale("log")

        if cfg["x_log_scale"]:
            ax.set_xscale("log")

        # Set x and y label for histogram
        ax.set(xlabel=cfg["xlabel"], ylabel=cfg["ylabel"])
        ax.set_title(cfg["title"])

        plt.xticks(rotation=cfg["x_rotate"])

        ax.get_legend().set_visible(False)
        # Save histogram as .pdf file
        plt.savefig(cfg["hist_name"] + "ratio_plot" + ".pdf",bbox_inches='tight')
        
        #Close figure again
        plt.close(f)

    
    def plot(self, cfg, data):
        """
        Create and save the histogram specified by the config

        :param cfg: Dictionary with the config for the plot
        :param data: Data, used for the plotting
        :return:
        """
        # Filter data with condition
        f_data = data.copy()
        f_data = f_data.loc[eval(cfg["condition"])].reset_index(drop=True)
        f_data[cfg["x"]] = f_data[cfg["x"]].clip(cfg["underflow"],cfg["overflow"])

        #Map classes of hue to other value with potentially other type
        if cfg["hue_mapping"]:
            f_data[cfg["hue"]] = f_data[cfg["hue"]].map(cfg["hue_mapping"][1])
            f_data[cfg["hue"]] = f_data[cfg["hue"]].astype(cfg["hue_mapping"][0])

        # Now use the filtered data to create the histogram
        f, ax = plt.subplots(figsize=cfg["fig_size"])
        sns.despine(f)  # Remove top and right spine of histogram
        if (cfg["type"] == "step"):
            ax = sns.histplot(data=f_data,
                              x=cfg["x"],
                              element="step",
                              hue=cfg["hue"],
                              bins=cfg["bins"],
                              alpha=0,
                              ax=ax,
                              stat=cfg["stat"],
                              common_norm=False,
                              palette=cfg["class_color"],
                              hue_order=cfg["hue_order"],
            )
        elif (cfg["type"] == "stack"):
            ax = sns.histplot(data=f_data,
                              x=cfg["x"],
                              multiple="stack",
                              hue=cfg["hue"],
                              bins=cfg["bins"],
                              stat=cfg["stat"],
                              ax=ax,
                              palette=cfg["class_color"],
                              hue_order=cfg["hue_order"],
            )
        elif (cfg["type"] == "2D"):
            ax = sns.histplot(data=f_data,
                              x=cfg["x"],
                              y=cfg["y"],
                              bins=cfg["bins"],
                              cbar=True,
                              stat=cfg["stat"],
                              ax=ax,
            )
        else:
            assert False, "No valid plotting option"
        # Set xticks for the histogram
        if cfg["xticks"]:
            ax.set_xticks(cfg["xticks"])

        if cfg["y_log_scale"]:
            ax.set_yscale("log")

        if cfg["x_log_scale"]:
            ax.set_xscale("log")

        # Set x and y label for histogram
        ax.set(xlabel=cfg["xlabel"], ylabel=cfg["ylabel"])
        ax.set_title(cfg["title"])

        plt.xticks(rotation=cfg["x_rotate"])

        # Save histogram as .pdf file
        plt.savefig(cfg["hist_name"] + ".pdf",bbox_inches='tight')
        
        #Close figure again
        plt.close(f)
        
    def create_plots(self):

        for i in range(len(self.cfgs)):
            self.cfg_setter(self.cfgs[i])
            
        #Plot every config with custom files
        for cfg in self.cfgs:
            if cfg["files"]:
                print(f'LOAD DATA OF FILES: {cfg["files"]}')
                data = self.load_data(variables=cfg["variables"],dt_var=cfg["datetime_var"],cfg=cfg)
                self.plot(cfg,data)
                if cfg["ratio_plot"]:
                    self.ratio_plot(cfg,data)

        #Sort all configs by path, if cfg["files"] has files specified, they are thrown out
        cfgs_dict_path = self.group_cfg_path(self.cfgs)


        cfgs_dict_date = {}
        # Create configs dicts, which collects all cfgs for each non
        # overlapping date_range for every path
        for path in cfgs_dict_path:
            cfgs_dict_date[path] = self.group_cfg_date(cfgs_dict_path[path])

        # Create plots for all the different paths and date_ranges
        for path in cfgs_dict_date:
            for date_range in cfgs_dict_date[path]:
                
                print(f"LOAD DATA OF PATH {path} for the date range {date_range}")

                # Load data for all the variables used in the cfgs for each
                # date_range
                variables = self.cfgs_to_variables(cfgs=cfgs_dict_date[path][date_range],option="variables")
                dt_var = self.cfgs_to_variables(cfgs=cfgs_dict_date[path][date_range],option="datetime_var")
            
                #Make sure that modificationtime is always loaded and also transformed to datetime
                if ("modificationtime" not in variables):
                    variables += ["modificationtime"]
                
                if ("modificationtime" not in dt_var):
                    dt_var += ["modificationtime"]
                
                data = self.load_data(variables=variables, dt_var=dt_var,date_range=date_range, path=path)
            
                # Create plots for each config
                for cfg in cfgs_dict_date[path][date_range]:
                    self.plot(cfg, data)
                    if cfg["ratio_plot"]:
                        self.ratio_plot(cfg,data)
