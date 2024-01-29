import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import datetime

"""
example_conifg = {start_date: Start date to read the data from (Last modification time), in format YYYY_MM_DD, (string)
                  end_date: End date to read the data from (Last modification time), in format YYYY_MM_DD, (string)
                  variables: Specifies which columns to keep (For condition and for plotting) --> Saves memory (List of strings)
                  title: Title of the histogram (String),
                  hist_name: Name of output pdf (string),
                  fig_size: Size of the output figure in (width,height) format (tuple of floats),
                  bins: Number of bins (int) or location of bin edges (list) or "auto" -> bins set automatically by seaborn (string),
                  ylabel: Label of y axis (string),
                  xlabel: Label of x axis (string),
                  xticks: xticks for the histogram (List of floats), have to be set to empty list if no xticks want to be specified,
                  normalize: Decision, wheter to normalize the histogram or not (Boolean),
                  hue: Column name, whose unique values are used as class color for stacked histo (string) or None if no classes (None-type),
                  x: Column name, used to plot on the x-axis (string),
                  y: Column name, used to plot on the y-axis for 2D histograms (string). Set to None as default if only 1D histograms desired (None-type)
                  x_rotate: Degree, to define how much the x-labels are rotated (float),
                  datetime_var: Variables which have to turned into datetime (list),
                  condition: String of conditions, transformed later to expression (USE data[variable] instead of variable) (string),
                  type: Type of plot, choose either 'stack', '2D', or 'step' (string) 
                  y_log_scale: Boolean, to decide whether we want to scale the y-axis logarithmic or not (bool)
                  x_log_scale: Boolean, to decide whether we want to scale the x-axis logarithmic or not (bool)
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

    def group_cfg(self, configs):
        """
        Create config groups, whose dates do not overlap to not load data twice

        :param configs: List of configs from which plots is to be created
        :return: Dictionary, with tuple as key (start_date,end_date) and list as value [configs belonging to one group]
        """

        # Initialize dictionary to put the grouped configs in
        cfg_groups = {}

        for i, config in enumerate(configs):

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

    def load_data(self, date_range, variables, dt_var):
        """
        Load the data from the files specified in config and concatenate into one pandas dataframe

        :param date_range: Date range from which days to load the data from
        :param variables: List with variables, defining which columns will be loaded
        :return pd.concat(data): Returns all the csv files concatenated into one pandas dataframe.
        """
        data = []
        start = date_range[0]
        end = date_range[1]

        while start <= end:
            # Load file into pandas dataframe and append to data
            date = str(start).replace("-", "_")
            data.append(pd.read_csv(f"/share/scratch1/es-atlas/atlas_jobs_enr_skimmed/atlas_jobs_enr-{date}.csv",usecols=variables))
            start += datetime.timedelta(days=1)

        if len(data) == 0:
            raise FileNotFoundError(
                "No files found to load into pandas dataframe!")
        
        #Concatenate all days and tranform datetime columns to datetime objects
        concat_data = pd.concat(data) if len(data) > 1 else data[0]
        concat_data[dt_var] = concat_data[dt_var].apply(pd.to_datetime, errors="coerce")

        return concat_data

    def plot(self, cfg, data):
        """
        Create and save the histogram specified by the config

        :param cfg: Dictionary with the config for the plot
        :param data: Data, used for the plotting
        :return:
        """
        # Filter data with condition
        f_data = data.loc[eval(cfg["condition"])].reset_index(drop=True)
        # Now use the filtered data to create the histogram
        f, ax = plt.subplots(figsize=cfg["fig_size"])
        sns.despine(f)  # Remove top and right spine of histogram
        if ((cfg["normalize"]) & (cfg["type"] == "step")):
            ax = sns.histplot(data=f_data,
                              x=cfg["x"],
                              element="step",
                              hue=cfg["hue"],
                              bins=cfg["bins"],
                              alpha=0,
                              ax=ax,
                              stat="probability",
                              common_norm=False,
            )
        elif ((not cfg["normalize"]) & (cfg["type"] == "step")):
            ax = sns.histplot(data=f_data,
                              x=cfg["x"],
                              element="step",
                              hue=cfg["hue"],
                              bins=cfg["bins"],
                              alpha=0,
                              ax=ax,
            )
        elif ((cfg["type"] == "stack") & (cfg["normalize"])):
            ax = sns.histplot(data=f_data,
                              x=cfg["x"],
                              multiple="stack",
                              hue=cfg["hue"],
                              bins=cfg["bins"],
                              stat="probability",
                              ax=ax,
            )
        elif ((cfg["type"] == "stack") & (not cfg["normalize"])):
            ax = sns.histplot(data=f_data,
                              x=cfg["x"],
                              multiple="stack",
                              hue=cfg["hue"],
                              bins=cfg["bins"],
                              ax=ax,
            )
        elif ((cfg["type"] == "2D") & (not cfg["normalize"])):
            ax = sns.histplot(data=f_data,
                              x=cfg["x"],
                              y=cfg["y"],
                              bins=cfg["bins"],
                              cbar=True,
                              ax=ax,
            )
        elif ((cfg["type"] == "2D") & (cfg["normalize"])):
            ax = sns.histplot(data=f_data,
                              x=cfg["x"],
                              y=cfg["y"],
                              bins=cfg["bins"],
                              cbar=True,
                              stat="probability",
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

    def create_plots(self):
        
        # First create configs dict, which collects all cfgs for each non
        # overlapping date_range
        cfgs_dict = self.group_cfg(self.cfgs)

        # Create plots for all the date_ranges
        for date_range in cfgs_dict:

            # Load data for all the variables used in the cfgs for each
            # date_range
            variables = self.cfgs_to_variables(cfgs=cfgs_dict[date_range],option="variables")
            dt_var = self.cfgs_to_variables(cfgs=cfgs_dict[date_range],option="datetime_var")
            
            #Make sure that modificationtime is always loaded and also transformed to datetime
            if ("modificationtime" not in variables):
                variables += ["modificationtime"]
                
            if ("modificationtime" not in dt_var):
                dt_var += ["modificationtime"]
                
            data = self.load_data(date_range=date_range, variables=variables, dt_var=dt_var)
            
            # Create plots for each config
            for cfg in cfgs_dict[date_range]:
                self.plot(cfg, data)
