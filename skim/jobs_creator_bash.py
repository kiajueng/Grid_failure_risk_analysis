import datetime
import os 

def create_skim_script(number,year,month,day):
    """Function to create skimming script for one day of data
    
    :param number: Process number used in HTCondor submission file 
    :param year: Year, in which data is loaded
    :param month: Month, in which data is loaded
    :param day: Day, in which data is loaded
    """
    
    #Create directory on scratch if not already exist
    f = open(f"condor_skim/skim_{number}.sh", "w", encoding="utf-8")
    
    f.write("#!/bin/bash\n")
    f.write("\n")
    
    f.write("source /cvmfs/sft.cern.ch/lcg/views/LCG_95/x86_64-centos7-gcc8-opt/setup.sh\n")
    f.write("source /cvmfs/grid.cern.ch/umd-c7ui-latest/etc/profile.d/setup-c7-ui-example.sh\n")
    f.write("export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase\n")
    f.write("export RUCIO_ACCOUNT=kyang\n")
    f.write("source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh\n")
    f.write("source /cvmfs/atlas.cern.ch/repo/sw/tdaq/tools/cmake_tdaq/bin/cm_setup.sh tdaq-09-04-00\n")
    f.write("localSetupRucioClients\n")
    f.write("\n")

    #Set date as variables, used to load data from rucio
    f.write(f"year='{year}'\n")
    f.write(f"month='{month}'\n")
    f.write(f"day='{day}'\n")
    f.write("\n")
    
    #Load data for certain date from rucio
    f.write('rucio download $(rucio list-files user.swozniew:atlas_jobs_enr | grep "${year}_${month}_${day}" | '+"awk '{ print $2; }')\n")
    f.write("\n")
    
    #Execute python script and exit with exit code of python script
    f.write(f"python3 /home/kyang/master/skim/skim.py --day $day --month $month --year $year\n") 
    f.write("python_exit_code=$? #Get the exit code of the python script -> In HTCondor exit code not passed on properly\n")
    f.write("exit $python_exit_code\n")
    f.close()
    
    os.chmod(f"condor_skim/skim_{number}.sh",0o755)
    return

#Initialize start date
s_day = 1 
s_month = 8
s_year = 2023
s_date = datetime.date(s_year,s_month,s_day)

#Initialize end date
e_day = 30 
e_month = 11 
e_year = 2023
e_date = datetime.date(e_year,e_month,e_day)

#Initialize file index
i=0

while s_date <= e_date:
    
    #Get the year,month,day of the date
    year = str(s_date).split("-")[0]
    month = str(s_date).split("-")[1]
    day = str(s_date).split("-")[2]
    
    #Use function create_skim_script to generate executable for htcondor
    create_skim_script(i,year,month,day)
    
    #Increase date and index for next job
    i += 1
    s_date += datetime.timedelta(days=1)
    
print("Number of jobs to be submitted: ",i)
