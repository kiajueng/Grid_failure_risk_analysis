#!/bin/bash 

echo "Activate VE"
source /home/kyang/master_grid/myenv/bin/activate

echo "Run script"
python3 /home/kyang/master_grid/ml/model/run_weights_no_cpu_eff.py

echo "Deactivate VE"
deactivate
