#!/bin/bash 

echo "Activate VE"
source /home/kyang/master_grid/myenv/bin/activate

echo "Run script"
python3 /home/kyang/master_grid/ml/model/run.py

echo "Deactivate VE"
deactivate
