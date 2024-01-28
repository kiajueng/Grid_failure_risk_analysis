#!/bin/bash

file="TO_BE_FILLED"
source /home/kyang/master_grid/myenv/bin/activate

python3 /home/kyang/master_grid/ml/weights/add_weights.py -f $file

deactivate
