#!/bin/bash

day=FILLED_DAY
month=FILLED_MONTH
year=FILLED_YEAR

source /home/kyang/master_grid/myenv/bin/activate

python3 /home/kyang/master_grid/ml/model/preprocessing/preprocess.py --day $day --month $month --year $year

deactivate
