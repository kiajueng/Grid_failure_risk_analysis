import glob

#Open the file from which to copy
to_copy = open("/home/kyang/master_grid/ml/weights/add_weights.sh", "r")

#Create for each .csv file one new python file which is then submitted
for i,file in enumerate(glob.glob("/share/scratch1/es-atlas/atlas_jobs_enr_skimmed/*2023_12_20*")):
    
    #Open a file in which the copy is writte and the file is modified
    copy = open(f"/home/kyang/master_grid/ml/weights/condor_submit/add_weights_{i}.sh","w")
    
    #Copy line for line and replace the TO_BE_FILLED string with the actual .csv file
    for line in to_copy:
        if 'TO_BE_FILLED' in line:
            copy.write(line.replace('TO_BE_FILLED',f'{file}'))
        else:
            copy.write(line)
    copy.close()
    
    #Set the to_copy line to zero so it can be looped again
    to_copy.seek(0)

to_copy.close()
            
