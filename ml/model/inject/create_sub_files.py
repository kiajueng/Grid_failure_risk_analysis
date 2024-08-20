import argparse
import datetime

if __name__=='__main__':
    
    parser=argparse.ArgumentParser()
    parser.add_argument("--start_day", type=int, required=True)
    parser.add_argument("--start_month", type=int, required=True)
    parser.add_argument("--start_year", type=int, required=True)
    parser.add_argument("--end_day", type=int, required=True)
    parser.add_argument("--end_month", type=int, required=True)
    parser.add_argument("--end_year", type=int, required=True)
    args = parser.parse_args()

    #Initialize start and end date
    start_date = datetime.date(args.start_year,args.start_month,args.start_day)
    end_date = datetime.date(args.end_year,args.end_month,args.end_day)

    #Counting index
    count = 0
    
    #Open the file to copy from
    to_copy = open("/home/kyang/master_grid/ml/model/inject/inject.sh","r")

    
    while start_date <= end_date:

        flag = False

        with open("/home/kyang/master_grid/ml/model/inject/done.txt","r") as f_read:
            for line in f_read:
                if str(start_date) in line:
                    flag = True

        if flag: 
            start_date += datetime.timedelta(days=1)
            continue


        with open(f"/home/kyang/master_grid/ml/model/inject/condor_submit/inject_{count}.sh", "w") as f:
            for line in to_copy:
                if "FILLED_DAY" in line:
                    f.write(line.replace("FILLED_DAY",f"{start_date.day}"))
                elif "FILLED_MONTH" in line:
                    f.write(line.replace("FILLED_MONTH",f"{start_date.month}"))
                elif "FILLED_YEAR" in line:
                    f.write(line.replace("FILLED_YEAR",f"{start_date.year}"))
                else:
                    f.write(line)
        
        count += 1
        start_date += datetime.timedelta(days=1)
        to_copy.seek(0)

    to_copy.close()

print(f"NUMBER OF JOBS: {count}")
