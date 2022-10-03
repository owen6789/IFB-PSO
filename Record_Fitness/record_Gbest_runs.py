import csv
import time
import numpy as np
import os
import string

def export_Gbest_runs(algorithm_name,benchmark_name,Gbest_runs):

    file_name = algorithm_name + ' v.s. ' + benchmark_name + \
                time.strftime(" %Y-%m-%d-%H-%M-%S")+".csv"

    runs = Gbest_runs.shape[0]
    iters = Gbest_runs.shape[1]
    #calculate average of Gbest 
    Gbest_runs_avg = np.zeros([iters])
    for i in range(iters):
        for j in range(runs):
             Gbest_runs_avg[i] += Gbest_runs[j][i]
        Gbest_runs_avg[i] = Gbest_runs_avg[i]/runs

    iters_arr = []
    for i in range(0,iters):
        iters_arr.append("Iter"+str(i+1))
    
    #check the file existed
    absolutepath = os.path.abspath(__file__)
    fileDirectory = os.path.dirname(absolutepath)
    parentDirectory = os.path.dirname(fileDirectory)
    with open(parentDirectory +'/Experimental_Records_cec/'+benchmark_name+'/'+file_name,'w',newline='') as csvfile:
        
        writer = csv.writer(csvfile)
        row_1 = np.concatenate([[algorithm_name],iters_arr])
        writer.writerow(row_1)

        #write Gbest
        for i in range(runs):
            row = np.concatenate([['run'+str(i+1)],Gbest_runs[i]])
            writer.writerow(row)

        #write avg
        row = np.concatenate([['average'],Gbest_runs_avg])
        writer.writerow(row)
        
        

    return 0


