import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter  

def draw_fit_vs_iters(record_filename_arr):

    #read csv file
    algorithm_num = len(record_filename_arr)
    plt.figure()
    al_count = 0

    for record_filename in record_filename_arr:
        
        with open(record_filename,newline='') as csvfile:

            rows = csv.reader(csvfile)

            row_count = 0
            for row in rows:
                
                if row_count == 1:

                    algorithm_name = row[0]
                    data_arr = np.array(row[1:])
                    iters = len(data_arr)
                    #create an array 1~iterations
                    x_iters = np.linspace(1,iters,iters)
                    #print(x_iters)
                    #print(data_arr)
                    

                row_count += 1
        #get the reciprocal of lost value => fitness value
        for i in range(int(len(data_arr))):
            
            fit_val = 1/float(data_arr[i])
            data_arr[i] = fit_val
            
            
        record = data_arr.astype('float32')
        #start to draw image
        if al_count == 0:
            plt.plot(x_iters,record,color='blue',label = 'PSO')
        elif al_count == 1:
            plt.plot(x_iters,record,color='red', label = 'IFB-PSO')

        al_count += 1

    plt.xlim(1,iters)
    plt.ylim((0,5))
    plt.legend(loc='upper left')
    plt.xlabel('Iterations')
    plt.ylabel('Fitness Value')
    plt.show()
    

file_name_arr = ['experiment 2020-01-06-19-19-32.csv','experiment 2020-01-08-14-12-06.csv']
draw_fit_vs_iters(file_name_arr)