import csv
import time
import numpy as np

def export_fitness(algorithm, fitness_data):

    File_name = "experiment "+time.strftime("%Y-%m-%d-%H-%M-%S")+".csv"
    
    iter_arr = []
    for i in range(0,int(len(fitness_data))):
        iter_arr.append("Iter"+str(i+1))
    
    row_1 = np.concatenate([["Optimizer"],iter_arr])
    row_2 = np.concatenate([[algorithm],fitness_data])

    print(row_1)
    print(row_2)

    with open(File_name,'w',newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row_1)
        writer.writerow(row_2)
    
        

    


    
    