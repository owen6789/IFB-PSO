import sys
sys.path.append('..')

import random
import numpy as np
from NiaPy.algorithms.basic import BackwardParticleSwarmAlgorithm
from Record_Fitness.record_Gbest_runs import export_Gbest_runs

Export = True #export data or not

runs = 15
Pp = 100
FE = 1000000
iters = int(FE/Pp)

#parameter for weight
W = 0.7
#hyperparameter for IFB-PSO
fg = int(iters*0.1)

#array to record data
Gbest_runs_PSO = np.zeros([int(runs),int(iters)*Pp])

# reruning part
benchmark_function = ['cec1','cec2','cec3', 'cec4','cec5', 'cec6', 'cec7', 'cec8','cec9','cec10', 'cec11','cec12', 'cec13','cec14',\
    'cec15','cec16','cec17', 'cec18','cec19','cec20','cec21', 'cec22', 'cec23', 'cec24', 'cec25', 'cec26', 'cec27', 'cec28',\
        'cec29','cec30']
D_f = [10,10,10,10,10,10,10,10,10,10,10,10, 10, 10, 10, 10, 10, 10, 10, 10, 10,10, 10, 10, 10, 10, 10, 10, 10, 10]
V_MAX = np.array([20, 20, 20, 2, 2, 4, 10, 20, 2, 20, 20, 20, 20, 20, 20, 20, 20, 20, 10, 20, 20, 20, 20, 20, 20, 20, 20, 10, 20, 20])
V_MIN = V_MAX*(-1)
if __name__ == "__main__":

    for i in range(1):

        b_count = 0

        for benchmark_name in benchmark_function:

            for run in range(runs):

                #IFB-W-PSO
                PSO = BackwardParticleSwarmAlgorithm(D=D_f[b_count], NP=Pp, nFES=FE, C1=1.5, C2=1.5, w=W, vMin=V_MIN[b_count], vMax=V_MAX[b_count], benchmark=benchmark_name, imp_check=fg, strategy=1)
                
                #run algorithms
                #IFB-W-PSO
                print('run%d:IFB-W-PSO v.s. %s' %(run+1,benchmark_name))
                PSO.run()

                #print Gbest
                print('IFB-W-PSO Gbest Fitness:' + str(PSO.gBestFitness))

                #record Gbest fitness
                Gbest_runs_PSO[run] = PSO.gBestFitness_record

            b_count += 1
            #record to CSV file
            export_Gbest_runs('IFB-W-PSO',benchmark_name,Gbest_runs_PSO)
