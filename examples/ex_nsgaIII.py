from neorl import NSGAIII
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  #This is important for 3d plotting 
from neorl.benchmarks.dtlz import DTLZ2   #Multi-objective benchmark
from neorl.multi.tools import uniform_reference_points  #for plotting reference points
import numpy as np

#parameters
NOBJ = 3   #number of objectives to optimize
nx=12
lambda_ = 92
problem = DTLZ2(n_var=nx, n_obj=NOBJ)   #adapted and taken from pymop package
dtlz2 = problem.evaluate                #objective function

#Setup the parameter space
BOUNDS={}
for i in range(1,nx+1):
    BOUNDS['x'+str(i)]=['float', 0, 1]

nsgaiii=NSGAIII(mode='min', bounds=BOUNDS, fit=dtlz2, lambda_=lambda_, mutpb=0.1,
     cxmode='blend', cxpb=0.8, ncores=1, p = nx ,sorting = 'log',seed=1)
x_best, y_best, es_hist=nsgaiii.evolute(ngen=400, verbose=1)
y_nsga3=np.array(es_hist['global_fitness'])

#plot pareto front
ref_points = uniform_reference_points(nobj = NOBJ, p = nx)  #reference points  
pf = problem.pareto_front(ref_points)                        #get optimal pareto front
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(ref_points[:,0], ref_points[:,1],ref_points[:,2], marker = 'o',color = 'red')
ax.scatter(y_nsga3[:,0],y_nsga3[:,1], y_nsga3[:,2], marker = '*',color = 'blue')
ax.scatter(pf[:,0], pf[:,1], pf[:,2], marker = 'x',color = 'black')
ax.set_xlabel('f1'); ax.set_ylabel('f2'); ax.set_zlabel('f3')
plt.legend(["ref points","NSGA-III","ideal pareto front"])
plt.tight_layout()
plt.show()