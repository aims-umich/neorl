from neorl import TS
import numpy as np

#this example is not finalized, it uses TS to solve two combinatorial problems:
#Travel salesman problem
#Job scheduling problem

#############################################
#### -- Scheduling Problem related input
#############################################

def input_data(njobs=30):
    '''
    Job Scheduling Problem Data
    Returns a dict of jobs number as Key and 
    weight, processing time (hours) and due date (hours) as values.
    '''
    
    if njobs==10:
        JobData={1: {'weight': 0.5, 'processing_time': 1.5, 'due_date': 4},
                 2: {'weight': 0.8, 'processing_time': 2.2, 'due_date': 4},
                 3: {'weight': 0.6, 'processing_time': 0.5, 'due_date': 4},
                 4: {'weight': 0.4, 'processing_time': 1.6, 'due_date': 6},
                 5: {'weight': 0.4, 'processing_time': 3.5, 'due_date': 6},
                 6: {'weight': 0.1, 'processing_time': 2.1, 'due_date': 6},
                 7: {'weight': 0.2, 'processing_time': 2.5, 'due_date': 8},
                 8: {'weight': 0.5, 'processing_time': 0.6, 'due_date': 8},
                 9: {'weight': 0.3, 'processing_time': 3.2, 'due_date': 8},
                 10: {'weight': 0.6, 'processing_time': 5.2, 'due_date': 8}}
        
    elif njobs==20:
        
        JobData={1: {'weight': 0.5, 'processing_time': 0.5, 'due_date': 8},
                 2: {'weight': 0.8, 'processing_time': 3.6, 'due_date': 6},
                 3: {'weight': 0.6, 'processing_time': 2.8000000000000003, 'due_date': 5},
                 4: {'weight': 0.4, 'processing_time': 1.9000000000000001, 'due_date': 12},
                 5: {'weight': 0.4, 'processing_time': 1.0, 'due_date': 10},
                 6: {'weight': 0.1, 'processing_time': 1.8, 'due_date': 11},
                 7: {'weight': 0.2, 'processing_time': 2.6, 'due_date': 6},
                 8: {'weight': 0.5, 'processing_time': 4.699999999999999, 'due_date': 18},
                 9: {'weight': 0.3, 'processing_time': 1.8, 'due_date': 8},
                 10: {'weight': 0.6, 'processing_time': 1.6, 'due_date': 14},
                 11: {'weight': 0.5, 'processing_time': 0.2, 'due_date': 6},
                 12: {'weight': 0.8, 'processing_time': 0.4, 'due_date': 30},
                 13: {'weight': 0.6, 'processing_time': 2.2, 'due_date': 22},
                 14: {'weight': 0.4, 'processing_time': 1.2000000000000002, 'due_date': 6},
                 15: {'weight': 0.4, 'processing_time': 0.1, 'due_date': 38},
                 16: {'weight': 0.1, 'processing_time': 2.0, 'due_date': 16},
                 17: {'weight': 0.2, 'processing_time': 2.9, 'due_date': 10},
                 18: {'weight': 0.5, 'processing_time': 0.1, 'due_date': 12},
                 19: {'weight': 0.3, 'processing_time': 2.5, 'due_date': 20},
                 20: {'weight': 0.6, 'processing_time': 0.9, 'due_date': 15}}
                
    elif njobs==30:
    
        JobData={1: {'weight': 0.5, 'processing_time': 6, 'due_date': 29},
                 2: {'weight': 0.9, 'processing_time': 1, 'due_date': 38},
                 3: {'weight': 0.0, 'processing_time': 2, 'due_date': 28},
                 4: {'weight': 0.3, 'processing_time': 6, 'due_date': 35},
                 5: {'weight': 0.6, 'processing_time': 5, 'due_date': 43},
                 6: {'weight': 0.3, 'processing_time': 4, 'due_date': 44},
                 7: {'weight': 0.9, 'processing_time': 6, 'due_date': 26},
                 8: {'weight': 0.4, 'processing_time': 1, 'due_date': 18},
                 9: {'weight': 0.2, 'processing_time': 2, 'due_date': 48},
                 10: {'weight': 0.4, 'processing_time': 4, 'due_date': 37},
                 11: {'weight': 1.0, 'processing_time': 1, 'due_date': 26},
                 12: {'weight': 0.1, 'processing_time': 4, 'due_date': 36},
                 13: {'weight': 0.4, 'processing_time': 6, 'due_date': 24},
                 14: {'weight': 0.8, 'processing_time': 3, 'due_date': 45},
                 15: {'weight': 0.1, 'processing_time': 4, 'due_date': 41},
                 16: {'weight': 1.0, 'processing_time': 1, 'due_date': 18},
                 17: {'weight': 0.1, 'processing_time': 3, 'due_date': 46},
                 18: {'weight': 0.4, 'processing_time': 4, 'due_date': 36},
                 19: {'weight': 0.7, 'processing_time': 8, 'due_date': 20},
                 20: {'weight': 1.0, 'processing_time': 8, 'due_date': 18},
                 21: {'weight': 0.2, 'processing_time': 2, 'due_date': 25},
                 22: {'weight': 0.1, 'processing_time': 2, 'due_date': 34},
                 23: {'weight': 0.0, 'processing_time': 2, 'due_date': 36},
                 24: {'weight': 0.6, 'processing_time': 4, 'due_date': 42},
                 25: {'weight': 0.1, 'processing_time': 5, 'due_date': 29},
                 26: {'weight': 0.9, 'processing_time': 6, 'due_date': 26},
                 27: {'weight': 0.7, 'processing_time': 3, 'due_date': 33},
                 28: {'weight': 0.6, 'processing_time': 1, 'due_date': 24},
                 29: {'weight': 0.4, 'processing_time': 1, 'due_date': 25},
                 30: {'weight': 0.9, 'processing_time': 4, 'due_date': 26}}
    else:
        raise Exception('--error: choose njobs as 10, 20, or 30')
        
    return JobData



def Objfun(solution):
    '''Takes a set of scheduled jobs, dict (input data)
    Return the objective function value of the solution
    '''
    t = 0   #starting time
    objfun_value = 0
    for job in solution:
        C_i = t + dictt[job]["processing_time"]  # Completion time
        d_i = dictt[job]["due_date"]   # due date of the job
        T_i = max(0, C_i - d_i)    #tardiness for the job
        W_i = dictt[job]["weight"]  # job's weight
        objfun_value +=  W_i * T_i
        t = C_i
    return objfun_value


#############################################
#### -- TSP related input
#############################################
def Gen_TSP_Data():
    #---51 cities
    #locations
    city_loc_list = [[37,52],[49,49],[52,64],[20,26],[40,30],[21,47],[17,63],[31,62],[52,33],[51,21],[42,41],[31,32],[5,25]\
                ,[12, 42],[36, 16],[52, 41],[27, 23],[17, 33],[13, 13],[57, 58],[62, 42],[42, 57],[16, 57],[8 ,52],[7 ,38],[27, 68],[30, 48]\
                ,[43, 67],[58, 48],[58, 27],[37, 69],[38, 46],[46, 10],[61, 33],[62, 63],[63, 69],[32, 22],[45, 35],[59, 15],[5 ,6],[10, 17]\
                ,[21, 10],[5 ,64],[30, 15],[39, 10],[32, 39],[25, 32],[25, 55],[48, 28],[56, 37],[30, 40]]
    increment = 0
    global dictt
    dictt = {}
    i = 0
    for data in city_loc_list:
        dictt[str(i+1)] = [data[0],data[1]]
        i += 1
    #optimal solution for comparison
    optimum_tour_city = [1,22,8,26,31,28,3,36,35,20,2,29,21,16,50,34,30,9,49,10,39,33,45,15,44,42,40,19,41,13,25,14,24,43,7,23,48\
                         ,6,27,51,46,12,47,18,4,17,37,5,38,11,32]
    optimum_tour_cost = Tour(optimum_tour_city)
    return optimum_tour_city,optimum_tour_cost

def Tour(tour):
    cost = 0
    for increment in range(0,len(tour) - 1): # compute euclidean distance from cities to cities
        loc1 = np.array([dictt[str(tour[increment])][0],dictt[str(tour[increment])][1]])
        loc2 = np.array([dictt[str(tour[increment + 1])][0],dictt[str(tour[increment + 1])][1]])
        dist = np.sqrt(np.sum(np.power(loc1 - loc2,2)))
        cost += int(round(dist))
    loc1 = np.array([dictt[str(tour[-1])][0],dictt[str(tour[-1])][1]])
    loc2 = np.array([dictt[str(tour[0])][0],dictt[str(tour[0])][1]])
    dist = np.sqrt(np.sum(np.power(loc1 - loc2,2)))
    cost += int(round(dist)) 
    return - cost    

if __name__=='__main__':
    if 0:
        #Setup the parameter space
        optimum_tour_city,optimum_tour_cost = Gen_TSP_Data()
        print("------- Running TSP -------")
        print("Optimum tour cost:",optimum_tour_cost)
        nx=51
        BOUNDS={}
        for i in range(1,nx+1):
            BOUNDS['x'+str(i)]=['int', 1, 51]
        ##use Instance_10.xlsx, Instance_20.xlsx, Instance_30.xlsx
        ts=TS(mode = "min", bounds = BOUNDS, fit = Tour, tabu_tenure=6, 
              penalization_weight = 0.8, swap_mode = "swap", ncores=1, seed=1)
        ts.evolute(ngen = 500)

    if 1:
        #Setup the parameter space
        print("------- Running JOB Scheduling -------")
        njobs=10
        dictt = input_data(njobs=njobs)
        BOUNDS={}
        for i in range(1,njobs+1):
            BOUNDS['x'+str(i)]=['int', 1, njobs]
        ts=TS(mode = "min", bounds = BOUNDS, fit = Objfun, 
              tabu_tenure=6, penalization_weight = 0.8, swap_mode = "swap", ncores=1, seed=1)
        ts.evolute(ngen = 30)
    
    
    