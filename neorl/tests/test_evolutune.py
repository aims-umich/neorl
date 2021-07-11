from neorl.tune import ESTUNE
from neorl import PSO

def test_evolutune():
    
    #Define the fitness function (for original optimisation)
    def sphere(individual):
        y=sum(x**2 for x in individual)
        return y
    
    def tune_fit(x):
        
        npar=x[0]
        c1=x[1] 
        c2=x[2]
        if x[3] == 1:
            speed_mech='constric'
        elif x[3] == 2:
            speed_mech='timew'
        elif x[3] == 3:
            speed_mech='globw'
    
        #--setup the parameter space
        nx=5
        BOUNDS={}
        for i in range(1,nx+1):
                BOUNDS['x'+str(i)]=['float', -100, 100]
    
        #--setup the ES algorithm
        pso=PSO(mode='min', bounds=BOUNDS, fit=sphere, npar=npar, c1=c1, c2=c2,
                 speed_mech=speed_mech, ncores=1, seed=1)
    
        #--Evolute the PSO object and obtains y_best
        #--turn off verbose for less algorithm print-out when tuning
        x_best, y_best, pso_hist=pso.evolute(ngen=30, verbose=0)
    
        return y_best #returns the best score
    

    #Setup the parameter space
    #VERY IMPORTANT: The order of these parameters MUST be similar to their order in tune_fit
    #see tune_fit
    param_grid={
    #def tune_fit(npar, c1, c2, speed_mech):
    'npar': ['int', 40, 60],        #npar is first (low=30, high=60, type=int/discrete)
    'c1': ['float', 2.05, 2.15],    #c1 is second (low=2.05, high=2.15, type=float/continuous)
    'c2': ['float', 2.05, 2.15],    #c2 is third (low=2.05, high=2.15, type=float/continuous)
    'speed_mech': ['int', 1, 3]}    #speed_mech is fourth (categorial variable encoded as integer, see tune_fit)
    
    #setup a evolutionary tune object
    etune=ESTUNE(mode='min', param_grid=param_grid, fit=tune_fit, ngen=10) #total cases is ngen * 10
    #tune the parameters with method .tune
    evolures=etune.tune(ncores=1, verbose=True)
    evolures = evolures.sort_values(['score'], axis='index', ascending=True) #rank the scores from min to max
    print(evolures)

test_evolutune()