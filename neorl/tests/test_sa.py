from neorl import SA
import random

def test_sa():
    #Define the fitness function
    def FIT(individual):
        """Sphere test objective function.
        """
        y=sum(x**2 for x in individual)
        return y
    
    #Setup the parameter space (d=5)
    nx=5
    BOUNDS={}
    for i in range(1,nx+1):
        BOUNDS['x'+str(i)]=['float', -100, 100]
    
    #define a custom moving function
    def my_move(x, **kwargs):
        #-----
        #this function selects two random indices in x and perturb their values
        #-----
        x_new=x.copy()
        indices=random.sample(range(0,len(x)), 2)
        for i in indices:
            x_new[i] = random.uniform(BOUNDS['x1'][1],BOUNDS['x1'][2])
        
        return x_new
    
    #setup and evolute SA
    sa=SA(mode='min', bounds=BOUNDS, fit=FIT, chain_size=30,chi=0.2, 
          move_func=my_move, reinforce_best=True, ncores=1, seed=1)
    x_best, y_best, sa_hist=sa.evolute(ngen=100, verbose=1)
    
test_sa()