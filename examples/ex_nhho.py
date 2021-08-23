from neorl import NHHO
import time

#Define the fitness function
def FIT(individual):
    """Sphere test objective function.
                    F(x) = sum_{i=1}^d xi^2
                    d=1,2,3,...
                    Range: [-100,100]
                    Minima: 0
    """
    y=sum(x**2 for x in individual)
    return y

#Setup the parameter space (d=5)
nx=5
BOUNDS={}
for i in range(1,nx+1):
    BOUNDS['x'+str(i)]=['float', -100, 100]

nn_params = {}
nn_params['num_nodes'] = [60, 30, 15]
nn_params['learning_rate'] = 8e-4
nn_params['epochs'] = 100
nn_params['plot'] = False #will accelerate training
nn_params['verbose'] = False #will accelerate training
nn_params['save_models'] = False  #will accelerate training

t0=time.time()
nhho = NHHO(mode='min', bounds=BOUNDS, fit=FIT, nhawks=20, 
            nn_params=nn_params, ncores=3, seed=1)
individuals, fitnesses = nhho.evolute(ngen=50, verbose=True)
print('Comp Time:', time.time()-t0)

#make evaluation of the best individuals using the real fitness function
real_fit=[FIT(item) for item in individuals]

#print the best individuals/fitness found
min_index=real_fit.index(min(real_fit))
print('------------------------ Final Summary --------------------------')
print('Best real individual:', individuals[min_index])
print('Best real fitness:', real_fit[min_index])
print('-----------------------------------------------------------------')
