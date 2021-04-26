from neorl import ES

def test_es():
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
    
    es=ES(mode='min', bounds=BOUNDS, fit=FIT, lambda_=80, mu=40, mutpb=0.25,
         cxmode='blend', cxpb=0.7, ncores=1, seed=1)
    x_best, y_best, es_hist=es.evolute(ngen=100, verbose=0)

test_es()