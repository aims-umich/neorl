########################
# Import Packages
########################

from neorl import HHO, ES, PESA, BAT, GWO, MFO, WOA, SSA, DE, JAYA, PESA2, PSO
import math

def test_mixedea():
    #################################
    # Define Vessel Function 
    #Mixed discrete/continuous/grid
    #################################
    def Vessel(individual):
        """
        Pressure vesssel design
        x1: thickness (d1)  --> discrete value multiple of 0.0625 in 
        x2: thickness of the heads (d2) ---> categorical value from a pre-defined grid
        x3: inner radius (r)  ---> cont. value between [10, 200]
        x4: length (L)  ---> cont. value between [10, 200]
        """
        
        #print(individual)
        x=individual.copy()
        x[0] *= 0.0625   #convert d1 to "in" 
    
        y = 0.6224*x[0]*x[2]*x[3]+1.7781*x[1]*x[2]**2+3.1661*x[0]**2*x[3]+19.84*x[0]**2*x[2];
    
        g1 = -x[0]+0.0193*x[2];
        g2 = -x[1]+0.00954*x[2];
        g3 = -math.pi*x[2]**2*x[3]-(4/3)*math.pi*x[2]**3 + 1296000;
        g4 = x[3]-240;
        g=[g1,g2,g3,g4]
        
        phi=sum(max(item,0) for item in g)
        eps=1e-5 #tolerance to escape the constraint region
        penality=1e7 #large penality to add if constraints are violated
        
        if phi > eps:  
            fitness=phi+penality
        else:
            fitness=y
        return fitness
    
    
    for item in ['float', 'grid', 'float/int', 'float/grid', 'int/grid', 'mixed', 'int']:
        bounds = {}
        btype=item  #float, int, grid, float/int, float/grid, int/grid, mixed. 
        
        print(item, 'is running -----')
        if btype=='mixed':
            bounds['x1'] = ['int', 1, 99]
            bounds['x2'] = ['grid', (0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625, 0.625)]
            bounds['x3'] = ['float', 10, 200]
            bounds['x4'] = ['float', 10, 200]
            bounds['x5'] = ['grid', ('Hi', 'Bye', 'New')]
            bounds['x6'] = ['int', -5, 5]
        
        elif btype=='int/grid':      
            bounds['x1'] = ['int', 1, 20]
            bounds['x2'] = ['grid', (0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625, 0.625)]
            bounds['x3'] = ['int', 10, 200]
            bounds['x4'] = ['int', 10, 200]
            bounds['x5'] = ['grid', ('Hi', 'Bye', 'New')]
            bounds['x6'] = ['int', -5, 5]
        
        elif btype=='float/grid':      
            bounds['x1'] = ['float', 1, 20]
            bounds['x2'] = ['grid', (0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625, 0.625)]
            bounds['x3'] = ['float', 10, 200]
            bounds['x4'] = ['float', 10, 200]
            bounds['x5'] = ['grid', ('Hi', 'Bye', 'New')]
            bounds['x6'] = ['float', -5, 5]
            
        elif btype=='float/int':      
            bounds['x1'] = ['int', 1, 20]
            bounds['x2'] = ['float', 1, 20]
            bounds['x3'] = ['int', 10, 200]
            bounds['x4'] = ['float', 10, 200]
            bounds['x5'] = ['float', -5, 5]
            bounds['x6'] = ['int', -5, 5]
        
        elif btype=='float':      
            bounds['x1'] = ['float', 1, 20]
            bounds['x2'] = ['float', 1, 20]
            bounds['x3'] = ['float', 10, 200]
            bounds['x4'] = ['float', 10, 200]
            bounds['x5'] = ['float', -5, 5]
            bounds['x6'] = ['float', -5, 5]
            
        elif btype=='int':      
            bounds['x1'] = ['int', 1, 20]
            bounds['x2'] = ['int', 1, 20]
            bounds['x3'] = ['int', 10, 200]
            bounds['x4'] = ['int', 10, 200]
            bounds['x5'] = ['int', -5, 5]
            bounds['x6'] = ['int', -5, 5]
            
        elif btype=='grid':      
            bounds['x1'] = ['grid', (0.0625, 0.125, 0.375, 0.4375, 0.5625, 0.625)]
            bounds['x2'] = ['grid', (0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625, 0.625)]
            bounds['x3'] = ['grid', (1,2,3,4,5)]
            bounds['x4'] = ['grid', (32,64,128)]
            bounds['x5'] = ['grid', ('Hi', 'Bye', 'New')]
            bounds['x6'] = ['grid', ('Cat', 'Dog', 'Bird', 'Fish')]
    
        ngen=5
        ########################
        # Setup and evolute HHO
        ########################
        hho = HHO(mode='min', bounds=bounds, fit=Vessel, nhawks=50, 
                          int_transform='minmax', ncores=1, seed=1)
        x_hho, y_hho, hho_hist=hho.evolute(ngen=ngen, verbose=0)
        assert Vessel(x_hho) == y_hho
        
        ########################
        # Setup and evolute ES 
        ########################
        es = ES(mode='min', fit=Vessel, cxmode='cx2point', bounds=bounds, 
                         lambda_=60, mu=30, cxpb=0.7, mutpb=0.2, seed=1)
        x_es, y_es, es_hist=es.evolute(ngen=ngen, verbose=0)
        assert Vessel(x_es) == y_es
        
        ########################
        # Setup and evolute PESA
        ########################
        pesa=PESA(mode='min', bounds=bounds, fit=Vessel, npop=60, mu=30, alpha_init=0.01,
                  alpha_end=1.0, cxpb=0.7, mutpb=0.2, alpha_backdoor=0.05)
        x_pesa, y_pesa, pesa_hist=pesa.evolute(ngen=ngen, verbose=0)
        assert Vessel(x_pesa) == y_pesa
        
        ########################
        # Setup and evolute BAT
        ########################
        bat=BAT(mode='min', bounds=bounds, fit=Vessel, nbats=50, fmin = 0 , fmax = 1, 
                A=0.5, r0=0.5, levy = True, seed = 1, ncores=1)
        x_bat, y_bat, bat_hist=bat.evolute(ngen=ngen, verbose=0)
        assert Vessel(x_bat) == y_bat
        
        ########################
        # Setup and evolute MFO
        ########################
        mfo=MFO(mode='min', bounds=bounds, fit=Vessel, nmoths=50, int_transform='minmax', ncores=1, seed=1)
        x_mfo, y_mfo, mfo_hist=mfo.evolute(ngen=ngen, verbose=0)
        assert Vessel(x_mfo) == y_mfo
    
        ########################
        # Setup and evolute JAYA
        ########################
        jaya=JAYA(mode='min', bounds=bounds, fit=Vessel, npop=60, int_transform='sigmoid', ncores=1, seed=1)
        x_jaya, y_jaya, jaya_hist=jaya.evolute(ngen=ngen, verbose=0)
        print(item)
        assert Vessel(x_jaya) == y_jaya
        
        ########################
        # Setup and evolute GWO
        ########################
        gwo=GWO(mode='min', fit=Vessel, bounds=bounds, nwolves=5, ncores=1, seed=1)
        x_gwo, y_gwo, gwo_hist=gwo.evolute(ngen=ngen, verbose=0)
        assert Vessel(x_gwo) == y_gwo
        
        ########################
        # Setup and evolute WOA
        ########################
        woa=WOA(mode='min', bounds=bounds, fit=Vessel, nwhales=20, a0=1.5, b=1, ncores=1, seed=1)
        x_woa, y_woa, woa_hist=woa.evolute(ngen=ngen, verbose=0)
        assert Vessel(x_woa) == y_woa
        
        ########################
        # Setup and evolute SSA
        ########################
        #setup and evolute SSA
        ssa=SSA(mode='min', bounds=bounds, fit=Vessel, nsalps=50, c1=None, int_transform='sigmoid', ncores=1, seed=1)
        x_ssa, y_ssa, ssa_hist=ssa.evolute(ngen=ngen, verbose=0)
        assert Vessel(x_ssa) == y_ssa
        
        ########################
        # Setup and evolute DE
        ########################
        de=DE(mode='min', bounds=bounds, fit=Vessel, npop=60, F=0.5, CR=0.7, int_transform='sigmoid', ncores=1, seed=1)
        x_de, y_de, de_hist=de.evolute(ngen=ngen, verbose=0)
        assert Vessel(x_de) == y_de
        
        ########################
        # Setup and evolute PESA2
        ########################
        pesa2=PESA2(mode='min', bounds=bounds, fit=Vessel, npop=60, nwolves=5)
        x_pesa2, y_pesa2, pesa2_hist=pesa2.evolute(ngen=ngen, replay_every=2, verbose=0)
        assert Vessel(x_pesa2) == y_pesa2
        
        ########################
        # Setup and evolute PSO
        ########################
        pso=PSO(mode='min', bounds=bounds, fit=Vessel, c1=2.05, c2=2.1, npar=50,
                        speed_mech='constric', ncores=1, seed=1)
        x_pso, y_pso, pso_hist=pso.evolute(ngen=ngen, verbose=0)
        assert Vessel(x_pso) == y_pso
        
        ########################
        # Comparison
        ########################
        print('---Best HHO Results---')
        print(x_hho)
        print(y_hho)
        print('---Best ES Results---')
        print(x_es)
        print(y_es)
        print('---Best PESA Results---')
        print(x_pesa)
        print(y_pesa)
        print('---Best BAT Results---')
        print(x_bat)
        print(y_bat)
        print('---Best GWO Results---')
        print(x_gwo)
        print(y_gwo)
        print('---Best WOA Results---')
        print(x_woa)
        print(y_woa)
        print('---Best SSA Results---')
        print(x_ssa)
        print(y_ssa)
        print('---Best MFO Results---')
        print(x_mfo)
        print(y_mfo)
        print('---Best DE Results---')
        print(x_de)
        print(y_de)
        print('---Best JAYA Results---')
        print(x_jaya)
        print(y_jaya)
        print('---Best PESA2 Results---')
        print(x_pesa2)
        print(y_pesa2)
        print('---Best PSO Results---')
        print(x_pso)
        print(y_pso)
    
test_mixedea()