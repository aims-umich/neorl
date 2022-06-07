########################
# Import Packages
########################

from neorl import DE, GWO, SSA, WOA, AEO, MFO, JAYA, HHO, PSO, ES, EPSO
from neorl import EDEV, HCLPSO, BAT, SA, CS, TS, PESA, PESA2
import math, random

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
    
    def init_sample(bounds):
        #generate an individual from a bounds dictionary
        indv=[]
        for key in bounds:
            if bounds[key][0] == 'int':
                indv.append(random.randint(bounds[key][1], bounds[key][2]))
            elif bounds[key][0] == 'float':
                indv.append(random.uniform(bounds[key][1], bounds[key][2]))
            elif bounds[key][0] == 'grid':
                indv.append(random.sample(bounds[key][1],1)[0])
            else:
                raise Exception ('unknown data type is given, either int, float, or grid are allowed for parameter bounds')   
        return indv
        
    ngen=5
    
    for item in ['mixed', 'grid', 'float/int', 'float/grid', 'int/grid', 'float', 'int']:
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
    
    
        npop=70
        x0=[]
        for i in range(npop):
            x0.append(init_sample(bounds))
            
        ########################
        # Setup and evolute GWO
        ########################
        gwo=GWO(mode='min', fit=Vessel, bounds=bounds, nwolves=npop, ncores=1, seed=1)
        x_gwo, y_gwo, gwo_hist=gwo.evolute(ngen=ngen, verbose=0)
        assert Vessel(x_gwo) == y_gwo
        x_gwo, y_gwo, gwo_hist=gwo.evolute(ngen=ngen, x0=x0, verbose=0)
        assert Vessel(x_gwo) == y_gwo
        
        ########################
        # Setup and evolute WOA
        ########################
        woa=WOA(mode='min', bounds=bounds, fit=Vessel, nwhales=npop, a0=1.5, b=1, ncores=1, seed=1)
        x_woa, y_woa, woa_hist=woa.evolute(ngen=ngen, verbose=0)
        assert Vessel(x_woa) == y_woa
        x_woa, y_woa, woa_hist=woa.evolute(ngen=ngen, x0=x0, verbose=0)
        assert Vessel(x_woa) == y_woa
    
        ########################
        # Setup and evolute SSA
        ########################
        #setup and evolute SSA
        ssa=SSA(mode='min', bounds=bounds, fit=Vessel, nsalps=npop, int_transform='sigmoid', ncores=1, seed=1)
        x_ssa, y_ssa, ssa_hist=ssa.evolute(ngen=ngen, verbose=0)
        assert Vessel(x_ssa) == y_ssa    
        x_ssa, y_ssa, ssa_hist=ssa.evolute(ngen=ngen, x0=x0, verbose=0)
        assert Vessel(x_ssa) == y_ssa
        
        ########################
        # Setup and evolute DE
        ########################
        de=DE(mode='min', bounds=bounds, fit=Vessel, npop=npop, F=0.5, CR=0.7, int_transform='sigmoid', ncores=1, seed=1)
        x_de, y_de, de_hist=de.evolute(ngen=ngen, verbose=0)
        assert Vessel(x_de) == y_de
        x_de, y_de, de_hist=de.evolute(ngen=ngen, x0=x0, verbose=0)
        assert Vessel(x_de) == y_de
        
        ########################
        # Setup and evolute MFO
        ########################
        mfo=MFO(mode='min', bounds=bounds, fit=Vessel, nmoths=npop, int_transform='minmax', ncores=1, seed=1)
        x_mfo, y_mfo, mfo_hist=mfo.evolute(ngen=ngen, verbose=0)
        assert Vessel(x_mfo) == y_mfo
        x_mfo, y_mfo, mfo_hist=mfo.evolute(ngen=ngen, x0=x0, verbose=0)
        assert Vessel(x_mfo) == y_mfo
    
        ########################
        # Setup and evolute JAYA
        ########################
        jaya=JAYA(mode='min', bounds=bounds, fit=Vessel, npop=npop, int_transform='sigmoid', ncores=1, seed=1)
        x_jaya, y_jaya, jaya_hist=jaya.evolute(ngen=ngen, verbose=0)
        assert Vessel(x_jaya) == y_jaya
        x_jaya, y_jaya, jaya_hist=jaya.evolute(ngen=ngen, x0=x0, verbose=0)
        assert Vessel(x_jaya) == y_jaya
        
        ########################
        # Setup and evolute HHO
        ########################
        hho = HHO(mode='min', bounds=bounds, fit=Vessel, nhawks=npop, 
                          int_transform='minmax', ncores=1, seed=1)
        x_hho, y_hho, hho_hist=hho.evolute(ngen=ngen, x0=x0, verbose=0)
        assert Vessel(x_hho) == y_hho
        
        ########################
        # Setup and evolute PSO
        ########################
        pso=PSO(mode='min', bounds=bounds, fit=Vessel, c1=2.05, c2=2.1, npar=npop,
                        speed_mech='constric', ncores=1, seed=1)
        x_pso, y_pso, pso_hist=pso.evolute(ngen=ngen, verbose=0)
        assert Vessel(x_pso) == y_pso
        x_pso, y_pso, pso_hist=pso.evolute(ngen=ngen, x0=x0, verbose=0)
        assert Vessel(x_pso) == y_pso
        
        ########################
        # Setup and evolute ES 
        ########################
        es = ES(mode='min', fit=Vessel, cxmode='cx2point', bounds=bounds, 
                         lambda_=npop, mu=5, cxpb=0.7, mutpb=0.2, seed=1)
        x_es, y_es, es_hist=es.evolute(ngen=ngen, verbose=0)
        assert Vessel(x_es) == y_es
        x_es, y_es, es_hist=es.evolute(ngen=ngen, x0=x0, verbose=0)
        assert Vessel(x_es) == y_es
    
        ########################
        # Setup and evolute ES 
        ########################
        #setup and evolute BAT
        bat=BAT(mode='min', bounds=bounds, fit=Vessel, nbats=npop, 
                fmin=0, fmax=1, A=1.0, r0=0.7,
                ncores=1, seed=1)
        x_bat, y_bat, bat_hist=bat.evolute(ngen=ngen, verbose=0)
        assert Vessel(x_bat) == y_bat
        x_bat, y_bat, bat_hist=bat.evolute(ngen=ngen, x0=x0, verbose=0)
        assert Vessel(x_bat) == y_bat
    
        
        ########################
        # Setup and evolute EPSO
        ########################   
        #setup and evolute EPSO
        epso=EPSO(mode='min', bounds=bounds, g1=int(npop/2), g2=int(npop/2), fit=Vessel, ncores=1, seed=None)
        x_epso, y_epso, epso_hist=epso.evolute(ngen=ngen, LP=1, verbose=0)
        assert Vessel(x_epso) == y_epso
        x_epso, y_epso, epso_hist=epso.evolute(ngen=ngen, x0=x0, LP=1, verbose=0)
        assert Vessel(x_epso) == y_epso
    
        ########################
        # Setup and evolute EDEV
        ########################  
        #setup and evolute EDEV
        edev=EDEV(mode='min', bounds=bounds, fit=Vessel, npop=npop, ncores=1, seed=1)
        x_edev, y_edev, edev_hist=edev.evolute(ngen=ngen, ng=1, verbose=0)
        assert Vessel(x_edev) == y_edev
        x_edev, y_edev, edev_hist=edev.evolute(ngen=ngen, x0=x0, ng=1, verbose=0)
        assert Vessel(x_edev) == y_edev
        
        ########################
        # Setup and evolute HCLPSO
        ########################  
        #setup and evolute HCLPSO
        hclpso=HCLPSO(mode='min', bounds=bounds, g1=int(npop/2), g2=int(npop/2), fit=Vessel, ncores=1, seed=1)
        x_hclpso, y_hclpso, hclpso_hist=hclpso.evolute(ngen=ngen, verbose=0)
        assert Vessel(x_hclpso) == y_hclpso
        x_hclpso, y_hclpso, hclpso_hist=hclpso.evolute(ngen=ngen, x0=x0, verbose=0)
        assert Vessel(x_hclpso) == y_hclpso
    
        ########################
        # Setup and evolute SA
        ########################  
        #define a custom moving function
        def my_move(x, **kwargs):
            #-----
            #this function selects two random indices in x and perturb their values
            #-----
            x_new=x.copy()
            #indices=random.sample(range(0,len(x)), 2)
            #for i in indices:
            #    x_new[i] = random.uniform(bounds['x1'][1],bounds['x1'][2])
            x_new=init_sample(sa.bounds)
            
            return x_new
    
        #setup and evolute parallel SA with `equilibrium` cooling
        sa=SA(mode='min', bounds=bounds, fit=Vessel, chain_size=50, chi=0.2, Tmax=10000,
              move_func=my_move, reinforce_best='soft', cooling='boltzmann', ncores=1, seed=1)
    
        x_sa, y_sa, sa_hist=sa.evolute(ngen=ngen, verbose=0)
        assert Vessel(x_sa) == y_sa
        x_sa, y_sa, sa_hist=sa.evolute(ngen=ngen, x0=x0[0], verbose=0)
        assert Vessel(x_sa) == y_sa
    
        ########################
        # Setup and evolute CS
        ########################  
        #setup and evolute CS
        cs = CS(mode = 'min', bounds = bounds, fit = Vessel, ncuckoos = npop, pa = 0.25, seed=None)
        x_cs, y_cs, cs_hist=cs.evolute(ngen = ngen, verbose=0)
        assert Vessel(x_cs) == y_cs
        x_cs, y_cs, cs_hist=cs.evolute(ngen = ngen, x0=x0, verbose=0)
        assert Vessel(x_cs) == y_cs
    
        ########################
        # Setup and evolute TS
        ########################  
        #setup and evolute TS
        ts=TS(mode = "min", bounds = bounds, fit = Vessel, tabu_tenure=60, 
              penalization_weight = 0.8, swap_mode = "perturb", ncores=1, seed=None)
        x_ts, y_ts, ts_hist=ts.evolute(ngen = ngen, verbose=0)
        assert Vessel(x_ts) == y_ts
        x_ts, y_ts, ts_hist=ts.evolute(ngen = ngen, x0=x0[0], verbose=0)
        assert Vessel(x_ts) == y_ts
        
        ########################
        # Setup and evolute PESA
        ######################## 
        pesa=PESA(mode='min', bounds=bounds, fit=Vessel, npop=npop, mu=40, alpha_init=0.2, 
                  alpha_end=1.0, alpha_backdoor=0.1, ncores=1)
        x_pesa, y_pesa, pesa_hist=pesa.evolute(ngen=ngen, verbose=0)
        assert Vessel(x_pesa) == y_pesa
        x_pesa, y_pesa, pesa_hist=pesa.evolute(ngen=ngen, x0=x0, verbose=0)
        assert Vessel(x_pesa) == y_pesa
    
        ########################
        # Setup and evolute PESA2
        ######################## 
        pesa2=PESA2(mode='min', bounds=bounds, fit=Vessel, npop=npop, nwolves=5, nwhales=5, ncores=1)
        x_pesa2, y_pesa2, pesa2_hist=pesa2.evolute(ngen=ngen, replay_every=2, verbose=0)
        assert Vessel(x_pesa2) == y_pesa2
        x_pesa2, y_pesa2, pesa2_hist=pesa2.evolute(ngen=ngen, x0=x0, replay_every=2, verbose=0)
        assert Vessel(x_pesa2) == y_pesa2
    
        ########################
        # Setup and evolute AEO
        ########################
        aeo = AEO(mode='min', bounds=bounds, optimizers=[mfo, de, gwo, woa, jaya, hho, pso, es, ssa], gen_per_cycle=3, fit = Vessel)
        x_aeo, y_aeo, aeo_hist = aeo.evolute(3, verbose = 1)
        assert Vessel(x_aeo) == y_aeo
       
test_mixedea()