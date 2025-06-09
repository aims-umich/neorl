import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

from neorl.hybrid.maeo import MAEO
from neorl.multi.tools import hypervolume_nd
from neorl.benchmarks.dtlz import DTLZ2

def plot_discovered_hypervolume(results, save_filename='maeo_discovered_hypervolume.png'):
    """Plot the evolution of maximum discovered hypervolume from cumulative history"""
    
    # Extract cumulative history and calculate hypervolume per cycle
    local_pop = results[2]['local_pop']  # metadata contains local_pop
    reference_point = np.array(results[2]['reference_point'])
    mode = results[2]['mode']
    
    # Group by cycle and calculate best hypervolume discovered so far
    cycle_hypervolumes = defaultdict(list)
    for record in local_pop:
        cycle_hypervolumes[record['cycle']].append(record['fitness'])
    
    discovered_hv_history = []
    all_discovered_fitness = []
    
    for cycle in sorted(cycle_hypervolumes.keys()):
        # Add this cycle's fitness values to cumulative set
        all_discovered_fitness.extend(cycle_hypervolumes[cycle])
        
        # Calculate hypervolume of all discovered solutions so far
        if mode == "min":
            # Filter dominated solutions for hypervolume calculation
            pareto_fitness = []
            for fitness in all_discovered_fitness:
                is_dominated = False
                for other_fitness in all_discovered_fitness:
                    if (np.all(np.array(other_fitness) <= np.array(fitness)) and 
                        np.any(np.array(other_fitness) < np.array(fitness))):
                        is_dominated = True
                        break
                if not is_dominated:
                    pareto_fitness.append(fitness)
        else:
            # Similar logic for maximization
            pareto_fitness = []
            for fitness in all_discovered_fitness:
                is_dominated = False
                for other_fitness in all_discovered_fitness:
                    if (np.all(np.array(other_fitness) >= np.array(fitness)) and 
                        np.any(np.array(other_fitness) > np.array(fitness))):
                        is_dominated = True
                        break
                if not is_dominated:
                    pareto_fitness.append(fitness)
        
        hv = hypervolume_nd(pareto_fitness, reference_point.tolist(), mode)
        discovered_hv_history.append(hv)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    cycles = range(1, len(discovered_hv_history) + 1)
    ax.plot(cycles, discovered_hv_history, 'g-', linewidth=2, marker='s', markersize=4)
    
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Maximum Discovered Hypervolume')
    ax.set_title('Evolution of Maximum Discovered Hypervolume (Cumulative)')
    ax.grid(True, alpha=0.3)
    
    # Add some styling
    ax.set_xlim(0.5, len(discovered_hv_history) + 0.5)
    
    # Add text box with final hypervolume
    if discovered_hv_history:
        final_hv = discovered_hv_history[-1]
        textstr = f'Max Discovered HV: {final_hv:.4f}'
        props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.5)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Discovered hypervolume plot saved to {save_filename}")

def save_history(results, filename='maeo_history.csv'):
    """Save optimization history to CSV - island-level summary"""
    local_pop = results[2]['local_pop']
    
    # Group by cycle and island to calculate hypervolumes and sizes
    cycle_island_data = defaultdict(lambda: defaultdict(list))
    
    for record in local_pop:
        cycle = record['cycle']
        island = record['island']
        cycle_island_data[cycle][island].append(record['fitness'])
    
    data = []
    reference_point = np.array(results[2]['reference_point'])
    mode = results[2]['mode']
    
    for cycle in sorted(cycle_island_data.keys()):
        for island in sorted(cycle_island_data[cycle].keys()):
            fitness_list = cycle_island_data[cycle][island]
            
            # Calculate hypervolume for this island's population
            from __main__ import hypervolume_nd
            hv = hypervolume_nd(fitness_list, reference_point.tolist(), mode)
            
            data.append({
                'cycle': cycle,
                'island': island,
                'hypervolume': hv,
                'population_size': len(fitness_list)
            })
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"History saved to {filename}")

def save_complete_history(results, filename='maeo_complete_history.csv'):
    """Save complete optimization history to CSV"""
    local_pop = results[2]['local_pop']
    
    if not local_pop:
        print("No cumulative history to save")
        return
    
    # Prepare data for CSV
    data_rows = []
    
    for record in local_pop:
        # Create base row with cycle and island info
        base_row = {
            'cycle': record['cycle'],
            'island': record['island']
        }
        
        # Add decision variables
        solution = record['solution']
        for i, var_val in enumerate(solution):
            base_row[f'var_{i+1}'] = var_val
        
        # Add objective values
        fitness = record['fitness']
        for i, obj_val in enumerate(fitness):
            base_row[f'obj_{i+1}'] = obj_val
        
        data_rows.append(base_row)
    
    # Create DataFrame and save
    df = pd.DataFrame(data_rows)
    df.to_csv(filename, index=False)
    print(f"Complete history saved to {filename} ({len(data_rows)} records)")

def plot_results(results, save_filename='maeo_results.png'):
    """Plot optimization results and save to file"""
    local_pop = results[2]['local_pop']
    mode = results[2]['mode']
    
    # Group data by cycle and island for plotting
    cycle_island_data = defaultdict(lambda: defaultdict(list))
    for record in local_pop:
        cycle_island_data[record['cycle']][record['island']].append(record['fitness'])
    
    # Calculate hypervolume and population size histories
    reference_point = np.array(results[2]['reference_point'])
    cycles = sorted(cycle_island_data.keys())
    islands = sorted(set(record['island'] for record in local_pop))
    
    hv_history = []
    size_history = []
    
    from __main__ import hypervolume_nd
    
    for cycle in cycles:
        cycle_hvs = []
        cycle_sizes = []
        for island in islands:
            if island in cycle_island_data[cycle]:
                fitness_list = cycle_island_data[cycle][island]
                hv = hypervolume_nd(fitness_list, reference_point.tolist(), mode)
                cycle_hvs.append(hv)
                cycle_sizes.append(len(fitness_list))
            else:
                cycle_hvs.append(0)
                cycle_sizes.append(0)
        hv_history.append(cycle_hvs)
        size_history.append(cycle_sizes)
    
    hv_history = np.array(hv_history)
    size_history = np.array(size_history)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot hypervolume evolution
    for i in range(hv_history.shape[1]):
        ax1.plot(range(1, len(cycles)+1), hv_history[:, i], label=f'Island {islands[i]}')
    ax1.set_xlabel('Cycle')
    ax1.set_ylabel('Hypervolume')
    ax1.set_title('Hypervolume Evolution')
    ax1.legend()
    ax1.grid(True)
    
    # Plot population size changes
    for i in range(size_history.shape[1]):
        ax2.plot(range(1, len(cycles)+1), size_history[:, i], label=f'Island {islands[i]}')
    ax2.set_xlabel('Cycle')
    ax2.set_ylabel('Population Size')
    ax2.set_title('Population Size Evolution')
    ax2.legend()
    ax2.grid(True)
    
    # Plot final Pareto front
    mode_str = "Minimized" if mode == "min" else "Maximized"
    final_pareto_fitness = np.array(results[1])  # y_best from results
    
    if len(final_pareto_fitness) > 0 and final_pareto_fitness.shape[1] >= 2:
        ax3.scatter(final_pareto_fitness[:, 0], final_pareto_fitness[:, 1], alpha=0.7)
        ax3.set_xlabel(f'Objective 1 ({mode_str})')
        ax3.set_ylabel(f'Objective 2 ({mode_str})')
        ax3.set_title('Final Pareto Front')
        ax3.grid(True)
    
    # Plot individual island final populations (from last cycle)
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    last_cycle = max(cycles)
    
    for i, island in enumerate(islands):
        if island in cycle_island_data[last_cycle]:
            fitness_data = np.array(cycle_island_data[last_cycle][island])
            if len(fitness_data) > 0 and fitness_data.shape[1] >= 2:
                ax4.scatter(fitness_data[:, 0], fitness_data[:, 1], 
                          color=colors[i % len(colors)], alpha=0.7, label=f'Island {island}')
    
    ax4.set_xlabel(f'Objective 1 ({mode_str})')
    ax4.set_ylabel(f'Objective 2 ({mode_str})')
    ax4.set_title('Individual Island Final Populations')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Results plot saved to {save_filename}")

if __name__ == "__main__":
    
    # Parameters
    NOBJ = 2   # number of objectives to optimize
    nx = 5     # number of variables
    lambda_ = 10
    problem = DTLZ2(n_var=nx, n_obj=NOBJ)
    dtlz2 = problem.evaluate 
    postprocessing = 0
    
    # Setup the parameter space
    BOUNDS = {}
    for i in range(1, nx+1):
        BOUNDS['x'+str(i)] = ['float', -10, 10]
    
    # Setup and run MAEO
    optimizers = ['NSGAII','NSGAII', 'NSGAIII', 'NSGAIII']  # Four different islands
    
    maeo = MAEO(
        bounds=BOUNDS,
        fit=dtlz2,
        optimizers=optimizers,
        gen_per_cycle=10,
        subcores=1,
        upcores=1,
        mode='min',
        lambda_=lambda_,
        seed=1
    )
    
    print("Starting MAEO optimization...")
    results = maeo.evolute(max_cycles=10, verbose=1)
    print("Optimization completed!")
    
    if postprocessing == 1:
        # Run postprocessing functions
        print("\nRunning postprocessing...")
        
        # Save complete history
        save_complete_history(results, 'maeo_complete_history.csv')
        
        # Save summary history  
        save_history(results, 'maeo_history.csv')
        
        # Plot results
        plot_results(results, 'maeo_results.png')
        
        # Create the discovered hypervolume plot
        plot_discovered_hypervolume(results, 'maeo_results_discovered_hv.png')
        
        print("\nPostprocessing completed!")
    else:
        print("\nPostprocessing and exporting options are available.")

    print(f"Final Pareto front contains {len(results[0])} solutions")
    print(f"Total solutions evaluated: {results[2]['total_solutions_evaluated']}")