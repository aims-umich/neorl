#    This file is part of NEORL.

#    Copyright (c) 2021 Exelon Corporation and MIT Nuclear Science and Engineering
#    NEORL is free software: you can redistribute it and/or modify
#    it under the terms of the MIT LICENSE

#    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#    SOFTWARE.

from neorl.multi.nsgaII import NSGAII
from neorl.multi.nsgaIII import NSGAIII
from neorl.multi.tools import hypervolume_nd, multinomial_sample, ind_performance_metric, isDominated
from neorl.multi.tools import non_dominated_sort_MAEO
from neorl.utils.seeding import set_neorl_seed
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import random

# MAEO Class
class MAEO:
    def __init__(self, bounds, fit, optimizers, gen_per_cycle, subcores=1, upcores=1, 
                 mode="min", seed=None, lambda_=25, **kwargs):
        self.bounds = bounds
        self.fit = fit
        self.optimizers = optimizers  # List of optimizer types or instances
        self.gen_per_cycle = gen_per_cycle
        self.subcores = subcores
        self.upcores = upcores
        self.mode = mode.lower()  # "min" or "max"
        self.seed = seed
        self.lambda_ = lambda_  # Initial population size
        self.kwargs = kwargs
        
        # Initialize populations and tracking variables. Population is a list instead of a dictionary
        self.populations = []
        self.optimizer_instances = []  # Store instances to maintain state
        self.cumulative_history = []  # Store all solutions ever discovered
        self.reference_point = None
        
        set_neorl_seed(seed)
    
    def create_optimizer_instances(self, lambda_list=None):
        """Create optimizer instances for each island with dynamic population size"""
        if lambda_list is None:
            lambda_list = [self.lambda_] * len(self.optimizers)
        
        instances = []
        for i, (opt_type, lambda_val) in enumerate(zip(self.optimizers, lambda_list)):
            if opt_type == 'NSGAII':
                algo = NSGAII(
                    mode=self.mode, bounds=self.bounds, fit=self.fit, lambda_=lambda_val,
                    mutpb=0.35, cxmode='blend', cxpb=0.65, ncores=self.subcores,
                    p=len(self.bounds), sorting='log', seed=i+1 if self.seed else None
                )
            else:  # NSGAIII
                algo = NSGAIII(
                    mode=self.mode, bounds=self.bounds, fit=self.fit, lambda_=lambda_val,
                    mutpb=0.35, cxmode='blend', cxpb=0.65, ncores=self.subcores,
                    p=len(self.bounds), sorting='log', seed=i+1 if self.seed else None
                )
            instances.append(algo)
        return instances
    
    def update_optimizer_populations(self, populations):
        """Update optimizer instances with new populations after migration"""
        for i, (optimizer, new_pop) in enumerate(zip(self.optimizer_instances, populations)):
            if hasattr(optimizer, 'lambda_'):
                # Update the population size parameter
                optimizer.lambda_ = len(new_pop)
            
            # Set the current population in the optimizer
            if hasattr(optimizer, 'population'):
                # Convert population format if needed
                optimizer_pop = []
                for ind in new_pop:
                    if hasattr(optimizer, 'Individual'):
                        # Create proper Individual objects
                        individual = optimizer.Individual(ind[0])
                        individual.fitness.values = tuple(ind[1]) if isinstance(ind[1], list) else ind[1]
                        optimizer_pop.append(individual)
                    else:
                        # Fallback: assume simple format
                        optimizer_pop.append(ind)
                
                optimizer.population = optimizer_pop
    
    def run_island(self, optimizer, island_id):
        """Run a single island optimization"""
        # Check if we have a current population to continue from
        if hasattr(optimizer, 'population') and optimizer.population and len(optimizer.population) > 0:
            # Extract solutions from current population format [solution, fitness]
            current_solutions = [ind[0] for ind in optimizer.population]
            return optimizer.evolute(ngen=self.gen_per_cycle, x0=current_solutions, verbose=0) # The new population is run with x0.
        else:
            # First generation, no existing population
            return optimizer.evolute(ngen=self.gen_per_cycle, verbose=0) # There is no x0.
    
    def parse_population(self, optimizer_result):
        """Parse the optimizer result to extract population data"""
        metadata = optimizer_result[2]
        
        # Use the full population from NSGAIII metadata
        if 'last_pop' in metadata:
            df_pop = metadata['last_pop']  # This is a pandas DataFrame from get_population_nsga
            
            if isinstance(df_pop, pd.DataFrame) and not df_pop.empty:
                # Extract variable columns (solutions)
                var_cols = [col for col in df_pop.columns if col.startswith('var')]
                obj_cols = [col for col in df_pop.columns if col.startswith('obj')]
                
                population = []
                for idx, row in df_pop.iterrows():
                    solution = row[var_cols].values.tolist()
                    fitness = row[obj_cols].values.tolist()
                    population.append([solution, fitness])
                
                return population
    
    def extract_pareto_front(self, population):
        """Extract non-dominated solutions from population"""
        if not population:
            return []
        
        # Extract fitness values
        fitness_values = np.array([ind[1] for ind in population])
        ranks, fronts = non_dominated_sort_MAEO(fitness_values, self.mode)
        
        if not fronts[0]:
            return []
        
        pareto_front = [population[i] for i in fronts[0]]
        return pareto_front
    
    def calculate_hypervolume(self, pareto_front, reference_point):
        """Calculate hypervolume of pareto front"""
        if not pareto_front or reference_point is None:
            return 0.0
        
        # Extract fitness values
        fitness_values = [ind[1] for ind in pareto_front]
        return hypervolume_nd(fitness_values, reference_point, self.mode)
    
    def update_reference_point(self, all_populations):
        """Set reference nadir point"""
        
        """Uncomment this block if you want to set the reference once from first cycle data 
        if self.reference_point is not None:
            return self.reference_point
        """
        all_fitness = []
        for pop in all_populations:
            for ind in pop:
                all_fitness.append(ind[1])
        
        if not all_fitness:
            return None
        
        all_fitness = np.array(all_fitness)
        
        if self.mode == "min":
            reference = np.max(all_fitness, axis=0)
        else:
            reference = np.min(all_fitness, axis=0)
        
        # Set reference point once and keep it fixed
        self.reference_point = reference.copy()
        return self.reference_point
    
    def calculate_migration_parameters(self, hypervolume_improvements, cycle, max_cycles):
        """Calculate migration parameters based on hypervolume improvements"""
        # Normalize hypervolume improvements
        hv_array = np.array(hypervolume_improvements)
        if np.max(hv_array) == np.min(hv_array):
            g_normalized = np.ones_like(hv_array) * 0.5
        else:
            g_normalized = (hv_array - np.min(hv_array)) / (np.max(hv_array) - np.min(hv_array))
        
        # Calculate alpha and q
        alpha = (cycle - 1) / (max_cycles - 1) if max_cycles > 1 else 0
        q = 2 * (1 - cycle) / (1 - max_cycles) - 1 if max_cycles > 1 else 0
        
        # Calculate h_i (proportion to remove from each population)
        g_alpha = g_normalized ** alpha
        sum_g_alpha = np.sum(g_alpha)
        if sum_g_alpha == 0:
            h_i = np.ones_like(g_alpha) * 0.1  # Conservative migration rate
        else:
            h_i = 0.1 + g_alpha / sum_g_alpha * 0.3  # Scale migration between 10-40%
        
        # Ensure h_i is in valid range [0, 1]
        h_i = np.clip(h_i, 0, 0.5)  # Max 50% migration
        
        return g_normalized, h_i, alpha
    
    def select_individuals_to_migrate(self, population, num_to_remove):
        """Select individuals to migrate based on performance scores"""
        if num_to_remove == 0 or len(population) <= num_to_remove:
            return population.copy(), []
        
        # Extract fitness values
        fitness_values = np.array([ind[1] for ind in population])
        performance_scores = ind_performance_metric(fitness_values, self.mode)
        
        # Sort by performance (higher is better for both min and max)
        sorted_indices = np.argsort(-performance_scores)
        
        # Select worst performers for migration
        selected_indices = sorted_indices[-num_to_remove:]
        
        migrants = [population[i] for i in selected_indices]
        remaining = [population[i] for i in range(len(population)) if i not in selected_indices]
        
        return remaining, migrants

    def disable_underpopulated_islands(self, optimizer_instances, populations, niche_size=1):
        """
        Disable islands whose population size is less than the given niching size.
        Migrates their remaining individuals to surviving islands.
        """
        updated_optimizers = []
        updated_populations = []
        rescued_individuals = []

        for i, (opt, pop) in enumerate(zip(optimizer_instances, populations)):
            if len(pop) < niche_size:
                rescued_individuals.extend(pop)
            else:
                updated_optimizers.append(opt)
                updated_populations.append(pop)

        if rescued_individuals and updated_populations:
            num_islands = len(updated_populations)
            for idx, ind in enumerate(rescued_individuals):
                updated_populations[idx % num_islands].append(ind)

        return updated_optimizers, updated_populations
    
    def distribute_migrants(self, all_migrants, g_normalized, alpha):
        """Distribute migrants to populations using multinomial sampling"""
        if not all_migrants:
            return [[] for _ in range(len(g_normalized))]
        
        K = len(all_migrants)
        
        # Calculate probabilities for multinomial distribution
        g_alpha = g_normalized ** alpha
        sum_g_alpha = np.sum(g_alpha)
        
        if sum_g_alpha == 0:
            probs = np.ones(len(g_normalized)) / len(g_normalized)
        else:
            probs = g_alpha / sum_g_alpha
        
        # Sample from multinomial distribution
        counts = multinomial_sample(K, probs)
        
        # Shuffle migrants and distribute
        random.shuffle(all_migrants)
        distributions = []
        start_idx = 0
        
        for count in counts:
            distributions.append(all_migrants[start_idx:start_idx + count])
            start_idx += count
        
        return distributions
    
    def store_cumulative_history(self, cycle, populations):
        """Store complete history of all solutions from all islands for this cycle"""
        cycle_data = []
        
        for island_id, population in enumerate(populations):
            for individual in population:
                solution = individual[0]  # Decision variables
                fitness = individual[1]   # Objective values
                
                cycle_data.append({
                    'cycle': cycle,
                    'island': island_id + 1,
                    'solution': solution.copy() if isinstance(solution, list) else solution,
                    'fitness': fitness.copy() if isinstance(fitness, list) else fitness
                })
        
        self.cumulative_history.extend(cycle_data)
    
    def evolute(self, max_cycles, x0=None, verbose=False):
        """
        This function evolutes the NSGA-III algorithm for number of generations.
        
        :param max_cycles: (int) number of cycles to evolute the islands
        :param x0: (list of lists) initial positions of individuals in problem space
        :param verbose: (bool) print statistics to screen
        
        :return: (tuple) (best individuals, best fitnesses, and a dictionary of history)
        """
        # Validate x0 if provided
        if x0 is not None:
            if len(x0) != len(self.optimizers):
                raise ValueError(f"x0 must have {len(self.optimizers)} sublists (one for each island), but got {len(x0)}")
        
        # Initialize optimizer instances
        initial_lambda_list = [self.lambda_] * len(self.optimizers)
        self.optimizer_instances = self.create_optimizer_instances(initial_lambda_list)
        
        # Set initial populations from x0 if provided
        if x0 is not None:
            self.populations = []
            for i, island_x0 in enumerate(x0):
                # Create population format [solution, fitness] with dummy fitness
                # The actual fitness will be calculated in the first evolute call
                population = [[solution, None] for solution in island_x0]
                self.populations.append(population)
        
        previous_hypervolumes = [0.0] * len(self.optimizer_instances)
        
        for cycle in range(1, max_cycles + 1):
            # Update optimizer populations with current populations (after migration or x0)
            if self.populations:
                self.update_optimizer_populations(self.populations)
            
            # Run all islands in parallel (no changes needed here)
            raw_results = Parallel(n_jobs=self.upcores, backend="loky")(
                delayed(self.run_island)(opt, i) for i, opt in enumerate(self.optimizer_instances)
            )
            
            # Parse results to get proper population format
            self.populations = [self.parse_population(result) for result in raw_results]
            
            # Update global reference point
            self.reference_point = self.update_reference_point(self.populations)
            
            # Calculate hypervolumes and improvements
            current_hypervolumes = []
            hypervolume_improvements = []
            
            for i, pop in enumerate(self.populations):
                pareto_front = self.extract_pareto_front(pop)
                if pareto_front and self.reference_point is not None:
                    hv = self.calculate_hypervolume(pareto_front, self.reference_point)
                else:
                    hv = 0.0
                
                current_hypervolumes.append(hv)
                improvement = hv - previous_hypervolumes[i]
                hypervolume_improvements.append(improvement)
            
            # Store cumulative history
            self.store_cumulative_history(cycle, self.populations)
            
            # Migration phase (skip for last cycle)
            if cycle < max_cycles:
                g_normalized, h_i, alpha = self.calculate_migration_parameters(
                    hypervolume_improvements, cycle, max_cycles
                )
                
                # Collect migrants
                all_migrants = []
                new_populations = []
                
                for i, (pop, h_val) in enumerate(zip(self.populations, h_i)):
                    num_to_remove = int(h_val * len(pop))
                    remaining, migrants = self.select_individuals_to_migrate(pop, num_to_remove)
                    new_populations.append(remaining)
                    all_migrants.extend(migrants)
                
                # Distribute migrants
                if all_migrants:
                    distributions = self.distribute_migrants(all_migrants, g_normalized, alpha)
                    
                    # Add migrants to populations
                    for i, migrants in enumerate(distributions):
                        new_populations[i].extend(migrants)
                
                self.populations = new_populations
                
                # Handle underpopulated islands
                self.optimizer_instances, self.populations = self.disable_underpopulated_islands(
                    self.optimizer_instances, self.populations, niche_size=1
                )
                
                # Recreate optimizer instances with updated lambda values
                new_lambda_list = [len(pop) for pop in self.populations]
                self.optimizer_instances = self.create_optimizer_instances(new_lambda_list)
                
                # Update optimizer populations for next cycle
                self.update_optimizer_populations(self.populations)
            
            previous_hypervolumes = current_hypervolumes.copy()
        
        # Prepare final results
        # Extract final global Pareto front from all discovered solutions
        all_discovered_solutions = []
        for record in self.cumulative_history:
            all_discovered_solutions.append([record['solution'], record['fitness']])
        
        final_pareto_front = self.extract_pareto_front(all_discovered_solutions)
        
        # Extract solutions and fitness values
        pareto_solutions = [ind[0] for ind in final_pareto_front]
        pareto_fitness = [ind[1] for ind in final_pareto_front]
        
        # Prepare metadata
        metadata = {
            'reference_point': self.reference_point.tolist() if self.reference_point is not None else None,
            'local_pop': self.cumulative_history,  # Complete history with cycle info
            'mode': self.mode,
            'total_cycles': max_cycles,
            'total_solutions_evaluated': len(self.cumulative_history),
            'final_pareto_size': len(final_pareto_front),
            'bounds': self.bounds,
            'optimizers': self.optimizers
        }
        
        return [pareto_solutions, pareto_fitness, metadata]