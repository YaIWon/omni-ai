"""
GENETIC ALGORITHM OPTIMIZER FOR SELF-EVOLUTION
"""
import numpy as np
from deap import base, creator, tools, algorithms
import random
from typing import List, Dict, Callable, Tuple
import multiprocessing

class MultiObjectiveGeneticOptimizer:
    """Multi-objective genetic algorithm optimizer"""
    
    def __init__(self, 
                 fitness_function: Callable,
                 gene_structure: Dict,
                 population_size: int = 100,
                 generations: int = 50,
                 cx_prob: float = 0.7,
                 mut_prob: float = 0.2):
        
        self.fitness_function = fitness_function
        self.gene_structure = gene_structure
        self.population_size = population_size
        self.generations = generations
        
        # Setup DEAP
        self._setup_deap()
        
        # Statistics
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)
    
    def optimize(self) -> Dict:
        """Run multi-objective optimization"""
        
        # Create initial population
        pop = self.toolbox.population(n=self.population_size)
        
        # Evaluate initial population
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Run NSGA-II algorithm
        pop, logbook = algorithms.eaMuPlusLambda(
            pop, self.toolbox,
            mu=self.population_size,
            lambda_=self.population_size * 2,
            cxpb=self.cx_prob, mutpb=self.mut_prob,
            ngen=self.generations,
            stats=self.stats, halloffame=None,
            verbose=True
        )
        
        # Get Pareto front
        pareto_front = tools.sortNondominated(pop, len(pop))[0]
        
        return {
            'pareto_front': pareto_front,
            'logbook': logbook,
            'best_solutions': self._extract_best_solutions(pareto_front)
        }
