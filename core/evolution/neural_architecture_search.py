"""
NEURAL ARCHITECTURE SEARCH FOR SELF-EVOLUTION
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
import random
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json

class LayerType(Enum):
    """Neural network layer types"""
    LINEAR = "linear"
    CONV1D = "conv1d"
    CONV2D = "conv2d"
    LSTM = "lstm"
    GRU = "gru"
    ATTENTION = "attention"
    TRANSFORMER = "transformer"
    DROPOUT = "dropout"
    BATCHNORM = "batchnorm"
    LAYERNORM = "layernorm"

@dataclass
class LayerGene:
    """Gene representing a neural network layer"""
    layer_type: LayerType
    parameters: Dict
    activation: str = "relu"
    normalization: Optional[str] = None
    dropout: float = 0.0
    output_shape: Tuple = None
    
    def to_dict(self):
        return {
            "layer_type": self.layer_type.value,
            "parameters": self.parameters,
            "activation": self.activation,
            "normalization": self.normalization,
            "dropout": self.dropout,
            "output_shape": self.output_shape
        }
    
    def hash(self) -> str:
        """Unique hash for layer configuration"""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

@dataclass
class ArchitectureGene:
    """Complete neural architecture gene"""
    layers: List[LayerGene]
    connections: List[Tuple[int, int]]  # (from_layer_idx, to_layer_idx)
    input_shape: Tuple
    output_shape: Tuple
    fitness: float = 0.0
    generation: int = 0
    parent_hashes: List[str] = field(default_factory=list)
    
    def to_network(self) -> nn.Module:
        """Convert gene to PyTorch module"""
        return NeuralArchitecture(self)
    
    def mutate(self, mutation_rate: float = 0.1) -> 'ArchitectureGene':
        """Apply mutations"""
        mutated = ArchitectureGene(
            layers=[layer for layer in self.layers],
            connections=[conn for conn in self.connections],
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            generation=self.generation + 1,
            parent_hashes=[self.hash()]
        )
        
        # Layer mutations
        for i, layer in enumerate(mutated.layers):
            if random.random() < mutation_rate:
                mutated.layers[i] = self._mutate_layer(layer)
        
        # Connection mutations
        if random.random() < mutation_rate:
            if mutated.connections and random.random() < 0.5:
                # Remove connection
                idx = random.randint(0, len(mutated.connections) - 1)
                mutated.connections.pop(idx)
            else:
                # Add connection
                from_idx = random.randint(0, len(mutated.layers) - 1)
                to_idx = random.randint(0, len(mutated.layers) - 1)
                if from_idx != to_idx and (from_idx, to_idx) not in mutated.connections:
                    mutated.connections.append((from_idx, to_idx))
        
        # Add/remove layers
        if random.random() < mutation_rate * 0.5:
            if len(mutated.layers) < 20:  # Max layers
                mutated.layers.insert(
                    random.randint(0, len(mutated.layers)),
                    self._random_layer()
                )
        
        if random.random() < mutation_rate * 0.5 and len(mutated.layers) > 1:
            mutated.layers.pop(random.randint(0, len(mutated.layers) - 1))
        
        return mutated
    
    def crossover(self, other: 'ArchitectureGene') -> 'ArchitectureGene':
        """Crossover with another architecture"""
        # Single-point crossover
        crossover_point = random.randint(1, min(len(self.layers), len(other.layers)) - 1)
        
        new_layers = (
            self.layers[:crossover_point] + 
            other.layers[crossover_point:]
        )
        
        # Combine connections
        new_connections = list(set(self.connections + other.connections))
        
        child = ArchitectureGene(
            layers=new_layers,
            connections=new_connections,
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            generation=max(self.generation, other.generation) + 1,
            parent_hashes=[self.hash(), other.hash()]
        )
        
        return child
    
    def hash(self) -> str:
        """Unique architecture hash"""
        layers_hash = "".join([layer.hash() for layer in self.layers])
        conns_hash = "".join([f"{f}-{t}" for f, t in self.connections])
        return hashlib.sha256(f"{layers_hash}|{conns_hash}".encode()).hexdigest()[:32]

class NeuralArchitectureSearch:
    """Neural Architecture Search with evolutionary algorithms"""
    
    def __init__(self, 
                 population_size: int = 50,
                 input_shape: Tuple = (1, 784),
                 output_shape: Tuple = (10,),
                 objectives: List[str] = None):
        
        self.population_size = population_size
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.objectives = objectives or ["accuracy", "latency", "memory"]
        
        self.population: List[ArchitectureGene] = []
        self.archive: Dict[str, ArchitectureGene] = {}  # Pareto front
        self.generation = 0
        
        # Initialize population
        self._initialize_population()
    
    def _initialize_population(self):
        """Initialize random population"""
        for _ in range(self.population_size):
            # Random number of layers
            num_layers = random.randint(3, 10)
            layers = []
            
            for i in range(num_layers):
                layer = self._random_layer()
                layers.append(layer)
            
            # Create random connections (feedforward for now)
            connections = [(i, i+1) for i in range(num_layers - 1)]
            
            arch = ArchitectureGene(
                layers=layers,
                connections=connections,
                input_shape=self.input_shape,
                output_shape=self.output_shape,
                generation=0
            )
            
            self.population.append(arch)
    
    def evolve(self, 
               fitness_function, 
               generations: int = 100,
               mutation_rate: float = 0.2,
               crossover_rate: float = 0.7) -> Dict:
        """Evolve architectures"""
        
        history = {
            "best_fitness": [],
            "population_diversity": [],
            "pareto_front_size": [],
            "architectures_evaluated": 0
        }
        
        for gen in range(generations):
            self.generation = gen
            
            # Evaluate fitness
            fitness_scores = []
            for arch in self.population:
                fitness = fitness_function(arch)
                arch.fitness = fitness
                fitness_scores.append(fitness)
                
                # Update archive (Pareto front)
                self._update_archive(arch)
            
            history["architectures_evaluated"] += len(self.population)
            history["best_fitness"].append(max(fitness_scores))
            history["population_diversity"].append(self._calculate_diversity())
            history["pareto_front_size"].append(len(self.archive))
            
            # Selection
            selected = self._select_population()
            
            # Create new population
            new_population = []
            while len(new_population) < self.population_size:
                if random.random() < crossover_rate and len(selected) >= 2:
                    # Crossover
                    parent1, parent2 = random.sample(selected, 2)
                    child = parent1.crossover(parent2)
                else:
                    # Mutation or clone
                    parent = random.choice(selected)
                    child = parent.mutate(mutation_rate)
                
                new_population.append(child)
            
            self.population = new_population[:self.population_size]
            
            # Log progress
            if gen % 10 == 0:
                print(f"Generation {gen}: Best fitness = {max(fitness_scores):.4f}")
        
        return {
            "history": history,
            "best_architecture": max(self.population, key=lambda x: x.fitness),
            "pareto_front": list(self.archive.values()),
            "final_population": self.population
        }
    
    def _update_archive(self, arch: ArchitectureGene):
        """Update Pareto archive"""
        arch_hash = arch.hash()
        
        if arch_hash not in self.archive:
            # Check if dominated by any in archive
            dominated = False
            to_remove = []
            
            for existing_hash, existing_arch in self.archive.items():
                if self._dominates(existing_arch, arch):
                    dominated = True
                    break
                elif self._dominates(arch, existing_arch):
                    to_remove.append(existing_hash)
            
            if not dominated:
                # Remove dominated architectures
                for hash_to_remove in to_remove:
                    del self.archive[hash_to_remove]
                
                # Add new architecture
                self.archive[arch_hash] = arch
    
    def _dominates(self, arch1: ArchitectureGene, arch2: ArchitectureGene) -> bool:
        """Check if arch1 dominates arch2"""
        # Multi-objective domination
        # For simplicity, assume higher fitness is better
        return arch1.fitness > arch2.fitness
    
    def _select_population(self) -> List[ArchitectureGene]:
        """Select parents using tournament selection"""
        selected = []
        tournament_size = 3
        
        for _ in range(self.population_size):
            tournament = random.sample(self.population, tournament_size)
            winner = max(tournament, key=lambda x: x.fitness)
            selected.append(winner)
        
        return selected
    
    def _calculate_diversity(self) -> float:
        """Calculate population diversity"""
        if len(self.population) <= 1:
            return 0.0
        
        hashes = [arch.hash() for arch in self.population]
        unique_hashes = len(set(hashes))
        return unique_hashes / len(self.population)

class NeuralArchitecture(nn.Module):
    """Dynamic neural network from architecture gene"""
    
    def __init__(self, gene: ArchitectureGene):
        super().__init__()
        self.gene = gene
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleDict()
        self.normalizations = nn.ModuleDict()
        
        # Build layers
        for i, layer_gene in enumerate(gene.layers):
            layer = self._create_layer(layer_gene)
            self.layers.append(layer)
            
            # Add activation
            if layer_gene.activation:
                activation = self._get_activation(layer_gene.activation)
                self.activations[f"act_{i}"] = activation
            
            # Add normalization
            if layer_gene.normalization:
                norm = self._get_normalization(layer_gene)
                self.normalizations[f"norm_{i}"] = norm
        
        # Build connection graph
        self.connection_matrix = self._build_connection_matrix()
    
    def forward(self, x):
        """Forward pass with dynamic connections"""
        # Track outputs of each layer
        layer_outputs = {}
        
        # Process through computational graph
        visited = set()
        
        def process_layer(idx):
            if idx in visited:
                return layer_outputs.get(idx)
            
            visited.add(idx)
            
            # Get input from connected layers
            inputs = []
            for from_idx, to_idx in self.gene.connections:
                if to_idx == idx:
                    input_tensor = process_layer(from_idx)
                    if input_tensor is not None:
                        inputs.append(input_tensor)
            
            if not inputs:
                # Start layer (input)
                layer_input = x
            else:
                # Combine inputs (concatenate for now)
                layer_input = torch.cat(inputs, dim=1)
            
            # Apply layer
            layer_output = self.layers[idx](layer_input)
            
            # Apply activation
            if f"act_{idx}" in self.activations:
                layer_output = self.activations[f"act_{idx}"](layer_output)
            
            # Apply normalization
            if f"norm_{idx}" in self.normalizations:
                layer_output = self.normalizations[f"norm_{idx}"](layer_output)
            
            # Apply dropout
            if self.gene.layers[idx].dropout > 0:
                layer_output = F.dropout(layer_output, self.gene.layers[idx].dropout)
            
            layer_outputs[idx] = layer_output
            return layer_output
        
        # Process all layers
        outputs = []
        for i in range(len(self.layers)):
            output = process_layer(i)
            if output is not None:
                outputs.append(output)
        
        # Combine outputs (average for now)
        if outputs:
            final_output = torch.mean(torch.stack(outputs), dim=0)
        else:
            final_output = x
        
        return final_output
