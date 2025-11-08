import numpy as np
import random
from typing import List, Tuple, Optional
from dataclasses import dataclass
import pickle
import copy
import time
import pygame
from enum import Enum


# ============================================================================
# NEURAL NETWORK
# ============================================================================

input_size = 55  # 8 sectors × 6 values + 7 self stats
output_size = 6

class ActivationType(Enum):
    """Available activation function types."""
    TANH = "tanh"
    RELU = "relu"
    SIGMOID = "sigmoid"
    LEAKY_RELU = "leaky_relu"
    SWISH = "swish"
    LINEAR = "linear"

class ActivationFunctions:
    """Collection of activation functions and their derivatives."""
    
    @staticmethod
    def activate(x: np.ndarray, activation_type: ActivationType) -> np.ndarray:
        """Apply activation function."""
        if activation_type == ActivationType.TANH:
            return np.tanh(x)
        elif activation_type == ActivationType.RELU:
            return np.maximum(0, x)
        elif activation_type == ActivationType.SIGMOID:
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        elif activation_type == ActivationType.LEAKY_RELU:
            return np.where(x > 0, x, x * 0.01)
        elif activation_type == ActivationType.SWISH:
            return x * (1 / (1 + np.exp(-np.clip(x, -500, 500))))
        elif activation_type == ActivationType.LINEAR:
            return x
        else:
            return np.tanh(x)  # Default
        
class RecurrentNeuralNetwork:
    """
    Recurrent neural network with reservoir computing architecture.
    Creates a 'soup of neurons' with recurrent connections.
    """
    
    def __init__(self, input_size: int, reservoir_size: int, output_size: int,
                 spectral_radius: float = 0.9, sparsity: float = 0.1, input_scaling: float = 1.0):
        """
        Initialize a recurrent neural network.
        
        Args:
            input_size: Number of input neurons
            reservoir_size: Size of recurrent reservoir layer
            output_size: Number of output neurons
            spectral_radius: Controls stability (0.9 = good default)
            sparsity: How sparse reservoir connections are (0.1 = 10% connected)
            input_scaling: Scaling factor for input weights
        """
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        
        # Reservoir state (neuron activations)
        self.reservoir_state = np.zeros(reservoir_size)
        
        # Input -> Reservoir weights (UPDATED: add sparsity mask)
        self.input_weights = np.random.randn(input_size, reservoir_size) * input_scaling
        # Make 50% of input connections zero (allows evolution to add connections where needed)
        input_mask = np.random.random((input_size, reservoir_size)) < 0.5
        self.input_weights *= input_mask
        
        # Reservoir -> Reservoir weights (the "soup")
        self.reservoir_weights = self._initialize_reservoir_weights()
        
        # Reservoir -> Output weights (UPDATED: start with more connections)
        self.output_weights = np.random.randn(reservoir_size, output_size) * 0.3
        # Start with 30% of reservoir neurons connected to each output
        # This gives outputs more information to work with initially
        output_mask = np.random.random((reservoir_size, output_size)) < 0.3
        self.output_weights *= output_mask
        
        self.output_bias = np.random.randn(output_size) * 0.5
        
        # Activation functions for each reservoir neuron
        self.reservoir_activations = [
            random.choice(list(ActivationType)) 
            for _ in range(reservoir_size)
        ]
        
        # Output activation functions
        self.output_activations = [ActivationType.TANH] * output_size

        self.last_input = None
    
    def _initialize_reservoir_weights(self) -> np.ndarray:
        """
        Initialize sparse recurrent reservoir weights with controlled spectral radius.
        This creates the 'neural soup' effect.
        """
        # Create sparse random matrix
        W = np.random.randn(self.reservoir_size, self.reservoir_size)
        
        # Apply sparsity
        mask = np.random.rand(self.reservoir_size, self.reservoir_size) < self.sparsity
        W = W * mask
        
        # Scale by spectral radius for stability
        eigenvalues = np.linalg.eigvals(W)
        max_eigenvalue = np.max(np.abs(eigenvalues))
        if max_eigenvalue > 0:
            W = W * (self.spectral_radius / max_eigenvalue)
        
        return W
    
    def forward(self, inputs: np.ndarray, leak_rate: float = 0.3) -> np.ndarray:
        """
        Forward pass through recurrent network.
        
        Args:
            inputs: Input vector
            leak_rate: How much previous state affects current (0.3 = 30% leak)
            
        Returns:
            Output vector
        """

        self.last_input = inputs.copy()
        # Input -> Reservoir
        input_activation = np.dot(inputs, self.input_weights)
        
        # Reservoir recurrent update (the magic happens here)
        reservoir_input = input_activation + np.dot(self.reservoir_state, self.reservoir_weights)
        
        # Apply activation functions per neuron
        new_state = np.zeros(self.reservoir_size)
        for i in range(self.reservoir_size):
            new_state[i] = ActivationFunctions.activate(
                reservoir_input[i:i+1], 
                self.reservoir_activations[i]
            )[0]
        
        # Leak integration (combines old state with new)
        self.reservoir_state = (1 - leak_rate) * self.reservoir_state + leak_rate * new_state
        
        # Reservoir -> Output
        output_activation = np.dot(self.reservoir_state, self.output_weights) + self.output_bias
        
        # Apply output activations
        outputs = np.zeros(self.output_size)
        for i in range(self.output_size):
            outputs[i] = ActivationFunctions.activate(
                output_activation[i:i+1],
                self.output_activations[i]
            )[0]
        
        return outputs
    
    def reset_state(self):
        """Reset reservoir state (call at start of episode)."""
        self.reservoir_state = np.zeros(self.reservoir_size)
    
    def mutate(self, mutation_rate: float = 0.1, mutation_strength: float = 0.3,
           structural_mutation_rate: float = 0.05, activation_mutation_rate: float = 0.05):
        """
        Mutate network weights, structure, and activation functions.
        
        Args:
            mutation_rate: Probability of mutating each weight
            mutation_strength: Standard deviation of weight mutations
            structural_mutation_rate: Probability of structural changes
            activation_mutation_rate: Probability of changing activation function
        """
        # === WEIGHT MUTATIONS ===
        # Mutate input weights
        mask = np.random.random(self.input_weights.shape) < mutation_rate
        self.input_weights += mask * np.random.randn(*self.input_weights.shape) * mutation_strength
        
        # Mutate output weights
        mask = np.random.random(self.output_weights.shape) < mutation_rate
        self.output_weights += mask * np.random.randn(*self.output_weights.shape) * mutation_strength
        
        # Mutate output bias
        mask = np.random.random(self.output_bias.shape) < mutation_rate
        self.output_bias += mask * np.random.randn(*self.output_bias.shape) * mutation_strength
        
        # Mutate reservoir weights (careful - affects stability)
        mask = np.random.random(self.reservoir_weights.shape) < mutation_rate * 0.3  # Lower rate
        self.reservoir_weights += mask * np.random.randn(*self.reservoir_weights.shape) * mutation_strength * 0.5
        
        # Re-scale spectral radius after mutation
        eigenvalues = np.linalg.eigvals(self.reservoir_weights)
        max_eigenvalue = np.max(np.abs(eigenvalues))
        if max_eigenvalue > 0:
            self.reservoir_weights = self.reservoir_weights * (self.spectral_radius / max_eigenvalue)
        
        # === ACTIVATION FUNCTION MUTATIONS ===
        for i in range(self.reservoir_size):
            if random.random() < activation_mutation_rate:
                self.reservoir_activations[i] = random.choice(list(ActivationType))
        
        for i in range(self.output_size):
            if random.random() < activation_mutation_rate:
                self.output_activations[i] = random.choice(list(ActivationType))
        
        # === STRUCTURAL MUTATIONS ===
        if random.random() < structural_mutation_rate:
            mutation_type = random.choice([
                'add_reservoir_connection',
                'remove_reservoir_connection',
                'add_input_connection',      # NEW
                'remove_input_connection',   # NEW
                'add_output_connection',     # NEW
                'remove_output_connection',  # NEW
                'resize_reservoir',
                'mutate_spectral_radius'
            ])
            
            if mutation_type == 'add_reservoir_connection':
                # Add new reservoir connection
                i, j = random.randint(0, self.reservoir_size-1), random.randint(0, self.reservoir_size-1)
                self.reservoir_weights[i, j] = np.random.randn() * 0.5
                
            elif mutation_type == 'remove_reservoir_connection':
                # Remove reservoir connection
                i, j = random.randint(0, self.reservoir_size-1), random.randint(0, self.reservoir_size-1)
                self.reservoir_weights[i, j] = 0
            
            # NEW: Input-to-reservoir connection mutations
            elif mutation_type == 'add_input_connection':
                # Add new input-to-reservoir connection
                # input_weights shape: (input_size, reservoir_size)
                inp_idx = random.randint(0, self.input_size - 1)
                res_idx = random.randint(0, self.reservoir_size - 1)
                self.input_weights[inp_idx, res_idx] = np.random.randn() * 0.3
                
            elif mutation_type == 'remove_input_connection':
                # Remove input-to-reservoir connection
                inp_idx = random.randint(0, self.input_size - 1)
                res_idx = random.randint(0, self.reservoir_size - 1)
                self.input_weights[inp_idx, res_idx] = 0
            
            # NEW: Reservoir-to-output connection mutations
            elif mutation_type == 'add_output_connection':
                # Add new reservoir-to-output connection
                # output_weights shape: (reservoir_size, output_size)
                res_idx = random.randint(0, self.reservoir_size - 1)
                out_idx = random.randint(0, self.output_size - 1)
                self.output_weights[res_idx, out_idx] = np.random.randn() * 0.3
                
            elif mutation_type == 'remove_output_connection':
                # Remove reservoir-to-output connection
                res_idx = random.randint(0, self.reservoir_size - 1)
                out_idx = random.randint(0, self.output_size - 1)
                self.output_weights[res_idx, out_idx] = 0

            elif mutation_type == 'mutate_spectral_radius':
                # Adjust spectral radius by a small amount
                old_radius = self.spectral_radius
                
                # Mutate: ±0.05 to ±0.15
                delta = random.uniform(-0.15, 0.15)
                new_radius = self.spectral_radius + delta
                
                # Clamp to reasonable range
                # < 1.0 = stable/fading memory
                # = 1.0 = edge of chaos (ideal for many tasks)
                # > 1.0 = unstable/chaotic
                self.spectral_radius = np.clip(new_radius, 0.5, 1.2)
                
                # Rescale reservoir weights to new spectral radius
                eigenvalues = np.linalg.eigvals(self.reservoir_weights)
                max_eigenvalue = np.max(np.abs(eigenvalues))
                if max_eigenvalue > 0:
                    self.reservoir_weights = self.reservoir_weights * (self.spectral_radius / max_eigenvalue)
                
            elif mutation_type == 'resize_reservoir':
                # Change reservoir size slightly
                size_change = random.choice([-2, -1, 1, 2])
                new_size = max(10, min(200, self.reservoir_size + size_change))
                if new_size != self.reservoir_size:
                    self._resize_reservoir(new_size)
    
    def _resize_reservoir(self, new_size: int):
        """Resize the reservoir (grow or shrink)."""
        old_size = self.reservoir_size
        
        if new_size > old_size:
            # Growing - add new neurons
            diff = new_size - old_size
            
            # Expand input weights
            new_input_weights = np.random.randn(self.input_size, diff) * 0.5
            self.input_weights = np.hstack([self.input_weights, new_input_weights])
            
            # Expand reservoir weights
            new_reservoir_weights = np.zeros((new_size, new_size))
            new_reservoir_weights[:old_size, :old_size] = self.reservoir_weights
            # Add new random connections
            new_reservoir_weights[old_size:, :] = np.random.randn(diff, new_size) * 0.3
            new_reservoir_weights[:, old_size:] = np.random.randn(new_size, diff) * 0.3
            self.reservoir_weights = new_reservoir_weights
            
            # Expand output weights
            new_output_weights = np.random.randn(diff, self.output_size) * 0.5
            self.output_weights = np.vstack([self.output_weights, new_output_weights])
            
            # Add activation functions for new neurons
            self.reservoir_activations.extend([
                random.choice(list(ActivationType)) for _ in range(diff)
            ])
            
            # Expand state
            self.reservoir_state = np.zeros(new_size)
            
        else:
            # Shrinking - remove neurons
            # Keep the most connected neurons
            connection_counts = np.sum(np.abs(self.reservoir_weights), axis=1)
            keep_indices = np.argsort(connection_counts)[-new_size:]
            
            self.input_weights = self.input_weights[:, keep_indices]
            self.reservoir_weights = self.reservoir_weights[keep_indices][:, keep_indices]
            self.output_weights = self.output_weights[keep_indices, :]
            self.reservoir_activations = [self.reservoir_activations[i] for i in keep_indices]
            self.reservoir_state = np.zeros(new_size)
        
        self.reservoir_size = new_size
    
    def clone(self) -> 'RecurrentNeuralNetwork':
        """Create a deep copy of this network."""
        cloned = RecurrentNeuralNetwork(
            self.input_size, 
            self.reservoir_size, 
            self.output_size,
            self.spectral_radius,
            self.sparsity
        )
        
        cloned.input_weights = self.input_weights.copy()
        cloned.reservoir_weights = self.reservoir_weights.copy()
        cloned.output_weights = self.output_weights.copy()
        cloned.output_bias = self.output_bias.copy()
        cloned.reservoir_activations = self.reservoir_activations.copy()
        cloned.output_activations = self.output_activations.copy()
        cloned.reservoir_state = self.reservoir_state.copy()
        
        return cloned
    
    def crossover(self, other: 'RecurrentNeuralNetwork', crossover_rate: float = 0.5) -> 'RecurrentNeuralNetwork':
        """
        Create offspring by crossing over with another network.
        """
        # Use the larger reservoir size
        if self.reservoir_size >= other.reservoir_size:
            offspring = self.clone()
            other_parent = other
        else:
            offspring = other.clone()
            other_parent = self
        
        # Crossover output weights
        if offspring.output_weights.shape == other_parent.output_weights.shape:
            mask = np.random.random(offspring.output_weights.shape) < crossover_rate
            offspring.output_weights = np.where(mask, offspring.output_weights, other_parent.output_weights)
            
            mask = np.random.random(offspring.output_bias.shape) < crossover_rate
            offspring.output_bias = np.where(mask, offspring.output_bias, other_parent.output_bias)
        
        # Crossover activation functions
        for i in range(min(len(offspring.reservoir_activations), len(other_parent.reservoir_activations))):
            if random.random() < crossover_rate:
                offspring.reservoir_activations[i] = other_parent.reservoir_activations[i]
        
        return offspring
    
    def get_complexity(self) -> int:
        """Get network complexity (total parameters)."""
        return (self.input_weights.size + 
                self.reservoir_weights.size + 
                self.output_weights.size + 
                self.output_bias.size)
    
    def get_architecture_string(self) -> str:
        """Get a string describing the network architecture."""
        # Count non-zero connections
        input_connections = np.count_nonzero(self.input_weights)
        reservoir_connections = np.count_nonzero(self.reservoir_weights)
        output_connections = np.count_nonzero(self.output_weights)
        
        return (f"RNN[{self.input_size}→{self.reservoir_size}→{self.output_size}] "
                f"SR:{self.spectral_radius:.2f} "  # NEW: Show spectral radius
                f"C:{input_connections}+{reservoir_connections}+{output_connections}")

class NeuralNetwork:
    """Feedforward neural network with variable architecture."""
    
    def __init__(self, input_size: int, hidden_layers: List[int], output_size: int):
        """
        Initialize a random neural network.
        
        Args:
            input_size: Number of input neurons
            hidden_layers: List of hidden layer sizes
            output_size: Number of output neurons
        """
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers.copy()  # Store architecture
        self.layers = [input_size] + hidden_layers + [output_size]
        self.weights = []
        self.biases = []
        
        # Initialize random weights and biases
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights and biases for current architecture."""
        self.weights = []
        self.biases = []
        
        for i in range(len(self.layers) - 1):
            weight_matrix = np.random.randn(self.layers[i], self.layers[i + 1]) * 0.5
            bias_vector = np.random.randn(self.layers[i + 1]) * 0.5
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass through the network.
        
        Args:
            inputs: Input vector
            
        Returns:
            Output vector
        """
        activation = inputs
        
        for i in range(len(self.weights)):
            z = np.dot(activation, self.weights[i]) + self.biases[i]
            activation = self._activation_function(z)
        
        return activation
    
    def _activation_function(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function (tanh by default)."""
        return np.tanh(x)
    
    def mutate(self, mutation_rate: float = 0.1, mutation_strength: float = 0.5,
           structural_mutation_rate: float = 0.1):
        """
        Mutate weights, biases, AND network structure.
        NOW ENFORCES MINIMUM 3 HIDDEN LAYERS.
        
        Args:
            mutation_rate: Probability of mutating each weight
            mutation_strength: Standard deviation of mutation
            structural_mutation_rate: Probability of structural changes
        """
        # STRUCTURAL MUTATIONS (change architecture)
        if random.random() < structural_mutation_rate:
            mutation_type = random.choice([
                'add_node', 'remove_node', 'add_layer', 'remove_layer'
            ])
            
            if mutation_type == 'add_node' and len(self.hidden_layers) > 0:
                # Add a node to a random hidden layer
                layer_idx = random.randint(0, len(self.hidden_layers) - 1)
                self.hidden_layers[layer_idx] += 1
                self._resize_network()
                
            elif mutation_type == 'remove_node' and len(self.hidden_layers) > 0:
                # Remove a node from a random hidden layer (min 2 nodes per layer)
                layer_idx = random.randint(0, len(self.hidden_layers) - 1)
                if self.hidden_layers[layer_idx] > 2:
                    self.hidden_layers[layer_idx] -= 1
                    self._resize_network()
                    
            elif mutation_type == 'add_layer':
                # Add a new hidden layer (size based on neighbors)
                if len(self.hidden_layers) == 0:
                    new_size = (self.input_size + self.output_size) // 2
                else:
                    insert_pos = random.randint(0, len(self.hidden_layers))
                    if insert_pos == 0:
                        new_size = (self.input_size + self.hidden_layers[0]) // 2
                    elif insert_pos == len(self.hidden_layers):
                        new_size = (self.hidden_layers[-1] + self.output_size) // 2
                    else:
                        new_size = (self.hidden_layers[insert_pos-1] + self.hidden_layers[insert_pos]) // 2
                    insert_pos = insert_pos
                
                new_size = max(2, new_size)  # Minimum 2 nodes
                insert_pos = random.randint(0, len(self.hidden_layers))
                self.hidden_layers.insert(insert_pos, new_size)
                self._resize_network()
                
            elif mutation_type == 'remove_layer':
                # UPDATED: Check minimum layer count BEFORE removing
                if len(self.hidden_layers) > 3:
                    # Safe to remove - still have more than 3 layers
                    layer_idx = random.randint(0, len(self.hidden_layers) - 1)
                    self.hidden_layers.pop(layer_idx)
                    self._resize_network()
                else:
                    # At minimum (3 layers) - add a layer instead of removing
                    insert_pos = random.randint(0, len(self.hidden_layers))
                    if insert_pos == 0:
                        new_size = (self.input_size + self.hidden_layers[0]) // 2
                    elif insert_pos == len(self.hidden_layers):
                        new_size = (self.hidden_layers[-1] + self.output_size) // 2
                    else:
                        new_size = (self.hidden_layers[insert_pos-1] + self.hidden_layers[insert_pos]) // 2
                    
                    new_size = max(2, new_size)
                    self.hidden_layers.insert(insert_pos, new_size)
                    self._resize_network()
        
        # WEIGHT MUTATIONS (change values)
        for i in range(len(self.weights)):
            # Mutate weights
            weight_mask = np.random.random(self.weights[i].shape) < mutation_rate
            mutations = np.random.randn(*self.weights[i].shape) * mutation_strength
            self.weights[i] += weight_mask * mutations
            
            # Mutate biases
            bias_mask = np.random.random(self.biases[i].shape) < mutation_rate
            bias_mutations = np.random.randn(*self.biases[i].shape) * mutation_strength
            self.biases[i] += bias_mask * bias_mutations
    
    def _resize_network(self):
        """Rebuild network after structural mutation, preserving weights where possible."""
        old_weights = self.weights.copy()
        old_biases = self.biases.copy()
        old_layers = self.layers.copy()
        
        # Update layer sizes
        self.layers = [self.input_size] + self.hidden_layers + [self.output_size]
        
        # Initialize new weights
        self._initialize_weights()
        
        # Copy over old weights where dimensions match
        for i in range(min(len(old_weights), len(self.weights))):
            old_shape = old_weights[i].shape
            new_shape = self.weights[i].shape
            
            # Copy overlapping weights
            rows = min(old_shape[0], new_shape[0])
            cols = min(old_shape[1], new_shape[1])
            self.weights[i][:rows, :cols] = old_weights[i][:rows, :cols]
            
            # Copy overlapping biases
            bias_size = min(old_biases[i].shape[0], self.biases[i].shape[0])
            self.biases[i][:bias_size] = old_biases[i][:bias_size]
    
    def clone(self) -> 'NeuralNetwork':
        """
        Create a deep copy of this network.
        
        Returns:
            Cloned neural network
        """
        # Create new network with same architecture
        cloned = NeuralNetwork(self.input_size, self.hidden_layers.copy(), self.output_size)
        
        # Deep copy weights and biases
        cloned.weights = [w.copy() for w in self.weights]
        cloned.biases = [b.copy() for b in self.biases]
        
        return cloned
    
    def crossover(self, other: 'NeuralNetwork', crossover_rate: float = 0.5) -> 'NeuralNetwork':
        """
        Create offspring by crossing over with another network.
        Works with different architectures by choosing one parent's structure.
        
        Args:
            other: Other parent network
            crossover_rate: Probability of taking gene from this parent vs other
            
        Returns:
            New neural network offspring
        """
        # Choose which parent's architecture to use (favor more complex)
        if len(self.hidden_layers) >= len(other.hidden_layers):
            offspring = self.clone()
            other_parent = other
        else:
            offspring = other.clone()
            other_parent = self
        
        # Try to crossover weights where architectures overlap
        min_layers = min(len(offspring.weights), len(other_parent.weights))
        
        for i in range(min_layers):
            my_shape = offspring.weights[i].shape
            other_shape = other_parent.weights[i].shape
            
            # Only crossover if shapes match
            if my_shape == other_shape:
                mask = np.random.random(my_shape) < crossover_rate
                offspring.weights[i] = np.where(mask, offspring.weights[i], other_parent.weights[i])
                
                bias_mask = np.random.random(offspring.biases[i].shape) < crossover_rate
                offspring.biases[i] = np.where(bias_mask, offspring.biases[i], other_parent.biases[i])
            else:
                # Partial crossover for mismatched shapes
                rows = min(my_shape[0], other_shape[0])
                cols = min(my_shape[1], other_shape[1])
                
                for r in range(rows):
                    for c in range(cols):
                        if random.random() < crossover_rate:
                            offspring.weights[i][r, c] = other_parent.weights[i][r, c]
                
                bias_size = min(offspring.biases[i].shape[0], other_parent.biases[i].shape[0])
                for b in range(bias_size):
                    if random.random() < crossover_rate:
                        offspring.biases[i][b] = other_parent.biases[i][b]
        
        return offspring
    
    def get_complexity(self) -> int:
        """
        Get a measure of network complexity (total number of parameters).
        
        Returns:
            Total number of weights and biases
        """
        total = 0
        for w in self.weights:
            total += w.size
        for b in self.biases:
            total += b.size
        return total
    
    def get_architecture_string(self) -> str:
        """
        Get a string representation of the architecture.
        
        Returns:
            Architecture as string, e.g., "16-12-8-4"
        """
        return "-".join(map(str, self.layers))


# ============================================================================
# CIRCLE ENTITY
# ============================================================================

@dataclass
class Circle:
    """Physical representation of an AI agent."""
    
    x: float
    y: float
    radius: float
    color: Tuple[int, int, int]
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    
    def update_position(self, dt: float = 1.0):
        """Update position based on velocity."""
        self.x += self.velocity_x * dt
        self.y += self.velocity_y * dt
    
    def distance_to(self, other: 'Circle') -> float:
        """Calculate distance to another circle."""
        dx = self.x - other.x
        dy = self.y - other.y
        return np.sqrt(dx**2 + dy**2)
    
    def distance_to_point(self, x: float, y: float) -> float:
        """Calculate distance to a point."""
        dx = self.x - x
        dy = self.y - y
        return np.sqrt(dx**2 + dy**2)
    
    def is_colliding(self, other: 'Circle') -> bool:
        """Check if colliding with another circle."""
        return self.distance_to(other) < (self.radius + other.radius)


# ============================================================================
# FOOD
# ============================================================================

@dataclass
class Food:
    """Food item that agents can consume for energy."""
    
    x: float
    y: float
    radius: float = 5.0
    energy_value: float = 50.0
    color: Tuple[int, int, int] = (0, 255, 0)  # Green
    consumed: bool = False
    
    def distance_to_point(self, x: float, y: float) -> float:
        """Calculate distance to a point."""
        dx = self.x - x
        dy = self.y - y
        return np.sqrt(dx**2 + dy**2)
    
    def is_touching_circle(self, circle: Circle) -> bool:
        """Check if a circle is touching this food."""
        distance = self.distance_to_point(circle.x, circle.y)
        return distance < (self.radius + circle.radius)
    

@dataclass
class Water:
    """Water source that agents can drink from for hydration."""
    
    x: float
    y: float
    radius: float = 20.0  # LARGER radius (was 6.0)
    water_amount: float = 5000.0  # Total water available
    max_water_amount: float = 5000.0
    water_per_drink: float = 1  # Amount given per drink
    hydration_value: float = 25.0  # How much hydration agent gets
    color: Tuple[int, int, int] = (50, 150, 255)  # Blue
    depleted: bool = False  # Changed from 'consumed' to 'depleted'
    
    def distance_to_point(self, x: float, y: float) -> float:
        """Calculate distance to a point."""
        dx = self.x - x
        dy = self.y - y
        return np.sqrt(dx**2 + dy**2)
    
    def is_touching_circle(self, circle: Circle) -> bool:
        """Check if a circle is touching this water."""
        distance = self.distance_to_point(circle.x, circle.y)
        return distance < (self.radius + circle.radius)
    
    def drink(self) -> float:
        """
        Agent drinks from this water source.
        
        Returns:
            Amount of hydration provided
        """
        if self.depleted or self.water_amount <= 0:
            return 0
        
        # Deplete water amount
        drink_amount = min(self.water_per_drink, self.water_amount)
        self.water_amount -= drink_amount
        
        # Mark as depleted if empty
        if self.water_amount <= 0:
            self.depleted = True
        
        return self.hydration_value
    
    def get_fill_percentage(self) -> float:
        """Get percentage of water remaining (0-1)."""
        return self.water_amount / self.max_water_amount


# ============================================================================
# GARDEN CLASS (NEW)
# ============================================================================
@dataclass
class Garden:
    """Garden plot that agents can create and plant food in."""
    
    x: float
    y: float
    radius: float = 15.0
    owner_id: Optional[int] = None  # Agent who created it
    planted: bool = False
    watered: bool = False
    growth_timer: int = 0
    growth_time: int = 500  # Timesteps to grow
    cooldown_timer: int = 0
    cooldown_time: int = 300  # Timesteps before garden can be used again
    food_multiplier: int = 5  # 1 food -> 5 food
    color: Tuple[int, int, int] = (139, 69, 19)  # Brown
    harvested: bool = False

    times_harvested: int = 0  # Track harvest count
    last_interaction_timestep: int = 0  # NEW: When was it last interacted with
    inactivity_threshold: int = 10000  # NEW: Remove after 10k timesteps of no interaction
    
    def distance_to_point(self, x: float, y: float) -> float:
        """Calculate distance to a point."""
        dx = self.x - x
        dy = self.y - y
        return np.sqrt(dx**2 + dy**2)
    
    def is_touching_circle(self, circle: Circle) -> bool:
        """Check if a circle is touching this garden."""
        distance = self.distance_to_point(circle.x, circle.y)
        return distance < (self.radius + circle.radius)
    
    def update(self):
        """Update garden growth."""
        self.inactivity_threshold -= 1
        
        if self.planted and self.watered and not self.harvested:
            self.growth_timer += 1

        if self.harvested and self.cooldown_timer < self.cooldown_time:
            self.cooldown_timer += 1
            
            # Reset garden when cooldown completes
            if self.cooldown_timer >= self.cooldown_time:
                self.reset()
    
    def reset(self):
        """Reset garden to be usable again."""
        self.planted = False
        self.watered = False
        self.growth_timer = 0
        self.harvested = False
        self.cooldown_timer = 0
        print(f"  ♻️  Garden at ({self.x:.0f}, {self.y:.0f}) is ready to use again!")

    def mark_interaction(self, current_timestep: int):
        """NEW: Mark that this garden was interacted with."""
        self.inactivity_threshold = 10000
    
    def plant(self, current_timestep: int):
        """NEW: Plant the garden and mark interaction."""
        if self.is_available():
            self.planted = True
            self.mark_interaction(current_timestep)
    
    def water(self, current_timestep: int):
        """NEW: Water the garden and mark interaction."""
        if self.planted and not self.watered:
            self.watered = True
            self.mark_interaction(current_timestep)

    def is_ready_to_harvest(self) -> bool:
        """Check if garden is ready to harvest."""
        return (self.planted and self.watered and 
                self.growth_timer >= self.growth_time and 
                not self.harvested)
    
    def get_growth_percentage(self) -> float:
        """Get growth percentage (0-1)."""
        if not self.planted or not self.watered:
            return 0.0
        return min(1.0, self.growth_timer / self.growth_time)
    
    def is_available(self) -> bool:
        """Check if garden is available for planting."""
        return not self.planted and not self.harvested
    
    def is_on_cooldown(self) -> bool:
        """Check if garden is on cooldown."""
        return self.harvested and self.cooldown_timer < self.cooldown_time
    
    def get_cooldown_percentage(self) -> float:
        """Get cooldown percentage (0-1)."""
        if not self.is_on_cooldown():
            return 1.0
        return self.cooldown_timer / self.cooldown_time
    
    def harvest(self, current_timestep: int) -> int:
        """
        Mark garden as harvested and return food yield.
        
        Args:
            current_timestep: Current simulation timestep
            
        Returns:
            Amount of food produced
        """
        if not self.is_ready_to_harvest():
            return 0
        
        self.harvested = True
        self.times_harvested += 1
        self.mark_interaction(current_timestep)
        
        return self.food_multiplier
    
    def should_be_removed(self, current_timestep: int) -> bool:
        """
        NEW: Check if garden should be removed due to inactivity.
        
        Args:
            current_timestep: Current simulation timestep
            
        Returns:
            True if garden has been inactive for too long
        """
        #timesteps_since_interaction = current_timestep - self.last_interaction_timestep
        #if self.inactivity_threshold < 0:
            #print(f"  💀  Removed garden at ({self.x:.0f}, {self.y:.0f}) due to inactivity")
        return self.inactivity_threshold < 0
    
    def get_color(self) -> Tuple[int, int, int]:
        """Get color based on garden state."""
        if self.is_on_cooldown():
            # Fade from gray to brown during cooldown
            progress = self.get_cooldown_percentage()
            gray = 120
            target_r, target_g, target_b = 139, 69, 19
            r = int(gray + (target_r - gray) * progress)
            g = int(gray + (target_g - gray) * progress)
            b = int(gray + (target_b - gray) * progress)
            return (r, g, b)
        elif self.harvested:
            return (120, 120, 120)  # Gray - on cooldown
        elif self.is_ready_to_harvest():
            return (0, 200, 0)  # Bright green - ready
        elif self.planted and self.watered:
            # Grow from brown to green
            progress = self.get_growth_percentage()
            r = int(139 * (1 - progress))
            g = int(69 + (200 - 69) * progress)
            b = int(19 * (1 - progress))
            return (r, g, b)
        elif self.planted and not self.watered:
            return (160, 82, 45)  # Light brown - needs water
        else:
            return (139, 69, 19)  # Brown - empty plot

# ============================================================================
# AI AGENT
# ============================================================================

class Agent:
    def __init__(self, agent_id: int, x: float, y: float, radius: float = 10.0,
                 input_size: int = input_size,
                 hidden_layers: Optional[List[int]] = None,  # Keep for compatibility
                 output_size: int = output_size, 
                 brain: Optional[RecurrentNeuralNetwork] = None,  # Changed type hint
                 generation_born: int = 1):
        """Initialize an AI agent with recurrent brain."""
        self.id = agent_id
        self.generation_born = generation_born
        self.circle = Circle(
            x=x, 
            y=y, 
            radius=radius,
            color=self._random_color()
        )
        
        # Create recurrent brain if not provided
        if brain is None:
            # Default reservoir size - can be evolved
            reservoir_size = 50  # Good starting point
            self.brain = RecurrentNeuralNetwork(
                input_size=input_size,
                reservoir_size=reservoir_size,
                output_size=output_size,
                spectral_radius=0.9,  # Stable dynamics
                sparsity=0.1  # 10% connections
            )
        else:
            self.brain = brain
        
        self.fitness = 0.0
        self.alive = True
        
        
        # Resource stats
        self.food = 100.0  # Renamed from energy
        self.max_food = 150.0
        self.food_decay_rate = 0.08
        self.movement_penalty = 0.005
        
        self.water = 100.0  # NEW: Water stat
        self.max_water = 150.0
        self.water_decay_rate = 0.08  # Water same as food, without movement penalty

        self.food_inventory = 0
        self.max_inventory = 20
        self.times_eaten_from_inventory = 0

        #garden stats
        self.gardens_created = 0
        self.gardens_planted = 0
        self.gardens_watered = 0
        self.gardens_harvested = 0
        
        self.health = 100.0  # NEW: Health stat
        self.max_health = 100.0
        self.health_decay_when_starving = 0.5  # Health loss per timestep when resources depleted
        self.health_regen_rate = 0.1  # Slow health regeneration when well-fed
        
        self.food_collected = 0
        self.water_collected = 0  # NEW: Track water consumption
        self.age = 0
        self.children = 0
        self.current_intent = 0
        self.food_shared = 0
        self.food_received = 0
        self.damage_dealt = 0
        self.damage_taken = 0
        
        # Reproduction parameters
        self.reproduction_cooldown = 3000
        self.reproduction_cooldown_time = 2000
        self.reproduction_food_threshold = 0  # Renamed from energy
        self.reproduction_water_threshold = 0  # NEW: Also need water to reproduce
        self.reproduction_food_cost = 0.0
        self.reproduction_water_cost = 0.0  # NEW: Water cost for reproduction
    
    def _random_color(self) -> Tuple[int, int, int]:
        """Generate random RGB color."""
        return (random.randint(50, 255), 
                random.randint(50, 255), 
                random.randint(50, 255))
    
    def generate_name(self):
        first_name_1 = ['Kim', 'Yorg', 'Bing', 'Bop', 'Bong', 'Tod', 'Shwing', 'Ging', 'Biff', 'Bip', 'Won', 'Boo', 'Pow', 'Baff', 'Flip', 'Shoop']
        first_name_2 = ['yorg', 'bing', 'ing', 'bop', 'bong', 'tod', 'shwing', 'ging', 'biff', 'bip', 'won', 'boo', 'pow', 'baff', ' Von', ' The', 'hep', 'grep', 'fed', 'san', 'gorp', 'shoop']
        
    def reset_brain_state(self):
        """Reset the recurrent network's internal state."""
        if hasattr(self.brain, 'reset_state'):
            self.brain.reset_state()
    
    def sense(self, environment: 'Environment') -> np.ndarray:
        """
        Gather sensory information from the environment (RADIAL SECTOR VISION).
        
        Sensor inputs (55 total):
        
        === RADIAL SECTORS (48 inputs) ===
        8 sectors × 6 values each:
        - Sector 0 (0°-45°):   [food_dist, water_dist, agent_dist, agent_intent, garden_dist, garden_status]
        - Sector 1 (45°-90°):  [food_dist, water_dist, agent_dist, agent_intent, garden_dist, garden_status]
        - Sector 2 (90°-135°): [food_dist, water_dist, agent_dist, agent_intent, garden_dist, garden_status]
        - Sector 3 (135°-180°): [food_dist, water_dist, agent_dist, agent_intent, garden_dist, garden_status]
        - Sector 4 (180°-225°): [food_dist, water_dist, agent_dist, agent_intent, garden_dist, garden_status]
        - Sector 5 (225°-270°): [food_dist, water_dist, agent_dist, agent_intent, garden_dist, garden_status]
        - Sector 6 (270°-315°): [food_dist, water_dist, agent_dist, agent_intent, garden_dist, garden_status]
        - Sector 7 (315°-360°): [food_dist, water_dist, agent_dist, agent_intent, garden_dist, garden_status]
        
        All distances normalized to 0-1 range (0 = far/none, 1 = very close)
        Agent intent: -1 (hostile) to +1 (friendly)
        Garden status: 0=empty, 0.33=planted, 0.66=watered, 1.0=ready to harvest
        Detection range: 400 pixels
        
        === SELF STATE (7 inputs) ===
        48: Health level (0-1)
        49: Food level (0-1)
        50: Water level (0-1)
        51: Food inventory (0-1)
        52: Distance to nearest wall (0-1, higher = closer)
        53: Relative age (0-1, 0=youngest, 1=oldest)
        54: Bias (always 1.0)
        """
        sensors = np.zeros(input_size)
        
        # === RADIAL SECTOR VISION ===
        num_sectors = 8
        sector_angle = 2 * np.pi / num_sectors
        max_detection_range = 400.0
        
        for sector_idx in range(num_sectors):
            angle_start = sector_idx * sector_angle
            angle_end = (sector_idx + 1) * sector_angle
            
            base_idx = sector_idx * 6
            
            # Find nearest food in this sector
            nearest_food_dist = self._find_nearest_in_sector(
                environment.food, angle_start, angle_end, max_detection_range,
                filter_fn=lambda f: not f.consumed
            )
            sensors[base_idx + 0] = nearest_food_dist
            
            # Find nearest water in this sector
            nearest_water_dist = self._find_nearest_in_sector(
                environment.water, angle_start, angle_end, max_detection_range,
                filter_fn=lambda w: not w.depleted
            )
            sensors[base_idx + 1] = nearest_water_dist
            
            # Find nearest agent in this sector (returns distance and intent)
            nearest_agent_dist, nearest_agent_intent = self._find_nearest_agent_in_sector(
                environment.agents, angle_start, angle_end, max_detection_range
            )
            sensors[base_idx + 2] = nearest_agent_dist
            sensors[base_idx + 3] = nearest_agent_intent
            
            # Find nearest garden in this sector (returns distance and status)
            nearest_garden_dist, nearest_garden_status = self._find_nearest_garden_in_sector(
                environment.gardens, angle_start, angle_end, max_detection_range
            )
            sensors[base_idx + 4] = nearest_garden_dist
            sensors[base_idx + 5] = nearest_garden_status
        
        # === SELF STATE ===
        base_idx = num_sectors * 6  # 48
        
        sensors[base_idx + 0] = self.health / self.max_health
        sensors[base_idx + 1] = self.food / self.max_food
        sensors[base_idx + 2] = self.water / self.max_water
        sensors[base_idx + 3] = self.food_inventory / self.max_inventory
        
        # Distance to nearest wall
        dist_to_left = self.circle.x
        dist_to_right = environment.width - self.circle.x
        dist_to_top = self.circle.y
        dist_to_bottom = environment.height - self.circle.y
        nearest_wall_distance = min(dist_to_left, dist_to_right, dist_to_top, dist_to_bottom)
        
        max_wall_distance = 500.0
        sensors[base_idx + 4] = 1.0 - np.clip(nearest_wall_distance / max_wall_distance, 0, 1)
        
        # Relative age (0 = youngest, 1 = oldest agent)
        alive_agents = [a for a in environment.agents if a.alive]
        if alive_agents:
            max_age = max(a.age for a in alive_agents)
            if max_age > 0:
                sensors[base_idx + 5] = self.age / max_age
            else:
                sensors[base_idx + 5] = 0.0
        else:
            sensors[base_idx + 5] = 0.0
        
        sensors[base_idx + 6] = 1.0  # Bias
    
        return sensors

    def _find_nearest_in_sector(self, objects: list, angle_start: float, angle_end: float,
                                max_distance: float, filter_fn=None, is_agent: bool = False) -> float:
        """
        Find the nearest object within a radial sector.
        
        Args:
            objects: List of objects to search
            angle_start: Start angle of sector (radians)
            angle_end: End angle of sector (radians)
            max_distance: Maximum detection distance
            filter_fn: Optional filter function
            is_agent: True if objects are agents (different position access)
            
        Returns:
            Normalized distance (0 = far/none, 1 = very close)
        """
        if filter_fn:
            objects = [obj for obj in objects if filter_fn(obj)]
        
        if not objects:
            return 0.0
        
        nearest_distance = max_distance
        
        for obj in objects:
            # Get object position
            if is_agent:
                obj_x, obj_y = obj.circle.x, obj.circle.y
            else:
                obj_x, obj_y = obj.x, obj.y
            
            # Calculate angle to object
            dx = obj_x - self.circle.x
            dy = obj_y - self.circle.y
            distance = np.sqrt(dx**2 + dy**2)
            
            # Skip if too far
            if distance > max_distance:
                continue
            
            # Calculate angle
            angle = np.arctan2(dy, dx)
            if angle < 0:
                angle += 2 * np.pi
            
            # Check if in sector (handle wraparound at 0/2π)
            in_sector = False
            if angle_end > angle_start:
                # Normal case
                in_sector = angle_start <= angle < angle_end
            else:
                # Wraparound case (sector crosses 0°)
                in_sector = angle >= angle_start or angle < angle_end
            
            if in_sector and distance < nearest_distance:
                nearest_distance = distance
        
        # Normalize: closer = higher value (1.0 at distance 0, 0.0 at max_distance)
        if nearest_distance >= max_distance:
            return 0.0
        else:
            return 1.0 - (nearest_distance / max_distance)
        
    def _find_nearest_agent_in_sector(self, agents: list, angle_start: float, angle_end: float,
                                   max_distance: float) -> Tuple[float, float]:
        """
        Find the nearest agent within a radial sector and return distance + intent.
        
        Args:
            agents: List of agents to search
            angle_start: Start angle of sector (radians)
            angle_end: End angle of sector (radians)
            max_distance: Maximum detection distance
            
        Returns:
            Tuple of (normalized_distance, intent)
            - normalized_distance: 0 = far/none, 1 = very close
            - intent: -1 = hostile, 0 = neutral, +1 = friendly
        """
        # Filter to other alive agents
        other_agents = [a for a in agents if a.id != self.id and a.alive]
        
        if not other_agents:
            return (0.0, 0.0)
        
        nearest_distance = max_distance
        nearest_agent = None
        
        for agent in other_agents:
            obj_x, obj_y = agent.circle.x, agent.circle.y
            
            # Calculate angle to agent
            dx = obj_x - self.circle.x
            dy = obj_y - self.circle.y
            distance = np.sqrt(dx**2 + dy**2)
            
            # Skip if too far
            if distance > max_distance:
                continue
            
            # Calculate angle
            angle = np.arctan2(dy, dx)
            if angle < 0:
                angle += 2 * np.pi
            
            # Check if in sector (handle wraparound at 0/2π)
            in_sector = False
            if angle_end > angle_start:
                in_sector = angle_start <= angle < angle_end
            else:
                in_sector = angle >= angle_start or angle < angle_end
            
            if in_sector and distance < nearest_distance:
                nearest_distance = distance
                nearest_agent = agent
        
        # Return normalized distance and intent
        if nearest_agent is None:
            return (0.0, 0.0)
        else:
            normalized_dist = 1.0 - (nearest_distance / max_distance)
            intent = nearest_agent.current_intent  # Get the agent's current intent
            return (normalized_dist, intent)
        
    def _find_nearest_garden_in_sector(self, gardens: list, angle_start: float, angle_end: float,
                                    max_distance: float) -> Tuple[float, float]:
        """
        Find the nearest garden within a radial sector and return distance + status.
        
        Args:
            gardens: List of gardens to search
            angle_start: Start angle of sector (radians)
            angle_end: End angle of sector (radians)
            max_distance: Maximum detection distance
            
        Returns:
            Tuple of (normalized_distance, status)
            - normalized_distance: 0 = far/none, 1 = very close
            - status: 0=empty, 0.33=planted, 0.66=watered, 1.0=ready
        """
        # Filter to non-harvested gardens
        active_gardens = [g for g in gardens if not g.harvested]
        
        if not active_gardens:
            return (0.0, 0.0)
        
        nearest_distance = max_distance
        nearest_garden = None
        
        for garden in active_gardens:
            obj_x, obj_y = garden.x, garden.y
            
            # Calculate angle to garden
            dx = obj_x - self.circle.x
            dy = obj_y - self.circle.y
            distance = np.sqrt(dx**2 + dy**2)
            
            # Skip if too far
            if distance > max_distance:
                continue
            
            # Calculate angle
            angle = np.arctan2(dy, dx)
            if angle < 0:
                angle += 2 * np.pi
            
            # Check if in sector (handle wraparound at 0/2π)
            in_sector = False
            if angle_end > angle_start:
                in_sector = angle_start <= angle < angle_end
            else:
                in_sector = angle >= angle_start or angle < angle_end
            
            if in_sector and distance < nearest_distance:
                nearest_distance = distance
                nearest_garden = garden
        
        # Return normalized distance and status
        if nearest_garden is None:
            return (0.0, 0.0)
        else:
            normalized_dist = 1.0 - (nearest_distance / max_distance)
            
            # Encode garden status
            if not nearest_garden.planted:
                status = 0.0  # Empty plot
            elif nearest_garden.planted and not nearest_garden.watered:
                status = 0.33  # Planted but needs water
            elif nearest_garden.planted and nearest_garden.watered and not nearest_garden.is_ready_to_harvest():
                status = 0.66  # Growing
            else:
                status = 1.0  # Ready to harvest
            
            return (normalized_dist, status)

    def _angle_to_target(self, target_x: float, target_y: float) -> float:
        """Calculate normalized angle to a target (-1 to 1)."""
        dx = target_x - self.circle.x
        dy = target_y - self.circle.y
        angle = np.arctan2(dy, dx)
        return angle / np.pi
    
    def think(self, sensor_inputs: np.ndarray) -> np.ndarray:
        """Process sensor inputs through neural network."""
        return self.brain.forward(sensor_inputs)
    
    def act(self, outputs: np.ndarray, environment: 'Environment'):
        """
        Convert neural network outputs to actions.
        
        Args:
            outputs: Neural network output vector (6 total)
                [0]: Movement angle (-1 to 1, maps to -π to π)
                [1]: Movement speed (-1 to 1, normalized to 0-1)
                [2]: Intent signal (-1 = hostile, +1 = friendly)
                [3]: Create garden (>0.5 triggers)
                [4]: Plant in garden (>0.5 triggers)
                [5]: Water garden (>0.5 triggers)
        """
        max_speed = 5.0
        
        # Convert angle output (-1 to 1) to radians (-π to π)
        angle = outputs[0] * np.pi
        
        # Speed (normalize from tanh output [-1, 1] to [0, 1] range)
        speed = (outputs[1] + 1) / 2
        speed = speed * max_speed
        
        # Set velocity based on angle and speed
        self.circle.velocity_x = np.cos(angle) * speed
        self.circle.velocity_y = np.sin(angle) * speed

        # Intent signal (output 2) - store for other agents to sense
        self.current_intent = np.clip(outputs[2], -1.0, 1.0)

        # Create garden
        create_garden_signal = outputs[3]
        if create_garden_signal > 0.5:
            self.try_create_garden(environment)
        
        # Plant in garden
        plant_signal = outputs[4]
        if plant_signal > 0.5:
            self.try_plant_garden(environment)
        
        # Water garden
        water_signal = outputs[5]
        if water_signal > 0.5:
            self.try_water_garden(environment)

    def try_create_garden(self, environment: 'Environment'):
        """NEW: Try to create a garden plot."""
        # Check if agent has enough food inventory
        if self.food_inventory < 10:
            return
        
        # Check if there's already a garden too close
        min_garden_distance = 50
        for garden in environment.gardens:
            if self.circle.distance_to_point(garden.x, garden.y) < min_garden_distance:
                return  # Too close to existing garden
        
        # Create garden at agent's position (slightly offset)
        garden_x = self.circle.x + random.uniform(-20, 20)
        garden_y = self.circle.y + random.uniform(-20, 20)
        
        # Keep within bounds
        garden_x = max(30, min(garden_x, environment.width - 30))
        garden_y = max(30, min(garden_y, environment.height - 30))
        
        # Spend food from inventory
        self.food_inventory -= 10
        
        # Create garden
        garden = Garden(x=garden_x, y=garden_y, owner_id=self.id)
        environment.gardens.append(garden)
        
        self.gardens_created += 1
        self.fitness += 20  # Reward for creating garden
        
        print(f"  🌱 Agent {self.id} created garden at ({garden_x:.0f}, {garden_y:.0f})")
    
    def try_plant_garden(self, environment: 'Environment'):
        """Try to plant food in a nearby garden."""
        if self.food_inventory < 1:
            return
        
        # Find nearest empty garden
        for garden in environment.gardens:
            if not garden.is_available():  # Use the new method
                continue
            
            if garden.is_touching_circle(self.circle):
                # Plant food
                self.food_inventory -= 1
                garden.plant(environment.timestep)  # UPDATED: Use new method with timestep
                
                self.gardens_planted += 1
                self.fitness += 15
                
                print(f"  🌾 Agent {self.id} planted in garden at ({garden.x:.0f}, {garden.y:.0f})")
                return

    def try_water_garden(self, environment: 'Environment'):
        """Try to water a planted garden."""
        if self.water < 30:
            return
        
        # Find nearest planted but not watered garden
        for garden in environment.gardens:
            if not garden.planted or garden.watered or garden.harvested:
                continue
            
            if garden.is_touching_circle(self.circle):
                # Water garden
                self.water -= 30
                garden.water(environment.timestep)  # UPDATED: Use new method with timestep
                
                self.gardens_watered += 1
                self.fitness += 25
                
                print(f"  💧 Agent {self.id} watered garden at ({garden.x:.0f}, {garden.y:.0f})")
                return

    def try_harvest_garden(self, environment: 'Environment'):
        """Try to harvest a ready garden."""
        for garden in environment.gardens:
            if not garden.is_ready_to_harvest():
                continue
            
            if garden.is_touching_circle(self.circle):
                # Harvest food (returns amount of food produced)
                food_gained = garden.harvest(environment.timestep)  # UPDATED: Pass timestep, returns food amount
                
                if food_gained == 0:  # Safety check
                    continue
                
                # Add harvested food to inventory or directly to food level
                for _ in range(food_gained):
                    if self.food_inventory < self.max_inventory:
                        self.food_inventory += 1
                    else:
                        self.food = min(self.food + 10, self.max_food)
                
                self.gardens_harvested += 1
                self.food_collected += food_gained
                self.fitness += 50
                
                print(f"  🎉 Agent {self.id} harvested {food_gained} food from garden at ({garden.x:.0f}, {garden.y:.0f}) - Total harvests: {garden.times_harvested}")
                
                return
    
    def update(self, environment: 'Environment', dt: float = 1.0):
        """Main update loop for the agent."""
        if not self.alive:
            return
        
        # Sense -> Think -> Act cycle
        sensor_data = self.sense(environment)
        actions = self.think(sensor_data)
        self.act(actions, environment)
        
        # Update physical state
        self.circle.update_position(dt)
        self.update_resources(dt)
        self.update_health(dt)
        self.check_boundaries(environment)
        self.age += 1

        self.try_harvest_garden(environment)

        #self.calculate_fitness()
        
        # Update reproduction cooldown
        if self.reproduction_cooldown > 0:
            self.reproduction_cooldown -= 1
        
        # Check if agent died
        if self.health <= 0:
            self.alive = False
            self.reset_brain_state()
    
    def update_resources(self, dt: float):
        """Update food and water levels (both decrease over time)."""
        # Food decreases based on movement and metabolism
        movement_cost = (abs(self.circle.velocity_x) + abs(self.circle.velocity_y)) * self.movement_penalty
        self.food -= (self.food_decay_rate + movement_cost) * dt
        self.food = max(0, min(self.food, self.max_food))
        
        # Water decreases faster than food
        water_movement_cost = (abs(self.circle.velocity_x) + abs(self.circle.velocity_y)) * 0
        self.water -= (self.water_decay_rate + water_movement_cost) * dt
        self.water = max(0, min(self.water, self.max_water))

        if self.food < 100:
            self.eat_from_inventory()
    
    def update_health(self, dt: float):
        """Update health based on food and water availability."""
        # Lose health if starving or dehydrated
        if self.food <= 0 or self.water <= 0:
            self.health -= self.health_decay_when_starving * dt
        else:
            # Regenerate health slowly if well-fed and hydrated
            if self.food > 50 and self.water > 50 and self.health < self.max_health:
                self.health += self.health_regen_rate * dt
        
        self.health = max(0, min(self.health, self.max_health))
    
    def eat_from_inventory(self):
        """NEW: Eat food from inventory to restore food level."""
        if self.food_inventory > 0 and self.food < self.max_food:
            self.food_inventory -= 1
            food_gained = 50.0  # Same as Food.energy_value
            self.food = min(self.food + food_gained, self.max_food)
            self.times_eaten_from_inventory += 1
            self.fitness += 5  # Small reward for smart eating

    def consume_food(self, food):
        """MODIFIED: Collect food into inventory instead of eating immediately."""
        if self.food_inventory < self.max_inventory:
            # Add to inventory
            self.food_inventory += 1
            self.food_collected += 1
            self.fitness += 10
        else:
            # Inventory full - eat immediately
            self.food = min(self.food + food.energy_value, self.max_food)
            self.food_collected += 1
            self.fitness += 10
    
    def consume_water(self, water):
        """Consume water and gain hydration."""
        self.water = min(self.water + water.water_value, self.max_water)
        self.water_collected += 10
        #self.fitness += 10
       
    def consume_water_amount(self, amount: float):
        """
        Consume a specific amount of water (for depletable sources).
        
        Args:
            amount: Amount of hydration to gain
        """
        self.water = self.water + amount
        # Only count as "collected" once we get a full drink
        if amount >= 20:
            self.water_collected += 1
            #self.fitness += 3
    
    def can_reproduce(self) -> bool:
        """Check if agent is ready to reproduce."""
        return (self.alive and 
                self.food >= self.reproduction_food_threshold and
                self.water >= self.reproduction_water_threshold and
                self.health > 90 and
                self.reproduction_cooldown <= 0)
    
    def reproduce_with(self, other: 'Agent', child_id: int, generation: int) -> 'Agent':
        """Create offspring with another agent."""
        # Position child between parents
        child_x = (self.circle.x + other.circle.x) / 2
        child_y = (self.circle.y + other.circle.y) / 2
        child_x += random.uniform(-20, 20)
        child_y += random.uniform(-20, 20)
        
        # Create child brain through crossover and mutation
        child_brain = self.brain.crossover(other.brain, crossover_rate=0.5)
        mutation_type = random.uniform(0, 10)
        if mutation_type > 4 and mutation_type < 8: #40% of the time give weight mutations, 40% only crossover, 20% give structural mutations (gotta have some freaks)
            child_brain.mutate(
                mutation_rate=0.1, 
                mutation_strength=0.2,
                structural_mutation_rate=0.05,
                activation_mutation_rate=0.05
            )
        elif mutation_type >= 8:
            child_brain.mutate(
                mutation_rate=0.001, 
                mutation_strength=0.2,
                structural_mutation_rate=0.95,
                activation_mutation_rate=0.005
            )

        child_brain.reset_state()
        
        # Create child agent with correct input size
        child = Agent(
            agent_id=child_id,
            x=child_x,
            y=child_y,
            input_size=input_size,  # ADDED: Explicitly set input size
            brain=child_brain,
            generation_born=generation
        )
        
        # Child starts with resources from parents
        child.food = 100
        child.water = 100
        
        # Deduct resources from parents
        self.food -= self.reproduction_food_cost
        self.water -= self.reproduction_water_cost
        other.food -= other.reproduction_food_cost
        other.water -= other.reproduction_water_cost
        
        # Set cooldown for both parents
        self.reproduction_cooldown = self.reproduction_cooldown_time
        other.reproduction_cooldown = other.reproduction_cooldown_time
        
        # Track offspring
        self.children += 1
        other.children += 1
        self.fitness += 15
        other.fitness += 15
        
        return child
    
    def check_boundaries(self, environment: 'Environment'):
        """Handle boundary collisions (bounce off walls)."""
        # Left wall
        if self.circle.x - self.circle.radius < 0:
            self.circle.x = self.circle.radius
            self.circle.velocity_x *= -0.5
        
        # Right wall
        if self.circle.x + self.circle.radius > environment.width:
            self.circle.x = environment.width - self.circle.radius
            self.circle.velocity_x *= -0.5
        
        # Top wall
        if self.circle.y - self.circle.radius < 0:
            self.circle.y = self.circle.radius
            self.circle.velocity_y *= -0.5
        
        # Bottom wall
        if self.circle.y + self.circle.radius > environment.height:
            self.circle.y = environment.height - self.circle.radius
            self.circle.velocity_y *= -0.5
    
    def calculate_fitness(self):
        """Calculate fitness score for evolution."""
        survival_score = self.age * 1
        food_score = self.food_collected * 10 + 1
        water_score = self.water_collected * 1 + 1
        resource_score = (self.food + self.water) * 0.3
        health_score = self.health * 0.5
        reproduction_score = self.children * 1000
        
        # Small complexity bonus only if agent is successful
        complexity_bonus = 0
        if self.food_collected > 2 and self.water_collected > 2:
            complexity = self.brain.get_complexity()
            complexity_bonus = np.log(complexity) * 2

        garden_bonus = self.gardens_created * 1000
        garden_bonus += self.gardens_planted * 200
        garden_bonus += self.gardens_watered * 200
        garden_bonus += self.gardens_harvested * 500  # Big bonus for farming

        resource_bonus = min(2000, food_score * water_score)

        social_bonus = self.food_shared * 50
        
        #self.fitness = survival_score + food_score + water_score + resource_score + health_score + reproduction_score + complexity_bonus
        self.fitness = survival_score + reproduction_score + complexity_bonus + garden_bonus + resource_bonus + social_bonus
        return self.fitness


# ============================================================================
# ENVIRONMENT
# ============================================================================

class Environment:
    """Main environment that manages all agents and simulation."""
    
    def __init__(self, width: int = 3000, height: int = 2000, num_agents: int = 20):
        """Initialize the environment."""
        self.width = width
        self.height = height
        self.agents: List[Agent] = []
        self.food: List[Food] = []
        self.water: List[Water] = []
        self.gardens: List[Garden] = []
        self.timestep = 0
        self.last_save_timestep = 0
        self.generation = 1
        self.next_agent_id = 0

        # Camera (will be initialized in render)
        self.camera: Optional[Camera] = None
        
        # UI state for scrolling
        self.scroll_offset = 0
        self.panel_width = 320

        # Selected agent for brain visualization
        self.selected_agent: Optional[Agent] = None
        self.show_brain_viz = False
        self.brain_viz_close_rect = None
        
        # Food spawning parameters
        self.food_spawn_interval = 30
        self.max_food = 250
        self.food_per_spawn = 10
        
        # Water spawning parameters - UPDATED
        self.water_spawn_interval = 500  # Spawn very rarely
        self.max_water = 15  # Only 10 water sources max
        self.water_per_spawn = 1  # Spawn one at a time
        
        # Population parameters
        self.max_population = 1000
        self.min_population = 10
        
        # Evolution parameters
        self.generation_length = 5000
        self.cull_percentage = 0.3
        self.mutation_rate = 0.15
        self.mutation_strength = 0.3

         # HALL OF FAME - preserve best agents across all generations
        self.hall_of_fame: List[dict] = []  # Stores {brain, fitness, generation, stats}
        self.hall_of_fame_size = 10  # Keep top 10 agents of all time
        self.best_fitness_ever = 0
        
        self._spawn_agents(num_agents)
        self._spawn_initial_food()
        self._spawn_initial_water()
        self._spawn_initial_garden()
    
    def _spawn_initial_water(self):
        """Spawn initial water sources (only 3)."""
        for _ in range(15):  # Changed from 20 to 3
            self._spawn_water()
    
    def _spawn_water(self):
        """Spawn a single water source at a random location, minimum 500px from other water."""
        active_water_count = len([w for w in self.water if not w.depleted])
        
        if active_water_count >= self.max_water:
            return
        
        min_distance = 300  # Minimum distance from other water sources
        max_attempts = 50   # Try up to 50 times to find a good spot
        
        for attempt in range(max_attempts):
            x = random.uniform(40, self.width - 40)
            y = random.uniform(40, self.height - 40)
            
            # Check distance to all existing water sources
            too_close = False
            for existing_water in self.water:
                if not existing_water.depleted:
                    dx = x - existing_water.x
                    dy = y - existing_water.y
                    distance = np.sqrt(dx**2 + dy**2)
                    
                    if distance < min_distance:
                        too_close = True
                        break
            
            # If far enough from all water sources, spawn here
            if not too_close:
                water = Water(x=x, y=y)
                self.water.append(water)
                return
        
        # If we couldn't find a spot after max_attempts, spawn anyway
        # (This handles edge cases where the world is too small or too crowded)
        x = random.uniform(40, self.width - 40)
        y = random.uniform(40, self.height - 40)
        water = Water(x=x, y=y)
        self.water.append(water)

    def _spawn_initial_garden(self):
        """NEW: Spawn one garden at the start to help AIs learn."""
        # Place garden near center of world
        garden_x = self.width / 2 + random.uniform(-100, 100)
        garden_y = self.height / 2 + random.uniform(-100, 100)
        
        garden = Garden(x=garden_x, y=garden_y, owner_id=None)
        self.gardens.append(garden)
        
        print(f"🌱 Initial tutorial garden spawned at ({garden_x:.0f}, {garden_y:.0f})")
    
    def spawn_water_periodically(self):
        """Spawn water very rarely when depleted."""
        if self.timestep % self.water_spawn_interval == 0:
            for _ in range(self.water_per_spawn):
                active_water_count = len([w for w in self.water if not w.depleted])
                if active_water_count < self.max_water:
                    x = random.uniform(40, self.width - 40)
                    y = random.uniform(40, self.height - 40)
                    water = Water(x=x, y=y)
                    self.water.append(water)
                    print(f"  💧 New water source spawned at ({x:.0f}, {y:.0f})")
                else:
                    break
    
    def _spawn_agents(self, num_agents: int, brains: Optional[List[NeuralNetwork]] = None):
        """Spawn agents at random positions."""
        for i in range(num_agents):
            x = random.uniform(50, self.width - 50)
            y = random.uniform(50, self.height - 50)
            brain = brains[i] if brains and i < len(brains) else None
            agent = Agent(
                agent_id=self.next_agent_id, 
                x=x, 
                y=y, 
                brain=brain,
                generation_born=self.generation
            )
            self.agents.append(agent)
            self.next_agent_id += 1
    
    def add_agent(self, agent: Agent):
        """Add a new agent to the environment (used for births)."""
        self.agents.append(agent)

    def update_hall_of_fame(self, print_stats=False):
        """
        Update the hall of fame with best performers from current generation.
        PREVENTS DUPLICATE AGENTS - each agent ID can only appear once.
        PRESERVES loaded Hall of Fame entries.
        """
        # Get all agents with positive fitness
        candidates = [a for a in self.agents if a.fitness > 0]
        
        # Build existing Hall of Fame lookup dictionary
        hof_by_agent_id = {}
        hof_without_id = []  # Old entries without agent_id
        
        for record in self.hall_of_fame:
            if 'agent_id' in record and record['agent_id'] is not None:
                hof_by_agent_id[record['agent_id']] = record
            else:
                # Keep old entries without IDs - they're from loaded saves
                hof_without_id.append(record)
        
        # If no good candidates this generation, keep existing Hall of Fame unchanged
        if not candidates:
            if print_stats:
                print("  ℹ️  No agents with positive fitness this generation - Hall of Fame unchanged")
            return
        
        # Sort candidates by fitness
        candidates.sort(key=lambda a: a.fitness, reverse=True)
        
        # Add top performers to hall of fame (only if not already present or if fitness improved)
        for agent in candidates[:5]:  # Consider top 5 from this generation
            agent_record = {
                'agent_id': agent.id,  # Track agent ID to prevent duplicates
                'brain': agent.brain.clone(),
                'fitness': agent.fitness,
                'generation': self.generation,
                'food_collected': agent.food_collected,
                'water_collected': agent.water_collected,
                'age': agent.age,
                'children': agent.children,
                'architecture': agent.brain.get_architecture_string(),
                'complexity': agent.brain.get_complexity()
            }
            
            if agent.id in hof_by_agent_id:
                # Agent already in hall of fame - UPDATE their entry if fitness improved
                old_fitness = hof_by_agent_id[agent.id]['fitness']
                if agent.fitness > old_fitness:
                    hof_by_agent_id[agent.id] = agent_record
                    if print_stats:
                        print(f"  🔄 Updated Agent {agent.id} in HOF: {old_fitness:.2f} → {agent.fitness:.2f}")
                # else: keep old record with better fitness
            else:
                # New agent - ADD to hall of fame
                hof_by_agent_id[agent.id] = agent_record
                if print_stats:
                    print(f"  ✨ Added Agent {agent.id} to HOF: {agent.fitness:.2f}")
        
        # Rebuild hall of fame list from dictionary + old entries
        # IMPORTANT: Preserve old entries without IDs (from loaded saves)
        self.hall_of_fame = list(hof_by_agent_id.values()) + hof_without_id
        
        # Sort hall of fame by fitness
        self.hall_of_fame.sort(key=lambda x: x['fitness'], reverse=True)
        
        # Keep only top performers
        self.hall_of_fame = self.hall_of_fame[:self.hall_of_fame_size]
        
        # Update best fitness ever
        if self.hall_of_fame:
            self.best_fitness_ever = max(self.best_fitness_ever, self.hall_of_fame[0]['fitness'])
            if print_stats:
                print(f"\n🏆 HALL OF FAME UPDATED:")
                print(f"   Best Ever: {self.best_fitness_ever:.2f} (Gen {self.hall_of_fame[0]['generation']})")
                print(f"   Full HOF:")
                for i, record in enumerate(self.hall_of_fame):
                    agent_id_str = f"Agent {record['agent_id']}" if 'agent_id' in record and record['agent_id'] is not None else "Legacy"
                    print(f"     {i+1}. {agent_id_str} | Fitness {record['fitness']:.2f} | "
                        f"Gen {record['generation']} | "
                        f"Arch: {record['architecture']} | "
                        f"Food: {record['food_collected']}, Water: {record['water_collected']}")
                
    def spawn_from_hall_of_fame(self, num_agents: int, diversity_ratio: float = 0.3):
        """
        Spawn agents using hall of fame brains.
        
        Args:
            num_agents: Number of agents to spawn
            diversity_ratio: Ratio of random agents to add (0.3 = 30% random)
        """
        if not self.hall_of_fame:
            # No hall of fame yet, spawn random
            print("   No hall of fame available, spawning random agents")
            self._spawn_random_agents(num_agents)
            return
        
        num_random = int(num_agents * diversity_ratio)
        num_from_hof = num_agents - num_random
        
        print(f"   Spawning {num_from_hof} from Hall of Fame, {num_random} random for diversity")
        
        # Spawn agents from hall of fame
        for i in range(num_from_hof):
            # Select from hall of fame (weighted towards best)
            # Use top performers more often
            if i < num_from_hof // 2:
                # First half: use top performers
                hof_idx = i % min(3, len(self.hall_of_fame))
            else:
                # Second half: use any hall of fame member
                hof_idx = random.randint(0, len(self.hall_of_fame) - 1)
            
            record = self.hall_of_fame[hof_idx]
            
            # Clone and mutate the brain
            brain = record['brain'].clone()
            brain.mutate(
                mutation_rate=0.1,
                mutation_strength=0.2,
                structural_mutation_rate=0.05
            )
            
            x = random.uniform(50, self.width - 50)
            y = random.uniform(50, self.height - 50)
            
            agent = Agent(
                agent_id=self.next_agent_id,
                x=x,
                y=y,
                input_size=input_size,
                brain=brain,
                generation_born=self.generation + 1
            )
            self.agents.append(agent)
            self.next_agent_id += 1
        
        # Spawn random agents for diversity
        self._spawn_random_agents(num_random)

    def _spawn_random_agents(self, num_agents: int):
        """Spawn completely random agents with recurrent brains."""
        for i in range(num_agents):
            x = random.uniform(50, self.width - 50)
            y = random.uniform(50, self.height - 50)
            
            # Random reservoir sizes
            reservoir_size = random.choice([30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400])
            spectral_radius = random.uniform(0.85, 0.95)
            sparsity = random.uniform(0.05, 0.15)
            
            # Create recurrent brain
            brain = RecurrentNeuralNetwork(
                input_size=input_size,
                reservoir_size=reservoir_size,
                output_size=output_size,
                spectral_radius=spectral_radius,
                sparsity=sparsity
            )
            
            agent = Agent(
                agent_id=self.next_agent_id,
                x=x,
                y=y,
                input_size=input_size,
                brain=brain,
                generation_born=self.generation + 1
            )
            self.agents.append(agent)
            self.next_agent_id += 1
    
    def _spawn_initial_food(self):
        """Spawn initial food items."""
        for _ in range(100):
            self._spawn_food()
    
    
    
    def _spawn_food(self):
        """Spawn a single food item at a random location."""
        active_food_count = len([f for f in self.food if not f.consumed])
        
        if active_food_count >= self.max_food:
            return
        
        x = random.uniform(20, self.width - 20)
        y = random.uniform(20, self.height - 20)
        food = Food(x=x, y=y)
        self.food.append(food)
    
    
    
    def spawn_food_periodically(self):
        """Spawn food at regular intervals."""
        if self.timestep % self.food_spawn_interval == 0:
            for _ in range(self.food_per_spawn):
                active_food_count = len([f for f in self.food if not f.consumed])
                if active_food_count < self.max_food:
                    x = random.uniform(20, self.width - 20)
                    y = random.uniform(20, self.height - 20)
                    food = Food(x=x, y=y)
                    self.food.append(food)
                else:
                    break
    
    
    
    def update(self, dt: float = 1.0):
        """Update all agents and environment state."""
        # Spawn resources periodically
        self.spawn_food_periodically()
        self.spawn_water_periodically()
        
        # Update all agents
        for agent in self.agents:
            agent.update(self, dt)

        #update gardens
        for garden in self.gardens:
            garden.update()
        
        # NEW: Remove inactive gardens periodically
        if self.timestep % 100 == 0:
            initial_garden_count = len(self.gardens)
            self.gardens = [g for g in self.gardens if not g.should_be_removed(self.timestep)]
            removed = initial_garden_count - len(self.gardens)
            if removed > 0:
                print(f"  🗑️  Removed {removed} inactive garden(s)")

        # Handle interactions
        self.handle_food_collection()
        self.handle_water_collection()
        self.handle_reproduction()
        self.handle_collisions()
        self.handle_interactions()

        # NEW: Check if selected agent is still valid
        if self.selected_agent is not None:
            if not self.selected_agent.alive or self.selected_agent not in self.agents:
                print(f"⚠️  Selected agent {self.selected_agent.id} is no longer valid - clearing selection")
                self.selected_agent = None
                self.show_brain_viz = False
        
        # Clean up consumed resources periodically
        if self.timestep % 100 == 0:
            self.food = [f for f in self.food if not f.consumed]
            self.water = [w for w in self.water if not w.depleted]  # Changed to depleted
        
        # Remove dead agents periodically
        if self.timestep % 50 == 0:
            self.remove_dead_agents()

        if self.timestep % 100 == 0:
            for agent in self.agents:
                agent.calculate_fitness()
            self.update_hall_of_fame()
            
        
        # Check end conditions
        if self.should_evolve():
            self.evolve_population()
        
        self.timestep += 1
    
    def handle_food_collection(self):
        """Handle agents collecting food."""
        for agent in self.agents:
            if not agent.alive:
                continue
            
            for food in self.food:
                if food.consumed:
                    continue
                
                if food.is_touching_circle(agent.circle):
                    agent.consume_food(food)
                    food.consumed = True
    
    def handle_water_collection(self):
        """Handle agents drinking from water sources (only when thirsty)."""
        for agent in self.agents:
            if not agent.alive:
                continue
            
            # Only drink if agent needs water (not already full)
            if agent.water >= agent.max_water:
                continue  # Skip if already full
            
            for water in self.water:
                if water.depleted:
                    continue
                
                if water.is_touching_circle(agent.circle):
                    # Agent drinks from water source (depletes it slowly)
                    hydration = water.drink()
                    if hydration > 0:
                        agent.consume_water_amount(hydration)
                        # Only one drink per timestep
                        break
    
    def handle_reproduction(self):
        """Handle reproduction between agents with high resources."""
        """if len(self.agents) >= self.max_population: no max population
            return"""
        
        alive_agents = [a for a in self.agents if a.alive and a.can_reproduce()]
        
        checked_pairs = set()
        for i, agent1 in enumerate(alive_agents):
            for agent2 in alive_agents[i+1:]:
                pair_id = tuple(sorted([agent1.id, agent2.id]))
                if pair_id in checked_pairs:
                    continue
                checked_pairs.add(pair_id)
                
                distance = agent1.circle.distance_to(agent2.circle)
                reproduction_distance = (agent1.circle.radius + agent2.circle.radius) * 2
                
                if distance <= reproduction_distance:
                    child = agent1.reproduce_with(agent2, self.next_agent_id, self.generation)
                    self.add_agent(child)
                    self.next_agent_id += 1
                    
                    print(f"  🐣 Birth! Agent {agent1.id} + Agent {agent2.id} -> Agent {child.id} "
                          f"(Gen {self.generation}, Pop: {len(self.agents)})")
                    
                    if len(self.agents) >= self.max_population:
                        return
    
    def handle_collisions(self):
        """Handle collisions between agents."""
        for i, agent1 in enumerate(self.agents):
            if not agent1.alive:
                continue
                
            for agent2 in self.agents[i+1:]:
                if not agent2.alive:
                    continue
                
                if agent1.circle.is_colliding(agent2.circle):
                        # === SOCIAL INTERACTIONS ===
                    
                    # Agent 1's intent towards Agent 2
                    if agent1.current_intent < -0.5:
                        # Agent 1 is hostile - damages Agent 2
                        damage = 25.0
                        agent2.health -= damage
                        agent1.damage_dealt += damage
                        agent2.damage_taken += damage
                        # No fitness bonus for harming
                        # Check if Agent 2 died from this attack
                        if agent2.health <= 0 and agent2.alive:
                            agent2.alive = False
                            print(f"  💀 Agent {agent1.id} killed Agent {agent2.id} "
                              f"(Gen {agent1.generation_born} vs Gen {agent2.generation_born}, "
                              f"Age {agent2.age})")
                        
                    elif agent1.current_intent > 0.5 and agent1.food_inventory > 0:
                        # Agent 1 is friendly - shares food with Agent 2
                        agent1.food_inventory -= 1
                        agent2.food_inventory += 1
                        agent1.food_shared += 1
                        agent2.food_received += 1
                        agent1.fitness += 50  # Fitness bonus for sharing!
                    
                    # Agent 2's intent towards Agent 1
                    if agent2.current_intent < -0.5:
                        # Agent 2 is hostile - damages Agent 1
                        damage = 25.0
                        agent1.health -= damage
                        agent2.damage_dealt += damage
                        agent1.damage_taken += damage
                        # No fitness bonus for harming
                        # Check if Agent 2 died from this attack
                        if agent1.health <= 0 and agent1.alive:
                            agent1.alive = False
                            print(f"  💀 Agent {agent2.id} killed Agent {agent1.id} "
                              f"(Gen {agent2.generation_born} vs Gen {agent1.generation_born}, "
                              f"Age {agent1.age})")
                        
                    elif agent2.current_intent > 0.5 and agent2.food_inventory > 0:
                        # Agent 2 is friendly - shares food with Agent 1
                        agent2.food_inventory -= 1
                        agent1.food_inventory += 1
                        agent2.food_shared += 1
                        agent1.food_received += 1
                        agent2.fitness += 50  # Fitness bonus for sharing!

                    # Simple elastic collision
                    agent1.circle.velocity_x, agent2.circle.velocity_x = \
                        agent2.circle.velocity_x * 0.8, agent1.circle.velocity_x * 0.8
                    agent1.circle.velocity_y, agent2.circle.velocity_y = \
                        agent2.circle.velocity_y * 0.8, agent1.circle.velocity_y * 0.8
                    
                    # Separate circles
                    dx = agent2.circle.x - agent1.circle.x
                    dy = agent2.circle.y - agent1.circle.y
                    distance = np.sqrt(dx**2 + dy**2)
                    
                    if distance > 0:
                        overlap = (agent1.circle.radius + agent2.circle.radius) - distance
                        dx /= distance
                        dy /= distance
                        
                        agent1.circle.x -= dx * overlap / 2
                        agent1.circle.y -= dy * overlap / 2
                        agent2.circle.x += dx * overlap / 2
                        agent2.circle.y += dy * overlap / 2
    
    def handle_interactions(self):
        """Handle other agent interactions."""
        pass
    
    def remove_dead_agents(self):
        """Remove dead agents from the simulation."""
        initial_count = len(self.agents)
        self.agents = [a for a in self.agents if a.alive]
        removed = initial_count - len(self.agents)
        
        if removed > 0:
            print(f"  💀 Removed {removed} dead agent(s). Population: {len(self.agents)}")
    
    def get_nearby_agents(self, agent: Agent, radius: float) -> List[Agent]:
        """Get agents within a certain radius."""
        nearby = []
        for other in self.agents:
            if other.id != agent.id and other.alive:
                if agent.circle.distance_to(other.circle) <= radius:
                    nearby.append(other)
        return nearby
    
    def get_statistics(self) -> dict:
        """Get current simulation statistics."""
        alive_agents = [a for a in self.agents if a.alive]
        active_food = [f for f in self.food if not f.consumed]
        active_water = [w for w in self.water if not w.depleted]
        
        # Calculate generation composition
        current_gen_agents = [a for a in alive_agents if a.generation_born == self.generation]
        legacy_agents = [a for a in alive_agents if a.generation_born < self.generation]

        active_gardens = [g for g in self.gardens if not g.harvested]
        
        stats = {
            'timestep': self.timestep,
            'generation': self.generation,
            'alive_agents': len(alive_agents),
            'total_agents': len(self.agents),
            'current_gen': len(current_gen_agents),
            'legacy_agents': len(legacy_agents),
            'active_food': len(active_food),
            'total_food': len(self.food),
            'active_water': len(active_water),
            'total_water': len(self.water),
            'avg_food': np.mean([a.food for a in alive_agents]) if alive_agents else 0,
            'avg_water': np.mean([a.water for a in alive_agents]) if alive_agents else 0,
            'avg_health': np.mean([a.health for a in alive_agents]) if alive_agents else 0,
            'avg_age': np.mean([a.age for a in alive_agents]) if alive_agents else 0,
            'total_food_collected': sum(a.food_collected for a in self.agents),
            'total_water_collected': sum(a.water_collected for a in self.agents),
            'max_fitness': max([a.fitness for a in self.agents]) if self.agents else 0,
            'active_gardens': len(active_gardens),
            'total_gardens': len(self.gardens),
            'gardens_ready': len([g for g in active_gardens if g.is_ready_to_harvest()]),
        }
        return stats
    
    # ... (keep the existing evolve_population method from before) ...
    

    
    def evolve_population(self):
        """
        Implement evolution with Hall of Fame preservation and injection.
        """
        print(f"\n{'='*60}")
        print(f"EVOLUTION: Generation {self.generation} -> {self.generation + 1}")
        print(f"{'='*60}")
        
        # Calculate fitness for all agents
        for agent in self.agents:
            agent.calculate_fitness()
        
        # UPDATE HALL OF FAME BEFORE EVOLUTION
        self.update_hall_of_fame(True)
        
        # Sort agents by fitness
        self.agents.sort(key=lambda a: a.fitness, reverse=True)
        
        alive_agents = [a for a in self.agents if a.alive]
        good_performers = [a for a in self.agents if a.fitness > 0]
        
        # Print statistics
        if alive_agents:
            best_fitness = alive_agents[0].fitness
            avg_fitness = np.mean([a.fitness for a in alive_agents])
            best_food = alive_agents[0].food_collected
            avg_food = np.mean([a.food_collected for a in alive_agents])
            avg_water = np.mean([a.water_collected for a in alive_agents])
            total_children = sum(a.children for a in self.agents)
            
            # Architecture statistics
            best_arch = alive_agents[0].brain.get_architecture_string()
            best_complexity = alive_agents[0].brain.get_complexity()
            avg_complexity = np.mean([a.brain.get_complexity() for a in alive_agents])
            min_complexity = min([a.brain.get_complexity() for a in alive_agents])
            max_complexity = max([a.brain.get_complexity() for a in alive_agents])

            total_gardens_created = sum(a.gardens_created for a in self.agents)
            total_gardens_harvested = sum(a.gardens_harvested for a in self.agents)
            
            print(f"Population: {len(alive_agents)} alive, {len(self.agents)} total")
            print(f"Good Performers: {len(good_performers)} (fitness > 0)")
            print(f"Best Fitness: {best_fitness:.2f} (Agent {alive_agents[0].id})")
            print(f"Best All-Time: {self.best_fitness_ever:.2f}")
            print(f"Best Architecture: {best_arch} ({best_complexity} parameters)")
            print(f"Complexity Range: {min_complexity} - {max_complexity} (avg: {avg_complexity:.0f})")
            print(f"Avg Fitness: {avg_fitness:.2f}")
            print(f"Best Food Collected: {best_food}")
            print(f"Avg Food Collected: {avg_food:.2f}")
            print(f"Avg Water Collected: {avg_water:.2f}")
            print(f"Total Offspring Born: {total_children}")
            print(f"Total Gardens Created: {total_gardens_created}")
            print(f"Total Gardens Harvested: {total_gardens_harvested}")
            
            # Show top 3 architectures
            print(f"\nTop 3 This Generation:")
            for i, agent in enumerate(alive_agents[:3]):
                arch = agent.brain.get_architecture_string()
                params = agent.brain.get_complexity()
                inv = agent.food_inventory
                gardens = f"G:{agent.gardens_created}/{agent.gardens_harvested}"
                print(f"  {i+1}. Agent {agent.id}: {arch} ({params} params, "
                      f"fitness: {agent.fitness:.1f}, inv: {inv}, {gardens})")
        else:
            print(f"⚠️  TOTAL EXTINCTION - All agents died!")
            print(f"Good Performers: 0")
        
        # DETERMINE SPAWNING STRATEGY
        if len(good_performers) == 0:
            # NO good performers - use HALL OF FAME
            print(f"❌ No good performers found (all fitness <= 0)")
            
            if self.hall_of_fame:
                print(f"🏆 Spawning from HALL OF FAME (with diversity)")
                self.agents = []
                self.spawn_from_hall_of_fame(20, diversity_ratio=0.2)  # 80% hall of fame, 20% random
            else:
                # Very first generations - no hall of fame yet
                print(f"🔄 No hall of fame yet - spawning random population")
                self.agents = []
                self._spawn_random_agents(20)
            
        else:
            # We have good performers
            print(f"✓ Found {len(good_performers)} good performer(s)")
            
            # Determine how many agents we need
            target_population = max(30, self.min_population)
            current_alive = len(alive_agents)
            
            if current_alive < target_population:
                # Need to spawn more agents from good performers
                num_to_spawn = target_population - current_alive
                
                # Calculate composition:
                # - Some from hall of fame (guaranteed quality)
                # - Most from current good performers (evolution)
                # - A few random (diversity)
                
                num_from_hof = min(10, num_to_spawn // 3)  # ~33% from hall of fame
                num_random = min(10, num_to_spawn // 3)   # ~33% architecture mutants
                num_from_current = num_to_spawn - num_from_hof - num_random
                
                print(f"📈 Spawning {num_to_spawn} agents:")
                print(f"   - {num_from_hof} from Hall of Fame (quality baseline)")
                print(f"   - {num_from_current} from current best (evolution)")
                print(f"   - {num_random} best agent mutants")
                
                # 1. Inject Hall of Fame agents
                if num_from_hof > 0 and self.hall_of_fame:
                    injected = self.inject_hall_of_fame_agents(num_from_hof, mutation_chance=0.6)
                    print(f"   ✓ Injected {injected} Hall of Fame agents")
                
                # 2. Spawn from current good performers
                if num_from_current > 0:
                    num_parents = max(3, len(good_performers) // 2)
                    parent_pool = good_performers[:num_parents]
                    
                    for _ in range(num_from_current):
                        parent = random.choice(parent_pool)
                        child_brain = parent.brain.clone()
                        
                        # Mutation
                        child_brain.mutate(
                            mutation_rate=0.05, 
                            mutation_strength=0.25,
                            structural_mutation_rate=0.1
                        )
                        
                        x = random.uniform(50, self.width - 50)
                        y = random.uniform(50, self.height - 50)
                        child = Agent(
                            agent_id=self.next_agent_id,
                            x=x,
                            y=y,
                            input_size=input_size,
                            brain=child_brain,
                            generation_born=self.generation + 1
                        )
                        self.add_agent(child)
                        self.next_agent_id += 1
                    
                    print(f"   ✓ Spawned {num_from_current} from top {num_parents} current performers")
                
                # 3. Spawn mutated best agents
                if num_random > 0:
                    self.spawn_best_agent_variants(num_random)
                    print(f"   ✓ Spawned {num_random} best agent mutants")
            
            else:
                # Population is healthy - still inject some hall of fame for stability
                print('Population is healthy! No evolution necessary')
                """num_to_cull = int(len(self.agents) * self.cull_percentage)
                
                if num_to_cull > 0:
                    # Cull worst performers
                    culled = self.agents[-num_to_cull:]
                    self.agents = self.agents[:-num_to_cull]
                    
                    # Replace some with hall of fame agents
                    num_hof_replacements = max(2, num_to_cull // 4)  # Replace 25% with HoF
                    
                    print(f"🗑️  Culled worst {num_to_cull} agents")
                    print(f"🏆 Replacing {num_hof_replacements} with Hall of Fame agents")
                    
                    if self.hall_of_fame:
                        injected = self.inject_hall_of_fame_agents(num_hof_replacements, mutation_chance=0.7)
                        print(f"   ✓ Injected {injected} Hall of Fame agents")
                    
                    # Mark culled agents as dead
                    for agent in culled:
                        agent.alive = False"""
        
        # Reset timestep for next generation cycle
        self.generation += 1
        self.timestep = 0
        
        final_alive = len([a for a in self.agents if a.alive])
        print(f"✓ Generation {self.generation} started with {final_alive} agents")
        print(f"{'='*60}\n")
    
    

    def inject_hall_of_fame_agents(self, num_agents: int, mutation_chance: float = 0.5):
        """
        Inject hall of fame agents into the current population.
        
        Args:
            num_agents: Number of hall of fame agents to inject
            mutation_chance: Probability that each hall of fame agent will be mutated
        """
        if not self.hall_of_fame:
            return 0
        
        injected = 0
        for i in range(num_agents):
            if len(self.hall_of_fame) == 0:
                break
            
            # Select from hall of fame (weighted towards best)
            # Top 3 are selected more often
            if random.random() < 0.6 and len(self.hall_of_fame) >= 3:
                hof_idx = random.randint(0, 2)  # Top 3
            else:
                hof_idx = random.randint(0, len(self.hall_of_fame) - 1)  # Any
            
            record = self.hall_of_fame[hof_idx]
            
            # Clone the brain
            brain = record['brain'].clone()
            
            # Mutate some of them for variation
            if random.random() < mutation_chance:
                brain.mutate(
                    mutation_rate=0.05,  # Light mutation
                    mutation_strength=0.15,
                    structural_mutation_rate=0.02  # Rare structural changes
                )
                mutation_label = "mutated"
            else:
                mutation_label = "pure"
            
            x = random.uniform(50, self.width - 50)
            y = random.uniform(50, self.height - 50)
            
            agent = Agent(
                agent_id=self.next_agent_id,
                x=x,
                y=y,
                input_size=input_size,
                brain=brain,
                generation_born=self.generation
            )
            self.agents.append(agent)
            self.next_agent_id += 1
            injected += 1
        
        return injected
   
    def spawn_best_agent_variants(self, num_agents: int, structural_mutation_rate: float = 0.8):
        """
        Spawn variants of the all-time best agent with structural mutations.
        Encourages neural network architectural evolution.
        
        Args:
            num_agents: Number of variants to spawn
            structural_mutation_rate: High rate of structural changes
        """
        if not self.hall_of_fame:
            # Fallback to random if no hall of fame
            self._spawn_random_agents(num_agents)
            return 0
        
        # Get the best agent ever
        best_record = self.hall_of_fame[0]
        
        spawned = 0
        for i in range(num_agents):
            # Clone the best brain
            brain = best_record['brain'].clone()
            
            # Apply STRUCTURAL mutations (encourage architecture evolution)
            brain.mutate(
                mutation_rate=0.05,  # Weight mutations
                mutation_strength=0.3,
                structural_mutation_rate=structural_mutation_rate  # HIGH structural mutation
            )
            
            x = random.uniform(50, self.width - 50)
            y = random.uniform(50, self.height - 50)
            
            agent = Agent(
                agent_id=self.next_agent_id,
                x=x,
                y=y,
                input_size=input_size,
                brain=brain,
                generation_born=self.generation
            )
            self.agents.append(agent)
            self.next_agent_id += 1
            spawned += 1
    
        return spawned
    
    def should_evolve(self) -> bool:
        """
        Determine if population should evolve.
        
        Returns:
            True if evolution should occur
        """
        # Evolve when generation time is reached (and < 10 agents) or all agents are dead
        alive_count = sum(1 for a in self.agents if a.alive)
        
        generation_time_reached = self.timestep > self.generation_length and self.timestep > 0 and alive_count < self.min_population
        all_dead = alive_count == 0
        
        return generation_time_reached or all_dead
    
    
    
    def _tournament_selection(self, population: List[Agent], tournament_size: int = 3) -> Agent:
        """
        Select an agent using tournament selection.
        
        Args:
            population: List of agents to select from
            tournament_size: Number of agents in tournament
            
        Returns:
            Selected agent
        """
        tournament = random.sample(population, min(tournament_size, len(population)))
        return max(tournament, key=lambda a: a.fitness)
    
    def handle_agent_list_click(self, mouse_x: int, mouse_y: int, viewport_width: int, viewport_height: int):
        """
        Handle clicks on the agent list panel.
        
        Args:
            mouse_x, mouse_y: Mouse position
            viewport_width, viewport_height: Size of viewport
        """
        panel_x = viewport_width
        header_height = 50
        scroll_area_y = header_height
        
        # Check if click is within the panel
        if mouse_x < panel_x or mouse_x > panel_x + self.panel_width:
            return
        
        if mouse_y < scroll_area_y or mouse_y > viewport_height:
            return
        
        # Calculate which agent was clicked
        item_height = 115
        click_y = mouse_y - scroll_area_y + self.scroll_offset
        clicked_index = int(click_y / item_height)
        
        # Get sorted agents (same order as display)
        alive_agents = [a for a in self.agents if a.alive]
        sorted_agents = sorted(alive_agents, key=lambda a: a.age, reverse=True)
        
        if 0 <= clicked_index < len(sorted_agents):
            clicked_agent = sorted_agents[clicked_index]
            
            # Verify agent is still alive
            if not clicked_agent.alive:
                print(f"⚠️  Agent {clicked_agent.id} is no longer alive")
                return
            
            self.selected_agent = clicked_agent
            self.show_brain_viz = True
            
            # Move camera to selected agent
            if self.camera:
                self.camera.x = self.selected_agent.circle.x
                self.camera.y = self.selected_agent.circle.y
            
            print(f"📍 Selected Agent {self.selected_agent.id}")
            print(f"   Brain type: {type(self.selected_agent.brain).__name__}")
            print(f"   Architecture: {self.selected_agent.brain.get_architecture_string()}")
            print(f"   show_brain_viz = {self.show_brain_viz}")

    def render_brain_visualization(self, screen, viewport_width: int, viewport_height: int):
        """
        Render a visual representation of the selected agent's brain.
        Shows nodes colored by their activation values.
        """
        # Validate selected agent
        if self.selected_agent is None:
            print("⚠️  No agent selected")
            self.show_brain_viz = False
            return None
        
        if not self.show_brain_viz:
            return None
        
        # Check if agent is still alive
        if not self.selected_agent.alive:
            print(f"⚠️  Selected agent {self.selected_agent.id} died")
            self.selected_agent = None
            self.show_brain_viz = False
            return None
        
        # Verify agent is still in the agents list
        if self.selected_agent not in self.agents:
            print(f"⚠️  Selected agent {self.selected_agent.id} no longer in environment")
            self.selected_agent = None
            self.show_brain_viz = False
            return None
        
        #import pygame
        
        agent = self.selected_agent
        brain = agent.brain
        
        #print(f"🧠 Rendering brain for Agent {agent.id}")  # Debug
        
        # Brain viz panel dimensions - MOVED TO LEFT SIDE
        panel_width = 450
        panel_height = 550
        panel_x = 20  # Left side with small margin
        panel_y = 20  # Top with small margin
        
        # Create semi-transparent background
        brain_surface = pygame.Surface((panel_width, panel_height))
        brain_surface.set_alpha(240)
        brain_surface.fill((30, 30, 40))
        screen.blit(brain_surface, (panel_x, panel_y))
        
        # Draw border
        pygame.draw.rect(screen, (100, 150, 200), (panel_x, panel_y, panel_width, panel_height), 3)
        
        # Header
        font = pygame.font.Font(None, 24)
        tiny_font = pygame.font.Font(None, 16)
        
        header_text = font.render(f"Agent {agent.id} Brain", True, (255, 255, 255))
        screen.blit(header_text, (panel_x + 10, panel_y + 10))
        
        # Close button
        close_text = font.render("X", True, (255, 100, 100))
        close_rect = pygame.Rect(panel_x + panel_width - 35, panel_y + 5, 30, 30)
        pygame.draw.rect(screen, (60, 30, 30), close_rect)
        screen.blit(close_text, (panel_x + panel_width - 28, panel_y + 8))
        
        # Architecture info
        arch_text = tiny_font.render(f"Architecture: {brain.get_architecture_string()}", True, (200, 200, 200))
        screen.blit(arch_text, (panel_x + 10, panel_y + 35))
        
        # Check if recurrent or feedforward
        is_recurrent = isinstance(brain, RecurrentNeuralNetwork)
        
        #print(f"   Is recurrent: {is_recurrent}")  # Debug
        
        if is_recurrent:
            # Render recurrent network visualization
            self._render_recurrent_brain(screen, brain, panel_x, panel_y, panel_width, panel_height, tiny_font)
        else:
            # Render feedforward network visualization
            self._render_feedforward_brain(screen, brain, panel_x, panel_y, panel_width, panel_height, tiny_font)
        
        return close_rect  # Return for click detection

    def _render_recurrent_brain(self, screen, brain: RecurrentNeuralNetwork, 
                        panel_x: int, panel_y: int, panel_width: int, panel_height: int, font):
        """Render recurrent neural network with reservoir visualization."""
        import pygame
        
        # Layout: Input nodes | Reservoir nodes | Output nodes
        node_radius = 6
        margin = 60
        content_y = panel_y + 60
        content_height = panel_height - 80
        
        # Calculate positions
        input_x = panel_x + margin
        reservoir_x = panel_x + panel_width // 2
        output_x = panel_x + panel_width - margin
        
        # Get current activations
        reservoir_state = brain.reservoir_state
        input_size = brain.input_size
        output_size = brain.output_size
        reservoir_size = brain.reservoir_size
        
        # Get current input values (stored during last forward pass)
        if hasattr(brain, 'last_input') and brain.last_input is not None:
            input_values = brain.last_input
        else:
            input_values = np.zeros(input_size)
        
        # Input node labels - UPDATED FOR 54 INPUTS
        input_labels = []
        
        # 8 sectors × 6 values = 48 inputs
        sector_names = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
        for i, sector in enumerate(sector_names):
            input_labels.append(f"{sector}:Food")
            input_labels.append(f"{sector}:Watr")
            input_labels.append(f"{sector}:Agnt")
            input_labels.append(f"{sector}:Intnt")
            input_labels.append(f"{sector}:Grdn")
            input_labels.append(f"{sector}:GStat")
        
        # Self-state inputs (6)
        input_labels.extend([
            "Health",
            "Food Lvl",
            "Water Lvl",
            "Inventory",
            "Wall Dist",
            "Rel Age",
            "Bias"
        ])
        
        # Ensure we have enough labels
        while len(input_labels) < input_size:
            input_labels.append(f"Input {len(input_labels)}")
        
        # Draw Input Nodes (vertically spaced, scrollable if needed)
        max_visible_inputs = 100  # Show up to 25 inputs
        input_spacing = min(content_height / (max_visible_inputs + 1), 20)
        
        if input_size > max_visible_inputs:
            # Show only first max_visible_inputs and indicate more exist
            visible_inputs = max_visible_inputs - 1
            show_more_indicator = True
        else:
            visible_inputs = input_size
            show_more_indicator = False
        
        input_start_y = content_y + 10
        input_positions = []
        
        for i in range(visible_inputs):
            y = input_start_y + i * input_spacing
            
            # Color based on input value
            if i < len(input_values):
                input_val = input_values[i]
                color = self._activation_to_color(input_val)
            else:
                input_val = 0
                color = (100, 150, 200)
            
            pygame.draw.circle(screen, color, (int(input_x), int(y)), node_radius)
            pygame.draw.circle(screen, (255, 255, 255), (int(input_x), int(y)), node_radius, 1)
            
            input_positions.append((input_x, y))
            
            # Draw input label (make font smaller for readability)
            if i < len(input_labels):
                label = font.render(input_labels[i], True, (180, 180, 180))
                screen.blit(label, (input_x - 80, y - 6))
                
                # Draw value (smaller, on right of node)
                value_text = font.render(f"{input_val:.2f}", True, (130, 130, 130))
                screen.blit(value_text, (input_x + 10, y - 6))
        
        # Show "..." indicator if there are more inputs
        if show_more_indicator:
            y = input_start_y + visible_inputs * input_spacing
            more_text = font.render(f"... +{input_size - visible_inputs} more", True, (150, 150, 150))
            screen.blit(more_text, (input_x - 70, y - 6))
        
        # Sample reservoir nodes if too many - FIX: Ensure indices are valid
        max_display_nodes = 500
        if reservoir_size > max_display_nodes:
            # Sample evenly, ensuring we don't exceed reservoir_size - 1
            display_indices = [min(int(i * reservoir_size / max_display_nodes), reservoir_size - 1) 
                            for i in range(max_display_nodes)]
            # Remove duplicates while preserving order
            seen = set()
            display_indices = [x for x in display_indices if not (x in seen or seen.add(x))]
            display_size = len(display_indices)
        else:
            display_indices = list(range(reservoir_size))
            display_size = reservoir_size
        
        # Draw Reservoir Nodes (circular arrangement)
        reservoir_center_y = content_y + content_height // 2
        reservoir_radius = min(80, content_height // 3)
        
        reservoir_positions = []
        for i, idx in enumerate(display_indices):
            angle = (i / display_size) * 2 * np.pi
            x = reservoir_x + reservoir_radius * np.cos(angle)
            y = reservoir_center_y + reservoir_radius * np.sin(angle)
            
            # Color based on activation value
            activation = reservoir_state[idx]
            color = self._activation_to_color(activation)
            
            pygame.draw.circle(screen, color, (int(x), int(y)), node_radius)
            pygame.draw.circle(screen, (255, 255, 255), (int(x), int(y)), node_radius, 1)
            
            reservoir_positions.append((x, y))
        
        # Create a mapping from reservoir indices to display positions
        index_to_display = {idx: i for i, idx in enumerate(display_indices)}
        
        # ========== DRAW INPUT-TO-RESERVOIR CONNECTIONS ==========
        input_connections = []
        for res_idx in display_indices:
            # Bounds check
            if res_idx >= reservoir_size or res_idx < 0:
                continue
                
            for inp_idx in range(min(visible_inputs, input_size)):  # Only draw connections to visible inputs
                # Check array bounds
                if res_idx < brain.input_weights.shape[0] and inp_idx < brain.input_weights.shape[1]:
                    weight = brain.input_weights[res_idx, inp_idx]  # [reservoir, input]
                    if abs(weight) > 0.001:
                        input_connections.append((inp_idx, res_idx, weight))
        
        # Store total before limiting
        total_input_connections = len(input_connections)

        # Limit if too many
        max_input_connections = 100
        if len(input_connections) > max_input_connections:
            input_connections.sort(key=lambda x: abs(x[2]), reverse=True)
            input_connections = input_connections[:max_input_connections]
        
        # Draw input connections (behind reservoir nodes)
        for inp_idx, res_idx, weight in input_connections:
            if res_idx in index_to_display and inp_idx < len(input_positions):
                res_display_idx = index_to_display[res_idx]
                if res_display_idx < len(reservoir_positions):
                    start_pos = input_positions[inp_idx]
                    end_pos = reservoir_positions[res_display_idx]
                    
                    # Color and thickness based on weight
                    weight_strength = min(abs(weight) / 2.0, 1.0)  # Scale down for visibility
                    
                    if weight > 0:
                        intensity = int(80 + 100 * weight_strength)
                        line_color = (intensity, intensity // 4, 0)
                    else:
                        intensity = int(80 + 100 * weight_strength)
                        line_color = (0, intensity // 4, intensity)
                    
                    line_width = 1
                    pygame.draw.line(screen, line_color,
                                (int(start_pos[0]), int(start_pos[1])),
                                (int(end_pos[0]), int(end_pos[1])), line_width)
        
        # ========== DRAW RESERVOIR-TO-RESERVOIR CONNECTIONS ==========
        reservoir_connections = []
        for i in display_indices:
            if i >= reservoir_size or i < 0:
                continue
            for j in display_indices:
                if j >= reservoir_size or j < 0:
                    continue
                # Check array bounds
                if j < brain.reservoir_weights.shape[0] and i < brain.reservoir_weights.shape[1]:
                    weight = brain.reservoir_weights[j, i]  # [to, from]
                    if abs(weight) > 0.001:
                        reservoir_connections.append((i, j, weight))
        
        # Store total before limiting
        total_reservoir_connections = len(reservoir_connections)

        # Limit number of connections if too many
        max_reservoir_connections = 100
        if len(reservoir_connections) > max_reservoir_connections:
            reservoir_connections.sort(key=lambda x: abs(x[2]), reverse=True)
            reservoir_connections = reservoir_connections[:max_reservoir_connections]
        
        # Draw reservoir connections
        for i, j, weight in reservoir_connections:
            if i in index_to_display and j in index_to_display:
                start_idx = index_to_display[i]
                end_idx = index_to_display[j]
                
                if start_idx < len(reservoir_positions) and end_idx < len(reservoir_positions):
                    start_pos = reservoir_positions[start_idx]
                    end_pos = reservoir_positions[end_idx]
                    
                    # Color based on weight sign and magnitude
                    weight_strength = min(abs(weight), 1.0)
                    
                    if weight > 0:
                        intensity = int(100 + 155 * weight_strength)
                        line_color = (intensity, intensity // 3, 0)
                    else:
                        intensity = int(100 + 155 * weight_strength)
                        line_color = (0, intensity // 3, intensity)
                    
                    line_width = max(1, int(weight_strength * 2))
                    
                    pygame.draw.line(screen, line_color,
                                (int(start_pos[0]), int(start_pos[1])),
                                (int(end_pos[0]), int(end_pos[1])), line_width)
        
        # Draw Output Nodes (vertically spaced)
        output_spacing = min(content_height / (output_size + 1), 40)
        output_start_y = content_y + (content_height - output_spacing * output_size) // 2
        
        # Get output activations
        output_activations = np.dot(reservoir_state, brain.output_weights) + brain.output_bias
        for j in range(output_size):
            output_activations[j] = ActivationFunctions.activate(
                np.array([output_activations[j]]),
                brain.output_activations[j]
            )[0]
        
        output_labels = ["Angle", "Speed", "Intent", "Garden", "Plant", "Water"]
        output_positions = []
        
        for i in range(output_size):
            y = output_start_y + i * output_spacing
            
            # Color based on activation
            activation = output_activations[i]
            color = self._activation_to_color(activation)
            
            pygame.draw.circle(screen, color, (int(output_x), int(y)), node_radius + 2)
            pygame.draw.circle(screen, (255, 255, 255), (int(output_x), int(y)), node_radius + 2, 1)
            
            output_positions.append((output_x, y))
            
            # Label
            if i < len(output_labels):
                label = font.render(output_labels[i], True, (200, 200, 200))
                screen.blit(label, (output_x + 15, y - 6))
                
                # Draw value
                value_text = font.render(f"{activation:.2f}", True, (150, 150, 150))
                screen.blit(value_text, (output_x + 80, y - 6))
        
        # ========== DRAW RESERVOIR-TO-OUTPUT CONNECTIONS ==========
        output_connections = []
        for out_idx in range(output_size):
            for res_idx in display_indices:
                if res_idx >= reservoir_size or res_idx < 0:
                    continue
                # Check array bounds
                # output_weights shape is (reservoir_size, output_size)
                if res_idx < brain.output_weights.shape[0] and out_idx < brain.output_weights.shape[1]:
                    weight = brain.output_weights[res_idx, out_idx]
                    if abs(weight) > 0.001:
                        output_connections.append((res_idx, out_idx, weight))

        # Store total before limiting
        total_output_connections = len(output_connections)

        # Limit if too many
        max_output_connections = 100
        if len(output_connections) > max_output_connections:
            output_connections.sort(key=lambda x: abs(x[2]), reverse=True)
            output_connections = output_connections[:max_output_connections]

        # Draw output connections
        for res_idx, out_idx, weight in output_connections:
            if res_idx in index_to_display and out_idx < len(output_positions):
                res_display_idx = index_to_display[res_idx]
                if res_display_idx < len(reservoir_positions):
                    start_pos = reservoir_positions[res_display_idx]
                    end_pos = output_positions[out_idx]
                    
                    # Color and thickness based on weight
                    weight_strength = min(abs(weight) / 2.0, 1.0)
                    
                    if weight > 0:
                        intensity = int(80 + 100 * weight_strength)
                        line_color = (intensity, intensity // 4, 0)
                    else:
                        intensity = int(80 + 100 * weight_strength)
                        line_color = (0, intensity // 4, intensity)
                    
                    line_width = 1
                    pygame.draw.line(screen, line_color,
                                (int(start_pos[0]), int(start_pos[1])),
                                (int(end_pos[0]), int(end_pos[1])), line_width)
        
        # Draw connection statistics
        total_all_connections = total_input_connections + total_reservoir_connections + total_output_connections
        displayed_connections = len(input_connections) + len(reservoir_connections) + len(output_connections)

        stats_text = font.render(
            f"Connections: {total_all_connections} ({total_input_connections}→R, {total_reservoir_connections}↔R, {total_output_connections}→O)", 
            True, (180, 180, 180)
        )
        screen.blit(stats_text, (panel_x + 10, panel_y + 50))
        
        # Legend
        legend_y = panel_y + panel_height - 25
        legend_text = font.render("Activation: ", True, (180, 180, 180))
        screen.blit(legend_text, (panel_x + 10, legend_y))
        
        # Color gradient legend
        for i in range(5):
            activation_val = -1 + (i / 4) * 2  # -1 to 1
            color = self._activation_to_color(activation_val)
            legend_x = panel_x + 100 + i * 25
            pygame.draw.circle(screen, color, (legend_x, legend_y + 8), 8)

    def _render_feedforward_brain(self, screen, brain: NeuralNetwork,
                                panel_x: int, panel_y: int, panel_width: int, panel_height: int, font):
        """Render feedforward neural network layer by layer."""
        import pygame
        
        node_radius = 6
        margin = 40
        content_y = panel_y + 60
        content_height = panel_height - 80
        
        layers = brain.layers
        num_layers = len(layers)
        
        # Calculate horizontal spacing
        layer_spacing = (panel_width - 2 * margin) / (num_layers - 1) if num_layers > 1 else 0
        
        # Store node positions for drawing connections
        layer_positions = []
        
        for layer_idx, layer_size in enumerate(layers):
            x = panel_x + margin + layer_idx * layer_spacing
            
            # Vertical spacing for nodes in this layer
            node_spacing = min(content_height / (layer_size + 1), 25)
            start_y = content_y + (content_height - node_spacing * layer_size) // 2
            
            positions = []
            for node_idx in range(layer_size):
                y = start_y + node_idx * node_spacing
                
                # Color based on layer type
                if layer_idx == 0:
                    # Input layer - blue
                    color = (100, 150, 200)
                elif layer_idx == num_layers - 1:
                    # Output layer - green/red based on activation
                    # Approximate activation
                    color = (100, 200, 100)
                else:
                    # Hidden layer - purple
                    color = (150, 100, 200)
                
                pygame.draw.circle(screen, color, (int(x), int(y)), node_radius)
                pygame.draw.circle(screen, (255, 255, 255), (int(x), int(y)), node_radius, 1)
                
                positions.append((x, y))
            
            layer_positions.append(positions)
        
        # Draw connections between layers (sample to avoid clutter)
        for layer_idx in range(len(layer_positions) - 1):
            current_layer = layer_positions[layer_idx]
            next_layer = layer_positions[layer_idx + 1]
            
            # Sample connections
            max_connections = 50
            total_possible = len(current_layer) * len(next_layer)
            
            if total_possible <= max_connections:
                # Draw all
                for start_pos in current_layer:
                    for end_pos in next_layer:
                        pygame.draw.line(screen, (60, 60, 80),
                                    (int(start_pos[0]), int(start_pos[1])),
                                    (int(end_pos[0]), int(end_pos[1])), 1)
            else:
                # Sample randomly
                for _ in range(max_connections):
                    start_pos = random.choice(current_layer)
                    end_pos = random.choice(next_layer)
                    pygame.draw.line(screen, (60, 60, 80),
                                (int(start_pos[0]), int(start_pos[1])),
                                (int(end_pos[0]), int(end_pos[1])), 1)

    def _activation_to_color(self, activation: float) -> Tuple[int, int, int]:
        """
        Convert activation value to a color.
        Negative = blue, Zero = gray, Positive = red/orange
        """
        # Clamp activation to -1 to 1 range
        activation = np.clip(activation, -1, 1)
        
        if activation < 0:
            # Negative: black -> blue
            intensity = int(255 * abs(activation))
            return (0, 0, intensity)
        else:
            # Positive: black -> red/orange
            intensity = int(255 * activation)
            return (intensity, intensity // 2, 0)
    
    def render(self, screen=None, font=None):
        
        """
        Render the environment with camera controls.
        """
        #import pygame
        
        # Initialize pygame if not already done
        viewport_width = 1000  # Main viewing area width
        viewport_height = 600  # Main viewing area height
        
        if screen is None:
            pygame.init()
            screen = pygame.display.set_mode((viewport_width + self.panel_width, viewport_height))
            pygame.display.set_caption("lil guy world")
        
        if font is None:
            font = pygame.font.Font(None, 24)
            small_font = pygame.font.Font(None, 18)
            tiny_font = pygame.font.Font(None, 14)
        else:
            small_font = pygame.font.Font(None, 18)
            tiny_font = pygame.font.Font(None, 14)
        
        # Initialize camera if not already done
        if self.camera is None:
            self.camera = Camera(viewport_width, viewport_height, self.width, self.height)
        
        # Clear screen
        screen.fill((255, 255, 255))
        
        # Draw world background (grid for reference)
        self._draw_grid(screen, viewport_width, viewport_height)
        
        # ========== MAIN SIMULATION AREA (with camera) ==========
        # Create a surface for the world view
        world_surface = pygame.Surface((viewport_width, viewport_height))
        world_surface.fill((245, 245, 250))
        
        # Draw food (only if visible)
        for food in self.food:
            if not food.consumed and self.camera.is_visible(food.x, food.y):
                screen_pos = self.camera.world_to_screen(food.x, food.y)
                scaled_radius = max(2, int(self.camera.scale_size(food.radius)))
                
                pygame.draw.circle(
                    screen,
                    food.color,
                    screen_pos,
                    scaled_radius
                )
                pygame.draw.circle(
                    screen,
                    (0, 200, 0),
                    screen_pos,
                    scaled_radius,
                    max(1, int(self.camera.zoom))
                )
        
        # Draw water (only if visible)
        for water in self.water:
            if not water.depleted and self.camera.is_visible(water.x, water.y):
                screen_pos = self.camera.world_to_screen(water.x, water.y)
                scaled_radius = max(3, int(self.camera.scale_size(water.radius)))
                
                fill_percentage = water.get_fill_percentage()
                blue_intensity = int(150 + (105 * fill_percentage))
                water_color = (50, 100 + int(50 * fill_percentage), blue_intensity)
                
                pygame.draw.circle(
                    screen,
                    water_color,
                    screen_pos,
                    scaled_radius
                )
                
                pygame.draw.circle(
                    screen,
                    (0, 100, 200),
                    screen_pos,
                    scaled_radius,
                    max(1, int(self.camera.zoom * 2))
                )
                
                # Draw water amount if zoomed in enough
                if self.camera.zoom > 0.5:
                    if fill_percentage < 1.0:
                        inner_radius = int(scaled_radius * fill_percentage)
                        if inner_radius > 2:
                            pygame.draw.circle(
                                screen,
                                (200, 230, 255),
                                screen_pos,
                                inner_radius
                            )
                    
                    if self.camera.zoom > 0.8:
                        water_text = tiny_font.render(f"{water.water_amount:.0f}", True, (255, 255, 255))
                        text_rect = water_text.get_rect(center=screen_pos)
                        bg_rect = text_rect.inflate(8, 4)
                        pygame.draw.rect(screen, (0, 50, 100), bg_rect)
                        pygame.draw.rect(screen, (255, 255, 255), bg_rect, 1)
                        screen.blit(water_text, text_rect)

        
        
        # Draw gardens (after food, before agents)
        for garden in self.gardens:
            if self.camera.is_visible(garden.x, garden.y):
                screen_pos = self.camera.world_to_screen(garden.x, garden.y)
                scaled_radius = max(3, int(self.camera.scale_size(garden.radius)))
                
                # Draw garden circle
                garden_color = garden.get_color()
                pygame.draw.circle(
                    screen,
                    garden_color,
                    screen_pos,
                    scaled_radius
                )
                
                # Draw outline
                outline_color = (255, 215, 0) if garden.is_ready_to_harvest() else (80, 40, 0)
                outline_width = max(1, int(self.camera.zoom * 2))
                pygame.draw.circle(
                    screen,
                    outline_color,
                    screen_pos,
                    scaled_radius,
                    outline_width
                )
                
                # Draw growth progress if zoomed in
                if self.camera.zoom > 0.5 and garden.planted and garden.watered:
                    progress = garden.get_growth_percentage()
                    if progress < 1.0:
                        # Draw progress arc
                        arc_rect = pygame.Rect(
                            screen_pos[0] - scaled_radius,
                            screen_pos[1] - scaled_radius,
                            scaled_radius * 2,
                            scaled_radius * 2
                        )
                        arc_angle = int(360 * progress)
                        if arc_angle > 0:
                            pygame.draw.arc(
                                screen,
                                (255, 255, 0),
                                arc_rect,
                                0,
                                np.radians(arc_angle),
                                max(2, int(self.camera.zoom * 3))
                            )
                    
                    # Draw timer text if zoomed in enough
                    if self.camera.zoom > 0.8:
                        time_left = garden.growth_time - garden.growth_timer
                        if garden.is_ready_to_harvest():
                            timer_text = tiny_font.render("READY!", True, (255, 255, 255))
                        else:
                            timer_text = tiny_font.render(f"{time_left}", True, (255, 255, 255))
                        
                        text_rect = timer_text.get_rect(center=screen_pos)
                        bg_rect = text_rect.inflate(6, 4)
                        pygame.draw.rect(screen, (0, 0, 0, 180), bg_rect)
                        screen.blit(timer_text, text_rect)
        
        # Draw agents (only if visible)
        for agent in self.agents:
            if agent.alive and self.camera.is_visible(agent.circle.x, agent.circle.y):
                screen_pos = self.camera.world_to_screen(agent.circle.x, agent.circle.y)
                scaled_radius = max(2, int(self.camera.scale_size(agent.circle.radius)))
                
                # Draw agent circle
                pygame.draw.circle(
                    screen,
                    agent.circle.color,
                    screen_pos,
                    scaled_radius
                )
                
                # Draw outline
                outline_color = (255, 215, 0) if agent.can_reproduce() else (0, 0, 0)
                outline_width = max(1, int(self.camera.zoom * 2)) if agent.can_reproduce() else max(1, int(self.camera.zoom))
                pygame.draw.circle(
                    screen,
                    outline_color,
                    screen_pos,
                    scaled_radius,
                    outline_width
                )

                # Draw agent ID nametag BELOW agent (changed from above)
                if self.camera.zoom > 0.4:
                    nametag_font = pygame.font.Font(None, max(14, int(16 * self.camera.zoom)))
                    nametag_text = nametag_font.render(f"#{agent.id}", True, (255, 255, 255))
                    nametag_rect = nametag_text.get_rect()
                    nametag_rect.centerx = screen_pos[0]
                    nametag_rect.top = screen_pos[1] + scaled_radius + 5  # CHANGED: now below agent
                    
                    # Background for nametag
                    bg_rect = nametag_rect.inflate(4, 2)
                    pygame.draw.rect(screen, (0, 0, 0, 180), bg_rect)
                    pygame.draw.rect(screen, (200, 200, 200), bg_rect, 1)
                    screen.blit(nametag_text, nametag_rect)
                
                # Draw status bars (only if zoomed in enough)
                if self.camera.zoom > 0.5:
                    bar_width = self.camera.scale_size(agent.circle.radius * 2.5)
                    bar_height = max(2, int(self.camera.scale_size(3)))
                    bar_x = screen_pos[0] - bar_width / 2
                    bar_y_start = screen_pos[1] - scaled_radius - int(self.camera.scale_size(18))
                    
                    # Health bar
                    pygame.draw.rect(screen, (100, 100, 100), (bar_x, bar_y_start, bar_width, bar_height))
                    health_width = bar_width * (agent.health / agent.max_health)
                    health_color = (
                        int(255 * (1 - agent.health / agent.max_health)),
                        int(255 * (agent.health / agent.max_health)),
                        0
                    )
                    pygame.draw.rect(screen, health_color, (bar_x, bar_y_start, health_width, bar_height))
                    
                    # Food bar
                    bar_y = bar_y_start + bar_height + 2
                    pygame.draw.rect(screen, (100, 100, 100), (bar_x, bar_y, bar_width, bar_height))
                    food_width = bar_width * (agent.food / agent.max_food)
                    pygame.draw.rect(screen, (0, 200, 0), (bar_x, bar_y, food_width, bar_height))
                    
                    # Water bar
                    bar_y = bar_y + bar_height + 2
                    pygame.draw.rect(screen, (100, 100, 100), (bar_x, bar_y, bar_width, bar_height))
                    water_width = bar_width * (agent.water / agent.max_water)
                    pygame.draw.rect(screen, (50, 150, 255), (bar_x, bar_y, water_width, bar_height))
        
        # Draw world bounds
        world_corners = [
            (0, 0), (self.width, 0),
            (self.width, self.height), (0, self.height), (0, 0)
        ]
        screen_corners = [self.camera.world_to_screen(x, y) for x, y in world_corners]
        pygame.draw.lines(screen, (200, 0, 0), False, screen_corners, 3)
        
        # Draw dividing line
        pygame.draw.line(screen, (100, 100, 100), (viewport_width, 0), (viewport_width, viewport_height), 2)
        
        # ========== TOP STATS PANEL ==========
        stats = self.get_statistics()
        active_water_sources = len([w for w in self.water if not w.depleted])
        
        if self.hall_of_fame:
            best_hof = self.hall_of_fame[0]
            hof_arch = best_hof['architecture']
            hof_fitness = best_hof['fitness']
            hof_gen = best_hof['generation']
            
            stats_texts = [
                f"Generation: {stats['generation']}",
                f"Timestep: {stats['timestep']}",
                f"Alive: {stats['alive_agents']}/{stats['total_agents']}",
                f"",
                f"Camera:",
                f"  Zoom: {self.camera.zoom:.2f}x",
                f"  Pos: ({int(self.camera.x)}, {int(self.camera.y)})",
                f"",
                f"Best: {stats['max_fitness']:.0f}",
                f"All-Time: {hof_fitness:.0f}",
                f"HoF: {len(self.hall_of_fame)}",
                f"Gardens: {stats['active_gardens']} ({stats['gardens_ready']} ready)"
            ]
        else:
            stats_texts = [
                f"Generation: {stats['generation']}",
                f"Timestep: {stats['timestep']}",
                f"Alive: {stats['alive_agents']}/{stats['total_agents']}",
                f"",
                f"Camera:",
                f"  Zoom: {self.camera.zoom:.2f}x",
                f"  Pos: ({int(self.camera.x)}, {int(self.camera.y)})",
            ]
        
        # Draw stats background
        stats_bg = pygame.Surface((220, 260))
        stats_bg.set_alpha(200)
        stats_bg.fill((50, 50, 50))
        screen.blit(stats_bg, (10, 10))
        
        # Draw stat text
        y_offset = 15
        for text in stats_texts:
            if text == "":
                y_offset += 10
            else:
                text_surface = tiny_font.render(text, True, (255, 255, 255))
                screen.blit(text_surface, (15, y_offset))
                y_offset += 18
        
        # ========== RIGHT PANEL - AGENT LIST ==========
        # ... (keep existing agent list code) ...
        
        # ========== CAMERA CONTROLS HELP ==========
        help_y = viewport_height - 110
        help_bg = pygame.Surface((220, 100))
        help_bg.set_alpha(200)
        help_bg.fill((50, 50, 50))
        screen.blit(help_bg, (10, help_y))
        
        help_texts = [
            "Camera Controls:",
            "  WASD - Pan camera",
            "  Q/E - Zoom out/in",
            "  Scroll - Zoom",
            "  Space - Center view",
        ]
        
        y_offset = help_y + 5
        for text in help_texts:
            text_surface = tiny_font.render(text, True, (255, 255, 255))
            screen.blit(text_surface, (15, y_offset))
            y_offset += 18
        
        

    
            
            # ========== RIGHT PANEL - AGENT LIST ==========
            # ... (keep the existing agent list panel code) ...
        
        
        
        
        
        # ========== RIGHT PANEL - AGENT LIST ==========
        panel_x = self.width
        panel_bg_color = (240, 240, 245)
        
        # Draw panel background
        pygame.draw.rect(screen, panel_bg_color, (panel_x, 0, self.panel_width, self.height))
        
        # Panel header
        panel_x = viewport_width
        panel_bg_color = (240, 240, 245)
        
        # Draw panel background
        pygame.draw.rect(screen, panel_bg_color, (panel_x, 0, self.panel_width, viewport_height))
        
        # Panel header
        header_height = 50
        pygame.draw.rect(screen, (60, 60, 80), (panel_x, 0, self.panel_width, header_height))
        
        header_text = font.render("Living Agents", True, (255, 255, 255))
        screen.blit(header_text, (panel_x + 10, 10))
        
        alive_agents = [a for a in self.agents if a.alive]
        alive_count_text = small_font.render(f"({len(alive_agents)} alive)", True, (200, 200, 200))
        screen.blit(alive_count_text, (panel_x + 10, 30))
        
        # Sort agents by age (oldest first)
        sorted_agents = sorted(alive_agents, key=lambda a: a.age, reverse=True)
        
        # Scrollable area
        scroll_area_y = header_height
        scroll_area_height = self.height - header_height
        item_height = 115  # Increased height for more stats
        
        # Calculate max scroll
        total_content_height = len(sorted_agents) * item_height
        max_scroll = max(0, total_content_height - scroll_area_height)
        max_scroll = 10000
        self.scroll_offset = max(0, min(self.scroll_offset, max_scroll))
        
        # Create scrollable surface
        scrollable_surface = pygame.Surface((self.panel_width, scroll_area_height))
        scrollable_surface.fill(panel_bg_color)
        
        # Draw each agent in the list
        y_pos = -self.scroll_offset
        for i, agent in enumerate(sorted_agents):
            # Only draw if visible
            if y_pos + item_height < 0 or y_pos > scroll_area_height:
                y_pos += item_height
                continue
            
                    # Agent item background
            item_bg_color = (250, 250, 255) if i % 2 == 0 else (245, 245, 250)
            pygame.draw.rect(scrollable_surface, item_bg_color, 
                            (5, y_pos + 2, self.panel_width - 10, item_height - 4))
            
            # Border
            pygame.draw.rect(scrollable_surface, (200, 200, 210),
                            (5, y_pos + 2, self.panel_width - 10, item_height - 4), 1)
            
            # Agent ID and Age
            id_text = small_font.render(f"Agent {agent.id}", True, (20, 20, 40))
            scrollable_surface.blit(id_text, (10, y_pos + 5))
            
            age_text = tiny_font.render(f"Age: {agent.age}", True, (80, 80, 100))
            scrollable_surface.blit(age_text, (10, y_pos + 23))
            
            gen_text = tiny_font.render(f"Gen: {agent.generation_born}", True, (80, 80, 100))
            scrollable_surface.blit(gen_text, (85, y_pos + 23))
            
            # Health, Food, Water stats with color coding
            health_color = (255, 0, 0) if agent.health < 30 else (200, 100, 0) if agent.health < 60 else (0, 150, 0)
            health_text = tiny_font.render(f"HP: {agent.health:.0f}", True, health_color)
            scrollable_surface.blit(health_text, (170, y_pos + 23))
            
            food_color = (255, 0, 0) if agent.food < 30 else (0, 150, 0)
            food_text = tiny_font.render(f"Food: {agent.food:.0f}", True, food_color)
            scrollable_surface.blit(food_text, (10, y_pos + 40))
            
            water_color = (255, 0, 0) if agent.water < 30 else (50, 100, 200)
            water_text = tiny_font.render(f"Water: {agent.water:.0f}", True, water_color)
            scrollable_surface.blit(water_text, (100, y_pos + 40))
            
            # Resources collected
            collected_text = tiny_font.render(f"F:{agent.food_collected} W:{agent.water_collected}", True, (100, 100, 120))
            scrollable_surface.blit(collected_text, (10, y_pos + 56))
            
            # Children
            children_text = tiny_font.render(f"Children: {agent.children}", True, (150, 0, 150))
            scrollable_surface.blit(children_text, (140, y_pos + 56))
            
            # Fitness
            fitness_text = tiny_font.render(f"Fitness: {agent.fitness:.0f}", True, (0, 100, 200))
            scrollable_surface.blit(fitness_text, (10, y_pos + 72))

            # Inventort
            inventory_text = tiny_font.render(f"Inventory: {agent.food_inventory}", True, (0, 100, 200))
            scrollable_surface.blit(inventory_text, (140, y_pos + 72))
            
            # Neural Network Architecture (highlighted)
            arch_string = agent.brain.get_architecture_string()
            complexity = agent.brain.get_complexity()
            
            # Architecture background
            pygame.draw.rect(scrollable_surface, (230, 240, 255),
                            (10, y_pos + 88, self.panel_width - 25, 22))
            pygame.draw.rect(scrollable_surface, (150, 180, 220),
                            (10, y_pos + 88, self.panel_width - 25, 22), 1)
            
            arch_text = tiny_font.render(f"Net: {arch_string}", True, (0, 50, 150))
            scrollable_surface.blit(arch_text, (15, y_pos + 92))
            
            params_text = tiny_font.render(f"({complexity})", True, (100, 100, 120))
            scrollable_surface.blit(params_text, (15 + arch_text.get_width() + 5, y_pos + 92))
            
            y_pos += item_height
        
        # Draw scrollable content to main screen
        screen.blit(scrollable_surface, (panel_x, scroll_area_y))
        
        # Draw scrollbar if needed
        if total_content_height > scroll_area_height:
            scrollbar_x = panel_x + self.panel_width - 15
            scrollbar_width = 10
            scrollbar_height = max(30, int((scroll_area_height / total_content_height) * scroll_area_height))
            scrollbar_y = scroll_area_y + int((self.scroll_offset / max_scroll) * (scroll_area_height - scrollbar_height))
            
            # Scrollbar track
            pygame.draw.rect(screen, (200, 200, 200),
                            (scrollbar_x, scroll_area_y, scrollbar_width, scroll_area_height))
            
            # Scrollbar handle
            pygame.draw.rect(screen, (100, 100, 120),
                            (scrollbar_x, scrollbar_y, scrollbar_width, scrollbar_height))
        
        # Draw legend in bottom left
        legend_y = self.height - 60
        legend_bg = pygame.Surface((200, 55))
        legend_bg.set_alpha(200)
        legend_bg.fill((50, 50, 50))
        screen.blit(legend_bg, (10, legend_y))
        
        legend_title = tiny_font.render("Status Bars:", True, (255, 255, 255))
        screen.blit(legend_title, (15, legend_y + 5))
        
        # Health bar example
        pygame.draw.rect(screen, (255, 100, 100), (15, legend_y + 22, 20, 3))
        health_label = tiny_font.render("Health", True, (255, 255, 255))
        screen.blit(health_label, (40, legend_y + 18))
        
        # Food bar example
        pygame.draw.rect(screen, (0, 200, 0), (110, legend_y + 22, 20, 3))
        food_label = tiny_font.render("Food", True, (255, 255, 255))
        screen.blit(food_label, (135, legend_y + 18))
        
        # Water bar example
        pygame.draw.rect(screen, (50, 150, 255), (15, legend_y + 37, 20, 3))
        water_label = tiny_font.render("Water", True, (255, 255, 255))
        screen.blit(water_label, (40, legend_y + 33))
        
        if self.show_brain_viz and self.selected_agent:
            # Store close rect for click detection (will be used in simulation loop)
            self.brain_viz_close_rect = self.render_brain_visualization(
                screen, viewport_width, viewport_height
            )
        else:
            self.brain_viz_close_rect = None

        # Update display
        pygame.display.flip()
        
        return screen, font
    
    
    def _draw_grid(self, screen, viewport_width, viewport_height):
        """Draw background grid for spatial reference."""
        #import pygame
        
        grid_spacing = 200  # Grid every 200 world units
        
        # Vertical lines
        start_x = int(self.camera.x / grid_spacing) * grid_spacing
        x = start_x
        while x < self.camera.x + viewport_width / self.camera.zoom:
            screen_pos = self.camera.world_to_screen(x, 0)
            if 0 <= screen_pos[0] <= viewport_width:
                pygame.draw.line(screen, (230, 230, 230), 
                            (screen_pos[0], 0), 
                            (screen_pos[0], viewport_height), 1)
            x += grid_spacing
        
        # Horizontal lines
        start_y = int(self.camera.y / grid_spacing) * grid_spacing
        y = start_y
        while y < self.camera.y + viewport_height / self.camera.zoom:
            screen_pos = self.camera.world_to_screen(0, y)
            if 0 <= screen_pos[1] <= viewport_height:
                pygame.draw.line(screen, (230, 230, 230), 
                            (0, screen_pos[1]), 
                            (viewport_width, screen_pos[1]), 1)
            y += grid_spacing




class Camera:
    """Camera for viewing and navigating the simulation world."""
    
    def __init__(self, viewport_width: int, viewport_height: int, 
                 world_width: int, world_height: int):
        """
        Initialize camera.
        
        Args:
            viewport_width: Width of the viewing area
            viewport_height: Height of the viewing area
            world_width: Total width of the simulation world
            world_height: Total height of the simulation world
        """
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.world_width = world_width
        self.world_height = world_height
        
        # Camera position (center of view)
        self.x = world_width / 2
        self.y = world_height / 2
        
        # Zoom level (1.0 = normal, 2.0 = zoomed in 2x, 0.5 = zoomed out 2x)
        self.zoom = 1.0
        self.min_zoom = 0.2
        self.max_zoom = 4.0
        
        # Camera movement speed
        self.pan_speed = 20
        self.zoom_speed = 0.1
    
    def world_to_screen(self, world_x: float, world_y: float) -> Tuple[int, int]:
        """
        Convert world coordinates to screen coordinates.
        
        Args:
            world_x, world_y: Position in world space
            
        Returns:
            Screen coordinates (x, y)
        """
        # Offset from camera center
        offset_x = (world_x - self.x) * self.zoom
        offset_y = (world_y - self.y) * self.zoom
        
        # Screen position (centered in viewport)
        screen_x = self.viewport_width / 2 + offset_x
        screen_y = self.viewport_height / 2 + offset_y
        
        return (int(screen_x), int(screen_y))
    
    def screen_to_world(self, screen_x: int, screen_y: int) -> Tuple[float, float]:
        """
        Convert screen coordinates to world coordinates.
        
        Args:
            screen_x, screen_y: Position on screen
            
        Returns:
            World coordinates (x, y)
        """
        # Offset from screen center
        offset_x = screen_x - self.viewport_width / 2
        offset_y = screen_y - self.viewport_height / 2
        
        # World position
        world_x = self.x + offset_x / self.zoom
        world_y = self.y + offset_y / self.zoom
        
        return (world_x, world_y)
    
    def is_visible(self, world_x: float, world_y: float, margin: float = 100) -> bool:
        """
        Check if a world position is visible on screen.
        
        Args:
            world_x, world_y: Position in world space
            margin: Extra margin around viewport
            
        Returns:
            True if position is visible
        """
        screen_x, screen_y = self.world_to_screen(world_x, world_y)
        return (-margin <= screen_x <= self.viewport_width + margin and
                -margin <= screen_y <= self.viewport_height + margin)
    
    def move(self, dx: float, dy: float):
        """Move camera by offset."""
        self.x += dx / self.zoom
        self.y += dy / self.zoom
        
        # Clamp to world bounds
        self.x = max(0, min(self.x, self.world_width))
        self.y = max(0, min(self.y, self.world_height))
    
    def zoom_at(self, screen_x: int, screen_y: int, zoom_delta: float):
        """
        Zoom in/out at a specific screen position.
        
        Args:
            screen_x, screen_y: Screen position to zoom towards
            zoom_delta: Change in zoom level
        """
        # Get world position before zoom
        world_pos_before = self.screen_to_world(screen_x, screen_y)
        
        # Update zoom
        old_zoom = self.zoom
        self.zoom *= (1 + zoom_delta)
        self.zoom = max(self.min_zoom, min(self.zoom, self.max_zoom))
        
        # Get world position after zoom
        world_pos_after = self.screen_to_world(screen_x, screen_y)
        
        # Adjust camera position to keep same world point under cursor
        self.x += world_pos_before[0] - world_pos_after[0]
        self.y += world_pos_before[1] - world_pos_after[1]
    
    def scale_size(self, world_size: float) -> float:
        """Scale a world size to screen size based on zoom."""
        return world_size * self.zoom

# ============================================================================
# SIMULATION
# ============================================================================

class Simulation:
    """Main simulation controller."""
    
    def __init__(self, environment: Environment, speed_multiplier: float = 1.0):
        """
        Initialize simulation.
        
        Args:
            environment: The environment to simulate
        """
        self.environment = environment
        self.running = False
        self.fps = 60
        self.speed_multiplier = speed_multiplier  # NEW
    
    def run(self, max_timesteps: Optional[int] = None, auto_save_interval: int = 10):
        """
        Main simulation loop with camera controls.
        """
        #import pygame
        
        self.running = True
        timestep = 0
        last_save_generation = 0
        
        # Initialize pygame display
        screen = None
        font = None
        clock = pygame.time.Clock()

        viewport_width = 1000
        viewport_height = 600
        
        print("Starting simulation with pygame rendering...")
        print(f"World size: {self.environment.width}x{self.environment.height}")
        print(f"Speed multiplier: {self.speed_multiplier}x")
        print("\nControls:")
        print("  ESC - Quit")
        print("  S - Save state")
        print("  WASD - Pan camera")
        print("  Q/E - Zoom out/in")
        print("  Mouse Wheel - Zoom at cursor")
        print("  Space - Center camera on world")
        print("  Arrow Keys - Scroll agent list")
        if auto_save_interval > 0:
            print(f"  Auto-save every {auto_save_interval} generations")

        brain_viz_close_rect = None  # Track close button
        
        try:
            while self.running:
                mouse_pos = pygame.mouse.get_pos()
                keys = pygame.key.get_pressed()
                
                # Handle pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                        
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            self.running = False
                        elif event.key == pygame.K_s:
                            self.save_best_run()
                        elif event.key == pygame.K_SPACE:
                            # Center camera on world
                            if self.environment.camera:
                                self.environment.camera.x = self.environment.width / 2
                                self.environment.camera.y = self.environment.height / 2
                        elif event.key == pygame.K_UP:
                            self.environment.scroll_offset -= 30
                        elif event.key == pygame.K_DOWN:
                            self.environment.scroll_offset += 30
                    
                    # Mouse click handling
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 1:  # Left click
                            print(f"🖱️  Mouse clicked at {mouse_pos}")
                            
                            # Check if clicking close button on brain viz
                            if self.environment.show_brain_viz and self.environment.brain_viz_close_rect:
                                print(f"   Brain viz is open, close rect: {self.environment.brain_viz_close_rect}")
                                if self.environment.brain_viz_close_rect.collidepoint(mouse_pos):
                                    print("   ✓ Clicked close button - closing brain viz")
                                    self.environment.show_brain_viz = False
                                    self.environment.selected_agent = None
                                    continue  # Don't process other clicks
                            
                            # Check if clicking agent list (only if not closing brain viz)
                            self.environment.handle_agent_list_click(
                                mouse_pos[0], mouse_pos[1], 
                                viewport_width, viewport_height
                            )

                    # Mouse wheel zoom
                    elif event.type == pygame.MOUSEWHEEL:
                        # Only zoom if mouse is over main viewport (not panel)
                        if mouse_pos[0] < 800:
                            if self.environment.camera:
                                zoom_delta = event.y * self.environment.camera.zoom_speed
                                self.environment.camera.zoom_at(mouse_pos[0], mouse_pos[1], zoom_delta)
                        else:
                            # Scroll agent list if over panel
                            scroll_speed = 30
                            self.environment.scroll_offset -= event.y * scroll_speed
                
                # Continuous camera controls (WASD)
                if self.environment.camera:
                    pan_speed = self.environment.camera.pan_speed
                    
                    if keys[pygame.K_w]:
                        self.environment.camera.move(0, -pan_speed)
                    if keys[pygame.K_s]:
                        self.environment.camera.move(0, pan_speed)
                    if keys[pygame.K_a]:
                        self.environment.camera.move(-pan_speed, 0)
                    if keys[pygame.K_d]:
                        self.environment.camera.move(pan_speed, 0)
                    
                    # Zoom with Q/E
                    if keys[pygame.K_q]:
                        self.environment.camera.zoom *= 0.98  # Zoom out
                        self.environment.camera.zoom = max(self.environment.camera.min_zoom, 
                                                        self.environment.camera.zoom)
                    if keys[pygame.K_e]:
                        self.environment.camera.zoom *= 1.02  # Zoom in
                        self.environment.camera.zoom = min(self.environment.camera.max_zoom, 
                                                        self.environment.camera.zoom)
                
                # Update simulation multiple times based on speed multiplier
                steps_this_frame = int(self.speed_multiplier)
                remainder = self.speed_multiplier - steps_this_frame
                
                # Always do at least the integer number of steps
                for _ in range(steps_this_frame):
                    self.step()
                    timestep += 1
                
                # Randomly do one more step based on remainder (for fractional speeds)
                if random.random() < remainder:
                    self.step()
                    timestep += 1
                
                # === AUTO-SAVE LOGIC ===
                current_gen = self.environment.generation
                current_timestep = self.environment.timestep
                last_save_timestep = self.environment.last_save_timestep
                
                # Method 1: Save on generation change (normal evolution)
                if auto_save_interval > 0:
                    if current_gen > last_save_generation and current_gen % auto_save_interval == 0:
                        self.save_best_run(f'autosave_gen{current_gen}.pkl')
                        last_save_generation = current_gen
                        last_save_timestep = current_timestep
                
                # Method 2: Save every 50k timesteps if stable (same generation for a long time)
                # This handles the case where agents reproduce and maintain stable population
                if current_timestep - last_save_timestep >= 50000:
                    print(f"\n⏰ Stable population detected - auto-saving at timestep {current_timestep}")
                    self.save_best_run(f'autosave_gen{current_gen}_t{current_timestep}.pkl')
                    last_save_timestep = current_timestep
                self.environment.last_save_timestep = last_save_timestep
                
                # Render
                screen, font = self.environment.render(screen, font)

                # NEW: Render brain visualization overlay
                """if self.environment.show_brain_viz and self.environment.selected_agent:
                    brain_viz_close_rect = self.environment.render_brain_visualization(
                        screen, viewport_width, viewport_height
                    )
                else:
                    brain_viz_close_rect = None"""
                
                # Control frame rate
                clock.tick(self.fps)
                
                # Check if max timesteps reached
                if max_timesteps and timestep >= max_timesteps:
                    print(f"\nReached max timesteps: {max_timesteps}")
                    break
        
        except KeyboardInterrupt:
            print("\n\nSimulation interrupted by user")
        
        finally:
            self.running = False
            
            # Save final state before closing
            print("\nSaving final state before exit...")
            self.save_best_run('final_state.pkl')
            
            pygame.quit()
            print("\nSimulation ended.")
            self._print_final_statistics()
        
            
    
    def step(self):
        """Execute one simulation step."""
        self.environment.update(dt=1.0)
    
    def _print_final_statistics(self):
        """Print final simulation statistics."""
        stats = self.environment.get_statistics()
        print(f"\n{'='*60}")
        print("FINAL STATISTICS")
        print(f"{'='*60}")
        print(f"Generation: {stats['generation']}")
        print(f"Total Timesteps: {stats['timestep']}")
        print(f"Agents Alive: {stats['alive_agents']}/{stats['total_agents']}")
        print(f"Total Food Collected: {stats['total_food_collected']}")
        print(f"Max Fitness Achieved: {stats['max_fitness']:.2f}")
        print(f"{'='*60}\n")
    
    def save_state(self, filename: str):
        """
        Save current simulation state including Hall of Fame, Gardens, and agent stats.
        
        Args:
            filename: File to save to
        """
        state = {
            'environment': {
                'width': self.environment.width,
                'height': self.environment.height,
                'timestep': self.environment.timestep,
                'generation': self.environment.generation,
                'next_agent_id': self.environment.next_agent_id,
                'best_fitness_ever': self.environment.best_fitness_ever,
            },
            'agents': [],
            'hall_of_fame': [],
            'gardens': [],
        }
        
        # Save agent brains AND stats
        for agent in self.environment.agents:
            agent_data = {
                # Recurrent network specific fields
                'brain_type': 'recurrent',
                'input_weights': agent.brain.input_weights.tolist(),
                'reservoir_weights': agent.brain.reservoir_weights.tolist(),
                'output_weights': agent.brain.output_weights.tolist(),
                'output_bias': agent.brain.output_bias.tolist(),
                'reservoir_size': agent.brain.reservoir_size,
                'spectral_radius': agent.brain.spectral_radius,
                'sparsity': agent.brain.sparsity,
                'reservoir_activations': [act.value for act in agent.brain.reservoir_activations],
                'output_activations': [act.value for act in agent.brain.output_activations],
                # Agent stats (UPDATED: save all current stats)
                'agent_id': agent.id,
                'alive': agent.alive,
                'health': agent.health,
                'food': agent.food,
                'water': agent.water,
                'food_inventory': agent.food_inventory,
                'age': agent.age,
                'fitness': agent.fitness,
                'food_collected': agent.food_collected,
                'water_collected': agent.water_collected,
                'children': agent.children,
                'generation_born': agent.generation_born,
                'gardens_created': agent.gardens_created,
                'gardens_planted': agent.gardens_planted,
                'gardens_watered': agent.gardens_watered,
                'gardens_harvested': agent.gardens_harvested,
                # Agent position and velocity
                'x': agent.circle.x,
                'y': agent.circle.y,
                'velocity_x': agent.circle.velocity_x,
                'velocity_y': agent.circle.velocity_y,
            }
            state['agents'].append(agent_data)
        
        # Save Hall of Fame
        for hof_record in self.environment.hall_of_fame:
            hof_data = {
                'brain_type': 'recurrent',
                'input_weights': hof_record['brain'].input_weights.tolist(),
                'reservoir_weights': hof_record['brain'].reservoir_weights.tolist(),
                'output_weights': hof_record['brain'].output_weights.tolist(),
                'output_bias': hof_record['brain'].output_bias.tolist(),
                'reservoir_size': hof_record['brain'].reservoir_size,
                'spectral_radius': hof_record['brain'].spectral_radius,
                'sparsity': hof_record['brain'].sparsity,
                'reservoir_activations': [act.value for act in hof_record['brain'].reservoir_activations],
                'output_activations': [act.value for act in hof_record['brain'].output_activations],
                'fitness': hof_record['fitness'],
                'generation': hof_record['generation'],
                'food_collected': hof_record['food_collected'],
                'water_collected': hof_record['water_collected'],
                'age': hof_record['age'],
                'children': hof_record['children'],
                'architecture': hof_record['architecture'],
                'complexity': hof_record['complexity'],
                'agent_id': hof_record.get('agent_id'),
            }
            state['hall_of_fame'].append(hof_data)
        
        # Save gardens
        for garden in self.environment.gardens:
            garden_data = {
                'x': garden.x,
                'y': garden.y,
                'radius': garden.radius,
                'owner_id': garden.owner_id,
                'planted': garden.planted,
                'watered': garden.watered,
                'growth_timer': garden.growth_timer,
                'growth_time': garden.growth_time,
                'cooldown_timer': garden.cooldown_timer,
                'cooldown_time': garden.cooldown_time,
                'food_multiplier': garden.food_multiplier,
                'harvested': garden.harvested,
                'times_harvested': garden.times_harvested,
                'last_interaction_timestep': garden.last_interaction_timestep,
                'inactivity_threshold': garden.inactivity_threshold,
            }
            state['gardens'].append(garden_data)
        
        with open(filename, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"\n💾 Simulation state saved to {filename}")
        print(f"   - {len(state['agents'])} agents")
        print(f"   - {len(state['hall_of_fame'])} Hall of Fame members")
        print(f"   - {len(state['gardens'])} gardens")
        print(f"   - Best fitness ever: {state['environment']['best_fitness_ever']:.2f}")

    def load_state(self, filename: str):
        """
        Load simulation state from file including Hall of Fame, Gardens, and agent stats.
        
        Args:
            filename: File to load from
        """
        with open(filename, 'rb') as f:
            state = pickle.load(f)
        
        # Restore environment settings
        self.environment.timestep = state['environment']['timestep']
        self.environment.generation = state['environment']['generation']
        self.environment.next_agent_id = state['environment']['next_agent_id']
        self.environment.best_fitness_ever = state['environment']['best_fitness_ever']
        
        # Clear existing agents
        self.environment.agents = []
        max_agent_id = -1
        
        # Restore agents with full state
        for agent_data in state['agents']:
            if agent_data.get('brain_type') == 'recurrent':
                # Load recurrent network
                brain = RecurrentNeuralNetwork(
                    input_size=input_size,
                    reservoir_size=agent_data['reservoir_size'],
                    output_size=output_size,
                    spectral_radius=agent_data['spectral_radius'],
                    sparsity=agent_data['sparsity']
                )
                brain.input_weights = np.array(agent_data['input_weights'])
                brain.reservoir_weights = np.array(agent_data['reservoir_weights'])
                brain.output_weights = np.array(agent_data['output_weights'])
                brain.output_bias = np.array(agent_data['output_bias'])
                brain.reservoir_activations = [ActivationType(act) for act in agent_data['reservoir_activations']]
                brain.output_activations = [ActivationType(act) for act in agent_data['output_activations']]
                brain.reset_state()  # Start with clean state
            else:
                # Fallback for old feedforward networks
                layers = agent_data['brain_layers']
                brain = NeuralNetwork(layers[0], layers[1:-1], layers[-1])
                brain.weights = [np.array(w) for w in agent_data['brain_weights']]
                brain.biases = [np.array(b) for b in agent_data['brain_biases']]
            
            # Create agent with saved position
            x = agent_data.get('x', random.uniform(50, self.environment.width - 50))
            y = agent_data.get('y', random.uniform(50, self.environment.height - 50))
            agent_id = agent_data.get('agent_id', 0)
            
            agent = Agent(
                agent_id=agent_id,
                x=x,
                y=y,
                brain=brain,
                generation_born=agent_data.get('generation_born', self.environment.generation)
            )
            max_agent_id = max(max_agent_id, agent_id)
            
            # Restore all agent stats (UPDATED)
            agent.alive = agent_data.get('alive', True)
            agent.health = agent_data.get('health', agent.max_health)
            agent.food = agent_data.get('food', agent.max_food)
            agent.water = agent_data.get('water', agent.max_water)
            agent.food_inventory = agent_data.get('food_inventory', 0)
            agent.age = agent_data.get('age', 0)
            agent.fitness = agent_data.get('fitness', 0)  # RESTORE FITNESS
            agent.food_collected = agent_data.get('food_collected', 0)
            agent.water_collected = agent_data.get('water_collected', 0)
            agent.children = agent_data.get('children', 0)
            agent.gardens_created = agent_data.get('gardens_created', 0)
            agent.gardens_planted = agent_data.get('gardens_planted', 0)
            agent.gardens_watered = agent_data.get('gardens_watered', 0)
            agent.gardens_harvested = agent_data.get('gardens_harvested', 0)
            
            # Restore velocity
            agent.circle.velocity_x = agent_data.get('velocity_x', 0)
            agent.circle.velocity_y = agent_data.get('velocity_y', 0)
            
            self.environment.agents.append(agent)
        
        self.environment.next_agent_id = max_agent_id + 1

        # Restore Hall of Fame
        self.environment.hall_of_fame = []
        if 'hall_of_fame' in state:
            for hof_data in state['hall_of_fame']:
                # Check if this is a recurrent network or feedforward network
                if hof_data.get('brain_type') == 'recurrent':
                    # Load recurrent network from Hall of Fame
                    brain = RecurrentNeuralNetwork(
                        input_size=input_size,
                        reservoir_size=hof_data['reservoir_size'],
                        output_size=output_size,
                        spectral_radius=hof_data['spectral_radius'],
                        sparsity=hof_data['sparsity']
                    )
                    brain.input_weights = np.array(hof_data['input_weights'])
                    brain.reservoir_weights = np.array(hof_data['reservoir_weights'])
                    brain.output_weights = np.array(hof_data['output_weights'])
                    brain.output_bias = np.array(hof_data['output_bias'])
                    brain.reservoir_activations = [ActivationType(act) for act in hof_data['reservoir_activations']]
                    brain.output_activations = [ActivationType(act) for act in hof_data['output_activations']]
                    brain.reset_state()  # Start with clean state
                    
                    # Get architecture string for recurrent network
                    architecture = brain.get_architecture_string()
                    complexity = brain.get_complexity()
                else:
                    # Fallback for old feedforward networks (backwards compatibility)
                    layers = hof_data['brain_layers']
                    brain = NeuralNetwork(layers[0], layers[1:-1], layers[-1])
                    brain.weights = [np.array(w) for w in hof_data['brain_weights']]
                    brain.biases = [np.array(b) for b in hof_data['brain_biases']]
                    
                    # Use stored architecture or generate it
                    architecture = hof_data.get('architecture', brain.get_architecture_string())
                    complexity = hof_data.get('complexity', brain.get_complexity())
                
                # Create Hall of Fame record
                hof_record = {
                    'brain': brain,
                    'fitness': hof_data['fitness'],
                    'generation': hof_data['generation'],
                    'food_collected': hof_data['food_collected'],
                    'water_collected': hof_data['water_collected'],
                    'age': hof_data['age'],
                    'children': hof_data['children'],
                    'architecture': architecture,
                    'complexity': complexity,
                    'agent_id': hof_data.get('agent_id')
                }
                self.environment.hall_of_fame.append(hof_record)
        
        # Restore gardens
        self.environment.gardens = []
        if 'gardens' in state:
            for garden_data in state['gardens']:
                garden = Garden(
                    x=garden_data['x'],
                    y=garden_data['y'],
                    radius=garden_data.get('radius', 15.0),
                    owner_id=garden_data.get('owner_id'),
                )
                garden.planted = garden_data.get('planted', False)
                garden.watered = garden_data.get('watered', False)
                garden.growth_timer = garden_data.get('growth_timer', 0)
                garden.growth_time = garden_data.get('growth_time', 500)
                garden.cooldown_timer = garden_data.get('cooldown_timer', 0)
                garden.cooldown_time = garden_data.get('cooldown_time', 300)
                garden.food_multiplier = garden_data.get('food_multiplier', 5)
                garden.harvested = garden_data.get('harvested', False)
                garden.times_harvested = garden_data.get('times_harvested', 0)
                garden.last_interaction_timestep = garden_data.get('last_interaction_timestep', 0)
                garden.inactivity_threshold = garden_data.get('inactivity_threshold', 10000)
                
                self.environment.gardens.append(garden)
        
        print(f"\n📂 Simulation state loaded from {filename}")
        print(f"   - Generation {self.environment.generation}")
        print(f"   - {len(self.environment.agents)} agents restored")
        print(f"   - {len(self.environment.hall_of_fame)} Hall of Fame members")
        print(f"   - {len(self.environment.gardens)} gardens restored")
        
        # Print agent stats summary
        if self.environment.agents:
            total_fitness = sum(a.fitness for a in self.environment.agents)
            avg_fitness = total_fitness / len(self.environment.agents)
            max_fitness = max(a.fitness for a in self.environment.agents)
            print(f"   - Agent fitness: avg={avg_fitness:.2f}, max={max_fitness:.2f}")
        
        if self.environment.hall_of_fame:
            best_hof = self.environment.hall_of_fame[0]
            print(f"   - Best ever: {best_hof['fitness']:.2f} (Gen {best_hof['generation']}, {best_hof['architecture']})")

    def save_best_run(self, filename: str = None):
        """
        Save the simulation with a filename indicating the best fitness.
        
        Args:
            filename: Optional custom filename
        """
        if filename is None:
            # Auto-generate filename with best fitness
            best_fitness = int(self.environment.best_fitness_ever)
            filename = f'best_run.pkl'
        
        self.save_state(filename)


# ============================================================================
# EXAMPLE USAGE / RUNTIME
# ============================================================================

def run_simulation():
    """
    Main runtime function to start the simulation.
    """
    print("="*60)
    print("AI CIRCLE ENVIRONMENT SIMULATION")
    print("="*60)
    print()
    
    # Create environment with 20 agents
    env = Environment(width=5000, height=5000, num_agents=20)
    
    # Create simulation
    sim = Simulation(env)
    
    # Run simulation for multiple generations
    # Will automatically evolve when generation completes
    sim.run(max_timesteps=10000)
    
    # Optionally save the best performing generation
    # sim.save_state('best_generation.pkl')
    
    print("\nSimulation complete!")

def run_simulation_with_saves():
    """
    Example: Run simulation with saving and loading.
    """
    load_previous = False  # Set to True to continue from save
    # Create environment and simulation
    if load_previous:
        env = Environment(width=3000, height=3000, num_agents=30)
    else:
        env = Environment(width=3000, height=3000, num_agents=0)
        env.evolve_population()
    sim = Simulation(env, 1)
    
    
    # Option 1: Load previous progress
    
    if load_previous:
        try:
            sim.load_state('final_state.pkl')
            print("Continuing from previous run!")
        except FileNotFoundError:
            print("No save file found, starting fresh")
        
    
    # Run simulation with auto-save every 10 generations
    sim.run(max_timesteps=None, auto_save_interval=10)
    
    print("\nFinal Hall of Fame:")
    for i, record in enumerate(env.hall_of_fame[:5]):
        print(f"{i+1}. Gen {record['generation']}: "
              f"Fitness {record['fitness']:.2f}, "
              f"Architecture {record['architecture']}")


    

def run_quick_test():
    """
    Quick test to verify everything works.
    """
    print("Running quick test (500 timesteps)...\n")
    
    env = Environment(width=1000, height=800, num_agents=0)
    env.evolve_population()
    sim = Simulation(env)
    
    # Run for just 500 steps
    i = 0
    #sim.save_state('save.pkl')
    #sim.load_state('save.pkl')
    while(True):
        for i in range(10000):
            time.sleep(0.001)
            sim.step()
            if i % 25 == 0:
                env.render()
        sim.save_state('save.pkl')
        if i == 1:
            break
    
    print("\nQuick test complete!")

def test_recurrent_network():
    """Test the recurrent neural network."""
    
    # Create network
    net = RecurrentNeuralNetwork(
        input_size=14,
        reservoir_size=50,
        output_size=6,
        spectral_radius=0.9,
        sparsity=0.1
    )
    
    print("Network created:")
    print(f"  Architecture: {net.get_architecture_string()}")
    print(f"  Complexity: {net.get_complexity()} parameters")
    print(f"  Spectral radius: {net.spectral_radius}")
    
    # Test forward pass
    test_input = np.random.randn(14)
    
    print("\nTesting temporal dynamics (should show recurrence):")
    for i in range(5):
        output = net.forward(test_input)
        print(f"  Step {i}: Output = {output[:3]}... (first 3 values)")
    
    # Test mutation
    print("\nTesting mutation:")
    original_arch = net.get_architecture_string()
    net.mutate(structural_mutation_rate=1.0)  # Force structural change
    print(f"  Before: {original_arch}")
    print(f"  After: {net.get_architecture_string()}")
    
    # Test activation function diversity
    print("\nActivation functions in use:")
    activation_counts = {}
    for act in net.reservoir_activations:
        activation_counts[act.value] = activation_counts.get(act.value, 0) + 1
    for act_type, count in activation_counts.items():
        print(f"  {act_type}: {count} neurons")


if __name__ == "__main__":
    # Choose which mode to run:
    
    # Full simulation with evolution
    #run_simulation()
    
    # Or run quick test
    #run_quick_test()
    pygame.init()
    run_simulation_with_saves()
    #test_recurrent_network()