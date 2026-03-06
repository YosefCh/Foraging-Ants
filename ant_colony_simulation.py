"""
Ant Colony Foraging Simulation
================================
A demonstration of stigmergy and distributed reinforcement learning through ant foraging behavior.

This module provides a configurable simulation comparing three strategies:
1. Ant Colony (with pheromones) - emergent intelligence
2. Random Exploration (no memory) - baseline
3. Ideal Allocation (omniscient) - theoretical upper bound
"""

import math
import random as rd
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy
import matplotlib.pyplot as plt
import warnings
import sys

try:
    from IPython.display import HTML
except ImportError:
    HTML = None


class AntColonySimulation:
    """
    Simulates ant colony foraging behavior with pheromone-based optimization.
    
    Compares emergent intelligence (pheromone trails) against random exploration
    and ideal allocation to measure efficiency gains from distributed reinforcement learning.
    """
    
    def __init__(self, config=None):
        """
        Initialize the ant colony simulation with configurable parameters.
        
        Parameters:
        -----------
        config : dict, optional
            Configuration dictionary with the following optional keys:
            
            Environment parameters:
            - 'nest_position': tuple, default (0, 0) - Nest coordinates
            - 'grid_size': int, default 40 - Size of the foraging area
            - 'num_food_sources': int, default 10 - Number of food patches
            
            ========================================================================
            The following parameters serve a dual purpose: 
            1. They define the limiting characteristics (quantity, quality, ease of access) of each food location.
            2. They establish the normalization ranges for these attributes, which are crucial for calculating pheromone deposits.
               This range is how the simulation imagines the ants perceive the environment and make decisions based on relative differences between food sources.
            
            - 'min_food_quantity': int, default 100 - Minimum food per patch
            - 'max_food_quantity': int, default 1000 - Maximum food per patch
            - 'min_quality': int, default 1 - Minimum food quality (1-10)
            - 'max_quality': int, default 10 - Maximum food quality (1-10)
            - 'min_ease': int, default 1 - Minimum ease of access (1-10)
            - 'max_ease': int, default 10 - Maximum ease of access (1-10)
            =========================================================================
            
            Ant behavior parameters:
            - 'ant_perception': int, default 5 - Angular detection radius (degrees)
            - 'max_ants': int, default 10000 - Maximum iterations for ant colony
            - 'scout_frequency': int, default 12 - Every Nth ant scouts randomly
            - 'evaporation_frequency': int, default 8 - Evaporation every N iterations
            - 'evaporation_rate': float, default 0.35 - Pheromone decay rate (0-1)
            - 'pheromone_threshold': float, default 0.05 - Minimum pheromone to follow
            
            Pheromone deposit parameters:
            - 'pheromone_midpoint': float, default 0.6 - Threshold for penalty/reward
            - 'pheromone_range': float, default 0.06 - Neutral zone around midpoint
            - 'penalty_exponent': float, default 1.5 - Low score penalty power
            - 'penalty_multiplier': float, default 0.75 - Low score penalty scale
            - 'reward_exponent': float, default 0.4 - High score reward power
            - 'reward_multiplier': float, default 1.85 - High score reward scale
            
            Food score weights:
            - 'weight_distance': float, default 0.4 - Distance importance
            - 'weight_quality': float, default 0.225 - Quality importance
            - 'weight_quantity': float, default 0.225 - Quantity importance
            - 'weight_ease': float, default 0.15 - Ease of access importance
            
            Random simulation parameters:
            - 'random_perception': int, default 3 - Angular detection for random ants
        """
        # Set default configuration
        self.config = {
            # Environment
            'nest_position': (0, 0),
            'grid_size': 40,
            'num_food_sources': 10,
            'min_food_quantity': 100,
            'max_food_quantity': 1000,
            'min_quality': 1,
            'max_quality': 10,
            'min_ease': 1,
            'max_ease': 10,
            
            # Ant behavior
            'ant_perception': 5,
            'max_ants': 10000,
            'scout_frequency': 12,
            'evaporation_frequency': 8,
            'evaporation_rate': 0.35,
            'pheromone_threshold': 0.05,
            
            # Pheromone deposit
            'pheromone_midpoint': 0.6,
            'pheromone_range': 0.06,
            'penalty_exponent': 1.5,
            'penalty_multiplier': 0.75,
            'reward_exponent': 0.4,
            'reward_multiplier': 1.85,
            
            # Food score weights
            'weight_distance': 0.4,
            'weight_quality': 0.225,
            'weight_quantity': 0.225,
            'weight_ease': 0.15,
            
            # Random simulation
            'random_perception': 3,
        }
        
        # Update with user-provided config (excluding static parameters)
        if config:
            # Static parameters that should not be overridden
            # These define the environment and ant perception baseline
            static_params = {
                'nest_position', 'grid_size', 'num_food_sources', 'pheromone_threshold',
                'min_food_quantity', 'max_food_quantity', 'min_quality', 'max_quality', 'min_ease', 'max_ease'
            }
            
            # Only update non-static parameters (silently ignore static ones)
            for key, value in config.items():
                if key not in static_params:
                    self.config[key] = value
        
        # Initialize simulation state
        self.environment_initialized = False
        self.ant_colony_completed = False
        self.random_completed = False
    
        
    def initialize_environment(self, seed=None):
        """
        Initialize the foraging environment with food sources.
        
        Parameters:
        -----------
        seed : int, optional
            Random seed for reproducibility
        """
        if seed is not None:
            rd.seed(seed)
        
        # Extract config values
        SIZE = self.config['grid_size']
        NEST = self.config['nest_position']
        num_sources = self.config['num_food_sources']
        
        # Generate food locations
        self.nest = NEST
        self.food_locations = [(rd.randint(-SIZE, SIZE), rd.randint(-SIZE, SIZE)) 
                               for _ in range(num_sources)]
        
        # Calculate angles from nest to food
        self.food_locations_angles = [
            round(math.degrees(math.atan2(y - NEST[1], x - NEST[0])) % 360, 2) 
            for x, y in self.food_locations
        ]
        
        # Calculate distances from nest to food
        self.food_distances = [
            math.sqrt((x - NEST[0]) ** 2 + (y - NEST[1]) ** 2) 
            for x, y in self.food_locations
        ]
        
        # Generate food attributes
        self.food_quality = [rd.randint(self.config['min_quality'], self.config['max_quality']) 
                            for _ in range(num_sources)]
        self.food_quantity = [rd.randint(self.config['min_food_quantity'], 
                                        self.config['max_food_quantity']) 
                             for _ in range(num_sources)]
        self.food_patch_size = [math.sqrt(x) / 10 for x in self.food_quantity]
        self.food_ease_of_access = [rd.randint(self.config['min_ease'], self.config['max_ease']) 
                                   for _ in range(num_sources)]
        
        # Save original quantities for analysis
        self.original_food_quantity = deepcopy(self.food_quantity)
        
        # Create copies for random simulation
        self._create_copies()
        
        # Calculate normalized values and scores
        self._calculate_normalized_values()
        self._calculate_food_scores()
        
        self.environment_initialized = True
        
        return self
    
    def _create_copies(self):
        """Create copies of environment data for random simulation."""
        self.food_locations_copy = deepcopy(self.food_locations)
        self.food_quantity_copy = deepcopy(self.food_quantity)
        self.food_ease_of_access_copy = deepcopy(self.food_ease_of_access)
        self.food_distances_copy = deepcopy(self.food_distances)
        self.food_locations_angles_copy = deepcopy(self.food_locations_angles)
        self.food_patch_size_copy = deepcopy(self.food_patch_size)
    
    def _normalize(self, data_list, feature_bottom, feature_top, invert=False):
        """Normalize data using MinMaxScaler."""
        temp_list = data_list.copy()
        temp_list.append(feature_bottom)
        temp_list.append(feature_top)
        scaler = MinMaxScaler(feature_range=(0.1, 1.1))
        normalized = scaler.fit_transform([[x] for x in temp_list])
        
        if invert:
            with_edges = [1.2 - x[0] for x in normalized]
            return with_edges[:-2]
        else:
            with_edges = [x[0] for x in normalized]
            return with_edges[:-2]
    
    def _calculate_normalized_values(self):
        """Calculate normalized values for all food attributes."""
        SIZE = self.config['grid_size']
        
        # Normalization ranges
        min_dist = 1
        max_dist = math.sqrt((SIZE - self.nest[0]) ** 2 + (SIZE - self.nest[1]) ** 2)
        
        self.norm_dist = self._normalize(self.food_distances, min_dist, max_dist, invert=True)
        self.norm_qual = self._normalize(self.food_quality, 
                                        self.config['min_quality'], 
                                        self.config['max_quality'], 
                                        invert=False)
        self.norm_quant = self._normalize(self.food_quantity, 
                                         self.config['min_food_quantity'], 
                                         self.config['max_food_quantity'], 
                                         invert=False)
        self.norm_ease = self._normalize(self.food_ease_of_access, 
                                        self.config['min_ease'], 
                                        self.config['max_ease'], 
                                        invert=False)
    
    def _calculate_food_scores(self):
        """Calculate composite food scores based on weighted attributes."""
        w_dist = self.config['weight_distance']
        w_qual = self.config['weight_quality']
        w_quant = self.config['weight_quantity']
        w_ease = self.config['weight_ease']
        
        self.food_scores = [
            nd * w_dist + nqlty * w_qual + nqty * w_quant + nea * w_ease
            for nd, nqlty, nqty, nea in zip(self.norm_dist, self.norm_qual, 
                                            self.norm_quant, self.norm_ease)
        ]
    
    def _within_range(self, angle, use_copy=False):
        """
        Check if angle detects any food source within perception range.
        
        Parameters:
        -----------
        angle : float
            Angle in degrees from nest
        use_copy : bool
            If True, use copied environment (for random simulation)
        
        Returns:
        --------
        int or False
            Index of closest food source within range, or False if none
        """
        if use_copy:
            angles = self.food_locations_angles_copy
            quantities = self.food_quantity_copy
            patch_sizes = self.food_patch_size_copy
            distances = self.food_distances_copy

        else:
            angles = self.food_locations_angles
            quantities = self.food_quantity
            patch_sizes = self.food_patch_size
            distances = self.food_distances
        
        perception = self.config['ant_perception']
        
        # Calculate angular distances (handle wraparound)
        angle_distances = [min(abs(angle - i), 360 - abs(angle - i)) for i in angles]
        angle_distances = [round(d, 1) for d in angle_distances]
        
        # Adjust for patch size
        distances_with_patch = [round(dist - (patch_sizes[idx] / 2), 1) 
                               for idx, dist in enumerate(angle_distances)]
        
        # Find sources within perception range with remaining food
        within_hit_radius = [idx for idx, dist in enumerate(distances_with_patch) 
                            if dist <= perception and quantities[idx] > 0]
        
        if not within_hit_radius:
            return False
        
        # Return closest food source
        best_idx = min(within_hit_radius, key=lambda idx: distances[idx])
        return best_idx
    
    def _pheromone_deposit(self, food_idx):
        """
        Calculate pheromone deposit based on food attributes.
        
        Uses non-linear scaling to amplify good sources and penalize bad ones.
        """
        # Recalculate quantity normalization (dynamic as food depletes)
        norm_quant = self._normalize(self.food_quantity, 
                                    self.config['min_food_quantity'], 
                                    self.config['max_food_quantity'], 
                                    invert=False)
        nqty = norm_quant[food_idx]
        
        # Use pre-calculated static normalized values
        nd = self.norm_dist[food_idx]
        nqlty = self.norm_qual[food_idx]
        nea = self.norm_ease[food_idx]
        
        # Calculate base deposit
        w_dist = self.config['weight_distance']
        w_qual = self.config['weight_quality']
        w_quant = self.config['weight_quantity']
        w_ease = self.config['weight_ease']
        
        base_deposit = nd * w_dist + nqlty * w_qual + nqty * w_quant + nea * w_ease
        
        # Apply non-linear scaling
        midpoint = self.config['pheromone_midpoint']
        range_val = self.config['pheromone_range']
        
        if base_deposit <= (midpoint - range_val):
            # Penalty for low scores
            deposit = ((base_deposit ** self.config['penalty_exponent']) * 
                      self.config['penalty_multiplier'])
        elif base_deposit >= (midpoint + range_val):
            # Reward for high scores
            excess = base_deposit - midpoint
            deposit = (midpoint + (excess ** self.config['reward_exponent']) * 
                      self.config['reward_multiplier'])
        else:
            # Neutral zone
            deposit = base_deposit
        
        return deposit
    
    def run_ant_colony_simulation(self, verbose=False):
        """
        Run the ant colony simulation with pheromone-based foraging.
        
        Parameters:
        -----------
        verbose : bool
            If True, print progress every 200 iterations
        
        Returns:
        --------
        dict
            Results containing paths, iterations, and ant behavior data
        """
        if not self.environment_initialized:
            raise RuntimeError("Environment not initialized. Call initialize_environment() first.")
        
        # Initialize simulation state
        max_ants = self.config['max_ants']
        scout_freq = self.config['scout_frequency']
        evap_freq = self.config['evaporation_frequency']
        evap_rate = self.config['evaporation_rate']
        threshold = self.config['pheromone_threshold']
        
        total_food = sum(self.food_quantity)
        iter_num = 0
        
        pheromone_levels = [0 for _ in range(len(self.food_locations))]
        food_paths = [0 for _ in range(len(self.food_locations))]
        actual_ant_paths = []
        iters = []
        
        if verbose:
            print(f"Starting Ant Colony Simulation...")
            print(f"Starting Food Supply: {total_food}\n")
        
        # Main simulation loop
        for i in range(max_ants):
            iter_num += 1
            
            # Check termination condition
            if max(pheromone_levels) <= threshold and sum(self.food_quantity) <= 0:
                if verbose:
                    print("All food depleted and no pheromones left. Ending simulation.")
                break
            
            # Get best pheromone path
            max_pher = max(pheromone_levels)
            best_path = max_pher if max_pher > threshold else 0
            
            # Progress display
            if verbose and iter_num % 200 == 0:
                print(f"Ant Iteration: {iter_num}")
                print(f"Paths: {food_paths}")
                print(f"Quantity: {self.food_quantity}")
                print(f"Pheromone Levels: {[round(x, 2) for x in pheromone_levels]}")
                print(f"Remaining: {sum(self.food_quantity)}\n")
            
            # Scout or follow pheromone
            if iter_num % scout_freq == 0 or best_path == 0:
                # Random exploration (scout)
                angle = rd.randint(0, 360)
                hit = self._within_range(angle, use_copy=False)
                
                if hit is not False:
                    food_paths[hit] += 1
                    if self.food_quantity[hit] > 0:
                        self.food_quantity[hit] -= 1
                        pheromone_levels[hit] += self._pheromone_deposit(hit)
                    actual_ant_paths.append((iter_num, angle, 'scout'))
                else:
                    actual_ant_paths.append((iter_num, angle, 'scout'))
            else:
                # Follow best pheromone path
                best_indices = [idx for idx, p in enumerate(pheromone_levels) if p == best_path]
                best_index = rd.choice(best_indices)
                food_paths[best_index] += 1
                
                if self.food_quantity[best_index] > 0:
                    self.food_quantity[best_index] -= 1
                    pheromone_levels[best_index] += self._pheromone_deposit(best_index)
                
                actual_ant_paths.append((iter_num, self.food_locations_angles[best_index], 'follower'))
            
            # Pheromone evaporation
            if iter_num % evap_freq == 0:
                pheromone_levels = [max(p * (1 - evap_rate), 0) for p in pheromone_levels]
            
            # Track when food depleted
            if sum(self.food_quantity) == 0:
                iters.append(iter_num)
        
        if verbose:
            depletion_iter = min(iters) if iters else "not depleted"
            print(f"\n✅ Ant Colony Simulation complete! {iter_num} ants processed")
            print(f"Food was depleted after {depletion_iter} iterations")
        
        # Store results
        self.ant_colony_paths = food_paths.copy()
        self.ant_colony_iterations = iter_num
        self.ant_colony_quantity_remaining = self.food_quantity.copy()
        self.actual_ant_paths = actual_ant_paths
        self.ant_colony_completed = True
        
        return {
            'paths': self.ant_colony_paths,
            'iterations': self.ant_colony_iterations,
            'remaining_food': self.ant_colony_quantity_remaining,
            'ant_behaviors': actual_ant_paths
        }
    
    def run_random_simulation(self, verbose=False):
        """
        Run random exploration simulation (no pheromones).
        
        Also captures a snapshot at the iteration when ant colony finished
        for fair comparison (both methods evaluated at same iteration count).
        
        Parameters:
        -----------
        verbose : bool
            If True, print progress every 200 iterations
        
        Returns:
        --------
        dict
            Results containing paths, iterations, and comparison snapshot
        """
        if not self.environment_initialized:
            raise RuntimeError("Environment not initialized. Call initialize_environment() first.")
        
        if not self.ant_colony_completed:
            raise RuntimeError("Ant colony simulation must be completed first for fair comparison.")
        
        total_food = sum(self.food_quantity_copy)
        iter_num = 0
        food_paths = [0 for _ in range(len(self.food_locations_copy))]
        actual_ant_paths = []
        
        # Track state at ant colony completion point for fair comparison
        random_paths_at_ant_completion = None
        food_collected_at_ant_completion = None
        
        if verbose:
            print(f"Starting Random Exploration Simulation...")
            print(f"Starting Food Supply: {total_food}")
            print(f"🎯 Will capture snapshot at iteration {self.ant_colony_iterations} (when ant colony finished)\n")
        
        # Run until all food is depleted
        while sum(self.food_quantity_copy) > 0:
            iter_num += 1
            
            # Capture snapshot when we reach ant colony's completion iteration
            if iter_num == self.ant_colony_iterations:
                random_paths_at_ant_completion = food_paths.copy()
                food_collected_at_ant_completion = [self.original_food_quantity[i] - self.food_quantity_copy[i] 
                                                    for i in range(len(self.original_food_quantity))]
                if verbose:
                    print(f"📸 SNAPSHOT at iteration {self.ant_colony_iterations}:")
                    print(f"   Random has collected: {sum(food_collected_at_ant_completion)}/{sum(self.original_food_quantity)} food items")
                    print(f"   Progress: {sum(food_collected_at_ant_completion)/sum(self.original_food_quantity)*100:.1f}%\n")
            
            # Progress display
            if verbose and iter_num % 200 == 0:
                print(f"Random Ant Iteration: {iter_num}")
                print(f"Paths: {food_paths}")
                print(f"Quantity: {self.food_quantity_copy}")
                print(f"Remaining: {sum(self.food_quantity_copy)}\n")
            
            # Random exploration
            angle = rd.randint(0, 360)
            hit = self._within_range(angle, use_copy=True)
            
            if hit is not False:
                food_paths[hit] += 1
                if self.food_quantity_copy[hit] > 0:
                    self.food_quantity_copy[hit] -= 1
            else:
                actual_ant_paths.append((iter_num, angle))
        
        if verbose:
            print(f"\n✅ Random Simulation complete! {iter_num} ants processed")
            print(f"All food depleted after {iter_num} iterations")
        
        # Store results (full completion)
        self.random_paths = food_paths.copy()
        self.random_iterations = iter_num
        self.random_quantity_remaining = self.food_quantity_copy.copy()
        
        # Store fair comparison results (at ant colony completion point)
        self.random_paths_at_comparison = random_paths_at_ant_completion.copy()
        self.food_collected_at_comparison = food_collected_at_ant_completion.copy()
        
        self.random_completed = True
        
        if verbose:
            print(f"\n📊 FULL COMPLETION (all food depleted):")
            print(f"   Random Visits: {self.random_paths}")
            print(f"   Total visits: {sum(self.random_paths)}")
            print(f"   Iterations to deplete food: {self.random_iterations}")
            print(f"\n🎯 FAIR COMPARISON (at iteration {self.ant_colony_iterations}):")
            print(f"   Random Visits: {self.random_paths_at_comparison}")
            print(f"   Total visits: {sum(self.random_paths_at_comparison)}")
            print(f"   Food collected: {sum(self.food_collected_at_comparison)}/{sum(self.original_food_quantity)} ({sum(self.food_collected_at_comparison)/sum(self.original_food_quantity)*100:.1f}%)")
        
        return {
            'paths': self.random_paths,
            'iterations': self.random_iterations,
            'remaining_food': self.random_quantity_remaining,
            'paths_at_comparison': self.random_paths_at_comparison,
            'food_collected_at_comparison': self.food_collected_at_comparison
        }
    
    def get_environment_summary(self):
        """
        Get a summary of the environment setup.
        
        Returns:
        --------
        IPython.display.HTML or str
            Formatted HTML table of food sources when IPython is available,
            otherwise a plain-text summary.
        """
        if not self.environment_initialized:
            return "Environment not initialized."

        rows = []
        for i in range(len(self.food_locations)):
            rows.append(
                {
                    "ID": f"F{i + 1}",
                    "Location": str(self.food_locations[i]),
                    "Dist": f"{self.food_distances[i]:.2f}",
                    "Angle": f"{self.food_locations_angles[i]:.2f}",
                    "Quality": f"{self.food_quality[i]:.0f}",
                    "Quantity": f"{self.original_food_quantity[i]}",
                    "Ease": f"{self.food_ease_of_access[i]:.0f}",
                    "Patch Size": f"{self.food_patch_size[i]:.1f}°",
                    "Score": f"{self.food_scores[i]:.2f}",
                }
            )

        if HTML is None:
            lines = ["🐜 FOOD SOURCES INITIALIZED:\n"]
            lines.append(
                f"{'ID':<4} {'Location':<15} {'Dist':<7} {'Angle':<7} {'Quality':<9} "
                f"{'Quantity':<10} {'Ease':<6} {'Patch-Size':<10} {'Score':<6}"
            )
            lines.append("=" * 85)

            for row in rows:
                lines.append(
                    f"{row['ID']:<4} {row['Location']:<15} {row['Dist']:<7} {row['Angle']:<7} "
                    f"{row['Quality']:<9} {row['Quantity']:<10} {row['Ease']:<6} "
                    f"{row['Patch Size']:<10} {row['Score']:<6}"
                )

            lines.append("\n✅ Environment ready for ant colony simulation!")
            return "\n".join(lines)

        header_html = """
<div style='margin-bottom: 10px;'>
  <h3 style='margin: 0 0 8px 0;'>🐜 Food Sources Initialized</h3>
  <p style='margin: 0; color: #444;'>
    This table summarizes the randomly generated environment before the simulation starts.
  </p>
</div>
"""

        column_order = ["ID", "Location", "Dist", "Angle", "Quality", "Quantity", "Ease", "Patch Size", "Score"]
        header_row = "".join(
            f"<th style='padding: 8px 10px; border-bottom: 2px solid #444; background: rgb(40, 140, 230); color:black; text-align: center;'>{col}</th>"
            for col in column_order
        )

        data_rows = []
        for row in rows:
            cells = []
            for col in column_order:
                extra_style = "font-weight: 700;" if col in {"ID", "Score"} else ""
                cells.append(
                    f"<td style='padding: 8px 10px; border-bottom: 1px solid #ddd; text-align: center; {extra_style}'>{row[col]}</td>"
                )
            data_rows.append("<tr>" + "".join(cells) + "</tr>")

        table_html = f"""
<table style='border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; font-size: 14px;'>
  <thead>
    <tr>{header_row}</tr>
  </thead>
  <tbody>
    {''.join(data_rows)}
  </tbody>
</table>
"""

        footer_html = """
<br>
<div style='margin-top: 10px; padding: 8px 10px; background: rgb(40, 140, 230); color:black; border-left: 4px solid #2e7d32;'><b>
  ✅ Environment ready for ant colony simulation!</b>
</div>
"""

        return HTML(header_html + table_html + footer_html)
    
    def visualize_environment(self, figsize=(12, 10)):
        """
        Visualize the nest and food locations with scores.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
        """
        if not self.environment_initialized:
            print("Environment not initialized.")
            return
        
        plt.figure(figsize=figsize)
        SIZE = self.config['grid_size']
        
        # Plot nest
        plt.scatter(self.nest[0], self.nest[1], s=300, c='red', marker='*', 
                   label='Nest', zorder=3, edgecolors='black', linewidths=2)
        
        # Plot food sources
        food_x = [loc[0] for loc in self.food_locations]
        food_y = [loc[1] for loc in self.food_locations]
        
        scatter = plt.scatter(food_x, food_y, s=100, c=self.food_scores, cmap='YlGn', 
                            vmin=0, vmax=1, edgecolors='black', linewidths=1.5, zorder=2)
        
        plt.colorbar(scatter, label='Food Score (darker = better)')
        
        # Draw food patch circles
        for i in range(len(self.food_locations)):
            angular_radius = self.food_patch_size[i] / 2
            spatial_radius = self.food_distances[i] * math.tan(math.radians(angular_radius))
            
            circle = plt.Circle((food_x[i], food_y[i]), spatial_radius, 
                              color='green', fill=False, linestyle='--', 
                              linewidth=1.5, alpha=0.5, 
                              label='Food Patch' if i == 0 else '')
            plt.gca().add_patch(circle)
        
        # Add labels
        for i in range(len(self.food_locations)):
            plt.annotate(f'F{i + 1}\n({self.food_patch_size[i]:.1f}°)', 
                        (food_x[i], food_y[i]), 
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, fontweight='bold')
        
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Ant Colony Foraging Environment\n'
                 '(Food sources with patch coverage, dashed circles show detection radius)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(-SIZE, SIZE)
        plt.ylim(-SIZE, SIZE)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
    
    def visualize_efficiency(self, figsize=(16, 6)):
        """
        Visualize efficiency comparison with pie charts showing hits vs misses.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
        """
        if not self.ant_colony_completed or not self.random_completed:
            print("Both simulations must be completed first.")
            return
        
        # Suppress matplotlib warnings
        warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
        
        # Calculate metrics
        total_food_items = sum(self.original_food_quantity)
        ideal_iterations = total_food_items
        ant_colony_iters = self.ant_colony_iterations
        random_iters = self.random_iterations
        
        # Print summary
        print("\n" + "="*80)
        print("⏱️  EFFICIENCY COMPARISON: Iterations to Deplete All Food")
        print("="*80)
        print(f"\nTotal Food Items: {total_food_items}")
        print(f"\n1. Ideal (Perfect Knowledge):  {ideal_iterations:,} iterations")
        print(f"   → Each food item collected exactly once (theoretical minimum)")
        print(f"\n2. Ant Colony (Pheromones):    {ant_colony_iters:,} iterations")
        print(f"   → {((ant_colony_iters / ideal_iterations) * 100):.1f}% of ideal")
        print(f"   → {ant_colony_iters - ideal_iterations:,} extra iterations")
        print(f"\n3. Random Exploration:         {random_iters:,} iterations")
        print(f"   → {((random_iters / ideal_iterations) * 100):.1f}% of ideal")
        print(f"   → {random_iters - ideal_iterations:,} extra iterations")
        print(f"\n💡 EFFICIENCY GAIN:")
        print(f"   Ant Colony is {((random_iters - ant_colony_iters) / random_iters * 100):.1f}% faster than Random")
        print(f"   Ant Colony saved {random_iters - ant_colony_iters:,} iterations!")
        print("="*80)
        
        # Calculate hits vs misses
        ant_colony_hits = total_food_items
        ant_colony_misses = ant_colony_iters - total_food_items
        random_hits = total_food_items
        random_misses = random_iters - total_food_items
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # --- Plot 1: Bar Chart - Iterations Comparison ---
        methods = ['Ideal\n(Perfect)', 'Ant Colony\n(Pheromones)', 'Random\n(No Memory)']
        iterations = [ideal_iterations, ant_colony_iters, random_iters]
        colors = ['gold', 'darkgreen', 'gray']
        alphas = [0.9, 0.85, 0.6]
        
        bars = ax1.bar(methods, iterations, color=colors, edgecolor='black', linewidth=2)
        for bar, alpha in zip(bars, alphas):
            bar.set_alpha(alpha)
        
        # Add value labels on bars
        for bar, iter_val in zip(bars, iterations):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{iter_val:,}',
                    ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        ax1.set_ylabel('Total Iterations to Deplete All Food', fontsize=13, fontweight='bold')
        ax1.set_title('Speed Comparison: How Many Iterations to Finish?', fontsize=15, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        ax1.set_ylim(0, max(iterations) * 1.15)
        
        # Add efficiency percentages
        efficiency_texts = [
            '100%\n(baseline)',
            f'{(ideal_iterations/ant_colony_iters*100):.1f}%\nof ideal',
            f'{(ideal_iterations/random_iters*100):.1f}%\nof ideal'
        ]
        for i, (bar, text) in enumerate(zip(bars, efficiency_texts)):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() * 0.5,
                    text,
                    ha='center', va='center', fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))
        
        # --- Plot 2: Pie Charts - Hits vs Misses ---
        # Create two side-by-side pie charts
        ax2.axis('off')  # Turn off the main axis
        
        # Create sub-axes for the two pie charts
        ax2_left = fig.add_axes([0.55, 0.15, 0.18, 0.7])
        ax2_right = fig.add_axes([0.77, 0.15, 0.18, 0.7])
        
        # Ant Colony pie chart
        ant_sizes = [ant_colony_hits, ant_colony_misses]
        ant_labels = [f'Hits\n{ant_colony_hits:,}\n({ant_colony_hits/ant_colony_iters*100:.1f}%)', 
                    f'Misses\n{ant_colony_misses:,}\n({ant_colony_misses/ant_colony_iters*100:.1f}%)']
        ant_colors = ['darkgreen', 'lightcoral']
        ant_explode = (0.05, 0)
        
        ax2_left.pie(ant_sizes, labels=ant_labels, colors=ant_colors, 
                    explode=ant_explode, autopct='', startangle=90,
                    textprops={'fontsize': 10, 'fontweight': 'bold'},
                    wedgeprops={'edgecolor': 'black', 'linewidth': 2})
        ax2_left.set_title('Ant Colony\n(Pheromones)', fontsize=12, fontweight='bold', pad=10)
        
        # Random pie chart
        random_sizes = [random_hits, random_misses]
        random_labels = [f'Hits\n{random_hits:,}\n({random_hits/random_iters*100:.1f}%)', 
                        f'Misses\n{random_misses:,}\n({random_misses/random_iters*100:.1f}%)']
        random_colors = ['darkgreen', 'lightcoral']
        random_explode = (0.05, 0)
        
        ax2_right.pie(random_sizes, labels=random_labels, colors=random_colors, 
                    explode=random_explode, autopct='', startangle=90,
                    textprops={'fontsize': 10, 'fontweight': 'bold'},
                    wedgeprops={'edgecolor': 'black', 'linewidth': 2})
        ax2_right.set_title('Random\n(No Memory)', fontsize=12, fontweight='bold', pad=10)
        
        # Add main title for the pie chart section
        fig.text(0.73, 0.92, 'Hits vs Misses: Successful Food Collection vs Wasted Trips', 
                ha='center', fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed breakdown
        print(f"\n📊 DETAILED BREAKDOWN:")
        print(f"   Ideal:       {ideal_iterations:,} iterations (0% waste)")
        print(f"   Ant Colony:  {ant_colony_iters:,} iterations ({(ant_colony_misses/ant_colony_iters*100):.1f}% waste)")
        print(f"   Random:      {random_iters:,} iterations ({(random_misses/random_iters*100):.1f}% waste)")
    
    def visualize_allocation(self, figsize=(20, 10)):
        """
        Visualize proportional visit allocation analysis.
        Shows how well visits are distributed according to food availability.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
        """
        if not self.ant_colony_completed or not self.random_completed:
            print("Both simulations must be completed first.")
            return
        
        import numpy as np
        
        # Suppress matplotlib warnings
        warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
        
        # Calculate ideal proportions (what % of total food each source represents)
        total_food = sum(self.original_food_quantity)
        ideal_proportions = [(qty / total_food * 100) for qty in self.original_food_quantity]
        
        # Calculate actual visit proportions for ant colony
        total_ant_visits = sum(self.ant_colony_paths)
        ant_visit_proportions = [(visits / total_ant_visits * 100) if total_ant_visits > 0 else 0 
                                for visits in self.ant_colony_paths]
        
        # Calculate actual visit proportions for random (at comparison point)
        total_random_visits = sum(self.random_paths_at_comparison)
        random_visit_proportions = [(visits / total_random_visits * 100) if total_random_visits > 0 else 0 
                                    for visits in self.random_paths_at_comparison]
        
        # Print detailed comparison table
        print("\n" + "="*120)
        print("📊 PROPORTIONAL VISIT ALLOCATION: Are visits distributed according to food availability?")
        print("="*120)
        print(f"\n{'Source':<8} {'Food':<10} {'Ideal %':<12} {'Ant Colony':<25} {'Random':<25} {'Score':<8}")
        print(f"{'ID':<8} {'Qty':<10} {'(Target)':<12} {'Visits (% of Total)':<25} {'Visits (% of Total)':<25} {'(Quality)':<8}")
        print("-"*120)
        
        for i in range(len(self.original_food_quantity)):
            # Calculate how close each method got to ideal proportion
            ant_deviation = ant_visit_proportions[i] - ideal_proportions[i]
            random_deviation = random_visit_proportions[i] - ideal_proportions[i]
            
            ant_marker = "✓" if abs(ant_deviation) < 2 else ("↑" if ant_deviation > 0 else "↓")
            random_marker = "✓" if abs(random_deviation) < 2 else ("↑" if random_deviation > 0 else "↓")
            
            print(f"F{i+1:<7} {self.original_food_quantity[i]:<10} {ideal_proportions[i]:>6.2f}%      "
                f"{self.ant_colony_paths[i]:<7} ({ant_visit_proportions[i]:>5.2f}%) {ant_marker}    "
                f"{self.random_paths_at_comparison[i]:<7} ({random_visit_proportions[i]:>5.2f}%) {random_marker}    "
                f"{self.food_scores[i]:<6.2f}")
        
        print("-"*120)
        print(f"{'TOTAL':<8} {total_food:<10} 100.00%      "
            f"{total_ant_visits:<7} (100.00%)       "
            f"{total_random_visits:<7} (100.00%)")
        print("="*120)
        
        # Calculate allocation accuracy metrics
        ant_allocation_error = np.mean([abs(ant_visit_proportions[i] - ideal_proportions[i]) 
                                        for i in range(len(self.original_food_quantity))])
        random_allocation_error = np.mean([abs(random_visit_proportions[i] - ideal_proportions[i]) 
                                            for i in range(len(self.original_food_quantity))])
        
        print(f"\n💡 ALLOCATION ACCURACY (Mean Absolute Deviation):")
        print(f"   📐 Formula: Average of |actual% - ideal%| across all sources")
        print(f"   Ant Colony: {ant_allocation_error:.2f}% average deviation from ideal")
        print(f"   Random:     {random_allocation_error:.2f}% average deviation from ideal")
        print(f"   Improvement: {((random_allocation_error - ant_allocation_error) / random_allocation_error * 100):.1f}% better allocation by ant colony!")
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)
        
        # --- Plot 1: Stacked Bar Chart - Proportional Distribution ---
        ax1 = fig.add_subplot(gs[0, :])
        x = np.arange(len(self.original_food_quantity))
        width = 0.25
        
        bars1 = ax1.bar(x - width, ideal_proportions, width, label='Ideal (Target)', 
                        color='gold', alpha=0.9, edgecolor='black', linewidth=1.5)
        bars2 = ax1.bar(x, ant_visit_proportions, width, label='Ant Colony (Actual)', 
                        color='darkgreen', alpha=0.8, edgecolor='black', linewidth=1.5)
        bars3 = ax1.bar(x + width, random_visit_proportions, width, label='Random (Actual)', 
                        color='gray', alpha=0.6, edgecolor='black', linewidth=1.5)
        
        ax1.set_xlabel('Food Source ID', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Percentage of Total (%)', fontsize=13, fontweight='bold')
        ax1.set_title('Visit Allocation Proportions by Food Source\n(Do visits match food availability?)', 
                    fontsize=15, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'F{i+1}' for i in range(len(self.original_food_quantity))])
        ax1.legend(loc='upper right', fontsize=12)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars (only show if > 3% to avoid clutter)
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 3:
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.1f}%',
                            ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # --- Plot 2: Deviation from Ideal (Left bottom) ---
        ax2 = fig.add_subplot(gs[1, 0])
        
        ant_deviations = [ant_visit_proportions[i] - ideal_proportions[i] 
                        for i in range(len(self.original_food_quantity))]
        random_deviations = [random_visit_proportions[i] - ideal_proportions[i] 
                            for i in range(len(self.original_food_quantity))]
        
        x_pos = np.arange(len(self.original_food_quantity))
        width = 0.35
        
        # Use consistent colors: green for ant colony, gray for random
        bars_dev1 = ax2.bar(x_pos - width/2, ant_deviations, width, label='Ant Colony', 
                            color='darkgreen', alpha=0.8, edgecolor='black', linewidth=1.5)
        bars_dev2 = ax2.bar(x_pos + width/2, random_deviations, width, label='Random', 
                            color='gray', alpha=0.6, edgecolor='black', linewidth=1.5)
        
        ax2.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Perfect (0% error)')
        ax2.set_xlabel('Food Source ID', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Deviation from Ideal (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Allocation Deviation by Source\n(Above red line = over-visited, Below = under-visited)', 
                    fontsize=13, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f'F{i+1}' for i in range(len(self.original_food_quantity))])
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add annotation box explaining the plot
        explanation = "📊 HOW TO READ THIS:\n" \
                    "• Bars ABOVE red line:\n" \
                    "  Source got MORE visits\n" \
                    "  than ideal (over-visited)\n\n" \
                    "• Bars BELOW red line:\n" \
                    "  Source got FEWER visits\n" \
                    "  than ideal (under-visited)\n\n" \
                    "• Closer to red line = better!"
        ax2.text(0.98, 0.02, explanation,
                transform=ax2.transAxes, fontsize=8,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='black', linewidth=1.5),
                family='monospace')
        
        # --- Plot 3: Allocation Quality vs Food Score (Right bottom) ---
        ax3 = fig.add_subplot(gs[1, 1])
        
        # Calculate allocation accuracy per source (lower is better)
        ant_accuracy = [abs(ant_visit_proportions[i] - ideal_proportions[i]) 
                        for i in range(len(self.original_food_quantity))]
        random_accuracy = [abs(random_visit_proportions[i] - ideal_proportions[i]) 
                        for i in range(len(self.original_food_quantity))]
        
        ax3.scatter(self.food_scores, ant_accuracy, s=200, c='darkgreen', alpha=0.7, 
                edgecolors='black', linewidths=2, label='Ant Colony', marker='o', zorder=3)
        ax3.scatter(self.food_scores, random_accuracy, s=200, c='gray', alpha=0.6, 
                edgecolors='black', linewidths=2, label='Random', marker='s', zorder=2)
        
        # Add labels
        for i in range(len(self.food_scores)):
            ax3.annotate(f'F{i+1}', (self.food_scores[i], ant_accuracy[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold',
                        color='darkgreen')
            ax3.annotate(f'F{i+1}', (self.food_scores[i], random_accuracy[i]), 
                        xytext=(5, -12), textcoords='offset points', fontsize=9, fontweight='bold',
                        color='gray')
        
        ax3.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Perfect (0% error)')
        ax3.set_xlabel('Food Source Score (Quality)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Allocation Error (% deviation)', fontsize=12, fontweight='bold')
        ax3.set_title('Correlation: Food Quality vs Allocation Accuracy\n(Lower is better - closer to ideal)', 
                    fontsize=13, fontweight='bold')
        ax3.legend(loc='upper right', fontsize=10)
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.set_ylim(bottom=-1)
        
        # Add insight text box
        well_allocated_ant = sum(1 for acc in ant_accuracy if acc < 2)
        well_allocated_random = sum(1 for acc in random_accuracy if acc < 2)
        
        textstr = f'Allocation Quality:\n' \
                f'Sources within ±2%:\n' \
                f'  Ant: {well_allocated_ant}/{len(self.original_food_quantity)}\n' \
                f'  Random: {well_allocated_random}/{len(self.original_food_quantity)}\n\n' \
                f'Avg error:\n' \
                f'  Ant: {ant_allocation_error:.2f}%\n' \
                f'  Random: {random_allocation_error:.2f}%\n\n' \
                f'🎯 Ant colony is\n{((random_allocation_error - ant_allocation_error) / random_allocation_error * 100):.0f}% more accurate!'
        ax3.text(0.02, 0.98, textstr,
                transform=ax3.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='black', linewidth=2),
                family='monospace', fontweight='bold')
        
        plt.suptitle(f'Proportional Visit Allocation Analysis (at iteration {self.ant_colony_iterations:,})', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.show()
        
        # Print detailed insights
        print(f"\n🎯 KEY INSIGHTS:")
        print(f"\n1. BEST ALLOCATED SOURCES (Ant Colony):")
        best_ant = sorted(enumerate(ant_accuracy), key=lambda x: x[1])[:3]
        for idx, error in best_ant:
            print(f"   F{idx+1}: {error:.2f}% error (ideal: {ideal_proportions[idx]:.1f}%, actual: {ant_visit_proportions[idx]:.1f}%)")
        
        print(f"\n2. WORST ALLOCATED SOURCES (Ant Colony):")
        worst_ant = sorted(enumerate(ant_accuracy), key=lambda x: x[1], reverse=True)[:3]
        for idx, error in worst_ant:
            print(f"   F{idx+1}: {error:.2f}% error (ideal: {ideal_proportions[idx]:.1f}%, actual: {ant_visit_proportions[idx]:.1f}%)")
        
        print(f"\n3. RANDOM ALLOCATION PATTERN:")
        print(f"   Average visit proportion: {np.mean(random_visit_proportions):.2f}% (should vary by source)")
        print(f"   Standard deviation: {np.std(random_visit_proportions):.2f}%")
        print(f"   → Random tends toward UNIFORM distribution (bad!)")
        
        print(f"\n4. ANT COLONY ALLOCATION PATTERN:")
        print(f"   Correlation with food scores: {np.corrcoef(self.food_scores, ant_visit_proportions)[0,1]:.3f}")
        print(f"   Correlation with food quantities: {np.corrcoef(self.original_food_quantity, ant_visit_proportions)[0,1]:.3f}")
        print(f"   → Pheromones create SMART allocation (good!)")
    
    def visualize_results(self, figsize_efficiency=(16, 6), figsize_allocation=(20, 10)):
        """
        Visualize all results: efficiency comparison and allocation analysis.
        
        Parameters:
        -----------
        figsize_efficiency : tuple
            Figure size for efficiency visualization
        figsize_allocation : tuple
            Figure size for allocation visualization
        """
        self.visualize_efficiency(figsize=figsize_efficiency)
        self.visualize_allocation(figsize=figsize_allocation)
