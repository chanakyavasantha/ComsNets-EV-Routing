# %% [markdown]
# #### Section 1: Importing Necessary Libraries

# %%
### Basic Imports
import numpy as np
import pandas as pd
import random
import toml
import os
import logging
import math
import json
from datetime import datetime
import seaborn as sns
### Matplot Lib Imports
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import List, Dict, Tuple, Optional
import itertools

### Parallel Processing Libraries
from functools import partial
import time
from concurrent.futures import ProcessPoolExecutor, as_completed,ThreadPoolExecutor
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import concurrent.futures

### Scipy Imports
from scipy.spatial import distance
from shapely.geometry import Point, MultiPoint
from shapely.ops import cascaded_union
from scipy.spatial import distance
from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.spatial.distance import cdist
### Other Imports
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any
from abc import ABC, abstractmethod
from matplotlib.colors import LinearSegmentedColormap


# %% [markdown]
# #### Section 1.1: Basic Utility Functions

# %%
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))

def create_distance_matrix(locations):
    n = len(locations)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i][j] = euclidean_distance(locations[i], locations[j])
    return matrix

def create_charging_distance_matrix(locations, charging_stations):
    matrix = np.zeros((len(locations), len(charging_stations)))
    for i, loc in enumerate(locations):
        for j, station in enumerate(charging_stations):
            matrix[i][j] = euclidean_distance(loc, station)
    return matrix

# %%
class EVRPSolution:
    def __init__(self):
        self.routes = []
        self.vehicle_types = []
        self.route_loads = []
        self.route_distances = []
        self.route_energies = []
        self.delivery_times = []
        self.computation_time = 0.0

    def add_route(self, route, vehicle_type, load):
        self.routes.append(route)
        self.vehicle_types.append(vehicle_type)
        self.route_loads.append(load)

# %%
class EVConfig:
    def __init__(self):
        self.categories = {
            'small': {
                'battery_capacity': 35,
                'base_weight': 1500,
                'load_capacity': 500
            },
            'medium': {
                'battery_capacity': 40,
                'base_weight': 1800,
                'load_capacity': 600
            },
            'large': {
                'battery_capacity': 45,
                'base_weight': 2000,
                'load_capacity': 700
            },
            'xlarge': {
                'battery_capacity': 50,
                'base_weight': 2200,
                'load_capacity': 800
            }
        }
        self.initial_charging = 100
        self.speed = 25
        self.energy_consumption_rate = 0.15
        self.weight_factor = 0.05
        self.battery_safety_margin = 40

# %%
class EVRPInstance:
    def __init__(self, instance_id, depot_location, customer_locations, 
                 charging_stations, customer_items_weights, charging_rate):
        self.instance_id = instance_id
        self.depot_location = depot_location
        self.customer_locations = customer_locations
        self.charging_stations = charging_stations
        self.customer_items_weights = customer_items_weights
        self.charging_rate = charging_rate
        
        # Create distance matrices
        self.distance_matrix = self._create_distance_matrix()
        self.charging_distance_matrix = self._create_charging_distance_matrix()

    def _create_distance_matrix(self):
        locations = [self.depot_location] + self.customer_locations
        return create_distance_matrix(locations)

    def _create_charging_distance_matrix(self):
        locations = [self.depot_location] + self.customer_locations
        return create_charging_distance_matrix(locations, self.charging_stations)

# %%
class GreedyEVRPSolver:
    def __init__(self, instance: EVRPInstance):
        self.instance = instance
        self.ev_config = EVConfig()
        self.best_solution = EVRPSolution()
        self.served_customers = set()
        self.fleet = {
            'xlarge': 0,
            'large': 0,
            'medium': 0,
            'small': 0
        }
        
    def calculate_min_vehicles_needed(self) -> Dict[str, int]:
        total_demand = sum(self.instance.customer_items_weights)
        unit_fleet_capacity = sum(
            self.ev_config.categories[v_type]['load_capacity']
            for v_type in ['xlarge', 'large', 'medium', 'small']
        )
        min_vehicles = math.ceil(total_demand / unit_fleet_capacity)
        return {
            'xlarge': min_vehicles,
            'large': min_vehicles,
            'medium': min_vehicles,
            'small': min_vehicles
        }

    def increase_fleet_binary(self, current_fleet: Dict[str, int]) -> Dict[str, int]:
        """Increase fleet following binary pattern"""
        # Get base value (minimum number of vehicles)
        base = min(current_fleet.values())
        
        # Convert current configuration to binary
        binary = ''
        for v_type in ['xlarge', 'large', 'medium', 'small']:
            binary += '1' if current_fleet[v_type] > base else '0'
        
        # Increment binary pattern
        current_value = int(binary, 2)
        next_value = current_value + 1
        
        # If we've used all patterns (1111), increment base
        if next_value > 15:  # 15 is 1111 in binary
            base += 1
            next_value = 0
        
        # Convert back to binary
        new_binary = format(next_value, '04b')
        
        # Create new fleet configuration
        new_fleet = {}
        for i, v_type in enumerate(['xlarge', 'large', 'medium', 'small']):
            new_fleet[v_type] = base + (1 if new_binary[i] == '1' else 0)
        
        return new_fleet
    
    def calculate_proportional_loads(self, fleet: Dict[str, int], total_demand: float) -> Dict[str, float]:
        proportional_loads = {}
        vehicle_capacities = {}
        total_vehicles = 0
        
        # Get vehicle capacities and count total vehicles
        for v_type, count in fleet.items():
            if count > 0:
                vehicle_capacities[v_type] = self.ev_config.categories[v_type]['load_capacity']
                total_vehicles += count
        
        if total_vehicles == 0:
            return {v_type: 0 for v_type in fleet}
        
        # Calculate base load per vehicle (equal distribution)
        base_load_per_vehicle = total_demand / total_vehicles
        
        # First pass: Assign base load or capacity, whichever is smaller
        remaining_demand = total_demand
        remaining_vehicles = total_vehicles
        
        for v_type in fleet:
            if fleet[v_type] > 0:
                capacity = vehicle_capacities[v_type]
                count = fleet[v_type]
                
                # Assign minimum of base load or capacity
                load_per_vehicle = min(base_load_per_vehicle, capacity)
                proportional_loads[v_type] = load_per_vehicle
                
                # Update remaining demand and vehicles
                actual_load = load_per_vehicle * count
                remaining_demand -= actual_load
                if load_per_vehicle < base_load_per_vehicle:
                    remaining_vehicles -= count
        
        # Second pass: Redistribute excess to vehicles with remaining capacity
        if remaining_demand > 0 and remaining_vehicles > 0:
            additional_load_per_vehicle = remaining_demand / remaining_vehicles
            
            for v_type in fleet:
                if fleet[v_type] > 0:
                    current_load = proportional_loads[v_type]
                    capacity = vehicle_capacities[v_type]
                    
                    if current_load < capacity:
                        # Can take more load
                        new_load = min(capacity, current_load + additional_load_per_vehicle)
                        proportional_loads[v_type] = new_load
        
        # Add zero loads for unused vehicle types
        for v_type in fleet:
            if v_type not in proportional_loads:
                proportional_loads[v_type] = 0
        
        # Print distribution information
        print(f"\nLoad Distribution Details:")
        print(f"Total Demand: {total_demand:.2f}")
        print(f"Base Load Per Vehicle: {base_load_per_vehicle:.2f}")
        print("\nPer Vehicle Type Assignment:")
        
        total_allocated = 0
        for v_type in proportional_loads:
            if fleet[v_type] > 0:
                load = proportional_loads[v_type]
                capacity = vehicle_capacities[v_type]
                count = fleet[v_type]
                total_type_load = load * count
                total_allocated += total_type_load
                
                print(f"{v_type:>8}: {load:.2f} kg/vehicle × {count} vehicles = {total_type_load:.2f} kg "
                    f"(capacity: {capacity:.2f} kg)")
        
        print(f"\nTotal Allocated: {total_allocated:.2f} kg")
        
        return proportional_loads

    def find_best_next_customer(self, current_pos: int, unserved_customers: List[int]) -> Optional[int]:
        best_customer = None
        min_distance = float('inf')
        
        for customer_idx in unserved_customers:
            if customer_idx in self.served_customers:
                continue
                
            customer_id = customer_idx + 1
            distance_to_customer = self.calculate_distance(current_pos, customer_id)
            
            # Simply choose the nearest customer
            if distance_to_customer < min_distance:
                min_distance = distance_to_customer
                best_customer = customer_idx
                
        return best_customer

    def create_route(self, unserved_customers: List[int], vehicle_type: str, 
                target_load: float) -> Tuple[List[int], float]:
        """
        Create a route allowing only one customer to exceed the proportional load target.
        
        Args:
            unserved_customers: List of customer indices not yet served
            vehicle_type: Type of vehicle ('small', 'medium', 'large', 'xlarge')
            target_load: Target proportional load for this vehicle type
            
        Returns:
            Tuple containing:
            - List[int]: Route (sequence of customer indices, starting and ending with depot)
            - float: Total load for the route
        """
        route = [0]  # Start from depot
        current_load = 0
        allowed_excess = True  # Flag to track if we can still allow one customer to exceed
        vehicle_capacity = self.ev_config.categories[vehicle_type]['load_capacity']
        
        while unserved_customers:
            next_customer = self.find_best_next_customer(route[-1], unserved_customers)
            
            if next_customer is None:
                break
                
            customer_demand = self.instance.customer_items_weights[next_customer]
            is_last_customer = len(unserved_customers) == 1
            
            # Check if adding this customer would exceed vehicle capacity
            if current_load + customer_demand > vehicle_capacity:
                break
                
            # Check if adding this customer would exceed target load
            exceeds_target = current_load + customer_demand > target_load
            
            if exceeds_target:
                if allowed_excess and not is_last_customer:
                    # Allow this customer but mark that we can't exceed again
                    allowed_excess = False
                else:
                    # We've already used our one excess or it's the last customer
                    break
                    
            customer_id = next_customer + 1
            route.append(customer_id)
            current_load += customer_demand
            
            self.served_customers.add(next_customer)
            unserved_customers.remove(next_customer)
            
        route.append(0)  # Return to depot
        return route, current_load

    def solve(self) -> EVRPSolution:
        """Main solving procedure with distance-based customer selection"""
        total_demand = sum(self.instance.customer_items_weights)
        # Start with minimum vehicles
        self.fleet = self.calculate_min_vehicles_needed()
        max_attempts = 16  # Maximum number of fleet configurations to try
        attempt = 0
        
        while attempt < max_attempts:
            print(f"\nAttempt {attempt + 1}: Trying fleet configuration: {self.fleet}")
            solution = EVRPSolution()
            solution.computation_time = 0.0
            self.served_customers.clear()
            
            proportional_loads = self.calculate_proportional_loads(self.fleet, total_demand)
            print(f"\nProportional target loads:")
            for v_type in ['small', 'medium', 'large', 'xlarge']:
                print(f"{v_type}: {proportional_loads[v_type]:.2f} kg (max capacity: {self.ev_config.categories[v_type]['load_capacity']} kg)")
            
            # Create routes for each vehicle type
            for v_type in ['small', 'medium', 'large', 'xlarge']:
                num_vehicles = self.fleet[v_type]
                
                print(f"\nAssigning customers to {v_type} vehicles")
                print(f"Target load per vehicle: {proportional_loads[v_type]:.2f} kg")
                
                for vehicle_num in range(num_vehicles):
                    if len(self.served_customers) == len(self.instance.customer_locations):
                        break
                        
                    unserved = [i for i in range(len(self.instance.customer_locations)) 
                            if i not in self.served_customers]
                    
                    route, load = self.create_route(unserved, v_type, 
                                                proportional_loads[v_type])  # Now using proportional load
                    
                    if route and len(route) > 2:  # If route contains any customers
                        route_with_charging = self.insert_charging_stations(route, load, v_type)
                        distance, energy, time, battery_levels = self.calculate_route_metrics(
                            route_with_charging, load, v_type)
                        
                        solution.add_route(route_with_charging, v_type, load)
                        solution.route_distances.append(distance)
                        solution.route_energies.append(energy)
                        solution.delivery_times.append(time)
                        
                        print(f"Created route for {v_type} vehicle {vehicle_num + 1}:")
                        print(f"Load: {load:.2f} kg")
                        print(f"Customers: {[c for c in route if c > 0]}")
            
            unserved_count = len(self.instance.customer_locations) - len(self.served_customers)
            if unserved_count == 0:
                print("\nAll customers served successfully!")
                return solution
            else:
                print(f"\nWarning: {unserved_count} customers remain unserved")
                print("Increasing fleet size...")
                self.fleet = self.increase_fleet_binary(self.fleet)
                attempt += 1
        
        print("\nNo feasible solution found within maximum attempts")
        return EVRPSolution()  # Return empty solution instead of None
    
    def insert_charging_stations(self, route: List[int], load: float, 
                               vehicle_type: str) -> List[int]:
        if len(route) <= 2:  # Only depot-customer-depot
            return route
            
        new_route = [0]  # Start at depot
        current_battery = self.ev_config.initial_charging
        current_load = load
        
        for i in range(1, len(route)):
            from_loc = route[i-1]
            to_loc = route[i]
            
            # Calculate energy needed for next leg
            distance = self.calculate_distance(from_loc, to_loc)
            energy_needed = self.calculate_energy_consumption(
                distance, current_load, vehicle_type)
            
            # Check if charging is needed
            if current_battery - energy_needed < self.ev_config.battery_safety_margin:
                # Find nearest charging station
                charging_station = self.find_nearest_charging_station(new_route[-1])
                new_route.append(charging_station)
                current_battery = self.ev_config.initial_charging
                
                # Recalculate energy needed from charging station
                distance = self.calculate_distance(charging_station, to_loc)
                energy_needed = self.calculate_energy_consumption(
                    distance, current_load, vehicle_type)
            
            new_route.append(to_loc)
            current_battery -= energy_needed
            
            # Update load after delivery (if it's a customer)
            if to_loc > 0:
                current_load -= self.instance.customer_items_weights[to_loc-1]
        
        return new_route

    def calculate_route_metrics(self, route: List[int], load: float,
                              vehicle_type: str) -> Tuple[float, float, float, List[Tuple[int, float]]]:
        """Calculate distance, energy consumption, delivery time and battery levels"""
        total_distance = 0
        total_energy = 0
        total_time = 0
        current_load = load
        current_battery = self.ev_config.initial_charging
        battery_levels = [(route[0], current_battery)]
        
        for i in range(len(route) - 1):
            from_loc = route[i]
            to_loc = route[i + 1]
            
            distance = self.calculate_distance(from_loc, to_loc)
            energy = self.calculate_energy_consumption(
                distance, current_load, vehicle_type)
            
            total_distance += distance
            total_energy += energy
            total_time += distance / self.ev_config.speed
            
            current_battery -= energy
            
            if from_loc < 0:  # At charging station
                charging_time = (self.ev_config.categories[vehicle_type]['battery_capacity'] / 
                               self.instance.charging_rate)
                total_time += charging_time
                current_battery = self.ev_config.initial_charging
            
            if to_loc > 0:  # Delivering to customer
                current_load -= self.instance.customer_items_weights[to_loc-1]
            
            battery_levels.append((to_loc, current_battery))
        
        return total_distance, total_energy, total_time, battery_levels

    def calculate_energy_consumption(self, distance: float, load: float, 
                                  vehicle_type: str) -> float:
        """Calculate energy consumption for a given distance and load"""
        vehicle_specs = self.ev_config.categories[vehicle_type]
        total_weight = vehicle_specs['base_weight'] + load
        return distance * (self.ev_config.energy_consumption_rate + 
                         (total_weight * self.ev_config.weight_factor/1000))

    def calculate_distance(self, from_location: int, to_location: int) -> float:
        """Calculate distance between two locations"""
        if from_location >= 0 and to_location >= 0:
            return self.instance.distance_matrix[from_location][to_location]
        elif from_location < 0:  # From charging station
            charging_station_index = -from_location - 1
            return self.instance.charging_distance_matrix[to_location][charging_station_index]
        else:  # To charging station
            charging_station_index = -to_location - 1
            return self.instance.charging_distance_matrix[from_location][charging_station_index]

    def find_nearest_charging_station(self, location: int) -> int:
        """Find the nearest charging station to a given location"""
        if location < 0:
            return location  # Already at a charging station
            
        distances = self.instance.charging_distance_matrix[location]
        nearest_index = np.argmin(distances)
        return -(nearest_index + 1)  # Convert to charging station index

# %%
def validate_input_data(instance):
    # Check if all required fields exist
    required_fields = ['instance_id', 'depot_location', 'customer_locations', 
                      'charging_stations', 'customer_items_weights', 'charging_rate']
    
    for field in required_fields:
        if field not in instance:
            raise ValueError(f"Missing required field: {field}")
    
    # Validate dimensions
    if len(instance['depot_location']) != 2:
        raise ValueError("Depot location must be a 2D point")
    
    if not all(len(loc) == 2 for loc in instance['customer_locations']):
        raise ValueError("All customer locations must be 2D points")
        
    if not all(len(loc) == 2 for loc in instance['charging_stations']):
        raise ValueError("All charging station locations must be 2D points")
        
    # Check if number of weights matches number of customers
    if len(instance['customer_items_weights']) != len(instance['customer_locations']):
        raise ValueError("Number of weights must match number of customers")
        
    # Validate numeric values
    if instance['charging_rate'] <= 0:
        raise ValueError("Charging rate must be positive")
        
    if any(w <= 0 for w in instance['customer_items_weights']):
        raise ValueError("All customer weights must be positive")

# %%
def read_toml_input(file_path):
    try:
        data = toml.load(file_path)
        
        # Extract required fields
        instance = {
            'instance_id': os.path.basename(file_path).split('.')[0],
            'depot_location': data['depot_location'],
            'customer_locations': data['customer_locations'],
            'charging_stations': data['charging_stations'],
            'customer_items_weights': data['customer_items_weights'],
            'charging_rate': data['charging_rate'],
            'vehicle_speed': data.get('vehicle_speed', 25),  # default if not specified
            'ev_parameters': data.get('ev_parameters', None)
        }
        
        # Validate data
        validate_input_data(instance)
        return instance
        
    except Exception as e:
        raise Exception(f"Error reading TOML file: {str(e)}")
    

def process_single_instance(toml_path: str) -> Dict[str, Any]:
    try:
        # Create instance and solve
        instance_data = read_toml_input(toml_path)
        instance = EVRPInstance(
            instance_id=instance_data['instance_id'],
            depot_location=instance_data['depot_location'],
            customer_locations=instance_data['customer_locations'],
            charging_stations=instance_data['charging_stations'],
            customer_items_weights=instance_data['customer_items_weights'],
            charging_rate=instance_data['charging_rate']
        )
        
        solver = GreedyEVRPSolver(instance)
        start_time = time.time()
        solution = solver.solve()
        computation_time = time.time() - start_time
        
        # Calculate metrics
        total_distance = sum(solution.route_distances)
        total_energy = sum(solution.route_energies)
        total_time = sum(solution.delivery_times)
        max_time = max(solution.delivery_times)
        print(solution.delivery_times)
        print(max_time)

        
        # Count vehicle types
        vehicle_counts = {}
        for v_type in solution.vehicle_types:
            vehicle_counts[v_type] = vehicle_counts.get(v_type, 0) + 1
            
        # Calculate number of charging stops
        charging_stops = sum(
            sum(1 for loc in route if loc < 0)
            for route in solution.routes
        )
        
        return {
            'instance_id': instance_data['instance_id'],
            'num_customers': len(instance_data['customer_locations']),
            'num_charging_stations': len(instance_data['charging_stations']),
            'total_delivery_weight': sum(instance_data['customer_items_weights']),
            'total_distance': total_distance,
            'total_energy': total_energy,
            'total_time': total_time,
            'max_time': max_time,
            'num_routes': len(solution.routes),
            'vehicle_distribution': vehicle_counts,
            'charging_stops': charging_stops,
            'computation_time': computation_time,
            'success': True,
            'error': None
        }
        
    except Exception as e:
        return {
            'instance_id': os.path.basename(toml_path).split('.')[0],
            'success': False,
            'error': str(e)
        }

def run_parallel_experiments(
    test_cases_dir: str,
    output_dir: str,
    max_workers: int = None
) -> None:
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all TOML files
    toml_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(test_cases_dir)
        for f in files if f.endswith('.toml')
    ]
    
    if not toml_files:
        raise ValueError(f"No TOML files found in {test_cases_dir}")
    
    # Initialize results storage
    results = []
    
    # Set up parallel processing
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_path = {
            executor.submit(process_single_instance, path): path
            for path in toml_files
        }
        
        # Process results as they complete
        for future in tqdm(as_completed(future_to_path), total=len(toml_files)):
            path = future_to_path[future]
            try:
                result = future.result()
                results.append(result)
                
                # Print progress
                instance_id = result.get('instance_id', 'unknown')
                if result['success']:
                    print(f"✓ {instance_id}: Completed successfully")
                else:
                    print(f"✗ {instance_id}: Failed - {result['error']}")
                    
            except Exception as e:
                print(f"Error processing {path}: {str(e)}")
    
    # Save detailed results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = os.path.join(output_dir, f'greedy_results_equal_proportional_loads_{timestamp}.json')
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create summary DataFrame
    summary_data = []
    for r in results:
        if r['success']:
            summary_data.append({
                'instance_id': r['instance_id'],
                'num_customers': r['num_customers'],
                'num_charging_stations': r['num_charging_stations'],
                'total_distance': r['total_distance'],
                'total_energy': r['total_energy'],
                'total_time': r['total_time'],
                'num_vehicles': r['num_routes'],
                'charging_stops': r['charging_stops'],
                'computation_time': r['computation_time'],
                'max_time': r['max_time']
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, f'greedy_summary_equal_proprtitonal_loads_{timestamp}.csv')
    summary_df.to_csv(summary_path, index=False)
    
    print("\nExperiment Results Summary:")
    print(f"Total instances processed: {len(results)}")
    print(f"Successful runs: {len(summary_data)}")
    print(f"Failed runs: {len(results) - len(summary_data)}")
    print(f"\nResults saved to:")
    print(f"Detailed results: {results_path}")
    print(f"Summary CSV: {summary_path}")

if __name__ == "__main__":
    # Example usage
    test_cases_dir = "../test_cases/"
    output_dir = "../results"
    
    # Run experiments using number of CPU cores - 1
    max_workers = max(1, os.cpu_count() - 1)
    run_parallel_experiments(test_cases_dir, output_dir, max_workers)

# %%


# %%



