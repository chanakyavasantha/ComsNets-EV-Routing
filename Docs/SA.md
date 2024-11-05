# Simulated Annealing Algorithm for EVRP with Load-Based Neighbor Generation

## Core Neighbor Generation Strategy
```
Generate_Neighbor(current_solution):
    1. Identify Over-Occupied Routes:
       - For each route in current solution:
           - Calculate actual load vs target load
           - Sort routes by occupation ratio (actual/target)
           - Select most over-occupied route
    
    2. Select Customer to Reassign:
       - From over-occupied route:
           - Get list of customers and their demands
           - Randomly select a customer
           - Consider position in route for time impact
    
    3. Find Suitable Target Vehicle:
       - For each vehicle with available capacity:
           - Calculate remaining capacity
           - Check if customer demand fits
           - Check if adding customer maintains feasible delivery time
           - Create candidate list of feasible vehicles
    
    4. Reassign Customer:
       - Select best target vehicle from candidates
       - Remove customer from over-occupied route
       - Insert customer in best position in target route
       - Update route metrics
    
    return modified_solution
```

## Main SA Algorithm Structure
```
Simulated_Annealing(instance, greedy_solution):
    // Initialize
    current_solution = remove_charging_stations(greedy_solution)
    best_solution = current_solution
    T = calculate_initial_temperature()
    
    while T > T_final:
        // Get over-occupied routes
        over_occupied = find_over_occupied_routes(current_solution)
        
        if not over_occupied:
            // If no over-occupied routes, try to optimize delivery time
            neighbor = optimize_delivery_time(current_solution)
        else:
            // Generate neighbor by redistributing from over-occupied route
            neighbor = Generate_Neighbor(current_solution)
        
        // Calculate maximum delivery times
        current_max_time = calculate_max_delivery_time(current_solution)
        neighbor_max_time = calculate_max_delivery_time(neighbor)
        
        delta = neighbor_max_time - current_max_time
        
        // Accept or reject
        if delta < 0 or random(0,1) < exp(-delta/T):
            current_solution = neighbor
            if neighbor_max_time < best_max_time:
                best_solution = neighbor
                best_max_time = neighbor_max_time
        
        // Cool down and track metrics
        T = T * cooling_rate
        record_iteration_metrics()
    
    return insert_charging_stations(best_solution)
```

## Helper Functions

```python
def find_over_occupied_routes(solution):
    over_occupied = []
    for route_idx, route in enumerate(solution.routes):
        vehicle_type = solution.vehicle_types[route_idx]
        current_load = sum(customer_demands[c-1] for c in route if c > 0)
        capacity = vehicle_capacities[vehicle_type]
        
        # Calculate occupation ratio
        occupation_ratio = current_load / capacity
        if occupation_ratio > TARGET_OCCUPATION:
            over_occupied.append({
                'route_idx': route_idx,
                'ratio': occupation_ratio,
                'excess': current_load - (TARGET_OCCUPATION * capacity)
            })
    
    return sorted(over_occupied, key=lambda x: x['ratio'], reverse=True)

def find_best_insertion_position(route, customer, vehicle_type):
    best_position = None
    best_time = float('inf')
    
    for i in range(1, len(route)):
        # Try inserting customer at position i
        test_route = route[:i] + [customer] + route[i:]
        route_time = calculate_route_time(test_route, vehicle_type)
        
        if route_time < best_time:
            best_time = route_time
            best_position = i
    
    return best_position, best_time

def calculate_load_distribution(solution):
    distribution = {}
    for route_idx, route in enumerate(solution.routes):
        vehicle_type = solution.vehicle_types[route_idx]
        load = sum(customer_demands[c-1] for c in route if c > 0)
        capacity = vehicle_capacities[vehicle_type]
        distribution[route_idx] = {
            'load': load,
            'capacity': capacity,
            'utilization': load/capacity
        }
    return distribution
```

## Metrics and Analysis

1. **Load Distribution Tracking**
```python
def track_load_metrics():
    metrics = {
        'max_utilization': max(route['utilization'] for route in load_distribution.values()),
        'avg_utilization': sum(route['utilization'] for route in load_distribution.values()) / len(load_distribution),
        'std_utilization': calculate_std([route['utilization'] for route in load_distribution.values()]),
        'over_occupied_count': len([r for r in load_distribution.values() if r['utilization'] > TARGET_OCCUPATION])
    }
    return metrics
```

2. **Visualization Plots**
```
Generate_Analysis_Plots():
    1. Load Distribution Plot:
       - X-axis: Route number
       - Y-axis: Load utilization
       - Show target occupation line
       - Compare SA vs Greedy
    
    2. Convergence Plot:
       - X-axis: Iteration
       - Y-axis: 
           - Maximum delivery time
           - Number of over-occupied routes
           - Maximum utilization
    
    3. Route Changes Plot:
       - Track customer reassignments
       - Show impact on delivery times
```

## Implementation Notes

1. **Target Occupation**
   - Set target route occupation (e.g., 85% of vehicle capacity)
   - Use as threshold for identifying over-occupied routes

2. **Customer Selection**
   - Prioritize customers with larger demands from over-occupied routes
   - Consider impact on route timing when selecting

3. **Vehicle Selection**
   - Consider both current load and potential delivery time impact
   - Maintain vehicle type constraints from greedy solution

4. **Acceptance Criteria**
   - Primary: Reduction in over-occupation
   - Secondary: Improvement in maximum delivery time
   - Accept worse solutions based on temperature cooling

5. **Tracking**
   - Monitor load distribution changes
   - Track delivery time improvements
   - Record customer reassignments