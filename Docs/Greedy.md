# Electric Vehicle Routing Problem (EVRP) Algorithm

## Input Parameters
- `D`: Depot location coordinates (x, y)
- `C`: Set of customer locations with coordinates (x, y)
- `S`: Set of charging station locations with coordinates (x, y)
- `W`: Set of customer demands (weights)
- `R`: Charging rate (kWh/h)
- Vehicle categories (small, medium, large, xlarge) with:
  - Battery capacity (kWh)
  - Base weight (kg)
  - Load capacity (kg)

## Algorithm

### Phase 1: Initialization
```
Algorithm 1: Initialize_EVRP
Input: D, C, S, W, R
Output: Instance configuration and initial fleet

1. CREATE distance matrix DM[i,j] for all locations using Euclidean distance
2. CREATE charging distance matrix CM[i,j] between locations and charging stations
3. SET initial_charging ← 100%
4. SET battery_safety_margin ← 40%
5. SET total_demand ← SUM(W)
6. SET fleet ← Calculate_Minimum_Fleet(total_demand)
   WHERE fleet = {
       'small': min_vehicles,
       'medium': min_vehicles,
       'large': min_vehicles,
       'xlarge': min_vehicles
   }
7. SET served_customers ← ∅
8. RETURN initialized instance
```

### Phase 2: Main Solver
```
Algorithm 2: Solve_EVRP
Input: EVRP instance
Output: Solution with routes and metrics

1. SET max_attempts ← 16
2. SET attempt ← 0
3. SET best_solution ← ∅

4. WHILE attempt < max_attempts DO:
    5. CLEAR served_customers
    6. SET solution ← new EVRPSolution()
    
    7. CALCULATE proportional_loads ← Calculate_Load_Distribution(fleet, total_demand)
    
    8. FOR each vehicle_type in ['small', 'medium', 'large', 'xlarge']:
        9. SET num_vehicles ← fleet[vehicle_type]
        
        10. FOR v in range(num_vehicles):
            11. IF |served_customers| = |C| THEN:
                12. RETURN solution
            
            13. SET unserved ← C - served_customers
            14. SET route, load ← Create_Route(unserved, vehicle_type, proportional_loads[vehicle_type])
            
            15. IF route.length > 2 THEN:
                16. SET route_with_charging ← Insert_Charging_Stations(route, load, vehicle_type)
                17. CALCULATE route_metrics ← Calculate_Route_Metrics(route_with_charging)
                18. ADD route_with_charging to solution
                19. UPDATE solution metrics
    
    20. IF |served_customers| = |C| THEN:
        21. RETURN solution
    
    22. SET fleet ← Increase_Fleet_Binary(fleet)
    23. INCREMENT attempt
    
24. RETURN empty solution
```

### Phase 3: Route Creation
```
Algorithm 3: Create_Route
Input: unserved_customers, vehicle_type, target_load
Output: route, current_load

1. SET route ← [depot]
2. SET current_load ← 0
3. SET current_battery ← initial_charging
4. SET tolerance ← 0.10

5. WHILE unserved_customers not empty DO:
    6. SET next_customer ← Find_Best_Next_Customer(route[-1], unserved_customers)
    
    7. IF next_customer is null THEN:
        8. BREAK
    
    9. SET customer_demand ← W[next_customer]
    10. SET is_last_customer ← |unserved_customers| = 1
    
    11. IF exceeds_capacity(current_load + customer_demand, vehicle_type) THEN:
        12. BREAK
    
    13. IF not is_last_customer AND exceeds_target_load(current_load + customer_demand, target_load, tolerance) THEN:
        14. BREAK
    
    15. APPEND next_customer to route
    16. INCREMENT current_load by customer_demand
    17. ADD next_customer to served_customers
    
18. APPEND depot to route
19. RETURN route, current_load
```

### Phase 4: Charging Station Insertion
```
Algorithm 4: Insert_Charging_Stations
Input: route, load, vehicle_type
Output: route_with_charging

1. IF route.length ≤ 2 THEN:
    2. RETURN route

3. SET new_route ← [depot]
4. SET current_battery ← initial_charging
5. SET current_load ← load

6. FOR i in range(1, route.length):
    7. SET from_loc ← route[i-1]
    8. SET to_loc ← route[i]
    
    9. CALCULATE energy_needed ← Calculate_Energy(from_loc, to_loc, current_load, vehicle_type)
    
    10. IF current_battery - energy_needed < battery_safety_margin THEN:
        11. SET charging_station ← Find_Nearest_Charging_Station(new_route[-1])
        12. APPEND charging_station to new_route
        13. SET current_battery ← initial_charging
        14. RECALCULATE energy_needed
    
    15. APPEND to_loc to new_route
    16. DECREMENT current_battery by energy_needed
    
    17. IF to_loc is customer THEN:
        18. DECREMENT current_load by W[to_loc]

19. RETURN new_route
```

### Phase 5: Fleet Management
```
Algorithm 5: Increase_Fleet_Binary
Input: current_fleet
Output: new_fleet

1. SET base ← MIN(current_fleet.values())
2. SET binary ← Convert_Fleet_To_Binary(current_fleet, base)
3. SET current_value ← BINARY_TO_INT(binary)
4. SET next_value ← current_value + 1

5. IF next_value > 15 THEN:
    6. INCREMENT base
    7. SET next_value ← 0

8. SET new_binary ← INT_TO_BINARY(next_value, 4)
9. SET new_fleet ← Create_Fleet_From_Binary(new_binary, base)
10. RETURN new_fleet
```

## Complexity Analysis
- Time Complexity: O(N²) for N customers in the basic implementation
- Space Complexity: O(N²) for distance matrices
- Additional complexity factors:
  - Number of vehicle types
  - Number of charging stations
  - Maximum attempts for fleet configurations

## Output
- Set of routes for each vehicle
- Vehicle type assignments
- Load distribution
- Route metrics:
  - Total distance
  - Energy consumption
  - Delivery times
  - Battery levels
- Visualization of solution