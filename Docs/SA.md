# Simulated Annealing Electric Vehicle Routing Problem (SAEVRP) Algorithm

## Input Parameters
- `D`: Depot location coordinates (x, y)
- `C`: Set of customer locations with coordinates (x, y)
- `S`: Set of charging station locations with coordinates (x, y)
- `W`: Set of customer demands (weights)
- `R`: Charging rate (kWh/h)
- `T0`: Initial temperature
- `Tf`: Final temperature
- `α`: Cooling rate
- `L`: Number of iterations at each temperature
- Vehicle categories (small, medium, large, xlarge) with parameters from EVConfig

## Algorithm

### Phase 1: Initialization
```
Algorithm 1: Initialize_SAEVRP
Input: D, C, S, W, R, T0, Tf, α, L
Output: Initial solution and SA parameters

1. CREATE greedy_solver ← GreedyEVRPSolver(instance)
2. SET initial_solution ← greedy_solver.solve()
3. SET current_solution ← initial_solution
4. SET best_solution ← initial_solution
5. SET temperature ← T0
6. SET iteration_count ← 0
7. RETURN initialized state
```

### Phase 2: Main SA Solver
```
Algorithm 2: Solve_SAEVRP
Input: EVRP instance, SA parameters
Output: Best solution found

1. Initialize_SAEVRP()
2. WHILE temperature > Tf DO:
    3. FOR iteration in range(L):
        4. SET new_solution ← Generate_Neighbor(current_solution)
        5. IF Is_Feasible(new_solution) THEN:
            6. SET Δ ← Calculate_Delta(current_solution, new_solution)
            7. IF Δ < 0 OR random(0,1) < exp(-Δ/temperature) THEN:
                8. SET current_solution ← new_solution
                9. IF Cost(new_solution) < Cost(best_solution) THEN:
                    10. SET best_solution ← new_solution
    
    11. SET temperature ← temperature * α
    12. INCREMENT iteration_count

13. RETURN best_solution
```

### Phase 3: Neighbor Generation
```
Algorithm 3: Generate_Neighbor
Input: current_solution
Output: new_solution

1. SET new_solution ← Deep_Copy(current_solution)
2. SET operator ← Random_Select_Operator([
    'Intra_Route_Exchange',
    'Inter_Route_Exchange',
    'Route_Split',
    'Route_Merge',
    'Vehicle_Type_Change',
    'Charging_Station_Relocate'
])

3. SWITCH operator:
    CASE 'Intra_Route_Exchange':
        4. SELECT random route from new_solution
        5. SWAP two random customers within route
        6. UPDATE charging stations

    CASE 'Inter_Route_Exchange':
        7. SELECT two different random routes
        8. SWAP customers between routes
        9. UPDATE charging stations and vehicle loads

    CASE 'Route_Split':
        10. SELECT longest route
        11. SPLIT into two routes
        12. ASSIGN appropriate vehicle types
        13. UPDATE charging stations

    CASE 'Route_Merge':
        14. SELECT two compatible routes
        15. MERGE if capacity constraints allow
        16. UPDATE charging stations

    CASE 'Vehicle_Type_Change':
        17. SELECT random route
        18. CHANGE vehicle type
        19. VERIFY capacity constraints
        20. UPDATE charging stations

    CASE 'Charging_Station_Relocate':
        21. SELECT random route
        22. RELOCATE charging station to alternative
        23. UPDATE energy calculations

24. RETURN new_solution
```

### Phase 4: Solution Evaluation
```
Algorithm 4: Calculate_Delta
Input: current_solution, new_solution
Output: cost difference

1. SET current_cost ← Calculate_Total_Cost(current_solution)
2. SET new_cost ← Calculate_Total_Cost(new_solution)
3. RETURN new_cost - current_cost

Function: Calculate_Total_Cost(solution)
1. SET total_cost ← 0
2. SET distance_weight ← w1
3. SET energy_weight ← w2
4. SET vehicle_weight ← w3

5. FOR each route in solution:
    6. ADD distance_weight * route.total_distance to total_cost
    7. ADD energy_weight * route.total_energy to total_cost
    8. ADD vehicle_weight * vehicle_count to total_cost

9. RETURN total_cost
```

### Phase 5: Feasibility Check
```
Algorithm 5: Is_Feasible
Input: solution
Output: boolean

1. FOR each route in solution:
    2. IF route.total_load > vehicle_capacity THEN:
        3. RETURN false
    
    4. IF route.total_energy > battery_capacity THEN:
        5. RETURN false
    
    6. FOR each segment in route:
        7. IF segment.battery_level < safety_margin THEN:
            8. RETURN false

9. IF any customer not served exactly once THEN:
    10. RETURN false

11. RETURN true
```

## Complexity Analysis
- Time Complexity: O(L * N * M) where:
  - L: Total number of iterations (L * number of temperature steps)
  - N: Number of customers
  - M: Number of charging stations
- Space Complexity: O(N²) for distance matrices and solution storage

## Solution Components
1. Total Cost Function:
   ```
   Total_Cost = w1 * total_distance + 
                w2 * total_energy_consumption +
                w3 * number_of_vehicles
   ```

2. Cooling Schedule:
   ```
   T = T0 * α^k
   where k is the iteration number
   ```

3. Acceptance Probability:
   ```
   P(Δ,T) = exp(-Δ/T)
   where Δ is the cost difference
   ```

## Output
- Best solution found containing:
  - Optimized routes
  - Vehicle assignments
  - Charging station placements
  - Total cost metrics
  - Energy consumption
  - Computation statistics