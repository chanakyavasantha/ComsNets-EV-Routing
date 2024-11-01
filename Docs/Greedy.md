# Greedy Algorithm for Electric Vehicle Routing Problem (EVRP)

## Algorithm Overview

## Section 0: EVRP Input & Solution Format
plaintext
Algorithm: GreedyEVRP
Input: 
- depot_location: coordinates (x,y)
- customer_locations: list of (x,y) coordinates
- customer item weights: list of weights that needs to be delivered to the customers
- charging_stations: list of (x,y) coordinates
- vehicle_types: specifications of different EV types

Output: 
- Solution containing routes, vehicle assignments, and charging stops

```
1. Solution Class Structure
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

2. Example Solution Format
# Example solution for 5 customers and 3 charging stations
solution = {
    "routes": [
        {
            "sequence": [0, 2, -1, 4, 0],     # 0=depot, 2,4=customers, -1=charging_station_1
            "vehicle_type": "small",
            "load": 110.0,                    # kg
            "distance": 85.3,                 # km
            "energy": 15.2,                   # kWh
            "delivery_time": 4.1              # hours
        },
        {
            "sequence": [0, 1, -2, 3, 5, 0],
            "vehicle_type": "medium",
            "load": 185.0,
            "distance": 102.7,
            "energy": 18.9,
            "delivery_time": 5.3
        }
    ],
    "summary": {
        "total_distance": 188.0,              # km
        "total_energy": 34.1,                 # kWh
        "total_time": 9.4,                    # hours
        "computation_time": 1.23              # seconds
    }
}

3. Location Indexing Convention

Depot: Always indexed as 0
Customers: Positive integers starting from 1

Customer 1 → index 1
Customer 2 → index 2
etc.


Charging Stations: Negative integers

Charging Station 1 → index -1
Charging Station 2 → index -2
etc.



Example:
pythonCopy# Sample route interpretation
route = [0, 2, -1, 4, 0]
# Means: Depot → Customer2 → ChargingStation1 → Customer4 → Depot

```
## Section 1: Main Algorithm Structure

```plaintext
PROCEDURE ModifiedGreedyEVRP_Solve:
    1. Calculate Initial Fleet Size
        total_demand ← Sum of all customer demands
        unit_fleet_capacity ← (700 + 600 + 500)  // Sum of one of each vehicle type
        min_vehicles_needed ← Ceiling(total_demand / (0.25 * unit_fleet_capacity))
        
    2. Initialize Fleet Management
        fleet ← Initialize_Fleet(min_vehicles_needed)  // Starts with min_vehicles_needed of each type
        
    3. Initialize Solution
        unserved_customers ← all_customers
        solution ← empty_solution
        available_vehicles ← Generate_Vehicle_Sequence(fleet)
        // Available vehicles will be in order: [large, large, ..., medium, medium, ..., small, small, ...]
        
    4. Main Routing Loop
        current_vehicle_index ← 0
        
        WHILE unserved_customers is not empty:
            IF current_vehicle_index >= len(available_vehicles):
                Increase_Fleet_Size()
                Regenerate_Vehicle_Sequence()
                current_vehicle_index ← 0
                
            vehicle_type ← available_vehicles[current_vehicle_index]
            route ← Create_New_Route(vehicle_type)
            Add route to solution
            current_vehicle_index += 1
            
    Return solution
```

### Example:

```
Initial State:
- Customers: [1, 2, 3, 4, 5]
- Demands: [90, 50, 65, 60, 95]
- Total Demand: 360kg
- Unit Fleet Capacity (1 of each): 1800kg
- Min Vehicles Calculation:
  360 ≤ 0.25 * n * 1800
  360 ≤ 450n
  n ≥ 0.8
  min_vehicles_needed = 1 (ceiling)

Initial Fleet (1 of each type):
- 1 large (700kg)
- 1 medium (600kg)
- 1 small (500kg)

Vehicle Assignment Sequence:
1. First Route: Large Vehicle (700kg)
   - Serves: [2, 4, 3] (175kg total)
   
2. Second Route: Medium Vehicle (600kg)
   - Serves: [1] (90kg)
   
3. Third Route: Small Vehicle (500kg)
   - Serves: [5] (95kg)

All customers served with initial fleet ✓
```


## Section 2. Fleet Management
```
PROCEDURE Calculate_Min_Vehicles_Needed(total_demand):
    unit_fleet_capacity ← 700 + 600 + 500  // Sum of one of each vehicle type
    min_vehicles ← Ceiling(total_demand / (0.25 * unit_fleet_capacity))
    Return min_vehicles

PROCEDURE Initialize_Fleet(min_vehicles_needed):
    fleet ← {
        'large': min_vehicles_needed,
        'medium': min_vehicles_needed,
        'small': min_vehicles_needed
    }
    Return fleet

PROCEDURE Generate_Vehicle_Sequence(fleet):
    sequence ← []
    FOR vehicle_type in ['large', 'medium', 'small']:
        FOR i in range(fleet[vehicle_type]):
            Add vehicle_type to sequence
    Return sequence

PROCEDURE Increase_Fleet_Size:
    FOR each vehicle_type in fleet:
        fleet[vehicle_type] += 1


```

## Section 3: Route Creation

```plaintext
PROCEDURE Create_New_Route:
    route ← [depot]
    current_load ← 0
    current_battery ← 100%
    vehicle_type ← 'xlarge'  // Start with largest vehicle but will optimize later
    
    WHILE can_add_more_customers:
        next_customer ← Find_Best_Next_Customer(
            current_location,
            unserved_customers,
            current_load,
            current_battery
        )
        
        IF next_customer is None:
            BREAK
            
        distance_to_customer ← Calculate_Distance(current_location, next_customer)
        distance_to_depot ← Calculate_Distance(current_location, depot)
        
        IF distance_to_customer > distance_to_depot:
            BREAK  // End route here and start new one
            
        Add next_customer to route
        Update current_load
        Update current_battery
        Remove next_customer from unserved_customers
        
    Add depot to route
    Try_Smaller_Vehicle(route)  // Try to optimize down from xlarge
    Insert_Charging_Stations(route)
    
    Return route
```

### Example:
```plaintext
Creating Route 1:
Current Location: depot (60, 60)
1. Start: [0]
2. Find nearest: Customer 2 at (46, 58), demand=50kg
   Route: [0 → 2]
3. Find next nearest: Customer 4 at (23, 29), demand=60kg
   Route: [0 → 2 → 4]
4. Return to depot: [0 → 2 → 4 → 0]
5. Optimize vehicle: medium vehicle (total load = 110kg)
6. Add charging: [0 → 2 → CS1 → 4 → 0]
```

## Section 4. Customer Selection

```plaintext
PROCEDURE Find_Best_Next_Customer:
    best_customer ← None
    min_distance ← infinity
    
    FOR each customer in unserved_customers:
        // Calculate distances
        distance_to_customer ← Calculate_Distance(current_location, customer)
        distance_to_depot ← Calculate_Distance(current_location, depot)
        
        // Skip if customer is farther than depot
        IF distance_to_customer > distance_to_depot:
            CONTINUE
            
        IF Is_Feasible(customer, current_load, current_battery):
            IF distance_to_customer < min_distance:
                min_distance ← distance_to_customer
                best_customer ← customer
                
    Return best_customer

Example:
Current Location: (60, 60)
Unserved Customers:
1. Customer A: 
   - Location: (44, 93)
   - Distance: 35.4km
   - Distance to depot: 40.2km
   - Status: Consider (35.4 < 40.2) ✓

2. Customer B:
   - Location: (46, 58)
   - Distance: 14.2km
   - Distance to depot: 40.2km
   - Status: Consider (14.2 < 40.2) ✓ 
   - Current best (shortest distance) ✓

3. Customer C:
   - Location: (95, 70)
   - Distance: 45.3km
   - Distance to depot: 40.2km
   - Status: Skip (45.3 > 40.2) ✗

Select: Customer B (shortest feasible distance)
```

### Example:
```plaintext
Current Location: (60, 60)
Unserved Customers:
- Customer 1: (44, 93), demand=90kg, distance=35.4
- Customer 2: (46, 58), demand=50kg, distance=14.2
- Customer 3: (87, 43), demand=65kg, distance=32.1

Selection Process:
1. Check Customer 1: feasible, distance=35.4
2. Check Customer 2: feasible, distance=14.2 ← Best so far
3. Check Customer 3: feasible, distance=32.1

Select: Customer 2 (shortest feasible distance)
```

## Section 5. Feasibility Check

```plaintext
PROCEDURE Is_Feasible(customer, current_load, current_battery):
    // 1. Basic Load Check
    new_load ← current_load + customer_demand
    IF new_load > vehicle_specs.capacity:
        Return False
        
    // 2. Battery Check for Direct Travel
    distance_to_customer ← Calculate_Distance(current_location, customer)
    energy_to_customer ← Calculate_Energy_Consumption(
        distance_to_customer,
        new_load
    )
    
    // 3. Check Return to Depot Possibility
    distance_to_depot ← Calculate_Distance(customer, depot)
    energy_to_depot ← Calculate_Energy_Consumption(
        distance_to_depot,
        new_load - customer_demand  // Load after delivery
    )
    
    total_energy_needed ← energy_to_customer + energy_to_depot
    remaining_battery ← current_battery - energy_to_customer
    
    // 4. Check if Direct Route is Possible
    IF remaining_battery >= energy_to_depot + safety_margin:
        Return True
        
    // 5. Check if Route with Charging is Possible
    nearest_cs ← Find_Nearest_Charging_Station(customer)
    IF nearest_cs exists:
        energy_to_cs ← Calculate_Energy_Consumption(
            current_location,
            nearest_cs,
            current_load
        )
        energy_cs_to_customer ← Calculate_Energy_Consumption(
            nearest_cs,
            customer,
            current_load
        )
        energy_customer_to_depot ← Calculate_Energy_Consumption(
            customer,
            depot,
            current_load - customer_demand
        )
        
        // Check if can reach CS and then complete route
        IF current_battery >= energy_to_cs + safety_margin AND
           battery_capacity >= energy_cs_to_customer + 
                             energy_customer_to_depot + 
                             safety_margin:
            Return True
            
    Return False

Example:
Check Customer 5:
1. Load Check:
   - Current load: 200kg
   - Customer demand: 95kg
   - New total: 295kg < 800kg ✓
   
2. Direct Battery Check:
   - Current battery: 70%
   - To customer: 30%
   - To depot: 45%
   - Remaining after customer: 40%
   - Needed for depot: 45%
   - Direct route not possible ✗
   
3. Charging Station Check:
   - Nearest CS distance: 15km
   - Energy to CS: 20%
   - Energy CS to customer: 25%
   - Energy customer to depot: 45%
   - Can reach CS: Yes (70% > 20% + safety_margin) ✓
   - Can complete route after charging: Yes ✓
   
Result: Feasible (with charging) ✓
```

### Example:
```plaintext
Check Customer 5:
1. Capacity Check:
   - Current load: 50kg
   - Customer demand: 95kg
   - Total: 145kg < 800kg (XLARGE) ✓
   
2. Energy Check:
   - Current battery: 70%
   - Energy to customer: 30%
   - Energy to depot: 45%
   - Total needed: 75% > 70% ✗
   
Result: Not Feasible (battery constraint)
```

## Section 5. Charging Station Insertion

```plaintext
PROCEDURE Insert_Charging_Stations:
    new_route ← [depot]
    battery ← 100%
    safety_margin ← 20%
    
    FOR each customer in route:
        energy_needed ← Calculate_Energy_To_Next(
            current_location,
            next_location,
            current_load
        )
        
        IF battery - energy_needed < safety_margin:
            cs ← Find_Nearest_Charging_Station()
            Add cs to new_route
            battery ← 100%
            
        Add customer to new_route
        battery ← battery - energy_needed
        
    Return new_route
```

### Example:
```plaintext
Original Route: [0 → 2 → 4 → 0]

Energy Analysis:
1. Depot → 2: 20% used (80% remaining)
2. 2 → 4: 40% used (40% remaining)
3. 4 → Depot: 45% needed (Not enough!)

Insert Charging:
- After Customer 2, before Customer 4
Final Route: [0 → 2 → CS1 → 4 → 0]
```




### Example:
```plaintext
Route: [0 → 2 → CS1 → 4 → 0]
Total Load: 110kg

Vehicle Options:
1. small: 500kg ✓ (First sufficient capacity)
2. medium: 600kg
3. large: 700kg
4. xlarge: 800kg

Select: small vehicle type
```

## Implementation Notes

1. **Distance Calculation**:
```python
def calculate_distance(point1, point2):
    return sqrt((point2[0] - point1[0])**2 + 
                (point2[1] - point1[1])**2)
```

2. **Energy Consumption**:
```python
def calculate_energy(distance, load):
    base_rate = 0.15  # kWh/km
    weight_factor = 0.05  # per 1000kg
    return distance * (base_rate + load * weight_factor/1000)
```

3. **Battery Management**:
```python
def is_battery_sufficient(route, battery_level):
    required_energy = sum(calculate_energy(leg) 
                         for leg in route_legs)
    return battery_level >= required_energy + safety_margin
```



