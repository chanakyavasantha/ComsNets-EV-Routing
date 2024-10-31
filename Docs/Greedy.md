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
class Solution:
    def __init__(self):
        self.routes = []              # List of Route objects
        self.total_distance = 0.0     # Total distance traveled by all vehicles
        self.total_energy = 0.0       # Total energy consumed by all vehicles
        self.total_time = 0.0         # Total delivery time
        self.computation_time = 0.0   # Time taken to compute solution
class Route:
    def __init__(self):
        self.sequence = []        # List of locations (depot=0, customer>0, charging_station<0)
        self.vehicle_type = ""    # Type of EV assigned (small/medium/large/xlarge)
        self.load = 0.0          # Total load for this route
        self.distance = 0.0      # Total distance of route
        self.energy = 0.0        # Total energy consumption
        self.delivery_time = 0.0 # Total time including travel and charging


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
PROCEDURE GreedyEVRP_Solve:
    unserved_customers ← all_customers
    solution ← empty_solution
    
    WHILE unserved_customers is not empty:
        route ← Create_New_Route()
        Add route to solution
        
    Return solution
```

### Example:
```plaintext
Initial State:
- Customers: [1, 2, 3, 4, 5]
- Demands: [90, 50, 65, 60, 95]

Iteration 1:
- Create Route 1: [0 → 2 → 4 → 0] (demands 50 + 60 = 110kg)
Remaining: [1, 3, 5]

Iteration 2:
- Create Route 2: [0 → 3 → 0] (demand 65kg)
Remaining: [1, 5]

Iteration 3:
- Create Route 3: [0 → 1 → 5 → 0] (demands 90 + 95 = 185kg)
Remaining: []
```

## Section 2: Route Creation

```plaintext
PROCEDURE Create_New_Route:
    route ← [depot]
    current_load ← 0
    vehicle_type ← 'xlarge'  // Start with largest vehicle
    
    WHILE can_add_more_customers:
        next_customer ← Find_Best_Next_Customer(
            current_location,
            unserved_customers,
            current_load
        )
        
        IF next_customer is None:
            BREAK
            
        Add next_customer to route
        Update current_load
        Remove next_customer from unserved_customers
        
    Add depot to route
    Optimize_Vehicle_Type(route)
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

## Section 3. Customer Selection

```plaintext
PROCEDURE Find_Best_Next_Customer:
    best_customer ← None
    min_distance ← infinity
    
    FOR each customer in unserved_customers:
        IF Is_Feasible(customer, current_load):
            distance ← Calculate_Distance(current_location, customer)
            IF distance < min_distance:
                min_distance ← distance
                best_customer ← customer
                
    Return best_customer
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

## Section 4. Feasibility Check

```plaintext
PROCEDURE Is_Feasible:
    // Check vehicle capacity
    IF current_load + customer_demand > max_vehicle_capacity:
        Return False
        
    // Check battery range
    energy_to_customer ← Calculate_Energy(current_location, customer)
    energy_to_depot ← Calculate_Energy(customer, depot)
    total_energy ← energy_to_customer + energy_to_depot
    
    IF total_energy > battery_capacity:
        Return False
        
    Return True
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

## Section 6. Vehicle Type Optimization

```plaintext
PROCEDURE Optimize_Vehicle_Type:
    route_load ← Calculate_Total_Load(route)
    
    FOR vehicle_type in sorted_by_capacity:
        IF vehicle_type.capacity >= route_load:
            Return vehicle_type
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