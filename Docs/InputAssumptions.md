#### Key Notes:
- [x] 0:Depot,
- [x] Number of Customer = self.number_of_customers (without including depot),
- [x] Charging Stations are represented with negative indexing,

### EV Categories and Energy Calculations:

1. Small EV (35 kWh):
```
Base Weight: 1500 kg
Load Capacity: 500 kg
Battery: 35 kWh

Energy Consumption:
- Empty: 0.15 + (1500/1000)*0.05 = 0.225 kWh/km
- Full Load: 0.15 + (2000/1000)*0.05 = 0.25 kWh/km

Range at Full Load:
- Max Range = 0.8 * 35 kWh / 0.25 kWh/km = 112 km
- Min Range = 0.2 * 35 kWh / 0.25 kWh/km = 28 km
```

2. Medium EV (40 kWh):
```
Base Weight: 1800 kg
Load Capacity: 600 kg
Battery: 40 kWh

Energy Consumption:
- Empty: 0.15 + (1800/1000)*0.05 = 0.24 kWh/km
- Full Load: 0.15 + (2400/1000)*0.05 = 0.27 kWh/km

Range at Full Load:
- Max Range = 0.8 * 40 kWh / 0.27 kWh/km = 118.5 km
- Min Range = 0.2 * 40 kWh / 0.27 kWh/km = 29.6 km
```

3. Large EV (45 kWh):
```
Base Weight: 2000 kg
Load Capacity: 700 kg
Battery: 45 kWh

Energy Consumption:
- Empty: 0.15 + (2000/1000)*0.05 = 0.25 kWh/km
- Full Load: 0.15 + (2700/1000)*0.05 = 0.285 kWh/km

Range at Full Load:
- Max Range = 0.8 * 45 kWh / 0.285 kWh/km = 126.3 km
- Min Range = 0.2 * 45 kWh / 0.285 kWh/km = 31.6 km
```

4. Extra Large EV (50 kWh):
```
Base Weight: 2200 kg
Load Capacity: 800 kg
Battery: 50 kWh

Energy Consumption:
- Empty: 0.15 + (2200/1000)*0.05 = 0.26 kWh/km
- Full Load: 0.15 + (3000/1000)*0.05 = 0.30 kWh/km

Range at Full Load:
- Max Range = 0.8 * 50 kWh / 0.30 kWh/km = 133.3 km
- Min Range = 0.2 * 50 kWh / 0.30 kWh/km = 33.3 km
```


# EV Energy Efficiency Per kg-km

## Energy Consumption Calculation:
For each vehicle: Energy = base_rate + (total_weight * weight_factor)
where:
- base_rate = 0.15 kWh/km
- weight_factor = 0.05 per 1000kg

## Per Vehicle Category Analysis:

1. Small EV:
```
Base Energy (empty) = 0.15 + (1500/1000)*0.05 = 0.225 kWh/km
Full Load Energy = 0.15 + (2000/1000)*0.05 = 0.25 kWh/km
Energy for payload (500kg) = 0.25 - 0.225 = 0.025 kWh/km
Energy per kg-km = 0.025/500 = 0.00005 kWh/kg/km
```

2. Medium EV:
```
Base Energy (empty) = 0.15 + (1800/1000)*0.05 = 0.24 kWh/km
Full Load Energy = 0.15 + (2400/1000)*0.05 = 0.27 kWh/km
Energy for payload (600kg) = 0.27 - 0.24 = 0.03 kWh/km
Energy per kg-km = 0.03/600 = 0.00005 kWh/kg/km
```

3. Large EV:
```
Base Energy (empty) = 0.15 + (2000/1000)*0.05 = 0.25 kWh/km
Full Load Energy = 0.15 + (2700/1000)*0.05 = 0.285 kWh/km
Energy for payload (700kg) = 0.285 - 0.25 = 0.035 kWh/km
Energy per kg-km = 0.035/700 = 0.00005 kWh/kg/km
```

4. X-Large EV:
```
Base Energy (empty) = 0.15 + (2200/1000)*0.05 = 0.26 kWh/km
Full Load Energy = 0.15 + (3000/1000)*0.05 = 0.30 kWh/km
Energy for payload (800kg) = 0.30 - 0.26 = 0.04 kWh/km
Energy per kg-km = 0.04/800 = 0.00005 kWh/kg/km
```

# Enhanced EV Summary Table

| Category | Battery | Base Weight | Max Load | Full Load Range (Min-Max) | Energy Consumption (Empty-Full) | Energy per kg-km | Empty km/kWh | Full Load km/kWh |
|----------|---------|-------------|-----------|--------------------------|--------------------------------|------------------|--------------|-----------------|
| Small    | 35 kWh  | 1500 kg    | 500 kg   | 28 - 112 km             | 0.225 - 0.25 kWh/km           | 0.00005 kWh/kg/km| 4.44 km/kWh  | 4.00 km/kWh     |
| Medium   | 40 kWh  | 1800 kg    | 600 kg   | 29.6 - 118.5 km         | 0.24 - 0.27 kWh/km            | 0.00005 kWh/kg/km| 4.17 km/kWh  | 3.70 km/kWh     |
| Large    | 45 kWh  | 2000 kg    | 700 kg   | 31.6 - 126.3 km         | 0.25 - 0.285 kWh/km           | 0.00005 kWh/kg/km| 4.00 km/kWh  | 3.51 km/kWh     |
| X-Large  | 50 kWh  | 2200 kg    | 800 kg   | 33.3 - 133.3 km         | 0.26 - 0.30 kWh/km            | 0.00005 kWh/kg/km| 3.85 km/kWh  | 3.33 km/kWh     |

## Key Finding:
Interestingly, all vehicle categories have the same energy efficiency per kg-km (0.00005 kWh/kg/km). This is because the weight factor (0.05 per 1000kg) is linear and proportional to the vehicle's capacity.

## Implications for Vehicle Selection:
1. Since energy efficiency per kg-km is constant across all vehicles, the primary selection criteria should be:
   - Minimizing empty runs (base energy consumption)
   - Maximizing capacity utilization
   - Route length constraints (battery capacity)

2. Vehicle selection strategy should focus on:
   - Using smaller vehicles for smaller loads to minimize base energy consumption
   - Using larger vehicles only when load consolidation can maximize capacity utilization
   - Matching vehicle battery capacity to route length requirements






###  Customer Item Weights:
```
Range: 50-100 kg (Multiple of 5)
- Minimum Package Weight: 50 kg
- Maximum Package Weight: 100 kg
```

### Vehicle Speed:
```
Constant Operating Speed: 25 kmph (Considering Traffic)
(Fixed for all vehicle categories and conditions)
```

### Charging Rate:
```
Constant Charging Rate: 22 kW 
(Standard Level 2 AC charging for all vehicles)
```

These constants will apply uniformly across all EV categories:
- Small EV (35 kWh)
- Medium EV (40 kWh)
- Large EV (45 kWh)
- Extra Large EV (50 kWh)




### Distance Parameters:

1. For Customer Locations:
```
Distance from Depot to Customers:
- Minimum: 10 km
- Maximum: 60 km
(Ensures coverage within city/suburban limits)

Distance between Customers:
- Minimum: 5 km (avoid unrealistic clustering)
```

2. For Charging Station Placement:
```
Distance between Charging Stations:
- Minimum: 20 km
(Based on EV ranges and safety margins)

Distance from Customers to Nearest Charging Station:
- Maximum: 30 km
(Ensures reachability with low battery)

Coverage Requirements:
- Each customer should have at least one charging station within 30 km
- Each charging station should serve minimum 3-4 customers
```





