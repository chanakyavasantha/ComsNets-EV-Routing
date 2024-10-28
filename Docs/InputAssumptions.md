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

Summary Table:
| Category | Battery | Base Weight | Max Load | Full Load Range (Min-Max) | Energy Consumption (Empty-Full) |
|----------|---------|-------------|-----------|--------------------------|--------------------------------|
| Small    | 35 kWh  | 1500 kg    | 500 kg   | 28 - 112 km             | 0.225 - 0.25 kWh/km           |
| Medium   | 40 kWh  | 1800 kg    | 600 kg   | 29.6 - 118.5 km         | 0.24 - 0.27 kWh/km            |
| Large    | 45 kWh  | 2000 kg    | 700 kg   | 31.6 - 126.3 km         | 0.25 - 0.285 kWh/km           |
| X-Large  | 50 kWh  | 2200 kg    | 800 kg   | 33.3 - 133.3 km         | 0.26 - 0.30 kWh/km            |



###  Customer Item Weights:
```
Range: 5-50 kg
- Minimum Package Weight: 5 kg
- Maximum Package Weight: 50 kg
```

### Vehicle Speed:
```
Constant Operating Speed: 25 kmph
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
- Minimum: 5 km
- Maximum: 30 km
(Ensures coverage within city/suburban limits)

Distance between Customers:
- Minimum: 2 km (avoid unrealistic clustering)
- Maximum: 15 km (reasonable travel between deliveries)
```

2. For Charging Station Placement:
```
Distance between Charging Stations:
- Minimum: 10 km
- Maximum: 25 km
(Based on EV ranges and safety margins)

Distance from Customers to Nearest Charging Station:
- Maximum: 15 km
(Ensures reachability with low battery)

Coverage Requirements:
- Each customer should have at least one charging station within 15 km
- Each charging station should serve minimum 3-4 customers
```

Rationale:
1. Minimum distances ensure realistic spacing
2. Maximum distances are set to:
   - Allow round trips on 20% battery (minimum range ~28 km for smallest EV)
   - Enable reaching next charging station even at low battery
   - Cover typical urban/suburban delivery areas
3. Charging station spacing ensures:
   - No "charging deserts"
   - Efficient coverage without redundancy
   - Safety margin for battery depletion
