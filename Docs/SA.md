INITIAL SOLUTION ALGORITHM FOR EV ROUTING

Objective: Create initial routes by clustering customers geographically while respecting vehicle capacities

Input:
- customer_locations: List of (x,y) coordinates for each customer
- customer_demands: List of demands (weights) for each customer
- depot_location: (x,y) coordinate of depot
- vehicle_types: Dictionary of available vehicle types and their specifications
    {
        'small':  {'capacity': 500, 'battery': 35, 'weight': 1500},
        'medium': {'capacity': 600, 'battery': 40, 'weight': 1800},
        'large':  {'capacity': 700, 'battery': 45, 'weight': 2000},
        'xlarge': {'capacity': 800, 'battery': 50, 'weight': 2200}
    }

Output:
- Solution object containing:
    - routes: List of routes, each route is list of customer indices
    - vehicle_types: List of assigned vehicle type for each route
    - loads: List of total load for each route

Algorithm Steps:

1. DETERMINE NUMBER OF CLUSTERS:
   function calculate_initial_clusters(customer_demands, min_vehicle_capacity):
       total_demand = sum(customer_demands)
       return ceiling(total_demand / min_vehicle_capacity)

2. PERFORM K-MEANS CLUSTERING:
   function cluster_customers(customer_locations, num_clusters):
       Input: customer locations as (x,y) coordinates
       Output: cluster assignments for each customer
       
       Initialize K-means with num_clusters centroids
       Run K-means until convergence
       Return cluster assignments

3. CREATE ROUTES FROM CLUSTERS:
   function create_routes_from_clusters(cluster_assignments, customer_locations):
       For each cluster:
           Initialize empty route
           Add depot (0) as first point
           
           // Use nearest neighbor within cluster
           current = depot
           unvisited = customers in current cluster
           While unvisited not empty:
               next = find_nearest(current, unvisited)
               Add next to route
               Remove next from unvisited
               current = next
           
           Add depot (0) as last point
           Add route to routes list

4. ASSIGN VEHICLE TYPES:
   function assign_vehicle_types(routes, customer_demands, vehicle_types):
       For each route:
           Calculate total_load for route
           Find smallest vehicle type that can handle load:
               For each vehicle_type in ascending capacity order:
                   If vehicle_type.capacity >= total_load:
                       Assign vehicle_type to route
                       Break

5. MAIN INITIALIZATION FUNCTION:
   function initialize_solution():
       // Step 1: Calculate number of clusters
       num_clusters = calculate_initial_clusters(
           customer_demands, 
           min_vehicle_capacity=500
       )
       
       // Step 2: Cluster customers
       cluster_assignments = cluster_customers(
           customer_locations, 
           num_clusters
       )
       
       // Step 3: Create routes
       routes = create_routes_from_clusters(
           cluster_assignments,
           customer_locations
       )
       
       // Step 4: Assign vehicles
       vehicle_types = assign_vehicle_types(
           routes,
           customer_demands,
           vehicle_types
       )
       
       Return Solution(routes, vehicle_types)

Example Usage:
Input:
- 8 customers with locations and demands (from previous example)
- Depot at (0,0)

Output Example:
{
    'routes': [
        [0, 1, 4, 2, 0],    # Cluster 1 route
        [0, 3, 7, 0],       # Cluster 2 route
        [0, 5, 8, 6, 0]     # Cluster 3 route
    ],
    'vehicle_types': [
        'small',            # For route 1
        'small',            # For route 2
        'small'             # For route 3
    ],
    'loads': [
        370,               # Total load for route 1
        330,               # Total load for route 2
        430                # Total load for route 3
    ]
}


NEIGHBORHOOD OPERATIONS FOR EV ROUTING

Objective: Define operations that generate neighboring solutions by modifying customer sequences

Types of Operations:

1. INTER-ROUTE OPERATIONS (Between Routes):

   a) SWAP Operation:
      Input: Two routes R1, R2 and positions i, j
      Function: Exchange customers at position i in R1 with position j in R2
      
      Example:
      R1: [0 → 1 → 2 → 3 → 0]
      R2: [0 → 4 → 5 → 6 → 0]
      Swap(R1[2], R2[2])
      Result:
      R1: [0 → 1 → 5 → 3 → 0]
      R2: [0 → 4 → 2 → 6 → 0]

   b) RELOCATE Operation:
      Input: Source route Rs, destination route Rd, position i
      Function: Move customer from position i in Rs to best position in Rd
      
      Example:
      R1: [0 → 1 → 2 → 3 → 0]
      R2: [0 → 4 → 5 → 6 → 0]
      Relocate(R1[2] to R2)
      Result:
      R1: [0 → 1 → 3 → 0]
      R2: [0 → 4 → 2 → 5 → 6 → 0]

2. INTRA-ROUTE OPERATIONS (Within Route):

   a) 2-OPT Operation:
      Input: Single route R, positions i, j
      Function: Reverse sequence between positions i and j
      
      Example:
      R: [0 → 1 → 2 → 3 → 4 → 0]
      2-opt(R, 2, 4)
      Result:
      R: [0 → 1 → 4 → 3 → 2 → 0]

   b) RELOCATE Operation:
      Input: Route R, positions i, j
      Function: Move customer from position i to j
      
      Example:
      R: [0 → 1 → 2 → 3 → 4 → 0]
      Relocate(R, 2, 4)
      Result:
      R: [0 → 1 → 3 → 4 → 2 → 0]

Implementation:

```python
class NeighborhoodOperations:
    def swap_between_routes(self, solution, r1_idx, r2_idx, pos1, pos2):
        """
        Swap customers between two routes
        
        Args:
            solution: Current solution
            r1_idx, r2_idx: Indices of routes to swap between
            pos1, pos2: Positions in routes to swap
            
        Returns:
            New solution with swap performed
        """
        new_solution = deepcopy(solution)
        route1 = new_solution.routes[r1_idx]
        route2 = new_solution.routes[r2_idx]
        
        # Perform swap
        route1[pos1], route2[pos2] = route2[pos2], route1[pos1]
        
        # Update vehicle assignments if needed
        self._update_vehicle_types([r1_idx, r2_idx], new_solution)
        
        return new_solution

    def relocate_between_routes(self, solution, source_idx, dest_idx, pos):
        """
        Relocate customer from one route to another
        
        Args:
            solution: Current solution
            source_idx, dest_idx: Route indices
            pos: Position to move from source route
            
        Returns:
            New solution with relocation performed
        """
        new_solution = deepcopy(solution)
        source_route = new_solution.routes[source_idx]
        dest_route = new_solution.routes[dest_idx]
        
        # Remove customer from source route
        customer = source_route.pop(pos)
        
        # Find best insertion position in destination route
        best_pos = self._find_best_insertion(dest_route, customer)
        dest_route.insert(best_pos, customer)
        
        # Update vehicle assignments
        self._update_vehicle_types([source_idx, dest_idx], new_solution)
        
        return new_solution

    def two_opt_within_route(self, solution, route_idx, i, j):
        """
        Perform 2-opt operation within a route
        
        Args:
            solution: Current solution
            route_idx: Route to modify
            i, j: Positions to reverse between
            
        Returns:
            New solution with 2-opt performed
        """
        new_solution = deepcopy(solution)
        route = new_solution.routes[route_idx]
        
        # Reverse segment
        route[i:j+1] = reversed(route[i:j+1])
        
        return new_solution

    def relocate_within_route(self, solution, route_idx, pos1, pos2):
        """
        Relocate customer within a route
        
        Args:
            solution: Current solution
            route_idx: Route to modify
            pos1, pos2: Position to move from/to
            
        Returns:
            New solution with relocation performed
        """
        new_solution = deepcopy(solution)
        route = new_solution.routes[route_idx]
        
        # Perform relocation
        customer = route.pop(pos1)
        route.insert(pos2, customer)
        
        return new_solution

    def _find_best_insertion(self, route, customer):
        """Helper function to find best position to insert customer"""
        best_pos = 1  # Start after depot
        best_cost = float('inf')
        
        for i in range(1, len(route)):
            # Try insertion at position i
            test_route = route[:i] + [customer] + route[i:]
            cost = self._calculate_route_duration(test_route)
            
            if cost < best_cost:
                best_cost = cost
                best_pos = i
                
        return best_pos

    def _update_vehicle_types(self, route_indices, solution):
        """Helper function to update vehicle types after route modification"""
        for idx in route_indices:
            route = solution.routes[idx]
            total_load = sum(self.customer_demands[c-1] for c in route[1:-1])
            solution.vehicle_types[idx] = self._select_vehicle_type(total_load)