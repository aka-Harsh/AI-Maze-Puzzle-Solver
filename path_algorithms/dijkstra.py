import heapq

class Dijkstra:
    """
    Dijkstra's algorithm for maze solving.
    Finds shortest path in weighted graphs.
    For unweighted graphs (like mazes), equivalent to BFS.
    """
    
    def __init__(self):
        self.name = "Dijkstra"
        self.description = "Dijkstra's Algorithm - Finds shortest path in weighted graphs"
        self.complexity = "O((V + E) log V)"
    
    def solve(self, grid, start, end):
        """
        Solve maze using Dijkstra's algorithm.
        
        Args:
            grid: 2D list where 0=path, 1=wall
            start: (row, col) tuple for start position
            end: (row, col) tuple for end position
            
        Returns:
            Tuple of (path, visited_nodes)
        """
        if not grid or start == end:
            return [start] if start == end else [], []
        
        rows, cols = len(grid), len(grid[0])
        
        # Check validity
        if (start[0] < 0 or start[0] >= rows or start[1] < 0 or start[1] >= cols or
            end[0] < 0 or end[0] >= rows or end[1] < 0 or end[1] >= cols or
            grid[start[0]][start[1]] == 1 or grid[end[0]][end[1]] == 1):
            return [], []
        
        # Initialize Dijkstra's algorithm
        distances = {}
        previous = {}
        visited_order = []
        
        # Priority queue: (distance, position)
        pq = [(0, start)]
        distances[start] = 0
        
        while pq:
            current_dist, current_pos = heapq.heappop(pq)
            
            # Skip if we've already processed this node with a better distance
            if current_pos in visited_order:
                continue
            
            visited_order.append(current_pos)
            
            # Found the target
            if current_pos == end:
                path = self._reconstruct_path(previous, start, end)
                return path, visited_order
            
            # Explore neighbors
            for neighbor in self._get_neighbors(current_pos, grid):
                if neighbor in visited_order:
                    continue
                
                # Calculate distance to neighbor
                edge_weight = self._get_edge_weight(current_pos, neighbor)
                new_distance = current_dist + edge_weight
                
                # If we found a shorter path to neighbor
                if neighbor not in distances or new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    previous[neighbor] = current_pos
                    heapq.heappush(pq, (new_distance, neighbor))
        
        # No path found
        return [], visited_order
    
    def _get_neighbors(self, pos, grid):
        """Get valid neighboring positions."""
        rows, cols = len(grid), len(grid[0])
        neighbors = []
        
        # 4-directional movement
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for dr, dc in directions:
            new_row, new_col = pos[0] + dr, pos[1] + dc
            
            if (0 <= new_row < rows and 0 <= new_col < cols and
                grid[new_row][new_col] == 0):
                neighbors.append((new_row, new_col))
        
        return neighbors
    
    def _get_edge_weight(self, pos1, pos2):
        """
        Calculate edge weight between two adjacent positions.
        Can be modified to handle different terrain costs.
        """
        # For basic maze: uniform cost
        return 1
    
    def _reconstruct_path(self, previous, start, end):
        """Reconstruct path from previous dictionary."""
        path = []
        current = end
        
        while current is not None:
            path.append(current)
            current = previous.get(current)
        
        path.reverse()
        return path
    
    def get_complexity(self):
        """Return the time complexity of the algorithm."""
        return self.complexity
    
    def get_description(self):
        """Return description of the algorithm."""
        return self.description
    
    def is_optimal(self):
        """Return True if algorithm guarantees optimal solution."""
        return True
    
    def get_algorithm_info(self):
        """Get detailed information about the algorithm."""
        return {
            'name': self.name,
            'description': self.description,
            'time_complexity': 'O((V + E) log V)',
            'space_complexity': 'O(V)',
            'optimal': True,
            'complete': True,
            'characteristics': [
                'Explores nodes in order of distance',
                'Maintains distance to all nodes',
                'Uses priority queue',
                'Guarantees shortest path'
            ],
            'best_use_cases': [
                'Weighted graphs',
                'When edge costs vary',
                'Network routing',
                'GPS navigation'
            ],
            'limitations': [
                'Slower than A* with good heuristic',
                'Explores all directions equally',
                'Higher memory usage than DFS'
            ]
        }


class DijkstraWithTerrain(Dijkstra):
    """
    Dijkstra variant that handles different terrain costs.
    """
    
    def __init__(self, terrain_costs=None):
        super().__init__()
        self.name = "Dijkstra (Terrain)"
        self.description = "Dijkstra's Algorithm with variable terrain costs"
        # Default terrain costs
        self.terrain_costs = terrain_costs or {
            0: 1,    # Normal path
            2: 2,    # Slow terrain
            3: 3,    # Very slow terrain
            4: 5,    # Difficult terrain
        }
    
    def _get_edge_weight(self, pos1, pos2):
        """Calculate edge weight based on terrain cost."""
        # Use destination cell's terrain cost
        terrain_type = 0  # Default to normal path
        return self.terrain_costs.get(terrain_type, 1)


class BidirectionalDijkstra(Dijkstra):
    """
    Bidirectional Dijkstra that searches from both start and end.
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Bidirectional Dijkstra"
        self.description = "Dijkstra's Algorithm searching from both ends"
        self.complexity = "O((V + E) log V)"
    
    def solve(self, grid, start, end):
        """
        Solve using bidirectional Dijkstra.
        """
        if not grid or start == end:
            return [start] if start == end else [], []
        
        rows, cols = len(grid), len(grid[0])
        
        # Check validity
        if (start[0] < 0 or start[0] >= rows or start[1] < 0 or start[1] >= cols or
            end[0] < 0 or end[0] >= rows or end[1] < 0 or end[1] >= cols or
            grid[start[0]][start[1]] == 1 or grid[end[0]][end[1]] == 1):
            return [], []
        
        # Initialize forward search
        forward_distances = {start: 0}
        forward_previous = {}
        forward_pq = [(0, start)]
        forward_visited = set()
        
        # Initialize backward search
        backward_distances = {end: 0}
        backward_previous = {}
        backward_pq = [(0, end)]
        backward_visited = set()
        
        visited_order = []
        best_distance = float('inf')
        meeting_point = None
        
        while forward_pq or backward_pq:
            # Forward search step
            if forward_pq:
                f_dist, f_pos = heapq.heappop(forward_pq)
                
                if f_pos not in forward_visited:
                    forward_visited.add(f_pos)
                    visited_order.append(f_pos)
                    
                    # Check if paths meet
                    if f_pos in backward_visited:
                        total_dist = forward_distances[f_pos] + backward_distances[f_pos]
                        if total_dist < best_distance:
                            best_distance = total_dist
                            meeting_point = f_pos
                    
                    # Expand forward
                    for neighbor in self._get_neighbors(f_pos, grid):
                        if neighbor not in forward_visited:
                            new_dist = f_dist + self._get_edge_weight(f_pos, neighbor)
                            if neighbor not in forward_distances or new_dist < forward_distances[neighbor]:
                                forward_distances[neighbor] = new_dist
                                forward_previous[neighbor] = f_pos
                                heapq.heappush(forward_pq, (new_dist, neighbor))
            
            # Backward search step
            if backward_pq:
                b_dist, b_pos = heapq.heappop(backward_pq)
                
                if b_pos not in backward_visited:
                    backward_visited.add(b_pos)
                    visited_order.append(b_pos)
                    
                    # Check if paths meet
                    if b_pos in forward_visited:
                        total_dist = forward_distances[b_pos] + backward_distances[b_pos]
                        if total_dist < best_distance:
                            best_distance = total_dist
                            meeting_point = b_pos
                    
                    # Expand backward
                    for neighbor in self._get_neighbors(b_pos, grid):
                        if neighbor not in backward_visited:
                            new_dist = b_dist + self._get_edge_weight(b_pos, neighbor)
                            if neighbor not in backward_distances or new_dist < backward_distances[neighbor]:
                                backward_distances[neighbor] = new_dist
                                backward_previous[neighbor] = b_pos
                                heapq.heappush(backward_pq, (new_dist, neighbor))
            
            # Check termination condition
            if meeting_point and (not forward_pq or forward_pq[0][0] > best_distance) and \
               (not backward_pq or backward_pq[0][0] > best_distance):
                break
        
        if meeting_point:
            path = self._reconstruct_bidirectional_path(
                forward_previous, backward_previous, start, end, meeting_point
            )
            return path, visited_order
        
        return [], visited_order
    
    def _reconstruct_bidirectional_path(self, forward_previous, backward_previous, start, end, meeting_point):
        """Reconstruct path from bidirectional search."""
        # Forward path (start to meeting point)
        forward_path = []
        current = meeting_point
        while current is not None:
            forward_path.append(current)
            current = forward_previous.get(current)
        forward_path.reverse()
        
        # Backward path (meeting point to end)
        backward_path = []
        current = backward_previous.get(meeting_point)
        while current is not None:
            backward_path.append(current)
            current = backward_previous.get(current)
        
        return forward_path + backward_path