from collections import deque

class BFS:
    """
    Breadth-First Search algorithm for maze solving.
    Guarantees the shortest path in terms of number of steps.
    """
    
    def __init__(self):
        self.name = "BFS"
        self.description = "Breadth-First Search - Explores all neighbors before moving deeper"
        self.complexity = "O(V + E)"
    
    def solve(self, grid, start, end):
        """
        Solve maze using Breadth-First Search algorithm.
        
        Args:
            grid: 2D list where 0=path, 1=wall
            start: (row, col) tuple for start position
            end: (row, col) tuple for end position
            
        Returns:
            Tuple of (path, visited_nodes)
            path: List of (row, col) coordinates from start to end
            visited_nodes: List of all nodes visited during search
        """
        if not grid or start == end:
            return [start] if start == end else [], []
        
        rows, cols = len(grid), len(grid[0])
        
        # Check if start and end are valid
        if (start[0] < 0 or start[0] >= rows or start[1] < 0 or start[1] >= cols or
            end[0] < 0 or end[0] >= rows or end[1] < 0 or end[1] >= cols or
            grid[start[0]][start[1]] == 1 or grid[end[0]][end[1]] == 1):
            return [], []
        
        # Initialize BFS
        queue = deque([start])
        visited = {start}
        parent = {start: None}
        visited_order = [start]
        
        # Directions: up, down, left, right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while queue:
            current = queue.popleft()
            
            # Check if we reached the destination
            if current == end:
                path = self._reconstruct_path(parent, start, end)
                return path, visited_order
            
            # Explore neighbors
            for dr, dc in directions:
                new_row, new_col = current[0] + dr, current[1] + dc
                neighbor = (new_row, new_col)
                
                # Check bounds and walls
                if (0 <= new_row < rows and 0 <= new_col < cols and
                    grid[new_row][new_col] == 0 and neighbor not in visited):
                    
                    visited.add(neighbor)
                    visited_order.append(neighbor)
                    parent[neighbor] = current
                    queue.append(neighbor)
        
        # No path found
        return [], visited_order
    
    def _reconstruct_path(self, parent, start, end):
        """
        Reconstruct path from parent dictionary.
        
        Args:
            parent: Dictionary mapping node to its parent
            start: Start position
            end: End position
            
        Returns:
            List of coordinates representing the path
        """
        path = []
        current = end
        
        while current is not None:
            path.append(current)
            current = parent[current]
        
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
        """
        Get detailed information about the algorithm.
        
        Returns:
            Dictionary with algorithm details
        """
        return {
            'name': self.name,
            'description': self.description,
            'time_complexity': 'O(V + E)',
            'space_complexity': 'O(V)',
            'optimal': True,
            'complete': True,
            'characteristics': [
                'Explores nodes level by level',
                'Guarantees shortest path',
                'Uses queue (FIFO) data structure',
                'Systematic exploration'
            ],
            'best_use_cases': [
                'Finding shortest path in unweighted graphs',
                'When optimal solution is required',
                'Exploring state spaces systematically'
            ],
            'limitations': [
                'Can be memory intensive for large graphs',
                'Explores many unnecessary nodes',
                'Not suitable for weighted graphs'
            ]
        }


class BFSWithStats(BFS):
    """
    Enhanced BFS that tracks additional statistics during execution.
    """
    
    def __init__(self):
        super().__init__()
        self.stats = {}
    
    def solve(self, grid, start, end):
        """
        Solve maze with detailed statistics tracking.
        """
        import time
        
        start_time = time.time()
        
        # Reset stats
        self.stats = {
            'nodes_expanded': 0,
            'nodes_generated': 0,
            'max_queue_size': 0,
            'branching_factor': 0,
            'depth_reached': 0
        }
        
        if not grid or start == end:
            return [start] if start == end else [], []
        
        rows, cols = len(grid), len(grid[0])
        
        # Check validity
        if (start[0] < 0 or start[0] >= rows or start[1] < 0 or start[1] >= cols or
            end[0] < 0 or end[0] >= rows or end[1] < 0 or end[1] >= cols or
            grid[start[0]][start[1]] == 1 or grid[end[0]][end[1]] == 1):
            return [], []
        
        # Initialize BFS with stats tracking
        queue = deque([(start, 0)])  # (position, depth)
        visited = {start}
        parent = {start: None}
        visited_order = [start]
        
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        total_children = 0
        
        while queue:
            current_pos, depth = queue.popleft()
            self.stats['nodes_expanded'] += 1
            self.stats['depth_reached'] = max(self.stats['depth_reached'], depth)
            
            if current_pos == end:
                end_time = time.time()
                self.stats['execution_time'] = (end_time - start_time) * 1000  # ms
                self.stats['branching_factor'] = (total_children / self.stats['nodes_expanded'] 
                                                if self.stats['nodes_expanded'] > 0 else 0)
                
                path = self._reconstruct_path(parent, start, end)
                return path, visited_order
            
            children_count = 0
            
            # Explore neighbors
            for dr, dc in directions:
                new_row, new_col = current_pos[0] + dr, current_pos[1] + dc
                neighbor = (new_row, new_col)
                
                if (0 <= new_row < rows and 0 <= new_col < cols and
                    grid[new_row][new_col] == 0 and neighbor not in visited):
                    
                    visited.add(neighbor)
                    visited_order.append(neighbor)
                    parent[neighbor] = current_pos
                    queue.append((neighbor, depth + 1))
                    
                    children_count += 1
                    self.stats['nodes_generated'] += 1
            
            total_children += children_count
            self.stats['max_queue_size'] = max(self.stats['max_queue_size'], len(queue))
        
        # No path found
        end_time = time.time()
        self.stats['execution_time'] = (end_time - start_time) * 1000
        self.stats['branching_factor'] = (total_children / self.stats['nodes_expanded'] 
                                        if self.stats['nodes_expanded'] > 0 else 0)
        
        return [], visited_order
    
    def get_stats(self):
        """Return detailed statistics from last execution."""
        return self.stats.copy()


class BidirectionalBFS(BFS):
    """
    Bidirectional BFS that searches from both start and end simultaneously.
    Can be more efficient than regular BFS for long paths.
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Bidirectional BFS"
        self.description = "Searches from both start and end simultaneously"
        self.complexity = "O(b^(d/2))"
    
    def solve(self, grid, start, end):
        """
        Solve maze using bidirectional BFS.
        """
        if not grid or start == end:
            return [start] if start == end else [], []
        
        rows, cols = len(grid), len(grid[0])
        
        # Check validity
        if (start[0] < 0 or start[0] >= rows or start[1] < 0 or start[1] >= cols or
            end[0] < 0 or end[0] >= rows or end[1] < 0 or end[1] >= cols or
            grid[start[0]][start[1]] == 1 or grid[end[0]][end[1]] == 1):
            return [], []
        
        # Initialize two searches
        forward_queue = deque([start])
        backward_queue = deque([end])
        
        forward_visited = {start}
        backward_visited = {end}
        
        forward_parent = {start: None}
        backward_parent = {end: None}
        
        visited_order = [start, end]
        
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while forward_queue or backward_queue:
            # Forward search
            if forward_queue:
                current = forward_queue.popleft()
                
                # Check if we meet the backward search
                if current in backward_visited:
                    path = self._reconstruct_bidirectional_path(
                        forward_parent, backward_parent, start, end, current
                    )
                    return path, visited_order
                
                # Expand forward
                for dr, dc in directions:
                    new_row, new_col = current[0] + dr, current[1] + dc
                    neighbor = (new_row, new_col)
                    
                    if (0 <= new_row < rows and 0 <= new_col < cols and
                        grid[new_row][new_col] == 0 and neighbor not in forward_visited):
                        
                        forward_visited.add(neighbor)
                        visited_order.append(neighbor)
                        forward_parent[neighbor] = current
                        forward_queue.append(neighbor)
            
            # Backward search
            if backward_queue:
                current = backward_queue.popleft()
                
                # Check if we meet the forward search
                if current in forward_visited:
                    path = self._reconstruct_bidirectional_path(
                        forward_parent, backward_parent, start, end, current
                    )
                    return path, visited_order
                
                # Expand backward
                for dr, dc in directions:
                    new_row, new_col = current[0] + dr, current[1] + dc
                    neighbor = (new_row, new_col)
                    
                    if (0 <= new_row < rows and 0 <= new_col < cols and
                        grid[new_row][new_col] == 0 and neighbor not in backward_visited):
                        
                        backward_visited.add(neighbor)
                        visited_order.append(neighbor)
                        backward_parent[neighbor] = current
                        backward_queue.append(neighbor)
        
        return [], visited_order
    
    def _reconstruct_bidirectional_path(self, forward_parent, backward_parent, start, end, meeting_point):
        """
        Reconstruct path from bidirectional search.
        """
        # Forward path (start to meeting point)
        forward_path = []
        current = meeting_point
        while current is not None:
            forward_path.append(current)
            current = forward_parent[current]
        forward_path.reverse()
        
        # Backward path (meeting point to end)
        backward_path = []
        current = backward_parent[meeting_point]
        while current is not None:
            backward_path.append(current)
            current = backward_parent[current]
        
        # Combine paths
        return forward_path + backward_path