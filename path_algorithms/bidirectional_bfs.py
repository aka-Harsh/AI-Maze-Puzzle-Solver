from collections import deque

class BidirectionalBFS:
    """
    Bidirectional Breadth-First Search algorithm for maze solving.
    Searches from both start and end simultaneously.
    More efficient than regular BFS for long paths.
    """
    
    def __init__(self):
        self.name = "Bidirectional BFS"
        self.description = "Bidirectional BFS - Searches from both start and end simultaneously"
        self.complexity = "O(b^(d/2))"
    
    def solve(self, grid, start, end):
        """
        Solve maze using Bidirectional BFS algorithm.
        
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
        
        # Initialize forward search (from start)
        forward_queue = deque([start])
        forward_visited = {start}
        forward_parent = {start: None}
        
        # Initialize backward search (from end)
        backward_queue = deque([end])
        backward_visited = {end}
        backward_parent = {end: None}
        
        visited_order = [start, end]
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while forward_queue or backward_queue:
            # Forward search step
            if forward_queue:
                current = forward_queue.popleft()
                
                # Check if we meet the backward search
                if current in backward_visited:
                    path = self._reconstruct_bidirectional_path(
                        forward_parent, backward_parent, start, end, current
                    )
                    return path, visited_order
                
                # Expand forward search
                for dr, dc in directions:
                    new_row, new_col = current[0] + dr, current[1] + dc
                    neighbor = (new_row, new_col)
                    
                    if (0 <= new_row < rows and 0 <= new_col < cols and
                        grid[new_row][new_col] == 0 and neighbor not in forward_visited):
                        
                        forward_visited.add(neighbor)
                        visited_order.append(neighbor)
                        forward_parent[neighbor] = current
                        forward_queue.append(neighbor)
            
            # Backward search step
            if backward_queue:
                current = backward_queue.popleft()
                
                # Check if we meet the forward search
                if current in forward_visited:
                    path = self._reconstruct_bidirectional_path(
                        forward_parent, backward_parent, start, end, current
                    )
                    return path, visited_order
                
                # Expand backward search
                for dr, dc in directions:
                    new_row, new_col = current[0] + dr, current[1] + dc
                    neighbor = (new_row, new_col)
                    
                    if (0 <= new_row < rows and 0 <= new_col < cols and
                        grid[new_row][new_col] == 0 and neighbor not in backward_visited):
                        
                        backward_visited.add(neighbor)
                        visited_order.append(neighbor)
                        backward_parent[neighbor] = current
                        backward_queue.append(neighbor)
        
        # No path found
        return [], visited_order
    
    def _reconstruct_bidirectional_path(self, forward_parent, backward_parent, start, end, meeting_point):
        """
        Reconstruct path from bidirectional search.
        
        Args:
            forward_parent: Parent dictionary from forward search
            backward_parent: Parent dictionary from backward search
            start: Start position
            end: End position
            meeting_point: Point where searches meet
            
        Returns:
            Complete path from start to end
        """
        # Build forward path (start to meeting point)
        forward_path = []
        current = meeting_point
        while current is not None:
            forward_path.append(current)
            current = forward_parent[current]
        forward_path.reverse()
        
        # Build backward path (meeting point to end)
        backward_path = []
        current = backward_parent[meeting_point]
        while current is not None:
            backward_path.append(current)
            current = backward_parent[current]
        
        # Combine paths (avoid duplicating meeting point)
        return forward_path + backward_path
    
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
            'time_complexity': 'O(b^(d/2))',
            'space_complexity': 'O(b^(d/2))',
            'optimal': True,
            'complete': True,
            'characteristics': [
                'Searches from both ends simultaneously',
                'More efficient than single-direction BFS',
                'Reduces search space significantly',
                'Guarantees shortest path'
            ],
            'best_use_cases': [
                'Long paths in large graphs',
                'When start and end are far apart',
                'Memory-efficient pathfinding',
                'Symmetric search problems'
            ],
            'limitations': [
                'More complex implementation',
                'Requires knowledge of goal state',
                'May not help if path is short',
                'Overhead of managing two searches'
            ]
        }


class BidirectionalBFSWithHeuristic(BidirectionalBFS):
    """
    Bidirectional BFS enhanced with heuristic guidance.
    Prioritizes expansion based on distance to the other search frontier.
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Bidirectional BFS (Heuristic)"
        self.description = "Bidirectional BFS with heuristic guidance"
    
    def solve(self, grid, start, end):
        """
        Solve using bidirectional BFS with heuristic prioritization.
        """
        if not grid or start == end:
            return [start] if start == end else [], []
        
        rows, cols = len(grid), len(grid[0])
        
        # Check validity
        if (start[0] < 0 or start[0] >= rows or start[1] < 0 or start[1] >= cols or
            end[0] < 0 or end[0] >= rows or end[1] < 0 or end[1] >= cols or
            grid[start[0]][start[1]] == 1 or grid[end[0]][end[1]] == 1):
            return [], []
        
        # Use priority queues instead of regular queues
        import heapq
        
        forward_queue = [(0, start)]
        forward_visited = {start: 0}
        forward_parent = {start: None}
        
        backward_queue = [(0, end)]
        backward_visited = {end: 0}
        backward_parent = {end: None}
        
        visited_order = [start, end]
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while forward_queue or backward_queue:
            # Alternate between forward and backward search
            # Choose the search with shorter estimated distance to goal
            
            if forward_queue and (not backward_queue or forward_queue[0][0] <= backward_queue[0][0]):
                # Forward search step
                current_dist, current = heapq.heappop(forward_queue)
                
                if current in backward_visited:
                    path = self._reconstruct_bidirectional_path_with_costs(
                        forward_parent, backward_parent, forward_visited, 
                        backward_visited, start, end, current
                    )
                    return path, visited_order
                
                # Expand forward
                for dr, dc in directions:
                    new_row, new_col = current[0] + dr, current[1] + dc
                    neighbor = (new_row, new_col)
                    
                    if (0 <= new_row < rows and 0 <= new_col < cols and
                        grid[new_row][new_col] == 0):
                        
                        new_dist = current_dist + 1
                        
                        if neighbor not in forward_visited or new_dist < forward_visited[neighbor]:
                            forward_visited[neighbor] = new_dist
                            forward_parent[neighbor] = current
                            
                            # Priority based on distance from start + heuristic to end
                            priority = new_dist + self._manhattan_distance(neighbor, end)
                            heapq.heappush(forward_queue, (priority, neighbor))
                            
                            if neighbor not in [item[1] for item in visited_order]:
                                visited_order.append(neighbor)
            
            elif backward_queue:
                # Backward search step
                current_dist, current = heapq.heappop(backward_queue)
                
                if current in forward_visited:
                    path = self._reconstruct_bidirectional_path_with_costs(
                        forward_parent, backward_parent, forward_visited,
                        backward_visited, start, end, current
                    )
                    return path, visited_order
                
                # Expand backward
                for dr, dc in directions:
                    new_row, new_col = current[0] + dr, current[1] + dc
                    neighbor = (new_row, new_col)
                    
                    if (0 <= new_row < rows and 0 <= new_col < cols and
                        grid[new_row][new_col] == 0):
                        
                        new_dist = current_dist + 1
                        
                        if neighbor not in backward_visited or new_dist < backward_visited[neighbor]:
                            backward_visited[neighbor] = new_dist
                            backward_parent[neighbor] = current
                            
                            # Priority based on distance from end + heuristic to start
                            priority = new_dist + self._manhattan_distance(neighbor, start)
                            heapq.heappush(backward_queue, (priority, neighbor))
                            
                            if neighbor not in [item[1] for item in visited_order]:
                                visited_order.append(neighbor)
        
        return [], visited_order
    
    def _manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _reconstruct_bidirectional_path_with_costs(self, forward_parent, backward_parent, 
                                                  forward_costs, backward_costs, start, end, meeting_point):
        """Reconstruct path with cost information."""
        # Forward path
        forward_path = []
        current = meeting_point
        while current is not None:
            forward_path.append(current)
            current = forward_parent[current]
        forward_path.reverse()
        
        # Backward path
        backward_path = []
        current = backward_parent[meeting_point] if meeting_point in backward_parent else None
        while current is not None:
            backward_path.append(current)
            current = backward_parent[current]
        
        return forward_path + backward_path


class BidirectionalAstar(BidirectionalBFS):
    """
    Bidirectional A* search algorithm.
    Combines the efficiency of bidirectional search with A* heuristics.
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Bidirectional A*"
        self.description = "Bidirectional A* - Combines bidirectional search with heuristics"
        self.complexity = "O(b^(d/2))"
    
    def solve(self, grid, start, end):
        """
        Solve using Bidirectional A* algorithm.
        """
        if not grid or start == end:
            return [start] if start == end else [], []
        
        rows, cols = len(grid), len(grid[0])
        
        # Check validity
        if (start[0] < 0 or start[0] >= rows or start[1] < 0 or start[1] >= cols or
            end[0] < 0 or end[0] >= rows or end[1] < 0 or end[1] >= cols or
            grid[start[0]][start[1]] == 1 or grid[end[0]][end[1]] == 1):
            return [], []
        
        import heapq
        
        # Forward A* search
        forward_open = [(0, start)]
        forward_g = {start: 0}
        forward_f = {start: self._manhattan_distance(start, end)}
        forward_parent = {start: None}
        forward_closed = set()
        
        # Backward A* search
        backward_open = [(0, end)]
        backward_g = {end: 0}
        backward_f = {end: self._manhattan_distance(end, start)}
        backward_parent = {end: None}
        backward_closed = set()
        
        visited_order = []
        mu = float('inf')  # Upper bound on optimal solution cost
        meeting_point = None
        
        while forward_open or backward_open:
            # Forward search
            if forward_open:
                current_f, current = heapq.heappop(forward_open)
                
                if current in forward_closed:
                    continue
                
                forward_closed.add(current)
                visited_order.append(current)
                
                # Check if paths meet
                if current in backward_closed:
                    cost = forward_g[current] + backward_g[current]
                    if cost < mu:
                        mu = cost
                        meeting_point = current
                
                # Termination condition
                if current_f >= mu:
                    break
                
                # Expand neighbors
                for neighbor in self._get_neighbors(current, grid):
                    if neighbor in forward_closed:
                        continue
                    
                    tentative_g = forward_g[current] + 1
                    
                    if neighbor not in forward_g or tentative_g < forward_g[neighbor]:
                        forward_g[neighbor] = tentative_g
                        forward_f[neighbor] = tentative_g + self._manhattan_distance(neighbor, end)
                        forward_parent[neighbor] = current
                        heapq.heappush(forward_open, (forward_f[neighbor], neighbor))
            
            # Backward search
            if backward_open:
                current_f, current = heapq.heappop(backward_open)
                
                if current in backward_closed:
                    continue
                
                backward_closed.add(current)
                visited_order.append(current)
                
                # Check if paths meet
                if current in forward_closed:
                    cost = forward_g[current] + backward_g[current]
                    if cost < mu:
                        mu = cost
                        meeting_point = current
                
                # Termination condition
                if current_f >= mu:
                    break
                
                # Expand neighbors
                for neighbor in self._get_neighbors(current, grid):
                    if neighbor in backward_closed:
                        continue
                    
                    tentative_g = backward_g[current] + 1
                    
                    if neighbor not in backward_g or tentative_g < backward_g[neighbor]:
                        backward_g[neighbor] = tentative_g
                        backward_f[neighbor] = tentative_g + self._manhattan_distance(neighbor, start)
                        backward_parent[neighbor] = current
                        heapq.heappush(backward_open, (backward_f[neighbor], neighbor))
        
        if meeting_point:
            path = self._reconstruct_bidirectional_path(
                forward_parent, backward_parent, start, end, meeting_point
            )
            return path, visited_order
        
        return [], visited_order
    
    def _get_neighbors(self, pos, grid):
        """Get valid neighboring positions."""
        rows, cols = len(grid), len(grid[0])
        neighbors = []
        
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for dr, dc in directions:
            new_row, new_col = pos[0] + dr, pos[1] + dc
            
            if (0 <= new_row < rows and 0 <= new_col < cols and
                grid[new_row][new_col] == 0):
                neighbors.append((new_row, new_col))
        
        return neighbors
    
    def _manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance heuristic."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])