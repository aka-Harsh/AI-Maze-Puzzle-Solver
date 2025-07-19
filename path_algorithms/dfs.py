class DFS:
    """
    Depth-First Search algorithm for maze solving.
    Explores as far as possible before backtracking.
    Does not guarantee shortest path.
    """
    
    def __init__(self):
        self.name = "DFS"
        self.description = "Depth-First Search - Explores as far as possible before backtracking"
        self.complexity = "O(V + E)"
    
    def solve(self, grid, start, end):
        """
        Solve maze using iterative DFS for better consistency.
        """
        if not grid or not grid[0]:
            return [], []
            
        if start == end:
            return [start], [start]
        
        rows, cols = len(grid), len(grid[0])
        
        # Check validity
        if (start[0] < 0 or start[0] >= rows or start[1] < 0 or start[1] >= cols or
            end[0] < 0 or end[0] >= rows or end[1] < 0 or end[1] >= cols or
            grid[start[0]][start[1]] == 1 or grid[end[0]][end[1]] == 1):
            return [], []
        
        # Use iterative DFS for more predictable behavior
        stack = [start]
        visited = {start}
        parent = {start: None}
        visited_order = [start]
        
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left
        
        while stack:
            current = stack.pop()
            
            if current == end:
                path = self._reconstruct_path(parent, start, end)
                return path, visited_order
            
            # Explore neighbors (in reverse order due to stack LIFO)
            for dr, dc in reversed(directions):
                new_row, new_col = current[0] + dr, current[1] + dc
                neighbor = (new_row, new_col)
                
                if (0 <= new_row < rows and 0 <= new_col < cols and
                    grid[new_row][new_col] == 0 and neighbor not in visited):
                    
                    visited.add(neighbor)
                    visited_order.append(neighbor)
                    parent[neighbor] = current
                    stack.append(neighbor)
        
        return [], visited_order
    
    def solve_iterative(self, grid, start, end):
        """
        Iterative implementation of DFS using a stack.
        """
        if not grid or start == end:
            return [start] if start == end else [], []
        
        rows, cols = len(grid), len(grid[0])
        
        # Check validity
        if (start[0] < 0 or start[0] >= rows or start[1] < 0 or start[1] >= cols or
            end[0] < 0 or end[0] >= rows or end[1] < 0 or end[1] >= cols or
            grid[start[0]][start[1]] == 1 or grid[end[0]][end[1]] == 1):
            return [], []
        
        stack = [start]
        visited = {start}
        parent = {start: None}
        visited_order = [start]
        
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        
        while stack:
            current = stack.pop()
            
            if current == end:
                path = self._reconstruct_path(parent, start, end)
                return path, visited_order
            
            # Explore neighbors (in reverse order due to stack LIFO)
            for dr, dc in reversed(directions):
                new_row, new_col = current[0] + dr, current[1] + dc
                neighbor = (new_row, new_col)
                
                if (0 <= new_row < rows and 0 <= new_col < cols and
                    grid[new_row][new_col] == 0 and neighbor not in visited):
                    
                    visited.add(neighbor)
                    visited_order.append(neighbor)
                    parent[neighbor] = current
                    stack.append(neighbor)
        
        return [], visited_order
    
    def _reconstruct_path(self, parent, start, end):
        """Reconstruct path from parent dictionary."""
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
        return False
    
    def get_algorithm_info(self):
        """Get detailed information about the algorithm."""
        return {
            'name': self.name,
            'description': self.description,
            'time_complexity': 'O(V + E)',
            'space_complexity': 'O(V)',
            'optimal': False,
            'complete': True,
            'characteristics': [
                'Explores deeply before backtracking',
                'Uses stack (LIFO) data structure',
                'Memory efficient',
                'May find suboptimal paths'
            ],
            'best_use_cases': [
                'When memory is limited',
                'Exploring all possible paths',
                'When any solution is acceptable'
            ],
            'limitations': [
                'Does not guarantee shortest path',
                'Can get stuck in deep branches',
                'May be slower than BFS for short paths'
            ]
        }