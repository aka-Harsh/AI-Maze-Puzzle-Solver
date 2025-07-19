import heapq
import math

class GreedyBestFirst:
    """
    Greedy Best-First Search algorithm for maze solving.
    Uses only heuristic function to guide search.
    Fast but does not guarantee optimal path.
    """
    
    def __init__(self, heuristic='manhattan'):
        self.name = "Greedy"
        self.description = "Greedy Best-First Search - Uses only heuristic to guide search"
        self.complexity = "O(b^m)"
        self.heuristic_type = heuristic
    
    def solve(self, grid, start, end):
        """
        Solve maze using Greedy Best-First Search algorithm.
        
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
        
        # Initialize Greedy search
        open_set = []
        heapq.heappush(open_set, (self._heuristic(start, end), start))
        
        came_from = {}
        visited = set()
        visited_order = []
        
        while open_set:
            current_h, current = heapq.heappop(open_set)
            
            if current in visited:
                continue
            
            visited.add(current)
            visited_order.append(current)
            
            # Found the goal
            if current == end:
                path = self._reconstruct_path(came_from, current)
                return path, visited_order
            
            # Explore neighbors
            for neighbor in self._get_neighbors(current, grid):
                if neighbor not in visited:
                    came_from[neighbor] = current
                    heapq.heappush(open_set, (h_score, neighbor))
        
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
    
    def _heuristic(self, pos1, pos2):
        """
        Calculate heuristic distance between two positions.
        """
        if self.heuristic_type == 'manhattan':
            return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
        elif self.heuristic_type == 'euclidean':
            return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
        elif self.heuristic_type == 'chebyshev':
            return max(abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1]))
        elif self.heuristic_type == 'octile':
            dx = abs(pos1[0] - pos2[0])
            dy = abs(pos1[1] - pos2[1])
            return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)
        else:
            return 0
    
    def _reconstruct_path(self, came_from, current):
        """Reconstruct path from came_from dictionary."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
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
            'time_complexity': 'O(b^m)',
            'space_complexity': 'O(b^m)',
            'optimal': False,
            'complete': False,
            'heuristic': self.heuristic_type,
            'characteristics': [
                'Uses only heuristic function',
                'Fast and memory efficient',
                'Greedy selection of next node',
                'Can get stuck in local optima'
            ],
            'best_use_cases': [
                'When speed is more important than optimality',
                'Good heuristic available',
                'Limited memory environments',
                'Real-time applications'
            ],
            'limitations': [
                'Does not guarantee shortest path',
                'Can get stuck in dead ends',
                'Incomplete (may not find solution)',
                'Heavily dependent on heuristic quality'
            ]
        }


class GreedyWithBacktracking(GreedyBestFirst):
    """
    Greedy Best-First Search with backtracking capability.
    Improves completeness by allowing backtracking when stuck.
    """
    
    def __init__(self, heuristic='manhattan', max_backtracks=10):
        super().__init__(heuristic)
        self.name = "Greedy (Backtrack)"
        self.description = "Greedy Best-First Search with backtracking"
        self.max_backtracks = max_backtracks
    
    def solve(self, grid, start, end):
        """
        Solve with backtracking when greedy search gets stuck.
        """
        if not grid or start == end:
            return [start] if start == end else [], []
        
        rows, cols = len(grid), len(grid[0])
        
        # Check validity
        if (start[0] < 0 or start[0] >= rows or start[1] < 0 or start[1] >= cols or
            end[0] < 0 or end[0] >= rows or end[1] < 0 or end[1] >= cols or
            grid[start[0]][start[1]] == 1 or grid[end[0]][end[1]] == 1):
            return [], []
        
        visited_global = set()
        visited_order = []
        backtracks = 0
        
        def greedy_search_with_backtrack(current_start, local_visited):
            nonlocal backtracks, visited_global, visited_order
            
            open_set = []
            heapq.heappush(open_set, (self._heuristic(current_start, end), current_start))
            
            came_from = {}
            visited = local_visited.copy()
            path_so_far = []
            
            while open_set:
                current_h, current = heapq.heappop(open_set)
                
                if current in visited:
                    continue
                
                visited.add(current)
                visited_global.add(current)
                visited_order.append(current)
                path_so_far.append(current)
                
                if current == end:
                    path = self._reconstruct_path(came_from, current)
                    return path
                
                neighbors = self._get_neighbors(current, grid)
                valid_neighbors = [n for n in neighbors if n not in visited]
                
                if not valid_neighbors and backtracks < self.max_backtracks:
                    # Stuck - try backtracking
                    backtracks += 1
                    if len(path_so_far) > 1:
                        # Backtrack to a previous position
                        backtrack_pos = path_so_far[-2]
                        result = greedy_search_with_backtrack(backtrack_pos, visited)
                        if result:
                            return result
                
                for neighbor in valid_neighbors:
                    came_from[neighbor] = current
                    h_score = self._heuristic(neighbor, end)
                    heapq.heappush(open_set, (h_score, neighbor))
            
            return None
        
        result = greedy_search_with_backtrack(start, set())
        return result or [], visited_order


class BeamSearch(GreedyBestFirst):
    """
    Beam Search variant that maintains multiple best paths.
    Combines greedy approach with limited breadth.
    """
    
    def __init__(self, beam_width=3, heuristic='manhattan'):
        super().__init__(heuristic)
        self.name = "Beam Search"
        self.description = f"Beam Search with width {beam_width}"
        self.beam_width = beam_width
    
    def solve(self, grid, start, end):
        """
        Solve using beam search algorithm.
        """
        if not grid or start == end:
            return [start] if start == end else [], []
        
        rows, cols = len(grid), len(grid[0])
        
        # Check validity
        if (start[0] < 0 or start[0] >= rows or start[1] < 0 or start[1] >= cols or
            end[0] < 0 or end[0] >= rows or end[1] < 0 or end[1] >= cols or
            grid[start[0]][start[1]] == 1 or grid[end[0]][end[1]] == 1):
            return [], []
        
        # Initialize beam search
        current_level = [(self._heuristic(start, end), start, [start])]
        visited_global = {start}
        visited_order = [start]
        
        while current_level:
            next_level = []
            
            for _, current_pos, path in current_level:
                if current_pos == end:
                    return path, visited_order
                
                # Generate successors
                for neighbor in self._get_neighbors(current_pos, grid):
                    if neighbor not in visited_global:
                        visited_global.add(neighbor)
                        visited_order.append(neighbor)
                        
                        new_path = path + [neighbor]
                        h_score = self._heuristic(neighbor, end)
                        next_level.append((h_score, neighbor, new_path))
            
            # Keep only the best beam_width candidates
            next_level.sort(key=lambda x: x[0])
            current_level = next_level[:self.beam_width]
        
        return [], visited_order


class GreedyHillClimbing(GreedyBestFirst):
    """
    Hill Climbing variant that always moves to the best neighbor.
    Simple but can get stuck in local optima.
    """
    
    def __init__(self, heuristic='manhattan'):
        super().__init__(heuristic)
        self.name = "Hill Climbing"
        self.description = "Hill Climbing - Always moves to best neighbor"
    
    def solve(self, grid, start, end):
        """
        Solve using hill climbing algorithm.
        """
        if not grid or start == end:
            return [start] if start == end else [], []
        
        rows, cols = len(grid), len(grid[0])
        
        # Check validity
        if (start[0] < 0 or start[0] >= rows or start[1] < 0 or start[1] >= cols or
            end[0] < 0 or end[0] >= rows or end[1] < 0 or end[1] >= cols or
            grid[start[0]][start[1]] == 1 or grid[end[0]][end[1]] == 1):
            return [], []
        
        current = start
        path = [current]
        visited_order = [current]
        visited = {current}
        
        while current != end:
            neighbors = self._get_neighbors(current, grid)
            valid_neighbors = [n for n in neighbors if n not in visited]
            
            if not valid_neighbors:
                # Stuck - no valid moves
                break
            
            # Choose neighbor with best heuristic value
            best_neighbor = min(valid_neighbors, key=lambda pos: self._heuristic(pos, end))
            
            # Check if we're making progress
            if self._heuristic(best_neighbor, end) >= self._heuristic(current, end):
                # Not improving - might be stuck in local optimum
                # Try a random valid neighbor instead
                import random
                if len(valid_neighbors) > 1:
                    best_neighbor = random.choice(valid_neighbors)
            
            current = best_neighbor
            visited.add(current)
            visited_order.append(current)
            path.append(current)
            
            # Prevent infinite loops
            if len(path) > rows * cols:
                break
        
        if current == end:
            return path, visited_order
        else:
            return [], visited_order