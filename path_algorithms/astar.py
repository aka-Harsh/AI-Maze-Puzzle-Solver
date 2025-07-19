import heapq
import math

class AStar:
    """
    A* Search algorithm for maze solving.
    Uses heuristic function to guide search towards the goal.
    Optimal with admissible heuristic.
    """
    
    def __init__(self, heuristic='manhattan'):
        self.name = "A*"
        self.description = "A* Search - Uses heuristic to guide search towards goal"
        self.complexity = "O(b^d)"
        self.heuristic_type = heuristic
    
    def solve(self, grid, start, end):
        """
        Solve maze using A* Search algorithm.
        Improved implementation with better tie-breaking and path reconstruction.
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
        
        # Initialize A* with tie-breaking
        open_set = []
        counter = 0
        heapq.heappush(open_set, (0, counter, start))
        
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, end)}
        
        visited_order = []
        closed_set = set()
        
        while open_set:
            current_f, _, current = heapq.heappop(open_set)
            
            if current in closed_set:
                continue
            
            closed_set.add(current)
            visited_order.append(current)
            
            if current == end:
                path = self._reconstruct_path(came_from, current)
                return path, visited_order
            
            # Explore neighbors
            for neighbor in self._get_neighbors(current, grid):
                if neighbor in closed_set:
                    continue
                
                tentative_g_score = g_score[current] + self._distance(current, neighbor)
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self._heuristic(neighbor, end)
                    
                    counter += 1
                    heapq.heappush(open_set, (f_score[neighbor], counter, neighbor))
        
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
            return 0  # Dijkstra's algorithm (no heuristic)
    
    def _distance(self, pos1, pos2):
        """Calculate actual distance between adjacent positions."""
        return 1  # Unit cost for grid movement
    
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
        return self.heuristic_type in ['manhattan', 'euclidean']  # With admissible heuristics
    
    def get_algorithm_info(self):
        """Get detailed information about the algorithm."""
        return {
            'name': self.name,
            'description': self.description,
            'time_complexity': 'O(b^d)',
            'space_complexity': 'O(b^d)',
            'optimal': self.is_optimal(),
            'complete': True,
            'heuristic': self.heuristic_type,
            'characteristics': [
                'Uses both cost and heuristic',
                'Optimal with admissible heuristic',
                'More efficient than Dijkstra',
                'Goal-directed search'
            ],
            'best_use_cases': [
                'When heuristic information is available',
                'Shortest path in weighted graphs',
                'Game AI pathfinding',
                'Navigation systems'
            ],
            'limitations': [
                'Requires good heuristic function',
                'Can be memory intensive',
                'Performance depends on heuristic quality'
            ]
        }


class AStarWithDiagonals(AStar):
    """
    A* variant that allows diagonal movement.
    """
    
    def __init__(self, heuristic='octile'):
        super().__init__(heuristic)
        self.name = "A* (8-directional)"
        self.description = "A* Search with diagonal movement allowed"
    
    def _get_neighbors(self, pos, grid):
        """Get valid neighboring positions including diagonals."""
        rows, cols = len(grid), len(grid[0])
        neighbors = []
        
        # 8-directional movement
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        for dr, dc in directions:
            new_row, new_col = pos[0] + dr, pos[1] + dc
            
            if (0 <= new_row < rows and 0 <= new_col < cols and
                grid[new_row][new_col] == 0):
                
                # Check if diagonal movement is blocked
                if abs(dr) == 1 and abs(dc) == 1:  # Diagonal move
                    # Ensure adjacent cells are not walls (prevent corner cutting)
                    if (grid[pos[0] + dr][pos[1]] == 0 and 
                        grid[pos[0]][pos[1] + dc] == 0):
                        neighbors.append((new_row, new_col))
                else:  # Orthogonal move
                    neighbors.append((new_row, new_col))
        
        return neighbors
    
    def _distance(self, pos1, pos2):
        """Calculate actual distance between adjacent positions."""
        dr = abs(pos1[0] - pos2[0])
        dc = abs(pos1[1] - pos2[1])
        
        if dr == 1 and dc == 1:  # Diagonal
            return math.sqrt(2)
        else:  # Orthogonal
            return 1


class AStarJumpPointSearch(AStar):
    """
    A* with Jump Point Search optimization for grid-based pathfinding.
    Significantly reduces the number of nodes explored.
    """
    
    def __init__(self):
        super().__init__('octile')
        self.name = "A* JPS"
        self.description = "A* with Jump Point Search optimization"
    
    def solve(self, grid, start, end):
        """
        Solve using A* with Jump Point Search.
        """
        if not grid or start == end:
            return [start] if start == end else [], []
        
        rows, cols = len(grid), len(grid[0])
        
        # Check validity
        if (start[0] < 0 or start[0] >= rows or start[1] < 0 or start[1] >= cols or
            end[0] < 0 or end[0] >= rows or end[1] < 0 or end[1] >= cols or
            grid[start[0]][start[1]] == 1 or grid[end[0]][end[1]] == 1):
            return [], []
        
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, end)}
        
        visited_order = []
        closed_set = set()
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current in closed_set:
                continue
            
            closed_set.add(current)
            visited_order.append(current)
            
            if current == end:
                path = self._reconstruct_path(came_from, current)
                return path, visited_order
            
            # Get jump points instead of regular neighbors
            jump_points = self._get_jump_points(current, grid, end, came_from.get(current))
            
            for jp in jump_points:
                if jp in closed_set:
                    continue
                
                tentative_g_score = g_score[current] + self._heuristic(current, jp)
                
                if jp not in g_score or tentative_g_score < g_score[jp]:
                    came_from[jp] = current
                    g_score[jp] = tentative_g_score
                    f_score[jp] = tentative_g_score + self._heuristic(jp, end)
                    
                    heapq.heappush(open_set, (f_score[jp], jp))
        
        return [], visited_order
    
    def _get_jump_points(self, current, grid, goal, parent):
        """
        Find jump points from current position.
        """
        jump_points = []
        
        if parent is None:
            # Initial exploration in all directions
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                         (-1, -1), (-1, 1), (1, -1), (1, 1)]
        else:
            # Determine natural directions based on parent
            directions = self._get_natural_directions(parent, current)
        
        for direction in directions:
            jp = self._jump(current, direction, grid, goal)
            if jp:
                jump_points.append(jp)
        
        return jump_points
    
    def _jump(self, current, direction, grid, goal):
        """
        Recursively jump in a direction until finding a jump point.
        """
        dr, dc = direction
        next_pos = (current[0] + dr, current[1] + dc)
        
        # Check bounds and walls
        if (next_pos[0] < 0 or next_pos[0] >= len(grid) or
            next_pos[1] < 0 or next_pos[1] >= len(grid[0]) or
            grid[next_pos[0]][next_pos[1]] == 1):
            return None
        
        # Found goal
        if next_pos == goal:
            return next_pos
        
        # Check for forced neighbors (jump point condition)
        if self._has_forced_neighbors(next_pos, direction, grid):
            return next_pos
        
        # For diagonal movement, check orthogonal directions
        if dr != 0 and dc != 0:
            if (self._jump(next_pos, (dr, 0), grid, goal) or
                self._jump(next_pos, (0, dc), grid, goal)):
                return next_pos
        
        # Continue jumping
        return self._jump(next_pos, direction, grid, goal)
    
    def _has_forced_neighbors(self, pos, direction, grid):
        """
        Check if position has forced neighbors (making it a jump point).
        """
        dr, dc = direction
        r, c = pos
        rows, cols = len(grid), len(grid[0])
        
        # Check for forced neighbors based on direction
        if dr == 0:  # Horizontal movement
            if dc == 1:  # Moving right
                return ((r - 1 >= 0 and grid[r - 1][c] == 1 and r - 1 >= 0 and c + 1 < cols and grid[r - 1][c + 1] == 0) or
                        (r + 1 < rows and grid[r + 1][c] == 1 and r + 1 < rows and c + 1 < cols and grid[r + 1][c + 1] == 0))
            else:  # Moving left
                return ((r - 1 >= 0 and grid[r - 1][c] == 1 and r - 1 >= 0 and c - 1 >= 0 and grid[r - 1][c - 1] == 0) or
                        (r + 1 < rows and grid[r + 1][c] == 1 and r + 1 < rows and c - 1 >= 0 and grid[r + 1][c - 1] == 0))
        elif dc == 0:  # Vertical movement
            if dr == 1:  # Moving down
                return ((c - 1 >= 0 and grid[r][c - 1] == 1 and r + 1 < rows and c - 1 >= 0 and grid[r + 1][c - 1] == 0) or
                        (c + 1 < cols and grid[r][c + 1] == 1 and r + 1 < rows and c + 1 < cols and grid[r + 1][c + 1] == 0))
            else:  # Moving up
                return ((c - 1 >= 0 and grid[r][c - 1] == 1 and r - 1 >= 0 and c - 1 >= 0 and grid[r - 1][c - 1] == 0) or
                        (c + 1 < cols and grid[r][c + 1] == 1 and r - 1 >= 0 and c + 1 < cols and grid[r - 1][c + 1] == 0))
        
        return False
    
    def _get_natural_directions(self, parent, current):
        """
        Get natural directions to explore based on parent-current relationship.
        """
        dr = current[0] - parent[0]
        dc = current[1] - parent[1]
        
        directions = []
        
        if dr != 0 and dc != 0:  # Diagonal parent
            directions.extend([(dr, 0), (0, dc), (dr, dc)])
        elif dr != 0:  # Vertical parent
            directions.extend([(dr, 0), (dr, -1), (dr, 1)])
        elif dc != 0:  # Horizontal parent
            directions.extend([(0, dc), (-1, dc), (1, dc)])
        
        return directions